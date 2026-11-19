"""에피소드 끝난 뒤 판정: 실기 에피소드 bag 에서 뽑은 경계 사진(real_frames 의 bounds.jsonl)에 VLM 경계 판정을 돌리고 사람이 볼 HTML 을 만든다.
로봇 · ROS 와 무관한 오프라인 도구다(실시간 노드 아님).

  score: python3 -m vlm_supervisor.bench.episode_review score <bounds.jsonl> --url http://127.0.0.1:8767 [--out review.jsonl]
         python3 -m vlm_supervisor.bench.episode_review score <bounds.jsonl> --from-bench <run_bench results.jsonl> [--variant short2]
  page:  python3 -m vlm_supervisor.bench.episode_review page <review.jsonl> [--frames <사진 폴더>] [--out index.html] [--cfg]
  agree: python3 -m vlm_supervisor.bench.episode_review agree <review.jsonl> <labels.jsonl> [--cfg]

경계 판정: 경계 묶음 모든 노드의 node_checks 를 슬롯 기준으로 합친다. 확정 답 하나라도 기대와 다르면 막음, 아니면 오류 · low_mass 가
하나라도 있으면 판정불가, 모두 기대대로면 통과. 검사가 없거나 전 사진이 없으면(short2 는 두 장) 묻지 않고 판정불가.
score 는 이미 review.jsonl 에 있는 경계 id 를 건너뛴다(재실행 안전). 다시 묻고 싶은 경계는 그 줄을 지우고 돌린다.
종합 판정(combine): 노드 역할(role)별로 러너와 VLM 중 정답 쪽을 고른다. state 노드 검사는 묻지 않는다(go_home_end · final 의 all_in_holders 만 예외).
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import yaml

from vlm_supervisor.runner.vlm_http import VlmHttpJudge

CFG = Path(__file__).resolve().parents[1] / "config" / "judge_short.yaml"
OK = "COMPLETED"
VERDICT_KO = {"pass": "통과", "block": "막음", "undecided": "판정불가"}
EPISODE_NODES, EPISODE_SLOT = ("go_home_end", "final"), "all_in_holders"
UNPLACED = "?unplaced"     # bound_checks 키: 어느 place 노드에도 못 붙인 cup_in_holder_b 검사(노드 id 와 겹치지 않는다)


def merge_checks(nodes: list, node_checks: dict) -> list[dict]:
    """경계 묶음 노드들의 검사를 슬롯 기준으로 합친다(처음 나온 순서). 같은 슬롯의 기대값이 다르면 설정 오류다."""
    seen: dict[str, str] = {}
    for n in nodes:
        for c in node_checks.get(n["id"]) or []:
            if seen.setdefault(c["slot"], c["ok"]) != c["ok"]:
                raise ValueError(f"슬롯 {c['slot']} 의 기대값이 노드마다 다르다({n['id']})")
    return [{"slot": s, "expect": ok} for s, ok in seen.items()]


def judge_check(chk: dict, rec: dict) -> dict:
    """VlmHttpJudge 답 하나 → 검사 기록(p_yes · thr · 답 · 기대대로인지)."""
    out = {**chk, "p_yes": rec.get("p_yes"), "thr": rec.get("thr"), "mass": rec.get("mass"), "ans": None, "pass": None}
    if "scores" in rec:
        out["ans"] = "yes" if rec["scores"]["yes"] > rec["scores"]["no"] else "no"
        out["pass"] = out["ans"] == chk["expect"]
    elif "error" in rec:
        out["error"] = rec["error"]
    else:
        out["hold"] = "band" if rec.get("band") else "low_mass"
    return out


def verdict(checks: list[dict]) -> tuple[str, str]:
    if any(c.get("pass") is False for c in checks):
        return "block", ""
    if any("error" in c for c in checks):
        return "undecided", "오류"
    if any(c.get("pass") is None for c in checks):         # 묻지 않은 검사(before 없음)도 여기
        return "undecided", "low_mass"
    return "pass", ""


def review_row(b: dict, cfg: dict, answer) -> dict:
    """bounds 행 하나 → review 행. answer(b, [문구]) → [judge 답]."""
    nodes = [{"id": n["id"], "result": n.get("result"), "reason": n.get("reason") or ""} for n in b["nodes"]]
    row = {"id": b["id"], "seed": b["seed"], "k": b["k"], "node": b["node"], "nodes": nodes,
           "runner_ok": all(n["result"] == OK for n in nodes), "jpg": b["jpg"], "before": b.get("before"),
           "end": b.get("end", False)}
    nc = cfg.get("node_checks") or {}
    checks = [{**c, "q": cfg["questions"][c["slot"]]} for c in merge_checks(b["nodes"], bound_checks(b["nodes"], nc))]
    if not checks:
        return combine({**row, "checks": [], "verdict": "undecided", "why": "검사 없음", "sec": 0.0}, nc)
    if not b.get("before"):
        return combine({**row, "checks": checks, "verdict": "undecided", "why": "before 없음", "sec": 0.0}, nc)
    t0 = time.monotonic()
    recs = answer(b, [c["q"] for c in checks])
    checks = [judge_check(c, r) for c, r in zip(checks, recs)]
    v, why = verdict(checks)
    return combine({**row, "checks": checks, "verdict": v, "why": why, "sec": round(time.monotonic() - t0, 3)}, nc)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{i} 줄을 읽을 수 없다({exc}). 그 줄을 고치거나 지운 뒤 다시 돌린다") from None
    return rows


def http_answer(judge: VlmHttpJudge, frames: Path):
    return lambda b, qs: judge.ask([(frames / b["before"]).read_bytes(), (frames / b["jpg"]).read_bytes()], qs)


def bench_answer(judge: VlmHttpJudge, results: Path, variant: str):
    """run_bench results.jsonl 의 응답을 (경계 id, 프롬프트) 로 찾아 같은 문턱으로 다시 판정한다. 프롬프트가 다르면 못 찾는다."""
    resp = {}
    for r in read_jsonl(results):
        if r.get("variant") == variant:
            resp[(r["id"].rsplit("/", 1)[0], r["prompt"])] = r.get("resp") or {"error": r.get("error", "resp 없음")}

    def answer(b, qs):
        out = []
        for q in qs:
            r = resp.get((b["id"], judge.prompt.format(question=q)))
            if r is None:
                out.append({"error": "bench 결과에 없음"})
            elif "error" in r or not isinstance(r.get("scores"), dict):
                out.append({"error": f"bench: {r.get('error', '점수 없음')}"})
            else:
                out.append(judge.read_resp(r, q))
        return out
    return answer


def latest(rows: list[dict]) -> list[dict]:
    """같은 경계 id 가 여러 줄이면 마지막 줄을 쓴다(오류 경계를 다시 물은 결과). 순서는 처음 나온 자리."""
    last = {r["id"]: r for r in rows}
    return list(last.values())


def score(bounds: Path, out: Path, cfg_path: Path, answer_of) -> tuple[int, int]:
    """오류(서버 끊김·timeout)로 판정불가가 된 경계는 끝난 것으로 치지 않고 다시 묻는다."""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    rows = latest(read_jsonl(out)) if out.exists() else []
    done = {r["id"] for r in rows if r.get("why") != "오류"}
    bl = read_jsonl(bounds)
    dup = len(bl) - len({b["id"] for b in bl})
    if dup:
        raise SystemExit(f"{bounds}: 경계 id 가 {dup} 개 겹친다")
    todo = [b for b in bl if b["id"] not in done]
    with out.open("a", encoding="utf-8") as f:
        for b in todo:
            f.write(json.dumps(review_row(b, cfg, answer_of), ensure_ascii=False) + "\n")
            f.flush()
    return len(todo), len(done)


def role(node_id: str) -> str:
    """outcome: 결과는 VLM. drop: 결과는 러너, VLM 은 떨어뜨림 감시. state: 로봇 상태로 확인하는 이동·세팅이라 러너만."""
    if node_id.startswith(("grasp_", "place_")):
        return "outcome"
    if node_id.startswith(("to_tilt_start_", "tilt_")):
        return "drop"
    return "state"


def _place_for(slot: str, places: list[str]) -> str | None:
    """cup_in_holder_b:<물체> → 그 물체를 놓는 place 노드(접미사 대응). 이름이 안 맞으면 None — 다른 물체 노드로 대신하지 않는다."""
    obj = slot.split(":", 1)[1].upper()
    return next((p for p in places if obj in (p[len("place_"):].upper(), "SRC_" + p[len("place_"):].upper())), None)


def bound_checks(nodes: list, node_checks: dict) -> dict:
    """경계 묶음의 노드 → VLM 검사. state 노드 검사는 빼되 에피소드 노드의 all_in_holders 는 남기고,
    state 노드의 cup_in_holder_b:<물체> 는 같은 묶음의 place 노드 결과로 옮긴다(놓기 끝 사진은 손이 컵을 가려 place 자체엔 검사가 없다).
    묶음 밖(설정 전체)에도 그 물체의 place 노드가 없으면 설정 오류라 UNPLACED 로 모은다(combine 이 사람 확인으로 올린다)."""
    out = {n["id"]: [] for n in nodes}
    places = [n["id"] for n in nodes if n["id"].startswith("place_")]
    known = places + [k for k in node_checks if k.startswith("place_")]
    for n in nodes:
        nid = n["id"]
        for c in node_checks.get(nid) or []:
            if role(nid) != "state" or (nid in EPISODE_NODES and c["slot"] == EPISODE_SLOT):
                out[nid].append(c)
            elif c["slot"].startswith("cup_in_holder_b:"):
                p = _place_for(c["slot"], places)
                if p is None and _place_for(c["slot"], known) is None:
                    p = UNPLACED
                    out.setdefault(UNPLACED, [])
                if p and c["slot"] not in {x["slot"] for x in out[p]}:
                    out[p].append(c)
    return out


def combine(r: dict, node_checks: dict) -> dict:
    """review 행 + cfg node_checks → 행에 노드별 종합 결과(combined)와 경계 final 을 붙인다. 저장된 checks 를 슬롯 → 노드로 다시 나눈다."""
    nc = bound_checks(r["nodes"], node_checks)
    by_slot = {c["slot"]: c for c in r.get("checks") or []}
    out = []
    for n in r["nodes"]:
        ro, nid = role(n["id"]), n["id"]
        runner = "성공" if n["result"] == OK else "실패"
        e = {"node": nid, "role": ro, "runner": runner, "vlm": "비전 대상 아님", "final": runner, "note": ""}
        want = [c["slot"] for c in nc.get(nid) or []]
        cs = [by_slot[s] for s in want if s in by_slot]
        lost = [s for s in want if s not in by_slot]     # 설정엔 있는데 저장 검사엔 없다 — "검사 없음"과 다르다
        v = verdict(cs)[0] if cs else None
        if ro != "state" and lost and v != "block":      # 남은 검사가 막으면 그대로 실패, 아니면 성공으로 넘기지 않는다
            e["vlm"], e["final"], e["note"] = "검사 누락", "사람 확인", f"검사 누락: {', '.join(lost)}"
        elif ro != "state":
            e["vlm"] = VERDICT_KO[v] if v else "검사 없음"
            if ro == "outcome" and v == "pass":
                e["final"] = "성공"
                if runner == "실패":
                    e["note"] = f"과정 경고: {(n.get('reason') or n['result'])[:60]}"
            elif ro == "outcome" and v == "block":
                e["final"] = "실패"
            elif ro == "outcome":          # 러너의 잡기 · 놓기 실패 신호는 과정 신호라 결과로 믿지 않는다
                e["final"], e["note"] = ("성공", "VLM 판정 못 함") if runner == "성공" else ("사람 확인", "러너 실패 · VLM 판정 못 함")
            elif v == "block":
                e["final"], e["note"] = "실패", "떨어뜨림 의심"
            elif v == "pass":
                e["note"] = "떨어뜨림 없음"
        out.append(e)
    ep = [n["id"] for n in r["nodes"] if EPISODE_SLOT in {c["slot"] for c in nc.get(n["id"]) or []}]
    if ep and EPISODE_SLOT in by_slot:
        v = verdict([by_slot[EPISODE_SLOT]])[0]
        out.append({"node": EPISODE_SLOT, "role": "episode", "runner": "-", "vlm": VERDICT_KO[v],
                    "final": {"pass": "성공", "block": "실패"}.get(v, "사람 확인"), "note": ""})
    elif ep:
        out.append({"node": EPISODE_SLOT, "role": "episode", "runner": "-", "vlm": "검사 없음",
                    "final": "사람 확인", "note": "VLM 검사 없음"})
    for c in nc.get(UNPLACED, []):
        out.append({"node": c["slot"], "role": "episode", "runner": "-", "vlm": "검사 없음",
                    "final": "사람 확인", "note": "놓기 노드 못 찾음"})
    finals = [e["final"] for e in out]
    final = "실패" if "실패" in finals else "사람 확인" if "사람 확인" in finals else "성공"
    return {**r, "combined": out, "final": final}


# ---------------- page ----------------
E = html.escape

CSS = """
body{font-family:system-ui,'Noto Sans KR',sans-serif;margin:0 auto;max-width:1300px;padding:16px 20px;color:#1b1f24;background:#fafbfc;line-height:1.45}
h1{font-size:1.4rem}h2{font-size:1.15rem;margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:4px}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top;font-size:.9rem}
th{background:#eef1f4}.note{color:#57606a;font-size:.88rem}
.comb td{white-space:nowrap}.comb td.note{white-space:normal;min-width:16rem}
.card{background:#fff;border:1px solid #d0d7de;border-radius:6px;padding:10px 12px;margin:10px 0;display:flex;gap:14px;flex-wrap:wrap}
.card.mm{border:2px solid #bf3989}.photos{display:flex;gap:6px}.photos figure{margin:0;text-align:center;font-size:.8rem;color:#57606a}
.photos img{width:240px;height:auto;border:1px solid #d0d7de;display:block}.noimg{width:240px;height:135px;display:flex;align-items:center;justify-content:center;background:#eef1f4}
.info{flex:1;min-width:420px}.info h3{margin:0 0 6px;font-size:1rem}.v{font-weight:600;padding:1px 8px;border-radius:10px}
.v.pass{background:#dafbe1;color:#116329}.v.block{background:#ffebe9;color:#a40e26}.v.undecided{background:#fff8c5;color:#7d4e00}
.rfail{color:#a40e26}.chk{font-family:ui-monospace,monospace;font-size:.82rem}.chk .bad{color:#a40e26;font-weight:600}
fieldset{border:1px solid #d0d7de;border-radius:4px;margin-top:6px;font-size:.88rem}fieldset label{margin-right:12px;cursor:pointer}
.only-mm .card:not(.mm),.only-mm section.ep:not(:has(.mm)){display:none}
#out{width:100%;height:120px;font-family:ui-monospace,monospace;font-size:.8rem}button{font-size:.9rem;padding:4px 12px;cursor:pointer}
"""

JS = """
const KEY='episode_review_marks:'+location.pathname;
let marks={};try{marks=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const radios=document.querySelectorAll('input[type=radio][data-id]');
function count(){document.getElementById('nmark').textContent=Object.keys(marks).length}
radios.forEach(r=>{if(marks[r.dataset.id]===r.value)r.checked=true;
  r.addEventListener('change',()=>{marks[r.dataset.id]=r.value;try{localStorage.setItem(KEY,JSON.stringify(marks))}catch(e){};count()})});
count();
document.getElementById('onlymm').addEventListener('change',e=>document.body.classList.toggle('only-mm',e.target.checked));
document.getElementById('copy').addEventListener('click',()=>{
  const t=Object.entries(marks).map(([id,m])=>JSON.stringify({id:id,mark:m,runner_ok:INFO[id]&&INFO[id][0],vlm:INFO[id]&&INFO[id][1]})).join('\\n');
  const o=document.getElementById('out');o.value=t;o.select();
  if(navigator.clipboard)navigator.clipboard.writeText(t).catch(()=>{});});
"""

MARKS = [("runner", "러너 맞음"), ("vlm", "VLM 맞음"), ("both_wrong", "둘 다 틀림"), ("unknown", "모름")]


def _img(rel: str, path: str | None, cap: str) -> str:
    if not path:
        return f'<figure><div class="noimg">없음</div>{cap}</figure>'
    src = E(quote(f"{rel}/{path}"))
    return f'<figure><a href="{src}" target="_blank"><img src="{src}" loading="lazy" alt="{E(cap)} 사진 {E(path)}"></a>{cap}</figure>'


def _check_line(c: dict) -> str:
    if c.get("pass") is None and "ans" not in c:            # 묻지 않은 검사(before 없음)
        return f'<div class="chk">{E(c["slot"])}: 묻지 않음 (기대 {E(c["expect"])})</div>'
    p = "-" if c.get("p_yes") is None else f'{c["p_yes"]:.3f}'
    t = "-" if c.get("thr") is None else f'{c["thr"]:.2f}'
    res = c.get("ans") or c.get("hold") or f'오류 {c.get("error", "")}'
    cls = ' class="bad"' if c.get("pass") is False else ""
    return (f'<div class="chk" title="{E(c["q"])}">{E(c["slot"])}: p_yes {p} / 문턱 {t} → '
            f'<span{cls}>{E(res)}</span> (기대 {E(c["expect"])})</div>')


FINAL_CLS = {"성공": "pass", "실패": "block", "사람 확인": "undecided"}


def _final(f: str) -> str:
    return f'<span class="v {FINAL_CLS.get(f, "undecided")}">{E(f)}</span>'


def _comb_table(r: dict) -> str:
    trs = "".join(f'<tr><td>{E(e["node"])}</td><td>{E(e["role"])}</td><td>{E(e["runner"])}</td><td>{E(e["vlm"])}</td>'
                  f'<td>{_final(e["final"])}</td><td class="note">{E(e["note"])}</td></tr>' for e in r["combined"])
    return ('<table class="comb"><tr><th>노드</th><th>역할</th><th>러너</th><th>VLM</th><th>종합</th><th>비고</th></tr>'
            f'{trs}</table>')


def _card(r: dict, rel: str, node_checks: dict) -> str:
    rid = r["id"]
    used = {c["slot"] for cs in bound_checks(r["nodes"], node_checks).values() for c in cs}
    runner = "<br>".join(
        f'<span class="{"" if n["result"] == OK else "rfail"}">{E(n["id"])}: {E(str(n["result"]))}</span>'
        + (f' <span class="note">{E(n["reason"])}</span>' if n["reason"] else "") for n in r["nodes"])
    why = f' <span class="note">({E(r["why"])})</span>' if r["why"] else ""
    radios = "".join(f'<label><input type="radio" name="m:{E(rid)}" data-id="{E(rid)}" value="{v}"> {lab}</label>'
                     for v, lab in MARKS)
    return (f'<div class="card{"" if r["final"] == "성공" else " mm"}" id="{E(rid)}">'
            f'<div class="photos">{_img(rel, r["before"], "전")}{_img(rel, r["jpg"], "후")}</div>'
            f'<div class="info"><h3>{r["k"]:02d} · {E(" + ".join(n["id"] for n in r["nodes"]))} → 종합 {_final(r["final"])}</h3>'
            + _comb_table(r) +
            f'<div><b>러너 기록</b><br>{runner}</div>'
            f'<div style="margin-top:6px"><b>VLM</b> <span class="v {r["verdict"]}">{VERDICT_KO[r["verdict"]]}</span>{why}</div>'
            + "".join(_check_line(c) if c["slot"] in used else
                      f'<div class="chk note">{E(c["slot"])}: 상태 노드 검사라 종합에서 뺐다</div>' for c in r["checks"]) +
            f'<fieldset><legend>누가 맞았나</legend>{radios}</fieldset></div></div>')


def page(rows: list[dict], frames: Path, out: Path, node_checks: dict) -> dict:
    """종합 결과는 review 행의 checks 로 다시 계산한다(옛 review.jsonl 도 서버 없이 페이지를 만들 수 있다)."""
    rel = Path(os.path.relpath(frames, out.parent)).as_posix()
    rows = [combine(r, node_checks) for r in rows]
    eps = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (r["seed"], r["k"])):
        eps.setdefault(r["seed"], []).append(r)
    cnt = collections.Counter(r["verdict"] for r in rows)
    whys = collections.Counter(r["why"] for r in rows if r["verdict"] == "undecided")
    fin = collections.Counter(r["final"] for r in rows)
    bad = [r for r in rows if r["final"] != "성공"]
    warn = [(r, e) for r in rows for e in r["combined"] if e["note"].startswith("과정 경고")]
    summ = {"episodes": len(eps), "bounds": len(rows), **{k: cnt.get(k, 0) for k in VERDICT_KO},
            "undecided_why": dict(whys), "runner_fail": sum(not r["runner_ok"] for r in rows),
            "final": {k: fin.get(k, 0) for k in FINAL_CLS},
            "final_fail": [r["id"] for r in bad if r["final"] == "실패"],
            "final_human": [r["id"] for r in bad if r["final"] == "사람 확인"],
            "process_warn": [r["id"] for r, _ in warn]}
    bad_rows = "".join(f'<tr><td><a href="#{E(r["id"])}">{E(r["id"])}</a></td><td>{_final(r["final"])}</td>'
                       f'<td>{_comb_table(r)}</td></tr>' for r in bad)
    warn_rows = "".join(f'<tr><td><a href="#{E(r["id"])}">{E(r["id"])}</a></td><td>{E(e["node"])}</td>'
                        f'<td>{E(e["note"])}</td></tr>' for r, e in warn)
    info = json.dumps({r["id"]: [r["runner_ok"], r["verdict"]] for r in rows}, ensure_ascii=False).replace("</", "<\\/")
    body = [
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>에피소드 경계 판정 확인</title>'
        f'<style>{CSS}</style></head><body>',
        '<h1>에피소드 경계 판정 확인</h1>',
        '<p class="note">종합 결과는 노드 역할로 정한다. state(이동·세팅)는 러너, outcome(grasp · place)은 VLM, '
        'drop(to_tilt_start · tilt)은 러너에 VLM 떨어뜨림 감시를 더한다. go_home_end · final 의 all_in_holders 는 에피소드 결과다. '
        '사진을 보고 누가 맞았는지 표시하면 이 브라우저(localStorage)에 남고, 아래 버튼으로 jsonl 텍스트를 복사한다.</p>',
        '<table><tr><th>에피소드</th><th>경계</th><th>종합 성공</th><th>종합 실패</th><th>사람 확인</th>'
        '<th>VLM 통과 · 막음 · 판정불가</th><th>러너 실패 기록</th><th>과정 경고</th></tr>'
        f'<tr><td>{summ["episodes"]}</td><td>{summ["bounds"]}</td><td>{fin["성공"]}</td><td>{fin["실패"]}</td>'
        f'<td>{fin["사람 확인"]}</td><td>{summ["pass"]} · {summ["block"]} · {summ["undecided"]} '
        f'<span class="note">{E(", ".join(f"{k} {v}" for k, v in whys.items()))}</span></td>'
        f'<td>{summ["runner_fail"]}</td><td>{len(warn)}</td></tr></table>',
        '<h2>종합 결과 실패 · 사람 확인 경계</h2>',
        (f'<table><tr><th>경계</th><th>종합</th><th>노드별</th></tr>{bad_rows}</table>' if bad else '<p>없음</p>'),
        '<h2>과정 경고 (러너 실패 → 종합 성공)</h2>',
        '<p class="note">러너는 과정 신호(래치 전 손 폄, lift_guard 등)로 실패를 냈지만 VLM 이 결과를 성공으로 본 노드.</p>',
        (f'<table><tr><th>경계</th><th>노드</th><th>러너 사유</th></tr>{warn_rows}</table>' if warn else '<p>없음</p>'),
        '<p><label><input type="checkbox" id="onlymm"> 실패 · 사람 확인 경계만 보기</label> · 표시한 경계 <span id="nmark">0</span>개 '
        '<button id="copy" type="button">표시 결과 jsonl 복사</button></p>'
        '<textarea id="out" readonly aria-label="표시 결과 jsonl"></textarea>',
    ]
    for seed, rs in eps.items():
        body.append(f'<section class="ep"><h2>에피소드 {E(seed)} <span class="note">경계 {len(rs)}</span></h2>')
        body.extend(_card(r, rel, node_checks) for r in rs)
        body.append("</section>")
    body.append(f"<script>const INFO={info};{JS}</script></body></html>")
    out.write_text("\n".join(body), encoding="utf-8")
    return summ


def truth(lab: dict) -> bool | None:
    """사람 표시 → 경계 정답(성공 여부). 정답을 모르는 표시(VLM 맞음인데 VLM 판정불가 · 둘 다 틀림 · 모름)는 None."""
    if lab["mark"] == "runner":
        return bool(lab["runner_ok"])
    if lab["mark"] == "vlm" and lab.get("vlm") in ("pass", "block"):
        return lab["vlm"] == "pass"
    return None


def bound_role(r: dict) -> str:
    """경계 묶음의 역할: outcome 노드가 있으면 outcome, 그다음 drop, 나머지 state."""
    roles = {role(n["id"]) for n in r["nodes"]}
    return "outcome" if "outcome" in roles else "drop" if "drop" in roles else "state"


def agree(rows: list[dict], labels: list[dict], node_checks: dict) -> dict:
    """정답 있는 경계에서 러너만 · 종합 판정의 맞음 · 틀림 · 사람 확인 수를 경계 역할별로 센다."""
    by_id = {r["id"]: combine(r, node_checks) for r in rows}
    labels = list({x["id"]: x for x in labels}.values())     # 같은 경계를 다시 표시했으면 마지막 표시
    tab = {k: collections.Counter() for k in ("state", "outcome", "drop")}
    wrong = {"runner": [], "combined": []}
    for lab in labels:
        r = by_id.get(lab["id"])
        t = truth(lab)
        if r is None or t is None:
            continue
        c = tab[bound_role(r)]
        c["n"] += 1
        ok = r["runner_ok"] == t
        c["runner_ok" if ok else "runner_bad"] += 1
        if not ok:
            wrong["runner"].append(lab["id"])
        if r["final"] == "사람 확인":
            c["comb_human"] += 1
        elif (r["final"] == "성공") == t:
            c["comb_ok"] += 1
        else:
            c["comb_bad"] += 1
            wrong["combined"].append(lab["id"])
    return {"table": {k: dict(v) for k, v in tab.items()}, "wrong": wrong,
            "both_wrong": [x["id"] for x in labels if x["mark"] == "both_wrong"],
            "no_truth": sum(truth(x) is None for x in labels), "missing": [x["id"] for x in labels if x["id"] not in by_id]}


def agree_text(a: dict) -> str:
    lines = ["| 역할 | 정답 경계 | 러너 맞음 | 러너 틀림 | 종합 맞음 | 종합 틀림 | 종합 사람 확인 |", "|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for k, c in a["table"].items():
        tot.update(c)
        lines.append(f"| {k} | " + " | ".join(str(c.get(f, 0)) for f in
                     ("n", "runner_ok", "runner_bad", "comb_ok", "comb_bad", "comb_human")) + " |")
    lines.append("| 합계 | " + " | ".join(str(tot.get(f, 0)) for f in
                 ("n", "runner_ok", "runner_bad", "comb_ok", "comb_bad", "comb_human")) + " |")
    lines += ["", f"러너 틀린 경계: {', '.join(a['wrong']['runner']) or '없음'}",
              f"종합 틀린 경계: {', '.join(a['wrong']['combined']) or '없음'}",
              f"둘 다 틀림(정답 없음): {', '.join(a['both_wrong']) or '없음'}",
              f"정답 없는 표시 수: {a['no_truth']}" + (f" · review 에 없는 id: {', '.join(a['missing'])}" if a["missing"] else "")]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="bounds.jsonl → review.jsonl")
    s.add_argument("bounds", type=Path)
    s.add_argument("--cfg", type=Path, default=CFG)
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="VLM 서버(tools/vlm_server.py)")
    src.add_argument("--from-bench", type=Path, help="run_bench results.jsonl 을 다시 판정(서버 없이)")
    s.add_argument("--variant", default="short2")
    s.add_argument("--timeout", type=float, default=60.0)
    s.add_argument("--band", type=float, default=0.0)
    s.add_argument("--out", type=Path, help="기본 <bounds 폴더>/review.jsonl")
    p = sub.add_parser("page", help="review.jsonl → index.html")
    p.add_argument("review", type=Path)
    p.add_argument("--frames", type=Path, help="사진 폴더(bounds.jsonl 의 jpg 기준). 기본 review.jsonl 폴더")
    p.add_argument("--out", type=Path, help="기본 <review 폴더>/index.html")
    p.add_argument("--cfg", type=Path, default=CFG, help="노드 → 검사 슬롯(node_checks)을 읽을 설정")
    g = sub.add_parser("agree", help="사람 표시(labels.jsonl)와 러너 · 종합 판정 비교")
    g.add_argument("review", type=Path)
    g.add_argument("labels", type=Path)
    g.add_argument("--cfg", type=Path, default=CFG)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        out = a.out or a.bounds.parent / "review.jsonl"
        judge = VlmHttpJudge(a.cfg, a.url or "", a.timeout, a.band)
        answer = (bench_answer(judge, a.from_bench, a.variant) if a.from_bench
                  else http_answer(judge, a.bounds.parent))
        new, old = score(a.bounds, out, a.cfg, answer)
        print(f"{out}: 새 경계 {new}, 이미 있던 경계 {old}")
        return 0
    nc = yaml.safe_load(a.cfg.read_text(encoding="utf-8")).get("node_checks") or {}
    if a.cmd == "agree":
        print(agree_text(agree(latest(read_jsonl(a.review)), read_jsonl(a.labels), nc)))
    else:
        out = a.out or a.review.parent / "index.html"
        summ = page(latest(read_jsonl(a.review)), a.frames or a.review.parent, out, nc)
        print(f"{out}: {json.dumps(summ, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

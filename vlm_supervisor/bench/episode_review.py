"""에피소드 끝난 뒤 판정: 실기 에피소드 bag 에서 뽑은 경계 사진(real_frames 의 bounds.jsonl)에 VLM 경계 판정을 돌리고 사람이 볼 HTML 을 만든다.
로봇 · ROS 와 무관한 오프라인 도구다(실시간 노드 아님).

  score: python3 -m vlm_supervisor.bench.episode_review score <bounds.jsonl> --url http://127.0.0.1:8767 [--out review.jsonl]
         python3 -m vlm_supervisor.bench.episode_review score <bounds.jsonl> --from-bench <run_bench results.jsonl> [--variant short2]
  page:  python3 -m vlm_supervisor.bench.episode_review page <review.jsonl> [--frames <사진 폴더>] [--out index.html]

경계 판정: 경계 묶음 모든 노드의 node_checks 를 슬롯 기준으로 합친다. 확정 답 하나라도 기대와 다르면 막음, 아니면 오류 · low_mass 가
하나라도 있으면 판정불가, 모두 기대대로면 통과. 검사가 없거나 전 사진이 없으면(short2 는 두 장) 묻지 않고 판정불가.
score 는 이미 review.jsonl 에 있는 경계 id 를 건너뛴다(재실행 안전). 다시 묻고 싶은 경계는 그 줄을 지우고 돌린다.
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
    if any(c["pass"] is False for c in checks):
        return "block", ""
    if any("error" in c for c in checks):
        return "undecided", "오류"
    if any(c["pass"] is None for c in checks):
        return "undecided", "low_mass"
    return "pass", ""


def review_row(b: dict, cfg: dict, answer) -> dict:
    """bounds 행 하나 → review 행. answer(b, [문구]) → [judge 답]."""
    nodes = [{"id": n["id"], "result": n.get("result"), "reason": n.get("reason") or ""} for n in b["nodes"]]
    row = {"id": b["id"], "seed": b["seed"], "k": b["k"], "node": b["node"], "nodes": nodes,
           "runner_ok": all(n["result"] == OK for n in nodes), "jpg": b["jpg"], "before": b.get("before"),
           "end": b.get("end", False)}
    checks = [{**c, "q": cfg["questions"][c["slot"]]} for c in merge_checks(b["nodes"], cfg.get("node_checks") or {})]
    if not checks:
        return {**row, "checks": [], "verdict": "undecided", "why": "검사 없음", "sec": 0.0}
    if not b.get("before"):
        return {**row, "checks": checks, "verdict": "undecided", "why": "before 없음", "sec": 0.0}
    t0 = time.monotonic()
    recs = answer(b, [c["q"] for c in checks])
    checks = [judge_check(c, r) for c, r in zip(checks, recs)]
    v, why = verdict(checks)
    return {**row, "checks": checks, "verdict": v, "why": why, "sec": round(time.monotonic() - t0, 3)}


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


def mismatch(r: dict) -> bool:
    """러너 기록과 VLM 이 다르게 본 경계: 러너 실패 코드인데 VLM 통과, 또는 러너 COMPLETED 인데 VLM 막음."""
    return (not r["runner_ok"] and r["verdict"] == "pass") or (r["runner_ok"] and r["verdict"] == "block")


# ---------------- page ----------------
E = html.escape

CSS = """
body{font-family:system-ui,'Noto Sans KR',sans-serif;margin:0 auto;max-width:1300px;padding:16px 20px;color:#1b1f24;background:#fafbfc;line-height:1.45}
h1{font-size:1.4rem}h2{font-size:1.15rem;margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:4px}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top;font-size:.9rem}
th{background:#eef1f4}.note{color:#57606a;font-size:.88rem}
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


def _card(r: dict, rel: str) -> str:
    rid = r["id"]
    runner = "<br>".join(
        f'<span class="{"" if n["result"] == OK else "rfail"}">{E(n["id"])}: {E(str(n["result"]))}</span>'
        + (f' <span class="note">{E(n["reason"])}</span>' if n["reason"] else "") for n in r["nodes"])
    why = f' <span class="note">({E(r["why"])})</span>' if r["why"] else ""
    radios = "".join(f'<label><input type="radio" name="m:{E(rid)}" data-id="{E(rid)}" value="{v}"> {lab}</label>'
                     for v, lab in MARKS)
    return (f'<div class="card{" mm" if mismatch(r) else ""}" id="{E(rid)}">'
            f'<div class="photos">{_img(rel, r["before"], "전")}{_img(rel, r["jpg"], "후")}</div>'
            f'<div class="info"><h3>{r["k"]:02d} · {E(" + ".join(n["id"] for n in r["nodes"]))}</h3>'
            f'<div><b>러너 기록</b><br>{runner}</div>'
            f'<div style="margin-top:6px"><b>VLM</b> <span class="v {r["verdict"]}">{VERDICT_KO[r["verdict"]]}</span>{why}</div>'
            + "".join(_check_line(c) for c in r["checks"]) +
            f'<fieldset><legend>누가 맞았나</legend>{radios}</fieldset></div></div>')


def page(rows: list[dict], frames: Path, out: Path) -> dict:
    rel = Path(os.path.relpath(frames, out.parent)).as_posix()
    eps = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (r["seed"], r["k"])):
        eps.setdefault(r["seed"], []).append(r)
    cnt = collections.Counter(r["verdict"] for r in rows)
    whys = collections.Counter(r["why"] for r in rows if r["verdict"] == "undecided")
    mm = [r for r in rows if mismatch(r)]
    summ = {"episodes": len(eps), "bounds": len(rows), **{k: cnt.get(k, 0) for k in VERDICT_KO},
            "undecided_why": dict(whys), "runner_fail": sum(not r["runner_ok"] for r in rows),
            "mismatch": [r["id"] for r in mm]}
    mm_rows = "".join(
        f'<tr><td><a href="#{E(r["id"])}">{E(r["id"])}</a></td><td>{E(" + ".join(n["id"] for n in r["nodes"]))}</td>'
        f'<td>{E(", ".join(str(n["result"]) for n in r["nodes"]))}</td>'
        f'<td><span class="v {r["verdict"]}">{VERDICT_KO[r["verdict"]]}</span></td></tr>' for r in mm)
    info = json.dumps({r["id"]: [r["runner_ok"], r["verdict"]] for r in rows}, ensure_ascii=False).replace("</", "<\\/")
    body = [
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>에피소드 경계 판정 확인</title>'
        f'<style>{CSS}</style></head><body>',
        '<h1>에피소드 경계 판정 확인</h1>',
        '<p class="note">러너 기록은 러너가 스스로 남긴 결과라 정답이 아니다. 사진을 보고 누가 맞았는지 표시한다. '
        '표시는 이 브라우저(localStorage)에 남고, 아래 버튼으로 jsonl 텍스트를 복사한다.</p>',
        '<table><tr><th>에피소드</th><th>경계</th><th>통과</th><th>막음</th><th>판정불가</th><th>러너 실패 기록</th><th>러너·VLM 불일치</th></tr>'
        f'<tr><td>{summ["episodes"]}</td><td>{summ["bounds"]}</td><td>{summ["pass"]}</td><td>{summ["block"]}</td>'
        f'<td>{summ["undecided"]} <span class="note">{E(", ".join(f"{k} {v}" for k, v in whys.items()))}</span></td>'
        f'<td>{summ["runner_fail"]}</td><td>{len(mm)}</td></tr></table>',
        '<h2>러너와 VLM 이 다르게 본 경계</h2>',
        '<p class="note">러너 실패 코드인데 VLM 통과, 또는 러너 COMPLETED 인데 VLM 막음. 판정불가 경계는 넣지 않았다.</p>',
        (f'<table><tr><th>경계</th><th>노드</th><th>러너 기록</th><th>VLM</th></tr>{mm_rows}</table>' if mm
         else '<p>없음</p>'),
        '<p><label><input type="checkbox" id="onlymm"> 불일치 경계만 보기</label> · 표시한 경계 <span id="nmark">0</span>개 '
        '<button id="copy" type="button">표시 결과 jsonl 복사</button></p>'
        '<textarea id="out" readonly aria-label="표시 결과 jsonl"></textarea>',
    ]
    for seed, rs in eps.items():
        body.append(f'<section class="ep"><h2>에피소드 {E(seed)} <span class="note">경계 {len(rs)}</span></h2>')
        body.extend(_card(r, rel) for r in rs)
        body.append("</section>")
    body.append(f"<script>const INFO={info};{JS}</script></body></html>")
    out.write_text("\n".join(body), encoding="utf-8")
    return summ


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
    a = ap.parse_args(argv)
    if a.cmd == "score":
        out = a.out or a.bounds.parent / "review.jsonl"
        judge = VlmHttpJudge(a.cfg, a.url or "", a.timeout, a.band)
        answer = (bench_answer(judge, a.from_bench, a.variant) if a.from_bench
                  else http_answer(judge, a.bounds.parent))
        new, old = score(a.bounds, out, a.cfg, answer)
        print(f"{out}: 새 경계 {new}, 이미 있던 경계 {old}")
    else:
        out = a.out or a.review.parent / "index.html"
        summ = page(latest(read_jsonl(a.review)), a.frames or a.review.parent, out)
        print(f"{out}: {json.dumps(summ, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

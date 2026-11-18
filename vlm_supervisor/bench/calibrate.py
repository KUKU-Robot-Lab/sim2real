"""짧은 예/아니오 판정의 문턱 정하기와 노드 단위 평가(11.17).

run_bench 의 short2(전·후 사진 2장) 결과(p_yes)로 질문별 문턱을 정한다. 같은 시드의 사진은 서로 독립이 아니어서 시드 단위로 나눈다.
  --split half(기본): 시드를 해시로 반씩 나눠 앞 절반(train)에서 문턱을 고르고 뒤 절반(test)에서 잰다. yaml 문턱 = train 문턱.
  --split loeo: 시드(실기는 에피소드) 하나를 빼고 고른 문턱을 뺀 시드에 적용해 잰다. yaml 문턱 = 전체로 고른 문턱.
경계 묶음: items 옆 boundaries.jsonl 에 nodes(한 경계에 묶인 노드들)가 있으면 그 노드들의 검사를 합친다(같은 슬롯은 뒤 노드 것).

노드 판정: NODE_CHECKS 의 질문이 모두 성공 쪽 답이면 통과, 하나라도 어긋나면 실패 표시.
  거짓 통과 = 참값으로 실패인 경계를 통과시킴(감독에서 위험한 쪽), 거짓 경보 = 참값 성공인데 실패 표시.

    python3 -m vlm_supervisor.bench.calibrate --results <short/results.jsonl> --items <bench/items.jsonl> --out config/judge_short.yaml
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import yaml

# 노드 → [(슬롯, 성공일 때 답)]. 슬롯은 build_set 항목 id 의 끝 조각(question_id:인자…).
# 약한 질문(cup_not_in_holder, cups_on_table)은 넣지 않는다(11.17 시험대: 문턱을 골라도 균형 0.75 · 0.84).
# holders_empty · holders_count0 도 뺀다(실기 6 에피소드: yes/no 간격 0.01~0.05).
# cup_released_in_holder 는 뺀다(11.18 실기 13 에피소드: short2 yes 재현 11/33). 홀더 안착은 cup_in_holder_b 가 본다.
# place_pink 끝 사진은 손이 컵을 아직 감싸 사람도 못 정한다 — 판정은 다음 노드(home_r_after_pink)로 넘긴다.
NODE_CHECKS: dict[str, list[tuple[str, str]]] = {
    "scene": [("holders_any_cup", "no")],
    "grasp_blue": [("cup_in_hand:right:SRC_BLUE", "yes"), ("cup_lifted:SRC_BLUE", "yes")],
    "grasp_shaker": [("cup_in_hand:left:SHAKER", "yes"), ("cup_lifted:SHAKER", "yes")],
    "to_tilt_start_r1": [("cup_in_hand:right:SRC_BLUE", "yes")],
    "to_tilt_start_l": [("cup_in_hand:left:SHAKER", "yes")],
    "tilt_blue": [("cup_in_hand:right:SRC_BLUE", "yes"), ("cup_in_hand:left:SHAKER", "yes")],
    "place_blue": [("cup_in_holder_b:SRC_BLUE", "yes")],
    "home_r_after_blue": [("cup_in_holder_b:SRC_BLUE", "yes"), ("cup_in_hand:left:SHAKER", "yes")],
    "grasp_pink": [("cup_in_hand:right:SRC_PINK", "yes"), ("cup_lifted:SRC_PINK", "yes")],
    "to_tilt_start_r2": [("cup_in_hand:right:SRC_PINK", "yes")],
    "tilt_pink": [("cup_in_hand:right:SRC_PINK", "yes"), ("cup_in_hand:left:SHAKER", "yes")],
    "place_pink": [],
    "home_r_after_pink": [("cup_in_holder_b:SRC_PINK", "yes"), ("cup_in_hand:left:SHAKER", "yes")],  # 실기 yaml 전용
    "place_shaker": [("cup_in_holder_b:SHAKER", "yes")],
    "home_l_after_shaker": [("cup_in_holder_b:SHAKER", "yes")],  # 실기 yaml 전용
    "go_home_end": [("all_in_holders", "yes")],
    "final": [("all_in_holders", "yes")],
}
GRID = [i / 100 for i in range(1, 100)]


def half(seed: str) -> str:
    return "test" if int(hashlib.md5(seed.encode()).hexdigest(), 16) % 2 else "train"


def slot(item_id: str) -> str:
    return item_id.rsplit("/", 1)[-1]


def boundary(item_id: str) -> str:
    return item_id.rsplit("/", 1)[0]


def load(results: Path, items: Path, variant: str) -> list[dict]:
    text = {}
    for line in items.read_text().splitlines():
        d = json.loads(line)
        text[d["id"]] = d["question"]
    rows = []
    for line in results.read_text().splitlines():
        d = json.loads(line)
        if d.get("variant") != variant or not isinstance(d.get("p_yes"), (int, float)):
            continue
        d["question"] = text.get(d["id"], "")
        rows.append(d)
    return rows


def recall(rows: list[dict], t: float) -> tuple[float, float]:
    pos = [r for r in rows if r["truth"] == "yes"]
    neg = [r for r in rows if r["truth"] == "no"]
    yr = sum(r["p_yes"] >= t for r in pos) / len(pos) if pos else float("nan")
    nr = sum(r["p_yes"] < t for r in neg) / len(neg) if neg else float("nan")
    return yr, nr


def pick(rows: list[dict]) -> float | None:
    """균형 정확도가 가장 높은 문턱. 같으면 가운데(0.5)에 가까운 쪽."""
    if not any(r["truth"] == "yes" for r in rows) or not any(r["truth"] == "no" for r in rows):
        return None
    best = max(GRID, key=lambda t: (sum(recall(rows, t)) / 2, -abs(t - 0.5)))
    return best


def answer(p_yes: float, t: float) -> str:
    return "yes" if p_yes >= t else "no"


def held_out_thr(rows: list[dict], split: str) -> dict[str, float]:
    """평가 행 id → 그 행의 시드를 빼고 고른 문턱. half 는 test 절반 행만, loeo 는 모든 행."""
    by_q: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    out = {}
    for rs in by_q.values():
        if split == "half":
            t = pick([r for r in rs if half(r["seed"]) == "train"])
            if t is not None:
                out.update({r["id"]: t for r in rs if half(r["seed"]) == "test"})
            continue
        for sd in {r["seed"] for r in rs}:
            t = pick([r for r in rs if r["seed"] != sd])
            # 뺀 시드에만 한쪽 답이 있으면 고를 수 없다 — 문턱 없을 때 runner 처럼 0.5
            out.update({r["id"]: 0.5 if t is None else t for r in rs if r["seed"] == sd})
    return out


def load_groups(items: Path) -> dict[str, list[str]]:
    """경계 id → 묶인 노드들(items 옆 boundaries.jsonl 의 nodes). 없으면 빈 dict(항목의 node 하나로 판정)."""
    p = items.parent / "boundaries.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            b = json.loads(line)
            if b.get("nodes"):
                out[b["id"]] = b["nodes"]
    return out


def node_eval(rows: list[dict], thr_of: dict[str, float], groups: dict[str, list[str]] | None = None) -> dict:
    """thr_of 에 있는 행만 경계로 묶어 판정한다. 결과는 경계의 마지막 노드(항목 node) 이름으로 센다."""
    by_b: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        if r["id"] in thr_of:
            by_b[boundary(r["id"])][slot(r["id"])] = r
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for b, slots in by_b.items():
        node = slots[next(iter(slots))]["node"]
        checks = {s: ok for n in (groups or {}).get(b, [node]) for s, ok in NODE_CHECKS.get(n, [])}
        if not checks:
            continue
        if any(s not in slots for s in checks):  # 사람 라벨이 비어 항목이 안 만들어진 슬롯
            out[node]["skip"] += 1
            continue
        truth_ok = all(slots[s]["truth"] == ok for s, ok in checks.items())
        vlm_ok = all(answer(slots[s]["p_yes"], thr_of[slots[s]["id"]]) == ok for s, ok in checks.items())
        out[node][("ok" if truth_ok else "fail") + ("_pass" if vlm_ok else "_flag")] += 1
    return out


def rl_ws_rel(p: Path) -> str:
    """rl_ws 아래 경로는 rl_ws 기준 상대 경로로(yaml 에 홈 경로를 박지 않는다)."""
    parts = p.resolve().parts
    return str(Path(*parts[parts.index("rl_ws") + 1:])) if "rl_ws" in parts else str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--variant", default="short2")
    ap.add_argument("--split", choices=["half", "loeo"], default="half")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = load(a.results, a.items, a.variant)
    meta = json.loads(a.results.read_text().splitlines()[0]).get("meta", {})
    items_sha = hashlib.sha256(a.items.read_bytes()).hexdigest()
    if meta.get("items_sha256") not in (None, items_sha):
        print(f"items sha256 {items_sha} 가 results meta {meta['items_sha256']} 와 다르다", file=sys.stderr)
        return 1
    by_q: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    used = {s.split(":")[0] for checks in NODE_CHECKS.values() for s, _ in checks}
    thr_of = held_out_thr(rows, a.split)
    thr, report = {}, []
    tname = "train" if a.split == "half" else "전체"
    report.append(f"| 질문 | 문턱({tname}) | test yes | test no | test 균형 | 0.5 일 때 test 균형 |  ({a.split})")
    report.append("|---|---|---|---|---|---|")
    for q in sorted(by_q):
        t = pick([r for r in by_q[q] if half(r["seed"]) == "train"] if a.split == "half" else by_q[q])
        if t is None:
            continue
        thr[q] = t
        te = [r for r in by_q[q] if r["id"] in thr_of]
        pos = [r for r in te if r["truth"] == "yes"]
        neg = [r for r in te if r["truth"] == "no"]
        y = sum(r["p_yes"] >= thr_of[r["id"]] for r in pos)
        n = sum(r["p_yes"] < thr_of[r["id"]] for r in neg)
        bal = (y / len(pos) + n / len(neg)) / 2 if pos and neg else float("nan")
        y5, n5 = recall(te, 0.5)
        report.append(f"| {q}{'' if q in used else ' (판정에 안 씀)'} | {t:.2f} | {y}/{len(pos)} | {n}/{len(neg)} | {bal:.1%} | {(y5 + n5) / 2:.1%} |")
    report.append("")
    report.append("| 노드 | test 경계 | 참 실패 | 거짓 통과(실패를 통과) | 거짓 경보(성공을 실패 표시) | 슬롯 빠져 못 잼 |")
    report.append("|---|---|---|---|---|---|")
    ev = node_eval(rows, thr_of, load_groups(a.items))
    tot = collections.Counter()
    for node in NODE_CHECKS:
        c = ev.get(node)
        if not c:
            continue
        tot.update(c)
        fail = c["fail_pass"] + c["fail_flag"]
        ok = c["ok_pass"] + c["ok_flag"]
        report.append(f"| {node} | {fail + ok} | {fail} | {c['fail_pass']}/{fail} | {c['ok_flag']}/{ok} | {c['skip']} |")
    fail = tot["fail_pass"] + tot["fail_flag"]
    ok = tot["ok_pass"] + tot["ok_flag"]
    report.append(f"| 합 | {fail + ok} | {fail} | {tot['fail_pass']}/{fail} | {tot['ok_flag']}/{ok} | {tot['skip']} |")
    print("\n".join(report))
    slots = {s for checks in NODE_CHECKS.values() for s, _ in checks}
    text = {slot(r["id"]): r["question"] for r in rows if slot(r["id"]) in slots}
    cfg = {
        "about": "짧은 예/아니오 판정 설정 — vlm_supervisor.bench.calibrate 가 만든다. 손으로 고치지 말고 다시 만든다.",
        "source": {"results": rl_ws_rel(a.results), "items": rl_ws_rel(a.items), "items_sha256": items_sha,
                   "variant": a.variant, "split": a.split},
        "prompt": meta.get(a.variant),
        "choices": ["yes", "no"],
        "min_mass": 0.5,
        "thresholds": {q: thr[q] for q in sorted(used) if q in thr},
        "questions": dict(sorted(text.items())),
        "node_checks": {n: [{"slot": s, "ok": ok} for s, ok in checks] for n, checks in NODE_CHECKS.items()},
    }
    missing = sorted(slots - set(text))
    if missing:
        print(f"질문 문구가 없는 슬롯: {missing}", file=sys.stderr)
        return 1
    a.out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=200))
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

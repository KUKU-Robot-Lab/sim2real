"""짧은 예/아니오 판정의 문턱 정하기와 노드 단위 평가(11.17).

run_bench 의 short2(전·후 사진 2장) 결과(p_yes)로 질문별 문턱을 정한다. 시드를 해시로 반씩 나눠 앞 절반(train)에서 문턱을 고르고
뒤 절반(test)에서 잰다. 같은 시드의 사진은 서로 독립이 아니어서 시드 단위로 나눈다.

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
# cup_released_in_holder 는 short2 에서 "이번 동작에서 놓았나"로 읽혀(방금 놓음 0.88~0.99, 몇 단계 전 0.1~0.4) 놓기 노드에서만 쓴다.
# place_pink 끝 사진은 손이 컵을 아직 감싸 사람도 못 정한다 — 판정은 다음 노드(home_r_after_pink)로 넘긴다.
NODE_CHECKS: dict[str, list[tuple[str, str]]] = {
    "scene": [("holders_any_cup", "no")],
    "grasp_blue": [("cup_in_hand:right:SRC_BLUE", "yes"), ("cup_lifted:SRC_BLUE", "yes")],
    "grasp_shaker": [("cup_in_hand:left:SHAKER", "yes"), ("cup_lifted:SHAKER", "yes")],
    "to_tilt_start_r1": [("cup_in_hand:right:SRC_BLUE", "yes")],
    "to_tilt_start_l": [("cup_in_hand:left:SHAKER", "yes")],
    "tilt_blue": [("cup_in_hand:right:SRC_BLUE", "yes"), ("cup_in_hand:left:SHAKER", "yes")],
    "place_blue": [("cup_in_holder_b:SRC_BLUE", "yes"), ("cup_released_in_holder:SRC_BLUE", "yes")],
    "home_r_after_blue": [("cup_in_holder_b:SRC_BLUE", "yes"), ("cup_in_hand:left:SHAKER", "yes")],
    "grasp_pink": [("cup_in_hand:right:SRC_PINK", "yes"), ("cup_lifted:SRC_PINK", "yes")],
    "to_tilt_start_r2": [("cup_in_hand:right:SRC_PINK", "yes")],
    "tilt_pink": [("cup_in_hand:right:SRC_PINK", "yes"), ("cup_in_hand:left:SHAKER", "yes")],
    "place_pink": [],
    "home_r_after_pink": [("cup_in_holder_b:SRC_PINK", "yes"), ("cup_in_hand:left:SHAKER", "yes")],  # 실기 yaml 전용
    "place_shaker": [("cup_in_holder_b:SHAKER", "yes"), ("cup_released_in_holder:SHAKER", "yes")],
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


def node_eval(rows: list[dict], thr: dict[str, float], part: str) -> dict:
    by_b: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        if half(r["seed"]) == part:
            by_b[boundary(r["id"])][slot(r["id"])] = r
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for b, slots in by_b.items():
        node = slots[next(iter(slots))]["node"]
        checks = NODE_CHECKS.get(node)
        if not checks or any(s not in slots for s, _ in checks):
            continue
        truth_ok = all(slots[s]["truth"] == ok for s, ok in checks)
        qid = lambda s: s.split(":")[0]  # noqa: E731
        vlm_ok = all(answer(slots[s]["p_yes"], thr[qid(s)]) == ok for s, ok in checks)
        out[node][("ok" if truth_ok else "fail") + ("_pass" if vlm_ok else "_flag")] += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--variant", default="short2")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = load(a.results, a.items, a.variant)
    meta = json.loads(a.results.read_text().splitlines()[0]).get("meta", {})
    by_q: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    used = {s.split(":")[0] for checks in NODE_CHECKS.values() for s, _ in checks}
    thr, report = {}, []
    report.append("| 질문 | 문턱(train) | test yes 재현율 | test no 재현율 | test 균형 | 0.5 일 때 test 균형 |")
    report.append("|---|---|---|---|---|---|")
    for q in sorted(by_q):
        tr = [r for r in by_q[q] if half(r["seed"]) == "train"]
        te = [r for r in by_q[q] if half(r["seed"]) == "test"]
        t = pick(tr)
        if t is None:
            continue
        thr[q] = t
        yr, nr = recall(te, t)
        y5, n5 = recall(te, 0.5)
        report.append(f"| {q}{'' if q in used else ' (판정에 안 씀)'} | {t:.2f} | {yr:.1%} | {nr:.1%} | {(yr + nr) / 2:.1%} | {(y5 + n5) / 2:.1%} |")
    report.append("")
    report.append("| 노드 | test 경계 | 참 실패 | 거짓 통과(실패를 통과) | 거짓 경보(성공을 실패 표시) |")
    report.append("|---|---|---|---|---|")
    ev = node_eval(rows, thr, "test")
    tot = collections.Counter()
    for node in NODE_CHECKS:
        c = ev.get(node)
        if not c:
            continue
        tot.update(c)
        fail = c["fail_pass"] + c["fail_flag"]
        ok = c["ok_pass"] + c["ok_flag"]
        report.append(f"| {node} | {fail + ok} | {fail} | {c['fail_pass']}/{fail} | {c['ok_flag']}/{ok} |")
    fail = tot["fail_pass"] + tot["fail_flag"]
    ok = tot["ok_pass"] + tot["ok_flag"]
    report.append(f"| 합 | {fail + ok} | {fail} | {tot['fail_pass']}/{fail} | {tot['ok_flag']}/{ok} |")
    print("\n".join(report))
    slots = {s for checks in NODE_CHECKS.values() for s, _ in checks}
    text = {slot(r["id"]): r["question"] for r in rows if slot(r["id"]) in slots}
    cfg = {
        "about": "짧은 예/아니오 판정 설정 — vlm_supervisor.bench.calibrate 가 만든다. 손으로 고치지 말고 다시 만든다.",
        "source": {"results": str(a.results), "items_sha256": meta.get("items_sha256"), "variant": a.variant},
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

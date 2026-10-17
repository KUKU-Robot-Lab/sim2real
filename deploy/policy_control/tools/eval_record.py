#!/usr/bin/env python3
"""실기 평가 회차 기록 — 파지 s2r 확인(결정 1) · tilt 10 회(결정 4 · 11). 로봇을 움직이지 않는다(파일만 쓴다).

10.17 사용자 결정(docs/PLAN_GRASP_TILT_EVAL_2026-10-17.md):
  결정 1  파지: 들어 올린 뒤 정책이 멈추고 손이 3 s 유지되면 1 회 성공. 쪽당 5 회, 4/5 이상이면 경로 A.
  결정 4  tilt 성공 = 시뮬 판정과 같게. D4(rh_ref_s2h4 ep5000 params/env.yaml) 판정:
            success_fill_ratio 1.0(쓴 비드 전부가 받는 컵) · success_spill_max 0.25 · success_xy_thresh 0.20 m(두 컵 중심)
            · success_hold_steps 20(1/3 s) — 받는 컵에 전부 들어가면 흘림은 0 이다.
          실기 = 끝난 뒤(복귀까지 마치고 두 컵이 선 채) 받는 컵 비드 = 넣은 비드 전부(10/10). 두 컵 거리는 궤적이 정한다.
          sim 은 에피소드 중 한 순간이라도 1/3 s 성립하면 성공이고, 실기는 끝에 센다 — 복귀 중 흘림까지 실패로 세는 보수적인 쪽이다.
  결정 11 비드 10 개.

    python3 deploy/policy_control/tools/eval_record.py tilt --path A --in-rcv 10 --in-src 0 --spilled 0 \\
        --grasp-right ok --grasp-left ok --video IMG_0012.MOV --note "첫 회 비드 없이 → 다음부터"
    python3 deploy/policy_control/tools/eval_record.py grasp --side right --ok --lift-cm 7.5 --hold-s 3
    python3 deploy/policy_control/tools/eval_record.py grasp --side left --fail released_in_hold
    python3 deploy/policy_control/tools/eval_record.py summary
기록: logs/eval_real/<YYYYMMDD>/{tilt,grasp}.jsonl · 같은 이름 .csv(표로 볼 때). 회차 번호는 그날 파일의 줄 수 + 1.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Mapping

SIM2REAL = Path(__file__).resolve().parents[3]
ROOT = SIM2REAL / "logs" / "eval_real"
#: D4 rh_ref_s2h4 ep5000 params/env.yaml(hdgp pour_fabric_mimic 성공 판정) — 결정 4 의 숫자
D4_SUCCESS = {"success_fill_ratio": 1.0, "success_spill_max": 0.25, "success_xy_thresh": 0.20, "success_hold_steps": 20}
N_BEADS = 10                                   # 결정 11
PATHS = ("A", "B", "C")
GRASP_RESULTS = ("ok", "approach_miss", "pushed", "dropped_in_lift", "released_in_hold", "not_closed", "other")
GRASP_FAILS = GRASP_RESULTS[1:]


class EvalError(ValueError):
    pass


def judge_tilt(n_beads: int, in_rcv: int, in_src: int, spilled: int) -> dict:
    """비드 세기 → 판정(결정 4). 세 칸 합이 넣은 수와 같아야 한다(빠뜨린 비드가 없게)."""
    vals = {"n_beads": n_beads, "in_rcv": in_rcv, "in_src": in_src, "spilled": spilled}
    if any(int(v) != v or v < 0 for v in vals.values()) or n_beads <= 0:
        raise EvalError(f"비드 수는 0 이상 정수(넣은 수 > 0): {vals}")
    if in_rcv + in_src + spilled != n_beads:
        raise EvalError(f"받는 컵 {in_rcv} + 병 {in_src} + 흘림 {spilled} = {in_rcv + in_src + spilled} ≠ 넣은 {n_beads} — 다시 센다")
    fill, spill = in_rcv / n_beads, spilled / n_beads
    ok = fill >= D4_SUCCESS["success_fill_ratio"] and spill <= D4_SUCCESS["success_spill_max"]
    return {**vals, "fill": round(fill, 3), "spill": round(spill, 3), "success": bool(ok)}


def judge_grasp(result: str, hold_s: float | None) -> dict:
    """결정 1 — ok 이고 유지 3 s 이상이면 성공."""
    if result not in GRASP_RESULTS:
        raise EvalError(f"파지 결과 {result!r} — {GRASP_RESULTS}")
    ok = result == "ok" and hold_s is not None and float(hold_s) >= 3.0
    return {"result": result, "hold_s": hold_s, "success": bool(ok)}


def _day_dir(root: Path) -> Path:
    d = root / time.strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def append(root: Path, kind: str, row: Mapping) -> dict:
    d = _day_dir(root)
    jl = d / f"{kind}.jsonl"
    n = sum(1 for line in jl.read_text(encoding="utf-8").splitlines() if line.strip()) if jl.is_file() else 0
    row = {"trial": n + 1, "at": time.strftime("%F %T"), **row}
    with open(jl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [json.loads(x) for x in jl.read_text(encoding="utf-8").splitlines() if x.strip()]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(d / f"{kind}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return row


def latest_episode(prefix: str) -> str:
    runs = sorted((SIM2REAL / "logs" / "episode").glob(f"*_{prefix}*"), key=lambda p: p.stat().st_mtime)
    return str(runs[-1].relative_to(SIM2REAL)) if runs else ""


def summary(root: Path, day: str = "") -> dict:
    d = root / (day or time.strftime("%Y%m%d"))
    out: dict = {"day": d.name}
    for kind in ("tilt", "grasp"):
        p = d / f"{kind}.jsonl"
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []
        if kind == "tilt":
            by = {}
            for r in rows:
                by.setdefault(r["path"], []).append(r)
            out["tilt"] = {k: {"n": len(v), "success": sum(r["success"] for r in v),
                               "fill_mean": round(sum(r["fill"] for r in v) / len(v), 3),
                               "spilled_total": sum(r["spilled"] for r in v)} for k, v in by.items()}
        else:
            by = {}
            for r in rows:
                by.setdefault(r["side"], []).append(r)
            out["grasp"] = {k: {"n": len(v), "success": sum(r["success"] for r in v),
                                "fails": {f: sum(r["result"] == f for r in v) for f in GRASP_FAILS if any(r["result"] == f for r in v)}}
                            for k, v in by.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    sub = ap.add_subparsers(dest="kind", required=True)
    t = sub.add_parser("tilt", help="tilt 한 회")
    t.add_argument("--path", choices=PATHS, required=True)
    t.add_argument("--n-beads", type=int, default=N_BEADS)
    t.add_argument("--in-rcv", type=int, required=True, help="받는 컵(왼손 shaker) 안 비드")
    t.add_argument("--in-src", type=int, required=True, help="병(오른손) 안에 남은 비드")
    t.add_argument("--spilled", type=int, required=True, help="두 컵 밖(탁자 · 바닥) 비드")
    t.add_argument("--grasp-right", choices=GRASP_RESULTS, default="ok")
    t.add_argument("--grasp-left", choices=GRASP_RESULTS, default="ok")
    t.add_argument("--duration-s", type=float, default=None)
    t.add_argument("--video", default="")
    t.add_argument("--episode-dir", default="", help="없으면 logs/episode 의 가장 최근 tilt_base_* 실행 폴더")
    t.add_argument("--note", default="")
    g = sub.add_parser("grasp", help="파지 s2r 확인 한 회(결정 1)")
    g.add_argument("--side", choices=("right", "left"), required=True)
    res = g.add_mutually_exclusive_group(required=True)
    res.add_argument("--ok", action="store_true")
    res.add_argument("--fail", choices=GRASP_FAILS)
    g.add_argument("--lift-cm", type=float, default=None, help="최대 들림 높이")
    g.add_argument("--hold-s", type=float, default=None, help="멈춘 뒤 손이 유지된 시간")
    g.add_argument("--video", default="")
    g.add_argument("--episode-dir", default="")
    g.add_argument("--note", default="")
    s = sub.add_parser("summary", help="그날 기록 요약")
    s.add_argument("--day", default="")
    a = ap.parse_args(argv)
    try:
        if a.kind == "summary":
            print(json.dumps(summary(a.root, a.day), ensure_ascii=False, indent=1))
            return 0
        if a.kind == "tilt":
            j = judge_tilt(a.n_beads, a.in_rcv, a.in_src, a.spilled)
            row = append(a.root, "tilt", {"path": a.path, **j, "grasp_right": a.grasp_right, "grasp_left": a.grasp_left,
                                          "duration_s": a.duration_s, "video": a.video,
                                          "episode_dir": a.episode_dir or latest_episode("tilt_base"), "note": a.note,
                                          "criterion": "끝에 받는 컵 = 넣은 비드 전부(D4 success_fill_ratio 1.0)"})
        else:
            j = judge_grasp("ok" if a.ok else a.fail, a.hold_s)
            row = append(a.root, "grasp", {"side": a.side, **j, "lift_cm": a.lift_cm, "video": a.video,
                                           "episode_dir": a.episode_dir or latest_episode("grasp_"), "note": a.note})
    except EvalError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2
    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

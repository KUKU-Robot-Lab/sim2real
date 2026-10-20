#!/usr/bin/env python3
"""tilt 시작 자세(세팅 자세)로 — 두 팔을 한 팔씩(오른팔 → 왼팔). 상황판 control 탭 [tilt_now] 가 rehome 처럼 부른다.
★10.19 사용자 "내가 말한 tilt_now 는 틸트 초기 자세(세팅 자세)로 가라고 하는 것(rehome 처럼)". 그 자세에서 컵을 쥐인 뒤
붓기는 [tilt_pour](tilt_pour.py) · tilt 에피소드.

끝 자세 = 붓기 시작 이정표(자산 policy.yaml start_pose = config/homes/rh56f1_pour_start.yaml, hdgp D4 기준 행 single734_t120 행 0 —
두 컵을 (0.25, ∓0.12, 0.41) 에 똑바로 든 자세). 에피소드 tilt_base_* 의 to_tilt_start_r · to_tilt_start_l(episode_ros._run_planned
carry)과 같은 두 걸음을 팔마다:

  1. plan_carry.py --sides <팔> — 실측 → ready_carry(hdgp Mover) 계획 · 검사(계획 · 도착 · 탁자 여유 · 관절 속도 · 두 팔 사이).
     하나라도 걸리면 rc 2 — 그 팔부터 움직이지 않는다(오른팔이 걸리면 왼팔도 안 간다).
  2. replay_to_pd.py — 그 팔 관절 7 개만 재생(다른 팔 · 두 손은 pd 가 붙든 채). 재생 직전 실측이 계획 첫 자세에서
     start_gap_rad 넘게 벗어나면 replay 가 거부한다.

손은 건드리지 않는다(빈손이든 쥔 손이든 그대로 — 손 안 컵 자리를 가상 컵으로 두고 탁자 · 두 팔 사이를 검사한다).
기록: logs/tilt/<날짜_시각>_setting/carry_<팔>/(check.json · carry_cmd.npz).

    python3 deploy/policy_control/tools/tilt_now.py                 # 계획 · 검사만(왼팔은 오른팔이 끝 자세에 갔다고 보고 계획)
    python3 deploy/policy_control/tools/tilt_now.py --execute       # 오른팔 계획 → 재생 → 왼팔 계획 → 재생
종료 코드: 0 = 끝 · 2 = 검사 실패(그 팔부터 움직이지 않음) · 3 = 핀 · 기준 파일 불일치 · 그 밖 = 계획 · 재생 실패.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from tilt_pour import ASSET, SIM2REAL, TOOLS, fresh_dir, run

SIDES = ("right", "left")
ARM_JOINTS = {s: [f"{s[0]}_aj_{i}" for i in range(1, 8)] for s in SIDES}


def current_q() -> dict[str, list[float]]:
    """실측 두 팔(계획만 모드에서 왼팔 계획의 출발 — 실기에서만 부른다)."""
    sys.path.insert(0, str(TOOLS))
    from plan_tilt import current_q as measure
    return measure("")


def fail_summary(out: Path) -> str:
    """plan_carry check.json 의 걸린 항목."""
    try:
        chk = json.loads((out / "check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    bad = []
    for side, c in (chk.get("checks") or {}).items():
        if c.get("ok", True):
            continue
        why = []
        if not c.get("plan_ok", True):
            why.append(f"계획 실패(앵커 {c.get('anchor_err_mm')} mm · {c.get('anchor_err_deg')}°)")
        if not c.get("arrived", True):
            why.append(f"도착 못 함(관절 {c.get('end_q_err_rad')} rad · 컵 {c.get('end_cup_err_mm')} mm)")
        if float(c.get("table_clear_m", 1.0)) < 0.01:
            why.append(f"탁자 여유 {c.get('table_clear_m')} m < 0.01")
        if float(c.get("vmax_rad_s", 0.0)) > 2.0:
            why.append(f"관절 속도 {c.get('vmax_rad_s')} rad/s > 2.0")
        bad.append(f"{side}: {' · '.join(why) or '검사 실패'}")
    arms = chk.get("arms_clear") or {}
    if arms and not arms.get("ok", True):
        bad.append(f"두 팔 사이(팔 점 {arms.get('min_arm_point_m')} m · 컵 {arms.get('min_cup_gap_m')} m)")
    return " · ".join(bad)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--out-root", type=Path, default=SIM2REAL / "logs" / "tilt")
    ap.add_argument("--sides", default="right,left", help="한 팔씩 이 순서로(right · left · right,left)")
    ap.add_argument("--rate-scale", type=float, default=1.0)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    sides = [x.strip() for x in a.sides.split(",") if x.strip()]
    if not sides or any(x not in SIDES for x in sides) or len(set(sides)) != len(sides):
        print(f"✗ --sides {a.sides!r} — right · left 중")
        return 1
    if not 0.1 <= a.rate_scale <= 1.0:
        print(f"✗ --rate-scale {a.rate_scale} — 0.1~1.0")
        return 1
    cfg = yaml.safe_load((a.asset / "policy.yaml").read_text(encoding="utf-8")) or {}
    gap = float(cfg.get("start_gap_rad", 0.05))
    goal = yaml.safe_load((SIM2REAL / cfg["start_pose"]).read_text(encoding="utf-8"))
    out = fresh_dir(a.out_root, "setting")
    q = None if a.execute else current_q()            # 계획만: 앞 팔은 끝 자세에 갔다고 보고 다음 팔을 계획한다
    for k, side in enumerate(sides):
        o = out / f"carry_{side}"
        argv_plan = [sys.executable, str(TOOLS / "plan_carry.py"), "--asset", str(a.asset), "--out", str(o), "--sides", side]
        if q is not None:
            if k > 0:
                for done in sides[:k]:
                    q[done] = [float(v) for v in goal[done]]
            out.mkdir(parents=True, exist_ok=True)
            qj = out / f"q_{side}.json"
            qj.write_text(json.dumps(q))
            argv_plan += ["--q-json", str(qj)]
        rc = run(argv_plan)
        if rc != 0:
            print(f"[tilt_now] {side} 계획 · 검사 rc {rc} — {side} 부터 움직이지 않았다 · 걸린 항목: "
                  f"{fail_summary(o) or '(check.json 없음)'} · {o}/check.json", flush=True)
            return rc
        if not a.execute:
            print(f"[tilt_now] {side} 검사 통과(계획만) · {o}", flush=True)
            continue
        rc = run([sys.executable, str(TOOLS / "replay_to_pd.py"), "--npz", str(o / "carry_cmd.npz"),
                  "--joints", ",".join(ARM_JOINTS[side]), "--rate-scale", repr(float(a.rate_scale)),
                  "--max-start-gap", repr(gap), "--execute"])
        if rc != 0:
            print(f"[tilt_now] {side} 재생 실패 rc {rc} · {o}", flush=True)
            return rc
        print(f"[tilt_now] {side} 세팅 자세 도착 · {o}", flush=True)
    print(f"[tilt_now] {'계획만 — --execute 로 재생' if not a.execute else '두 팔 세팅 자세 끝'} · {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

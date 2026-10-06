#!/usr/bin/env python3
"""정책이 끝난 자리 → 정책 시작 자세(계약 home_arm = 홈 경로 끝)까지 경로를 실측에서 계획한다. 구독만 한다(발행 없음).

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/plan_rehome.py --side left
    → logs/policy_control/rehome_<side>.npz (replay_to_pd 가 재생한다)

09.28 사용자: "정책 진행하고 종료한 뒤에 다시 홈자세로 하고 반복해야할수도". 그날 dg5f_m/cup_pick/left_a01 은 팔을 1.1 rad 옮겨
컵 옆 낮은 곳에 손을 두고 멈췄다. 거기서 시작 자세까지 관절공간 직선은 새끼 손끝이 상판을 지난다(검사기 최악 −0.21 m) —
그래서 직선(goto_home · plan_approach_to_start)이 아니라 홈 경로를 만든 계획기(plan_home_path: 직선 → 실패하면 RRT,
같은 세계 · 여유 2 cm · 실측 손 자세)를 실측 시작 자세로 부른다. 그날 자세에서 16 s 에 RRT 가 여유 2.5 cm 경로를 냈다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM2REAL = HERE.parents[2]
SAMPLE = SIM2REAL / "deploy" / "s2r_console" / "tools" / "sample_joints.py"
RL_WS = SIM2REAL.parent
#: plan_home_path 기본값은 DG-5F 세계다(10.01 — 빼먹으면 DG-5F 로 검사한다). 로봇마다 세계 · 계약 · 프로필 · pd 를 넘긴다.
#: rh56f1 = 미션 저장 홈 경로(paths/home_rh56f1_*.npz)를 만든 것과 같은 세계(10.04 에피소드 실행기 궤적).
ROBOTS = {
    "dg5f": [],
    "rh56f1": ["--urdf", str(RL_WS / "hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"),
               "--contract", str(SIM2REAL / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"),
               "--profile", str(RL_WS / "robot_control/src/robot_control/profiles/openarm_rh56f1.yaml"),
               "--env-yaml", str(SIM2REAL / "deploy/policies/rh56f1/aglt/right_cyl60g/params/env.yaml"),
               "--pd-config", str(SIM2REAL / "deploy/policy_control/config/pd_rh56f1.yaml")],
}


#: 실측에서 기다릴 관절(robot 별) — 손 상태가 늦게 붙으면 '손 관절 상태가 없다'로 계획이 멈춘다(10.04 에피소드 연속 실행)
HAND_NEED = {"rh56f1": ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1"), "dg5f": ()}


def measure(side: str = "right", robot: str = "dg5f") -> dict[str, float]:
    p = side[0]
    need = [f"{p}_aj_{i}" for i in range(1, 8)] + [f"{p}_hj_{j}" for j in HAND_NEED.get(robot, ())]
    out = subprocess.run([sys.executable, str(SAMPLE), "--seconds", "0.6", "--need", ",".join(need)],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"✗ 관절 상태를 못 읽었다: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])["q"]


def planner_argv(side: str, q: dict[str, float], out: Path, robot: str = "dg5f") -> list[str]:
    """실측 → plan_home_path 인자. 팔 7 관절이 없으면 SystemExit, 손은 있는 것만 넘긴다(측정 손 자세로 검사)."""
    p = side[0]
    arm = [f"{p}_aj_{i}" for i in range(1, 8)]
    missing = [j for j in arm if j not in q]
    if missing:
        raise SystemExit(f"✗ 관절 상태 없음 {missing} — 드라이버가 떠 있는가")
    if all(q[j] == 0.0 for j in arm):
        raise SystemExit("✗ 팔 관절이 전부 정확히 0.0 — 엔코더를 못 읽는 것이다")
    hand = {k: v for k, v in q.items() if k.startswith(f"{p}_hj_")}
    if not hand:
        raise SystemExit(f"✗ 손 관절 상태가 없다({side}) — 손 드라이버가 떠 있는가(손 자세로 충돌을 검사한다)")
    return [sys.executable, str(HERE / "plan_home_path.py"), "--side", side,
            # `--opt=값` 으로 붙인다 — 값이 '-0.1…' 처럼 음수로 시작하면 argparse 가 옵션으로 읽는다(09.28 실기 rc=2)
            "--start=" + ",".join(f"{q[j]:.5f}" for j in arm), "--goal", "contract",
            "--hand-start", "measured", "--hand-q=" + ",".join(f"{k}={v:.5f}" for k, v in sorted(hand.items())),
            "--out", str(out), *ROBOTS[robot]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--robot", choices=sorted(ROBOTS), default="dg5f", help="계획 세계(자산 · 계약 · 프로필 · pd)")
    args = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        raise SystemExit("✗ ROS_DOMAIN_ID 가 비었거나 0 — 거부")
    out = args.out or (SIM2REAL / "logs" / "policy_control" / f"rehome_{args.side}.npz")
    if out.exists():
        out.unlink()                         # 옛 계획이 남아 있으면 계획이 실패해도 재생 단계가 그것을 튼다
    argv = planner_argv(args.side, measure(args.side, args.robot), out, args.robot)
    print(f"[rehome] {args.side} 실측 → 정책 시작 자세 계획 (직선 → 실패하면 RRT, 실측 손 · 여유 2 cm)", flush=True)
    rc = subprocess.run(argv).returncode
    if rc != 0 or not out.exists():
        print("✗ 되돌아올 경로를 못 만들었다 — 팔을 움직이지 말 것(비상 복귀 reset_* 또는 사람이 본다)", file=sys.stderr)
        return rc or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

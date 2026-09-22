#!/usr/bin/env python3
"""지금 팔 관절을 재서 그 자리에서 계약 홈(정책 에피소드 시작 자세)까지 충돌 없는 경로를 계획한다. 구독만 한다.

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/plan_home_from_robot.py --side right

1. `/joint_states` 를 잠깐 구독해 이 팔의 7관절을 잰다(deploy/s2r_console/tools/sample_joints.py).
2. 그 값을 시작점으로 plan_home_path.py 를 부른다 — 손은 계약 home_hand · 전부 0 두 자세, 반대 팔은 홈 · 0 두 자세
   모두에서 통과해야 한다(손 · 반대 팔의 실제 자세를 가정하지 않는다). RRT 가 실패하면 시드를 바꿔 몇 번 더.
3. 결과를 logs/policy_control/home_path_<side>_current.npz 에 쓴다 — 미션의 재생 · 되짚기 단계가 이 파일을 쓴다.

09.22: 모델의 차렷(관절 0)은 손끝이 받침판 속이라 0 에서 계획한 경로의 첫 구간은 검증이 약했다. 실측 시작점으로 다시 계획한다.
경로를 못 찾으면 0 이 아닌 값으로 끝난다 — 그 단계에서 멈추고 팔은 움직이지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[3]
SAMPLE = SIM2REAL / "deploy" / "s2r_console" / "tools" / "sample_joints.py"
PLANNER = Path(__file__).resolve().parent / "plan_home_path.py"
SEEDS = tuple(range(12))    # 옆 벌림을 묶으면 RRT 성공률이 낮다(09.22 실측 8 번 중 1 번) — 시드를 넉넉히
MAX_ABDUCTION = 0.9         # j2 상한 [rad] — 팔을 옆으로 크게 벌리지 않고 j1·j4 위주로(09.22 사용자). 이전 경로는 약 1.0
ARGPARSE_ERROR = 2          # argparse 가 인자 오류로 끝낼 때의 코드


def current_path(side: str) -> Path:
    return SIM2REAL / "logs" / "policy_control" / f"home_path_{side}_current.npz"


def measure(side: str) -> list[float]:
    out = subprocess.run([sys.executable, str(SAMPLE), "--seconds", "0.6"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"[plan] 관절 상태를 못 읽었다: {out.stderr.strip()[-300:]}")
    q = json.loads(out.stdout.strip().splitlines()[-1])["q"]
    names = [f"{side[0]}_aj_{i}" for i in range(1, 8)]
    missing = [n for n in names if n not in q]
    if missing:
        raise SystemExit(f"[plan] {side} 팔 관절 상태가 없다: {missing} — 드라이버가 떠 있는가")
    return [float(q[n]) for n in names]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--with-cup", action="store_true", help="컵(스폰 중심)도 장애물로")
    args = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        raise SystemExit("[plan] ROS_DOMAIN_ID 가 비었거나 0 — 거부")
    start = measure(args.side)
    print(f"[plan] {args.side} 실측 시작 자세 {[round(v, 4) for v in start]}", flush=True)
    out = current_path(args.side)
    # `--start=` 로 붙인다 — 음수로 시작하면 argparse 가 값이 아니라 옵션으로 읽는다(09.22 왼팔 fake 에서 밟았다)
    base = [sys.executable, str(PLANNER), "--side", args.side, "--start=" + ",".join(f"{v:.6f}" for v in start),
            "--goal", "contract", "--hand-start", "both", "--other-arm", "both", "--out", str(out),
            f"--max-abduction={MAX_ABDUCTION}"]
    if args.with_cup:
        base.append("--with-cup")
    for seed in SEEDS:
        rc = subprocess.run([*base, "--seed", str(seed)]).returncode
        if rc == 0:
            print(f"[plan] 경로 → {out.relative_to(SIM2REAL)} (seed {seed})", flush=True)
            return 0
        if rc == ARGPARSE_ERROR:                     # 시드를 바꿔도 소용없다 — 호출이 틀렸다
            print("[plan] ✗ 계획기 인자 오류 — 이 도구의 결함이다", flush=True)
            return 3
        print(f"[plan] seed {seed} 실패 — 다음 시드", flush=True)
    print("[plan] ✗ 충돌 없는 경로를 찾지 못했다 — 팔을 움직이지 말 것", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())

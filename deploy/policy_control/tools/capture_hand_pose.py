#!/usr/bin/env python3
"""지금 손 자세를 **경로 기준 자세**로 기록한다 — pd 설정 `hand_path_pose` 에 쓴다. 구독만 한다(발행 없음).

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/capture_hand_pose.py --side both            # 보기만
    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/capture_hand_pose.py --side both --write    # 설정에 기록

09.23 사용자: "오른손/왼손 모두 주먹과 비슷한 자세로 내가 만들 거고, 팔처럼 트래젝토리 세팅 자세로 두 손들도
만든 다음에 실행되게 하면 되잖아?" — 그 순서다. 손을 원하는 모양(주먹)으로 만들어 두고 이 도구로 기록하면,
그 값을 **경로 계획과 실기가 같이** 쓴다.

  ① 실측 손을 읽는다(양손 또는 한 손)
  ② 벤더 관절 한계에서 `--margin` 안쪽으로 물린다 — 한계값 그대로 두면 pd 가 기계 끝점으로 민다(09.23 손가락 꺾임)
  ③ `--write` 면 short pd 설정 3개(원본 · 발행 사본 · fake)에 같은 값을 쓴다
  ④ 기록한 뒤에는 **경로를 다시 계획해야 한다** — 경로의 여유는 손 모양으로 계산한 값이다

`plan_home_path --hand-start pd` 와 `pd/hand_path` 가 이 값을 읽는다.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SIM2REAL = HERE.parents[2]
SAMPLE = SIM2REAL / "deploy" / "s2r_console" / "tools" / "sample_joints.py"
CONFIG_DIR = SIM2REAL / "deploy" / "policy_control" / "config"
CONFIGS = ("pd_dg5f_m_short.yaml", "pd_dg5f_m_short_exec.yaml", "pd_dg5f_m_short_fake.yaml")
PROFILE = SIM2REAL.parent / "robot_control" / "src" / "robot_control" / "profiles" / "openarm_tesollo.yaml"
MARGIN = 0.05                   # [rad] 관절 한계에서 물릴 거리


def measure() -> dict[str, float]:
    out = subprocess.run([sys.executable, str(SAMPLE), "--seconds", "0.8"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"✗ 관절 상태를 못 읽었다: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])["q"]


def limits() -> dict[str, tuple[float, float]]:
    prof = yaml.safe_load(PROFILE.read_text())
    return {j["canonical"]: (float(j["lower"]), float(j["upper"])) for j in prof["joints"] if "lower" in j}


def clamped(pose: dict[str, float], lim: dict[str, tuple[float, float]], margin: float) -> tuple[dict, list]:
    """(한계 안쪽으로 물린 자세, 물린 관절 목록). 순수."""
    out, moved = {}, []
    for j, v in sorted(pose.items()):
        lo, hi = lim.get(j, (-np.inf, np.inf))
        w = float(np.clip(v, lo + margin, hi - margin)) if lo + margin <= hi - margin else (lo + hi) / 2
        out[j] = round(w, 3)
        if abs(w - v) > 1e-9:
            moved.append((j, round(v, 3), out[j]))
    return out, moved


def block_text(pose: dict[str, dict[str, float]]) -> str:
    return "".join(f"  {side}:\n" + "".join(f"    {j}: {v}\n" for j, v in pose[side].items())
                   for side in sorted(pose))


def write_configs(pose: dict[str, dict[str, float]]) -> list[str]:
    """pd 설정 3개의 `hand_path_pose` 블록을 같은 값으로 바꾼다. 주석은 그대로 둔다."""
    done = []
    for name in CONFIGS:
        p = CONFIG_DIR / name
        s = p.read_text()
        m = re.search(r"^hand_path_pose:.*?\n(?:^ .*\n|^#.*\n)*", s, re.M)
        if m is None:
            raise SystemExit(f"✗ {p}: hand_path_pose 블록을 찾지 못했다")
        head = [ln for ln in m.group(0).splitlines(keepends=True) if ln.startswith(("hand_path_pose:", "  #"))]
        p.write_text(s[:m.start()] + "".join(head) + block_text(pose) + s[m.end():])
        done.append(name)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left", "both"), default="both")
    ap.add_argument("--margin", type=float, default=MARGIN, help="관절 한계에서 물릴 거리 [rad]")
    ap.add_argument("--write", action="store_true", help="pd 설정 3개에 기록한다 — 없으면 보여만 준다")
    args = ap.parse_args()

    q = measure()
    lim = limits()
    sides = ("right", "left") if args.side == "both" else (args.side,)
    pose: dict[str, dict[str, float]] = {}
    for side in sides:
        got = {j: v for j, v in q.items() if j.startswith(f"{side[0]}_hj_")}
        if len(got) < 20:
            raise SystemExit(f"✗ {side} 손 관절이 {len(got)} 개뿐이다 — 손 드라이버가 떠 있는가")
        pose[side], moved = clamped(got, lim, args.margin)
        print(f"[capture] {side} {len(pose[side])} 관절 · 한계 안쪽으로 물린 것 {len(moved)}")
        for j, was, now in moved:
            print(f"    {j} {was:+.3f} → {now:+.3f}")
    print("\n" + block_text(pose), end="")
    if not args.write:
        print("[capture] 보기만 — --write 를 붙이면 pd 설정에 기록한다")
        return 0

    old = {}
    for side in sides:                                   # 기록 전 값 — 얼마나 달라지는지 보여 준다
        raw = (yaml.safe_load((CONFIG_DIR / CONFIGS[0]).read_text()).get("hand_path_pose") or {}).get(side) or {}
        old[side] = raw
    if len(sides) == 1:                                  # 한 손만 기록해도 반대 손 값은 지키지 않는다
        other = "left" if sides[0] == "right" else "right"
        keep = (yaml.safe_load((CONFIG_DIR / CONFIGS[0]).read_text()).get("hand_path_pose") or {}).get(other)
        if keep:
            pose[other] = {str(j): float(v) for j, v in keep.items()}
    print("[capture] 기록:", ", ".join(write_configs(pose)))
    for side in sides:
        if old[side]:
            d = max(abs(pose[side][j] - float(old[side].get(j, pose[side][j]))) for j in pose[side])
            print(f"[capture] {side} 이전 기준 자세와 최대 차이 {d:.3f} rad")
    print("★경로를 다시 계획할 것 — 경로의 여유는 손 모양으로 계산한 값이다:\n"
          "   python3 deploy/policy_control/tools/plan_home_path.py --side right --start=<실측 7관절> "
          "--goal contract --hand-start pd --other-arm both --via j1j4 "
          "--out deploy/policy_control/paths/home_right.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())

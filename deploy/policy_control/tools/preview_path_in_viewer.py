#!/usr/bin/env python3
"""plan_home_path.py 가 만든 npz 를 읽기 전용 Isaac 뷰어(robot/isaacsim_bridge/viewer)에 UDP 로 흘려 보여 준다.

ROS 없음 · 실기 무관. 뷰어는 127.0.0.1:47811 에서 JSON 패킷(viewer/packet.py 형식)을 받아 관절만 쓴다.
패킷 = 움직이는 팔 프레임 + 반대팔(계약 home_arm 또는 0) + 양손(계약 home_hand) + 머리 0.

    .venv/bin/python deploy/policy_control/tools/preview_path_in_viewer.py --npz logs/policy_control/home_path_right.npz
    ... --speed 4            # 4배속
    ... --hold-frac 0.5      # 경로 중간 자세만 반복 송신(스크린샷용), --hold-sec 동안
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

SIM2REAL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM2REAL / "robot/isaacsim_bridge/viewer"))
import packet  # noqa: E402  (뷰어의 형식 정의를 그대로 쓴다 — 읽기만)

CONTRACT_DEFAULT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m-short_bi_rl/deploy_contract.json"
SEND_HZ = 30.0


def fixed_joints(contract: dict, side: str, other_arm: str) -> dict[str, float]:
    other = "left" if side == "right" else "right"
    sd, od = contract["sides"][side], contract["sides"][other]
    out = {**sd["home_hand"], **od["home_hand"], "head_j_pan": 0.0, "head_j_tilt": 0.0}
    arm = od["home_arm"] if other_arm == "home" else [0.0] * len(od["arm_joints"])
    out.update(zip(od["arm_joints"], arm))
    return out


def packets(frames: np.ndarray, joints: list[str], fixed: dict[str, float]) -> list[dict[str, float]]:
    return [{**fixed, **dict(zip(joints, map(float, q)))} for q in frames]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", type=Path, default=SIM2REAL / "logs/policy_control/home_path_right.npz")
    ap.add_argument("--contract", type=Path, default=CONTRACT_DEFAULT)
    ap.add_argument("--other-arm", choices=("home", "zero"), default="home")
    ap.add_argument("--host", default=packet.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=packet.DEFAULT_PORT + 1,
                    help="뷰어의 미리보기 포트(기본 = 실기 포트 + 1) — 실기 관절 값보다 우선해서 보인다")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속(>0)")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--hold-frac", type=float, default=None, help="0..1 — 이 위치 프레임 하나만 반복 송신")
    ap.add_argument("--hold-sec", type=float, default=60.0)
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("로컬 뷰어 전용 — host 는 127.0.0.1")
    if args.speed <= 0:
        raise SystemExit("--speed > 0")

    d = np.load(args.npz)
    frames = np.asarray(d["arm_target"], dtype=float)
    dt = float(d["meta_step_dt"])
    joints = [str(j) for j in d["meta_joints"]] if "meta_joints" in d else None
    if not joints:
        raise SystemExit("npz 에 meta_joints 가 없다")
    side = "right" if joints[0].startswith("r_") else "left"
    with open(args.contract) as f:
        fixed = fixed_joints(json.load(f), side, args.other_arm)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    if args.hold_frac is not None:
        k = int(round(np.clip(args.hold_frac, 0, 1) * (len(frames) - 1)))
        pkt = packets(frames[k:k + 1], joints, fixed)[0]
        print(f"[preview] 프레임 {k}/{len(frames) - 1} 고정 송신 {args.hold_sec:.0f} s -> udp://{addr[0]}:{addr[1]} "
              f"q={np.round(frames[k], 4).tolist()}", flush=True)
        t_end = time.time() + args.hold_sec
        while time.time() < t_end:
            sock.sendto(packet.encode(time.time(), pkt), addr)
            time.sleep(1.0 / SEND_HZ)
        return 0

    pk = packets(frames, joints, fixed)
    period = dt / args.speed
    print(f"[preview] {len(frames)} 프레임 · {(len(frames) - 1) * dt:.1f} s 경로 · {args.speed:g}배속 -> "
          f"udp://{addr[0]}:{addr[1]}", flush=True)
    for _ in range(args.loops):
        for p in pk:
            sock.sendto(packet.encode(time.time(), p), addr)
            time.sleep(period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

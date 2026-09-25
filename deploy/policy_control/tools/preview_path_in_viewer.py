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
PD_CONFIG_DEFAULT = Path(__file__).resolve().parent.parent / "config" / "pd_dg5f_m_short.yaml"
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


def hand_pose(spec: str | None, side: str, contract_path: Path, pd_config: Path) -> dict[str, float]:
    """`--hand` 값 → 그 팔의 손 관절 {이름: 값}. 빈 값·keep 이면 빈 dict(보내지 않는다).

    09.22 교훈: 손을 말없이 계약 홈(편 손)으로 덮어 보내면 화면에서 손가락이 갑자기 펴진다.
    그래서 기본은 **보내지 않는 것**이고, 보낼 때는 어느 자세인지 이름으로 고른다.
    """
    if not spec or spec == "keep":
        return {}
    if spec == "contract":
        with open(contract_path) as f:
            return {k: float(v) for k, v in json.load(f)["sides"][side]["home_hand"].items()}
    if spec == "pd":
        import yaml  # noqa: PLC0415
        pose = (yaml.safe_load(Path(pd_config).read_text()).get("hand_path_pose") or {}).get(side)
        if not pose:
            raise SystemExit(f"✗ {pd_config} 에 hand_path_pose.{side} 가 없다")
        return {str(k): float(v) for k, v in pose.items()}
    out = {}
    for item in spec.split(","):
        if item.strip():
            k, _, v = item.partition("=")
            out[k.strip()] = float(v)
    if not out:
        raise SystemExit(f"✗ --hand 를 알 수 없다: {spec!r}")
    return out


def blend(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    """손 자세 a → b 를 t(0..1) 로 섞는다. 순수."""
    return {k: float(a.get(k, v)) + (float(v) - float(a.get(k, v))) * t for k, v in b.items()}


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
    ap.add_argument("--reverse", action="store_true",
                    help="끝에서 앞으로 — 리셋(복귀)은 저장 경로를 되짚는 것이라 그 모습을 보려면 이것이 필요하다")
    ap.add_argument("--hand", default="keep",
                    help="이동하는 동안 **그 팔의 손** 자세: keep(기본 — 뷰어가 가진 값 그대로) · "
                         "pd(pd yaml hand_path_pose = 주먹, 봉투 구 안) · contract(계약 home_hand = 편 손) · "
                         "'r_hj_index_2=1.2,...' 실측 CSV. 팔이 움직이는 동안 손은 그대로 있는다(pd home_hand: keep)")
    ap.add_argument("--hand-from", default=None,
                    help="--hand 와 같은 형식. 주면 맨 앞 --hand-ramp-s 동안 이 자세에서 --hand 자세로 옮겨 보여 준다 "
                         "(복귀 전 pd/hand_rest 처럼 손이 먼저 움직이는 구간)")
    ap.add_argument("--hand-ramp-s", type=float, default=2.0, help="--hand-from 구간 길이 [s]")
    ap.add_argument("--pd-config", type=Path, default=PD_CONFIG_DEFAULT,
                    help="--hand pd 가 읽을 pd yaml (hand_path_pose)")
    ap.add_argument("--hold-frac", type=float, default=None, help="0..1 — 이 위치 프레임 하나만 반복 송신")
    ap.add_argument("--hold-sec", type=float, default=60.0)
    ap.add_argument("--with-fixed", action="store_true",
                    help="손 · 반대 팔 · 목도 계약 홈 값으로 덮어 보낸다. 기본은 **움직이는 팔 관절만** — 나머지는 뷰어가 실기 값을 "
                         "그대로 둔다(09.22: 손을 계약 홈(편 손)으로 덮어 보내 손가락이 갑자기 펴진 것처럼 보였다)")
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("로컬 뷰어 전용 — host 는 127.0.0.1")
    if args.speed <= 0:
        raise SystemExit("--speed > 0")

    d = np.load(args.npz)
    frames = np.asarray(d["arm_target"], dtype=float)
    if args.reverse:
        frames = frames[::-1]
    dt = float(d["meta_step_dt"])
    joints = [str(j) for j in d["meta_joints"]] if "meta_joints" in d else None
    if not joints:
        raise SystemExit("npz 에 meta_joints 가 없다")
    side = "right" if joints[0].startswith("r_") else "left"
    fixed = {}
    if args.with_fixed:
        with open(args.contract) as f:
            fixed = fixed_joints(json.load(f), side, args.other_arm)
    hand = hand_pose(args.hand, side, args.contract, args.pd_config)
    hand_from = hand_pose(args.hand_from, side, args.contract, args.pd_config) if args.hand_from else None
    fixed = {**fixed, **hand}
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
    if hand_from is not None:                 # 손이 먼저 움직이는 구간 — 팔은 첫 프레임에 선 채로
        n = max(1, int(round(args.hand_ramp_s * SEND_HZ)))
        arm0 = dict(zip(joints, map(float, frames[0])))
        head = [{**fixed, **blend(hand_from, hand, i / max(1, n - 1)), **arm0} for i in range(n)]
        print(f"[preview] 손 이동 {args.hand_from} → {args.hand} · {args.hand_ramp_s:g} s (팔은 정지)", flush=True)
        for p in head:
            sock.sendto(packet.encode(time.time(), p), addr)
            time.sleep(1.0 / SEND_HZ)
    print(f"[preview] {len(frames)} 프레임 · {(len(frames) - 1) * dt:.1f} s 경로 · {args.speed:g}배속"
          f"{' · 역재생(리셋)' if args.reverse else ''} -> udp://{addr[0]}:{addr[1]}", flush=True)
    for _ in range(args.loops):
        for p in pk:
            sock.sendto(packet.encode(time.time(), p), addr)
            time.sleep(period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

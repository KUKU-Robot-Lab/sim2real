#!/usr/bin/env python3
"""board_viewer 시험용 가짜 패널 — ROS 없이 board_packet 계약대로 UDP 만 보낸다(로봇에는 아무것도 안 간다).

오른팔 홈(deploy_contract home_arm, 검지만 편 손)에서 검지 끝을 ChArUco 교차점 bc08 위 3 cm 로 옮겼다가 천천히 내려 닿고,
다시 홈으로 돌아가기를 되풀이한다. q_cmd 는 q_show 보다 0.4 s 앞선 자세(고스트), tip = 자산 URDF FK.

    python3 fake_board_sender.py [--port 47813] [--seconds 0]
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S2R = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(S2R / "deploy/policy_control"))
import board_packet  # noqa: E402
from policy_control.fk_numpy import UrdfTree  # noqa: E402

URDF = S2R.parent / "hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
CONTRACT = S2R / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
ARM = [f"r_aj_{i}" for i in range(1, 8)]
HAND = ["r_hj_thumb_1", "r_hj_thumb_2", "r_hj_index_1", "r_hj_middle_1", "r_hj_ring_1", "r_hj_pinky_1"]
POINT = [1.57, 0.0, 0.0, 1.4, 1.4, 1.4]           # board_jog.POINT
# 보드: 인쇄면 위(보드 z = base −z), 탁자 0.205 위 3 mm 판. 원점(바깥 모서리) (0.10, −0.24) — x 0.10~0.31 이라 홀더 2(링 x 0.339~)와 안 겹친다.
T_BOARD = np.array([[1, 0, 0, 0.10], [0, -1, 0, -0.24], [0, 0, -1, 0.208], [0, 0, 0, 1]], float)
TARGET = T_BOARD[:3, :3] @ np.array([0.09, 0.06, 0.0]) + T_BOARD[:3, 3]     # bc08 = ((8%6+1)·0.03, (8//6+1)·0.03)
LEAD_S, PERIOD_S, HZ = 0.4, 16.0, 30.0


def tip(tree: UrdfTree, q: np.ndarray) -> np.ndarray:
    full = tree.expand_mimic(dict(zip(ARM, q)) | dict(zip(HAND, POINT)))
    return tree.transforms(["r_hl_index_tip"], full)["r_hl_index_tip"][:3, 3]


def ik(tree: UrdfTree, q0: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """위치만 맞추는 DLS IK(시험 자세 만들기용)."""
    q = q0.copy()
    for _ in range(100):
        e = goal - tip(tree, q)
        if np.linalg.norm(e) < 2e-4:
            return q
        J = np.stack([(tip(tree, q + d) - tip(tree, q)) / 1e-6 for d in np.eye(7) * 1e-6], 1)
        q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
    raise RuntimeError(f"IK 수렴 실패: 남은 {np.linalg.norm(goal - tip(tree, q)) * 1000:.1f} mm")


def path(home: np.ndarray, above: np.ndarray, touch: np.ndarray, s: float) -> np.ndarray:
    """주기 안 위치 s∈[0,1): 홈→위 3 cm(0.4) · 내려감(0.25) · 머묾(0.1) · 홈으로(0.25)."""
    def mix(a, b, u):
        u = float(np.clip(u, 0.0, 1.0))
        return a + (b - a) * (3 * u * u - 2 * u ** 3)
    if s < 0.40:
        return mix(home, above, s / 0.40)
    if s < 0.65:
        return mix(above, touch, (s - 0.40) / 0.25)
    if s < 0.75:
        return touch
    return mix(touch, home, (s - 0.75) / 0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=board_packet.DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=0.0, help=">0 이면 이 시간 뒤 끝")
    a = ap.parse_args()
    tree = UrdfTree(URDF)
    home = np.array(json.loads(CONTRACT.read_text())["sides"]["right"]["home_arm"], float)
    above = ik(tree, home, TARGET + [0, 0, 0.03])
    touch = ik(tree, above, TARGET)
    print(f"[fake_board_sender] home tip {np.round(tip(tree, home), 3)} → target {np.round(TARGET, 3)} · udp 127.0.0.1:{a.port}", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
    n = 0
    while a.seconds <= 0 or time.time() - t0 < a.seconds:
        t = time.time() - t0
        q_show = path(home, above, touch, (t % PERIOD_S) / PERIOD_S)
        q_cmd = path(home, above, touch, ((t + LEAD_S) % PERIOD_S) / PERIOD_S)
        p = tip(tree, q_show)
        data = board_packet.encode(time.time(), ARM + HAND, list(q_show) + POINT, list(q_cmd) + POINT, T_BOARD.tolist(),
                                   TARGET, p, bool(np.linalg.norm(p - TARGET) < 0.004), "dry")
        sock.sendto(data, ("127.0.0.1", a.port))
        n += 1
        if n % 150 == 0:
            print(f"[fake_board_sender] {n} 패킷 · tip-target {np.linalg.norm(p - TARGET) * 1000:.1f} mm", flush=True)
        time.sleep(1.0 / HZ)


if __name__ == "__main__":
    main()

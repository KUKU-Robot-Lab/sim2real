#!/usr/bin/env python3
"""fake RH56F1 손 — 벤더 드라이버(robot_control rh56f1_driver) 대신 같은 토픽 · 메시지 · 단위로 선다. fake 도메인 전용.

09.29 사용자: rh56f1 제어 연결. 실기 손 없이 pd 백엔드 → 드라이버 토픽 → 상태 노드 → pd/정책 경로를 돌린다.

  구독  /hand_<side>/angle_set  (SetAngle1, 슬롯 순 0.1°, -1 = 그 축은 둔다)
  발행  /hand_<side>/angle_actual (GetAngleAct1, 250 Hz — EtherCAT state_hz) · /hand_<side>/touch_data (TouchData1, 전부 0)

손가락은 목표 레지스터로 **전 행정 1 s**(벤더 speedSet 2000 기본)의 속도로 간다. 접촉 · 힘 제한은 흉내내지 않는다.
실기 도메인(126)과 0 은 거부한다.

    python3 scripts/fakes/fake_rh56f1_hand.py --side right [--open-pose 1.57,0,0,0,0,0]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_SIM2REAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SIM2REAL / "deploy" / "policy_control"))
from policy_control import rh56f1_map  # noqa: E402

REAL_DOMAIN = 126
RATE_HZ = 250.0           # 10.03 손 = EtherCAT(state_hz 250) — hand_check 가 그 80 % 를 기대한다(50 Hz RS485 시절 값이면 fake 미션이 멈춘다)
#: 전 행정(레지스터 끝 ↔ 끝) 걸리는 시간 [s]
STROKE_S = 1.0


def step(cur: np.ndarray, target: np.ndarray, span: np.ndarray, dt: float) -> np.ndarray:
    """한 틱 — 축마다 전 행정 STROKE_S 속도로 목표에 다가간다. 순수."""
    lim = span / STROKE_S * dt
    return cur + np.clip(target - cur, -lim, lim)


def apply_command(target: np.ndarray, values) -> np.ndarray:
    """SetAngle1 → 새 목표. -1 인 축은 그대로 둔다. 순수."""
    v = np.asarray(list(values), dtype=float)
    return np.where(v < 0, target, v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--map", default=str(rh56f1_map.DEFAULT_PATH))
    ap.add_argument("--open-pose", default="1.57,0,0,0,0,0", help="시작 손 자세(joint_order rad) — hdgp 프로필 open")
    args = ap.parse_args(argv)
    dom = os.environ.get("ROS_DOMAIN_ID", "").strip()
    if dom in ("", "0", str(REAL_DOMAIN)):
        print(f"fake 손은 fake 도메인에서만 — ROS_DOMAIN_ID={dom!r}", file=sys.stderr)
        return 2

    import rclpy
    from rclpy.node import Node
    from rh56f1_interfaces.msg import GetAngleAct1, SetAngle1, TouchData1

    hmap = rh56f1_map.load(args.map)
    pose = [float(v) for v in args.open_pose.split(",")]
    cur = np.asarray(hmap.to_register(pose, allow_unverified=True), dtype=float)
    target = cur.copy()
    span = np.zeros(6)
    for a in hmap.axes:
        span[a.slot] = abs(a.reg[1] - a.reg[0])
    names = [""] * 6
    for a in hmap.axes:
        names[a.slot] = f"{args.side[0]}_hj_{a.name}"

    rclpy.init()
    node = Node(f"fake_rh56f1_{args.side}")
    ns = f"/hand_{args.side}"
    act = node.create_publisher(GetAngleAct1, f"{ns}/angle_actual", 10)
    touch = node.create_publisher(TouchData1, f"{ns}/touch_data", 10)

    def on_set(msg) -> None:
        nonlocal target
        target = apply_command(target, msg.joint_values)

    def tick() -> None:
        nonlocal cur
        cur = step(cur, target, span, 1.0 / RATE_HZ)
        m = GetAngleAct1()
        m.header.stamp = node.get_clock().now().to_msg()
        m.joint_values = [int(round(v)) for v in cur]
        m.joint_names = names
        act.publish(m)
        t = TouchData1()
        t.header.stamp = m.header.stamp
        touch.publish(t)

    node.create_subscription(SetAngle1, f"{ns}/angle_set", on_set, 10)
    node.create_timer(1.0 / RATE_HZ, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

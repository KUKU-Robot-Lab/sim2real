#!/usr/bin/env python3
"""팔 · 손 관절 상태를 잠깐 구독해 canonical 이름(r_aj_* · r_hj_* …)의 JSON 한 줄로 낸다. 구독만 한다.

    ROS_DOMAIN_ID=97 python3 sample_joints.py [--seconds 1.0]

run_fake_mission.py 가 단계마다 부른다 — 콘솔(API) 프로세스가 rclpy 를 import 하지 않게 따로 뜬다.
이름 변환은 Isaac 뷰어와 같은 표(robot_control 프로필 openarm_tesollo.yaml, joint_map.py)를 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

VIEWER = Path(__file__).resolve().parents[3] / "robot" / "isaacsim_bridge" / "viewer"
sys.path.insert(0, str(VIEWER))
from joint_map import load_profile_table, map_joint_state  # noqa: E402

TOPICS = ("/joint_states", "/dg5f_right/joint_states", "/dg5f_left/joint_states")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        print("ROS_DOMAIN_ID 가 비었거나 0 — 거부", file=sys.stderr)
        return 2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    table = load_profile_table()
    values: dict[str, float] = {}
    seen: set[str] = set()

    def cb(topic):
        def _cb(msg):
            seen.add(topic)
            values.update(map_joint_state(list(msg.name), list(msg.position), table).values)
        return _cb

    rclpy.init()
    node = Node("sample_joints")
    for t in TOPICS:
        node.create_subscription(JointState, t, cb(t), 10)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds or len(seen) < len(TOPICS) and time.monotonic() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    print(json.dumps({"topics": sorted(seen), "q": values}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

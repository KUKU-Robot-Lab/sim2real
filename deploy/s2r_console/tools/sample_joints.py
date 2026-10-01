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

#: 팔 · DG5F 손 · RH56F1 손(rh56f1_state_node — canonical r_hj_* 이름). 10.01: RH56F1 손 토픽이 없어 fake 홈 손 판정이
#: '관절 상태 없음'으로 실패하고, "손은 아직 움직이지 않았다"는 inf 로 공허하게 통과했다.
ARM_TOPIC = "/joint_states"
WAIT_MAX_S = 5.0
TOPICS = (ARM_TOPIC, "/dg5f_right/joint_states", "/dg5f_left/joint_states",
          "/hand_right/joint_states", "/hand_left/joint_states")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--need", default="", help="canonical 관절 CSV — 이 관절들이 다 들어올 때까지(최대 5 s) 기다린다. "
                                             "10.01: 발견(discovery)이 늦으면 손 · 팔 토픽 하나를 통째로 놓쳤다")
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

    def wanted() -> set[str]:
        """기다릴 토픽 — 팔 /joint_states 는 늘(어느 로봇이든 있다), 손 토픽은 그래프에 발행자가 보이는 것만.
        10.01 실기: 발행자 있는 것만 기다리게 하자 0.6 s 안에 발견(discovery)이 /joint_states 를 아직 못 봐
        팔 관절 없이 끝나는 일이 4 번에 1~2 번 났다 — 팔 토픽은 발견 여부와 무관하게 받을 때까지 기다린다."""
        return {ARM_TOPIC} | {t for t in TOPICS if node.count_publishers(t) > 0}

    need = {n.strip() for n in args.need.split(",") if n.strip()}

    def done() -> bool:
        return wanted() <= seen and need <= values.keys()

    while time.monotonic() - t0 < args.seconds or not done() and time.monotonic() - t0 < WAIT_MAX_S:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    print(json.dumps({"topics": sorted(seen), "q": values}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

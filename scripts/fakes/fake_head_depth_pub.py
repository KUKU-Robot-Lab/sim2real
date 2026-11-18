#!/usr/bin/env python3
"""fake 머리 깊이 — /head_depth/image(32FC1, 160x90, m, 균일 깊이)를 낸다. rh_cap 깊이 학생 노드 fake 리허설용(11.18).

    ROS_DOMAIN_ID=97 python3 scripts/fakes/fake_head_depth_pub.py [--depth 0.6] [--hz 60]

head_depth_node 와 같은 토픽 · 형식 · QoS(qos_profile_sensor_data). camera_info 는 내지 않는다. /head_depth/config 는 --free-head 와
같은 대조 키(CONFIG)를 같은 QoS(transient_local latched)로 한 번 낸다 — cap 학생 노드가 reset 때 대조한다.
stamp = 벽시계(학생 노드가 time.time_ns() − stamp 로 나이를 잰다). 실기 도메인(config/ros_domain.env)과 0 은 거부한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "policy_control"))
from policy_control.ros_domain import REAL_DOMAIN  # noqa: E402  config/ros_domain.env 한 곳(10.30)

TOPIC, H, W = "/head_depth/image", 90, 160      # cap_student.IMG_HW
CONFIG_TOPIC = "/head_depth/config"
# head_depth_node.free_head_config(8, …) 의 대조 키 값 복사 — 그 모듈은 torch · cv2 를 끌어와 import 하지 않는다.
#   일치는 tests/policy_control/test_pc_cap_student_node.py 가 묶는다.
CONFIG = {"block": 8, "out_wh": [W, H], "floor_cut": False, "head_gate": False, "clip_m": [0.01, 10.0],
          "floor_cut_z_m": None, "extrinsics_md5": None, "rgb": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--depth", type=float, default=0.6, help="균일 깊이 [m](학습 범위 0.2~1.5, 밖은 0)")
    ap.add_argument("--hz", type=float, default=60.0)
    args = ap.parse_args()
    dom = os.environ.get("ROS_DOMAIN_ID", "").strip()
    if not dom.isdecimal() or int(dom) in (0, REAL_DOMAIN):   # "050" 도 실기 50
        raise SystemExit(f"fake 도메인에서만(지금 ROS_DOMAIN_ID={dom!r})")
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("fake_head_depth_pub")
    latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub_cfg = node.create_publisher(String, CONFIG_TOPIC, latched)
    pub_cfg.publish(String(data=json.dumps({**CONFIG, "fake": True})))
    pub = node.create_publisher(Image, TOPIC, qos_profile_sensor_data)
    msg = Image(height=H, width=W, encoding="32FC1", is_bigendian=0, step=W * 4)
    msg.header.frame_id = "camera_color_optical_frame"
    msg.data = np.full((H, W), args.depth, "<f4").tobytes()

    def tick() -> None:
        ns = time.time_ns()
        msg.header.stamp.sec, msg.header.stamp.nanosec = ns // 10**9, ns % 10**9
        pub.publish(msg)

    node.create_timer(1.0 / args.hz, tick)
    node.get_logger().info(f"{TOPIC} 32FC1 {W}x{H} = {args.depth} m @ {args.hz:g} Hz")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""fake 목 드라이버(연속 목표 포함, 11.18) — 실 head_joint_publisher 와 같은 HeadDriver 에 fake 모터를 붙인다.

fake_head_driver.py 는 HeadDriver 의 publish_angles(pan, tilt, 시각) 3 인자 규약(11.05) 전 것이라 첫 tick 에 죽고,
/head/target_stream 도 받지 않는다. 이 파일은 /head/joint_states 를 내고 /head/target_stream 을 받아 fake 모터를 움직인다.

    ROS_DOMAIN_ID=97 python3 scripts/fakes/fake_head_stream_driver.py [--pan 0 --tilt 40]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for sub in ("scripts", "scripts/nodes", "scripts/fakes", "deploy/policy_control"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / sub))
import head_motion  # noqa: E402
from fake_head_driver import FakeMotors  # noqa: E402
from head_driver_core import HeadDriver, stream_limits_for  # noqa: E402
from head_home import (ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE, DEFAULT_CONFIG, TORQUE_ON,  # noqa: E402
                       load_head_home, max_tilt_for)
from head_joint_publisher import DEFAULT_TOPIC, joint_state_fields  # noqa: E402
from policy_control.ros_domain import REAL_DOMAIN  # noqa: E402


class StreamMotors(FakeMotors):
    """연속 모드가 쓰는 read1(토크 · 모드) · write4(목표 틱)를 더한 fake 모터. 틱 ↔ 각은 FakeMotors 와 같은 (deg+180)·4095/360."""

    def __init__(self, pan: float, tilt: float, mode: int, bias: dict):
        super().__init__(pan, tilt)
        self.mode, self.bias = mode, bias

    def read1(self, _dxl_id: int, addr: int, _what: str) -> int:
        return TORQUE_ON if addr == ADDR_TORQUE_ENABLE else self.mode if addr == ADDR_OPERATING_MODE else 0

    def write4(self, dxl_id: int, _addr: int, tick: int, _what: str) -> None:
        self.goal[dxl_id] = (tick - int(self.bias.get(dxl_id, 0))) * 360.0 / 4095 - 180.0   # goal_tick 이 더한 bias 를 뺀다


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pan", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, default=40.0)
    ap.add_argument("--max-tilt", type=float, default=82.0)
    ap.add_argument("--rate-hz", type=float, default=20.0)
    args = ap.parse_args()
    dom = os.environ.get("ROS_DOMAIN_ID", "").strip()
    if not dom.isdecimal() or int(dom) in (0, REAL_DOMAIN):   # "050" 도 실기 50
        raise SystemExit(f"fake 도메인에서만(지금 ROS_DOMAIN_ID={dom!r})")
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    rclpy.init()
    node = Node("fake_head_stream_driver")
    pub = node.create_publisher(JointState, DEFAULT_TOPIC, 10)
    status_pub = node.create_publisher(String, head_motion.STATUS_TOPIC, 10)
    config = load_head_home(DEFAULT_CONFIG)
    motors = StreamMotors(args.pan, args.tilt, int(config.operating_mode), dict(config.goal_bias_tick))

    def publish_angles(pan: float, tilt: float, read_s: float) -> None:
        m = JointState()
        m.header.stamp = Time(nanoseconds=round(read_s * 1e9)).to_msg()
        m.name, m.position = joint_state_fields(pan, tilt)
        pub.publish(m)

    def wall(mono: float) -> float:
        return node.get_clock().now().nanoseconds * 1e-9 - (time.monotonic() - mono)

    max_tilt = max_tilt_for(config, args.max_tilt, args.max_tilt)
    driver = HeadDriver(motors, config, pan_id=1, tilt_id=2, publish_angles=publish_angles,
                        publish_status=lambda t: status_pub.publish(String(data=t)), apply=motors.apply, max_tilt_deg=max_tilt,
                        log=node.get_logger().info, period_s=1.0 / args.rate_hz, wall=wall,
                        stream_limits=stream_limits_for(config, 1, 2, max_tilt))
    node.create_subscription(String, head_motion.REQUEST_TOPIC, lambda m: driver.on_request(m.data), 10)
    node.create_subscription(String, head_motion.STREAM_TOPIC, lambda m: driver.on_stream(m.data), 10)
    node.create_timer(1.0 / args.rate_hz, driver.tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # SIGTERM 도 조용히 끝낸다
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

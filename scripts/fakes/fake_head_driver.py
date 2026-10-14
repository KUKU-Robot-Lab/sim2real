#!/usr/bin/env python3
"""가짜 목 드라이버 — `head_joint_publisher.py` 와 같은 토픽·이동 규약을 하드웨어 없이 낸다(fake 도메인 리허설용).

실제 `HeadDriver` 상태기계를 가짜 모터(일정 속도로 목표를 따라감)와 엮는다. 도메인 97/99 에서만 쓸 것.
    ROS_DOMAIN_ID=99 python3 scripts/fakes/fake_head_driver.py [--pan 0 --tilt -20]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

for sub in ("", "nodes"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / sub))

import head_motion  # noqa: E402
from head_driver_core import HeadDriver  # noqa: E402
from head_home import DEFAULT_CONFIG, load_head_home  # noqa: E402
from head_joint_publisher import DEFAULT_TOPIC, joint_state_fields  # noqa: E402


class FakeMotors:
    """목표 쪽으로 deg_per_s 속도로 움직이는 두 모터. `apply` 가 목표를 받는다."""

    def __init__(self, pan: float, tilt: float, deg_per_s: float = 60.0):
        self.pos = {1: pan, 2: tilt}
        self.goal = dict(self.pos)
        self.deg_per_s, self._t = deg_per_s, time.monotonic()

    def read_present_tick(self, dxl_id: int) -> int:
        now = time.monotonic()
        step = self.deg_per_s * (now - self._t)
        self._t = now
        for i in self.pos:
            self.pos[i] += max(-step, min(step, self.goal[i] - self.pos[i]))
        return round((self.pos[dxl_id] + 180.0) * 4095 / 360.0)

    def apply(self, _controller, config, dxl_id: int) -> None:
        self.goal[dxl_id] = config.targets_deg[dxl_id]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pan", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, default=-20.0)
    ap.add_argument("--rate-hz", type=float, default=20.0)
    args = ap.parse_args()
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    rclpy.init()
    node = Node("fake_head_driver")
    pub, status_pub = node.create_publisher(JointState, DEFAULT_TOPIC, 10), node.create_publisher(String, head_motion.STATUS_TOPIC, 10)
    motors = FakeMotors(args.pan, args.tilt)

    def publish_angles(pan: float, tilt: float) -> None:
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name, msg.position = joint_state_fields(pan, tilt)
        pub.publish(msg)

    driver = HeadDriver(motors, load_head_home(DEFAULT_CONFIG), pan_id=1, tilt_id=2, publish_angles=publish_angles,
                        publish_status=lambda text: status_pub.publish(String(data=text)), apply=motors.apply,
                        log=node.get_logger().info)
    node.create_subscription(String, head_motion.REQUEST_TOPIC, lambda m: driver.on_request(m.data), 10)
    node.create_timer(1.0 / args.rate_hz, driver.tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable

import rclpy
from builtin_interfaces.msg import Time
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


@dataclass
class JointSnapshot:
    stamp_ns: int
    position: float
    velocity: float
    effort: float


class JointErrorRecorder(Node):
    def __init__(self) -> None:
        super().__init__("joint_error_recorder")

        self.declare_parameter("real_joint_states_topic", "/isaacsim/joint_states")
        self.declare_parameter("sim_joint_states_topic", "/isaacsim/sim_joint_states")
        self.declare_parameter("output_path", "/tmp/isaacsim_joint_error.csv")
        self.declare_parameter("sample_period_sec", 0.05)
        self.declare_parameter("max_state_age_sec", 0.25)
        self.declare_parameter("tracked_joints", [])
        self.declare_parameter("summary_topic", "/isaacsim/joint_error_summary")

        self._real_topic = str(self.get_parameter("real_joint_states_topic").value)
        self._sim_topic = str(self.get_parameter("sim_joint_states_topic").value)
        self._output_path = str(self.get_parameter("output_path").value)
        self._sample_period_sec = float(self.get_parameter("sample_period_sec").value)
        self._max_state_age = Duration(
            seconds=float(self.get_parameter("max_state_age_sec").value)
        )
        self._tracked_joints = {
            str(joint_name)
            for joint_name in self.get_parameter("tracked_joints").value
            if str(joint_name)
        }
        self._summary_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("summary_topic").value), 10
        )

        self._real_states: dict[str, JointSnapshot] = {}
        self._sim_states: dict[str, JointSnapshot] = {}

        output_dir = os.path.dirname(self._output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._csv_file = open(self._output_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "stamp_sec",
                "joint",
                "real_position",
                "sim_position",
                "position_error",
                "real_velocity",
                "sim_velocity",
                "velocity_error",
                "real_effort",
                "sim_effort",
                "effort_error",
            ]
        )
        self._csv_file.flush()

        self.create_subscription(
            JointState, self._real_topic, self._real_joint_state_cb, 20
        )
        self.create_subscription(
            JointState, self._sim_topic, self._sim_joint_state_cb, 20
        )
        self.create_timer(self._sample_period_sec, self._sample_and_record)

        joint_filter = (
            f"{len(self._tracked_joints)} explicit joints"
            if self._tracked_joints
            else "all overlapping joints"
        )
        self.get_logger().info(
            "Recording joint error: "
            f"real={self._real_topic} sim={self._sim_topic} output={self._output_path} "
            f"filter={joint_filter}"
        )

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, "_csv_file") and self._csv_file and not self._csv_file.closed:
                self._csv_file.flush()
                self._csv_file.close()
        finally:
            return super().destroy_node()

    def _real_joint_state_cb(self, msg: JointState) -> None:
        self._update_cache(self._real_states, msg)

    def _sim_joint_state_cb(self, msg: JointState) -> None:
        self._update_cache(self._sim_states, msg)

    def _update_cache(self, cache: dict[str, JointSnapshot], msg: JointState) -> None:
        if not msg.name:
            return

        stamp_ns = self._stamp_to_nanoseconds(msg.header.stamp)
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds

        for index, joint_name in enumerate(msg.name):
            cache[joint_name] = JointSnapshot(
                stamp_ns=stamp_ns,
                position=msg.position[index] if index < len(msg.position) else 0.0,
                velocity=msg.velocity[index] if index < len(msg.velocity) else 0.0,
                effort=msg.effort[index] if index < len(msg.effort) else 0.0,
            )

    def _sample_and_record(self) -> None:
        joint_names = self._matching_joint_names()
        if not joint_names:
            return

        now_ns = self.get_clock().now().nanoseconds
        max_age_ns = self._max_state_age.nanoseconds

        position_sq_sum = 0.0
        velocity_sq_sum = 0.0
        max_abs_position_error = 0.0
        row_count = 0

        for joint_name in joint_names:
            real_state = self._real_states.get(joint_name)
            sim_state = self._sim_states.get(joint_name)
            if real_state is None or sim_state is None:
                continue
            if (now_ns - real_state.stamp_ns) > max_age_ns:
                continue
            if (now_ns - sim_state.stamp_ns) > max_age_ns:
                continue

            position_error = sim_state.position - real_state.position
            velocity_error = sim_state.velocity - real_state.velocity
            effort_error = sim_state.effort - real_state.effort

            self._csv_writer.writerow(
                [
                    f"{now_ns / 1e9:.6f}",
                    joint_name,
                    f"{real_state.position:.9f}",
                    f"{sim_state.position:.9f}",
                    f"{position_error:.9f}",
                    f"{real_state.velocity:.9f}",
                    f"{sim_state.velocity:.9f}",
                    f"{velocity_error:.9f}",
                    f"{real_state.effort:.9f}",
                    f"{sim_state.effort:.9f}",
                    f"{effort_error:.9f}",
                ]
            )

            position_sq_sum += position_error * position_error
            velocity_sq_sum += velocity_error * velocity_error
            max_abs_position_error = max(max_abs_position_error, abs(position_error))
            row_count += 1

        if row_count == 0:
            return

        self._csv_file.flush()

        summary = Float64MultiArray()
        summary.data = [
            math.sqrt(position_sq_sum / row_count),
            max_abs_position_error,
            math.sqrt(velocity_sq_sum / row_count),
            float(row_count),
        ]
        self._summary_pub.publish(summary)

    def _matching_joint_names(self) -> Iterable[str]:
        if self._tracked_joints:
            return sorted(
                joint_name
                for joint_name in self._tracked_joints
                if joint_name in self._real_states and joint_name in self._sim_states
            )
        return sorted(set(self._real_states).intersection(self._sim_states))

    @staticmethod
    def _stamp_to_nanoseconds(stamp: Time) -> int:
        return (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointErrorRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

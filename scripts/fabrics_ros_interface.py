#!/usr/bin/env python3

"""ROS 2 command interface that FABRICS or any Python control loop can import."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray


class Sim2RealCommandPublisher(Node):
    def __init__(self) -> None:
        super().__init__("sim2real_command_publisher")
        self.left_arm_pub = self.create_publisher(Float64MultiArray, "/isaacsim/left_arm_cmd", 10)
        self.left_gripper_pub = self.create_publisher(Float64, "/isaacsim/left_gripper_cmd", 10)
        self.right_arm_pub = self.create_publisher(Float64MultiArray, "/isaacsim/right_arm_cmd", 10)
        self.right_hand_pub = self.create_publisher(Float64MultiArray, "/isaacsim/right_hand_cmd", 10)

    def send_left_arm(self, values: list[float]) -> None:
        if len(values) != 7:
            raise ValueError(f"left arm expects 7 values, got {len(values)}")
        msg = Float64MultiArray()
        msg.data = list(values)
        self.left_arm_pub.publish(msg)

    def send_left_gripper(self, value: float) -> None:
        msg = Float64()
        msg.data = float(value)
        self.left_gripper_pub.publish(msg)

    def send_right_arm(self, values: list[float]) -> None:
        if len(values) != 7:
            raise ValueError(f"right arm expects 7 values, got {len(values)}")
        msg = Float64MultiArray()
        msg.data = list(values)
        self.right_arm_pub.publish(msg)

    def send_right_hand(self, values: list[float]) -> None:
        if len(values) != 20:
            raise ValueError(f"right hand expects 20 values, got {len(values)}")
        msg = Float64MultiArray()
        msg.data = list(values)
        self.right_hand_pub.publish(msg)

    def send_left_full(self, arm: list[float], gripper: float) -> None:
        self.send_left_arm(arm)
        self.send_left_gripper(gripper)

    def send_right_full(self, arm: list[float], hand: list[float]) -> None:
        self.send_right_arm(arm)
        self.send_right_hand(hand)

    def spin_once(self) -> None:
        rclpy.spin_once(self, timeout_sec=0.0)

    def close(self) -> None:
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def create_publisher() -> Sim2RealCommandPublisher:
    if not rclpy.ok():
        rclpy.init()
    return Sim2RealCommandPublisher()

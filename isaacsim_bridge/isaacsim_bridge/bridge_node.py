#!/usr/bin/env python3

from collections import OrderedDict

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from control_msgs.action import GripperCommand
    from rclpy.action import ActionClient
except ImportError:
    GripperCommand = None
    ActionClient = None


LEFT_ARM_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

RIGHT_ARM_JOINTS = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

RIGHT_HAND_JOINTS = [
    "rj_dg_1_1",
    "rj_dg_1_2",
    "rj_dg_1_3",
    "rj_dg_1_4",
    "rj_dg_2_1",
    "rj_dg_2_2",
    "rj_dg_2_3",
    "rj_dg_2_4",
    "rj_dg_3_1",
    "rj_dg_3_2",
    "rj_dg_3_3",
    "rj_dg_3_4",
    "rj_dg_4_1",
    "rj_dg_4_2",
    "rj_dg_4_3",
    "rj_dg_4_4",
    "rj_dg_5_1",
    "rj_dg_5_2",
    "rj_dg_5_3",
    "rj_dg_5_4",
]

DEFAULT_LEFT_ARM_MIN = [-3.14] * 7
DEFAULT_LEFT_ARM_MAX = [3.14] * 7
DEFAULT_RIGHT_ARM_MIN = [-3.14] * 7
DEFAULT_RIGHT_ARM_MAX = [3.14] * 7
DEFAULT_RIGHT_HAND_MIN = [-1.57] * 20
DEFAULT_RIGHT_HAND_MAX = [1.57] * 20
DEFAULT_LEFT_GRIPPER_MIN = 0.0
DEFAULT_LEFT_GRIPPER_MAX = 0.04


class IsaacSimBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("isaacsim_bridge")

        self.declare_parameter("trajectory_time_sec", 0.2)
        self.declare_parameter(
            "left_arm_topic", "/left_joint_trajectory_controller/joint_trajectory"
        )
        self.declare_parameter(
            "right_arm_topic", "/right_joint_trajectory_controller/joint_trajectory"
        )
        self.declare_parameter(
            "right_hand_topic", "/dg5f_right/dg5f_right_controller/joint_trajectory"
        )
        self.declare_parameter(
            "left_gripper_action", "/left_gripper_controller/gripper_cmd"
        )
        self.declare_parameter(
            "merged_joint_states_topic", "/isaacsim/joint_states"
        )
        self.declare_parameter("left_arm_min", DEFAULT_LEFT_ARM_MIN)
        self.declare_parameter("left_arm_max", DEFAULT_LEFT_ARM_MAX)
        self.declare_parameter("right_arm_min", DEFAULT_RIGHT_ARM_MIN)
        self.declare_parameter("right_arm_max", DEFAULT_RIGHT_ARM_MAX)
        self.declare_parameter("right_hand_min", DEFAULT_RIGHT_HAND_MIN)
        self.declare_parameter("right_hand_max", DEFAULT_RIGHT_HAND_MAX)
        self.declare_parameter("left_gripper_min", DEFAULT_LEFT_GRIPPER_MIN)
        self.declare_parameter("left_gripper_max", DEFAULT_LEFT_GRIPPER_MAX)

        self._trajectory_time_sec = float(
            self.get_parameter("trajectory_time_sec").value
        )
        self._left_arm_min = list(self.get_parameter("left_arm_min").value)
        self._left_arm_max = list(self.get_parameter("left_arm_max").value)
        self._right_arm_min = list(self.get_parameter("right_arm_min").value)
        self._right_arm_max = list(self.get_parameter("right_arm_max").value)
        self._right_hand_min = list(self.get_parameter("right_hand_min").value)
        self._right_hand_max = list(self.get_parameter("right_hand_max").value)
        self._left_gripper_min = float(self.get_parameter("left_gripper_min").value)
        self._left_gripper_max = float(self.get_parameter("left_gripper_max").value)
        self._emergency_stop_active = False

        self._left_arm_pub = self.create_publisher(
            JointTrajectory, self.get_parameter("left_arm_topic").value, 10
        )
        self._right_arm_pub = self.create_publisher(
            JointTrajectory, self.get_parameter("right_arm_topic").value, 10
        )
        self._right_hand_pub = self.create_publisher(
            JointTrajectory, self.get_parameter("right_hand_topic").value, 10
        )
        self._merged_joint_states_pub = self.create_publisher(
            JointState, self.get_parameter("merged_joint_states_topic").value, 10
        )

        self._left_gripper_client = None
        if GripperCommand is not None and ActionClient is not None:
            self._left_gripper_client = ActionClient(
                self,
                GripperCommand,
                self.get_parameter("left_gripper_action").value,
            )
        else:
            self.get_logger().warning(
                "control_msgs is not installed; left gripper action forwarding is disabled"
            )

        self._joint_state_cache: OrderedDict[str, dict] = OrderedDict()

        self.create_subscription(
            Float64MultiArray,
            "/isaacsim/left_arm_cmd",
            self._left_arm_cmd_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/isaacsim/right_arm_cmd",
            self._right_arm_cmd_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/isaacsim/right_hand_cmd",
            self._right_hand_cmd_cb,
            10,
        )
        self.create_subscription(
            Float64,
            "/isaacsim/left_gripper_cmd",
            self._left_gripper_cmd_cb,
            10,
        )
        self.create_subscription(
            Bool,
            "/isaacsim/emergency_stop",
            self._emergency_stop_cb,
            10,
        )
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 20)
        self.create_subscription(
            JointState, "/dg5f_right/joint_states", self._joint_state_cb, 20
        )

        self.get_logger().info(
            "Isaac Sim bridge ready. Input topics: "
            "/isaacsim/left_arm_cmd, /isaacsim/right_arm_cmd, "
            "/isaacsim/left_gripper_cmd, /isaacsim/right_hand_cmd, "
            "/isaacsim/emergency_stop"
        )

    def _left_arm_cmd_cb(self, msg: Float64MultiArray) -> None:
        self._publish_trajectory(
            self._left_arm_pub,
            LEFT_ARM_JOINTS,
            self._clamp_array(
                "left_arm_cmd",
                list(msg.data),
                self._left_arm_min,
                self._left_arm_max,
            ),
            "left_arm_cmd",
        )

    def _right_arm_cmd_cb(self, msg: Float64MultiArray) -> None:
        self._publish_trajectory(
            self._right_arm_pub,
            RIGHT_ARM_JOINTS,
            self._clamp_array(
                "right_arm_cmd",
                list(msg.data),
                self._right_arm_min,
                self._right_arm_max,
            ),
            "right_arm_cmd",
        )

    def _right_hand_cmd_cb(self, msg: Float64MultiArray) -> None:
        self._publish_trajectory(
            self._right_hand_pub,
            RIGHT_HAND_JOINTS,
            self._clamp_array(
                "right_hand_cmd",
                list(msg.data),
                self._right_hand_min,
                self._right_hand_max,
            ),
            "right_hand_cmd",
        )

    def _left_gripper_cmd_cb(self, msg: Float64) -> None:
        if self._emergency_stop_active:
            return

        if self._left_gripper_client is None or GripperCommand is None:
            self.get_logger().warning(
                "left gripper command received but control_msgs is unavailable; command dropped"
            )
            return

        if not self._left_gripper_client.server_is_ready():
            self.get_logger().warning(
                "left_gripper_controller action server is not ready; gripper command dropped"
            )
            return

        goal = GripperCommand.Goal()
        goal.command.position = min(
            max(float(msg.data), self._left_gripper_min),
            self._left_gripper_max,
        )
        goal.command.max_effort = 0.0
        self._left_gripper_client.send_goal_async(goal)

    def _emergency_stop_cb(self, msg: Bool) -> None:
        next_state = bool(msg.data)
        if next_state == self._emergency_stop_active:
            return

        self._emergency_stop_active = next_state
        if self._emergency_stop_active:
            self.get_logger().warning(
                "Emergency stop active; bridge is dropping all outgoing commands"
            )
        else:
            self.get_logger().info("Emergency stop cleared; command forwarding resumed")

    def _publish_trajectory(
        self,
        publisher,
        joint_names: list[str],
        positions: list[float],
        source_name: str,
    ) -> None:
        if self._emergency_stop_active:
            return

        if len(positions) != len(joint_names):
            self.get_logger().warning(
                f"{source_name} expected {len(joint_names)} values, got {len(positions)}"
            )
            return

        msg = JointTrajectory()
        msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(self._trajectory_time_sec),
            nanosec=int((self._trajectory_time_sec % 1.0) * 1e9),
        )
        msg.points.append(point)

        publisher.publish(msg)

    def _clamp_array(
        self,
        source_name: str,
        values: list[float],
        lower: list[float],
        upper: list[float],
    ) -> list[float]:
        if len(values) != len(lower) or len(values) != len(upper):
            return values

        clamped = []
        was_clamped = False
        for value, min_value, max_value in zip(values, lower, upper):
            next_value = min(max(value, min_value), max_value)
            if next_value != value:
                was_clamped = True
            clamped.append(next_value)

        if was_clamped:
            self.get_logger().warning(
                f"{source_name} exceeded configured limits; values were clamped"
            )

        return clamped

    def _joint_state_cb(self, msg: JointState) -> None:
        if not msg.name:
            return

        for index, joint_name in enumerate(msg.name):
            cached = self._joint_state_cache.get(joint_name, {})
            cached["position"] = (
                msg.position[index] if index < len(msg.position) else 0.0
            )
            cached["velocity"] = (
                msg.velocity[index] if index < len(msg.velocity) else 0.0
            )
            cached["effort"] = msg.effort[index] if index < len(msg.effort) else 0.0
            self._joint_state_cache[joint_name] = cached

        merged = JointState()
        merged.header.stamp = self.get_clock().now().to_msg()
        merged.name = list(self._joint_state_cache.keys())
        merged.position = [
            self._joint_state_cache[name]["position"] for name in merged.name
        ]
        merged.velocity = [
            self._joint_state_cache[name]["velocity"] for name in merged.name
        ]
        merged.effort = [
            self._joint_state_cache[name]["effort"] for name in merged.name
        ]

        self._merged_joint_states_pub.publish(merged)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IsaacSimBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

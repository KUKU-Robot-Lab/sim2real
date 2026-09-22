"""
Subscribe to /isaacsim/* ROS 2 command topics inside Isaac Sim and mirror them onto the loaded robot.

Use in Isaac Sim Script Editor after the robot USD is loaded:
    Open this file in Isaac Sim Script Editor and run it.

Then press Play. After that, terminal-side publishers like manual_command_pub.py
will move the Isaac Sim shadow robot too.
"""

from __future__ import annotations

import asyncio

import omni.timeline

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64, Float64MultiArray
except ImportError as exc:
    raise RuntimeError(
        "rclpy/std_msgs are not available inside Isaac Sim. Enable the ROS 2 bridge extension "
        "and ensure the ROS 2 Python environment is available."
    ) from exc

from isaacsim.core.prims import SingleArticulation


ROBOT_PRIM_CANDIDATES = [
    "/openarm_dual_modular",
    "/openarm_dual_modular/root_joint",
]

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

LEFT_GRIPPER_JOINT = "openarm_left_finger_joint1"

_shadow_bridge = None
_update_task = None


class RosToSimShadowBridge(Node):
    def __init__(self) -> None:
        super().__init__("ros_to_sim_shadow")
        self.timeline = omni.timeline.get_timeline_interface()
        self.articulation = self._attach_articulation()
        self.left_arm_indices = [self.articulation.get_dof_index(name) for name in LEFT_ARM_JOINTS]
        self.right_arm_indices = [self.articulation.get_dof_index(name) for name in RIGHT_ARM_JOINTS]
        self.right_hand_indices = [self.articulation.get_dof_index(name) for name in RIGHT_HAND_JOINTS]
        self.left_gripper_index = self.articulation.get_dof_index(LEFT_GRIPPER_JOINT)

        self.create_subscription(Float64MultiArray, "/isaacsim/left_arm_cmd", self._left_arm_cb, 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_arm_cmd", self._right_arm_cb, 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_hand_cmd", self._right_hand_cb, 10)
        self.create_subscription(Float64, "/isaacsim/left_gripper_cmd", self._left_gripper_cb, 10)

        self.get_logger().info("ROS-to-sim shadow bridge ready.")

    def _attach_articulation(self) -> SingleArticulation:
        last_exc = None
        for candidate in ROBOT_PRIM_CANDIDATES:
            try:
                articulation = SingleArticulation(prim_path=candidate, name="ros_to_sim_shadow_articulation")
                articulation.initialize()
                print(f"ROS-to-sim shadow attached to: {candidate}")
                return articulation
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"Could not attach to articulation. Tried: {ROBOT_PRIM_CANDIDATES}. Last error: {last_exc}"
        )

    def _left_arm_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 7 or not self.timeline.is_playing():
            return
        self.articulation.set_joint_positions([list(msg.data)], joint_indices=self.left_arm_indices)

    def _right_arm_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 7 or not self.timeline.is_playing():
            return
        self.articulation.set_joint_positions([list(msg.data)], joint_indices=self.right_arm_indices)

    def _right_hand_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 20 or not self.timeline.is_playing():
            return
        self.articulation.set_joint_positions([list(msg.data)], joint_indices=self.right_hand_indices)

    def _left_gripper_cb(self, msg: Float64) -> None:
        if not self.timeline.is_playing():
            return
        self.articulation.set_joint_positions([[float(msg.data)]], joint_indices=[self.left_gripper_index])


async def _spin_bridge():
    global _shadow_bridge

    if not rclpy.ok():
        rclpy.init(args=None)

    _shadow_bridge = RosToSimShadowBridge()

    while _shadow_bridge is not None:
        rclpy.spin_once(_shadow_bridge, timeout_sec=0.0)
        await __import__("omni.kit.app").kit.app.get_app().next_update_async()


def stop_ros_to_sim_shadow():
    global _shadow_bridge, _update_task

    if _shadow_bridge is not None:
        _shadow_bridge.destroy_node()
        _shadow_bridge = None
    if rclpy.ok():
        rclpy.shutdown()
    _update_task = None
    print("ROS-to-sim shadow bridge stopped.")


if _update_task is None:
    _update_task = asyncio.ensure_future(_spin_bridge())
    print("ROS-to-sim shadow bridge starting. Press Play, then publish to /isaacsim/* topics.")
    print("Call stop_ros_to_sim_shadow() in Script Editor to stop it.")

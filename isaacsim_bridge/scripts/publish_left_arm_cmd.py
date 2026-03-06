"""
Create a ROS2Publisher graph for /isaacsim/left_arm_cmd and seed it with a 7-DoF command.

Use in Isaac Sim Script Editor:
    Open this file in Isaac Sim Script Editor and run it.

After it runs:
- press Play
- the graph publishes std_msgs/msg/Float64MultiArray on /isaacsim/left_arm_cmd
- the same 7-DoF command is also applied to the Isaac Sim robot articulation

You can then call:

    set_left_arm_cmd([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

from the Script Editor to change the published command.
"""

import asyncio


GRAPH_PATH = "/ActionGraph/LeftArmCmdPublisher"
PUBLISHER_NODE = GRAPH_PATH + "/LeftArmPublisher"
ROBOT_PRIM_PATH = "/openarm_dual_modular"
ROBOT_PRIM_CANDIDATES = [
    ROBOT_PRIM_PATH,
    ROBOT_PRIM_PATH + "/root_joint",
]
DEFAULT_CMD = [0.0] * 7
LEFT_ARM_JOINT_NAMES = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

_LEFT_ARM_ARTICULATION = None
_LEFT_ARM_DOF_INDICES = None


async def _build_graph():
    import omni.graph.core as og
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(GRAPH_PATH):
        stage.RemovePrim(GRAPH_PATH)

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Ros2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("LeftArmPublisher", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            keys.SET_VALUES: [
                ("LeftArmPublisher.inputs:messagePackage", "std_msgs"),
                ("LeftArmPublisher.inputs:messageSubfolder", "msg"),
                ("LeftArmPublisher.inputs:messageName", "Float64MultiArray"),
                ("LeftArmPublisher.inputs:topicName", "/isaacsim/left_arm_cmd"),
                ("LeftArmPublisher.inputs:queueSize", 10),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "LeftArmPublisher.inputs:execIn"),
                ("Ros2Context.outputs:context", "LeftArmPublisher.inputs:context"),
            ],
        },
    )

    # Let the generic publisher generate dynamic ports for Float64MultiArray.
    app = __import__("omni.kit.app").kit.app.get_app()
    await app.next_update_async()
    await app.next_update_async()

    _set_left_arm_ros_msg(DEFAULT_CMD)

    print(f"Created graph: {GRAPH_PATH}")
    print("Topic: /isaacsim/left_arm_cmd")
    print(f"Initial command: {DEFAULT_CMD}")
    print("Play first, then call set_left_arm_cmd([...]) to mirror onto the sim robot.")
    print("Press Play, then verify with:")
    print("  ros2 topic echo /isaacsim/left_arm_cmd")
    print("  ros2 topic echo /left_joint_trajectory_controller/joint_trajectory")


def set_left_arm_cmd(values):
    if len(values) != 7:
        raise ValueError(f"left arm command must have exactly 7 values, got {len(values)}")

    _set_left_arm_ros_msg(list(values))
    _apply_left_arm_to_sim(list(values))
    print(f"Updated /isaacsim/left_arm_cmd: {list(values)}")


def _set_left_arm_ros_msg(values):
    import omni.graph.core as og

    og.Controller.attribute(PUBLISHER_NODE + ".inputs:data").set(list(values))
    og.Controller.attribute(PUBLISHER_NODE + ".inputs:layout:data_offset").set(0)
    og.Controller.attribute(PUBLISHER_NODE + ".inputs:layout:dim").set([])


def _apply_left_arm_to_sim(values):
    global _LEFT_ARM_ARTICULATION, _LEFT_ARM_DOF_INDICES

    try:
        from isaacsim.core.prims import SingleArticulation
        import omni.timeline
    except ImportError:
        print("isaacsim.core.prims.SingleArticulation is unavailable; sim shadow update skipped")
        return

    if not omni.timeline.get_timeline_interface().is_playing():
        print("Timeline is not playing; sim shadow update skipped. Press Play first.")
        return

    if _LEFT_ARM_ARTICULATION is None:
        for candidate in ROBOT_PRIM_CANDIDATES:
            try:
                _LEFT_ARM_ARTICULATION = SingleArticulation(prim_path=candidate, name="openarm_left_shadow")
                _LEFT_ARM_ARTICULATION.initialize()
                _LEFT_ARM_DOF_INDICES = [
                    _LEFT_ARM_ARTICULATION.get_dof_index(joint_name) for joint_name in LEFT_ARM_JOINT_NAMES
                ]
                print(f"Left arm shadow articulation attached to: {candidate}")
                break
            except Exception as exc:
                _LEFT_ARM_ARTICULATION = None
                _LEFT_ARM_DOF_INDICES = None
                last_exc = exc
        if _LEFT_ARM_ARTICULATION is None:
            print(
                "Could not initialize left arm articulation. "
                f"Tried: {ROBOT_PRIM_CANDIDATES}. "
                f"Last error: {last_exc}"
            )
            return

    try:
        _LEFT_ARM_ARTICULATION.set_joint_positions([values], joint_indices=_LEFT_ARM_DOF_INDICES)
    except Exception as exc:
        print(f"Could not apply left arm command to sim articulation: {exc}")


asyncio.ensure_future(_build_graph())

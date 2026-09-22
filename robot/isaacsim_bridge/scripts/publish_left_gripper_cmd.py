"""
Create a ROS2Publisher graph for /isaacsim/left_gripper_cmd and mirror the command to the sim robot.

Use in Isaac Sim Script Editor:
    Open this file in Isaac Sim Script Editor and run it.

After it runs:
- press Play
- the graph publishes std_msgs/msg/Float64 on /isaacsim/left_gripper_cmd
- the same command is applied to the Isaac Sim left gripper joint

Then call:

    set_left_gripper_cmd(0.02)
"""

import asyncio


GRAPH_PATH = "/ActionGraph/LeftGripperCmdPublisher"
PUBLISHER_NODE = GRAPH_PATH + "/LeftGripperPublisher"
ROBOT_PRIM_PATH = "/openarm_dual_modular"
ROBOT_PRIM_CANDIDATES = [
    ROBOT_PRIM_PATH,
    ROBOT_PRIM_PATH + "/root_joint",
]
DEFAULT_CMD = 0.0
LEFT_GRIPPER_JOINT_NAME = "openarm_left_finger_joint1"

_LEFT_GRIPPER_ARTICULATION = None
_LEFT_GRIPPER_DOF_INDEX = None


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
                ("LeftGripperPublisher", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            keys.SET_VALUES: [
                ("LeftGripperPublisher.inputs:messagePackage", "std_msgs"),
                ("LeftGripperPublisher.inputs:messageSubfolder", "msg"),
                ("LeftGripperPublisher.inputs:messageName", "Float64"),
                ("LeftGripperPublisher.inputs:topicName", "/isaacsim/left_gripper_cmd"),
                ("LeftGripperPublisher.inputs:queueSize", 10),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "LeftGripperPublisher.inputs:execIn"),
                ("Ros2Context.outputs:context", "LeftGripperPublisher.inputs:context"),
            ],
        },
    )

    app = __import__("omni.kit.app").kit.app.get_app()
    await app.next_update_async()
    await app.next_update_async()

    _set_left_gripper_ros_msg(DEFAULT_CMD)

    print(f"Created graph: {GRAPH_PATH}")
    print("Topic: /isaacsim/left_gripper_cmd")
    print(f"Initial command: {DEFAULT_CMD}")
    print("Play first, then call set_left_gripper_cmd(value) to mirror onto the sim robot.")
    print("Press Play, then verify with:")
    print("  ros2 topic echo /isaacsim/left_gripper_cmd")


def set_left_gripper_cmd(value):
    _set_left_gripper_ros_msg(float(value))
    _apply_left_gripper_to_sim(float(value))
    print(f"Updated /isaacsim/left_gripper_cmd: {float(value)}")


def _set_left_gripper_ros_msg(value):
    import omni.graph.core as og

    og.Controller.attribute(PUBLISHER_NODE + ".inputs:data").set(float(value))


def _apply_left_gripper_to_sim(value):
    global _LEFT_GRIPPER_ARTICULATION, _LEFT_GRIPPER_DOF_INDEX

    try:
        from isaacsim.core.prims import SingleArticulation
        import omni.timeline
    except ImportError:
        print("isaacsim.core.prims.SingleArticulation is unavailable; sim shadow update skipped")
        return

    if not omni.timeline.get_timeline_interface().is_playing():
        print("Timeline is not playing; sim shadow update skipped. Press Play first.")
        return

    if _LEFT_GRIPPER_ARTICULATION is None:
        last_exc = None
        for candidate in ROBOT_PRIM_CANDIDATES:
            try:
                _LEFT_GRIPPER_ARTICULATION = SingleArticulation(prim_path=candidate, name="openarm_left_gripper_shadow")
                _LEFT_GRIPPER_ARTICULATION.initialize()
                _LEFT_GRIPPER_DOF_INDEX = _LEFT_GRIPPER_ARTICULATION.get_dof_index(LEFT_GRIPPER_JOINT_NAME)
                print(f"Left gripper shadow articulation attached to: {candidate}")
                break
            except Exception as exc:
                _LEFT_GRIPPER_ARTICULATION = None
                _LEFT_GRIPPER_DOF_INDEX = None
                last_exc = exc
        if _LEFT_GRIPPER_ARTICULATION is None:
            print(
                "Could not initialize left gripper articulation. "
                f"Tried: {ROBOT_PRIM_CANDIDATES}. "
                f"Last error: {last_exc}"
            )
            return

    try:
        _LEFT_GRIPPER_ARTICULATION.set_joint_positions([[float(value)]], joint_indices=[_LEFT_GRIPPER_DOF_INDEX])
    except Exception as exc:
        print(f"Could not apply left gripper command to sim articulation: {exc}")


asyncio.ensure_future(_build_graph())

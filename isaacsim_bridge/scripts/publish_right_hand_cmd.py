"""
Create a ROS2Publisher graph for /isaacsim/right_hand_cmd and mirror the command to the sim robot.

Use in Isaac Sim Script Editor:
    Open this file in Isaac Sim Script Editor and run it.

After it runs:
- press Play
- the graph publishes std_msgs/msg/Float64MultiArray on /isaacsim/right_hand_cmd
- the same 20-DoF command is applied to the Isaac Sim right hand joints

Then call:

    set_right_hand_cmd([0.0] * 20)
"""

import asyncio


GRAPH_PATH = "/ActionGraph/RightHandCmdPublisher"
PUBLISHER_NODE = GRAPH_PATH + "/RightHandPublisher"
ROBOT_PRIM_PATH = "/openarm_dual_modular"
ROBOT_PRIM_CANDIDATES = [
    ROBOT_PRIM_PATH,
    ROBOT_PRIM_PATH + "/root_joint",
]
DEFAULT_CMD = [0.0] * 20
RIGHT_HAND_JOINT_NAMES = [
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

_RIGHT_HAND_ARTICULATION = None
_RIGHT_HAND_DOF_INDICES = None


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
                ("RightHandPublisher", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            keys.SET_VALUES: [
                ("RightHandPublisher.inputs:messagePackage", "std_msgs"),
                ("RightHandPublisher.inputs:messageSubfolder", "msg"),
                ("RightHandPublisher.inputs:messageName", "Float64MultiArray"),
                ("RightHandPublisher.inputs:topicName", "/isaacsim/right_hand_cmd"),
                ("RightHandPublisher.inputs:queueSize", 10),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RightHandPublisher.inputs:execIn"),
                ("Ros2Context.outputs:context", "RightHandPublisher.inputs:context"),
            ],
        },
    )

    app = __import__("omni.kit.app").kit.app.get_app()
    await app.next_update_async()
    await app.next_update_async()

    _set_right_hand_ros_msg(DEFAULT_CMD)

    print(f"Created graph: {GRAPH_PATH}")
    print("Topic: /isaacsim/right_hand_cmd")
    print(f"Initial command: {DEFAULT_CMD}")
    print("Play first, then call set_right_hand_cmd([...]) to mirror onto the sim robot.")
    print("Press Play, then verify with:")
    print("  ros2 topic echo /isaacsim/right_hand_cmd")


def set_right_hand_cmd(values):
    if len(values) != 20:
        raise ValueError(f"right hand command must have exactly 20 values, got {len(values)}")

    _set_right_hand_ros_msg(list(values))
    _apply_right_hand_to_sim(list(values))
    print(f"Updated /isaacsim/right_hand_cmd: {list(values)}")


def _set_right_hand_ros_msg(values):
    import omni.graph.core as og

    og.Controller.attribute(PUBLISHER_NODE + ".inputs:data").set(list(values))
    og.Controller.attribute(PUBLISHER_NODE + ".inputs:layout:data_offset").set(0)
    og.Controller.attribute(PUBLISHER_NODE + ".inputs:layout:dim").set([])


def _apply_right_hand_to_sim(values):
    global _RIGHT_HAND_ARTICULATION, _RIGHT_HAND_DOF_INDICES

    try:
        from isaacsim.core.prims import SingleArticulation
        import omni.timeline
    except ImportError:
        print("isaacsim.core.prims.SingleArticulation is unavailable; sim shadow update skipped")
        return

    if not omni.timeline.get_timeline_interface().is_playing():
        print("Timeline is not playing; sim shadow update skipped. Press Play first.")
        return

    if _RIGHT_HAND_ARTICULATION is None:
        last_exc = None
        for candidate in ROBOT_PRIM_CANDIDATES:
            try:
                _RIGHT_HAND_ARTICULATION = SingleArticulation(prim_path=candidate, name="tesollo_right_shadow")
                _RIGHT_HAND_ARTICULATION.initialize()
                _RIGHT_HAND_DOF_INDICES = [
                    _RIGHT_HAND_ARTICULATION.get_dof_index(joint_name) for joint_name in RIGHT_HAND_JOINT_NAMES
                ]
                print(f"Right hand shadow articulation attached to: {candidate}")
                break
            except Exception as exc:
                _RIGHT_HAND_ARTICULATION = None
                _RIGHT_HAND_DOF_INDICES = None
                last_exc = exc
        if _RIGHT_HAND_ARTICULATION is None:
            print(
                "Could not initialize right hand articulation. "
                f"Tried: {ROBOT_PRIM_CANDIDATES}. "
                f"Last error: {last_exc}"
            )
            return

    try:
        _RIGHT_HAND_ARTICULATION.set_joint_positions([list(values)], joint_indices=_RIGHT_HAND_DOF_INDICES)
    except Exception as exc:
        print(f"Could not apply right hand command to sim articulation: {exc}")


asyncio.ensure_future(_build_graph())

"""
Create an Isaac Sim Action Graph that publishes the shadow robot articulation state
to a dedicated ROS 2 topic for sim-vs-real comparison.

Use in Isaac Sim Script Editor after the robot USD is loaded:

    Open this file in Isaac Sim Script Editor and run it.

After the script finishes:
- press Play
- the robot's current articulation joint state is published on /isaacsim/sim_joint_states
- run joint_error_recorder against /isaacsim/joint_states (real) and /isaacsim/sim_joint_states (sim)
"""

from __future__ import annotations

import asyncio


ROBOT_PATH = "/openarm_dual_modular/root_joint"
GRAPH_PATH = "/ActionGraph/RosSimJointStatePublisher"
TOPIC_NAME = "/isaacsim/sim_joint_states"


async def _build() -> None:
    import omni.graph.core as og
    import omni.usd
    import usdrt.Sdf

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(GRAPH_PATH):
        stage.RemovePrim(GRAPH_PATH)

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("Tick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ],
            keys.SET_VALUES: [
                ("PublishJointState.inputs:topicName", TOPIC_NAME),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_PATH)]),
            ],
            keys.CONNECT: [
                ("Tick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishJointState.inputs:timeStamp",
                ),
            ],
        },
    )

    print(f"Created ROS joint state publisher graph: {GRAPH_PATH}")
    print(f"Publishing articulation state from: {ROBOT_PATH}")
    print(f"Topic: {TOPIC_NAME}")
    print("Press Play, then verify with:")
    print(f"  ros2 topic echo {TOPIC_NAME}")


asyncio.ensure_future(_build())

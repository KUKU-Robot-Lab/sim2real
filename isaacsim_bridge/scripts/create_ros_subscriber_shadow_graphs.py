"""
Create Action Graph subscribers inside Isaac Sim that mirror /isaacsim/* ROS 2 topics to the loaded robot.

Use in Isaac Sim Script Editor after the robot USD is loaded:
    Open this file in Isaac Sim Script Editor and run it.

After the script finishes:
- press Play
- terminal-side publishers like manual_command_pub.py will drive the shadow robot
"""

from __future__ import annotations

import asyncio


ROBOT_PATH = "/openarm_dual_modular"

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


async def _build():
    import omni.graph.core as og
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    graph_path = "/ActionGraph/RosToSimShadow"
    if stage.GetPrimAtPath(graph_path):
        stage.RemovePrim(graph_path)

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("Tick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("LeftArmSub", "isaacsim.ros2.bridge.ROS2Subscriber"),
                ("LeftArmCtrl", "isaacsim.core.nodes.IsaacArticulationController"),
                ("LeftGripperSub", "isaacsim.ros2.bridge.ROS2Subscriber"),
                ("LeftGripperArray", "omni.graph.nodes.ConstructArray"),
                ("LeftGripperCtrl", "isaacsim.core.nodes.IsaacArticulationController"),
                ("RightArmSub", "isaacsim.ros2.bridge.ROS2Subscriber"),
                ("RightArmCtrl", "isaacsim.core.nodes.IsaacArticulationController"),
                ("RightHandSub", "isaacsim.ros2.bridge.ROS2Subscriber"),
                ("RightHandCtrl", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            keys.SET_VALUES: [
                ("LeftArmSub.inputs:messagePackage", "std_msgs"),
                ("LeftArmSub.inputs:messageSubfolder", "msg"),
                ("LeftArmSub.inputs:messageName", "Float64MultiArray"),
                ("LeftArmSub.inputs:topicName", "/isaacsim/left_arm_cmd"),
                ("LeftArmCtrl.inputs:robotPath", ROBOT_PATH),
                ("LeftArmCtrl.inputs:jointNames", LEFT_ARM_JOINTS),

                ("LeftGripperSub.inputs:messagePackage", "std_msgs"),
                ("LeftGripperSub.inputs:messageSubfolder", "msg"),
                ("LeftGripperSub.inputs:messageName", "Float64"),
                ("LeftGripperSub.inputs:topicName", "/isaacsim/left_gripper_cmd"),
                ("LeftGripperArray.inputs:arraySize", 1),
                ("LeftGripperArray.inputs:arrayType", "double[]"),
                ("LeftGripperCtrl.inputs:robotPath", ROBOT_PATH),
                ("LeftGripperCtrl.inputs:jointNames", ["openarm_left_finger_joint1"]),

                ("RightArmSub.inputs:messagePackage", "std_msgs"),
                ("RightArmSub.inputs:messageSubfolder", "msg"),
                ("RightArmSub.inputs:messageName", "Float64MultiArray"),
                ("RightArmSub.inputs:topicName", "/isaacsim/right_arm_cmd"),
                ("RightArmCtrl.inputs:robotPath", ROBOT_PATH),
                ("RightArmCtrl.inputs:jointNames", RIGHT_ARM_JOINTS),

                ("RightHandSub.inputs:messagePackage", "std_msgs"),
                ("RightHandSub.inputs:messageSubfolder", "msg"),
                ("RightHandSub.inputs:messageName", "Float64MultiArray"),
                ("RightHandSub.inputs:topicName", "/isaacsim/right_hand_cmd"),
                ("RightHandCtrl.inputs:robotPath", ROBOT_PATH),
                ("RightHandCtrl.inputs:jointNames", RIGHT_HAND_JOINTS),
            ],
            keys.CONNECT: [
                ("Tick.outputs:tick", "LeftArmSub.inputs:execIn"),
                ("Context.outputs:context", "LeftArmSub.inputs:context"),

                ("Tick.outputs:tick", "LeftGripperSub.inputs:execIn"),
                ("Context.outputs:context", "LeftGripperSub.inputs:context"),

                ("Tick.outputs:tick", "RightArmSub.inputs:execIn"),
                ("Context.outputs:context", "RightArmSub.inputs:context"),

                ("Tick.outputs:tick", "RightHandSub.inputs:execIn"),
                ("Context.outputs:context", "RightHandSub.inputs:context"),
            ],
        },
    )

    # Let the generic subscribers generate dynamic output fields.
    app = __import__("omni.kit.app").kit.app.get_app()
    for _ in range(3):
        await app.next_update_async()

    # Connect dynamic outputs after they exist.
    og.Controller.connect(
        og.Controller.attribute(graph_path + "/LeftArmSub.outputs:data"),
        og.Controller.attribute(graph_path + "/LeftArmCtrl.inputs:positionCommand"),
    )
    og.Controller.connect(
        og.Controller.attribute(graph_path + "/LeftArmSub.outputs:execOut"),
        og.Controller.attribute(graph_path + "/LeftArmCtrl.inputs:execIn"),
    )

    og.Controller.connect(
        og.Controller.attribute(graph_path + "/LeftGripperSub.outputs:data"),
        og.Controller.attribute(graph_path + "/LeftGripperArray.inputs:input0"),
    )
    og.Controller.connect(
        og.Controller.attribute(graph_path + "/LeftGripperArray.outputs:array"),
        og.Controller.attribute(graph_path + "/LeftGripperCtrl.inputs:positionCommand"),
    )
    og.Controller.connect(
        og.Controller.attribute(graph_path + "/LeftGripperSub.outputs:execOut"),
        og.Controller.attribute(graph_path + "/LeftGripperCtrl.inputs:execIn"),
    )

    og.Controller.connect(
        og.Controller.attribute(graph_path + "/RightArmSub.outputs:data"),
        og.Controller.attribute(graph_path + "/RightArmCtrl.inputs:positionCommand"),
    )
    og.Controller.connect(
        og.Controller.attribute(graph_path + "/RightArmSub.outputs:execOut"),
        og.Controller.attribute(graph_path + "/RightArmCtrl.inputs:execIn"),
    )

    og.Controller.connect(
        og.Controller.attribute(graph_path + "/RightHandSub.outputs:data"),
        og.Controller.attribute(graph_path + "/RightHandCtrl.inputs:positionCommand"),
    )
    og.Controller.connect(
        og.Controller.attribute(graph_path + "/RightHandSub.outputs:execOut"),
        og.Controller.attribute(graph_path + "/RightHandCtrl.inputs:execIn"),
    )

    print(f"Created ROS subscriber shadow graph: {graph_path}")
    print("This graph mirrors these topics into the sim robot:")
    print("  /isaacsim/left_arm_cmd")
    print("  /isaacsim/left_gripper_cmd")
    print("  /isaacsim/right_arm_cmd")
    print("  /isaacsim/right_hand_cmd")
    print("Press Play, then publish from the terminal with manual_command_pub.py")


asyncio.ensure_future(_build())

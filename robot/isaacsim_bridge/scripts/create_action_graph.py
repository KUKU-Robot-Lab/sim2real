"""
Create an Isaac Sim Action Graph that publishes the bridge input topics.

Usage in Isaac Sim Script Editor:
1. Open Isaac Sim.
2. Open Window > Script Editor.
3. Paste this file or run it with `exec(open(...).read())`.
4. Press Play.

This script creates an Action Graph at /ActionGraph/Sim2RealBridgePublisher with:
- one On Playback Tick
- five ROS 2 publish nodes
- constant test sources

The publish nodes expose raw command topics that the ROS 2 bridge package consumes:
- /isaacsim/left_arm_cmd
- /isaacsim/right_arm_cmd
- /isaacsim/left_gripper_cmd
- /isaacsim/right_hand_cmd
- /isaacsim/emergency_stop
"""

GRAPH_PATH = "/ActionGraph/Sim2RealBridgePublisher"


def build_graph():
    import omni.graph.core as og
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(GRAPH_PATH):
        stage.RemovePrim(GRAPH_PATH)

    graph_spec = {
        og.Controller.Keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("left_arm_const", "omni.graph.nodes.ConstantDoubleArray"),
            ("right_arm_const", "omni.graph.nodes.ConstantDoubleArray"),
            ("right_hand_const", "omni.graph.nodes.ConstantDoubleArray"),
            ("left_gripper_const", "omni.graph.nodes.ConstantDouble"),
            ("estop_const", "omni.graph.nodes.ConstantBool"),
            ("left_arm_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
            ("right_arm_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
            ("right_hand_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
            ("left_gripper_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
            ("estop_pub", "isaacsim.ros2.bridge.ROS2Publisher"),
        ],
        og.Controller.Keys.CONNECT: [
            ("tick.outputs:tick", "left_arm_pub.inputs:execIn"),
            ("tick.outputs:tick", "right_arm_pub.inputs:execIn"),
            ("tick.outputs:tick", "right_hand_pub.inputs:execIn"),
            ("tick.outputs:tick", "left_gripper_pub.inputs:execIn"),
            ("tick.outputs:tick", "estop_pub.inputs:execIn"),
            ("left_arm_const.outputs:value", "left_arm_pub.inputs:data"),
            ("right_arm_const.outputs:value", "right_arm_pub.inputs:data"),
            ("right_hand_const.outputs:value", "right_hand_pub.inputs:data"),
            ("left_gripper_const.outputs:value", "left_gripper_pub.inputs:data"),
            ("estop_const.outputs:value", "estop_pub.inputs:data"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("left_arm_const.inputs:value", [0.0] * 7),
            ("right_arm_const.inputs:value", [0.0] * 7),
            ("right_hand_const.inputs:value", [0.0] * 20),
            ("left_gripper_const.inputs:value", 0.0),
            ("estop_const.inputs:value", False),
            ("left_arm_pub.inputs:messagePackage", "std_msgs"),
            ("left_arm_pub.inputs:messageSubfolder", "msg"),
            ("left_arm_pub.inputs:messageName", "Float64MultiArray"),
            ("left_arm_pub.inputs:topicName", "/isaacsim/left_arm_cmd"),
            ("right_arm_pub.inputs:messagePackage", "std_msgs"),
            ("right_arm_pub.inputs:messageSubfolder", "msg"),
            ("right_arm_pub.inputs:messageName", "Float64MultiArray"),
            ("right_arm_pub.inputs:topicName", "/isaacsim/right_arm_cmd"),
            ("right_hand_pub.inputs:messagePackage", "std_msgs"),
            ("right_hand_pub.inputs:messageSubfolder", "msg"),
            ("right_hand_pub.inputs:messageName", "Float64MultiArray"),
            ("right_hand_pub.inputs:topicName", "/isaacsim/right_hand_cmd"),
            ("left_gripper_pub.inputs:messagePackage", "std_msgs"),
            ("left_gripper_pub.inputs:messageSubfolder", "msg"),
            ("left_gripper_pub.inputs:messageName", "Float64"),
            ("left_gripper_pub.inputs:topicName", "/isaacsim/left_gripper_cmd"),
            ("estop_pub.inputs:messagePackage", "std_msgs"),
            ("estop_pub.inputs:messageSubfolder", "msg"),
            ("estop_pub.inputs:messageName", "Bool"),
            ("estop_pub.inputs:topicName", "/isaacsim/emergency_stop"),
        ],
    }

    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        graph_spec,
    )

    print(f"Created Action Graph: {GRAPH_PATH}")
    print("Bridge topics:")
    print("  /isaacsim/left_arm_cmd")
    print("  /isaacsim/right_arm_cmd")
    print("  /isaacsim/left_gripper_cmd")
    print("  /isaacsim/right_hand_cmd")
    print("  /isaacsim/emergency_stop")
    print("")
    print("If you re-run this script, the old graph at the same path is replaced.")
    print("Edit the Constant nodes in Action Graph to test raw command output.")
    print("For policy integration, replace the Constant nodes with your policy outputs.")


if __name__ == "__main__":
    build_graph()

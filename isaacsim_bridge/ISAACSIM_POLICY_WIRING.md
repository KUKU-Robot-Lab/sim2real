# Isaac Sim Policy Wiring

This file shows how to replace the test `Constant` nodes in the Action Graph with policy output nodes.

## Goal

Keep the same five bridge topics, but drive them from policy output instead of fixed values:

- `/isaacsim/left_arm_cmd`
- `/isaacsim/right_arm_cmd`
- `/isaacsim/left_gripper_cmd`
- `/isaacsim/right_hand_cmd`
- `/isaacsim/emergency_stop`

## Recommended Pattern

Use one `Script Node` that emits all outputs, then connect that node into the existing ROS 2 Publisher nodes.

## ScriptNode Template

Use:

- `scripts/policy_scriptnode_template.py`

In the Action Graph:

1. Add `omni.graph.scriptnode.ScriptNode`.
2. Add these outputs on the node:
   - `leftArmCmd` as `double[]`
   - `rightArmCmd` as `double[]`
   - `leftGripperCmd` as `double`
   - `rightHandCmd` as `double[]`
   - `estop` as `bool`
3. Paste the template into the Script Node.
4. Replace the placeholder zero outputs with your policy inference.

## Rewire Existing Graph

Disconnect these current connections:

- `left_arm_const.outputs:value -> left_arm_pub.inputs:data`
- `right_arm_const.outputs:value -> right_arm_pub.inputs:data`
- `left_gripper_const.outputs:value -> left_gripper_pub.inputs:data`
- `right_hand_const.outputs:value -> right_hand_pub.inputs:data`
- `estop_const.outputs:value -> estop_pub.inputs:data`

Connect instead:

- `policy_script.outputs:leftArmCmd -> left_arm_pub.inputs:data`
- `policy_script.outputs:rightArmCmd -> right_arm_pub.inputs:data`
- `policy_script.outputs:leftGripperCmd -> left_gripper_pub.inputs:data`
- `policy_script.outputs:rightHandCmd -> right_hand_pub.inputs:data`
- `policy_script.outputs:estop -> estop_pub.inputs:data`

## Shape Requirements

- left arm: exactly 7 values
- right arm: exactly 7 values
- left gripper: exactly 1 scalar
- right hand: exactly 20 values

If lengths do not match, the ROS 2 bridge drops the command and logs a warning.

## Safety Layer

The ROS 2 bridge now enforces:

- command clamping on all outputs
- emergency stop via `/isaacsim/emergency_stop`

So even when your policy is connected directly, out-of-range commands are clipped before they hit hardware.

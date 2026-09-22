"""
Template code for an Isaac Sim ScriptNode that replaces the Constant nodes
created by create_action_graph.py.

How to use:
1. Add a `Script Node` to the Action Graph.
2. Add these outputs on the node:
   - `leftArmCmd`    : double[]
   - `rightArmCmd`   : double[]
   - `leftGripperCmd`: double
   - `rightHandCmd`  : double[]
   - `estop`         : bool
3. Paste this file into the Script Node's script field.
4. Connect outputs to the matching ROS 2 Publisher `data` inputs.

Replace the marked section with your actual policy inference call.
"""


def setup(db):
    db.state.initialized = True


def compute(db):
    # Replace this block with actual policy output values.
    left_arm = [0.0] * 7
    right_arm = [0.0] * 7
    left_gripper = 0.0
    right_hand = [0.0] * 20
    estop = False

    db.outputs.leftArmCmd = left_arm
    db.outputs.rightArmCmd = right_arm
    db.outputs.leftGripperCmd = left_gripper
    db.outputs.rightHandCmd = right_hand
    db.outputs.estop = estop
    return True

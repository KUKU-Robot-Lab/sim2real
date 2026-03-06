# Isaac Sim Action Graph Setup

This file defines the exact Isaac Sim-side graph layout for the ROS 2 bridge in this repo.

## Goal

Publish the four bridge input topics from Isaac Sim:
Publish the five bridge input topics from Isaac Sim:

- `/isaacsim/left_arm_cmd`
- `/isaacsim/right_arm_cmd`
- `/isaacsim/left_gripper_cmd`
- `/isaacsim/right_hand_cmd`
- `/isaacsim/emergency_stop`

The ROS 2 package `isaacsim_bridge/bridge_node.py` subscribes to those topics and forwards them to the real controllers.

## Fastest Path

Run the Script Editor helper:

- `scripts/create_action_graph.py`

In Isaac Sim:

1. Open `Window > Script Editor`.
2. Run:

Open `scripts/create_action_graph.py` in Isaac Sim Script Editor and run it.

3. Press Play.

That creates:

- graph path: `/ActionGraph/Sim2RealBridgePublisher`
- four ROS 2 publishers
- five ROS 2 publishers
- constant command sources for quick testing, including emergency stop

## Node Layout

Required nodes:

- `On Playback Tick`
- `Read Simulation Time`
- `Constant Double Array` for left arm
- `Constant Double Array` for right arm
- `Constant Double Array` for right hand
- `Constant Double` for left gripper
- `Constant Bool` for emergency stop
- `ROS2 Publisher` x4
- `ROS2 Publisher` x1 for emergency stop

Message types:

- left arm: `std_msgs/msg/Float64MultiArray`
- right arm: `std_msgs/msg/Float64MultiArray`
- right hand: `std_msgs/msg/Float64MultiArray`
- left gripper: `std_msgs/msg/Float64`
- emergency stop: `std_msgs/msg/Bool`

## Topic Mapping

- left arm publisher topic: `/isaacsim/left_arm_cmd`
- right arm publisher topic: `/isaacsim/right_arm_cmd`
- left gripper publisher topic: `/isaacsim/left_gripper_cmd`
- right hand publisher topic: `/isaacsim/right_hand_cmd`
- emergency stop publisher topic: `/isaacsim/emergency_stop`

## Command Dimensions

- left arm: 7 values
- right arm: 7 values
- left gripper: 1 value
- right hand: 20 values

## How To Use With Policy Output

For initial connectivity testing, keep the `Constant` nodes.

When you connect a policy:

1. Replace each `Constant` node with the node that emits your policy output.
2. Keep the same topic names.
3. Preserve the exact vector lengths above.
4. Drive `/isaacsim/emergency_stop` high when you need command forwarding to stop.

For a direct ScriptNode template, use:

- `ISAACSIM_POLICY_WIRING.md`
- `scripts/policy_scriptnode_template.py`

## Quick ROS 2 Check

With the bridge running, confirm Isaac Sim is publishing:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 topic echo /isaacsim/left_arm_cmd
```

Then verify the bridge is relaying:

```bash
ros2 topic echo /left_joint_trajectory_controller/joint_trajectory
```

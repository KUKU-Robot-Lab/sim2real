# OpenArm Control

This folder contains the minimum custom bringup for the current OpenArm hardware layout:

- left arm with OpenArm gripper enabled
- right arm without OpenArm gripper

Use:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch "${REPO_DIR}/openarm_control/launch/openarm_left_gripper_bimanual_real.launch.py" \
  left_can_interface:=can1 \
  right_can_interface:=can0
```

For bench testing without real CAN hardware:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch "${REPO_DIR}/openarm_control/launch/openarm_left_gripper_bimanual_real.launch.py" \
  use_fake_hardware:=true
```

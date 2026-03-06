# Integrated Control

This folder contains the minimum combined control layer for:

- left OpenArm arm + left OpenArm gripper
- right OpenArm arm
- right Tesollo DG5 hand

Main launch:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch "${REPO_DIR}/integrated_control/launch/openarm_left_gripper_right_dg5_real.launch.py" \
  left_can_interface:=can1 \
  right_can_interface:=can0 \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

Files:

- `launch/openarm_left_gripper_right_dg5_real.launch.py`: combined bringup
- `config/openarm_left_gripper_bimanual_controllers.yaml`: OpenArm controller set with right gripper removed

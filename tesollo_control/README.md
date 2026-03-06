# Tesollo Control

This folder wraps the upstream `dg5f_right_driver` launch with the defaults used in the current setup.

Use:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch "${REPO_DIR}/tesollo_control/launch/dg5f_right_real.launch.py" \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

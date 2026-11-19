#!/usr/bin/env bash
# Install local SDK into a separate ROS-compatible Python 3.10 environment.
# No CAN access, no robot process is started.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENV_DIR=${ARM_ZERO_ENV_DIR:-$HOME/.cache/arm-zero/venv}
if [ ! -x "$ENV_DIR/bin/python" ]; then
  uv venv --python /usr/bin/python3 --system-site-packages "$ENV_DIR"
fi
export CMAKE_PREFIX_PATH="$ROOT/../robot_control/ros_ws/install/openarm_can${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
uv pip install --python "$ENV_DIR/bin/python" "$ROOT/../robot_control/ros_ws/src/openarm_can/python"
uv pip install --python "$ENV_DIR/bin/python" "numpy==1.26.4" "scipy==1.15.3" "trimesh==4.12.2" "PyYAML>=6,<7" "python-fcl==0.7.0.11"
source /opt/ros/humble/setup.bash
"$ENV_DIR/bin/python" -c 'import rclpy, openarm_can, numpy, yaml, trimesh; print("Calibration environment OK — no device access")'

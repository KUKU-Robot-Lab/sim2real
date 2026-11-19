#!/usr/bin/env bash
# User-operated calibration entry point. Setup/check never opens a device.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source /opt/ros/humble/setup.bash
source "$ROOT/../robot_control/ros_ws/install/setup.bash"
export ROS_DOMAIN_ID=50
PY=${ARM_ZERO_PYTHON:-$HOME/.cache/arm-zero/venv/bin/python}
if [ ! -x "$PY" ]; then
  echo '먼저 bash scripts/ops/arm_zero_env.sh 를 실행하세요.' >&2
  exit 2
fi
if [ "${1:-}" = '--check' ]; then
  exec "$PY" -c 'import sys, rclpy, openarm_can, numpy, yaml, trimesh; print("Python:", sys.version); print("ROS/CAN SDK/모델 의존성 import OK — 장치 접근 없음")'
fi
exec "$PY" "$ROOT/scripts/ops/arm_zero_session.py" "$@"

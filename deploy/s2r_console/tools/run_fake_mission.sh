#!/usr/bin/env bash
# fake 미션을 콘솔로 끝까지 밟는다 — console.sh 와 같은 환경(ROS + 워크스페이스 오버레이 + venv)으로.
#   deploy/s2r_console/tools/run_fake_mission.sh [--with-viewer] [--skip sensors ...]
set -eo pipefail
SIM2REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source /opt/ros/humble/setup.bash
for overlay in "$SIM2REAL/../robot_control/ros_ws/install" "$SIM2REAL/install"; do
  [ -f "$overlay/local_setup.bash" ] && source "$overlay/local_setup.bash"
done
[ -d "$SIM2REAL/.venv/bin" ] && export PATH="$SIM2REAL/.venv/bin:$PATH"
export PYTHONPATH="$SIM2REAL/deploy/s2r_console${PYTHONPATH:+:$PYTHONPATH}"
cd "$SIM2REAL"
exec "$SIM2REAL/.venv/bin/python" "$SIM2REAL/deploy/s2r_console/tools/run_fake_mission.py" "$@"

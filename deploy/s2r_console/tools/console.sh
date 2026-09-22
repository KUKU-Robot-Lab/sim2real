#!/usr/bin/env bash
# S2R 배포 콘솔을 띄운다 — 브라우저에서 http://127.0.0.1:8091
#
#   deploy/s2r_console/tools/console.sh                                  # 프로파일을 화면에서 고른다
#   deploy/s2r_console/tools/console.sh --profile pour_i18_fake --operator me
#   deploy/s2r_console/tools/console.sh --port 8092
#
# 이 스크립트가 있는 이유는 하나다: PYTHONPATH 를 *덮어쓰면* ROS 가 넣어 둔 rclpy 경로가 날아가
# 브리지가 "No module named rclpy" 로 죽는다 (09.21 에 직접 밟았다). 여기서는 앞에 *덧붙인다*.
set -eo pipefail
SIM2REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # deploy/s2r_console/tools → 저장소 루트
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PY="$SIM2REAL/.venv/bin/python"

if [ -f "$ROS_SETUP" ]; then
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
else
  echo "[console.sh] $ROS_SETUP 가 없다 — 브리지 없이 화면만 뜬다 (--no-bridge 와 같다)" >&2
  set -- --no-bridge "$@"
fi
[ -x "$PY" ] || PY="$(command -v python3)"

cd "$SIM2REAL"
export PYTHONPATH="$SIM2REAL/deploy/s2r_console${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m s2r_console "$@"

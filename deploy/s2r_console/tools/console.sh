#!/usr/bin/env bash
# S2R 배포 콘솔을 띄운다 — 브라우저에서 http://127.0.0.1:8091 (또는 --window 로 자기 창)
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

# 미션의 `ros2 launch openarm_bringup|dg5f_driver|policy_control` 는 워크스페이스 오버레이에 산다. /opt/ros 만 올리면
# "Package 'openarm_bringup' not found" 로 2 s 안에 죽는다(2026-09-22 실기 bringup). 자식은 이 셸의 환경을 받는다.
# 순서: 벤더 드라이버(robot_control) 위에 sim2real — 같은 이름이면 뒤가 이긴다.
ROBOT_CONTROL_INSTALL="${ROBOT_CONTROL_INSTALL:-$SIM2REAL/../robot_control/ros_ws/install}"
for overlay in "$ROBOT_CONTROL_INSTALL" "$SIM2REAL/install"; do
  if [ -f "$overlay/local_setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$overlay/local_setup.bash"
  else
    echo "[console.sh] ★오버레이 없음: $overlay — 그 안의 패키지를 쓰는 단계는 실패한다" >&2
  fi
done

cd "$SIM2REAL"
# 미션 명령의 `python3` 는 venv 여야 한다 — 콘솔은 자식에게 이 셸의 환경을 그대로 넘긴다. venv 를 활성화하지 않은
# 셸에서 띄우면 시스템 파이썬(torch 2.2 · rl_games 없음)이 잡혀 preflight 테스트가 떨어진다(2026-09-22 실측).
# venv 는 include-system-site-packages 라 ROS 경로(위 source 가 넣은 PYTHONPATH)도 그대로 보인다.
[ -d "$SIM2REAL/.venv/bin" ] && export PATH="$SIM2REAL/.venv/bin:$PATH"
export PYTHONPATH="$SIM2REAL/deploy/s2r_console${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m s2r_console "$@"

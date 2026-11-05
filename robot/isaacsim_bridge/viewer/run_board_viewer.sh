#!/usr/bin/env bash
# 보드 뷰어 런처 — 조작 패널(board_jog) UDP(127.0.0.1:47813)를 받아 Isaac 장면에 그린다. ROS 를 쓰지 않는다(발행 0).
#
#   ./run_board_viewer.sh                       # GUI(DISPLAY, 기본 :0)
#   BOARD_FAKE=1 ./run_board_viewer.sh          # 가짜 패널(fake_board_sender.py)도 같이 — 시험용
#   추가 인자는 board_viewer.py 로 간다(--eye, --lookat, --shot, --max_seconds …). 포트는 BOARD_PORT(기본 47813).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_WS="$(cd "${HERE}/../../../.." && pwd)"
ISAACLAB="${ISAACLAB_ROOT:-${RL_WS}/IsaacLab}"
PORT="${BOARD_PORT:-47813}"
export DISPLAY="${DISPLAY:-:0}"

# 실기 FP++ 노드(python)가 같은 GPU 에 늘 있으니 이름으로 막지 않고 여유 메모리로만 본다(뷰어 ~3-4 GB).
MIN_FREE_MIB="${VIEWER_MIN_FREE_MIB:-8000}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
if [[ -n "${FREE_MIB}" && "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]]; then
    echo "[run_board_viewer] GPU 여유 ${FREE_MIB} MiB < ${MIN_FREE_MIB} MiB — 띄우지 않는다(VIEWER_FORCE=1 로만 무시)" >&2
    [[ "${VIEWER_FORCE:-0}" == "1" ]] || exit 4
fi

if [[ "${BOARD_FAKE:-0}" == "1" ]]; then
    python3 "${HERE}/fake_board_sender.py" --port "${PORT}" &
    FAKE_PID=$!
    trap 'kill "${FAKE_PID}" 2>/dev/null || true' EXIT INT TERM
    echo "[run_board_viewer] fake sender pid=${FAKE_PID}"
fi

# ROS·sim2real .venv 흔적을 지운 환경(py3.10 이 Isaac py3.11 에 섞이지 않게, run_viewer.sh 와 같다)
strip_ros() { tr ':' '\n' <<<"${1:-}" | grep -v -e '^/opt/ros/' -e '/sim2real/.venv/' | paste -sd: -; }
env -u PYTHONPATH -u VIRTUAL_ENV -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION \
    -u ROS_PYTHON_VERSION -u CMAKE_PREFIX_PATH \
    LD_LIBRARY_PATH="$(strip_ros "${LD_LIBRARY_PATH:-}")" \
    PATH="$(strip_ros "${PATH}")" \
    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    "${ISAACLAB}/isaaclab.sh" -p "${HERE}/board_viewer.py" --port "${PORT}" "$@"

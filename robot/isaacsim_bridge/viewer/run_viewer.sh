#!/usr/bin/env bash
# 읽기 전용 Isaac 뷰어 런처 — relay(ROS Humble, 구독만) + Isaac 뷰어(ROS 없음)를 각자 환경으로 띄운다.
#
#   ROS_DOMAIN_ID=126 ./run_viewer.sh                 # GUI
#   ROS_DOMAIN_ID=126 ./run_viewer.sh --cup cup_big_s100
#   추가 인자는 isaac_viewer.py 로 간다. 포트는 VIEWER_PORT(기본 47811).
#
# 로봇에 명령을 보내는 경로는 없다. relay 는 JointState 구독만, 뷰어는 UDP 수신만 한다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_WS="$(cd "${HERE}/../../../.." && pwd)"
ISAACLAB="${ISAACLAB_ROOT:-${RL_WS}/IsaacLab}"
PORT="${VIEWER_PORT:-47811}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

if [[ -z "${ROS_DOMAIN_ID:-}" || "${ROS_DOMAIN_ID}" == "0" ]]; then
    echo "[run_viewer] ROS_DOMAIN_ID 가 비었거나 0 — 실기 도메인을 명시할 것(예: ROS_DOMAIN_ID=126). 거부." >&2
    exit 2
fi

# 같은 GPU 에서 학습 중이면 띄우지 않는다(PhysX 메모리 부족으로 학습 sim 이 죽는다).
if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | grep -Eiq 'python|kit|isaac'; then
    echo "[run_viewer] GPU 에 python/Isaac 계산 프로세스가 있다 — 학습 중일 수 있어 뷰어를 띄우지 않는다:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    [[ "${VIEWER_FORCE:-0}" == "1" ]] || exit 3
fi

# 이름에 python 이 없는 큰 GPU 사용자(09.22: VLLM::EngineCore 28 GB)도 있다 — 여유 메모리로도 본다.
MIN_FREE_MIB="${VIEWER_MIN_FREE_MIB:-8000}"
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
if [[ -n "${FREE_MIB}" && "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]]; then
    echo "[run_viewer] GPU 여유 메모리 ${FREE_MIB} MiB < ${MIN_FREE_MIB} MiB — 뷰어를 띄우지 않는다(다른 작업을 밀어낼 수 있다):" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    [[ "${VIEWER_FORCE:-0}" == "1" ]] || exit 4
fi

# 1) relay — ROS Humble 환경(서브셸에서만 source)
(
    set +u
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"
    set -u
    exec python3 "${HERE}/joint_state_relay.py" --port "${PORT}"
) &
RELAY_PID=$!
cleanup() {
    if kill -0 "${RELAY_PID}" 2>/dev/null; then
        kill -INT "${RELAY_PID}" 2>/dev/null || true
        wait "${RELAY_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM
echo "[run_viewer] relay pid=${RELAY_PID} (domain ${ROS_DOMAIN_ID}, udp 127.0.0.1:${PORT})"

# 2) Isaac 뷰어 — ROS 흔적을 지운 환경(py3.10 rclpy 가 py3.11 에 섞이지 않게)
# 콘솔이 띄우면 sim2real/.venv(py3.10) 도 PATH 앞에 있다 — Isaac 의 python 과 섞이지 않게 같이 지운다
strip_ros() { tr ':' '\n' <<<"${1:-}" | grep -v -e '^/opt/ros/' -e '/sim2real/.venv/' | paste -sd: -; }
env -u PYTHONPATH -u VIRTUAL_ENV -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION \
    -u ROS_PYTHON_VERSION -u CMAKE_PREFIX_PATH \
    LD_LIBRARY_PATH="$(strip_ros "${LD_LIBRARY_PATH:-}")" \
    PATH="$(strip_ros "${PATH}")" \
    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    "${ISAACLAB}/isaaclab.sh" -p "${HERE}/isaac_viewer.py" --port "${PORT}" "$@"

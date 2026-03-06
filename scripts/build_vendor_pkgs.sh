#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_BRIDGE_ONLY=0
if [[ "${1:-}" == "--bridge-only" ]]; then
  BUILD_BRIDGE_ONLY=1
fi

set +u
source /opt/ros/humble/setup.bash
set -u

cd "${REPO_DIR}"

if [[ "${BUILD_BRIDGE_ONLY}" -eq 1 ]]; then
  colcon build --base-paths \
    "${REPO_DIR}/isaacsim_bridge"
else
  colcon build --base-paths \
    "${REPO_DIR}/isaacsim_bridge" \
    "${REPO_DIR}/vendor/openarm/openarm_description" \
    "${REPO_DIR}/vendor/openarm/openarm_can" \
    "${REPO_DIR}/vendor/openarm/openarm_hardware" \
    "${REPO_DIR}/vendor/openarm/openarm_bringup" \
    "${REPO_DIR}/vendor/tesollo/dg_description" \
    "${REPO_DIR}/vendor/tesollo/dg_msgs" \
    "${REPO_DIR}/vendor/tesollo/delto_tcp_comm" \
    "${REPO_DIR}/vendor/tesollo/delto_hardware" \
    "${REPO_DIR}/vendor/tesollo/dg5f_driver"
fi

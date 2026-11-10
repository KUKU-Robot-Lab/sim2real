#!/bin/bash
# VLM 감독 실행기 — 5090 에서.
#   bash ~/rl_ws/vlm_sup/sup.sh quota       # 오늘 VLM 한도 · 토큰 장부
#   bash ~/rl_ws/vlm_sup/sup.sh preflight   # 실기 전 점검(읽기만)
#   bash ~/rl_ws/vlm_sup/sup.sh shadow      # 판단만 기록 — 승인은 사람이 콘솔에서
#   bash ~/rl_ws/vlm_sup/sup.sh active      # VLM 이 승인 · 멈춤까지 (시작 때 Enter 한 번)
#   bash ~/rl_ws/vlm_sup/sup.sh test [chain] # 가짜 실행기 + 녹화 프레임으로 끝까지(로봇 없음, 도메인 77 localhost)
#   공통 덧붙임: VLM_CHAIN=gemini-robotics-er-2-preview,local8b,local  GLEVEL=low  추가 인자는 vlm_supervisor.py 로
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S2R="$HOME/rl_ws/sim2real"
cmd="${1:-help}"; shift || true
source /opt/ros/humble/setup.bash
real_env() {
  source "$S2R/config/ros_domain.env"
  export ROS_DOMAIN_ID="$S2R_REAL_DOMAIN" ROS_LOCALHOST_ONLY=0
  source "$S2R/config/dds/lan_profile.sh"
}
case "$cmd" in
  quota) python3 "$HERE/vlm_backends.py" 2>&1 | grep -v -i "warn" ;;
  preflight) real_env; python3 "$HERE/preflight.py" 2>&1 | grep -v -i "RequestsDependencyWarning\|warnings.warn" ;;
  shadow|active) real_env; exec python3 -W ignore "$HERE/vlm_supervisor.py" --mode "$cmd" "$@" ;;
  test)
    export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1; unset FASTRTPS_DEFAULT_PROFILES_FILE
    export VLM_CHAIN="${1:-local}"; shift || true
    T=/tmp/vlm_sup_test; rm -rf "$T"; mkdir -p "$T"
    python3 "$HERE/fake_runner.py" "$HERE/test_frames/015557" "$T/approvals.jsonl" > "$T/fake.log" 2>&1 &
    FP=$!; trap 'kill $FP 2>/dev/null' EXIT; sleep 2
    python3 -W ignore "$HERE/vlm_supervisor.py" --mode active --yes --approvals "$T/approvals.jsonl" "$@" || true
    echo "--- fake runner"; cat "$T/fake.log" ;;
  *) sed -n 2,9p "$0" ;;
esac

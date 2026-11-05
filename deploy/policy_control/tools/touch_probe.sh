#!/usr/bin/env bash
# 손끝 접촉 보정 재생(touch_probe.py)을 실기 도메인 · 랜 DDS 설정 그대로 띄운다(콘솔 pd 와 같은 환경). 인자는 touch_probe.py 로 넘긴다.
#   deploy/policy_control/tools/touch_probe.sh --plan logs/analysis/fk_touch_probe_1105/plan_right.npz --out logs/analysis/fk_touch_probe_1105/rec/right_1.npz            # 계획만(발행 0)
#   deploy/policy_control/tools/touch_probe.sh --plan logs/analysis/fk_touch_probe_1105/plan_right.npz --out logs/analysis/fk_touch_probe_1105/rec/right_1.npz --step --execute  # 실기 — 사용자 승인 뒤
set -eo pipefail
cd "$(dirname "$0")/../../.."
source /opt/ros/humble/setup.bash
[ -f "$HOME/rl_ws/robot_control/ros_ws/install/setup.bash" ] && source "$HOME/rl_ws/robot_control/ros_ws/install/setup.bash"
source config/ros_domain.env
export ROS_DOMAIN_ID="$S2R_REAL_DOMAIN" ROS_LOCALHOST_ONLY=0
source config/dds/lan_profile.sh
exec python3 deploy/policy_control/tools/touch_probe.py "$@"

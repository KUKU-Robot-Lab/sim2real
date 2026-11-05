#!/usr/bin/env bash
# 보드 조그 기록 창을 실기 도메인 · 랜 DDS 설정 그대로 띄운다(teach_place.sh 와 같은 환경). 인자는 board_jog.py 로 넘긴다.
# python = sim2real/.venv(rclpy · PyQt5 · cv2 4.14 — --board_npz 의 plan_board 가 CharucoDetector 를 쓴다). 창은 5090 로컬 화면 :0.
#   deploy/policy_control/tools/board_jog.sh --side right --out logs/analysis/fk_board_jog_1105/rec/right_1.json            # dry(발행 0)
#   deploy/policy_control/tools/board_jog.sh --side right --out logs/analysis/fk_board_jog_1105/rec/right_1.json --execute  # 실기 — 사용자 승인 뒤
#   deploy/policy_control/tools/board_jog.sh --side right --boards charuco,grid50,grid50b --board_pose <boards.yaml> --out …/right_2.json  # 보드 여럿
#   deploy/policy_control/tools/board_jog.sh --side right --seed …/rec/right_3_relabel.json --out …/rec/right_4.json --execute  # 4 모서리 불러와 남은 점 전체 자동 + 점마다 사진
# 사진용 머리 카메라(--camera auto 기본): 영상이 없고 realsense2_camera_node 프로세스도 없으면 이 환경(도메인 · 아래 lan_profile)
# 그대로 camera_up.sh 와 같은 rs_launch 를 setsid 로 띄우고, 끝낼 때 그 그룹만 SIGTERM 한다(board_photo.Camera).
set -eo pipefail
cd "$(dirname "$0")/../../.."
source /opt/ros/humble/setup.bash
[ -f "$HOME/rl_ws/robot_control/ros_ws/install/setup.bash" ] && source "$HOME/rl_ws/robot_control/ros_ws/install/setup.bash"
source config/ros_domain.env
export ROS_DOMAIN_ID="$S2R_REAL_DOMAIN" ROS_LOCALHOST_ONLY=0
source config/dds/lan_profile.sh
export DISPLAY="${DISPLAY:-:0}"
exec .venv/bin/python deploy/policy_control/tools/board_jog.py "$@"

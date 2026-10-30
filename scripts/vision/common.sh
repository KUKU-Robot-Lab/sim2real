#!/bin/bash
# 인지 PC(vision-3090 · RH 로봇 쪽 인지 PC) 공통 설정. 모든 vision/*.sh 가 source 한다.
# ★set -u 금지: /opt/ros/humble/setup.bash 가 미정의 변수를 참조해 즉사한다.
set -eo pipefail
# ★10.30 사용자: 배포 도메인은 motion_acq(126 · 7)와 분리 — 값은 config/ros_domain.env 한 곳(python 은 policy_control/ros_domain.py)
source "$(dirname "${BASH_SOURCE[0]}")/../../config/ros_domain.env"
export ROS_DOMAIN_ID="$S2R_REAL_DOMAIN"
# ★영상 · FP++ 는 이 PC 안에서만 돈다(09.26). wifi AP 가 멀티캐스트를 막고 원격 데스크톱이 상향을 채우면 DDS 쓰기가
#  막혀 카메라가 멈췄다. 로봇 PC 로는 물체 자세만 UDP 로 넘긴다(pose_tx_up.sh · scripts/fpp_udp.py).
# ★10.30 0 으로 바꿨다(사용자 · s2r:deploy 합의): 목 드라이버(/head/joint_states, 0)와 카메라 IMU 를 head_view_feedback 이 함께
#  봐야 하는데 0 과 1 노드는 같은 PC 안에서도 서로 안 보인다(10.30 실측). 인지 묶음(camera_up · fpp_up · fpp_group_up ·
#  fpp_pose_tx)은 모두 이 값을 쓴다. 09.26 의 wifi 문제는 DDS 를 랜 NIC 에 묶는 Fast-DDS 프로파일(아래 config/dds/lan_profile.sh)로 막는다.
#  되돌리기: S2R_VISION_LOCALHOST=1.
export ROS_LOCALHOST_ONLY="${S2R_VISION_LOCALHOST:-0}"
# ★10.30 DDS 를 허브 랜 NIC 에만 묶는다 — FASTRTPS_DEFAULT_PROFILES_FILE(노드 · 드라이버)과 FPP_DDS_ARGS(컨테이너 docker run).
#  ROS_LOCALHOST_ONLY 를 정한 뒤에 source 한다(1 이면 내보내지 않는다).
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/../../config/dds/lan_profile.sh"
# 09.29: 경로를 이 파일 위치에서 만든다 — PC 마다 사용자 이름이 다르다(vision-3090 usr · arm4090/arm5080 user).
#        저장소 배치는 ~/rl_ws/{sim2real,perception_plus_plus} 로 같다. 다르면 PPP 를 export 해서 덮는다.
SIM2REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PPP="${PPP:-$(dirname "$SIM2REAL")/perception_plus_plus}"
export SIM2REAL
LOGDIR=/tmp/perception
mkdir -p "$LOGDIR"
source /opt/ros/humble/setup.bash

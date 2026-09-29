#!/bin/bash
# 인지 PC(vision-3090 · RH 로봇 쪽 인지 PC) 공통 설정. 모든 vision/*.sh 가 source 한다.
# ★set -u 금지: /opt/ros/humble/setup.bash 가 미정의 변수를 참조해 즉사한다.
set -eo pipefail
export ROS_DOMAIN_ID=126
# ★영상 · FP++ 는 이 PC 안에서만 돈다(09.26). wifi AP 가 멀티캐스트를 막고 원격 데스크톱이 상향을 채우면 DDS 쓰기가
#  막혀 카메라가 멈췄다. 로봇 PC 로는 물체 자세만 UDP 로 넘긴다(pose_tx_up.sh · scripts/fpp_udp.py).
export ROS_LOCALHOST_ONLY=1
# 09.29: 경로를 이 파일 위치에서 만든다 — PC 마다 사용자 이름이 다르다(vision-3090 usr · arm4090/arm5080 user).
#        저장소 배치는 ~/rl_ws/{sim2real,perception_plus_plus} 로 같다. 다르면 PPP 를 export 해서 덮는다.
SIM2REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PPP="${PPP:-$(dirname "$SIM2REAL")/perception_plus_plus}"
export SIM2REAL
LOGDIR=/tmp/perception
mkdir -p "$LOGDIR"
source /opt/ros/humble/setup.bash

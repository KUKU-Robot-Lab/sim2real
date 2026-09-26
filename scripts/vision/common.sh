#!/bin/bash
# vision-3090 전용 공통 설정. 모든 vision/*.sh 가 source 한다.
# ★set -u 금지: /opt/ros/humble/setup.bash 가 미정의 변수를 참조해 즉사한다.
set -eo pipefail
export ROS_DOMAIN_ID=126
# ★영상 · FP++ 는 이 PC 안에서만 돈다(09.26). wifi AP 가 멀티캐스트를 막고 원격 데스크톱이 상향을 채우면 DDS 쓰기가
#  막혀 카메라가 멈췄다. 로봇 PC 로는 물체 자세만 UDP 로 넘긴다(pose_tx_up.sh · scripts/fpp_udp.py).
export ROS_LOCALHOST_ONLY=1
SIM2REAL=/home/usr/rl_ws/sim2real
PPP=/home/usr/rl_ws/perception_plus_plus
LOGDIR=/tmp/perception
mkdir -p "$LOGDIR"
source /opt/ros/humble/setup.bash

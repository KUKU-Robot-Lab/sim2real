#!/bin/bash
source "$(dirname "$0")/common.sh"
# ★11.05 실행 파일 경로로만 — 이름만 쓰면 그 문자열을 명령줄에 담은 남의 셸까지 죽인다.
pkill -f lib/realsense2_camera/realsense2_camera_node || true
pkill -f "ros2 launch realsense2_camera" || true
echo "camera down"

#!/bin/bash
# camera_down.sh [head|global] — 기본 head. 그 카메라 노드만 끈다(launch 는 노드가 끝나면 같이 끝난다).
source "$(dirname "$0")/common.sh"
case "${1:-head}" in
  head) NS=camera ;; global) NS=global_camera ;;
  *) echo "usage: camera_down.sh [head|global]" >&2; exit 2 ;;
esac
# ★11.05 실행 파일 경로로만 — 이름만 쓰면 그 문자열을 명령줄에 담은 남의 셸까지 죽인다.
# ★11.18 ns 까지 본다 — D435i 두 대(arm4090): 경로 · 'ros2 launch realsense2_camera' 만 보면 다른 카메라도 같이 꺼진다.
pkill -f "lib/realsense2_camera/realsense2_camera_node .*__ns:=/$NS( |\$)" || true
echo "camera(${1:-head}) down"

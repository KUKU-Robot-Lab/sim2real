#!/bin/bash
# RealSense ROS 노드 기동(멱등). align_depth 필수 — FP++ 가 정렬 깊이를 구독한다.
source "$(dirname "$0")/common.sh"
if pgrep -f realsense2_camera_node >/dev/null; then echo "camera already up"; exit 0; fi
setsid bash -c 'source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126 ROS_LOCALHOST_ONLY=1;
  exec ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true' \
  </dev/null >"$LOGDIR/realsense.log" 2>&1 &
# ★타입을 적고, 실패하면 1 s 쉰다 — 타입 없이 부르면 토픽이 아직 안 보일 때 즉시 끝나서 30 번이 11 s 만에 소진됐다(09.22).
for _ in $(seq 1 30); do
  if timeout 3 ros2 topic echo --once /camera/camera/color/camera_info sensor_msgs/msg/CameraInfo >/dev/null 2>&1; then
    echo "camera up"; exit 0; fi
  sleep 1
done
echo "camera did not publish within 60s (see $LOGDIR/realsense.log)" >&2; exit 1

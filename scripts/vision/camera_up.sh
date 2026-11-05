#!/bin/bash
# RealSense ROS 노드 기동(멱등). align_depth 필수 — FP++ 가 정렬 깊이를 구독한다.
# ★11.03 IMU 를 켠다(/camera/camera/imu) — 미션 head_view_check 가 중력으로 카메라 숙임을 잰다(scripts/head_view_feedback.py).
#  realsense2_camera 4.58 기본은 gyro · accel 꺼짐 · unite_imu_method 0(합친 imu 토픽 없음).
source "$(dirname "$0")/common.sh"
ros2_daemon_heal
# ★11.05 실행 파일 경로로만 찾는다 — 'pgrep -f realsense2_camera_node' 는 그 문자열을 명령줄에 담은 남의 셸까지 잡아
#  카메라가 없는데 '떠 있다'로 보고, set -e 아래 pkill(대상 없음 → 1)에서 rc=1 로 죽었다(14:32 head_view_check).
NODE_PAT='lib/realsense2_camera/realsense2_camera_node'
if pgrep -f "$NODE_PAT" >/dev/null; then
  # ★11.03 IMU 없이 떠 있던 카메라(이전 camera_up)면 내리고 IMU 를 켜서 다시 띄운다 — 그대로 두면 head_view_check 가 늘 실패한다.
  if timeout 5 ros2 topic list 2>/dev/null | grep -qx /camera/camera/imu; then echo "camera already up"; exit 0; fi
  echo "camera up without IMU — restarting with IMU"
  pkill -f "realsense2_camera/launch/rs_launch.py" || true; pkill -f "$NODE_PAT" || true
  for _ in $(seq 1 10); do pgrep -f "$NODE_PAT" >/dev/null || break; sleep 0.5; done
fi
setsid bash -c "source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY;
  exec ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true \
    enable_gyro:=true enable_accel:=true unite_imu_method:=2" \
  </dev/null >"$LOGDIR/realsense.log" 2>&1 &
# ★타입을 적고, 실패하면 1 s 쉰다 — 타입 없이 부르면 토픽이 아직 안 보일 때 즉시 끝나서 30 번이 11 s 만에 소진됐다(09.22).
for _ in $(seq 1 30); do
  if timeout 3 ros2 topic echo --once /camera/camera/color/camera_info sensor_msgs/msg/CameraInfo >/dev/null 2>&1; then
    echo "camera up"; exit 0; fi
  sleep 1
done
echo "camera did not publish within 60s (see $LOGDIR/realsense.log)" >&2; exit 1

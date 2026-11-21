#!/bin/bash
# RealSense ROS 노드 기동(멱등). align_depth 필수 — FP++ 가 정렬 깊이를 구독한다.
#   camera_up.sh [head|global]   기본 head(머리 · ns /camera · 토픽 /camera/camera/*, 미션 · 도구가 부르는 그대로)
#                                global = 글로벌 카메라(ns /global_camera · 토픽 /global_camera/camera/*, IMU 끔)
# ★11.18 arm4090 에 D435i 가 두 대다 — serial_no 를 주지 않으면 어느 쪽을 잡을지 모른다. 시리얼은 사이트 파일 rs_cameras
#   (deploy/policy_control/config/sites/arm_v2.yaml, policy_control/site_ports.py camera_serial)에서 읽고, 못 읽으면 띄우지 않는다.
# ★11.03 머리는 IMU 를 켠다(/camera/camera/imu) — 미션 head_view_check 가 중력으로 카메라 숙임을 잰다(scripts/head_view_feedback.py).
#  realsense2_camera 4.58 기본은 gyro · accel 꺼짐 · unite_imu_method 0(합친 imu 토픽 없음).
source "$(dirname "$0")/common.sh"
ROLE="${1:-head}"
case "$ROLE" in
  head)   NS=camera;        EXTRA="enable_gyro:=true enable_accel:=true unite_imu_method:=2" ;;
  # 두 노드가 같은 camera_name 으로 /tf_static 에 camera_link 등을 내면 머리 TF 가 덮인다 — 글로벌은 TF 를 내지 않는다
  #  (우리 도구는 TF 대신 외부 파라미터 yaml 을 쓴다).
  global) NS=global_camera; EXTRA="publish_tf:=false" ;;
  *) echo "usage: camera_up.sh [head|global]" >&2; exit 2 ;;
esac
LOG="$LOGDIR/realsense$([ "$ROLE" = head ] || echo "_$ROLE").log"   # 머리는 옛 이름 그대로
SERIAL=$(python3 "$SIM2REAL/deploy/policy_control/policy_control/site_ports.py" camera "$ROLE") || exit 1
ros2_daemon_heal
# ★11.05 실행 파일 경로로만 찾는다 — 'pgrep -f realsense2_camera_node' 는 그 문자열을 명령줄에 담은 남의 셸까지 잡아
#  카메라가 없는데 '떠 있다'로 보고, set -e 아래 pkill(대상 없음 → 1)에서 rc=1 로 죽었다(14:32 head_view_check).
# ★11.18 노드 명령줄의 '-r __ns:=/<ns> ' 까지 본다 — 경로만 보면 다른 카메라 노드를 이 카메라로 보고, 아래 재기동이 그것까지 끈다.
NODE_PAT="lib/realsense2_camera/realsense2_camera_node .*__ns:=/$NS( |\$)"
if pgrep -f "$NODE_PAT" >/dev/null; then
  # ★11.03 IMU 없이 떠 있던 머리 카메라(이전 camera_up)면 내리고 IMU 를 켜서 다시 띄운다 — 그대로 두면 head_view_check 가 늘 실패한다.
  if [ "$ROLE" != head ] || timeout 5 ros2 topic list 2>/dev/null | grep -qx /camera/camera/imu; then echo "camera($ROLE) already up"; exit 0; fi
  echo "camera up without IMU — restarting with IMU"
  # 노드만 끈다 — 그 launch 는 노드가 끝나면 같이 끝난다(launch 명령줄로는 머리 · 글로벌을 못 가른다: 11.18 전 머리 launch 는 ns 인자가 없다)
  pkill -f "$NODE_PAT" || true
  for _ in $(seq 1 10); do pgrep -f "$NODE_PAT" >/dev/null || break; sleep 0.5; done
fi
setsid bash -c "source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY;
  exec ros2 launch realsense2_camera rs_launch.py camera_namespace:=$NS serial_no:=_$SERIAL align_depth.enable:=true $EXTRA" \
  </dev/null >"$LOG" 2>&1 &
# ★타입을 적고, 실패하면 1 s 쉰다 — 타입 없이 부르면 토픽이 아직 안 보일 때 즉시 끝나서 30 번이 11 s 만에 소진됐다(09.22).
for _ in $(seq 1 30); do
  if timeout 3 ros2 topic echo --once "/$NS/camera/color/camera_info" sensor_msgs/msg/CameraInfo >/dev/null 2>&1; then
    echo "camera($ROLE) up · serial $SERIAL"; exit 0; fi
  sleep 1
done
echo "camera($ROLE) did not publish within 60s (see $LOG)" >&2; exit 1

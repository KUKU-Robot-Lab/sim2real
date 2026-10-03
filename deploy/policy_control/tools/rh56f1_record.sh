#!/bin/bash
# RH56F1 실기 기록 — 팔 · 손을 **따로** bag 에 쓴다(10.03 실측: sqlite3 한 프로세스에 3750 msg/s 를 몰면 /joint_states 750 → 706 Hz
# 로 6 % 빠졌다. 둘로 나누면 /joint_states 750.1 Hz · 손 14 토픽 250.1 Hz 손실 0). mcap 저장기가 없어서 sqlite3 기준.
#
#   bash rh56f1_record.sh start [이름]     # ~/rl_ws/sim2real/logs/bags/<날짜_시각>_<이름>/{arm,hand}
#   bash rh56f1_record.sh stop             # 두 기록기에 SIGINT(정상 마무리) — PID 파일의 프로세스만
#   EXTRA="/policy_control/joint_target /status/pd_right" bash rh56f1_record.sh start   # 팔 쪽 bag 에 더할 토픽
#
# 도장: 손 토픽 header = 하드웨어 샘플 시각(EtherCAT 마스터가 PDO 를 받은 순간), 팔 /joint_states = ros2_control 갱신 시각.
# 크기: 손 약 0.7 MB/s · 팔 약 0.7 MB/s(10.03 측정 기준).
set -eo pipefail
CMD=${1:?start|stop}; NAME=${2:-run}
DIR=/tmp/rh56f1_record; mkdir -p "$DIR"
case "$CMD" in
start)
  for k in arm hand; do
    if [ -f "$DIR/$k.pid" ] && kill -0 "$(cat "$DIR/$k.pid")" 2>/dev/null; then echo "$k 기록 중이다 — 먼저 stop"; exit 1; fi
  done
  SIM2REAL="$(cd "$(dirname "$0")/../../.." && pwd)"
  OUT="$SIM2REAL/logs/bags/$(date +%Y%m%d_%H%M%S)_$NAME"; mkdir -p "$(dirname "$OUT")"
  HAND=""
  for s in right left; do for t in angle_actual force_actual current_actual touch_data joint_states tip_forces joint_forces ecat_status; do
    HAND="$HAND /hand_$s/$t"; done; done
  ENV="source /opt/ros/humble/setup.bash; source \$HOME/rl_ws/robot_control/ros_ws/install/setup.bash; export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-126}"
  setsid bash -c "echo \$\$ > '$DIR/arm.pid'; $ENV; exec ros2 bag record -o '$OUT/arm' /joint_states $EXTRA" </dev/null >"$DIR/arm.log" 2>&1 &
  setsid bash -c "echo \$\$ > '$DIR/hand.pid'; $ENV; exec ros2 bag record -o '$OUT/hand' $HAND" </dev/null >"$DIR/hand.log" 2>&1 &
  sleep 2
  echo "기록 시작 → $OUT  (arm pid $(cat "$DIR/arm.pid") · hand pid $(cat "$DIR/hand.pid"))"
  echo "$OUT" > "$DIR/last_out"
  ;;
stop)
  for k in arm hand; do
    [ -f "$DIR/$k.pid" ] || { echo "$k: 기록 중 아님"; continue; }
    PID=$(cat "$DIR/$k.pid")
    if kill -0 "$PID" 2>/dev/null && tr '\0' ' ' <"/proc/$PID/cmdline" | grep -q "bag record"; then
      kill -INT "$PID"
      for _ in $(seq 1 40); do kill -0 "$PID" 2>/dev/null || break; sleep 0.25; done
      kill -0 "$PID" 2>/dev/null && echo "$k pid $PID 가 10 s 넘게 살아 있다 — 확인 필요" >&2 || echo "$k 기록 끝"
    else
      echo "$k: pid $PID 는 기록기가 아니거나 끝났다"
    fi
    rm -f "$DIR/$k.pid"
  done
  [ -f "$DIR/last_out" ] && { source /opt/ros/humble/setup.bash; for k in arm hand; do ros2 bag info "$(cat "$DIR/last_out")/$k" 2>/dev/null | grep -E "Duration|Messages"; done; }
  ;;
*) echo "start|stop" >&2; exit 2;;
esac

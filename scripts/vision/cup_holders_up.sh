#!/bin/bash
# 컵홀더 자세 노드 기동(멱등) — 카메라(camera_up.sh)가 떠 있어야 한다. 머리는 head_home 뒤. 로봇은 움직이지 않는다.
# usage: cup_holders_up.sh [노드 인자…]   예: cup_holders_up.sh --write
# 내릴 때는 cup_holders_down.sh — 이 스크립트가 남긴 PID 파일의 프로세스만 끈다(패턴 kill 안 함).
source "$(dirname "$0")/common.sh"
PIDFILE="$LOGDIR/cup_holders.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "cup holders already up (pid $(cat "$PIDFILE"))"; exit 0; fi
setsid python3 "$SIM2REAL/scripts/nodes/cup_holder_pose_node.py" "$@" </dev/null >"$LOGDIR/cup_holders.log" 2>&1 &
echo $! >"$PIDFILE"
# 첫 장은 전체 무늬 탐색이라 ~6 s 걸린다
for _ in $(seq 1 30); do
  if timeout 3 ros2 topic echo --once /cup_holders/status std_msgs/msg/String >/dev/null 2>&1; then
    echo "cup holders up (pid $(cat "$PIDFILE"))"; exit 0; fi
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -20 "$LOGDIR/cup_holders.log" >&2; rm -f "$PIDFILE"; exit 1; }
  sleep 1
done
echo "status 가 60 s 안에 안 나왔다 (see $LOGDIR/cup_holders.log)" >&2; exit 1

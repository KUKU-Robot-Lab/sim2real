#!/bin/bash
# cup_holders_up.sh 가 띄운 노드만 끈다 — PID 파일의 프로세스가 정말 그 노드인지 cmdline 으로 확인한 뒤 SIGTERM.
source "$(dirname "$0")/common.sh"
PIDFILE="$LOGDIR/cup_holders.pid"
[ -f "$PIDFILE" ] || { echo "cup holders not running (no pid file)"; exit 0; }
PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null && tr '\0' ' ' <"/proc/$PID/cmdline" | grep -q "cup_holder_pose_node.py"; then
  kill -TERM "$PID"
  for _ in $(seq 1 10); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$PID" 2>/dev/null && { echo "pid $PID 가 SIGTERM 뒤 5 s 넘게 살아 있다 — 확인 필요" >&2; exit 1; }
  echo "cup holders down (pid $PID)"
else
  echo "pid $PID 는 cup_holder_pose_node 가 아니거나 이미 끝났다 — PID 파일만 지운다"
fi
rm -f "$PIDFILE"

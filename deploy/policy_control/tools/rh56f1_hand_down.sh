#!/bin/bash
# rh56f1_hand_up.sh 가 띄운 한 손만 내린다 — PID 파일의 프로세스가 정말 그 노드인지 cmdline 으로 확인한 뒤 SIGINT
# (노드가 마스터를 hold 50 주기 → INIT 으로 끝낸다). 패턴 kill 을 쓰지 않는다.
SIDE=${1:?right|left}
PIDF=/tmp/rh56f1_hand/$SIDE.pid
[ -f "$PIDF" ] || { echo "$SIDE: PID 파일 없음 — 떠 있지 않다"; exit 0; }
PID=$(cat "$PIDF")
CMD=$(tr '\0' ' ' <"/proc/$PID/cmdline" 2>/dev/null || true)
# EtherCAT = rh56f1_ecat_node.py --side <s> · RS485 = ros2 launch rh56f1_driver rh56f1_<s>_driver.launch.py
if kill -0 "$PID" 2>/dev/null && { [[ "$CMD" == *"rh56f1_ecat_node.py --side $SIDE"* ]] || [[ "$CMD" == *"rh56f1_${SIDE}_driver.launch.py"* ]]; }; then
  kill -INT "$PID"
  for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 0.25; done
  if kill -0 "$PID" 2>/dev/null; then echo "$SIDE pid $PID 가 5 s 넘게 살아 있다 — 확인 필요" >&2; exit 1; fi
  echo "$SIDE 내림 (pid $PID)"
else
  echo "$SIDE: pid $PID 는 이 손의 드라이버가 아니거나 이미 끝났다 — PID 파일만 지운다"
fi
rm -f "$PIDF"

#!/bin/bash
# FP++ 자세 → 로봇 PC UDP 송신기 기동(멱등). usage: pose_tx_up.sh <로봇 PC 주소> [port]
# 인지 런처가 ssh 접속 주소($SSH_CLIENT)를 넘긴다 — 주소를 이 저장소에 적어 두지 않는다.
source "$(dirname "$0")/common.sh"
DEST=${1:?dest address}; PORT=${2:-51126}
if pgrep -f "scripts/nodes/fpp_pose_tx.py" >/dev/null; then
  pgrep -af "scripts/nodes/fpp_pose_tx.py" | grep -q -- "--dest $DEST --port $PORT" && { echo "pose tx already up"; exit 0; }
  bash "$(dirname "$0")/pose_tx_down.sh" >/dev/null                         # 받는 곳이 바뀌었다 — 다시 띄운다
fi
setsid python3 "$SIM2REAL/scripts/nodes/fpp_pose_tx.py" --dest "$DEST" --port "$PORT" \
  </dev/null >"$LOGDIR/pose_tx.log" 2>&1 &
sleep 2; pgrep -f "scripts/nodes/fpp_pose_tx.py" >/dev/null && echo "pose tx up → $DEST:$PORT" \
  || { cat "$LOGDIR/pose_tx.log" >&2; exit 1; }

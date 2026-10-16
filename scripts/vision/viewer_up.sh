#!/bin/bash
# ROS 영상 구독 + MJPEG 8080 (모니터 없이). usage: viewer_up.sh <name>...
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { echo "need object names" >&2; exit 1; }
pgrep -f '^python3 cup_view_stream.py' >/dev/null && { echo "viewer already up"; exit 0; }
setsid bash -c 'cd "$SIM2REAL/scripts/vision" && exec python3 cup_view_stream.py --compressed --port 8080 --objects "$@"' viewer "$@" \
  </dev/null >"$LOGDIR/viewer.log" 2>&1 &
sleep 2; pgrep -f cup_view_stream.py >/dev/null && echo "viewer up" || { cat "$LOGDIR/viewer.log" >&2; exit 1; }

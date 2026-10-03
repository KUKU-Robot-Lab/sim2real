#!/bin/bash
# RH56F1 손 드라이버 한 손 기동(배경) — 미션 밖에서 실기 점검할 때. PID 파일에는 실제 노드 PID 가 들어간다.
#   bash rh56f1_hand_up.sh right --no-op            # SAFE_OP · 상태만(손 무동작)
#   bash rh56f1_hand_up.sh right                    # OP(첫 명령 전에는 제자리)
#   bash rh56f1_hand_up.sh right --op-enable        # OP 실험 손잡이 · --sync-type N 도 같다
#   bash rh56f1_hand_up.sh right --transport rs485  # 비상용 옛 배선(USB-RS485 를 다시 꽂은 뒤)
# 내릴 때: bash rh56f1_hand_down.sh right   (로그 /tmp/rh56f1_hand/<side>.log)
set -eo pipefail
SIDE=${1:?right|left}; shift
case "$SIDE" in right|left) ;; *) echo "side 는 right|left" >&2; exit 2;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR=/tmp/rh56f1_hand; mkdir -p "$DIR"
PIDF="$DIR/$SIDE.pid"; LOG="$DIR/$SIDE.log"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "$SIDE 이미 떠 있다 (pid $(cat "$PIDF")) — 먼저 rh56f1_hand_down.sh $SIDE"; exit 1; fi
# ★bash 가 자기 PID 를 쓰고 exec 로 드라이버 → 노드가 같은 PID 를 이어받는다(setsid 의 fork PID 를 쓰면 엉뚱한 것을 죽인다, 10.02)
setsid bash -c "echo \$\$ > '$PIDF'; source /opt/ros/humble/setup.bash; source \$HOME/rl_ws/robot_control/ros_ws/install/setup.bash; \
  export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-126}; exec python3 '$HERE/rh56f1_driver.py' --side $SIDE $*" </dev/null >"$LOG" 2>&1 &
for _ in $(seq 1 20); do
  sleep 0.5
  if grep -q "노드 연결\|angle_actual\|✗" "$LOG" 2>/dev/null; then break; fi
done
echo "$SIDE pid $(cat "$PIDF" 2>/dev/null) · 로그 $LOG"
grep "master\]\|ERROR\|✗" "$LOG" | sed 's/^.*\]: //' | tail -5 || true

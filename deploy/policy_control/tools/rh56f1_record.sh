#!/bin/bash
# RH56F1 실기 기록 — 팔 · 손을 **따로** bag 에 쓴다(10.03 실측: sqlite3 한 프로세스에 3750 msg/s 를 몰면 /joint_states 750 → 706 Hz
# 로 6 % 빠졌다. 둘로 나누면 /joint_states 750.1 Hz · 손 14 토픽 250.1 Hz 손실 0). mcap 저장기가 없어서 sqlite3 기준.
#
#   bash rh56f1_record.sh start <이름>     # ~/rl_ws/sim2real/logs/bags/<날짜_시각>_<이름>/{arm,hand}
#   bash rh56f1_record.sh stop  <이름>     # 그 이름의 두 기록기에 SIGINT(정상 마무리) — PID 파일의 프로세스만
#   bash rh56f1_record.sh stop             # 이름 없이 = 남은 기록 전부(shutdown)
#   EXTRA="/policy_control/joint_target /status/pd_right" bash rh56f1_record.sh start x   # 팔 쪽 bag 에 더할 토픽
#   HAND_SIDES="right" bash rh56f1_record.sh start aglt_right                             # 손 bag 에 그 손만(기본 양손)
#   CAMERA="/camera/camera/color/image_raw/compressed …" CAMERA_HZ=10 bash rh56f1_record.sh start x   # 세 번째 bag camera
#
# ★11.04 사용자: 머리 D435i 영상을 에피소드 bag 에 같이 — FP++ 자세 · 손바닥 · 정책 상태와 시간을 맞춰 겹쳐 본다
#   (scripts/analysis/episode_video_overlay.py). 영상은 따로 묶음 camera 에 쓴다(arm 에 섞으면 큰 메시지가 /joint_states 기록을
#   늦춘다 — 10.03 의 6 % 손실과 같은 이유). ros2 bag record 는 빈도를 못 낮춰 scripts/nodes/camera_bag_recorder.py 가 쓴다.
#
# ★10.09 실기: 양팔 정책을 동시에 돌리자 왼팔 기록 시작이 오른팔 기록을 '남은 기록'으로 끄고, 오른팔 단계의 기록 끝이 왼팔 기록을
#   껐다(PID 파일이 하나였다). 이제 이름마다 PID 파일이 따로다 — start 는 **같은 이름**의 남은 기록만, stop 은 그 이름만 끝낸다.
#
# 도장: 손 토픽 header = 하드웨어 샘플 시각(EtherCAT 마스터가 PDO 를 받은 순간), 팔 /joint_states = ros2_control 갱신 시각.
# 크기: 손 약 0.7 MB/s(양손) · 팔 약 0.7 MB/s(10.03 측정 기준).
# 크기: camera 10 Hz 약 4.2 MB/s(JPEG q95 1280x720 약 276 KB + 정렬 깊이 PNG 약 146 KB, 11.04 측정) = 분당 약 250 MB.
# 시험용 덮어쓰기: RH56F1_RECORD_DIR(PID) · RH56F1_RECORD_ENV(ROS 환경 명령) · RH56F1_RECORD_OUT(bag 위치) · RH56F1_RECORD_NO_INFO ·
#   RH56F1_CAMERA_CMD(카메라 기록기 명령).
set -eo pipefail
CMD=${1:?start|stop}; NAME=${2:-}
SIM2REAL="$(cd "$(dirname "$0")/../../.." && pwd)"
ROOT=${RH56F1_RECORD_DIR:-$SIM2REAL/logs/rh56f1_record}; mkdir -p "$ROOT"

stop_one() {   # $1 = 이름 디렉터리
  local d=$1 k PID
  for k in arm hand camera; do
    [ -f "$d/$k.pid" ] || continue
    PID=$(cat "$d/$k.pid")
    if kill -0 "$PID" 2>/dev/null && tr '\0' ' ' <"/proc/$PID/cmdline" | grep -qE "bag record|camera_bag_recorder"; then
      kill -INT "$PID"
      for _ in $(seq 1 40); do kill -0 "$PID" 2>/dev/null || break; sleep 0.25; done
      kill -0 "$PID" 2>/dev/null && echo "$(basename "$d") $k pid $PID 가 10 s 넘게 살아 있다 — 확인 필요" >&2 \
        || echo "$(basename "$d") $k 기록 끝"
    else
      echo "$(basename "$d") $k: pid $PID 는 기록기가 아니거나 끝났다"
    fi
    rm -f "$d/$k.pid"
  done
  # 요약은 보기용 — bag 이 비었거나 없어도 정지는 성공(shutdown 이 이 줄에서 멈추지 않게, 10.08 리뷰)
  if [ -z "${RH56F1_RECORD_NO_INFO:-}" ] && [ -f "$d/last_out" ]; then
    source /opt/ros/humble/setup.bash
    for k in arm hand camera; do ros2 bag info "$(cat "$d/last_out")/$k" 2>/dev/null | grep -E "Duration|Messages" || true; done
  fi
  rm -f "$d/last_out"   # 폴더가 재부팅 뒤에도 남는다 — 이름 없는 stop 이 끝난 기록을 다시 요약하지 않게
}

case "$CMD" in
start)
  [ -n "$NAME" ] || { echo "start 에는 이름이 필요하다(팔마다 따로 기록)" >&2; exit 2; }
  D="$ROOT/$NAME"; mkdir -p "$D"
  # ★10.08 리뷰: 앞 단계가 도중에 실패하면 기록기가 남는다 — 같은 이름을 다시 돌릴 때 막히지 않게 그 기록만 먼저 마무리한다
  for k in arm hand camera; do
    if [ -f "$D/$k.pid" ] && kill -0 "$(cat "$D/$k.pid")" 2>/dev/null; then
      echo "$NAME $k 기록이 남아 있다 — 마무리하고 새로 시작"; stop_one "$D"; break
    fi
  done
  OUT_ROOT=${RH56F1_RECORD_OUT:-$SIM2REAL/logs/bags}
  OUT="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)_$NAME"; mkdir -p "$OUT_ROOT"
  HAND=""
  # 10.08 명령(위치 angle_set · 어드민턴스 angle_target)과 어드민턴스가 연 칸 수 · 손가락 모드도 — 어드민턴스 정책 쥠 분석
  # 11.04 손끝 영점(tip_bias · tip_zero_status 는 transient_local — 기록 시작 전에 잡은 마지막 bias 도 bag 첫머리에 들어온다)
  for s in ${HAND_SIDES:-right left}; do for t in angle_actual force_actual current_actual touch_data joint_states tip_forces \
      tip_forces_zeroed tip_bias tip_zero_status joint_forces ecat_status angle_set angle_target admittance_offset finger_mode; do
    HAND="$HAND /hand_$s/$t"; done; done
  source "$(dirname "$0")/../../../config/ros_domain.env"   # 실기 도메인 한 곳(10.30 motion_acq 와 분리)
  ENV=${RH56F1_RECORD_ENV:-"source /opt/ros/humble/setup.bash; source \$HOME/rl_ws/robot_control/ros_ws/install/setup.bash; export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-$S2R_REAL_DOMAIN}"}
  setsid bash -c "echo \$\$ > '$D/arm.pid'; $ENV; exec ros2 bag record -o '$OUT/arm' /joint_states $EXTRA" </dev/null >"$D/arm.log" 2>&1 &
  setsid bash -c "echo \$\$ > '$D/hand.pid'; $ENV; exec ros2 bag record -o '$OUT/hand' $HAND" </dev/null >"$D/hand.log" 2>&1 &
  if [ -n "${CAMERA:-}" ]; then
    # nice 10 — 정책 · FP++ 보다 뒤에 선다(늦으면 프레임만 버린다, best effort 구독)
    CAM_CMD=${RH56F1_CAMERA_CMD:-"python3 $SIM2REAL/scripts/nodes/camera_bag_recorder.py"}
    setsid bash -c "echo \$\$ > '$D/camera.pid'; $ENV; exec nice -n 10 $CAM_CMD -o '$OUT/camera' --hz ${CAMERA_HZ:-10} $CAMERA" \
      </dev/null >"$D/camera.log" 2>&1 &
  fi
  for _ in $(seq 1 20); do [ -f "$D/arm.pid" ] && [ -f "$D/hand.pid" ] && { [ -z "${CAMERA:-}" ] || [ -f "$D/camera.pid" ]; } && break
    sleep 0.1; done
  sleep 1
  echo "기록 시작 → $OUT  (arm pid $(cat "$D/arm.pid") · hand pid $(cat "$D/hand.pid")${CAMERA:+ · camera pid $(cat "$D/camera.pid" 2>/dev/null)})"
  echo "$OUT" > "$D/last_out"
  ;;
stop)
  if [ -n "$NAME" ]; then
    if [ -d "$ROOT/$NAME" ]; then stop_one "$ROOT/$NAME"; else echo "$NAME: 기록 중 아님"; fi
  else
    for d in "$ROOT"/*/; do [ -d "$d" ] && stop_one "${d%/}"; done
    # 10.09 전(이름 없는 PID 파일)이 남아 있으면 그것도
    [ -f "$ROOT/arm.pid" ] || [ -f "$ROOT/hand.pid" ] && stop_one "$ROOT"
  fi
  exit 0
  ;;
*) echo "start|stop" >&2; exit 2;;
esac

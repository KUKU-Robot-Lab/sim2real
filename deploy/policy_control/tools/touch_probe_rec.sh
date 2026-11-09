#!/usr/bin/env bash
# touch_probe.sh 를 녹화(arm · hand · camera)와 같이 돌린다 — 11.05 사용자: "녹화 시작과 끝을 같이 맞춰야".
#   --execute 일 때만 녹화한다. 시작 전에 rh56f1_record.sh start, 끝나면(정상 · q/x · Ctrl-C · 터미널 닫힘 · SIGTERM) stop.
#   기록 이름 = touch_<--out 파일 이름>(예: touch_right_5) → logs/runs/<YYMMDD>/bags/<시각>_touch_right_5/{arm,hand,camera}
#   카메라 bag 에 /policy_control/joint_target 을 같이 실어 touch_probe 기록(frame_id 틱)과 시각을 잇는다.
#   deploy/policy_control/tools/touch_probe_rec.sh --plan $P/plan_right.npz --out $P/rec/right_5.npz --step --execute
# 녹화 환경에 랜 DDS 프로파일을 넣는다 — rh56f1_record.sh 기본 환경에는 없어서 11.05 녹화가 토픽을 하나도 못 받았다.
set -o pipefail
cd "$(dirname "$0")/../../.."
SIM2REAL=$PWD
# 인자는 touch_probe.py 와 같이 읽는다(그쪽은 allow_abbrev=False — 줄임 인자를 받지 않는다). --x=값 꼴도.
out="" plan="" side="" execute=0
prev=""
for a in "$@"; do
  case "$prev" in --out) out=$a ;; --plan) plan=$a ;; --side) side=$a ;; esac
  case "$a" in --out=*) out=${a#*=} ;; --plan=*) plan=${a#*=} ;; --side=*) side=${a#*=} ;; --execute) execute=1 ;; esac
  prev=$a
done
[ "$execute" = 1 ] || exec bash deploy/policy_control/tools/touch_probe.sh "$@"
[ -n "$out" ] || { echo "--out 이 없다" >&2; exit 2; }
[[ "$(basename "$out")" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--out 파일 이름은 [A-Za-z0-9._-] 만: $out" >&2; exit 2; }
# 손 녹화 쪽 = --side, 없으면 plan 파일 이름(touch_probe.py 는 그때 plan meta 의 side 를 쓴다)
if [ -z "$side" ]; then side=right; case "$(basename "$plan")" in *left*) side=left ;; esac; fi
case "$side" in right|left) ;; *) echo "--side 는 right|left: $side" >&2; exit 2 ;; esac
name="touch_$(basename "${out%.npz}")"
export RH56F1_RECORD_ENV="${RH56F1_RECORD_ENV:-source /opt/ros/humble/setup.bash; source \$HOME/rl_ws/robot_control/ros_ws/install/setup.bash; source $SIM2REAL/config/ros_domain.env; export ROS_DOMAIN_ID=\$S2R_REAL_DOMAIN ROS_LOCALHOST_ONLY=0; source $SIM2REAL/config/dds/lan_profile.sh}"
export CAMERA="${CAMERA:-/camera/camera/color/image_raw/compressed /camera/camera/aligned_depth_to_color/image_raw/compressedDepth /camera/camera/color/camera_info /policy_control/joint_target}"
export CAMERA_HZ="${CAMERA_HZ:-10}" HAND_SIDES="${HAND_SIDES:-$side}"
REC=deploy/policy_control/tools/rh56f1_record.sh
recdir="${RH56F1_RECORD_DIR:-$SIM2REAL/logs/runs/$(date +%y%m%d)/rh56f1_record}/$name"   # rh56f1_record.sh 의 PID 파일 자리
stopped=0
stop_rec() {
  [ "$stopped" = 1 ] && return
  stopped=1
  echo "녹화 끄는 중($name)…"
  local k p pids=""
  for k in arm hand camera; do [ -f "$recdir/$k.pid" ] && pids="$pids $k:$(cat "$recdir/$k.pid")"; done   # stop 이 PID 파일을 지운다 — 먼저 읽는다
  bash "$REC" stop "$name" || echo "★녹화 stop 실패(rc $?)" >&2
  for k in arm hand camera; do [ -f "$recdir/$k.pid" ] && echo "★녹화 PID 파일이 남았다: $recdir/$k.pid" >&2; done
  for p in $pids; do
    kill -0 "${p#*:}" 2>/dev/null && tr '\0' ' ' <"/proc/${p#*:}/cmdline" 2>/dev/null | grep -qE "bag record|camera_bag_recorder" \
      && echo "★녹화 ${p%%:*} pid ${p#*:} 가 아직 살아 있다 — 확인 필요(bash $REC stop $name)" >&2
  done
  return 0
}
# 시작 전에 건다 — start 가 기록기 일부만 띄우고 실패해도 끈다
trap stop_rec EXIT
trap 'exit 129' HUP
trap 'exit 143' TERM
bash "$REC" start "$name" || { echo "녹화 시작 실패 — 보정을 시작하지 않는다" >&2; exit 3; }
# 녹화기가 토픽을 잡을 때까지(카메라 · joint_target 구독 줄) 최대 10 s 기다린다 — 못 잡으면 알리고 그대로 간다
log="$recdir/camera.log"
for _ in $(seq 1 20); do grep -q "기록 시작" "$log" 2>/dev/null && break; sleep 0.5; done
grep -q "구독 /camera" "$log" 2>/dev/null || echo "★카메라 토픽을 못 잡았다 — 영상 없이 간다(camera_up.sh 확인)" >&2
bash deploy/policy_control/tools/touch_probe.sh "$@"

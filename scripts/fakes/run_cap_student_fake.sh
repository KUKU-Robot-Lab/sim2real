#!/usr/bin/env bash
# cap 깊이 학생 노드(rh_cap_student_node, 오른팔) fake 리허설 — 도메인 97 고정(11.18).
# fake plant(팔 /joint_states · 손 · 촉각 · 캡) + fake 머리 joint_states + fake 깊이 + fake 쉐이커 live_pose + 학생 노드(cpu)
# → cap_student_fake_drive.py 가 체인 한 번과 거부 경로 셋을 돌린다. pd 는 띄우지 않는다(plant 가 joint_target 을 바로 따른다).
# 띄운 PID 는 $LOG/pids.txt, 정리는 그 PID 에만 SIGTERM(SIGKILL 없음 — 안 죽으면 남은 PID 를 출력).
cd "$(dirname "$0")/../.."
source /opt/ros/humble/setup.bash   # set -u 전에(ament 설정이 빈 변수를 쓴다)
set -u -o pipefail   # 아래 drive | tee 의 종료코드 = drive 것(quick 거부 → 1)
export ROS_DOMAIN_ID=97 CUDA_VISIBLE_DEVICES=
REAL=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'deploy/policy_control'); from policy_control.ros_domain import REAL_DOMAIN; print(REAL_DOMAIN)")
[ "$ROS_DOMAIN_ID" = "$REAL" ] || [ "$ROS_DOMAIN_ID" = 50 ] && { echo "실기 도메인 거부"; exit 2; }
LOG=logs/runs/$(date +%y%m%d)/cap_fake/$(date +%H%M%S)
mkdir -p "$LOG"
SDIR=deploy/policies/rh56f1/cap/right_student_h01
PIDS=()
bg() {  # bg <이름> <명령…> — nice 15 로 띄우고 PID 를 적는다(nice 는 exec 하므로 $! = 실제 프로세스)
  local n=$1; shift
  nice -n 15 "$@" > "$LOG/$n.log" 2>&1 &
  PIDS+=($!); echo "$! $n" >> "$LOG/pids.txt"
}
cleanup() {
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -TERM "$p"; done
  for _ in $(seq 50); do alive=(); for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done; [ ${#alive[@]} = 0 ] && break; sleep 0.2; done
  echo "남은 PID: ${alive[*]:-없음}" | tee -a "$LOG/pids.txt"
}
trap cleanup EXIT
ps -eo pid,etimes,cmd | grep -E "s2r_console|pd_node|rh_.*_node|launch" | grep -v -E "grep|chrome|orca" > "$LOG/ps_before.txt"

ARMQ=$(.venv/bin/python -c "print(','.join(map(str, (-1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646))))")  # = 노드 LIVE_ARM_START_Q
bg plant python3 scripts/fakes/fake_rh_cap_plant.py --contract $SDIR/rh_cap_contract.json --arm-q=$ARMQ
# fake_head_driver.py 는 HeadDriver 3 인자 publish_angles 전 것이라 첫 tick 에 죽는다 → 연속 목표까지 받는 새 fake
bg head python3 scripts/fakes/fake_head_stream_driver.py --pan ${HEAD_PAN:-0} --tilt ${HEAD_TILT:-40}
# NO_DEPTH=1: 깊이 발행기 없이(→ reset 이 config 없음으로 거부해야 한다) · QUICK=1: reset → start → search 만
DEPTH_PID=0
[ "${NO_DEPTH:-0}" = 1 ] || { bg depth python3 scripts/fakes/fake_head_depth_pub.py --depth ${DEPTH_M:-0.6}; DEPTH_PID=${PIDS[-1]}; }
# 쉐이커 메쉬 원점 = 리시버 고정 자리 + 0.02 m(rcv_from_shaker 가 −0.02 내린다)
bg shaker python3 scripts/fakes/fake_cup_pose_pub.py --topic /objects/shaker_c_orange/live_pose --x 0.2519 --y 0.1241 --z 0.4284
bg node .venv-student/bin/python deploy/policy_control/policy_control/rh_cap_student_node.py --ros-args \
  -r __node:=rh_cap_node_right -p ns:=right_cap -p episode_topic:=/policy_control/right/episode \
  -p contract:=$SDIR/rh_cap_contract.json -p robot:=deploy/policy_control/config/robots/rh56f1_right_fake.yaml \
  -p device:=cpu -p student_dir:=$SDIR -p rcv_pose_topic:=/objects/shaker_c_orange/live_pose -p rcv_allow_fixed:=true \
  -p descend_xy_max_m:=${DESCEND_XY_MAX_M:-0.30} -p descend_len_max_m:=${DESCEND_LEN_MAX_M:-0.50}
# 위 하강 시작 검사 완화는 fake 전용: 깊이가 평평한 판이라 학생이 캡을 못 찾아 접근 끝이 캡 위가 아니다(11.18 181044: 43 mm > 40 mm 거부).
#   실기 기본값(0.04 · 0.12)은 노드 그대로. 하강 · 이송 · 끼우기 명령 경로를 끝까지 보려고 푼다.
nice -n 15 timeout 480 .venv-student/bin/python scripts/fakes/cap_student_fake_drive.py --out "$LOG" --depth-pid "$DEPTH_PID" ${QUICK:+--quick} \
  2>&1 | tee "$LOG/drive.log"
rc=$?
echo "로그: $LOG"
exit $rc

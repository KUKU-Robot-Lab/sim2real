#!/usr/bin/env bash
# joint family 배관 한 판 — fake 플랜트 + joint_node(한 팔), 실기 없음. 무엇을 증명하는지 먼저 적는다.
#
#   증명한다 : 계약이 로드되고 · 팔 · 손 joint_state 와 컵 포즈가 obs(133)로 조립되고 · LSTM 정책이 60 Hz 로 돌고 ·
#              디코더 목표가 pd 로 가고 · episode reset/start/stop 이 서고 · seq 가 안 빠진다.
#   증명 못 한다: 파지 · 들기 성공. MockArm 에 접촉도 컵 물리도 없다. 관측이 학습과 같은지는 trace 대조가 한다.
#
#   usage: ROS_DOMAIN_ID=97 deploy/policy_control/tools/joint_fake_run.sh [seconds] [logdir]
#   env:   RUN_DIR(기본 deploy/policies/dg5f_m/cup_pick/right_m15) · PD_CONTRACT(기본 logs/policy/asset_right_m15/deploy_contract.json)
#          ROBOT(기본 dg5f_m_right_fake) · SIDE(기본 right) · DEVICE(기본 cpu) · CUP("x y z", 기본 학습 스폰 중심)
#
#   ROS_DOMAIN_ID 는 실기(126)도 0/unset 도 아니어야 한다 — launch 가 0/unset 을 거부하고, 이 스크립트가 126 을 거부한다.
set -o pipefail
cd "$(dirname "$0")/../../.."   # deploy/policy_control/tools → 저장소 루트
source /opt/ros/humble/setup.bash && . .venv/bin/activate
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-97}"
[ "$ROS_DOMAIN_ID" = "126" ] && { echo "[joint_fake] 실기 도메인 126 에서는 돌리지 않는다"; exit 3; }

RUN_DIR="${RUN_DIR:-deploy/policies/dg5f_m/cup_pick/right_m15}"
SEC="${1:-20}"; LOG="${2:-logs/policy_control/joint_fake_$(date +%m%d_%H%M%S)}"; mkdir -p "$LOG"
CONTRACT="$RUN_DIR/joint_contract.json"
PD_CONTRACT="${PD_CONTRACT:-logs/policy/asset_right_m15/deploy_contract.json}"
ROBOT="${ROBOT:-dg5f_m_right_fake}"
SIDE="${SIDE:-right}"
# 학습 스폰 중심(object_spawn_center_override 0.25, −0.15) · 컵 원점 높이 0.2757(학습 trace 에피소드 시작 실측,
# 실기 FP++ 0.279 와 3.5 mm 차)
read -r CX CY CZ <<< "${CUP:-0.25 -0.15 0.2757}"
for f in "$CONTRACT" "$PD_CONTRACT"; do
  [ -f "$f" ] || { echo "[joint_fake] $f 가 없다 — build_deploy_contract.py 를 먼저"; exit 2; }
done
echo "[joint_fake] domain $ROS_DOMAIN_ID · run $RUN_DIR · $SIDE · cup($CX,$CY,$CZ) · log $LOG"

# ★자식이 자기 PID 를 직접 적는다(pour_fake_run.sh 와 같은 이유 — setsid 가 포크하면 $! 가 사라지는 부모다).
PIDFILE="$LOG/pids"; : > "$PIDFILE"
cleanup() {
  [ -s "$PIDFILE" ] || return 0
  while read -r g; do [ -n "$g" ] && kill -- -"$g" 2>/dev/null; done < "$PIDFILE"; sleep 1
  while read -r g; do [ -n "$g" ] && kill -9 -- -"$g" 2>/dev/null; done < "$PIDFILE"
}
trap cleanup EXIT
bg() { setsid bash -c 'echo $$ >> "$1"; shift; exec "$@"' _ "$PIDFILE" "$@" & }
call() { timeout 60 ros2 service call "/policy_control/episode/$1" std_srvs/srv/Trigger "{}" 2>&1 | tr -d '\n'; }
pd() { timeout 120 ros2 service call "/policy_control/pd_$SIDE/$1" std_srvs/srv/Trigger "{}" 2>&1 | tr -d '\n'; }

# 컵 포즈는 플랜트가 낸다(cup_x/y/z) — 따로 퍼블리셔를 띄우면 같은 토픽에 둘이 되어 플랜트 기본값(0.38, 0.19)이 섞인다(09.28).
bg ros2 launch deploy/policy_control/launch/fake_plant.launch.py side:="$SIDE" robot:="$ROBOT" \
    contract:="$PD_CONTRACT" plant_model:=rate hand_follow:=jtc cup_x:="$CX" cup_y:="$CY" cup_z:="$CZ" \
    > "$LOG/fake_plant.log" 2>&1
sleep 5

bg ros2 launch deploy/policy_control/launch/pd_controller.launch.py contract:="$PD_CONTRACT" robot:="$ROBOT" \
    pd_config:="${PD_CONFIG:-dg5f_m_short_fake}" sides:="$SIDE" execute:=true fake:=true use_source:=true \
    > "$LOG/pd.log" 2>&1
sleep 8
echo "[joint_fake] pd engage   : $(pd engage)"    | tee "$LOG/pd_stage.log"
echo "[joint_fake] pd goto_home: $(pd goto_home)" | tee -a "$LOG/pd_stage.log"
grep -q "goto_home.*success=True" "$LOG/pd_stage.log" || { echo "[joint_fake] goto_home 실패 — 중단"; exit 1; }

bg ros2 launch deploy/policy_control/launch/joint_chain.launch.py contract:="$CONTRACT" robot:="$ROBOT" \
    device:="${DEVICE:-cpu}" fake:=true use_source:=true > "$LOG/joint_chain.log" 2>&1
sleep 12

bg python deploy/policy_control/tools/status_to_csv.py --seconds "$SEC" --policy-dt 0.0166667 \
    --nodes joint_node --out "$LOG/status.csv" --jsonl "$LOG/status.jsonl" > "$LOG/status_summary.txt" 2>&1

RC=0
snap() {   # 팔 7 관절 한 줄 — pd 가 정책 목표를 실제로 따랐는지(팔이 움직였는지) 본다
  timeout 10 ros2 topic echo --once /joint_states sensor_msgs/msg/JointState 2>/dev/null | python -c "
import sys, yaml
d = yaml.safe_load('\n'.join(l for l in sys.stdin.read().split('---')[0].splitlines() if not l.startswith(chr(9))))  # QoS 이벤트 줄(탭) 제외
m = dict(zip(d['name'], d['position']))
print(' '.join(f'{m[n]:.4f}' for n in [f'openarm_${SIDE}_joint{i}' for i in range(1, 8)] if n in m) or ' '.join(f'{m[n]:.4f}' for n in [f'${SIDE:0:1}_aj_{i}' for i in range(1, 8)]))"
}
echo "[joint_fake] reset: $(call reset)" | tee "$LOG/episode.log"
echo "[joint_fake] start: $(call start)" | tee -a "$LOG/episode.log"
grep -q "start.*success=True" "$LOG/episode.log" || RC=1
Q0="$(snap)"
sleep "$SEC"
Q1="$(snap)"
python -c "
import sys; a, b = [list(map(float, x.split())) for x in sys.argv[1:3]]
print(f'[joint_fake] 팔 이동 {max(abs(p - q) for p, q in zip(a, b)):.3f} rad (start → stop 직전, 7 관절 중 최대)')" "$Q0" "$Q1" \
  | tee "$LOG/arm_moved.txt" || { echo "[joint_fake] 팔 관절을 못 읽었다"; RC=1; }
echo "[joint_fake] stop : $(call stop)" | tee -a "$LOG/episode.log"
echo "[joint_fake] pd release : $(pd release)" | tee -a "$LOG/pd_stage.log"

echo "[joint_fake] ---- 판정 ----" | tee "$LOG/verdict.txt"
cat "$LOG/status_summary.txt" | tee -a "$LOG/verdict.txt"
python - "$LOG" <<'PY' | tee -a "$LOG/verdict.txt"
import csv, json, pathlib, sys
log = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((log / "status.csv").open()))
bad = [r for r in rows if r.get("joint_node_ok") == "False"]
run = [json.loads(l) for l in (log / "status.jsonl").open() if '"running"' in l]
ms = sorted(float(r.get("proc_ms", 0)) for r in run if "proc_ms" in r)
p95 = ms[int(0.95 * (len(ms) - 1))] if ms else float("nan")
print(f"행 {len(rows)} · joint_node not-ok {len(bad)} · running 틱 {len(ms)} · proc p95 {p95:.1f} ms")
PY
grep -q "not-ok 0" "$LOG/verdict.txt" || RC=1
echo "[joint_fake] rc=$RC · $LOG"
exit $RC

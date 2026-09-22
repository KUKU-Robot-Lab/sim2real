#!/usr/bin/env bash
# pour 배관 한 판 — fake 플랜트 + pour_node, 실기 없음. 무엇을 증명하는지 먼저 적는다.
#
#   증명한다 : 계약이 로드되고 · 두 팔 joint_state 와 두 컵 포즈가 obs 로 조립되고 ·
#              정책이 60 Hz 로 돌고 · episode reset/start/stop 이 서고 · seq 가 안 빠진다.
#   증명 못 한다: 파지·붓기 성공. MockArm 에 커플링도 테이블 접촉도 없다(POLICY_CONTROL_STATUS:24).
#
#   usage: ROS_DOMAIN_ID=97 policy_control/tools/pour_fake_run.sh [seconds] [logdir]
#   env:   RUN(기본 pour_i18) · USE_FABRIC(기본 false — CUDA 가 비어 있을 때만 true) ·
#          PLANT_MODEL(기본 rate — 배선 검증용. pd 모델은 실측 팔 캘리브레이션
#          hdgp/log/logs/r2s_autotune/results/right_arm_best_calibration.json 을 요구하는데
#          이 PC 에 없다. 09.05 자산 정리 때 사라진 것으로 보인다) ·
#          DEVICE(기본 cpu) · FILL(기본 0.85, 계약에 기본값이 없어 반드시 넣어야 한다)
#
#   ROS_DOMAIN_ID 는 실기(126)도 0/unset 도 아니어야 한다 — launch 가 0/unset 을 거부하고,
#   이 스크립트가 126 을 거부한다.
set -o pipefail
cd "$(dirname "$0")/../.."
source /opt/ros/humble/setup.bash && . .venv/bin/activate
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-97}"
[ "$ROS_DOMAIN_ID" = "126" ] && { echo "[pour_fake] 실기 도메인 126 에서는 돌리지 않는다"; exit 3; }

RUN="${RUN:-pour_i18}"
SEC="${1:-20}"; LOG="${2:-logs/policy_control/pour_fake_$(date +%m%d_%H%M%S)}"; mkdir -p "$LOG"
CONTRACT="logs/policy/$RUN/pour_contract.json"
PD_CONTRACT="logs/policy/asset_$RUN/deploy_contract.json"
ROBOT="${ROBOT:-dg5f_m_bi_fake}"
SRC_TOPIC=/objects/pour_src_cup/pose
RCV_TOPIC=/objects/pour_rcv_cup/pose
for f in "$CONTRACT" "$PD_CONTRACT"; do
  [ -f "$f" ] || { echo "[pour_fake] $f 가 없다 — fetch_run.py + build_deploy_contract.py 를 먼저"; exit 2; }
done

# 컵 초기 위치는 그 런의 trace 첫 행에서 읽는다 — 손으로 옮겨 적지 않는다.
read -r SX SY SZ RX RY RZ < <(python - "$RUN" <<'PY'
import sys, numpy as np, pathlib
p = pathlib.Path("logs/policy") / sys.argv[1] / "trace.npz"
if p.is_file():
    z = np.load(p)
    v = list(z["src_cup_pos"][0, 0]) + list(z["rcv_cup_pos"][0, 0])
else:                       # trace 가 없으면 계약의 컵 입 높이로 대략 — 배선 확인에는 충분하다
    v = [0.365, -0.169, 0.282, 0.363, 0.168, 0.285]
print(" ".join(f"{float(x):.4f}" for x in v))
PY
)
echo "[pour_fake] domain $ROS_DOMAIN_ID · run $RUN · fabric ${USE_FABRIC:-false} · src($SX,$SY,$SZ) rcv($RX,$RY,$RZ) · log $LOG"

# ★자식이 **자기 PID 를 직접** 적는다. `setsid cmd & PIDS+=($!)` 는 틀린다 —
#   백그라운드 잡은 이미 그룹 리더라 setsid 가 포크하고, $! 는 곧 사라지는 부모다.
#   그러면 kill -- -$! 가 빈 그룹을 때리고 노드가 살아남아 **다음 판을 오염시킨다**(09.21 실측:
#   남은 fake controller_manager 가 컨트롤러를 물고 있어 engage 가 STRICT 거부됐다).
#   `bash -c` 가 setsid 아래에서 세션 리더가 되고 PID==PGID 이며, exec 로 그 PID 를 유지한다.
PIDFILE="$LOG/pids"; : > "$PIDFILE"
cleanup() {
  [ -s "$PIDFILE" ] || return 0
  while read -r g; do [ -n "$g" ] && kill -- -"$g" 2>/dev/null; done < "$PIDFILE"; sleep 1
  while read -r g; do [ -n "$g" ] && kill -9 -- -"$g" 2>/dev/null; done < "$PIDFILE"
}
trap cleanup EXIT
bg() { setsid bash -c 'echo $$ >> "$1"; shift; exec "$@"' _ "$PIDFILE" "$@" & }
call() { timeout 60 ros2 service call "/policy_control/episode/$1" std_srvs/srv/Trigger "{}" 2>&1 | tr -d '\n'; }

bg ros2 launch policy_control/launch/fake_plant.launch.py side:=both robot:="$ROBOT" \
    contract:="$PD_CONTRACT" plant_model:="${PLANT_MODEL:-rate}" > "$LOG/fake_plant.log" 2>&1
bg python scripts/fakes/fake_cup_pose_pub.py --topic "$SRC_TOPIC" --x "$SX" --y "$SY" --z "$SZ" > "$LOG/cup_src.log" 2>&1
bg python scripts/fakes/fake_cup_pose_pub.py --topic "$RCV_TOPIC" --x "$RX" --y "$RY" --z "$RZ" > "$LOG/cup_rcv.log" 2>&1
sleep 5

# fake 팔은 항상 차렷(0)에서 시작한다(fake_arm_side.py:107). 리셋 자세로 데려가는 것은 pd 의 일이다 —
# 이 순서가 RUNBOOK §3 의 실기 순서와 같다. 도메인은 fake, 플랜트도 fake 이므로 execute:=true 가 안전하다.
bg ros2 launch policy_control/launch/pd_controller.launch.py contract:="$PD_CONTRACT" robot:="$ROBOT" \
    pd_config:="${PD_CONFIG:-dg5f_m_short_fake}" sides:=right,left execute:=true fake:=true use_source:=true \
    > "$LOG/pd.log" 2>&1
sleep 8
pd() { timeout 120 ros2 service call "/policy_control/pd/$1" std_srvs/srv/Trigger "{}" 2>&1 | tr -d '\n'; }
echo "[pour_fake] pd engage   : $(pd engage)"    | tee "$LOG/pd_stage.log"
echo "[pour_fake] pd goto_home: $(pd goto_home)" | tee -a "$LOG/pd_stage.log"
grep -q "goto_home.*success=True" "$LOG/pd_stage.log" || { echo "[pour_fake] goto_home 실패 — 중단"; exit 1; }

bg ros2 launch policy_control/launch/pour_chain.launch.py contract:="$CONTRACT" robot:="$ROBOT" \
    src_cup_topic:="$SRC_TOPIC" rcv_cup_topic:="$RCV_TOPIC" device:="${DEVICE:-cpu}" \
    use_fabric:="${USE_FABRIC:-false}" fake:=true use_source:=true > "$LOG/pour_chain.log" 2>&1
bg python policy_control/policy_control/pour_guard_node.py --ros-args \
    -p src_cup_topic:="$SRC_TOPIC" -p rcv_cup_topic:="$RCV_TOPIC" > "$LOG/guard.log" 2>&1
sleep 8

bg python policy_control/tools/status_to_csv.py --seconds "$SEC" --policy-dt 0.0166667 \
    --nodes pour_node --out "$LOG/status.csv" --jsonl "$LOG/status.jsonl" > "$LOG/status_summary.txt" 2>&1

# fill_level 은 사람이 넣는 값이다(소스 컵이 지금 얼마나 차 있는가). 계약에 기본값이 없으면 start 가 거부된다.
timeout 20 ros2 topic pub --times 5 /policy_control/pour/fill_level std_msgs/msg/Float64 \
    "{data: ${FILL:-0.85}}" > "$LOG/fill.log" 2>&1

RC=0
# guard 가 살아 있는지는 status 한 줄로만 안다. run6 에서 guard 는 기동 즉시 ImportError 로 죽었는데
# 판정은 그걸 말하지 않았다 — 안전망 없이 돈 실행이 "통과"로 남으면 안 된다.
if timeout 15 ros2 topic echo --once /policy_control/status/pour_guard std_msgs/msg/String \
        > "$LOG/guard_status.log" 2>&1; then
    echo "[pour_fake] guard : status 수신" | tee "$LOG/guard_alive.txt"
else
    echo "[pour_fake] guard : status 없음 — guard 가 죽었다 ($LOG/guard.log)" | tee "$LOG/guard_alive.txt"
    RC=1
fi
echo "[pour_fake] reset: $(call reset)" | tee "$LOG/episode.log"
echo "[pour_fake] start: $(call start)" | tee -a "$LOG/episode.log"
grep -q "start.*success=True" "$LOG/episode.log" || RC=1
sleep "$SEC"
echo "[pour_fake] stop : $(call stop)" | tee -a "$LOG/episode.log"
echo "[pour_fake] pd release : $(pd release)" | tee -a "$LOG/pd_stage.log"

echo "[pour_fake] ---- 판정 ----" | tee "$LOG/verdict.txt"
cat "$LOG/status_summary.txt" | tee -a "$LOG/verdict.txt"
cat "$LOG/guard_alive.txt" >> "$LOG/verdict.txt"
python - "$LOG" <<'PY' | tee -a "$LOG/verdict.txt"
import csv, pathlib, sys
rows = list(csv.DictReader((pathlib.Path(sys.argv[1]) / "status.csv").open()))
bad = [r for r in rows if r.get("pour_node_ok") == "False"]
print(f"행 {len(rows)} · pour_node not-ok {len(bad)}")
PY
echo "[pour_fake] rc=$RC · $LOG"
exit $RC

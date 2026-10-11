#!/usr/bin/env bash
# 방안 2 재생 fake 리허설 — 한 팔: fake 플랜트(rate) + fake RH56F1 손 + 상태 노드 + pd(execute, fake)
#   → engage → goto_home → replay_traj_rh.py --execute(sim 성공 궤적, 팔 7 + 손 6) → release. 요약은 summary.txt.
#   usage: NPZ=<후보>.npz SIDE=left RATE=0.25 ROS_DOMAIN_ID=113 deploy/policy_control/tools/fake_replay_rh_run.sh [logdir]
#   확인: pd 가 HOLD 로 가지 않는다 · 손 모드 사건이 pd 에 닿아 어드민턴스 토픽(angle_target)으로 나간다 · 끝에 pd 가 내부 목표로 붙든다.
#   접촉·쥐는 힘은 fake 로 확인할 수 없다(fake 손 = 위치 추종) — 그건 sim 에서 본다.
set -o pipefail
cd "$(dirname "$0")/../../.."
source /opt/ros/humble/setup.bash && . .venv/bin/activate
[ -f "$HOME/rl_ws/robot_control/ros_ws/install/setup.bash" ] && source "$HOME/rl_ws/robot_control/ros_ws/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-113}" ROS_LOCALHOST_ONLY=1
case "$ROS_DOMAIN_ID" in 0|126) echo "✗ ROS_DOMAIN_ID $ROS_DOMAIN_ID 은 실기 도메인이다"; exit 3 ;; esac
: "${NPZ:?NPZ=<s2r_replay_v1 npz>}"
SIDE="${SIDE:-left}"; RATE="${RATE:-0.25}"
CONTRACT=logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json
ROBOT_BI=deploy/policy_control/config/robots/rh56f1_bi_fake.yaml
ROBOT=deploy/policy_control/config/robots/rh56f1_${SIDE}_fake.yaml
PD=deploy/policy_control/config/pd_rh56f1_fake.yaml
MAP=deploy/policy_control/config/rh56f1_hand_map.yaml
LOG="${1:-logs/policy_control/fake_replay_rh_${SIDE}_$(date +%m%d_%H%M%S)}"; mkdir -p "$LOG"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -- -"$p" 2>/dev/null; done; sleep 1; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 -- -"$p" 2>/dev/null; done; }
trap cleanup EXIT
launch_bg() { setsid "$@" & PIDS+=($!); }
call() {
  for attempt in 1 2; do
    out=$(timeout 120 ros2 service call /policy_control/$1 std_srvs/srv/Trigger "{}" 2>&1 | tr -d '\n'); echo "[run] $1: ${out:0:200}"
    echo "$out" | grep -q "response:" && break
  done
  echo "$out" | grep -q "success=True"
}
wait_topic() { for i in $(seq 1 60); do timeout 3 ros2 topic list 2>/dev/null | grep -qx "$1" && return 0; sleep 1; done; echo "[run] $1 없음"; return 1; }
echo "[run] fake replay · domain $ROS_DOMAIN_ID · $SIDE · rate $RATE · $NPZ · log $LOG"
# fake_plant 도 pd 와 같은 중력 모델을 쓴다(안 넘기면 기본 pd_dg5f_m_fake — fake_direct_run.sh 와 같은 이유)
launch_bg ros2 launch deploy/policy_control/launch/fake_plant.launch.py side:=both robot:=$ROBOT_BI contract:=$CONTRACT \
  pd_config:=$PD hands:=none plant_model:=rate "arm_start:=right=0,0,0,0.02,0,0,0;left=0,0,0,0.02,0,0,0" > "$LOG/fake_plant.log" 2>&1
launch_bg python3 scripts/fakes/fake_rh56f1_hand.py --side $SIDE > "$LOG/fake_hand.log" 2>&1
launch_bg python3 deploy/policy_control/policy_control/rh56f1_state_node.py --side $SIDE --map $MAP > "$LOG/hand_state.log" 2>&1
sleep 3
launch_bg ros2 launch deploy/policy_control/launch/pd_controller.launch.py contract:=$CONTRACT robot:=$ROBOT pd_config:=$PD \
  sides:=$SIDE execute:=true stage:=full fake:=true use_source:=true > "$LOG/pd.log" 2>&1
wait_topic /policy_control/status/pd_$SIDE || exit 4
# ★백그라운드(&) 명령의 표준입력은 /dev/null 이라 heredoc 을 python3 - 로 넘기면 빈 스크립트가 돈다 — 파일로 쓴 뒤 띄운다
cat > "$LOG/rec.py" <<'PY'
# 기록: joint_target · 팔 실측 · 손 실측 · pd phase · 손 토픽 수(위치 angle_set / 어드민턴스 angle_target)
import json, signal, sys, time
import numpy as np, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String
out, side = sys.argv[1], sys.argv[2]
rclpy.init(); n = rclpy.create_node("fake_replay_rec")
R = {"tgt": [], "arm": [], "hand": [], "phase": [], "ev": []}; cnt = {"angle_set": 0, "angle_target": 0}
arm_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
def tgt(m):
    if any(x.startswith(side[0] + "_aj_") for x in m.name):
        R["tgt"].append([time.time()] + list(m.position))
def arm(m):
    d = dict(zip(m.name, m.position))
    if all(a in d for a in arm_names): R["arm"].append([time.time()] + [d[a] for a in arm_names])
n.create_subscription(JointState, "/policy_control/joint_target", tgt, 50)
n.create_subscription(JointState, "/joint_states", arm, qos_profile_sensor_data)
n.create_subscription(JointState, f"/hand_{side}/joint_states", lambda m: R["hand"].append([time.time()] + list(m.position)), qos_profile_sensor_data)
n.create_subscription(String, f"/policy_control/status/pd_{side}", lambda m: R["phase"].append((time.time(), json.loads(m.data).get("phase"), json.loads(m.data).get("target"))), 10)
n.create_subscription(String, f"/policy_control/{side}/episode", lambda m: R["ev"].append((time.time(), m.data)), 10)
for topic in ("angle_set", "angle_target"):
    try:
        from rh56f1_interfaces.msg import SetAngle1 as _M  # noqa
    except Exception:
        _M = None
    if _M is not None:
        n.create_subscription(_M, f"/hand_{side}/{topic}", lambda m, t=topic: cnt.__setitem__(t, cnt[t] + 1), 50)
stop = {"v": False}
signal.signal(signal.SIGTERM, lambda *a: stop.update(v=True))
while not stop["v"]:
    rclpy.spin_once(n, timeout_sec=0.05)
np.savez(out, tgt=np.array(R["tgt"]), arm=np.array(R["arm"]), hand=np.array(R["hand"]),
         phase=np.array([json.dumps(p) for p in R["phase"]]), ev=np.array([json.dumps(e) for e in R["ev"]]),
         counts=np.array(json.dumps(cnt)))
print("saved", out, {k: len(v) for k, v in R.items()}, cnt, flush=True)
PY
launch_bg python3 "$LOG/rec.py" "$LOG/rec.npz" "$SIDE" > "$LOG/rec.log" 2>&1
REC_PID=${PIDS[-1]}
sleep 2; RC=0
call pd_$SIDE/engage || RC=1
[ $RC -eq 0 ] && { call pd_$SIDE/goto_home || RC=1; }
sleep 3
if [ $RC -eq 0 ]; then
  python3 deploy/policy_control/tools/replay_traj_rh.py --npz "$NPZ" --rate-scale $RATE --execute 2>&1 | tee "$LOG/replay.txt"
  RC=${PIPESTATUS[0]}
fi
sleep 2
call pd_$SIDE/release || true
kill -TERM -- -"$REC_PID" 2>/dev/null
for i in $(seq 1 20); do [ -f "$LOG/rec.npz" ] && break; sleep 0.5; done
python3 - "$LOG" "$NPZ" "$SIDE" "$RC" <<'PY' | tee "$LOG/summary.txt"
import json, sys
import numpy as np
log, npz, side, rc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
d = np.load(f"{log}/rec.npz"); ph = [json.loads(p) for p in d["phase"]]; ev = [json.loads(e)[1] for e in d["ev"]]
phases = [p[1] for p in ph]
print(f"replay rc {rc} · pd phases {sorted(set(phases))} · HOLD rows {phases.count('HOLD')} · 마지막 target {ph[-1][2] if ph else None}")
print("episode 사건:", [json.loads(e).get("event") + ":" + str(json.loads(e).get("hand_command", "-")) for e in ev])
print("손 토픽 수:", str(d["counts"]))
tgt, arm = d["tgt"], d["arm"]
if len(tgt) and len(arm):
    # 마지막 목표와 실측 차(정착), 재생 중 추종 오차(목표 시각 기준 가장 가까운 실측)
    idx = np.searchsorted(arm[:, 0], tgt[:, 0]).clip(0, len(arm) - 1)
    err = np.abs(tgt[:, 1:8] - arm[idx, 1:8]).max(axis=1)
    print(f"joint_target {len(tgt)} 행 · 팔 추종 오차 p50/p95/max {np.percentile(err, [50, 95, 100]).round(3).tolist()} rad"
          f" · 마지막 목표−실측 {np.abs(tgt[-1, 1:8] - arm[-1, 1:8]).max():.3f}")
z = np.load(npz)
print(f"궤적 끝 팔 {np.round(z['arm_q'][-1], 3).tolist()}")
if len(arm):
    print(f"fake 끝 팔 {np.round(arm[-1, 1:8], 3).tolist()}")
PY
exit $RC

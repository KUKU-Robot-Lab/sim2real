#!/bin/bash
# 에피소드 끝난 뒤 판정을 server GPU0 에서 한 번에 돌린다: GPU0 점유 확인 → 8B 서버 기동 → 준비 대기 → episode_review score → 서버 SIGTERM.
# 사용: vlm_supervisor/tools/review_on_server.sh <작업 폴더>   (작업 폴더 = real_frames --out 결과: bounds.jsonl + 에피소드별 jpg)
# 결과: <작업 폴더>/review.jsonl, 로그 review_score.log · server_8b.log, PID 기록 pids.log. 이미 있는 review.jsonl 은 이어서 채운다.
# 바꿀 값(환경 변수): VLM_PY, VLM_MODEL, VLM_PORT(8767), GPU0_MAX_MIB(2000, GPU0 에 이보다 큰 프로세스가 있으면 멈춘다),
#   GPU0_NEED_MIB(20000, GPU0 빈 메모리가 이보다 적으면 멈춘다), READY_SEC(900, 서버 준비 대기 상한)
set -u
W=$(realpath "${1:?작업 폴더를 준다}")
[ -f "$W/bounds.jsonl" ] || { echo "$W/bounds.jsonl 없음"; exit 1; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${VLM_PY:-$HOME/rl_ws/tools/venvs/vlm/bin/python}
MODEL=${VLM_MODEL:-$HOME/rl_ws/models/Cosmos-Reason2-8B}
PORT=${VLM_PORT:-8767}
MAX_MIB=${GPU0_MAX_MIB:-2000}
NEED_MIB=${GPU0_NEED_MIB:-20000}
READY_SEC=${READY_SEC:-900}
URL=http://127.0.0.1:$PORT
log() { echo "$(date +%F_%T) $*" | tee -a "$W/pids.log"; }

# GPU0 점유: 다른 큰 프로세스가 있으면 띄우지 않는다(남의 작업은 내리지 않는다)
APPS=$(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits) || { echo "nvidia-smi 실패"; exit 1; }
# 메모리 값이 숫자가 아니면([N/A] 등) 큰 프로세스로 본다
BIG=$(echo "$APPS" | awk -F', *' -v m="$MAX_MIB" 'NF>=2 && ($2 !~ /^[0-9]+$/ || $2+0 > m)')
if [ -n "$BIG" ]; then
  echo "GPU0 에 ${MAX_MIB} MiB 넘는 프로세스가 있어 멈춘다(pid, MiB):"; echo "$BIG"; exit 2
fi
FREE=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
case "$FREE" in ''|*[!0-9]*) echo "GPU0 빈 메모리를 읽지 못했다($FREE)"; exit 2;; esac
[ "$FREE" -ge "$NEED_MIB" ] || { echo "GPU0 빈 메모리 ${FREE} MiB < ${NEED_MIB} MiB 라 멈춘다"; exit 2; }
"$PY" -c "import requests, yaml" 2>/dev/null || { echo "$PY 에 requests·yaml 이 없다"; exit 1; }
if curl -sf "$URL/" >/dev/null; then echo "포트 $PORT 에 이미 서버가 있다(남의 서버일 수 있어 멈춘다)"; exit 2; fi

cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$REPO/deploy/policy_control
CUDA_VISIBLE_DEVICES=0 setsid "$PY" vlm_supervisor/tools/vlm_server.py --model "$MODEL" --host 127.0.0.1 --port "$PORT" > "$W/server_8b.log" 2>&1 &
P=$!
log "server python PID $P port $PORT model $MODEL start"

stop_server() {
  kill -0 "$P" 2>/dev/null || return 0
  kill -TERM "$P"
  for _ in $(seq 60); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
  if kill -0 "$P" 2>/dev/null; then
    log "server PID $P 가 SIGTERM 120 s 뒤에도 살아 있다. 손으로 확인(SIGKILL 은 하지 않음)"; return 3
  fi
  sleep 3
  LEFT=$(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', *' -v p="$P" '$1==p')
  if [ -n "$LEFT" ]; then log "server PID $P 가 GPU0 에 남아 있다: $LEFT"; return 3; fi
  log "server PID $P terminated, GPU0 에서 해제 확인"
}
on_exit() { RC=$?; stop_server || RC=3; exit $RC; }
trap on_exit EXIT
trap 'exit 143' TERM INT HUP

T0=$SECONDS
until curl -sf "$URL/" >/dev/null; do
  kill -0 "$P" 2>/dev/null || { log "server died (server_8b.log 확인)"; exit 1; }
  [ $((SECONDS - T0)) -lt "$READY_SEC" ] || { log "server 가 ${READY_SEC} s 안에 준비되지 않았다"; exit 1; }
  sleep 5
done
log "server ready"
"$PY" -m vlm_supervisor.bench.episode_review score "$W/bounds.jsonl" --url "$URL" --out "$W/review.jsonl" > "$W/review_score.log" 2>&1
RC=$?
log "score exit $RC ($(tail -1 "$W/review_score.log"))"
exit $RC

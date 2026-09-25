#!/usr/bin/env bash
# 저장 경로(npz)를 읽기 전용 Isaac 뷰어에서 재생하며 **영상**으로 남긴다. 로봇에는 아무것도 보내지 않는다.
#
#   deploy/policy_control/tools/record_path_video.sh deploy/policy_control/paths/home_right.npz
#   REVERSE=1 deploy/policy_control/tools/record_path_video.sh logs/policy_control/reset_right.npz   # 리셋(되짚기)
#   SPEED=2 OUT=~/rl_ws/our_source/x.mp4 ...
#
# env: REVERSE(0|1) · SPEED(기본 1) · OUT(기본 ~/rl_ws/our_source/<npz 이름>_<날짜>.mp4) · CUP · POLICY_DIR
#      HAND(keep|pd|contract|CSV — 이동 중 손 자세) · HAND_FROM(주면 맨 앞에서 손이 그 자세에서 HAND 로 움직인다)
#
# 손을 빼면(HAND 기본 keep) 화면의 손은 **정책 리셋 자세로 굳어 있다** — 실기의 손 모양이 아니다.
# 복귀 영상은 보통 `HAND=pd HAND_FROM=contract` 다: 손이 먼저 주먹(봉투 구 안)으로 오므라든 뒤 팔이 되짚는다.
#
# 09.23: 뷰어에 `--record <dir>` 를, 미리보기에 `--reverse` 를 붙여 만들었다. 렌더가 송신보다 느려 프레임이
# 빠지므로(1280x720 에서 8~9 fps) 영상 fps 는 **실측 간격**으로 맞춘다 — 그래야 실시간 속도로 보인다.
set -euo pipefail
NPZ="${1:?사용법: record_path_video.sh <npz> (env: REVERSE SPEED OUT)}"
SIM2REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SIM2REAL"
[ -f "$NPZ" ] || { echo "✗ npz 가 없다: $NPZ" >&2; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/path_video_XXXXXX")"
OUT="${OUT:-$HOME/rl_ws/our_source/$(basename "${NPZ%.npz}")_${STAMP}.mp4}"
LOG="$WORK/viewer.log"
mkdir -p "$(dirname "$OUT")" "$WORK/frames"

echo "[record] npz $NPZ · 역재생 ${REVERSE:-0} · ${SPEED:-1}배속 · 손 ${HAND:-keep}${HAND_FROM:+ (←$HAND_FROM)} → $OUT"
echo "[record] 뷰어 기동(헤드리스, 부팅 1~2 분) — 로그 $LOG"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-97}" setsid bash robot/isaacsim_bridge/viewer/run_viewer.sh \
    --headless --record "$WORK/frames" --record_idle_s 3 --max_seconds "${MAX_SECONDS:-600}" \
    ${CUP:+--cup "$CUP"} ${POLICY_DIR:+--policy_dir "$POLICY_DIR"} > "$LOG" 2>&1 &

for _ in $(seq 1 120); do
    grep -q "isaac_viewer] 녹화" "$LOG" 2>/dev/null && break
    grep -qE "Traceback|^Error" "$LOG" 2>/dev/null && { tail -20 "$LOG" >&2; exit 3; }
    sleep 5
done
grep -q "isaac_viewer] 녹화" "$LOG" || { echo "✗ 뷰어가 준비되지 않았다 — $LOG" >&2; exit 4; }

.venv/bin/python deploy/policy_control/tools/preview_path_in_viewer.py --npz "$NPZ" \
    --speed "${SPEED:-1}" ${REVERSE:+--reverse} --hand "${HAND:-keep}" \
    ${HAND_FROM:+--hand-from "$HAND_FROM"} ${HAND_RAMP_S:+--hand-ramp-s "$HAND_RAMP_S"}
sleep 5                                     # 뷰어가 idle 을 보고 녹화를 닫을 시간

OUT="$OUT" WORK="$WORK" python3 - <<'PY'
import cv2, glob, json, os
from pathlib import Path
work, out = Path(os.environ["WORK"]), Path(os.environ["OUT"])
rows = [l for l in (work / "viewer.log").read_text().splitlines() if l.startswith("VIEWER_RECORD")]
if not rows:
    raise SystemExit("✗ 녹화 요약(VIEWER_RECORD)이 없다 — 뷰어 로그를 볼 것")
fps = max(1.0, json.loads(rows[-1].split(" ", 1)[1])["fps"])
keep = [im for im in (cv2.imread(f) for f in sorted(glob.glob(str(work / "frames" / "f*.png"))))
        if im is not None and im.mean() > 4]        # 첫 몇 장은 annotator 가 비어 검다
if not keep:
    raise SystemExit("✗ 쓸 프레임이 없다")
h, w = keep[0].shape[:2]
vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for im in keep:
    vw.write(im)
vw.release()
print(f"[record] {out} · {len(keep)} 프레임 · {w}x{h} · {fps:.2f} fps · {len(keep)/fps:.1f} s "
      f"· {out.stat().st_size/1e6:.1f} MB")
PY
rm -rf "$WORK/frames"

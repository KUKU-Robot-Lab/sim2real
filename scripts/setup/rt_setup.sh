#!/bin/bash
# 로봇 PC 실시간 한도 — EtherCAT 마스터(SCHED_FIFO 80) · controller_manager(FIFO 50) 가 실시간 우선순위를 받게 한다.
#
# 10.03 사용자: "sim2real 은 CPU 최적화가 자동이어야 한다 — 어떤 PC 에서 세팅할지 모른다."
# 10.03 arm4090 실측: 두 프로세스 모두 'SCHED_FIFO … Operation not permitted' → 보통 우선순위로 돌았다(RTPRIO 한도 0).
# 코어 배치(어느 코어에 누구)는 노드가 그 PC 의 sysfs 를 읽어 스스로 정한다(policy_control/cpu_plan.py) — 여기서는
# 운영자 권한이 필요한 '한도'만 연다. PC 마다 한 번.
#
#   sudo bash scripts/setup/rt_setup.sh                # 한도 열기(이 사용자만)
#   sudo bash scripts/setup/rt_setup.sh --performance  # + 부팅 때 CPU governor 를 performance 로(전력 · 발열 늘어남)
#   bash scripts/setup/rt_setup.sh --check             # 무엇이 적용돼 있나(sudo 없이)
#   sudo bash scripts/setup/rt_setup.sh --undo         # 되돌리기
#
# 쓰는 파일 — 로그인 경로마다 한도를 주는 곳이 다르다:
#   /etc/security/limits.d/99-sim2real-rt.conf              ssh · 콘솔 · sudo 로그인(pam_limits)
#   /etc/systemd/system/user@<uid>.service.d/99-sim2real-rt.conf   이 사용자의 systemd 사용자 관리자 자체
#   ~/.config/systemd/user.conf  [Manager] DefaultLimit*     GNOME 터미널 등 사용자 관리자가 띄우는 프로그램
#     (arm4090: terminator · ros2_control_node 가 user@1000.service 아래에서 RTPRIO 0 이었다 — limits.d 만으로는 안 바뀐다)
# 적용은 다시 로그인한 뒤부터다. 사용자 관리자가 세션 사이에 살아 있을 수 있어 재부팅이 확실하다.
set -euo pipefail

RTPRIO=98
TAG=99-sim2real-rt.conf
GOV_UNIT=sim2real-cpu-performance.service
MODE=apply
PERF=0
for a in "$@"; do
  case "$a" in
    --check) MODE=check ;;
    --undo) MODE=undo ;;
    --performance) PERF=1 ;;
    -h|--help) sed -n 2,22p "$0"; exit 0 ;;
    *) echo "모르는 인자: $a (--check · --undo · --performance)"; exit 2 ;;
  esac
done

if [ "$MODE" = check ]; then
  TARGET=${SUDO_USER:-$USER}
else
  if [ "$(id -u)" -ne 0 ]; then echo "운영자 권한이 필요하다: sudo bash $0 $*"; exit 1; fi
  TARGET=${SUDO_USER:-}
  if [ -z "$TARGET" ] || [ "$TARGET" = root ]; then echo "로봇을 돌릴 사용자 셸에서 sudo 로 실행할 것(root 로그인 셸 말고)"; exit 1; fi
fi
TUID=$(id -u "$TARGET")
THOME=$(getent passwd "$TARGET" | cut -d: -f6)
LIMITS=/etc/security/limits.d/$TAG
UNIT_DIR=/etc/systemd/system/user@$TUID.service.d
UNIT_DROP=$UNIT_DIR/$TAG
USER_CONF=$THOME/.config/systemd/user.conf

check() {
  echo "사용자 $TARGET (uid $TUID)"
  for f in "$LIMITS" "$UNIT_DROP"; do
    if [ -f "$f" ]; then echo "  [있음] $f"; else echo "  [없음] $f"; fi
  done
  if grep -qs "^# sim2real-rt" "$USER_CONF"; then echo "  [있음] $USER_CONF (sim2real-rt 블록)"; else echo "  [없음] $USER_CONF (sim2real-rt 블록)"; fi
  if systemctl is-enabled "$GOV_UNIT" >/dev/null 2>&1; then echo "  [있음] $GOV_UNIT (governor performance)"; else echo "  [없음] $GOV_UNIT (선택)"; fi
  echo "  지금 이 셸: rtprio $(ulimit -r) · memlock $(ulimit -l) · governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '-')"
  echo "  필요: rtprio ≥ 80 (EtherCAT 마스터) · memlock unlimited"
}

if [ "$MODE" = check ]; then check; exit 0; fi

if [ "$MODE" = undo ]; then
  rm -f "$LIMITS" "$UNIT_DROP"
  rmdir "$UNIT_DIR" 2>/dev/null || true
  if [ -f "$USER_CONF" ]; then sudo -u "$TARGET" -- sed -i '/^# sim2real-rt/,+3d' "$USER_CONF"; fi   # 사용자 소유 그대로
  if systemctl is-enabled "$GOV_UNIT" >/dev/null 2>&1; then systemctl disable --now "$GOV_UNIT"; fi
  rm -f "/etc/systemd/system/$GOV_UNIT"
  systemctl daemon-reload
  echo "되돌렸다 — 다시 로그인(재부팅)하면 원래 한도로 돈다"
  exit 0
fi

cat > "$LIMITS" <<EOF
# sim2real rt_setup.sh — EtherCAT 마스터(SCHED_FIFO 80) · controller_manager(50) 실시간 우선순위, 마스터 mlockall
$TARGET - rtprio $RTPRIO
$TARGET - memlock unlimited
EOF
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DROP" <<EOF
# sim2real rt_setup.sh — 사용자 관리자 자체의 한도(이 아래에서 뜨는 GNOME 터미널 · 앱이 물려받는다)
[Service]
LimitRTPRIO=$RTPRIO
LimitMEMLOCK=infinity
EOF
# 사용자 홈의 파일은 그 사용자로 쓴다 — root 로 mkdir 하면 ~/.config 까지 root 소유가 되어 GNOME · dconf 쓰기가 막힌다
as_user() { sudo -u "$TARGET" -- "$@"; }
if ! grep -qs "^# sim2real-rt" "$USER_CONF"; then
  if grep -qs "^DefaultLimitRTPRIO=\|^DefaultLimitMEMLOCK=" "$USER_CONF"; then
    echo "  ⚠ $USER_CONF 에 이미 DefaultLimit 값이 있다 — 뒤에 붙는 이 값이 덮는다(systemd 는 나중 값을 쓴다)"
  fi
  as_user mkdir -p "$(dirname "$USER_CONF")"
  # 마지막 줄에 개행이 없으면 먼저 하나 — 표시줄이 남의 값 뒤에 붙지 않게
  if [ -s "$USER_CONF" ] && [ -n "$(tail -c1 "$USER_CONF")" ]; then as_user sh -c 'echo >> "$1"' _ "$USER_CONF"; fi
  # systemd 는 줄 끝 주석을 못 읽는다(섹션 머리에 붙이면 파일 전체가 깨진다) — 표시는 제 줄에. --undo 가 이 4 줄을 지운다
  as_user sh -c 'printf "# sim2real-rt (rt_setup.sh)\n[Manager]\nDefaultLimitRTPRIO=%s\nDefaultLimitMEMLOCK=infinity\n" "$2" >> "$1"' \
    _ "$USER_CONF" "$RTPRIO"
fi

if [ "$PERF" = 1 ]; then
  cat > "/etc/systemd/system/$GOV_UNIT" <<'EOF'
[Unit]
Description=sim2real: CPU governor performance (robot control timing)
After=sysinit.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g" || true; done'

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now "$GOV_UNIT"
else
  systemctl daemon-reload
fi

check
echo
echo "다음: 재부팅(또는 이 사용자의 모든 세션 로그아웃 → 다시 로그인) 뒤"
echo "      python3 scripts/setup/check_host.py --robot <rh56f1|dg5f> --only cpu    # rtprio ≥ 80 이면 통과"

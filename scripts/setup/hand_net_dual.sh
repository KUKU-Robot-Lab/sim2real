#!/usr/bin/env bash
# hand_net_dual.sh — DG-5F 양손 이더넷(Modbus TCP :502) 배선·주소·경로 점검/설정 (pc5090, 2026-09-14)
#
# 배선(2026-09-14 사용자 확정):
#   우손 = 본체 이더넷 eno1                          → 169.254.186.72 (dg5f_right_driver 기본값)
#   좌손 = USB-C 허브 이더넷 enx00e04c6806e1(r8152)  → 169.254.186.73 (dg5f_left_driver 기본값)
#
# 함정: 두 손이 같은 169.254.186.x 인데 NIC 이 둘이다. 게다가 169.254.0.0/16 경로가 wifi(wlp131s0)에
#   이미 잡혀 있어 서브넷(/16) 주소만 주면 패킷이 엉뚱한 NIC 로 나간다(08.03 교차배선 사고와 같은 종류,
#   09.07 에는 tailscale0 로 새서 controller_manager 가 접속 대기로 멈췄다).
#   그래서 손마다 /32 호스트 경로를 NIC 에 고정한다. NetworkManager 프로필(hand-right / hand-left)이라
#   재부팅·케이블 재연결 뒤에도 유지된다.
#
# 사용: bash scripts/setup/hand_net_dual.sh           # 점검만 (0 = 양손 정상, 1 = 실패 항목 있음)
#       bash scripts/setup/hand_net_dual.sh --apply   # nmcli 프로필 생성/갱신 후 점검
# env:  RIGHT_IF RIGHT_PEER RIGHT_SRC LEFT_IF LEFT_PEER LEFT_SRC 로 덮어쓴다(허브 교체로 NIC 이름이 바뀌면).
set -u

RIGHT_IF="${RIGHT_IF:-eno1}"
RIGHT_PEER="${RIGHT_PEER:-169.254.186.72}"
RIGHT_SRC="${RIGHT_SRC:-169.254.186.1}"
LEFT_IF="${LEFT_IF:-enx00e04c6806e1}"
LEFT_PEER="${LEFT_PEER:-169.254.186.73}"
LEFT_SRC="${LEFT_SRC:-169.254.186.2}"

fail=0
ok()  { printf '  ✓ %s\n' "$1"; }
bad() { printf '  ✗ %s\n' "$1"; fail=1; }

apply_profile() {  # name ifname src peer
    local name=$1 ifc=$2 src=$3 peer=$4
    local props=(connection.interface-name "$ifc" connection.autoconnect yes
                 connection.autoconnect-priority 10
                 ipv4.method manual ipv4.addresses "$src/32" ipv4.routes "$peer/32"
                 ipv4.never-default yes ipv6.method ignore)
    if nmcli -t -f NAME con show | grep -qx "$name"; then
        nmcli con modify "$name" "${props[@]}"
    else
        nmcli con add type ethernet con-name "$name" "${props[@]}"
    fi
    nmcli con up "$name" >/dev/null 2>&1 \
        || printf '  ⚠ %s 활성화 보류 — 링크가 없다(케이블·손 전원). 전원 후 nmcli con up %s\n' "$name" "$name"
}

check_hand() {  # label ifname src peer
    local label=$1 ifc=$2 src=$3 peer=$4
    echo "[$label] $ifc → $peer"
    if ! ip link show "$ifc" >/dev/null 2>&1; then
        bad "$ifc 인터페이스 없음 (허브 분리? 이름 변경? ip -br link 로 확인 후 ${label^^}_IF=…)"
        return
    fi
    local state dev
    state=$(ip -br link show "$ifc" | awk '{print $2}')
    if [ "$state" = "UP" ]; then ok "링크 UP"; else bad "링크 $state (케이블·손 전원)"; fi
    if ip -br addr show "$ifc" | grep -q "$src"; then ok "$ifc = $src"; else bad "$ifc 에 $src 없음 (--apply)"; fi
    dev=$(ip route get "$peer" 2>/dev/null | grep -o 'dev [^ ]*' | awk '{print $2}')
    if [ "$dev" = "$ifc" ]; then ok "경로 $peer → $dev"; else bad "경로 $peer → ${dev:-없음} (기대 $ifc) — 교차 위험"; fi
    if ping -c2 -W1 -I "$ifc" "$peer" >/dev/null 2>&1; then
        ok "ping $peer"
    else
        bad "ping $peer 무응답 (손 전원 · 손 IP 가 기본값과 다른지: arp-scan -I $ifc 169.254.186.0/24)"
    fi
}

if [ "$RIGHT_IF" = "$LEFT_IF" ] || [ "$RIGHT_PEER" = "$LEFT_PEER" ]; then
    echo "✗ 좌우 NIC 또는 손 IP 가 같다 — 이 스크립트는 손마다 NIC 하나를 전제한다"; exit 1
fi

if [ "${1:-}" = "--apply" ]; then
    echo "[apply] NetworkManager 프로필 hand-right / hand-left"
    apply_profile hand-right "$RIGHT_IF" "$RIGHT_SRC" "$RIGHT_PEER"
    apply_profile hand-left "$LEFT_IF" "$LEFT_SRC" "$LEFT_PEER"
fi

check_hand right "$RIGHT_IF" "$RIGHT_SRC" "$RIGHT_PEER"
check_hand left "$LEFT_IF" "$LEFT_SRC" "$LEFT_PEER"
echo
if [ "$fail" -eq 0 ]; then echo "양손 네트워크 정상"; else echo "실패 항목 있음"; fi
exit "$fail"

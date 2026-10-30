#!/bin/bash
# DDS 를 허브 랜 NIC 한 곳에 묶는 Fast-DDS 프로파일 — source 해서 쓴다. FASTRTPS_DEFAULT_PROFILES_FILE 과 FPP_DDS_ARGS 를 만든다.
# ★10.30 사용자: "5090 과 4090 은 ssh 가 아니라 랜 포트 연결로 통신해야 함".
#   ROS_LOCALHOST_ONLY=0 인 프로세스는 기본으로 모든 인터페이스(랜 · Wi-Fi 192.168.1.x · tailscale · docker 브리지)에 locator 를
#   광고하고, 쓰기 쪽은 상대의 locator 전부에 보낸다. 10.30 실측(도메인 50, ros2 CLI 200 Hz 발행): 프로파일 없이 105~118 Hz 수신,
#   양쪽에 프로파일을 켜면 200 Hz. common.sh 의 09.26 wifi 멀티캐스트 문제도 소켓이 랜 IP 에만 묶여 안 난다.
# 규칙: ROS_LOCALHOST_ONLY=1 로 띄우는 명령은 이 변수를 지운다(env -u FASTRTPS_DEFAULT_PROFILES_FILE 또는 unset).
#   둘이 같이 있으면 rmw_fastrtps 가 XML 의 랜 transport 를 그대로 두어 localhost 격리가 깨진다(10.30 실측: 4090 에서 보임).
#   이 파일은 source 시점에 ROS_LOCALHOST_ONLY=1 이면 내보내지 않는다. 실행 지점은 tests/test_dds_lan_profile.py 가 검사한다.
# 값: 랜 IP 는 이 호스트의 S2R_LAN_SUBNET(config/ros_domain.env) 주소에서 찾는다. S2R_LAN_IP 로 덮어쓸 수 있다(시험 · 특수 호스트).
#   랜 IP 가 없으면(laptop · server) 내보내지 않고 경고만 — DDS 는 기본대로 모든 인터페이스를 쓴다.
# set -e 인 호출자(common.sh · console.sh) 안에서도 안전해야 한다 — 실패할 수 있는 명령은 || true 로 감싼다.
_s2r_dds_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_s2r_dds_dir/../ros_domain.env"
if [ "${ROS_LOCALHOST_ONLY:-0}" = "1" ]; then
  unset FASTRTPS_DEFAULT_PROFILES_FILE
else
  _s2r_lan_ip="${S2R_LAN_IP:-}"
  if [ -z "$_s2r_lan_ip" ]; then
    _s2r_lan_ip="$(ip -4 -o addr show 2>/dev/null | awk -v p="$S2R_LAN_SUBNET" 'index($4,p)==1 {sub("/.*","",$4); print $4; exit}' || true)"
  fi
  if [ -n "$_s2r_lan_ip" ]; then
    _s2r_dds_out="${XDG_RUNTIME_DIR:-/tmp}/s2r_fastdds_lan_${_s2r_lan_ip}.xml"
    sed "s/@LAN_IP@/${_s2r_lan_ip}/g" "$_s2r_dds_dir/fastdds_lan.xml.in" > "$_s2r_dds_out"
    export FASTRTPS_DEFAULT_PROFILES_FILE="$_s2r_dds_out" S2R_LAN_IP="$_s2r_lan_ip"
  else
    echo "[dds/lan_profile.sh] ★이 호스트에 랜(${S2R_LAN_SUBNET}*) IP 가 없다 — DDS 가 모든 인터페이스를 쓴다" >&2
    unset FASTRTPS_DEFAULT_PROFILES_FILE
  fi
fi
# FP++ 컨테이너(fpp_up.sh · fpp_group_up.sh)에 같은 프로파일을 붙이는 docker run 인자. 프로파일이 없으면 빈 배열.
# 컨테이너 안은 humble 기본 rmw_fastrtps(RMW_IMPLEMENTATION 미설정, 10.30 s2r:vision 확인)라 같은 XML 이 그대로 먹는다.
FPP_DDS_ARGS=()
if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
  FPP_DDS_ARGS=(-v "$FASTRTPS_DEFAULT_PROFILES_FILE":/opt/params/fastdds_lan.xml:ro -e FASTRTPS_DEFAULT_PROFILES_FILE=/opt/params/fastdds_lan.xml)
fi
unset _s2r_dds_dir _s2r_lan_ip _s2r_dds_out

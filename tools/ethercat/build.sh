#!/bin/bash
# SOEM v1.4.0(사용자 빌드 ~/rl_ws/SOEM) 에 링크한다. 실행에는 cap_net_raw 가 필요하다(운영자가 setcap).
# /etc/sudoers.d/rh56f1-setcap 은 마스터 한 파일에 대해 정확히 아래 명령만 NOPASSWD 로 허용한다
# (인자 글자까지 같아야 맞는다: /usr/sbin/setcap cap_net_raw,cap_net_admin=eip <절대경로>).
set -e
SOEM=${SOEM:-$HOME/rl_ws/SOEM}
cd "$(dirname "$0")"
INC=(-I"$SOEM/soem" -I"$SOEM/osal" -I"$SOEM/osal/linux" -I"$SOEM/oshw/linux")
SETCAP=/usr/sbin/setcap
CAPS=cap_net_raw,cap_net_admin=eip

grant() {  # 권한이 없으면 sudoers 허용 명령으로 붙이고, 안 되면 운영자 명령을 알려 준다
  local f="$(pwd)/$1"
  if getcap "$f" | grep -q cap_net_raw; then echo "$1 · setcap 있음"; return; fi
  if sudo --non-interactive "$SETCAP" "$CAPS" "$f" 2>/dev/null; then echo "$1 · setcap 붙임"
  else echo "$1 — ★setcap 없음: sudo $SETCAP $CAPS $f"; fi
}

for t in ecat_rtt ecat_rh56f1 rh56f1_ecat_master; do
  # 다시 빌드하면 파일이 바뀌어 setcap 이 풀린다 — 내용이 같으면 건드리지 않는다
  gcc -O2 -Wall -o "$t.new" "$t.c" "${INC[@]}" "$SOEM/build/libsoem.a" -lpthread -lrt -lm
  if [ -f "$t" ] && cmp -s "$t" "$t.new"; then rm "$t.new"; echo "$t 그대로"
  else mv "$t.new" "$t"; echo "built $(pwd)/$t"; fi
  grant "$t"
done

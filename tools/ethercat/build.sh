#!/bin/bash
# SOEM v1.4.0(사용자 빌드 ~/rl_ws/SOEM) 에 링크한다. 실행에는 cap_net_raw 가 필요하다(운영자가 setcap).
set -e
SOEM=${SOEM:-$HOME/rl_ws/SOEM}
cd "$(dirname "$0")"
INC=(-I"$SOEM/soem" -I"$SOEM/osal" -I"$SOEM/osal/linux" -I"$SOEM/oshw/linux")
for t in ecat_rtt ecat_rh56f1 rh56f1_ecat_master; do
  # 다시 빌드하면 파일이 바뀌어 setcap 이 풀린다 — 내용이 같으면 건드리지 않는다
  gcc -O2 -Wall -o "$t.new" "$t.c" "${INC[@]}" "$SOEM/build/libsoem.a" -lpthread -lrt
  if [ -f "$t" ] && cmp -s "$t" "$t.new"; then rm "$t.new"; echo "$t 그대로(setcap 유지)"; else mv "$t.new" "$t"; echo "built $(pwd)/$t — 다음: sudo setcap cap_net_raw,cap_net_admin=ep $(pwd)/$t"; fi
done

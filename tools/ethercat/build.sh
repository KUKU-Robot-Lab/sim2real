#!/bin/bash
# SOEM v1.4.0(사용자 빌드 ~/rl_ws/SOEM) 에 링크한다. 실행에는 cap_net_raw 가 필요하다(운영자가 setcap).
set -e
SOEM=${SOEM:-$HOME/rl_ws/SOEM}
cd "$(dirname "$0")"
gcc -O2 -o ecat_rtt ecat_rtt.c -I"$SOEM/soem" -I"$SOEM/osal" -I"$SOEM/osal/linux" -I"$SOEM/oshw/linux" \
    "$SOEM/build/libsoem.a" -lpthread -lrt
echo "built $(pwd)/ecat_rtt — 다음: sudo setcap cap_net_raw,cap_net_admin=ep $(pwd)/ecat_rtt"

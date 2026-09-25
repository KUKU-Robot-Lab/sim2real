#!/usr/bin/env bash
# 프로그램 메뉴의 "S2R 콘솔" 이 부르는 진입점 — 창 모드로 띄우고 출력은 logs/s2r_console_app.log 에 쌓는다.
# 등록: install_desktop_entry.sh
SIM2REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "$SIM2REAL/logs"
exec "$SIM2REAL/deploy/s2r_console/tools/console.sh" --window "$@" >> "$SIM2REAL/logs/s2r_console_app.log" 2>&1

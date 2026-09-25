#!/usr/bin/env bash
# S2R 콘솔을 프로그램 메뉴(활동 검색 "S2R")에 등록한다 — 누르면 창 모드(--window)로 뜬다.
#
#   deploy/s2r_console/tools/install_desktop_entry.sh            # 등록
#   deploy/s2r_console/tools/install_desktop_entry.sh --remove   # 해제
#
# 터미널 없이 뜨므로 콘솔 출력은 logs/s2r_console_app.log 에 쌓인다(console_app.sh). 실기 드라이버를 남기고 닫히면
# 창이 닫힌 뒤 대화상자로도 알린다(window.notice).
set -euo pipefail
SIM2REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENTRY="${XDG_DATA_HOME:-$HOME/.local/share}/applications/s2r-console.desktop"

if [ "${1:-}" = "--remove" ]; then
  rm -f "$ENTRY" && echo "해제: $ENTRY"
  exit 0
fi

mkdir -p "$(dirname "$ENTRY")"
cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=S2R 콘솔
Comment=OpenArm 실기 배포 콘솔 (창 모드)
Exec="$SIM2REAL/deploy/s2r_console/tools/console_app.sh"
Icon=utilities-system-monitor
Terminal=false
Categories=Development;
StartupWMClass=s2r-console
EOF
command -v desktop-file-validate >/dev/null && desktop-file-validate "$ENTRY"
echo "등록: $ENTRY"

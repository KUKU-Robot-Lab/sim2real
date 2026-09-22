"""isaacsim_viewer 테스트가 뷰어 폴더의 순수 모듈(joint_map·packet·joint_state_relay)을 이름으로 임포트하게 한다."""

from __future__ import annotations

import sys
from pathlib import Path

VIEWER = Path(__file__).resolve().parents[2] / "robot" / "isaacsim_bridge" / "viewer"
if str(VIEWER) not in sys.path:
    sys.path.insert(0, str(VIEWER))

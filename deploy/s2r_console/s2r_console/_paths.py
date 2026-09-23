"""콘솔이 재사용하는 형제 트리를 `sys.path` 에 넣는 **한 곳**.

`scripts/`(mission_core · mission_stages), `scripts/ops/`(mission_run), `policy_control/`
(status_join · policy_registry). 콘솔은 이들을 복사하지 않고 그대로 부른다.
★`policy_control._paths` 는 부르지 않는다 — 그것은 hdgp·fabrics 트리까지 올리는데 콘솔은 필요 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[3]
if not (SIM2REAL / "config").is_dir() or not (SIM2REAL / "scripts").is_dir():   # 깊이가 틀리면 여기서 멈춘다
    raise ImportError(f"sim2real 루트를 잘못 잡았다: {SIM2REAL} (config/·scripts/ 가 없다)")
SCRIPTS = SIM2REAL / "scripts"
OPS = SCRIPTS / "ops"
POLICY_CONTROL = SIM2REAL / "deploy" / "policy_control"
POLICIES = SIM2REAL / "deploy" / "policies"
WEB = Path(__file__).resolve().parent / "web"
#: 벤더 관절 한계(로봇 프로파일) — 화면의 로봇 상태 표가 "끝점" 을 표시할 때 읽는다(09.23).
ROBOT_PROFILE = SIM2REAL.parent / "robot_control" / "src" / "robot_control" / "profiles" / "openarm_tesollo.yaml"


def install() -> None:
    for path in (SCRIPTS, OPS, POLICY_CONTROL):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


install()

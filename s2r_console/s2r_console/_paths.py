"""콘솔이 재사용하는 형제 트리를 `sys.path` 에 넣는 **한 곳**.

`scripts/`(mission_core · mission_stages), `scripts/ops/`(mission_run), `policy_control/`
(status_join · policy_registry). 콘솔은 이들을 복사하지 않고 그대로 부른다.
★`policy_control._paths` 는 부르지 않는다 — 그것은 hdgp·fabrics 트리까지 올리는데 콘솔은 필요 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[2]
SCRIPTS = SIM2REAL / "scripts"
OPS = SCRIPTS / "ops"
POLICY_CONTROL = SIM2REAL / "policy_control"
POLICIES = SIM2REAL / "policies"
WEB = Path(__file__).resolve().parent / "web"


def install() -> None:
    for path in (SCRIPTS, OPS, POLICY_CONTROL):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


install()

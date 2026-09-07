"""setup.py 가 선언한 실행파일은 설치본에 실제로 있어야 한다 (ROS 없음).

2026-09-07 우팔 실기: `episode_master` 를 09.06 에 추가하고 패키지를 다시 빌드하지 않아
`policy_chain.launch.py` 가 **실기에서** 죽었다(`executable 'episode_master' not found`).
선언과 설치본이 갈리는 것을 코드가 알 수 있는데 아무도 안 봤다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
SETUP = SIM2REAL / "policy_control/setup.py"
LIBEXEC = SIM2REAL / "install/policy_control/lib/policy_control"


def declared_scripts() -> list[str]:
    block = re.search(r'"console_scripts":\s*\[(.*?)\]', SETUP.read_text(encoding="utf-8"), re.S)
    assert block, "setup.py 에 console_scripts 가 없다"
    return re.findall(r'"\s*([\w_]+)\s*=', block.group(1))


def test_setup_declares_the_five_nodes():
    assert set(declared_scripts()) >= {"obs_node", "policy_node", "fabric_node", "pd_node", "episode_master"}


@pytest.mark.skipif(not LIBEXEC.is_dir(), reason="colcon 설치본 없음")
def test_every_declared_script_is_installed():
    missing = [s for s in declared_scripts() if not (LIBEXEC / s).exists()]
    assert not missing, (
        f"{missing} 가 선언돼 있는데 설치본에 없다 — colcon build 를 다시 해야 한다. "
        "launch 는 실기에서야 이걸 발견한다")

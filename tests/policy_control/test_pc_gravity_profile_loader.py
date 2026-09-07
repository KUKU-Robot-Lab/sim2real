"""옛 도구들은 삭제된 자산 manifest 없이도 프로필을 읽어야 한다 (ROS 없음).

2026-09-05 자산 교체로 `robot_control.profile.load_builtin_profile` 이 삭제된 manifest 를
요구하며 죽는다. 그 사실은 09.05 에 기록됐고 `policy_control` 과 `fake_arm_bridge` 에는
우회가 들어갔지만 `gravity_comp_node` 는 빠졌다 — 2026-09-07 우팔 실기에서 중력보상을
켜려는 순간 발견됐다(팔을 세워둔 채). 이 테스트가 그 재발을 막는다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SIM2REAL = Path(__file__).resolve().parents[2]
PROFILE = Path("/home/user/rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml")
needs_profile = pytest.mark.skipif(not PROFILE.exists(), reason="robot_control 프로필 없음")


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SIM2REAL / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_profile
def test_gravity_comp_reads_profile_without_asset_manifest():
    mod = _load("scripts/nodes/gravity_comp_node.py", "gravity_comp_tool")
    grp = mod.load_profile_group(PROFILE, "openarm_right_arm")

    assert list(grp.canonical) == [f"r_aj_{i}" for i in range(1, 8)]
    assert list(grp.sources) == [f"openarm_right_joint{i}" for i in range(1, 8)]
    assert list(grp.signs) == [1] * 7
    assert grp.tip == "r_hl_palm_ee", "중력 체인은 자산 tip(palm_ee)을 써야 한다"
    assert grp.effort_controller == "right_forward_effort_controller"


@needs_profile
def test_left_group_resolves_too():
    mod = _load("scripts/nodes/gravity_comp_node.py", "gravity_comp_tool2")
    grp = mod.load_profile_group(PROFILE, "openarm_left_arm")
    assert list(grp.canonical) == [f"l_aj_{i}" for i in range(1, 8)]
    assert grp.tip.endswith("_palm_ee") or grp.tip.endswith("_tcp")


def test_no_tool_reaches_for_the_dead_builtin_loader():
    """★09.05 함정: 이 로더는 삭제된 manifest 를 요구한다. 새로 쓰지 말 것."""
    offenders = []
    for f in SIM2REAL.glob("scripts/**/*.py"):
        text = f.read_text(encoding="utf-8", errors="replace")
        if "import load_builtin_profile" in text or "load_builtin_profile(" in text:
            offenders.append(f.relative_to(SIM2REAL))
    assert not offenders, (
        f"{offenders} 가 삭제된 manifest 를 요구하는 load_builtin_profile 을 쓴다 — "
        "프로필 yaml 을 직접 읽을 것")

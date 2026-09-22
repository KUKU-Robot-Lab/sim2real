"""실기 미션(config/mission_dg5f_m_control.yaml)의 순서 — 09.22 실기에서 막힌 순서가 다시 들어오지 않게 잠근다."""
from __future__ import annotations

from pathlib import Path

import yaml
from mission_core import load_mission
from mission_stages import load_runbook

PATH = Path(__file__).resolve().parents[2] / "config" / "mission_dg5f_m_control.yaml"
RAW = yaml.safe_load(PATH.read_text(encoding="utf-8"))
MISSION = load_mission(RAW)
BOOK = load_runbook(RAW.get("run"), MISSION)          # 정지 대상 · 순서 검사가 여기서 돈다
IDS = [s.id for s in MISSION.stages]


def _cmds(stage):
    return BOOK.commands[stage]


def _launches(stage, needle):
    return [i for i, c in enumerate(_cmds(stage)) if any(needle in a for a in c.argv)]


def test_every_stage_sits_in_one_of_the_six_groups_in_order():
    assert [g.title for g in MISSION.groups] == ["점검", "연결", "준비", "자세 이동", "정책 동작", "정리"]
    assert all(s.group for s in MISSION.stages)
    assert {g.id for g in MISSION.groups if g.motion} == {"motion", "policy"}


def test_each_arm_is_readied_engaged_and_homed_before_the_selftest():
    for side in ("right", "left"):
        order = [f"pd_load_{side}", f"pd_arm_{side}", f"home_{side}", f"selftest_{side}", f"release_{side}"]
        assert [IDS.index(s) for s in order] == sorted(IDS.index(s) for s in order), side
        assert any(a == "execute:=false" for c in _cmds(f"pd_load_{side}") for a in c.argv)
        arm = _cmds(f"pd_arm_{side}")
        assert arm[0].stop and any(a == "execute:=true" for a in arm[1].argv)          # 무발행을 내린 뒤 발행을 띄운다
        home = [" ".join(c.argv) for c in _cmds(f"home_{side}")]
        assert "--only pd_engage --hold-s 10" in home[0] and _cmds(f"home_{side}")[1].manual and "--only pd_goto_home" in home[2]
        assert f"{side[0]}_aj_4" not in " ".join(_cmds(f"selftest_{side}")[0].argv)    # 차렷에서 하한 0


def test_the_head_publisher_starts_only_after_head_home_released_the_port():
    cmds = _cmds("head_home")
    assert "head_home.py" in cmds[0].argv[-1] and not cmds[0].background
    assert "head_joint_publisher.py" in cmds[1].argv[-1] and cmds[1].background
    assert not any("head_joint_publisher" in a for s in IDS if s != "head_home" for c in _cmds(s) for a in c.argv)


def test_the_left_arm_never_starts_a_second_pd_node_beside_the_right_one():
    assert "pd_arm_right#1" in _cmds("pd_load_left")[0].stop
    assert set(_cmds("release_right")[-1].stop) == {"fabric_direct_right#0", "pd_arm_right#1"}


def test_shutdown_is_last_confirms_support_first_and_drops_the_arm_last():
    assert IDS[-1] == "shutdown"
    cmds = _cmds("shutdown")
    assert cmds[0].manual and "받침" in cmds[0].note
    stops = [c.stop for c in cmds if c.stop]
    arm = _launches("drivers", "openarm.bimanual.launch.py")[0]
    hands = _launches("drivers", "dg5f_driver")
    assert stops[-1] == (f"drivers#{arm}",) and set(stops[-2]) == {f"drivers#{i}" for i in hands}


def test_only_side_stages_and_optional_sensors_can_be_skipped():
    fixed = {s.id for s in MISSION.stages if not s.skippable}
    assert fixed == {"preflight", "drivers", "shutdown"}


def test_only_the_arm_stage_launches_pd_with_the_exec_config():
    # 발행 = launch execute:=true AND yaml execute. 무발행 단계는 원본(false), 발행 전환 단계만 사본(true).
    for side in ("right", "left"):
        load = " ".join(a for c in _cmds(f"pd_load_{side}") for a in c.argv)
        arm = " ".join(a for c in _cmds(f"pd_arm_{side}") for a in c.argv)
        assert "pd_config:={artifact:pd}" in load and "execute:=false" in load
        assert "pd_config:={artifact:pd_exec}" in arm and "execute:=true" in arm
        assert "pd_exec" in MISSION.stages[IDS.index(f"pd_arm_{side}")].artifacts      # 승인 근거 해시에 들어간다
    others = [s for s in IDS if not s.startswith("pd_arm_")]
    assert not any("pd_exec" in a for s in others for c in _cmds(s) for a in c.argv)

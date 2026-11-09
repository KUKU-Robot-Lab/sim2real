"""★261109 모터 무응답 감지 - 드라이버 `comm_age` 로 그 팔 지령을 멈춘다.

181759 에피소드: 오른팔 J1 이 CAN 응답을 멈췄는데 /joint_states 는 마지막 position 을 계속 내서
pd 는 멀쩡하다고 보고 지령을 계속 냈고, home 이 settle 시간 초과로만 끝났다.
"""
from types import SimpleNamespace

from policy_control.pd_arm import COMM_SILENT_SEC, ArmUnit


def _unit():
    return SimpleNamespace(
        arm_joints=["j1", "j2"],
        profile={"j1": {"source": "openarm_right_joint1"}, "j2": {"source": "openarm_right_joint2"}},
        robot_cfg=SimpleNamespace(sources={"arm": SimpleNamespace(stale_sec=0.5)}),
        comm_ages={}, t_comm_recv=None)


def test_silent_joint_is_reported():
    u = _unit()
    ArmUnit.on_comm_age(u, {"openarm_right_joint1": COMM_SILENT_SEC * 10, "openarm_right_joint2": 0.001}, now=1.0)
    assert ArmUnit.silent_joints(u, 1.1) == ["j1"]


def test_all_responding_is_clear():
    u = _unit()
    ArmUnit.on_comm_age(u, {"openarm_right_joint1": 0.001, "openarm_right_joint2": 0.002}, now=1.0)
    assert ArmUnit.silent_joints(u, 1.1) == []


def test_old_driver_without_comm_age_does_not_block():
    """구버전 드라이버(comm_age 없음)는 판정 불가 - 막지 않는다."""
    assert ArmUnit.silent_joints(_unit(), 5.0) == []


def test_stale_comm_report_does_not_block():
    """dynamic_joint_states 자체가 끊기면 이 판정은 끈다(온도 stale 규칙이 따로 본다)."""
    u = _unit()
    ArmUnit.on_comm_age(u, {"openarm_right_joint1": 9.0}, now=1.0)
    assert ArmUnit.silent_joints(u, 2.0) == []

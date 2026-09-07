"""팔 백엔드 없이 손만 pd 로 제어할 수 있어야 한다 (ROS 없음).

2026-09-07 사용자 지시: 손 PD 제어를 완성하는 동안 팔은 필요 없으니 **팔 제어를 끈 채**
진행한다. 지금 `ArmUnit` 은 둘을 묶고 있다 —

  · `_write` 가 `self.backends.arm.write(cmd)` 를 무조건 먼저 부른다.
  · `zero_release` 가 `backends.arm.zero_release()` 를 무조건 부른다.
  · `close`/`switch_*` 가 `backends.switch` 를 전제한다(팔 컨트롤러 교대용).

로봇 yaml 에서 **`arm` 소스는 남기고 `right_arm` 그룹만 빼면** 팔 지령 경로 없이 pd 가 돈다
(q_meas·FSM·워치독은 그대로 필요하다). 손은 자기 controller_manager(TCP)라 컨트롤러 교대도 없다.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
FULL = SIM2REAL / "policy_control/config/robots/dg5f_m_right_real.yaml"
HAND_ONLY = SIM2REAL / "policy_control/config/robots/dg5f_m_right_hand_only.yaml"


def test_hand_only_robot_yaml_exists_and_drops_only_the_arm_group():
    assert HAND_ONLY.exists(), "손 단독 로봇 yaml 이 있어야 한다"
    full, hand = yaml.safe_load(FULL.read_text()), yaml.safe_load(HAND_ONLY.read_text())

    assert set(hand["groups"]) == {"right_hand"}, "팔 그룹만 빠져야 한다"
    assert hand["groups"]["right_hand"] == full["groups"]["right_hand"], "손 그룹은 그대로"
    assert set(hand["sources"]) == set(full["sources"]), "소스는 그대로 — q_meas·워치독에 필요하다"
    assert hand["sources"]["arm"] == full["sources"]["arm"]


@pytest.mark.ros
def test_backends_build_without_an_arm_group(ros):
    from policy_control.contract import load_contract
    from policy_control.pd_arm import build_side_backends, hand_command_joints
    from policy_control.pd_law import load_pd_config
    from policy_control.sources import load_profile, load_robot_cfg

    contract = load_contract(SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json")
    cfg = load_pd_config(SIM2REAL / "policy_control/config/pd_dg5f_m.yaml")
    robot = load_robot_cfg(HAND_ONLY)
    side_cfg = contract.side("right")
    from rclpy.node import Node
    node = Node("hand_only_probe", context=ros)
    b = build_side_backends(node, robot.groups, cfg, side_cfg,
                            hand_command_joints(side_cfg), load_profile(robot.joint_profiles), execute=False)
    try:
        assert b.arm is None and b.switch is None, "팔 백엔드가 없어야 한다"
        assert b.hand is not None, "손 백엔드는 있어야 한다"
    finally:
        if b.hand_gains is not None:
            b.hand_gains.close()
        node.destroy_node()


def test_side_groups_allows_a_subset_but_never_an_undeclared_group():
    """계약이 선언한 그룹 중 **일부만** 제어하는 것은 의도적 선택(손 단독)이다.
    반대로 계약에 없는 그룹을 yaml 이 들고 오는 것은 배선 오류다."""
    from policy_control.contract import load_contract
    from policy_control.pd_arm import PdArmError, side_groups
    from policy_control.sources import load_robot_cfg

    contract = load_contract(SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json")
    side_cfg = contract.side("right")

    full = side_groups(load_robot_cfg(FULL), side_cfg)
    assert set(full) == {"right_arm", "right_hand"}

    hand = side_groups(load_robot_cfg(HAND_ONLY), side_cfg)
    assert set(hand) == {"right_hand"}, "부분집합은 허용된다"

    # ★"이 팔 소속"은 groups.<g>.side 로 표시된 것뿐이다 — 양팔 yaml 은 반대편 그룹도 담는다.
    class _Mine:
        groups = {"right_hand": {"backend": "dg5f_jtc", "side": "right"},
                  "mystery_group": {"backend": "arm_forward", "side": "right"}}
    with pytest.raises(PdArmError):
        side_groups(_Mine(), side_cfg)         # 이 팔 소속인데 계약에 없다 → 거부

    class _OtherSide:
        groups = {"right_hand": {"backend": "dg5f_jtc", "side": "right"},
                  "left_arm": {"backend": "arm_forward", "side": "left"}}
    assert set(side_groups(_OtherSide(), side_cfg)) == {"right_hand"}, "반대편 팔 그룹은 무시한다"


# ---------------------------------------------------------------- 팔 상태 없이도 손이 나간다
@pytest.mark.ros
def test_hand_only_does_not_require_arm_joint_state(ros):
    """손 드라이버만 떠 있어도 손 지령이 나가야 한다.

    2026-09-07: 손 단독 pd 가 `right: joint state missing ['arm']` 로 멈췄다. pd 는 팔 **지령**이
    아니라 팔 **관절 상태**를 FSM·추종오차에 쓰는데, 팔 백엔드가 없으면 그 추종오차는 우리가
    판정할 대상이 아니다(팔은 JTC 나 전원 차단 상태다). 팔 백엔드가 없으면 arm 소스를 요구하지
    않고, q_meas 는 목표를 그대로 써서 팔 법칙을 무해한 통과로 만든다.
    """
    import numpy as np
    from rclpy.node import Node

    from policy_control.contract import load_contract
    from policy_control.pd_arm import ArmUnit
    from policy_control.pd_law import load_pd_config
    from policy_control.sources import load_profile, load_robot_cfg

    node = Node("hand_only_unit", context=ros)
    try:
        contract = load_contract(SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json")
        robot = load_robot_cfg(HAND_ONLY)
        unit = ArmUnit(node, side_cfg=contract.side("right"), robot_cfg=robot,
                       cfg=load_pd_config(SIM2REAL / "policy_control/config/pd_dg5f_m.yaml"),
                       contract=contract, profile=load_profile(robot.joint_profiles),
                       stage_name="reduced", execute=False)
        state = unit.sources.snapshot(0.0)
        q, qd, why = unit._measured(state)
        assert why is None or "arm" not in why, f"팔 백엔드가 없으면 arm 결손을 이유로 멈추면 안 된다: {why}"
    finally:
        unit_close = getattr(locals().get("unit", None), "close", None)
        if unit_close: unit_close()
        node.destroy_node()

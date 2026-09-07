"""제어 전용(direct) 계약에서 fabric 리셋 홈은 **실측 자세**여야 한다 (ROS 없음).

정책 모드에서 fabric 의 리셋 q 는 계약 `side.fabric.home_q` — 곧 **액션의 default_config**
이자 cspace rest 다. 정책이 그 자세에서 출발하도록 학습됐으므로 옳다.

제어 전용 계약에는 정책이 없다. 그런데도 홈을 계약값(차렷)으로 잡으면, 팔이 다른 자세에
있을 때 fabric 이 **차렷 기준으로** 손바닥 위치를 계산한다. 2026-09-07 우팔 실기: 팔이
preset(팔꿈치 113.6°)에 있는데 fabric 의 palm_pose 가 차렷 위치를 가리켰고, fabric 이 낸
joint_target 이 실측과 **최대 96.5° (팔꿈치)** 어긋났다. pd 가 워치독 HOLD 라 안 끌려갔을 뿐,
engage 상태였다면 팔이 96° 스윙했다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
ASSET = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
LEFT = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="자산 계약 없음")
needs_left = pytest.mark.skipif(not LEFT.exists(), reason="정책 계약 없음")


def _node_mod():
    # ★파일 경로 로드(spec_from_file_location)는 sys.modules 에 등록되지 않아, 그 모듈이 정의한
    #   dataclass 가 자기 모듈을 못 찾고 죽는다. 패키지로 정상 import 한다.
    from policy_control import fabric_node
    return fabric_node


@needs_asset
def test_control_only_home_follows_the_measured_arm():
    from policy_control.contract import load_contract

    mod, c = _node_mod(), load_contract(ASSET)
    measured = np.radians([0.45, 17.41, -0.25, 113.56, -0.84, -0.23, 0.64])   # 실기 preset 실측
    home = mod.home_from_event({}, c, side="right", measured_arm_q=measured)

    assert np.allclose(home[:7], measured, atol=1e-9), (
        "제어 전용에서는 실측이 홈이어야 한다 — 아니면 fabric 이 차렷 기준으로 IK 를 푼다")
    assert len(home) == len(c.side("right").fabric.home_q)
    assert np.allclose(home[7:], c.side("right").fabric.home_q[7:]), "손 몫은 계약값 그대로"


@needs_asset
def test_without_measured_it_falls_back_to_the_contract():
    from policy_control.contract import load_contract

    mod, c = _node_mod(), load_contract(ASSET)
    home = mod.home_from_event({}, c, side="right")
    assert np.allclose(home, c.side("right").fabric.home_q)


@needs_left
def test_policy_contract_keeps_the_trained_cspace_rest():
    """★정책 계약은 절대 실측으로 바꾸지 않는다 — 학습된 default_config 가 홈이다."""
    from policy_control.contract import load_contract

    mod, c = _node_mod(), load_contract(LEFT)
    side = c.primary_side
    measured = np.full(len(c.side(side).arm_joints), 0.3)
    home = mod.home_from_event({}, c, side=side, measured_arm_q=measured)
    assert np.allclose(home, c.side(side).fabric.home_q), "정책 계약에서 실측을 홈으로 쓰면 안 된다"

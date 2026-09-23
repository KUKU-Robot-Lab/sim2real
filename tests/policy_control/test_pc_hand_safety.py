"""손 지령 안전 규칙 — 한계에서 물러난 목표 · 실측을 앞서지 않는 세트포인트 (ROS 없음, 순수 함수).

09.23 실기: `pd_hand_home` 이 계약의 손 홈(굽힘 관절 **하한 0.0 그 자체**)으로 밀어 손가락이 꺾였다.
드라이버가 error 423 을 내고 손이 0.47 rad 에서 멈췄는데도 세트포인트는 0 까지 전진했다 —
속도 제한이 **직전 세트포인트** 기준이라 손이 막힌 것을 보지 않는다.
"""
from __future__ import annotations

import numpy as np
import pytest

from policy_control.pd_backends import (HAND_LIMIT_MARGIN, HAND_MAX_LEAD, hand_lead_clamp, hand_safe_target)

pytestmark = pytest.mark.unit

# 벤더 프로파일(openarm_tesollo.yaml)의 실제 한계 — 굽힘 관절은 하한이 정확히 0 이다
LOWER = np.array([0.0, 0.0, 0.0, -2.70526])
UPPER = np.array([2.007129, 2.007129, 2.007129, 0.0])
HOME = np.array([0.0, 0.0, 0.0, -1.57])          # 계약 home_hand — 손가락 0, 엄지 −1.57


def test_the_contract_home_is_pulled_inside_the_joint_limits():
    safe = hand_safe_target(HOME, LOWER, UPPER)
    assert np.all(safe >= LOWER + HAND_LIMIT_MARGIN - 1e-12)
    assert np.all(safe <= UPPER - HAND_LIMIT_MARGIN + 1e-12)
    assert safe[0] == pytest.approx(HAND_LIMIT_MARGIN)        # 0.0 → 0.05 (기계 끝점에서 물러난다)
    assert safe[3] == pytest.approx(-1.57)                    # 한계 안쪽 값은 그대로


def test_a_target_far_inside_the_limits_is_untouched():
    q = np.array([1.0, 0.5, 1.5, -1.0])
    assert np.allclose(hand_safe_target(q, LOWER, UPPER), q)


def test_a_range_narrower_than_the_margin_falls_back_to_the_middle():
    lo, hi = np.array([0.0]), np.array([0.05])                # 여유 두 배보다 좁다
    assert hand_safe_target(np.array([0.0]), lo, hi) == pytest.approx(0.025)


def test_a_stalled_finger_stops_the_setpoint_instead_of_pushing_harder():
    # 손이 0.47 에서 막혀 있는데 세트포인트만 0 으로 가는 상황(09.23 실기)
    stalled = np.array([0.47, 0.47, 0.47, -1.2])
    marched = np.array([0.0, 0.0, 0.0, -1.57])
    held = hand_lead_clamp(marched, stalled)
    assert np.all(held >= stalled - HAND_MAX_LEAD - 1e-12)
    assert held[0] == pytest.approx(0.47 - HAND_MAX_LEAD)     # 미는 힘이 0.2 rad 어치로 묶인다
    assert held[3] == pytest.approx(-1.2 - HAND_MAX_LEAD)


def test_a_following_hand_is_not_slowed_down():
    meas = np.array([0.50, 0.50, 0.50, -1.30])
    cmd = np.array([0.45, 0.45, 0.45, -1.35])                 # 실측을 0.05 만 앞선다
    assert np.allclose(hand_lead_clamp(cmd, meas), cmd)


def test_without_a_measurement_the_clamp_does_nothing():
    cmd = np.array([0.1, 0.2])
    assert np.allclose(hand_lead_clamp(cmd, None), cmd)

"""RH56F1 손 행동 법칙 — 학습 코드(hdgp rh_aglt hand_action.py)와 수치가 같은가. 09.30 사용자: 손 액션 구조를 명확히."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from policy_control import rh56f1_hand as H

HDGP_HAND = Path.home() / "rl_ws/hdgp/source/openarm/openarm/agnostic/tasks/rh_aglt_r/hand_action.py"
OPEN = (1.57, 0.0, 0.0, 0.0, 0.0, 0.0)
GRIP = (1.20, 0.24, 1.08, 1.08, 0.85, 0.85)
LO = (0.0,) * 6
HI = (2.0943951, 0.474555, 1.5285594, 1.5285594, 1.5285594, 1.5285594)


def _law(**kw) -> H.HandLaw:
    base = dict(q_open=OPEN, q_grip=GRIP, lim_lo=LO, lim_hi=HI, range_mode="grip", ema=0.1, full_range_s=1.0,
                policy_hz=60.0, freeze=True, freeze_joints=(True,) * 6, freeze_threshold_n=1.0, hold="open")
    base.update(kw)
    return H.HandLaw(**base)


@pytest.mark.skipif(not HDGP_HAND.is_file(), reason="hdgp 없음")
def test_same_numbers_as_the_training_hand_action_for_random_rollouts():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("_hdgp_hand_action", HDGP_HAND)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    law = _law()
    q_open, q_grip = torch.tensor(OPEN), torch.tensor(GRIP)
    cap = torch.tensor(law.cap(), dtype=torch.float32)
    rng = np.random.default_rng(0)
    prev_np, prev_t = np.array(OPEN), q_open.unsqueeze(0).clone()
    for _ in range(300):
        a = rng.uniform(-1.3, 1.3, 6)                              # ±1 밖도 — 둘 다 자른다
        f = rng.uniform(0.0, 2.0, 5)
        freeze = law.touch_to_freeze(f)
        prev_np = law.step(prev_np, a, active=True, freeze=freeze)
        raw = ref.linear_grip_raw(torch.tensor(a, dtype=torch.float32).unsqueeze(0), q_open, q_grip)
        prev_t = ref.step_hand_target(prev_t, raw, alpha=0.1, cap=cap, freeze=torch.tensor(freeze).unsqueeze(0),
                                      q_open=q_open, q_grip=q_grip)
        np.testing.assert_allclose(prev_np, prev_t[0].numpy(), atol=1e-5)


def test_thumb_abduction_opens_at_plus_one_and_the_others_close():
    law = _law()
    raw = law.raw(np.ones(6))
    assert raw[0] == pytest.approx(1.57) and raw[1:] == pytest.approx(GRIP[1:])
    assert law.raw(np.zeros(6)) == pytest.approx((np.array(OPEN) + np.array(GRIP)) / 2)


def test_rate_cap_is_the_full_joint_range_in_full_range_s():
    law = _law(ema=1.0)
    q = law.step(np.array(OPEN), -np.ones(6) * -1, active=True)          # 한 스텝에 가려 해도
    assert np.all(np.abs(q - np.array(OPEN)) <= np.array(HI) / 60.0 + 1e-12)


def test_a_touching_finger_stops_closing_but_may_open():
    law = _law(ema=1.0)
    mid = (np.array(OPEN) + np.array(GRIP)) / 2
    freeze = law.touch_to_freeze([0, 5.0, 0, 0, 0])                   # 검지만 닿음
    closed = law.step(mid, np.ones(6), active=True, freeze=freeze)      # 닫으려 함
    assert closed[2] == pytest.approx(mid[2]) and closed[3] > mid[3]
    opened = law.step(mid, -np.ones(6), active=True, freeze=freeze)     # 펴는 것은 된다
    assert opened[2] < mid[2]


def test_hold_open_pins_the_hand_and_hold_follow_does_not():
    a = np.ones(6)
    assert _law(hold="open").step(np.array(GRIP), a, active=False) == pytest.approx(OPEN)
    moved = _law(hold="follow").step(np.array(OPEN), a, active=False)
    assert not np.allclose(moved, OPEN)


def test_limits_range_and_no_freeze_for_the_old_pour_fj_runs():
    law = _law(range_mode="limits", freeze=False)
    lo, hi = law.bounds()
    assert lo == pytest.approx(LO) and hi == pytest.approx(HI)
    assert law.touch_to_freeze([9, 9, 9, 9, 9]) is None


def test_tactile_obs_matches_training_clip_and_tanh():
    assert H.tactile_obs([-1, 0, 3, 10, 50], 10.0, 3.0) == pytest.approx(np.tanh(np.array([0, 0, 3, 10, 10]) / 3.0))


def test_bad_laws_are_refused():
    with pytest.raises(H.HandActionError):
        _law(range_mode="synergy")
    with pytest.raises(H.HandActionError):
        _law(q_grip=(3.0, 0, 0, 0, 0, 0))
    with pytest.raises(H.HandActionError):
        _law().step(np.array(OPEN), np.ones(5), active=True)


def test_open_floor_lifts_only_the_four_fingers_lower_bound():
    """hdgp pour_fabric_mimic side_rig(09.30): 실기 레지스터 1740 까지만 펴진다 → 네 손가락 목표 하한 0.065 rad."""
    lo, hi = _law(finger_open_floor=0.065).bounds()
    lo0, hi0 = _law().bounds()
    assert lo[2:] == pytest.approx([0.065] * 4) and lo[:2] == pytest.approx(lo0[:2]) and hi == pytest.approx(hi0)
    assert _law(finger_open_floor=0.065).raw(-np.ones(6))[2:] == pytest.approx([0.065] * 4)


def test_real_speed_cap_replaces_the_full_range_cap():
    caps = (2.1, 0.56, 2.1, 2.1, 2.1, 2.1)
    assert _law(vel_cap_rad_s=caps).cap() == pytest.approx(np.array(caps) / 60.0)
    q = _law(vel_cap_rad_s=caps, freeze=False).step(np.array(OPEN), np.ones(6), active=True)
    assert q[1] - OPEN[1] == pytest.approx(0.56 / 60.0)            # 엄지 굽힘은 따로 느리다


def test_bad_floor_and_speed_caps_are_refused():
    with pytest.raises(H.HandActionError):
        _law(vel_cap_rad_s=(2.1,) * 5)
    with pytest.raises(H.HandActionError):
        _law(vel_cap_rad_s=(2.1, 0.0, 2.1, 2.1, 2.1, 2.1))
    with pytest.raises(H.HandActionError):
        _law(finger_open_floor=-0.1)

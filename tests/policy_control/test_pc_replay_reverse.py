"""기록 궤적을 되짚어 내려온다 — preset 복귀는 goto_home 이 아니다 (ROS 없음).

★2026-09-07 실기 사고. preset(팔이 테이블 위로 뻗은 자세)에서 `pd/goto_home` 을 불렀다.
goto_home 은 계약 홈을 향한 **관절공간 직선 램프**라 관절 7개가 각자 목표로 직선으로 가고,
그 사이 손이 어디를 지나는지 아무도 계산하지 않는다. 사용자가 모터를 강제 종료했다.
[[replay-bridge-ramp-ignores-table]] 에 09.03 실충돌로 이미 적어둔 함정이었다.

올라온 길을 거꾸로 가면 **지나온 공간만** 지난다. 그래서 --reverse 다.

되짚기의 유일한 무보호 구간은 진입 램프(실측 → 첫 프레임)이고, 이것도 관절공간 직선이다.
되짚기는 "왔던 자리에 서 있다"가 전제이므로 그 간극이 크면 전제가 깨진 것이고, 램프로
메우는 대신 거부한다. 정방향(차렷 → preset)은 멀리서 시작하는 것이 정상이라 걸지 않는다.
"""
from __future__ import annotations

import numpy as np
import pytest

from policy_control import _paths  # noqa: F401
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "replay_to_pd", Path(__file__).resolve().parents[2] / "policy_control/tools/replay_to_pd.py")
R = importlib.util.module_from_spec(spec)
import sys
sys.modules["replay_to_pd"] = R
spec.loader.exec_module(R)

pytestmark = pytest.mark.unit

# 차렷(0) → 뻗은 자세로 올라가는 5 프레임짜리 기록
FRAMES = np.array([[0.0, 0.0], [0.25, 0.1], [0.50, 0.2], [0.75, 0.3], [1.00, 0.4]])
DT = 0.02


def test_reverse_walks_the_recording_backwards():
    plan, vel, ramp = R.build_plan(FRAMES, start=FRAMES[-1].copy(), pub_dt=DT, reverse=True)
    body = plan[ramp:]
    assert np.allclose(body, FRAMES[::-1])
    assert np.allclose(body[0], FRAMES[-1]) and np.allclose(body[-1], FRAMES[0])


def test_forward_is_unchanged():
    plan, vel, ramp = R.build_plan(FRAMES, start=np.zeros(2), pub_dt=DT, reverse=False)
    assert np.allclose(plan[ramp:], FRAMES)


def test_the_ramp_carries_no_velocity_feedforward_in_either_direction():
    for rev, start in ((True, FRAMES[-1].copy()), (False, np.zeros(2))):
        _, vel, ramp = R.build_plan(FRAMES, start=start, pub_dt=DT, reverse=rev)
        assert np.allclose(vel[:ramp], 0.0)


def test_reverse_refuses_to_start_far_from_where_the_recording_ended():
    """되짚기의 전제는 '왔던 자리에 서 있다' — 아니면 램프가 관절공간 직선으로 테이블을 지난다."""
    far = FRAMES[-1] + np.array([0.6, 0.0])
    with pytest.raises(SystemExit) as e:
        R.build_plan(FRAMES, start=far, pub_dt=DT, reverse=True)
    assert "reverse" in str(e.value) and "0.6" in str(e.value)


def test_reverse_allows_a_small_gap_because_the_arm_droops():
    near = FRAMES[-1] + np.array([0.10, -0.05])
    plan, _, ramp = R.build_plan(FRAMES, start=near, pub_dt=DT, reverse=True)
    assert ramp >= 1 and np.allclose(plan[ramp:], FRAMES[::-1])


def test_forward_may_start_anywhere_the_ramp_is_the_point():
    plan, _, ramp = R.build_plan(FRAMES, start=np.array([-1.5, 2.0]), pub_dt=DT, reverse=False)
    assert ramp > 1 and np.allclose(plan[ramp:], FRAMES)

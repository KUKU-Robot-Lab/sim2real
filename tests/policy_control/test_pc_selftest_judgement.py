"""셀프테스트 판정식 — 간섭은 직전 자세 기준, 크기는 관절 강성을 감안한다 (ROS 없음).

2026-09-07 우팔 실기에서 드러난 판정식 결함 둘:

① **간섭이 최초 기준자세와 비교된다.** 어떤 관절이 초반에 한 번 옮겨가 그대로 머물면,
   그 상수 오프셋이 **이후 모든 스텝에서 간섭으로 잡힌다**(실측: r_aj_5 가 28개 스텝 내내
   +0.0286 rad 로 동일하게 보고됨). 간섭은 "이 스텝 때문에 다른 관절이 움직였나"이므로
   **직전 자세** 기준이어야 한다.

② **작은 진폭에서 비율 대역이 도달 불가능하다.** 마찰·중력모델 오차로 생기는 정상상태
   편차는 진폭과 무관한 상수(실측 j1 ≈ 0.014 rad)다. 그래서 같은 로봇이 ±0.10 에서 0.86,
   ±0.02 에서 0.27 로 나온다. kp 10 인 손목은 ±0.05 에서도 대역(0.4~1.6)을 못 채운다.
   ⇒ 비율 대신 **잔류 편차 |실측 − 명령|** 을 관절별 허용치와 비교하는 경로를 둔다.
"""
from __future__ import annotations

import numpy as np
import pytest

from lowlevel_check_core import StepSpec, evaluate_step

pytestmark = pytest.mark.unit

J = [f"r_aj_{i}" for i in range(1, 8)]


def test_a_standing_offset_is_not_reported_as_crosstalk():
    """r_aj_5 가 스텝 전부터 +0.03 쏠려 있고 이 스텝으로는 안 움직였다 → 간섭 아님."""
    base = np.zeros(7)
    prev = base.copy(); prev[4] = 0.03                 # 최초 기준 대비 상수 오프셋
    end = prev.copy(); end[0] = 0.09                   # 이번 스텝으로 j1 만 움직였다
    spec = StepSpec("step", "r_aj_1", 0.10, 2.0)

    v = evaluate_step(J, base, end, spec, prev_q=prev)
    assert v.crosstalk < 0.01, f"직전 자세 기준이면 간섭 0 이어야 한다 (got {v.crosstalk})"
    assert v.ok, v.reason


def test_real_crosstalk_still_caught():
    base = np.zeros(7)
    prev = base.copy(); prev[4] = 0.03
    end = prev.copy(); end[0] = 0.09; end[4] += 0.02   # 이번 스텝에서 j5 가 실제로 움직였다
    v = evaluate_step(J, base, end, spec := StepSpec("step", "r_aj_1", 0.10, 2.0), prev_q=prev)
    assert not v.ok and "간섭" in v.reason, v.reason


def test_prev_q_defaults_to_base_so_old_callers_are_unchanged():
    base = np.zeros(7)
    end = base.copy(); end[0] = 0.09
    a = evaluate_step(J, base, end, StepSpec("step", "r_aj_1", 0.10, 2.0))
    b = evaluate_step(J, base, end, StepSpec("step", "r_aj_1", 0.10, 2.0), prev_q=base)
    assert a == b


def test_small_amplitude_passes_on_residual_when_the_joint_is_soft():
    """kp 10 인 손목: ±0.05 명령에 0.036 이동(비율 0.72)은 대역 밖이 아니지만,
    ±0.02 에 0.006(비율 0.30)은 대역 밖이다 — 잔류 0.014 rad 는 같다."""
    base = np.zeros(7)
    end = base.copy(); end[6] = 0.006
    spec = StepSpec("step", "r_aj_7", 0.02, 2.0)

    strict = evaluate_step(J, base, end, spec)
    assert not strict.ok and "크기" in strict.reason

    lenient = evaluate_step(J, base, end, spec, residual_tol_rad=0.016)
    assert lenient.ok, f"잔류 {abs(0.006-0.02):.4f} rad ≤ 허용 0.016 이면 통과해야 한다: {lenient.reason}"


def test_residual_path_does_not_excuse_a_wrong_sign():
    base = np.zeros(7)
    end = base.copy(); end[6] = -0.006
    v = evaluate_step(J, base, end, StepSpec("step", "r_aj_7", 0.02, 2.0), residual_tol_rad=0.05)
    assert not v.ok and "부호" in v.reason, v.reason

"""goto_home 정착 적분 — `model_tau_ff` 에서도 정상상태 처짐을 메운다 (ROS 없음, 순수 함수).

09.23 실기: 오른팔이 경로 끝(계약 홈)에 닿은 뒤 goto_home 이 30 s timeout 으로 거부됐다.
`kp(q*−q) + τ_ff ≡ kp(ref−q)` 항등 때문에 평형에서 처짐 τ/kp 가 남고(j7 2.06 N·m / kp 10 = 0.21 rad),
정착 보정이 `integral_droop` 계약에서만 돌아 그 처짐을 메울 수단이 없었다.
"""
from __future__ import annotations

import numpy as np
import pytest

from policy_control.pd_law import Settle, settle_bias, settle_limit

pytestmark = pytest.mark.unit

KP = np.array([70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0])      # 벤더 control_gains.yaml (팔 7)
TAU_G = np.array([-2.65, 2.86, 3.10, 4.04, 0.13, 0.13, 2.06])  # 09.23 실측 정지 토크 [N·m]
HOME = np.array([-1.1974, 0.6707, 0.1866, 1.731, 0.692, 0.0416, 0.946])
GAIN, CLAMP, CLAMP_NM = 0.01, 0.30, 5.0


def _measured(bias):
    """준정적 평형: 모터 토크 = kp·(home + bias − q) = τ_중력 → q = home + bias − τ/kp."""
    return HOME + bias - TAU_G / KP


def test_the_droop_the_real_arm_showed_is_exactly_tau_over_kp():
    q = _measured(np.zeros(7))
    assert np.abs(q - HOME).max() == pytest.approx(0.206, abs=0.001)        # j7
    assert np.argmax(np.abs(q - HOME)) == 6


def test_the_integral_brings_every_joint_inside_the_settle_tolerance():
    bias, tol = np.zeros(7), 0.01
    for _ in range(3000):                                   # 100 Hz · 30 s 창
        q = _measured(bias)
        if np.abs(q - HOME).max() < tol:
            break
        bias = settle_bias(bias, HOME, q, KP, gain=GAIN, clamp=CLAMP, clamp_nm=CLAMP_NM)
    else:
        pytest.fail(f"30 s 안에 정착하지 못했다 — 남은 오차 {np.abs(_measured(bias) - HOME).max():.4f}")
    assert np.abs(_measured(bias) - HOME).max() < tol
    assert np.abs(bias - TAU_G / KP).max() < tol            # bias 가 처짐을 그대로 메운다


def test_it_settles_within_five_seconds():
    bias = np.zeros(7)
    for _ in range(500):                                    # 100 Hz · 5 s
        bias = settle_bias(bias, HOME, _measured(bias), KP, gain=GAIN, clamp=CLAMP, clamp_nm=CLAMP_NM)
    assert np.abs(_measured(bias) - HOME).max() < 0.01


def test_the_limit_is_a_torque_budget_so_stiff_joints_cannot_add_a_big_angle():
    lim = settle_limit(KP, CLAMP, CLAMP_NM)
    assert lim[0] == pytest.approx(5.0 / 70.0) and lim[3] == pytest.approx(5.0 / 60.0)
    assert lim[6] == pytest.approx(CLAMP)                   # 손목(kp 10)은 각도 상한이 먼저 걸린다
    assert np.all(lim * KP <= CLAMP_NM + 1e-9) or lim[6] * KP[6] == pytest.approx(3.0)
    bias = settle_bias(np.zeros(7), HOME, HOME - 10.0, KP, gain=1.0, clamp=CLAMP, clamp_nm=CLAMP_NM)
    assert np.allclose(bias, lim)                           # 큰 오차에도 상한을 넘지 않는다


def test_gain_zero_keeps_the_old_behaviour():
    bias = settle_bias(np.ones(7) * 0.05, HOME, HOME - 1.0, KP, gain=0.0, clamp=CLAMP, clamp_nm=CLAMP_NM)
    assert np.allclose(bias, 0.05)


def test_settle_block_defaults_keep_old_configs_loadable():
    s = Settle(clamp=0.12, tol=0.01)
    assert s.gain == 0.0 and s.clamp_nm == 0.0              # 옛 pd yaml 은 보정 없음 그대로


# ---------------------------------------------------------------- 10.04 실기: 정책 뒤 goto_home 이 HOLD(한계 밖 목표)
def _unit(setpoint, lower, upper):
    """ArmUnit._settle 이 보는 것만 가진 가짜 팔(ROS 없음)."""
    from types import SimpleNamespace as NS

    from policy_control.pd_state import Phase
    return NS(phase=Phase.TRACKING, kp=KP, lower=np.asarray(lower, float), upper=np.asarray(upper, float),
              cfg=NS(settle=NS(tol=0.01, gain=GAIN, clamp=CLAMP, clamp_nm=CLAMP_NM)),
              side_cfg=NS(gravity=NS(mode="model_tau_ff", gain=None)),
              stage=NS(state=NS(law=NS(q_setpoint=np.asarray(setpoint, float)))))


RH_HOME = np.array([-1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646])        # rh56f1_aglt 오른팔 홈
RH_LO = np.array([-1.39626, -0.174533, -1.5708, 0.0, -1.5708, -0.785398, -1.5708])   # openarm_rh56f1 프로필
RH_HI = np.array([3.49066, 3.31612, 1.5708, 2.44346, 1.5708, 0.785398, 1.5708])


def test_no_settle_integration_while_the_ramp_is_still_travelling_home():
    """정책이 팔을 0.61 rad 끌고 간 뒤 goto_home: 램프 중 적분하면 손목 bias 가 0.3 까지 차 홈 + bias 가 한계를 0.05 넘게
    지나 'joint limit: target outside profile bounds' HOLD 였다(return_right 실패). 세트포인트가 홈에 닿을 때까지는 적분하지 않는다."""
    from policy_control.pd_arm import ArmUnit, Hold
    far = RH_HOME - np.array([0, 0, 0, 0.61, 0, 0.3, 0])                 # 팔은 아직 멀리
    unit = _unit(setpoint=far + 0.05, lower=RH_LO, upper=RH_HI)          # 램프 세트포인트도 아직 홈 전
    hold = Hold(q=RH_HOME.copy(), hand=None, bias=np.zeros(7), settle=True)
    for _ in range(500):
        hold = ArmUnit._settle(unit, hold, far)
    assert np.allclose(hold.bias, 0.0) and not hold.settled


def test_settle_bias_never_pushes_the_target_past_the_joint_limits():
    """세트포인트가 홈에 닿은 뒤 처짐을 메우되, 홈 + bias 는 한계 안 — 손목 j6 홈 0.5785 · 한계 0.7854 면 bias ≤ 0.2069
    (상한 0.3 이 아니라)."""
    from policy_control.pd_arm import ArmUnit, Hold
    unit = _unit(setpoint=RH_HOME, lower=RH_LO, upper=RH_HI)
    stuck = RH_HOME - np.array([0, 0, 0, 0, 0, 0.4, 0])                  # j6 가 걸려 홈에 못 간다
    hold = Hold(q=RH_HOME.copy(), hand=None, bias=np.zeros(7), settle=True)
    for _ in range(3000):
        unit.stage.state.law.q_setpoint = hold.q + hold.bias             # 세트포인트는 목표를 따라간다
        hold = ArmUnit._settle(unit, hold, stuck)
    target = hold.q + hold.bias
    assert np.all(target <= RH_HI + 1e-12) and np.all(target >= RH_LO - 1e-12)
    assert hold.bias[5] == pytest.approx(RH_HI[5] - RH_HOME[5])          # 한계까지만 민다
    assert hold.bias[5] > 0.1                                            # 적분은 실제로 돈다

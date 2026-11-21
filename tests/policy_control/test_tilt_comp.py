"""★11.16 tilt_comp: payload ff 가 with_payload 체인 중력 차와 같고, 적분은 deadband · windup · reset 을 지킨다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control import pd_law as L
from policy_control.tilt_comp import TiltComp, TiltCompCfg

URDF = Path(__file__).resolve().parents[3] / "hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
JOINTS = [f"l_aj_{i}" for i in range(1, 8)]


@pytest.mark.skipif(not URDF.exists(), reason="hdgp URDF 없음")
def test_payload_ff_matches_with_payload_chain():
    from robot_control.kinematics import chain_from_urdf, with_payload
    chain = chain_from_urdf(URDF.read_text(), JOINTS, "l_hl_palm_sensor")
    pay = (0.10, 0.0174, 0.0167, 0.1945)
    tc = TiltComp(TiltCompCfg(payload=pay), chain)
    q = np.array([0.3, 0.2, -0.1, 1.2, 0.4, -0.3, 0.2])
    expect = with_payload(chain, pay[0], list(pay[1:])).gravity_torque(q) - chain.gravity_torque(q)
    np.testing.assert_allclose(tc.step(q, q, 0.004), expect, atol=1e-9)


def test_integral_deadband_windup_reset():
    cfg = TiltCompCfg(ki=(0, 0, 0, 10.0, 0, 0, 0), limit=(0, 0, 0, 0.5, 0, 0, 0), deadband_deg=0.15)
    tc = TiltComp(cfg, chain=None)
    q = np.zeros(7)
    small = q.copy(); small[3] = np.deg2rad(0.1)            # deadband 안 → 적분 안 쌓임
    assert not tc.step(q, small, 0.01).any()
    big = q.copy(); big[3] = np.deg2rad(1.15)
    out = tc.step(q, big, 0.01)
    assert out[3] == pytest.approx(10.0 * big[3] * 0.01) and not np.delete(out, 3).any()
    for _ in range(10000):
        out = tc.step(q, big, 0.01)
    assert out[3] == pytest.approx(0.5)                        # windup 한계
    tc.reset()
    assert not tc.integral.any() and not tc.active


def test_config_parse_and_reject():
    assert L._tilt_comp_block(None) is None
    blk = L._tilt_comp_block({"left": {"payload": [0.1, 0, 0, 0.19], "ki": [0, 0, 0, 3, 0, 0, 0],
                                       "limit": [0, 0, 0, 0.6, 0, 0, 0]}})
    assert blk["left"].ki[3] == 3.0
    for bad in ({"middle": {}}, {"left": {"kp": 1}}, {"left": {"payload": [5, 0, 0, 0]}},
                {"left": {"ki": [-1] * 7}}):
        with pytest.raises(L.PdConfigError):
            L._tilt_comp_block(bad)
    with pytest.raises(ValueError):
        TiltComp(TiltCompCfg(payload=(0.1, 0, 0, 0)), chain=None)


def test_runtime_payload_change_keeps_com_and_refuses_bad_cases():
    """★11.17 실행 중 payload kg 변경 — 무게중심 그대로 · 켜진 구간 · 범위 밖 · 체인 없음은 거부(값 그대로)."""
    chain = object()                                          # set_payload_kg 는 체인 유무만 본다
    tc = TiltComp(TiltCompCfg(payload=(0.117, 0.0174, 0.0167, 0.1945)), chain)
    tc.set_payload_kg(0.0)
    assert tc.cfg.payload == (0.0, 0.0174, 0.0167, 0.1945) and not tc._theta.any()
    tc.set_payload_kg(0.2)
    np.testing.assert_allclose(tc._theta, [0.2, 0.2 * 0.0174, 0.2 * 0.0167, 0.2 * 0.1945])
    for bad in (-0.01, 0.51, float("nan")):
        with pytest.raises(ValueError):
            tc.set_payload_kg(bad)
    tc.active = True
    with pytest.raises(ValueError, match="켜진 구간"):
        tc.set_payload_kg(0.0)
    assert tc.cfg.payload[0] == 0.2
    with pytest.raises(ValueError, match="체인"):
        TiltComp(TiltCompCfg(), chain=None).set_payload_kg(0.1)
    TiltComp(TiltCompCfg(), chain=None).set_payload_kg(0.0)     # 0 은 체인 없이도 된다(적분만)


def test_arm_unit_refuses_payload_change_while_tracking():
    from types import SimpleNamespace

    from policy_control.pd_arm import ArmUnit
    from policy_control.pd_state import Phase
    tc = TiltComp(TiltCompCfg(payload=(0.117, 0, 0, 0.19)), object())
    unit = SimpleNamespace(side="left", phase=Phase.TRACKING, tilt_comp=tc)
    with pytest.raises(ValueError, match="TRACKING"):
        ArmUnit.set_tilt_payload(unit, 0.0)
    unit.phase = Phase.IDLE
    ArmUnit.set_tilt_payload(unit, 0.0)
    assert ArmUnit.tilt_payload(unit) == [0.0, 0.0, 0.0, 0.19]
    with pytest.raises(ValueError, match="블록"):
        ArmUnit.set_tilt_payload(SimpleNamespace(side="left", phase=Phase.IDLE, tilt_comp=None), 0.0)

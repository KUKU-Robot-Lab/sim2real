"""place_jog_exec: 유지 보정(발행 = 명령 + c, c = q_set − 실측) · 120 Hz · ≤ 0.30 rad/s / plan_place_fixed 구멍 xy 경고만."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"))
import place_jog_exec as J  # noqa: E402
import plan_place_fixed as PPF  # noqa: E402


def test_jog_stream_rides_pd_setpoint_so_small_jog_survives():
    meas = np.array([0.1, -0.2, 0.3, 0.4, 0.0, 0.1, -0.1])
    q_set = meas + np.array([0.004, -0.006, 0.0, 0.003, 0.0, -0.002, 0.001])   # 정상 오차 > Δq
    dq = np.full(7, 0.002)                                                     # ~1 mm 급 jog
    q, qd = J.jog_stream(meas, q_set, meas + dq)
    assert np.allclose(q[0], q_set) and np.allclose(q[-1], q_set + dq)        # 튐 없이 시작 · 세트포인트 기준으로 Δq 전부
    assert np.allclose(qd[0], 0) and np.allclose(qd[-1], 0)
    assert np.abs(np.diff(q, axis=0)).max() * J.PUB_HZ <= J.MAX_VEL + 1e-9
    big = J.jog_stream(meas, q_set, meas + 0.3)[0]
    assert np.abs(np.diff(big, axis=0)).max() * J.PUB_HZ <= J.MAX_VEL + 1e-9


def test_release_xy_is_warning_not_failure():
    checks = {"release_xy_m": {"value": 0.0063, "limit": 0.005, "ok": False}, "release_drop_m": {"value": 0.1, "ok": False}}
    PPF.warn_only(checks, "release_xy_m")
    assert checks["release_xy_m"]["ok"] and checks["release_xy_m"]["warn"] and checks["release_xy_m"]["value"] == 0.0063
    assert not PPF.all_ok(checks)                                              # 다른 검사는 그대로 막는다
    del checks["release_drop_m"]
    assert PPF.all_ok(checks)


def test_names_fault_needs_same_side_and_order():
    arm, hand = list(J.P.arm_names("right")), list(J.P.hand_names("right"))
    assert J.names_fault("right", arm, hand) is None
    assert J.names_fault("right", arm[::-1], hand)                             # 순서 다름
    assert J.names_fault("right", arm, list(J.P.hand_names("left")))           # 손이 다른 쪽
    assert J.names_fault("left", arm, hand)


def test_dq_fault_rejects_non_jog_size():
    assert J.MAX_VEL == min(0.30, J.PD_VEL_RAD_S)
    assert J.dq_fault(np.full(7, 0.01)) is None
    assert J.dq_fault(np.r_[np.zeros(6), J.MAX_JOG_DQ_RAD + 0.01])

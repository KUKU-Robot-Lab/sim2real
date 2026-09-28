"""물체 자세 추정 — FP++ 동안은 FP++, 파지 뒤에는 손에 붙인다(09.28 사용자: 파지 뒤 FP++ 는 제대로 안 될 것).

학습 trace 로 두 가지를 잠근다: 붙이는 시점이 들어 올리기 전후이고, 붙인 추정이 들어 올린 동안 실제 물체에서
크게 벗어나지 않는다(시뮬레이터에서 컵이 손 안에서 얼마나 미끄러지는가 = 이 가정의 오차).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control.joint_contract import load_contract
from policy_control.joint_object import AttachCfg, ObjectEstimator, matrix_to_quat
from policy_control.joint_obs import Pose, quat_to_matrix

pytestmark = pytest.mark.unit

RUN = Path(__file__).resolve().parents[2] / "deploy/policies/right_m15_e800"
UP = np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def c():
    if not (RUN / "joint_contract.json").is_file():
        pytest.skip("joint_contract.json 이 없다")
    return load_contract(RUN / "joint_contract.json")


def _closed(c, x: float) -> np.ndarray:
    lo, hi = np.asarray(c.hand_lo), np.asarray(c.hand_hi)
    return lo + x * (hi - lo)


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert np.allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)


def test_live_passes_through_and_a_single_jump_is_dropped(c):
    est, R = ObjectEstimator(c), np.eye(3)
    p, src = est.update(Pose(np.array([0.3, -0.1, 0.28]), UP), np.zeros(3), R, _closed(c, 0.0))
    assert src == "live" and np.allclose(p.pos, [0.3, -0.1, 0.28])
    p, src = est.update(Pose(np.array([0.5, -0.1, 0.28]), UP), np.zeros(3), R, _closed(c, 0.0))   # 20 cm 튐
    assert src == "held" and np.allclose(p.pos, [0.3, -0.1, 0.28])
    for _ in range(AttachCfg().reject_limit - 1):                                              # 계속 거기 있으면
        p, src = est.update(Pose(np.array([0.5, -0.1, 0.28]), UP), np.zeros(3), R, _closed(c, 0.0))
    assert src == "live" and np.allclose(p.pos, [0.5, -0.1, 0.28])                           # 진짜 옮겨진 것
    assert est.update(None, np.zeros(3), R, _closed(c, 0.0)) == (None, "missing")


def test_it_attaches_only_when_near_and_closed_for_enough_steps_then_follows_the_palm(c):
    est, R = ObjectEstimator(c), np.eye(3)
    cup = Pose(np.array([0.10, 0.0, 0.0]), UP)                                               # 손바닥에서 10 cm
    for _ in range(10):
        est.update(cup, np.zeros(3), R, _closed(c, 0.2))                                     # 가깝지만 안 닫힘
    assert not est.attached
    for _ in range(AttachCfg().attach_steps):
        est.update(cup, np.zeros(3), R, _closed(c, 0.7))
    assert est.attached
    # 손바닥이 z 로 90° 돌고 옮겨 가면 물체도 같이 — FP++ 가 엉뚱한 값을 내도 보지 않는다
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p, src = est.update(Pose(np.array([9.0, 9.0, 9.0]), UP), np.array([0.0, 0.0, 0.2]), Rz, _closed(c, 0.7))
    assert src == "attached" and np.allclose(p.pos, [0.0, 0.10, 0.2], atol=1e-9)
    assert np.allclose(quat_to_matrix(p.quat), Rz, atol=1e-9)
    assert est.update(None, np.zeros(3), R, _closed(c, 0.7))[1] == "attached"                # 가려져도 계속
    # 예측 1 cm 옆의 FP++ 도 받지 않는다 — 파지 뒤에는 FK 로만(09.28 사용자)
    p, src = est.update(Pose(np.array([0.11, 0.0, 0.0]), UP), np.zeros(3), R, _closed(c, 0.7))
    assert src == "attached" and np.allclose(p.pos, [0.10, 0.0, 0.0])
    est.reset()
    assert not est.attached and est.update(None, np.zeros(3), R, _closed(c, 0.7)) == (None, "missing")


def test_on_the_training_trace_the_attached_estimate_stays_on_the_lifted_cup(c):
    if not (RUN / "trace.npz").is_file():
        pytest.skip("trace 가 없다 — isaac_joint_trace.py")
    z = np.load(RUN / "trace.npz")
    d = {k: z[k][:, 0] for k in z.files}
    est = ObjectEstimator(c)
    start_z, attach_t, errs = None, [], []
    for t in range(len(d["ep_len"])):
        if d["ep_len"][t] == 0:
            est.reset()
            start_z = d["obj_pos"][t, 2]
        R = quat_to_matrix(d["palm_quat"][t])
        was = est.attached
        p, src = est.update(Pose(d["obj_pos"][t], d["obj_quat"][t]), d["palm_pos"][t], R, d["hand_qstar"][t])
        if est.attached and not was:
            attach_t.append((t, float(d["obj_pos"][t, 2] - start_z)))
        if src.startswith("attached") and d["obj_pos"][t, 2] - start_z > 0.02:
            errs.append(float(np.linalg.norm(p.pos - d["obj_pos"][t])))
    assert attach_t, "들어 올리는 동안 한 번도 붙지 않았다"
    assert all(lift < 0.02 for _, lift in attach_t), attach_t                               # 들기 전후에 붙는다
    # 붙인 뒤 손가락이 조이는 동안 컵이 손 안에서 움직인 만큼(학습 trace 2.3 cm)이 FK 고정의 대가 —
    # 실기에서 FP++ 로 다듬을 때 끌려간 6~18 cm 보다 작다(09.28)
    assert errs and max(errs) < 0.025, (len(errs), max(errs) if errs else None)
    print(f"붙인 시점 {attach_t} · 들어 올린 {len(errs)} 스텝 추정 오차 최대 {max(errs) * 100:.2f} cm")


def test_after_attaching_a_drifting_fpp_is_ignored_and_the_hand_carries_the_object(c):
    """가려진 FP++ 가 흘러가도(여기선 5 cm/스텝) 붙인 추정은 손바닥을 따른다."""
    if not (RUN / "trace.npz").is_file():
        pytest.skip("trace 가 없다")
    z = np.load(RUN / "trace.npz")
    d = {k: z[k][:, 0] for k in z.files}
    est, drift, errs = ObjectEstimator(c), None, []
    for t in range(len(d["ep_len"])):
        if d["ep_len"][t] == 0:
            est.reset()
            drift = None
        true = d["obj_pos"][t]
        if est.attached and drift is None:
            drift = 0.0
        if drift is not None:
            drift += 0.05                                     # 붙인 뒤 FP++ 가 엉뚱하게 흐른다
        fpp = Pose(true + np.array([drift or 0.0, 0.0, 0.0]), d["obj_quat"][t])
        p, src = est.update(fpp, d["palm_pos"][t], quat_to_matrix(d["palm_quat"][t]), d["hand_qstar"][t])
        if drift:
            assert src == "attached"
            errs.append(float(np.linalg.norm(p.pos - true)))
    assert errs and max(errs) < 0.03, max(errs)


def test_after_attaching_a_slowly_drifting_fpp_cannot_drag_the_estimate(c):
    """09.28 실기: 붙인 뒤 FP++ 가 스텝마다 3 cm 안으로 조금씩 흐르면 '다듬기'가 그것을 받아 상대 자세가 6~18 cm
    끌려갔다(컵은 손에 있었다 — 사용자 확인). 파지 뒤에는 FK 로만 — 느린 흐름도 받지 않는다."""
    est, R = ObjectEstimator(c), np.eye(3)
    cup = Pose(np.array([0.10, 0.0, 0.0]), UP)
    for _ in range(AttachCfg().attach_steps):
        est.update(cup, np.zeros(3), R, _closed(c, 0.7))
    assert est.attached
    tilted = np.array([np.cos(np.radians(23)), np.sin(np.radians(23)), 0.0, 0.0])      # 46° 로 기운 FP++
    for k in range(1, 40):                                                              # 0.5 cm/스텝 → 20 cm
        p, src = est.update(Pose(np.array([0.10, 0.005 * k, 0.0]), tilted), np.zeros(3), R, _closed(c, 0.7))
        assert src == "attached"
    assert np.allclose(p.pos, [0.10, 0.0, 0.0], atol=1e-9)
    assert np.allclose(p.quat, UP, atol=1e-9)

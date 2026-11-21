"""11.18 cap_scripts — rh_cap 깊이 학생의 하강 · 이송 스크립트와 캡 · 리시버 자세 공급(hdgp db1160bc rh_cap_env 식). rclpy 없음."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control import cap_scripts as S
from policy_control import rh_aglt as A
from policy_control import rh_cap as C

REPO = Path(__file__).resolve().parents[2]
HDGP = REPO.parent / "hdgp"
NO_HDGP = not (HDGP / "source/openarm/openarm/agnostic/modules/ready_carry").is_dir()
Q0 = np.array([-1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646])


@pytest.fixture(scope="module")
def c() -> C.CapContract:
    return C.load_contract(Path(__file__).resolve().parent / "fixtures" / "rh_cap_seat_contract.json")


class LinFK:
    """가짜 팔: 손바닥 위치 = q[:3](+ off), 방향 고정 — DLS 가 위치만 푼다. frozen=True 면 손바닥이 안 움직인다."""

    def __init__(self, frozen: bool = False) -> None:
        self.frozen = frozen

    def palm_pose(self, q, _hand):
        p = Q0[:3] if self.frozen else np.asarray(q, float)[:3]
        return SimpleNamespace(palm_pos=p.copy(), extra={"palm_rot": np.eye(3)})


def _m(q, tact=0.0):
    return SimpleNamespace(arm_q=np.asarray(q, float), palm_pos=np.asarray(q, float)[:3].copy(), palm_R=np.eye(3),
                           tactile_n=np.full(5, tact))


def _run_descend(c, fk, cap, tact=0.0, n=400):
    """완전 추종 가정(q = 누적 목표)으로 하강을 굴린다 → (끝 틱, 끝 사유, 마지막 q)."""
    d = S.Descend(c, fk, h=0.04)
    q = Q0.copy()
    d.start(fk.palm_pose(q, None).palm_pos, np.eye(3), *cap)
    for t in range(1, n + 1):
        a, done = d.step(_m(q, tact) if not fk.frozen else _m(Q0, tact), np.zeros(6), q)
        q = q + c.k_arm * a
        if done:
            return t, d.why, q, d
    raise AssertionError("하강이 끝나지 않았다")


# ---------------------------------------------------------------- 하강
def test_descend_goal_puts_pinch_ref_at_cap_origin_plus_h_along_axis():
    palm, R = np.array([0.3, -0.2, 0.4]), np.eye(3)
    cap = np.array([0.2, -0.18, 0.219])
    g = S.descend_goal(palm, R, cap, (1.0, 0, 0, 0), 0.04)
    np.testing.assert_allclose(g + R @ np.array(S.PINCH_OPEN_OFFSET), cap + [0, 0, 0.04])    # 끝에서 pinch_ref = 캡 + h·축
    q90 = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)                                   # x 축 90° — 캡 축 = −y
    g2 = S.descend_goal(palm, R, cap, q90, 0.04)
    np.testing.assert_allclose(g2 + np.array(S.PINCH_OPEN_OFFSET), cap + [0, -0.04, 0], atol=1e-12)


def test_descend_ends_on_reach_with_straight_line_speed(c):
    cap = (Q0[:3] + np.array(S.PINCH_OPEN_OFFSET) + [0.0, 0.0, -0.10], (1.0, 0, 0, 0))     # 집기점이 6 cm 내려가야 하는 자리
    t, why, q, d = _run_descend(c, LinFK(), cap)
    L = 0.06
    assert why == "reached" and abs(t - L / (S.DESCEND_SPEED / c.policy_hz)) <= 3            # 0.06 m/s · 60 Hz → 60 틱
    assert np.linalg.norm(q[:3] - d.goal) <= S.DESCEND_TOL_M


def test_descend_ends_on_touch_and_on_timeout(c):
    cap = (Q0[:3] + np.array(S.PINCH_OPEN_OFFSET) + [0.0, 0.0, -0.10], (1.0, 0, 0, 0))
    assert _run_descend(c, LinFK(), cap, tact=1.5)[:2] == (1, "touch")                       # 손끝 촉각 > 1 N
    assert _run_descend(c, LinFK(), cap, tact=np.nan)[1] == "reached"                        # 끊긴 촉각(NaN)은 닿음이 아니다
    assert _run_descend(c, LinFK(frozen=True), cap)[:2] == (S.DESCEND_MAX_STEPS, "timeout")  # 손바닥이 안 따라온다


# ---------------------------------------------------------------- 이송
def test_carry_anchor_is_cap_mouth_h_clr_above_receiver_mouth():
    rp = np.array([0.25, 0.12, 0.41])
    c_pre, ru = S.carry_anchor(rp, (1.0, 0, 0, 0))
    np.testing.assert_allclose(ru, [0, 0, 1])
    np.testing.assert_allclose(c_pre, rp + [0, 0, C.RCV_HALF_H + 0.03 + 0.015])              # 캡 입(원점 − 0.015)이 입구 + 3 cm
    q = (np.cos(0.1), np.sin(0.1), 0.0, 0.0)                                                 # 기운 리시버 — 축 따라
    c2, u2 = S.carry_anchor(rp, q)
    np.testing.assert_allclose(c2, rp + u2 * 0.13, atol=1e-12)


def test_carry_via_up_then_over_with_x_floor():
    c_pre, ru = np.array([0.25, 0.12, 0.54]), np.array([0.0, 0.0, 1.0])
    via = S.carry_via([0.05, -0.2, 0.30], c_pre, ru)
    np.testing.assert_allclose(via[1], c_pre + [0, 0, 0.06])                                  # 앵커 위 carry_above
    np.testing.assert_allclose(via[0], [0.10, -0.2, 0.60])                                    # 그 높이로 올림 · x 하한 0.10
    np.testing.assert_allclose(S.carry_via([0.3, -0.2, 0.7], c_pre, ru)[0], [0.3, -0.2, 0.7])  # 이미 높고 앞이면 그대로


def test_carry_region_matches_env_layout():
    offs, tils = S.carry_region(np.array([0.0, 0.0, 1.0]))
    assert offs.shape == tils.shape == (1 + 4 + 8 + 8, 3)
    np.testing.assert_allclose(offs[:5, 2], [0, -0.0125, -0.025, -0.0375, -0.05])
    np.testing.assert_allclose(np.abs(offs[5:13, :2]), 0.02)
    np.testing.assert_allclose(np.linalg.norm(tils[13:], axis=1), np.radians(5.0))


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_carry_with_pinned_mover_reaches_anchor(c):
    """배포 핀 Mover(225abc0)가 cap 이송 인자(via · Region 직접 · settle_steps)를 받고 앵커에 캡을 가져다 놓는다(완전 추종)."""
    from policy_control.place_ready_carry import load_chain
    s = c.side()
    RC, mch = load_chain(s.palm_body, list(s.arm_joints))
    car = S.Carry(RC, mch, k_arm=c.k_arm, arm_ema=c.arm_ema)
    import torch
    T = mch.fk(torch.tensor([list(Q0)], dtype=torch.float32))[0].double().numpy()
    cap_p = T[:3, :3] @ [0.0, 0.05, 0.10] + T[:3, 3]                                        # 손 안 캡(손바닥 기준 고정)
    T_to = car.cap_in_palm(Q0, cap_p, (1.0, 0, 0, 0))
    plan = car.start(Q0, T_to, S.RCV_FIXED_POS, S.RCV_FIXED_QUAT)
    q, n = Q0.copy(), 0
    while car.on:
        a, done = car.step(q, q)
        q = q + c.k_arm * c.arm_ema * a                                                       # 증분 = 행동 · k_arm · arm_ema
        n += 1
        assert n <= S.CARRY_MAX_STEPS + S.CARRY_SETTLE_STEPS + 5
    a, done = car.step(q, q)
    assert done and not a.any() and car.arrived                                               # 끝난 뒤 = 0(유지)
    end = (mch.fk(torch.tensor([list(q)], dtype=torch.float32)) @ T_to)[0, :3, 3].double().numpy()
    assert plan["ok"] and np.linalg.norm(end - plan["anchor"]) < 0.01


def fake_rc(ok=True, end_at=3, arrive=True, exc=None):
    """배포 핀 ready_carry 자리 — Mover 의 계획 ok · 끝 틱 · 도착 · start 예외를 정한다(hdgp 없이)."""
    import torch

    class Mover:
        def __init__(self, ch, cfg, n, dev, seed=0):
            self.on, self.T, self.k = torch.zeros(1, dtype=torch.bool), torch.tensor([end_at]), 0

        def start(self, *a, **kw):
            if exc is not None:
                raise exc
            self.on[0] = True
            return SimpleNamespace(ok=torch.tensor([ok]), margin=torch.tensor([0.1]), q=torch.zeros(1, 7))

        def step(self, q):
            self.k += 1
            done = self.k >= end_at
            self.on[0] = not done
            return torch.full((1, 7), 1e-4), None, torch.tensor([done]), torch.tensor([done and arrive])

    return SimpleNamespace(mover=SimpleNamespace(Mover=Mover, MoverCfg=lambda **kw: kw),
                           posture=SimpleNamespace(Region=lambda *a: a))


class EyeChain:
    def fk(self, q):
        import torch
        return torch.eye(4).repeat(q.shape[0], 1, 1)


def fake_carry(c, **kw) -> S.Carry:
    return S.Carry(fake_rc(**kw), EyeChain(), k_arm=c.k_arm, arm_ema=c.arm_ema)


def _eye_T():
    import torch
    return torch.eye(4)[None]


def test_carry_plan_not_ok_is_recorded_not_refused(c):
    car = fake_carry(c, ok=False)                                                             # 학습 env 도 ok 로 막지 않는다
    assert car.start(Q0, _eye_T(), S.RCV_FIXED_POS, S.RCV_FIXED_QUAT)["ok"] is False
    assert car.on and car.as_dict()["ok"] is False


def test_carry_start_exception_leaves_reset_state(c):
    sc = S.CapScripts(c, fake_carry(c, exc=RuntimeError("IK 발산")))
    sc.T_to = _eye_T()
    m = _m(Q0)
    for _ in range(2):                                                                        # 시작 안 된 Mover 가 "끝" 으로 읽히지 않는다
        with pytest.raises(S.ScriptError, match="IK 발산"):
            sc.carry_step(m, Q0, (S.RCV_FIXED_POS, S.RCV_FIXED_QUAT))
    assert sc.carry.mv is None and sc.as_dict()["carry"]["fail"]


def test_carry_tracking_guard_six_ticks(c):
    car = fake_carry(c, end_at=1000)
    car.start(Q0, _eye_T(), S.RCV_FIXED_POS, S.RCV_FIXED_QUAT)
    off = Q0 + S.TRACK_ABORT_RAD + 0.01
    for _ in range(S.TRACK_ABORT_TICKS - 1):
        car.step(Q0, off)
    car.step(Q0, Q0)                                                                          # 한 틱 따라오면 다시 센다
    for _ in range(S.TRACK_ABORT_TICKS - 1):
        car.step(Q0, off)
    with pytest.raises(S.ScriptError, match="못 따라간다"):
        car.step(Q0, off)
    with pytest.raises(S.ScriptError, match="못 따라간다"):                                  # reset 전까지 유지
        car.step(Q0, Q0)


def test_carry_end_without_arrival_holds_and_reports(c):
    car = fake_carry(c, end_at=2, arrive=False)
    T = _eye_T()
    T[0, :3, 3] = 0.1                                                                         # 캡 점 = 손바닥(EyeChain 원점) + 0.1·(1,1,1)
    car.start(Q0, T, S.RCV_FIXED_POS, S.RCV_FIXED_QUAT)
    assert [car.step(Q0, Q0)[1] for _ in range(4)] == [False] * 4 and car.missed              # 끝 신호 없음(자동 전환 없음)
    a, _ = car.step(Q0, Q0)
    assert not a.any()                                                                        # 목표 유지
    d = car.as_dict()
    assert d["carry_missed"] and d["miss_err_mm"] == pytest.approx(np.linalg.norm(0.1 - np.array(d["anchor"])) * 1e3, abs=0.2)
    car = fake_carry(c, end_at=2, arrive=True)
    car.start(Q0, _eye_T(), S.RCV_FIXED_POS, S.RCV_FIXED_QUAT)
    assert [car.step(Q0, Q0)[1] for _ in range(3)] == [False, True, True] and car.arrived and not car.missed


def test_descend_start_refusals(c):
    fk = LinFK()
    p = fk.palm_pose(Q0, None).palm_pos
    pinch = p + np.array(S.PINCH_OPEN_OFFSET)
    with pytest.raises(S.ScriptError, match="수평 50.0 mm"):                                 # 집기점이 캡에서 옆으로 5 cm
        S.Descend(c, fk).start(p, np.eye(3), pinch + [0.05, 0.0, -0.10], (1.0, 0, 0, 0))
    with pytest.raises(S.ScriptError, match="L 160.0 mm"):                                    # 캡이 20 cm 아래(선 16 cm)
        S.Descend(c, fk).start(p, np.eye(3), pinch + [0.0, 0.0, -0.20], (1.0, 0, 0, 0))
    sc = S.CapScripts(c, fake_carry(c))
    sc.fk = fk
    cap = (pinch + [0.0, 0.0, -0.10], np.array([1.0, 0, 0, 0]))
    with pytest.raises(S.ScriptError, match="촉각"):                                          # 첫 틱에 이미 닿음(기준값 뜸)
        sc.descend_step(_m(Q0, 1.5), np.zeros(6), Q0, cap)
    with pytest.raises(S.ScriptError, match="촉각"):                                          # 촉각이 내려가도 중간에서 시작하지 않는다
        sc.descend_step(_m(Q0, 0.0), np.zeros(6), Q0, cap)
    sc.reset()
    assert sc.descend_step(_m(Q0, 0.0), np.zeros(6), Q0, cap)[1] is False
    sc.reset()
    with pytest.raises(S.ScriptError, match="NaN"):                                           # 촉각이 끊겼으면 닿음 정지가 없다
        sc.descend_step(_m(Q0, np.nan), np.zeros(6), Q0, cap)
    with pytest.raises(S.ScriptError, match="NaN"):                                           # reset 까지 같은 오류
        sc.descend_step(_m(Q0, 0.0), np.zeros(6), Q0, cap)


class NanChain(EyeChain):
    def fk(self, q):
        return super().fk(q) * (1.0 + 0.0 * q.sum())                                            # q 의 NaN 이 그대로 번진다


def test_nan_arm_q_at_grasp_never_starts_carry(c):
    sc = S.CapScripts(c, S.Carry(fake_rc(), NanChain(), k_arm=c.k_arm, arm_ema=c.arm_ema))
    sc.fk = LinFK()
    sc.dsc = SimpleNamespace(step=lambda *a: (np.zeros(7), True))                              # 하강이 이번 틱에 끝난다
    m = _m(np.full(7, np.nan))
    with pytest.raises(S.ScriptError, match="NaN"):
        sc.descend_step(m, np.zeros(6), Q0, (S.CAP_FIXED_POS, S.CAP_FIXED_QUAT))
    assert sc.T_to is None
    with pytest.raises(S.ScriptError):
        sc.carry_step(_m(Q0), Q0, (S.RCV_FIXED_POS, S.RCV_FIXED_QUAT))
    assert sc.carry.mv is None


def test_dls_arm_action_equals_rh_cap_ik_lift_action(c):
    """같은 입력이면 하강 DLS 와 rh_cap 들기 DLS 가 같은 행동(실제 계약 FK · 손바닥 R ≠ I)."""
    from dataclasses import replace
    c = replace(c, grasp_ik_lift_m=0.1)                                                       # 고정 계약은 0.0 — 들기 높이가 0 이 아니게
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK
    s = c.side()
    fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
    hand = np.asarray(s.hand_open, float)
    pose = fk.palm_pose(Q0, hand)
    R = pose.extra["palm_rot"]
    assert not np.allclose(R, np.eye(3), atol=0.1)
    th = 0.05
    R0 = R @ np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])  # 방향 오차도 0 이 아니게
    p0, t, tgt = pose.palm_pos + [0.01, -0.005, 0.0], 7, Q0 + 0.01
    dz = min(t * c.grasp_ik_lift_speed / c.policy_hz, c.grasp_ik_lift_m)
    assert dz > 0
    want = C.ik_lift_action(c, fk, Q0, hand, tgt, p0, R0, t, np.asarray(s.arm_lo, float), np.asarray(s.arm_hi, float))
    got = S.dls_arm_action(c, fk, Q0, hand, tgt, p0 + [0.0, 0.0, dz], R0, damping=c.grasp_ik_damping)
    assert np.abs(want).max() > 1e-3
    np.testing.assert_allclose(got, want, atol=1e-12)


# ---------------------------------------------------------------- 자세 공급
def test_pick_pose_fresh_stale_missing():
    fixed = ((0.1, 0.2, 0.3), (1.0, 0, 0, 0))
    got = (10.0, np.array([0.5, 0.5, 0.5]), np.array([1.0, 0, 0, 0]), 9.9)
    p = S.pick_pose(got, fixed, 10.3, 0.5)
    assert p["src"] == "fpp" and p["age_s"] == pytest.approx(0.3)
    np.testing.assert_allclose(p["pos"], 0.5)
    p = S.pick_pose(got, fixed, 10.6, 0.5)
    assert p["src"] == "fixed" and p["age_s"] == pytest.approx(0.6)
    np.testing.assert_allclose(p["pos"], fixed[0])
    assert S.pick_pose(None, fixed, 10.0, 0.5)["src"] == "fixed" and S.pick_pose(None, fixed, 10.0, 0.5)["age_s"] is None
    r = S.pick_pose(got, fixed, 10.1, 0.5, S.rcv_from_shaker)                                 # 쉐이커 메쉬 원점 → env 리시버 원점
    np.testing.assert_allclose(r["pos"], [0.5, 0.5, 0.48])
    assert S.pick_pose(got, fixed, 10.3, 0.5, wall=10.3)["src"] == "fpp"                      # stamp 나이 0.4 s
    assert S.pick_pose(got, fixed, 10.3, 0.5, wall=10.5)["src"] == "fixed"                    # 수신은 새로워도 영상이 0.6 s 낡음
    assert S.pick_pose((*got[:3], 0.0), fixed, 10.3, 0.5, wall=99.0)["src"] == "fpp"          # stamp 0 = 수신 시각만
    assert S.pick_pose(got, fixed, 10.3, 0.5, wall=9.86)["src"] == "fpp"                      # stamp 가 0.04 s 미래 = 시계 어긋남 허용
    fut = S.pick_pose(got, fixed, 10.3, 0.5, wall=9.8)
    assert fut["src"] == "fixed" and "stamp 가 미래" in S.rcv_fixed_refusals(fut, False, 0.5)[0]


def test_rcv_from_shaker_moves_along_shaker_axis():
    q = (np.cos(0.25), np.sin(0.25), 0.0, 0.0)
    p, _ = S.rcv_from_shaker([0.0, 0.0, 0.0], q)
    np.testing.assert_allclose(p, -0.020 * A.quat_apply(q, np.array([0.0, 0.0, 1.0])))


def _pose(pos, quat=(1.0, 0, 0, 0), src="fixed"):
    q = np.asarray(quat, float)
    return {"src": src, "pos": np.asarray(pos, float), "quat": q / np.linalg.norm(q), "age_s": None}


def test_default_fixed_poses_are_inside_training_range():
    assert S.cap_refusals(_pose(S.CAP_FIXED_POS)) == []
    assert S.rcv_refusals(_pose(S.RCV_FIXED_POS, S.RCV_FIXED_QUAT)) == []


def test_pose_refusals_outside_training_range():
    assert S.cap_refusals(_pose([0.30 + 0.019, -0.35, 0.219])) == []                         # 소환 경계 + 폭 안
    assert "학습 범위" in S.cap_refusals(_pose([0.30 + 0.03, -0.185, 0.219], src="fpp"))[0]
    assert "학습 범위" in S.cap_refusals(_pose([0.215, -0.185, 0.30]))[0]                     # 상판에 선 높이가 아니다
    tilt = (np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0)                             # 20° 기운 캡
    assert "기울기" in S.cap_refusals(_pose(S.CAP_FIXED_POS, tilt))[0]
    assert "학습 범위" in S.rcv_refusals(_pose([0.25, 0.12, 0.50]))[0]
    assert "기울기" in S.rcv_refusals(_pose(S.RCV_FIXED_POS, (np.cos(np.radians(25)), np.sin(np.radians(25)), 0, 0)))[0]

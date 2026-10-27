"""rh_cap 계열(RH56F1 오른팔 캡 단계 정책, 10.12 사용자 간단 배포) — 계약 · 팔 상자 · IK 수직 들기 · 단계 성공 · 노드 계열 등록.

hdgp rh_cap_r(feat/rh-cap-split 95efd6b0): 관측 91(촉각 없음) · 행동 13 · 디코더 rh_aglt, 손 자세 rh56f1_right_captop,
목표 = 리시버 축 위 seat_offset, grasp = 상자 ± 0.3 rad · 쥔 판정 15 스텝 → IK 수직 0.10 m @ 0.06 m/s, seat = 캡을 손바닥에 붙인 채 끼움.
계약 고정본은 fixtures/rh_cap_{grasp,seat}_contract.json(스모크 런 · capg_j02 ep1200 촉각 뺀 체크포인트에서 빌드).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from policy_control import rh_aglt as A
from policy_control import rh_cap as C

FIX = Path(__file__).resolve().parent / "fixtures"
URDF = Path.home() / "rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
#: 사전 파지 뱅크 근처의 오른팔 자세(home 근처 — 손바닥이 테이블 위)
Q_PRE = np.array([-1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646])
#: IK 들기 검사용 — 관절 한계에서 먼 자세(home 은 들기 중 j1 이 하한 -1.4 에 닿아 방향 · 수평을 조금 내준다, sim DLS 도 같다)
Q_MID = np.array([-0.6, 0.2026, 0.6538, 1.7608, 0.3791, 0.2, 0.6646])


@pytest.fixture(scope="module")
def cg() -> C.CapContract:
    return C.load_contract(FIX / "rh_cap_grasp_contract.json")


@pytest.fixture(scope="module")
def cs() -> C.CapContract:
    return C.load_contract(FIX / "rh_cap_seat_contract.json")


@pytest.fixture(scope="module")
def fk(cg):
    if not URDF.is_file():
        pytest.skip("RH56F1 URDF 없음")
    from policy_control.fk_numpy import UrdfChainFK
    s = cg.side()
    return UrdfChainFK(URDF, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)


class Zero:
    def forward(self, obs):
        assert obs.shape == (91,)
        return np.zeros(13)

    def reset(self):
        pass


def _meas(c, fk, arm_q, hand_q, cap_pos, cap_quat=(1.0, 0.0, 0.0, 0.0), tact=(0.0,) * 5):
    s = c.side()
    pose = fk.palm_pose(arm_q, hand_q)
    return {"arm": A.RaMeas(arm_q=np.asarray(arm_q, float), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, hand_q)),
                            palm_pos=pose.palm_pos, palm_R=pose.extra["palm_rot"], tips=pose.tips,
                            cup_pos=np.asarray(cap_pos, float), cup_quat=np.asarray(cap_quat, float),
                            tactile_n=np.asarray(tact, float))}


def test_contract_keys_are_the_training_run(cg, cs):
    for c, ph in ((cg, "grasp"), (cs, "seat")):
        assert c.schema == C.SCHEMA and c.cap_phase == ph and c.task == f"open-rh_r_cap_{ph}"
        assert (c.obs_dim, c.action_dim, c.hold_steps, c.actor_tactile_obs) == (91, 13, 0, False)
        assert list(c.side().hand_open) == list(C.CAPTOP_OPEN) and list(c.side().hand_grip) == list(C.CAPTOP_GRIP)
        assert c.cup_half_height == pytest.approx(0.025) and c.seat_offset == pytest.approx(0.0783, abs=1e-4)
        assert (c.phase_arm_box_rad, c.phase_success_steps) == (0.3, 30)
        assert c.hand_command == "admittance"
    assert (cg.grasp_ik_lift_m, cg.grasp_ik_lift_after, cg.grasp_ik_lift_speed) == (0.1, 15, 0.06)
    assert (cs.seat_tol, cs.grade_perfect_tilt_deg) == (0.003, 5.0)


def test_box_keeps_q_raw_inside(cg):
    center = Q_PRE.copy()
    q_star = center + 0.29                                # 상자 끝 근처
    a = C.box_arm_action(cg, np.ones(7), q_star, center)
    assert np.all(q_star + cg.k_arm * a <= center + cg.phase_arm_box_rad + 1e-9)
    a = C.box_arm_action(cg, -np.ones(7), center - 0.29, center)
    assert np.all(center - 0.29 + cg.k_arm * a >= center - cg.phase_arm_box_rad - 1e-9)


def test_seat_goal_and_state(cs):
    rcv_pos, rcv_q = np.array([0.30, 0.10, 0.40]), np.array([1.0, 0.0, 0.0, 0.0])
    g = C.seat_goal(cs, rcv_pos, rcv_q)
    assert np.allclose(g.pos, rcv_pos + [0, 0, cs.seat_offset])
    st = C.seat_state(cs, g.pos, [0, 0, 1], rcv_pos, [0, 0, 1])
    assert st["seat_dz"] == pytest.approx(0.0) and st["lat"] == pytest.approx(0.0) and st["rel_tilt"] == pytest.approx(0.0)
    up = C.seat_state(cs, g.pos + [0, 0, 0.02], [0, 0, 1], rcv_pos, [0, 0, 1])
    assert up["seat_dz"] == pytest.approx(0.02)


def test_grasp_ik_lift_raises_palm_with_fixed_orientation(cg, fk):
    ch = C.CapChain(cg, Zero())
    ch.fk = fk
    hand = np.asarray(cg.side().hand_open, float)
    p0 = fk.palm_pose(Q_MID, hand)
    cap = p0.palm_pos + [0.0, 0.0, -0.06]
    ch.reset(_meas(cg, fk, Q_MID, hand, cap), receiver=(np.array([0.3, 0.12, 0.41]), np.array([1.0, 0, 0, 0])))
    q, grip = Q_MID.copy(), (3.0, 3.0, 0.0, 0.0, 0.0)          # 엄지 + 검지 손끝 > 1 N = 쥠
    z = []
    for t in range(cg.grasp_ik_lift_after + 120):
        _, a, tg = ch.step(_meas(cg, fk, q, hand, cap, tact=grip))
        q = tg["arm"][0]                                    # 완전 추종
        z.append(fk.palm_pose(q, hand).palm_pos[2])
    assert ch.lift_on
    lift = z[-1] - p0.palm_pos[2]
    assert 0.08 < lift < 0.105                              # 0.10 m 까지(EMA · 지연으로 조금 덜)
    R1 = fk.palm_pose(q, hand).extra["palm_rot"]
    ang = np.degrees(np.linalg.norm(C._rotvec(R1 @ ch.lift_R0.T)))
    assert ang < 2.0
    assert np.allclose(fk.palm_pose(q, hand).palm_pos[:2], ch.lift_p0[:2], atol=0.01)


def test_grasp_success_needs_lift_and_streak(cg, fk):
    ch = C.CapChain(cg, Zero())
    ch.fk = fk
    hand = np.asarray(cg.side().hand_open, float)
    p0 = fk.palm_pose(Q_PRE, hand).palm_pos
    cap0 = p0 + [0.0, 0.0, -0.06]
    ch.reset(_meas(cg, fk, Q_PRE, hand, cap0), receiver=(np.zeros(3), np.array([1.0, 0, 0, 0])))
    q, grip = Q_PRE.copy(), (3.0, 3.0, 0.0, 0.0, 0.0)
    for _ in range(400):
        palm = fk.palm_pose(q, hand).palm_pos
        _, _, tg = ch.step(_meas(cg, fk, q, hand, palm + [0.0, 0.0, -0.06], tact=grip))   # 캡이 손에 붙어 따라 올라온다
        q = tg["arm"][0]
        if ch.done:
            break
    assert ch.done and ch.last["lift_m"] >= cg.grasp_success_lift_m


def test_seat_success_on_seated_still_cap(cs, fk):
    ch = C.CapChain(cs, Zero())
    hand = np.asarray(cs.side().hand_grip, float) * 0.8
    pose = fk.palm_pose(Q_PRE, hand)
    rcv_q = np.array([1.0, 0.0, 0.0, 0.0])
    seat = pose.palm_pos + [0.0, 0.0, -0.05]
    rcv = seat - [0.0, 0.0, cs.seat_offset]
    ch.reset(_meas(cs, fk, Q_PRE, hand, seat), receiver=(rcv, rcv_q), held=(Q_PRE, hand))
    for _ in range(cs.phase_success_steps):
        ch.step(_meas(cs, fk, Q_PRE, hand, seat + [0.0, 0.0, 0.5]))      # 측정 캡은 무시 — 손바닥에 붙은 캡
    assert ch.done and abs(ch.last["seat_dz"]) < 1e-6


def test_seat_attached_cap_moves_with_palm(cs, fk):
    ch = C.CapChain(cs, Zero())
    hand = np.asarray(cs.side().hand_grip, float) * 0.8
    pose = fk.palm_pose(Q_PRE, hand)
    cap = pose.palm_pos + [0.01, 0.0, -0.05]
    ch.reset(_meas(cs, fk, Q_PRE, hand, cap), receiver=(np.zeros(3), np.array([1.0, 0, 0, 0])), held=(Q_PRE, hand))
    q2 = Q_PRE + np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    p2 = fk.palm_pose(q2, hand)
    pos, _ = ch.cup_from_palm(p2.palm_pos, p2.extra["palm_rot"])
    assert np.linalg.norm(pos - p2.palm_pos) == pytest.approx(np.linalg.norm(cap - pose.palm_pos), abs=1e-9)


class Rec:
    """일정한 행동(0.4)을 내고 받은 관측 · reset 횟수를 적는다."""

    def __init__(self):
        self.obs, self.resets = [], 0

    def forward(self, obs):
        self.obs.append(np.array(obs, float))
        return np.full(13, 0.4)

    def reset(self):
        self.resets += 1


def test_seat_reset_clears_prev_actions(cs, fk):
    """10.27 hdgp 연결 평가 분해(caps_n12 ep9000 · 시드 31-35): 직전 행동이 seat 에 이어지면 63 %, 0 으로 지우면 70 %.
    reset 은 직전 행동 · LSTM 을 지우고, 다시 시작한 첫 관측이 새 체인의 첫 관측과 같아야 한다."""
    hand = np.asarray(cs.side().hand_grip, float) * 0.8
    pose = fk.palm_pose(Q_PRE, hand)
    cap = pose.palm_pos + [0.0, 0.0, -0.05]
    rcv = (cap - [0.0, 0.0, cs.seat_offset], np.array([1.0, 0, 0, 0]))
    m = _meas(cs, fk, Q_PRE, hand, cap)
    used = Rec()
    ch = C.CapChain(cs, used)
    ch.reset(m, receiver=rcv, held=(Q_PRE, hand))
    for _ in range(5):
        ch.step(m)
    assert np.abs(ch.prev).max() > 0
    ch.reset(m, receiver=rcv, held=(Q_PRE, hand))
    assert used.resets == 2 and not ch.prev.any()
    ch.step(m)
    fresh = Rec()
    ch2 = C.CapChain(cs, fresh)
    ch2.reset(m, receiver=rcv, held=(Q_PRE, hand))
    ch2.step(m)
    np.testing.assert_array_equal(used.obs[-1], fresh.obs[0])


def test_reset_needs_receiver(cg, fk):
    ch = C.CapChain(cg, Zero())
    hand = np.asarray(cg.side().hand_open, float)
    with pytest.raises(C.RhCapError, match="receiver"):
        ch.reset(_meas(cg, fk, Q_PRE, hand, np.zeros(3)), receiver=None)


def test_seat_refuses_without_grasp_and_grasp_refuses_fallen_cap(cg, cs, fk):
    hand = np.asarray(cg.side().hand_open, float)
    m = _meas(cs, fk, Q_PRE, hand, np.zeros(3))
    assert C.start_refusals(cs, m) and "not grasped" in C.start_refusals(cs, m)[0]
    lying = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)           # 90° 누운 캡
    assert C.start_refusals(cg, _meas(cg, fk, Q_PRE, hand, np.zeros(3), cap_quat=lying))
    assert C.start_refusals(cg, _meas(cg, fk, Q_PRE, hand, np.zeros(3))) == []


def test_wrong_hand_profile_is_refused(cg):
    s = replace(cg.side(), hand_open=[0.0] * 6)
    with pytest.raises(A.RhAgltError):
        C.validate(replace(cg, sides={"arm": s}))


def test_node_family_is_registered():
    pytest.importorskip("rclpy")
    from policy_control import rh_cap_node  # noqa: F401
    from policy_control.pour_fj_node import FAMILIES, family_of
    assert family_of(FIX / "rh_cap_grasp_contract.json") == "rh_cap" and FAMILIES["rh_cap"].name == "rh_cap_node"


class Close:
    """팔 0 · 손 +1(닫기) — 막힘 신호 시험용."""
    def forward(self, obs):
        assert obs.shape == (91,)
        return np.concatenate([np.zeros(7), np.ones(6)])

    def reset(self):
        pass


@pytest.fixture(scope="module")
def cgb() -> C.CapContract:
    return C.load_contract(FIX / "rh_cap_grasp_hb_contract.json")


def test_hb_contract_carries_block_definition(cgb):
    """capg_j04 · caps_j01(10.12 hand_block 런) 계약 = 막힘 쥠 멈춤 · 쥔 판정 · 펌웨어 멈춤 없음."""
    cs_hb = C.load_contract(FIX / "rh_cap_seat_hb_contract.json")
    for c in (cgb, cs_hb):
        assert c.hand_block and c.grasp_rule == "thumb_flex_and_any_finger"
        assert c.hand_lead_cap_rad == pytest.approx(0.2) and c.actor_tactile_obs is False and c.obs_dim == 91


def test_hb_grasp_uses_block_not_tactile(cgb, fk):
    """hand_block 계약이면 손끝 촉각만으로는 쥔 판정이 서지 않고, 손이 막혀(실측이 q* 를 못 따라감) 있어야 선다."""
    hand = np.asarray(cgb.side().hand_open, float)
    cap = fk.palm_pose(Q_MID, hand).palm_pos + [0.0, 0.0, -0.06]
    ch = C.CapChain(cgb, Zero())
    ch.fk = fk
    ch.reset(_meas(cgb, fk, Q_MID, hand, cap), receiver=(np.array([0.3, 0.12, 0.41]), np.array([1.0, 0, 0, 0])))
    h = hand.copy()
    for _ in range(40):                                     # 손이 q* 를 완전히 따라감 · 촉각 3 N — 막힘 없음
        _, _, tg = ch.step(_meas(cgb, fk, Q_MID, h, cap, tact=(3.0, 3.0, 0.0, 0.0, 0.0)))
        h = tg["arm"][1]
    assert ch.g_streak == 0
    ch2 = C.CapChain(cgb, Close())
    ch2.fk = fk
    ch2.reset(_meas(cgb, fk, Q_MID, hand, cap), receiver=(np.array([0.3, 0.12, 0.41]), np.array([1.0, 0, 0, 0])))
    for _ in range(40):                                     # 닫으라는데 실측 손이 편 채로 멈춤 = 막힘
        ch2.step(_meas(cgb, fk, Q_MID, hand, cap))
    assert ch2.g_streak > 0

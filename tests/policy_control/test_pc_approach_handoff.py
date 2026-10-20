"""★10.20 경로 D — 접근 정책(ap05)이 사전 파지 자리에 멈추면 같은 rh_aglt 노드에서 rcg03(aglt_start=ready)이 넘겨받는다.

사용자: "ap_05로 가까이 가게한 다음에, rcg_03 행하면 잡을 것 같음". hdgp 0b79f245 rh_aglt_env(aglt_start=ready):
사전 파지 손바닥 = 물체 원점 + r_mult × 반지름 × 수평 바깥 + h × 위, 넘김 순간 팔 목표 = Mover 마지막 명령(≈ 실측),
대기 · 접근 · 정착 동안 기록 행동 = (팔 0, 손 폄).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from policy_control import approach_grasp as AG
from policy_control import rh_aglt as A
from policy_control.pour_fj_node import AgltChain, aglt_start_refusals, READY_POSE_TOL_M

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy" / "policies" / "rh56f1" / "aglt"
AP05, RCG03 = POL / "right_approach_ap05" / "rh_aglt_contract.json", POL / "right_rcg03" / "rh_aglt_contract.json"
URDF = Path.home() / "rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
Q_MID = np.array([-0.6, 0.2026, 0.6538, 1.7608, 0.3791, 0.2, 0.6646])   # test_pc_approach_grasp 과 같은 자세
OBJ = "source240_real"


@pytest.fixture(scope="module")
def ap05() -> A.RaContract:
    return A.load_contract(AP05)


@pytest.fixture(scope="module")
def rcg03() -> A.RaContract:
    return A.load_contract(RCG03)


@pytest.fixture(scope="module")
def fk(ap05):
    if not URDF.is_file():
        pytest.skip("RH56F1 URDF 없음")
    from policy_control.fk_numpy import UrdfChainFK
    s = ap05.side()
    return UrdfChainFK(URDF, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)


class Toward:
    """접근 정책 대역 — 손은 편 채, 팔은 Q_MID 로."""

    def __init__(self):
        self.ch = None

    def forward(self, obs):
        c, st = self.ch.c, self.ch.dec.state["arm"]
        a_arm = np.clip((Q_MID - st.arm_target) / np.asarray(c.k_arm, float), -1.0, 1.0)
        return np.concatenate([a_arm, AG.hand_action_for(self.ch.dec.law, np.asarray(c.side().hand_open, float))])

    def reset(self):
        pass


class Grasp:
    """넘겨받는 정책 대역 — 관측을 남기고, 팔은 그대로 · 손은 닫는 행동."""

    def __init__(self):
        self.obs, self.resets = [], 0

    def forward(self, obs):
        self.obs.append(np.asarray(obs, float).copy())
        return np.concatenate([np.zeros(7), np.ones(6) * 0.5])

    def reset(self):
        self.resets += 1


class Plant:
    """팔 = 목표 완전 추종. 컵은 Q_MID 손바닥에서 수평 −x 로 r_mult × 반지름(+ extra) 떨어진 곳 = 그 손바닥이 사전 파지 자리."""

    def __init__(self, c, fk, extra=(0.0, 0.0, 0.0), hz=60.0):
        s = c.side()
        self.fk, self.hz, self.names = fk, hz, list(s.hand_joints)
        self.hand = np.asarray(s.hand_open, float).copy()
        self.arm = Q_MID.copy()
        r = float(c.object_geometry[OBJ]["radius"]) * float(c.ready_r_mult[0])
        self.cup = self.palm()[0] - np.array([r, 0.0, 0.0]) + np.asarray(extra, float)
        self.arm = np.asarray(s.arm_home, float).copy()
        self.arm_prev = self.arm.copy()

    def palm(self):
        p = self.fk.palm_pose(self.arm, self.hand)
        return p.palm_pos, p.extra["palm_rot"], p.tips

    def meas(self):
        pos, R, tips = self.palm()
        m = A.RaMeas(arm_q=self.arm.copy(), arm_qd=(self.arm - self.arm_prev) * self.hz,
                     hand_q=dict(zip(self.names, self.hand)), palm_pos=pos, palm_R=R, tips=tips,
                     cup_pos=self.cup.copy(), cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5))
        return {"arm": m}

    def apply(self, targets):
        arm, hand = targets["arm"]
        self.arm_prev, self.arm = self.arm.copy(), np.asarray(arm, float).copy()
        self.hand = np.asarray(hand, float).copy()


def _chain(c, nxt=None, cfg=None):
    pol = Toward()
    ch = AgltChain(c, pol)
    pol.ch = ch
    ch.object_name = OBJ
    ch.ag = AG.ApproachGrasp(cfg or AG.ApproachGraspCfg(handoff=1), c.policy_hz)
    ch.ag.next = nxt
    return ch


def _run(ch, plant, n, until=lambda ch: False):
    for _ in range(n):
        _, _, tg = ch.step(plant.meas())
        plant.apply(tg)
        if ch.ag.end is not None or until(ch):
            break


# ---------------------------------------------------------------- 계약
def test_contracts_carry_the_training_start_and_the_pregrasp_range(ap05, rcg03):
    assert rcg03.arm_start == "ready" and ap05.arm_start == "home"
    for c in (ap05, rcg03):                       # 같은 env(2판 ready_pre_*) — 접근 끝과 넘겨받는 시작이 같은 자리
        assert c.ready_r_mult == [2.5, 2.5] and c.ready_h_m == [0.0, 0.0]


def test_old_contracts_without_the_field_still_start_at_home():
    old = json.loads((REPO / "deploy/policies_retired/rh56f1/aglt/right_rcg02/rh_aglt_contract.json").read_text())
    assert "arm_start" not in old
    assert A.RaContract.__dataclass_fields__["arm_start"].default == "home"


def test_ready_needs_a_pregrasp_range(rcg03):
    with pytest.raises(A.RhAgltError):
        A.validate(A.with_run(rcg03, ready_r_mult=[]))
    with pytest.raises(A.RhAgltError):
        A.validate(A.with_run(rcg03, ready_r_mult=[0.9, 2.5]))     # 손바닥이 물체 안
    with pytest.raises(A.RhAgltError):
        A.validate(A.with_run(rcg03, arm_start="bank"))


def test_ready_hold_action_is_arm_zero_and_the_open_hand(rcg03):
    a = A.ready_hold_action(rcg03)
    assert a.shape == (13,) and np.all(a[:7] == 0.0)
    law = A.hand_law(rcg03)
    assert np.allclose(law.raw(a[7:]), rcg03.side().hand_open, atol=1e-9)   # 손 행동이 편 손으로 풀린다


def test_ready_pose_error_measures_the_ring_around_the_object(rcg03):
    c = A.select_object(rcg03, OBJ)
    cup = np.array([0.3, -0.2, 0.45])
    r = 0.03 * 2.5
    assert A.ready_pose_error(c, cup, cup + [r, 0, 0]) == pytest.approx((2.5, 0.0, 0.0))
    rr, out_h, out_r = A.ready_pose_error(c, cup, cup + [0, r + 0.04, 0.02])
    assert rr == pytest.approx((r + 0.04) / 0.03) and out_r == pytest.approx(0.04) and out_h == pytest.approx(0.02)
    with pytest.raises(A.RhAgltError):            # 물체를 고르기 전에는 반지름을 모른다
        A.ready_pose_error(rcg03, cup, cup)


# ---------------------------------------------------------------- 시작 검사 · reset
def test_ready_start_checks_the_pregrasp_pose_not_home(rcg03, fk):
    c = A.select_object(rcg03, OBJ)
    plant = Plant(c, fk)
    plant.arm = Q_MID.copy()                      # 홈에서 멀지만 사전 파지 자리
    m = plant.meas()
    assert not [r for r in aglt_start_refusals(c, m, 0.15) if "arm" in r or "palm" in r]
    plant.cup = plant.cup + np.array([-(READY_POSE_TOL_M + 0.02), 0.0, 0.0])   # 컵이 더 멀다
    assert any("pre-grasp" in r for r in aglt_start_refusals(c, plant.meas(), 0.15))


def test_home_contracts_still_refuse_a_far_arm(ap05, fk):
    c = A.select_object(ap05, OBJ)
    plant = Plant(c, fk)
    plant.arm = Q_MID.copy()
    assert any("training start pose" in r for r in aglt_start_refusals(c, plant.meas(), 0.15))


def test_ready_reset_starts_the_decoder_at_the_measured_arm_and_records_the_scripted_action(rcg03, fk):
    pol = Grasp()
    ch = AgltChain(rcg03, pol)
    ch.object_name = OBJ
    plant = Plant(A.select_object(rcg03, OBJ), fk)
    plant.arm = Q_MID.copy()
    ch.reset(plant.meas())
    assert np.allclose(ch.dec.state["arm"].arm_target, Q_MID)       # 홈으로 끌고 가지 않는다
    for _ in range(rcg03.hold_steps):
        _, _, tg = ch.step(plant.meas())
        assert np.allclose(tg["arm"][0], Q_MID)                      # 대기 동안 그 자리
        assert np.allclose(ch.prev, A.ready_hold_action(ch.c))       # sim 기록 행동
    ch.step(plant.meas())
    assert not np.allclose(ch.prev, A.ready_hold_action(ch.c))       # 대기 뒤는 정책 행동


# ---------------------------------------------------------------- 넘김
def test_the_approach_hands_over_to_the_ready_policy_in_the_same_node(ap05, rcg03, fk):
    nxt = Grasp()
    ch = _chain(ap05, (rcg03, nxt))
    plant = Plant(A.select_object(ap05, OBJ), fk)
    ch.reset(plant.meas())
    _run(ch, plant, 60 * 10, until=lambda ch: ch.ag.phase == "policy")
    assert ch.ag.phase == "policy" and ch.ag.end is None
    assert ch.policy is nxt and nxt.resets == 1
    assert ch.c.arm_start == "ready" and ch.c.active_object == OBJ
    assert np.allclose(ch.dec.state["arm"].arm_target, plant.arm, atol=1e-12)  # 접근 정책이 그 틱에 보낸 팔 목표에서(끊김 없음)
    assert np.allclose(ch.dec.state["arm"].hand_target, ch.c.side().hand_open)
    assert np.allclose(ch.goals.cup_reset, plant.cup)                          # 목표 = 넘김 순간 컵 + 오프셋
    assert ch.step_i == 0 and ch.ag.last["handoff_t_s"] > 0
    _run(ch, plant, 30)
    assert len(nxt.obs) == 30 and ch.ag.phase == "policy" and ch.ag.end is None   # 그 뒤는 넘겨받은 정책이 돈다
    assert not ch.ag.scripted


def test_without_a_next_policy_the_handoff_ends_the_node_with_stop(ap05, fk):
    ch = _chain(ap05, None)
    plant = Plant(A.select_object(ap05, OBJ), fk)
    ch.reset(plant.meas())
    _run(ch, plant, 60 * 10)
    assert ch.ag.end is not None and ch.ag.end[0] == "stop" and "사전 파지" in ch.ag.end[1]


def test_an_approach_that_never_reaches_the_pregrasp_pose_aborts(ap05, rcg03, fk):
    nxt = Grasp()
    ch = _chain(ap05, (rcg03, nxt), AG.ApproachGraspCfg(handoff=1, approach_max_s=3.0))
    plant = Plant(A.select_object(ap05, OBJ), fk, extra=(-0.06, 0.0, 0.0))   # 6 cm 더 먼 컵
    ch.reset(plant.meas())
    _run(ch, plant, 60 * 10)
    assert ch.ag.end is not None and ch.ag.end[0] == "abort" and nxt.resets == 0


def test_reset_after_a_handoff_brings_back_the_approach_policy(ap05, rcg03, fk):
    nxt = Grasp()
    ch = _chain(ap05, (rcg03, nxt))
    plant = Plant(A.select_object(ap05, OBJ), fk)
    ch.reset(plant.meas())
    _run(ch, plant, 60 * 10, until=lambda ch: ch.ag.phase == "policy")
    plant2 = Plant(A.select_object(ap05, OBJ), fk)
    ch.reset(plant2.meas())
    assert ch.policy is not nxt and ch.c.arm_start == "home" and ch.ag.phase == "approach"


def test_handoff_time_covers_the_approach_and_the_next_policy():
    cfg = AG.ApproachGraspCfg(handoff=1)
    assert cfg.worst_s == cfg.approach_max_s + AG.DEFAULT_EPISODE_S
    assert AG.episode_s({"handoff": 1}) == 33.0
    with pytest.raises(AG.ApproachGraspError):
        AG.ApproachGraspCfg.parse("handoff=2")


# ---------------------------------------------------------------- 에피소드 · 미션
def test_episode_needs_both_the_handoff_key_and_the_policy_id():
    from policy_control import episode_spec as S
    raw = {"episode": {"name": "x"}, "objects": {"CUP": {"topic": "/objects/a/pose"}},
           "policies": {"r": {"policy": "rh56f1/aglt/right_approach_ap05", "kind": "aglt", "side": "right",
                              "approach_grasp": {"handoff": 1}}},
           "sequence": [{"id": "d", "type": "terminal", "result": "SUCCESS"}]}
    with pytest.raises(S.EpisodeSpecError):
        S.parse(raw)
    raw["policies"]["r"]["handoff"] = "rh56f1/aglt/right_rcg03"
    raw["policies"]["r"]["approach_grasp"] = {}
    with pytest.raises(S.EpisodeSpecError):
        S.parse(raw)


# ★10.20 저녁 사용자 "grasp_d 를 1-b 접근에 rcg03 으로 교체": grasp_d · tilt_base_d 는 더 이상 넘김을 쓰지 않는다 —
#   에피소드 · 미션 검사는 test_pc_path_d_reach.py. 넘김 기구(approach_grasp handoff)는 위 시험이 그대로 지킨다.


# ---------------------------------------------------------------- ★10.20 멈춤 = 학습 접근 성공(pos · 손바닥 still)
class Jitter(Plant):
    """실기 10.20: 손바닥은 서 있는데 관절 속도 실측이 0.3 rad/s 안팎(팔꿈치 · 손목을 계속 돌림)."""

    def meas(self):
        m = super().meas()
        return {"arm": A.RaMeas(**{**m["arm"].__dict__, "arm_qd": np.full(7, 0.3)})}


def test_joint_speed_noise_does_not_block_the_handoff(ap05, rcg03, fk):
    nxt = Grasp()
    ch = _chain(ap05, (rcg03, nxt))
    plant = Jitter(A.select_object(ap05, OBJ), fk)
    ch.reset(plant.meas())
    _run(ch, plant, 60 * 10, until=lambda ch: ch.ag.phase == "policy")
    assert ch.ag.phase == "policy" and ch.policy is nxt          # 옛 규칙(관절 < 0.05 rad/s)이면 시간 초과로 abort


def _ring(c, cup, out_r=0.0, out_h=0.0):
    r = float(c.object_geometry[OBJ]["radius"]) * float(c.ready_r_mult[0])
    return cup + np.array([r + out_r, 0.0, out_h])


def _feed(ag, chain, palms, cup):
    for p in palms:
        m = A.RaMeas(arm_q=Q_MID, arm_qd=np.zeros(7), hand_q={}, palm_pos=np.asarray(p, float), palm_R=np.eye(3),
                     tips=np.zeros((5, 3)), cup_pos=cup, cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5))
        ag.after_policy(chain, m)
        if ag.phase != "approach":
            return


def _bare(ap05):
    from types import SimpleNamespace as NS
    c = A.select_object(ap05, OBJ)
    seen = []
    chain = NS(c=c, step_i=10 ** 6, handover=lambda base, pol, m: seen.append(m))
    ag = AG.ApproachGrasp(AG.ApproachGraspCfg(handoff=1), c.policy_hz)
    ag.next = (None, None)
    return ag, chain, seen, c


def test_a_palm_moving_faster_than_the_training_limit_is_not_still(ap05):
    ag, chain, seen, c = _bare(ap05)
    cup = np.array([0.3, -0.2, 0.3])
    wob = [_ring(c, cup) + [0.0, (-1) ** k * 0.0004, 0.0] for k in range(240)]   # 0.0008 m/틱 = 0.048 m/s
    _feed(ag, chain, wob, cup)
    assert ag.phase == "approach" and not seen and ag.last["palm_mps"] == pytest.approx(0.048)
    _feed(ag, chain, [_ring(c, cup)] * 40, cup)
    assert ag.phase == "policy" and len(seen) == 1                         # 서고 30 틱(0.5 s)
    assert 0.5 - 1e-9 <= ag.last["handoff_t_s"] - 240 / c.policy_hz <= 0.55


def test_the_ring_distance_is_the_hypot_like_training(ap05):
    ag, chain, seen, c = _bare(ap05)
    cup = np.array([0.3, -0.2, 0.3])
    _feed(ag, chain, [_ring(c, cup, 0.015, 0.015)] * 120, cup)            # 각 1.5 cm · 합 2.1 cm — 옛 상자 규칙은 통과
    assert ag.phase == "approach" and not seen
    _feed(ag, chain, [_ring(c, cup, 0.012, 0.012)] * 60, cup)             # 합 1.7 cm
    assert ag.phase == "policy"


def test_still_mps_is_range_checked():
    assert AG.ApproachGraspCfg().still_mps == 0.03                        # hdgp approach_succ_speed_mps
    for bad in ("still_mps=0", "still_mps=0.5"):
        with pytest.raises(AG.ApproachGraspError):
            AG.ApproachGraspCfg.parse(bad)

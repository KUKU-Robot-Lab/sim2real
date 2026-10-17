"""경로 C(10.17 사용자 승인) — 접근 정책 → 스크립트 닫기 → 쥠 확인(네 손가락 2 축 래치 + 8 cm) → IK 들기 → 3 s 유지.

정책은 0 행동(팔 · 손 그대로)으로 대신하고, 손은 컵에 닿는 축만 그 각도에서 멈추는 간단한 플랜트로 돌린다.
계약은 deploy/policies/rh56f1/aglt/right_approach_rg3wc(rg3wc ep800 원본 — 관측 94 · 행동 13, rg3wo 와 해석이 같다).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from policy_control import approach_grasp as AG
from policy_control import hand_block as HB
from policy_control import rh_aglt as A
from policy_control.pour_fj_node import AgltChain

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "deploy/policies/rh56f1/aglt/right_approach_rg3wc/rh_aglt_contract.json"
URDF = Path.home() / "rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
Q_MID = np.array([-0.6, 0.2026, 0.6538, 1.7608, 0.3791, 0.2, 0.6646])   # test_pc_rh_cap 과 같은 관절 한계에서 먼 자세


@pytest.fixture(scope="module")
def c() -> A.RaContract:
    return A.select_object(A.load_contract(CONTRACT), "source240_real")


@pytest.fixture(scope="module")
def fk(c):
    if not URDF.is_file():
        pytest.skip("RH56F1 URDF 없음")
    from policy_control.fk_numpy import UrdfChainFK
    s = c.side()
    return UrdfChainFK(URDF, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)


class Toward:
    """접근 정책 대역 — 손은 편 채, 팔은 Q_MID 로(증분 행동). 체인의 디코더 목표를 읽는다."""

    def __init__(self):
        self.ch = None

    def forward(self, obs):
        assert obs.shape == (94,)
        c, st = self.ch.c, self.ch.dec.state["arm"]
        a_arm = np.clip((Q_MID - st.arm_target) / np.asarray(c.k_arm, float), -1.0, 1.0)
        return np.concatenate([a_arm, AG.hand_action_for(self.ch.dec.law, np.asarray(c.side().hand_open, float))])

    def reset(self):
        pass


class Plant:
    """팔 = 목표 완전 추종(속도는 차분), 손 = 목표를 따라가다 컵에 닿는 축(stop 비율)은 그 각도에서 멈춘다."""

    def __init__(self, c, fk, cup_rel=(0.0, 0.0, -0.05), stop=None, hz=60.0):
        s = c.side()
        self.c, self.fk, self.hz, self.names = c, fk, hz, list(s.hand_joints)
        self.o, self.g = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
        self.hand = self.o.copy()
        self.arm = Q_MID.copy()
        self.palm_mid = self.palm()[0]                               # 접근 끝 손바닥(팔이 Q_MID 에 섰을 때)
        self.cup = self.palm_mid + np.asarray(cup_rel, float)
        self.arm = np.asarray(s.arm_home, float).copy()              # 접근은 홈에서 시작(aglt reset 자세)
        self.arm_prev = self.arm.copy()
        stop = stop or {}
        self.limit = np.array([self.o[i] + stop.get(n.split("_hj_")[-1], 1.0) * (self.g[i] - self.o[i])
                               for i, n in enumerate(self.names)])

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
        sgn = np.sign(self.g - self.o)
        want = np.asarray(hand, float)
        self.hand = np.where(sgn * (want - self.limit) > 0, self.limit, want)   # 닫는 쪽으로 컵을 못 넘는다


def _chain(c, fk, cfg=None):
    pol = Toward()
    ch = AgltChain(c, pol)
    pol.ch = ch
    ch.object_name = "source240_real"
    ch.ag = AG.ApproachGrasp(cfg or AG.ApproachGraspCfg(), c.policy_hz)
    ch.fk = fk
    return ch


def _run(ch, plant, n):
    phases = []
    for _ in range(n):
        _, _, tg = ch.step(plant.meas())
        plant.apply(tg)
        phases.append(ch.ag.phase)
        if ch.ag.end is not None:
            break
    return phases


CUP_STOP = {"thumb_2": 0.55, "index_1": 0.5, "middle_1": 0.5, "ring_1": 0.6, "pinky_1": 0.65}


def test_cfg_param_off_on_overrides_and_unknown_keys():
    P = AG.ApproachGraspCfg
    assert P.parse("") is None and P.parse("on") == P()
    assert P.parse("stop_palm_m=0.05,judge_min_axes=3") == P(stop_palm_m=0.05, judge_min_axes=3)
    assert P.to_param({}) == "on" and P.to_param({"stop_palm_m": 0.05}) == "stop_palm_m=0.05"
    for bad in ("stop_palm=0.05", "stop_palm_m", "stop_palm_m=x", "judge_min_axes=5", '{"stop_palm_m": 0.05}'):
        with pytest.raises(AG.ApproachGraspError):
            P.parse(bad)


def test_close_contract_is_opp_k_with_the_deploy_block_nominal(c):
    cc = AG.close_contract(c)
    A.validate(cc)
    assert cc.grasp_rule == HB.GRASP_RULES[1] and cc.grasp_judge_axes == list(HB.FINGERS)
    assert cc.hand_block["g_rad"] == A.HAND_BLOCK_NOMINAL["g_rad"] and cc.hand_block["release_on_open"] is False
    assert (cc.obs_dim, cc.action_dim, cc.k_arm) == (c.obs_dim, c.action_dim, c.k_arm)   # 관측 · 행동 해석은 그대로


def test_hand_action_inverts_the_hand_law(c):
    law = A.hand_law(c)
    lo, hi = law.bounds()
    q = lo + 0.3 * (hi - lo)
    assert law.raw(AG.hand_action_for(law, q)) == pytest.approx(q)


def test_approach_stops_near_and_still_then_full_success(c, fk):
    plant = Plant(c, fk, cup_rel=(0.0, 0.0, -0.05), stop=CUP_STOP)
    ch = _chain(c, fk)
    ch.reset(plant.meas())
    phases = _run(ch, plant, 60 * 20)
    first_close = phases.index("close")
    still = round(AG.ApproachGraspCfg().still_s * c.policy_hz)
    assert c.hold_steps + still - 1 <= first_close < 60 * 10         # 대기 뒤, 팔이 서고 0.5 s 지나서(시간 한도 전)
    assert ch.ag.end is not None and ch.ag.end[0] == "stop", ch.ag.as_dict()
    assert {"lift", "settle"} <= set(phases)
    assert phases.count("settle") >= round(3.0 * c.policy_hz) - 1    # 3 s 유지
    rise = plant.palm()[0][2] - ch.ag.z0
    assert 0.09 - 1e-3 <= rise <= 0.105
    i_rot = [i for i, n in enumerate(plant.names) if n.endswith("thumb_1")][0]
    assert plant.hand[i_rot] == pytest.approx(np.asarray(c.side().hand_open)[i_rot], abs=1e-6)   # 엄지 회전은 접근 끝 그대로


def test_thumb_flex_waits_and_latched_axes_stop_closing(c, fk):
    plant = Plant(c, fk, stop=CUP_STOP)
    ch = _chain(c, fk)
    ch.reset(plant.meas())
    while ch.ag.phase == "approach":
        _, _, tg = ch.step(plant.meas())
        plant.apply(tg)
    flex = [i for i, n in enumerate(plant.names) if n.endswith("thumb_2")][0]
    q_flex0 = ch.ag.q_cmd[flex]
    for _ in range(round(AG.ApproachGraspCfg().thumb_delay_s * c.policy_hz) - 1):
        _, _, tg = ch.step(plant.meas())
        plant.apply(tg)
    assert ch.ag.q_cmd[flex] == pytest.approx(q_flex0)
    _run(ch, plant, 60 * 3)
    blk = ch.dec.block
    q = ch.ag.q_cmd.copy()
    for _ in range(30):
        _, _, tg = ch.step(plant.meas())
        plant.apply(tg)
        if ch.ag.end is not None:
            break
    assert np.all(ch.ag.q_cmd[blk.latch] == pytest.approx(q[blk.latch]))   # 래치 축은 더 닫지 않는다


def test_no_cup_means_no_lift_and_abort(c, fk):
    plant = Plant(c, fk, stop={})                                   # 아무 축도 막히지 않는다(빈손)
    ch = _chain(c, fk)
    ch.reset(plant.meas())
    phases = _run(ch, plant, 60 * 20)
    assert ch.ag.end is not None and ch.ag.end[0] == "abort" and "못 쥠" in ch.ag.end[1]
    assert "lift" not in phases
    assert plant.palm()[0] == pytest.approx(plant.palm_mid, abs=2e-3)   # 팔은 접근 끝 자리 그대로(들지 않는다)


def test_far_cup_is_not_a_grasp(c, fk):
    cfg = AG.ApproachGraspCfg(approach_max_s=1.0)                     # 멀어서 거리 조건으로는 안 멈춘다 → 시간으로 닫기
    plant = Plant(c, fk, cup_rel=(0.0, 0.0, -0.12), stop=CUP_STOP)   # 손가락은 막혀도(다른 물건) 컵이 12 cm 밖
    ch = _chain(c, fk, cfg)
    ch.reset(plant.meas())
    phases = _run(ch, plant, 60 * 20)
    assert "close" in phases and "lift" not in phases
    assert ch.ag.end[0] == "abort"


def test_grasp_loss_during_settle_aborts(c, fk):
    plant = Plant(c, fk, stop=CUP_STOP)
    ch = _chain(c, fk)
    ch.reset(plant.meas())
    while ch.ag.phase != "settle" and ch.ag.end is None:
        _, _, tg = ch.step(plant.meas())
        plant.apply(tg)
    assert ch.ag.phase == "settle"
    plant.limit = plant.g.copy()                                    # 컵이 빠졌다 — 손가락이 끝까지 닫힌다
    _run(ch, plant, 60 * 4)
    assert ch.ag.end is not None and ch.ag.end[0] == "abort" and "풀림" in ch.ag.end[1]


def test_reset_brings_back_the_approach_decoder(c, fk):
    plant = Plant(c, fk, stop=CUP_STOP)
    ch = _chain(c, fk)
    ch.reset(plant.meas())
    while not ch.ag.scripted:
        _, _, tg = ch.step(plant.meas())
        plant.apply(tg)
    assert ch.c.grasp_judge_axes == list(HB.FINGERS) and ch.c.grasp_rule == HB.GRASP_RULES[1]
    ch.reset(Plant(c, fk, stop=CUP_STOP).meas())
    assert ch.ag.phase == "approach" and ch.c.grasp_rule == c.grasp_rule and ch.dec.c is ch.c


def test_contract_matches_rg3wo_interpretation():
    """접근 정책과 rg3wo 는 관측 · 행동 해석이 같다(10.17 계약 비교) — 쥠 판정 키만 다르다."""
    a = json.loads(CONTRACT.read_text())
    b = json.loads((REPO / "deploy/policies/rh56f1/aglt/right_rg3wo/rh_aglt_contract.json").read_text())
    skip = {"checkpoint", "checkpoint_md5", "env_yaml_sha1", "agent_yaml_sha1", "run_dir", "notes", "hand_block",
            "grasp_rule", "grasp_judge_axes", "grasp_min_axes", "grasp_near_m"}
    assert {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}


def test_the_left_approach_is_the_right_one_seen_in_a_mirror():
    """왼손 접근 = 오른손 rg3wc ep800 거울 체크포인트(10.17) — 관측 94(촉각 없음 + one-hot 3)에서 a_R(o) = S · π_L(D · o)."""
    pytest.importorskip("torch")
    import test_pc_rh_aglt as T
    pol = REPO / "deploy/policies/rh56f1/aglt"
    cr = A.select_object(A.load_contract(pol / "right_approach_rg3wc/rh_aglt_contract.json"), "source240_real")
    cl = A.select_object(A.load_contract(pol / "left_approach_rg3wc/rh_aglt_contract.json"), "source240_real")
    if not (Path(cr.checkpoint).is_file() and Path(cl.checkpoint).is_file()):
        pytest.skip("가중치 없음(.gitignore)")
    from policy_control.joint_policy import JointPolicy
    pr, pl = JointPolicy(cr, "cpu"), JointPolicy(cl, "cpu")
    arm, M, MN = list(T.ARM_MIRROR_SIGN), list(T._M), list(T._MN)
    D = np.array(arm * 3 + [1.0] * 12 + M + MN + M + M + M + M * 5 + M * 4 + M + arm + [1.0] * 6 + [1.0] * 3)
    S = np.array(T.ACT_MIRROR)
    assert D.size == cr.obs_dim == 94
    dec = A.RaDecoder(cr)
    m = T._meas(cr)
    goal = A.first_goal(cr, m.cup_pos, m.cup_quat)
    rng = np.random.default_rng(3)
    prev, worst = np.zeros(13), 0.0
    for _ in range(60):
        o = A.build_obs(cr, m, dec, goal, prev) + rng.normal(0, 0.05, 94)
        a_r, a_l = pr.forward(o), pl.forward(D * o)
        worst = max(worst, float(np.abs(a_r - S * a_l).max()))
        prev = np.clip(a_r, -1, 1)
        dec.step(a_r, active=True, hand_q=np.asarray(cr.side().hand_open, float), obj_palm_m=0.1)
    assert worst < 1e-4, worst


@pytest.mark.parametrize("text", ["judge_ticks=0", "lift_m=1.0", "lift_m=0", "near_m=0.5", "loss_ticks=1000",
                                  "lift_damping=0", "lift_speed=0.5", "approach_max_s=0", "lift_tol_m=0.03,lift_m=0.03"])
def test_unsafe_values_are_refused_so_no_value_can_switch_off_a_gate(text):
    # 10.17 리뷰: judge_ticks=0 은 래치 없이 들고, lift_m · near_m · loss_ticks · damping 은 상한 · 하한이 없었다
    with pytest.raises(AG.ApproachGraspError):
        AG.ApproachGraspCfg.parse(text)


def test_episode_length_covers_the_slowest_path_and_off_keeps_the_aglt_default():
    cfg = AG.ApproachGraspCfg()
    slowest = cfg.approach_max_s + cfg.close_max_s + cfg.lift_m / cfg.lift_speed + cfg.lift_extra_s + cfg.hold_s
    assert AG.episode_s({}) >= slowest + AG.EPISODE_PAD_S - 1e-9
    assert AG.episode_s({"approach_max_s": 20}) == AG.episode_s({}) + 5.0
    assert AG.episode_s(None) == AG.DEFAULT_EPISODE_S == 15.0


def test_the_mission_and_the_runner_use_the_same_node_time_limit():
    import yaml
    from policy_control import episode_runner as R
    from policy_control import episode_spec as S
    repo = Path(__file__).resolve().parents[2]
    ep = S.load(repo / "config" / "episodes" / "grasp_c_right.yaml")
    assert R._node_max_s(ep.policies["rh_aglt_r"]) == AG.episode_s({})
    mission = yaml.safe_load((repo / "config" / "mission_rh56f1_fake.yaml").read_text())
    text = json.dumps(mission)
    assert f"max_episode_s:={AG.episode_s({}):.1f}" in text

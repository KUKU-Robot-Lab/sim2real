"""★10.25 준비 자세 이송(place_ready_carry) — newton_07_r · 06_l 계약 · PlaceChain 이송 → 정책 순서 · n05 는 그대로."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control import place_ready_carry as RCY
from policy_control import rh_aglt as A
from policy_control import rh_place as P

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy/policies/rh56f1/place"
HDGP = REPO.parent / "hdgp"


@pytest.mark.parametrize("run", ["right_n07", "left_n06"])
def test_n07_n06_contracts_carry_to_the_trained_anchor(run):
    c = P.load_contract(POL / run / "rh_place_contract.json")
    cfg = RCY.ReadyCarryCfg.from_dict(c.ready_carry)
    assert (cfg.dz, cfg.v_max, cfg.qd_max, cfg.plan_down, cfg.max_steps) == (0.085, 0.0015, 0.012, 0.04, 600)
    assert c.latch_s2r and not c.carry and c.attach_source == "bundle"


@pytest.mark.parametrize("run", ["right_n05", "left_n05"])
def test_n05_has_no_ready_carry(run):
    assert P.load_contract(POL / run / "rh_place_contract.json").ready_carry == {}


def test_carry_end_runs_refuse_to_build_without_ready_carry(tmp_path):
    import sys
    sys.path.insert(0, str(REPO / "deploy/policy_control/tools"))
    import build_rh_aglt_contract as B
    assert "carry_end" in A.read_env(POL / "right_n07/params/env.yaml")["start_bank_path"]
    with pytest.raises(P.RhPlaceError, match="--ready-carry"):
        B.main(["--run", str(POL / "right_n07"), "--attach", "bundle", "--out", str(tmp_path / "c.json")])


class _Pol:
    def __init__(self):
        self.n, self.resets = 0, 0

    def reset(self):
        self.resets += 1

    def forward(self, obs):
        self.n += 1
        return np.full(13, 0.3)


class _Est:
    def __init__(self):
        self.n, self.resets = 0, 0

    def reset(self):
        self.resets += 1

    def step(self, obs):
        self.n += 1
        return np.zeros(6)

    def rel_pose(self, p, q):
        from policy_control import place_latch_est as LE
        return LE.rel_pose(p, q, np.zeros(6))

    def as_dict(self):
        return {}


NO_HDGP = not (HDGP / "source/openarm/openarm/agnostic/modules/ready_carry").is_dir()


def _setup(policy="right_n07"):
    c = P.load_contract(POL / f"{policy}/rh_place_contract.json")
    s = c.side()
    RC, mch = RCY.load_chain(s.palm_body, list(s.arm_joints))
    pol, est = _Pol(), _Est()
    ch = P.PlaceChain(c, pol)
    ch.object_name, ch.est = "source240_real", est
    hand = np.asarray(s.hand_grip, float) * 0.5
    q = np.array([-0.0276, 0.9205, -0.6622, 0.7405, -0.1584, 0.7484, 1.5155])   # sim 잡기 뱅크 1023 r 8 행(들기 끝 실측)
    att = (np.array([0.0167, 0.0376, 0.0313]), np.array([0.4015, 0.5551, 0.4426, 0.5786]))   # 같은 행 컵 ← 손바닥
    holder = np.array([0.38, -0.20, 0.205 + P.PC.HOLDER_BASE_DZ])

    def meas(qa):
        import torch
        T = mch.fk(torch.tensor([list(qa)], dtype=torch.float32))[0].double().numpy()
        palm = T[:3, 3]
        return {"arm": A.RaMeas(arm_q=np.asarray(qa), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, hand)),
                                palm_pos=palm, palm_R=T[:3, :3], tips=np.tile(palm, (5, 1)), cup_pos=np.zeros(3),
                                cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5), joint_force=None)}

    return c, RC, mch, pol, est, ch, hand, q, att, holder, meas


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_chain_carries_first_then_starts_the_policy_from_zero():
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup()
    with pytest.raises(P.RhPlaceError, match="Mover 사슬"):
        ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    assert ch.carrying and ch.rcarry.plan["T_steps"] > 0
    r0 = pol.resets
    for _ in range(c.ready_carry["max_steps"] + 200):
        _, a, _ = ch.step(meas(q))
        if not ch.carrying:
            break
        assert pol.n == 0 and est.n == 0 and not ch.latch                 # 이송 중 정책 · 추정기 안 돈다
        q = ch.dec.state["arm"].arm_target.copy()                  # 디코더 팔 목표(EMA · ±1 행동) 완벽 추종
    d = ch.rcarry.as_dict()
    assert not ch.carrying and d["carry_ok"] and d["arrived"] and d["err_mm"] < 10.0
    assert pol.resets == r0 + 1 and est.resets >= 1 and ch.step_i == 0 and np.all(ch.prev == 0)
    ch.step(meas(q))
    assert pol.n == 1 and est.n == 1                                   # 다음 틱부터 정책
    assert ch.as_dict()["phase"] == "place" and ch.as_dict()["ready_carry"]["arrived"]


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
@pytest.mark.parametrize("policy", ["right_n07", "right_n10_e4000"])
def test_carry_dq_never_exceeds_the_action_limit(policy):
    """★11.10 Mover 관절 상한 = k_arm·arm_ema — 행동이 ±1 에 잘리지 않아 계획대로 도착한다(실기 161342: 잘린 틱 95 %, 끝 err 48 mm)."""
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup(policy)
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    rc = ch.rcarry
    assert rc.qd_max == pytest.approx(c.k_arm * c.arm_ema) and rc.qd_max < c.ready_carry["qd_max"]
    raw, orig = [], rc.mv.step

    def step(qq, _o=orig):
        out = _o(qq)
        raw.append(float(out[0].abs().max()) / rc.scale)
        return out
    rc.mv.step = step
    while ch.carrying:
        ch.step(meas(q))
        q = ch.dec.state["arm"].arm_target.copy()
    assert max(raw) <= 1.0 + 1e-5
    assert rc.start_ok and rc.as_dict()["err_mm"] < 10.0


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_carry_that_does_not_reach_the_start_does_not_start_the_policy():
    """★11.10 팔이 따라가지 못해 컵이 sim 시작 분포 밖에서 이송이 끝나면 정책 상태를 열지 않고 다음 틱을 거부한다(노드는 abort)."""
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup()
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    r0 = pol.resets
    for _ in range(c.ready_carry["max_steps"] + 200):              # 실측 q 고정 = 팔이 안 움직임 → 계획 시간 + settle 뒤 끝
        ch.step(meas(q))
        if not ch.carrying:
            break
    d = ch.rcarry.as_dict()
    assert not ch.carrying and not d["start_ok"] and d["err_mm"] > RCY.START_ERR_MAX_M * 1000
    assert pol.resets == r0 and pol.n == 0
    with pytest.raises(P.RhPlaceError, match="시작 분포 밖"):
        ch.step(meas(q))
    assert pol.n == 0

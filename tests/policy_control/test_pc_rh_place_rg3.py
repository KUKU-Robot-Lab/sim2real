"""10.12 새 place 계약(촉각 없는 actor + 물체 one-hot + 관측 래치 1 칸, 막힘 기반 놓음) — hdgp_place PlaceEnvMixin 과 같은 식."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from policy_control import rh_aglt as A
from policy_control import rh_place as P

RUN = Path(__file__).resolve().parents[2] / "deploy/policies/rh56f1/aglt/right_rg3mft"
BLOCK = {"g_rad": 0.0218, "eps_rad": 0.0109, "n_ticks": 3, "lag_ticks": 6,
         "axes": ["thumb_flex", "index", "middle", "ring", "pinky"]}
LATCH = {"dz": 0.055, "dxy": 0.004, "tilt_deg": 20.0, "bottom_xy": True}
OPEN = [0.0, 0.10, 0.10, 0.10, 0.10]
HOLDER = np.array([0.30, -0.20, 0.20])


def _contract(**kw) -> P.PlaceContract:
    a = A.load_contract(RUN / "rh_aglt_contract.json")
    raw = {k: v for k, v in asdict(a).items() if not k.startswith("goal_")}
    raw["sides"] = a.sides
    raw.update(schema=P.SCHEMA, task="open-rh_r_place", hold_steps=0, goal_offset=[0.0, 0.0, 0.0], actor_tactile_obs=False,
               obs_dim=95, hand_block=BLOCK, grasp_rule="thumb_flex_and_any_finger", hand_lead_cap_rad=0.2,
               extra_actor_obs=["latch"], seat_dz=0.06, target_holders=[1], release_steps=5, settle_steps=30,
               release_force_n=1.0, latch_obs=dict(LATCH), release_rule="block_follow_open", release_open_rad=list(OPEN))
    raw.update(kw)
    c = P.PlaceContract(**raw)
    P.validate(c)
    return c


class _Pol:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def forward(self, obs):
        return self.a


def _meas(c, hand, cup_z=0.30, cup_xy=(0.30, -0.20)):
    s = c.side()
    palm = np.array([cup_xy[0], cup_xy[1], cup_z + 0.02])
    return A.RaMeas(arm_q=np.asarray(s.arm_home), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, hand)),
                    palm_pos=palm, palm_R=np.eye(3), tips=np.tile(palm, (5, 1)),
                    cup_pos=np.array([cup_xy[0], cup_xy[1], cup_z]), cup_quat=np.array([1.0, 0, 0, 0]),
                    tactile_n=np.zeros(5), joint_force=None)


def test_the_contract_is_95_with_the_latch_last_and_refuses_bad_settings():
    c = _contract()
    assert A.obs_dim_of(c) == 95 and A.actor_layout(c)[-2:] == (("object_id", 3), ("latch", 1))
    with pytest.raises(P.RhPlaceError, match="release_open_rad"):
        _contract(release_open_rad=[0.0] * 5)
    with pytest.raises(P.RhPlaceError, match="release_rule"):
        _contract(release_rule="magic")
    with pytest.raises(A.RhAgltError):
        _contract(obs_dim=94)                                      # latch 칸이 빠진 폭은 거부


def test_the_latch_follows_the_seat_rule_and_stays_on():
    c = _contract()
    hand = np.asarray(c.side().hand_grip, float) * 0.5
    ch = P.PlaceChain(c, _Pol(np.zeros(13)))
    ch.object_name = "source240_real"
    seat_z = HOLDER[2] + c.seat_dz
    ch.reset({"arm": _meas(c, hand, cup_z=seat_z + 0.10)}, holder=HOLDER, held=(np.asarray(c.side().arm_home), hand))
    obs, _, _ = ch.step({"arm": _meas(c, hand, cup_z=seat_z + 0.10)})
    assert obs.size == 95 and obs[-1] == 0.0                       # 자리 위 10 cm — 아직
    obs, _, _ = ch.step({"arm": _meas(c, hand, cup_z=seat_z + 0.03)})
    assert obs[-1] == 1.0                                          # dz 0.055 안 · 바닥 중심 xy 0 · 기울기 0
    obs, _, _ = ch.step({"arm": _meas(c, hand, cup_z=seat_z + 0.20)})
    assert obs[-1] == 1.0                                          # 리셋까지 유지


def test_the_latch_needs_the_bottom_centre_within_dxy():
    c = _contract()
    hand = np.asarray(c.side().hand_grip, float) * 0.5
    ch = P.PlaceChain(c, _Pol(np.zeros(13)))
    ch.object_name = "source240_real"
    seat_z = HOLDER[2] + c.seat_dz
    ch.reset({"arm": _meas(c, hand, cup_z=seat_z)}, holder=HOLDER, held=(np.asarray(c.side().arm_home), hand))
    obs, _, _ = ch.step({"arm": _meas(c, hand, cup_z=seat_z, cup_xy=(0.306, -0.20))})   # 6 mm 옆
    assert obs[-1] == 0.0


def test_an_unknown_object_is_refused_at_reset():
    c = _contract()
    hand = np.asarray(c.side().hand_grip, float) * 0.5
    ch = P.PlaceChain(c, _Pol(np.zeros(13)))
    with pytest.raises(A.RhAgltError, match="물체"):
        ch.reset({"arm": _meas(c, hand)}, holder=HOLDER, held=(np.asarray(c.side().arm_home), hand))


def test_release_needs_open_fingers_that_follow_and_the_latch():
    c = _contract()
    s = c.side()
    lo, hi = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
    held = lo + 0.6 * (hi - lo)
    open_a = np.concatenate([np.zeros(7), P.open_hand_action(c)])  # 정책이 손을 편다
    ch = P.PlaceChain(c, _Pol(open_a))
    ch.object_name = "source240_real"
    seat_z = HOLDER[2] + c.seat_dz
    ch.reset({"arm": _meas(c, held, cup_z=seat_z + 0.20)}, holder=HOLDER, held=(np.asarray(s.arm_home), held))
    for _ in range(40):                                            # 래치 없이 손을 펴도 놓음이 아니다(자리 위 20 cm)
        tgt = ch.dec.state["arm"].hand_target.copy()
        ch.step({"arm": _meas(c, tgt, cup_z=seat_z + 0.20)})       # 실측 = 목표(막힘 없음, 따라감)
    assert not ch.settling and ch.free_streak >= c.release_steps
    ch.step({"arm": _meas(c, ch.dec.state["arm"].hand_target.copy(), cup_z=seat_z + 0.02)})
    assert ch.settling                                             # 래치가 켜진 스텝에 놓음
    assert ch.as_dict()["rule"] == "block_follow_open" and ch.as_dict()["latch"] is True


def test_a_finger_still_pressing_the_cup_blocks_the_release():
    c = _contract()
    s = c.side()
    lo, hi = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
    held = lo + 0.6 * (hi - lo)
    ch = P.PlaceChain(c, _Pol(np.concatenate([np.zeros(7), P.open_hand_action(c)])))
    ch.object_name = "source240_real"
    seat_z = HOLDER[2] + c.seat_dz
    ch.reset({"arm": _meas(c, held, cup_z=seat_z)}, holder=HOLDER, held=(np.asarray(s.arm_home), held))
    i_mid = s.hand_joints.index("r_hj_middle_1")
    for _ in range(40):
        q = ch.dec.state["arm"].hand_target.copy()
        q[i_mid] = held[i_mid]                                     # 중지는 컵에 걸려 안 열린다 → 목표를 못 따라감
        ch.step({"arm": _meas(c, q, cup_z=seat_z)})
    assert not ch.settling


def test_latch_settings_come_from_the_curriculum_end():
    env = {"release_latch_obs": True, "latch_dz": 0.055, "latch_dxy": 0.01, "latch_tilt_deg": 20.0, "latch_bottom_xy": True,
           "latch_obs_requires_grasp": False, "curr_enabled": True, "curr_latch_dxy": (0.01, 0.004),
           "curr_num_increments": 6, "curr_start_increment": 6}
    assert P.latch_obs_of(env) == LATCH
    with pytest.raises(P.RhPlaceError, match="커리큘럼"):
        P.latch_obs_of(dict(env, curr_start_increment=3))
    with pytest.raises(P.RhPlaceError, match="쥠"):
        P.latch_obs_of(dict(env, latch_obs_requires_grasp=True))
    assert P.latch_obs_of({}) == {}

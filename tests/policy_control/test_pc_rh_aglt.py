"""rh_aglt 계열(RH56F1 한 팔 aglt) — 계약 · 디코더 · 관측 96 · 노드 체인. 09.30 사용자: RH56F1 정책을 deploy 에 연결."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_control import pour_fj_node as N
from policy_control import rh56f1_hand as H
from policy_control import rh_aglt as A

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy" / "policies"
RIGHT, LEFT = POL / "right_rh_aglt_i03" / "rh_aglt_contract.json", POL / "left_rh_aglt_i05" / "rh_aglt_contract.json"
HOMES = REPO / "deploy/policy_control/config/homes/rh56f1_aglt.yaml"
HDGP = Path.home() / "rl_ws/hdgp/source/openarm/openarm/agnostic"


def _load_hdgp(rel: str):
    path = HDGP / rel
    if not path.is_file():
        pytest.skip(f"hdgp 없음: {path}")
    spec = importlib.util.spec_from_file_location("_hdgp_" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                   # dataclass 가 모듈을 찾는다
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", params=[RIGHT, LEFT], ids=["right", "left"])
def c(request):
    return A.load_contract(request.param)


def test_the_contracts_match_the_training_runs_and_the_mission_home(c):
    s = c.side()
    assert (c.obs_dim, c.action_dim, c.policy_hz, c.hold_steps, c.episode_s) == (96, 13, pytest.approx(60.0), 10, 15.0)
    assert c.recurrent and c.goal_offset == pytest.approx([0.0, 0.0, 0.14])
    homes = yaml.safe_load(HOMES.read_text())
    assert s.arm_home == pytest.approx(homes[s.side])                    # 미션 홈 = 학습 시작 자세
    p = s.side[0]
    assert s.hand_joints == [f"{p}_hj_{j}" for j in H.JOINTS] and s.palm_body == f"{p}_hl_palm_sensor"
    assert s.tip_bodies == [f"{p}_hl_{f}_tip" for f in H.FINGERS]
    assert s.hand_open == pytest.approx([1.57, 0, 0, 0, 0, 0]) and s.hand_grip == pytest.approx([1.2, 0.24, 1.08, 1.08, 0.85, 0.85])
    assert A.hand_law(c).hold == "open" and A.hand_law(c).range_mode == "grip"
    assert N.family_of(RIGHT if s.side == "right" else LEFT) == "rh_aglt"


def test_arm_law_is_the_training_increment_law(c):
    torch = pytest.importorskip("torch")
    ref = _load_hdgp("tasks/rh_aglt_r/arm_action.py")
    dec = A.RaDecoder(c)
    s = c.side()
    lo, hi = torch.tensor(s.arm_lo), torch.tensor(s.arm_hi)
    q_t = torch.tensor(s.arm_home).unsqueeze(0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = rng.uniform(-1.2, 1.2, 13)
        arm, _ = dec.step(a, active=True)["arm"]
        q_t = ref.step_arm_increment(q_t, torch.tensor(a[:7], dtype=torch.float32).unsqueeze(0), k=c.k_arm,
                                     alpha=c.arm_ema, q_lo=lo, q_hi=hi)
        np.testing.assert_allclose(arm, q_t[0].numpy(), atol=1e-5)


def test_hold_pins_the_arm_at_home_and_the_hand_open(c):
    dec = A.RaDecoder(c)
    arm, hand = dec.step(np.ones(13), active=False)["arm"]
    assert arm == pytest.approx(c.side().arm_home) and hand == pytest.approx(c.side().hand_open)


def test_keypoints_are_the_training_keypoints(c):
    torch = pytest.importorskip("torch")
    kg = _load_hdgp("modules/keypoint_goal.py")
    q = np.array([0.9, 0.1, -0.2, 0.3])
    q /= np.linalg.norm(q)
    pos = np.array([0.35, -0.16, 0.30])
    ref = kg.keypoints_world(torch.tensor(pos, dtype=torch.float32).unsqueeze(0), torch.tensor(q, dtype=torch.float32).unsqueeze(0),
                             kg.keypoint_offsets(c.cup_half_height, "cpu"))[0].numpy()
    np.testing.assert_allclose(A.keypoints(c, pos, q), ref, atol=1e-6)


def _meas(c, cup=(0.35, -0.16, 0.08), tact=(0, 0, 0, 0, 0)) -> A.RaMeas:
    s = c.side()
    return A.RaMeas(arm_q=np.asarray(s.arm_home), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, s.hand_open)),
                    palm_pos=np.array([0.2, -0.2, 0.3]), palm_R=np.eye(3), tips=np.tile([0.25, -0.2, 0.28], (5, 1)),
                    cup_pos=np.asarray(cup, float), cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.asarray(tact, float))


def test_observation_layout_is_96_in_the_training_order(c):
    dec = A.RaDecoder(c)
    m = _meas(c, tact=(0, 3.0, 0, 0, 20.0))
    goal = A.first_goal(c, m.cup_pos, m.cup_quat)
    prev = np.linspace(-1, 1, 13)
    obs = A.build_obs(c, m, dec, goal, prev)
    assert obs.size == 96 == sum(w for _, w in A.ACTOR_LAYOUT)
    off = {}
    i = 0
    for name, w in A.ACTOR_LAYOUT:
        off[name] = obs[i:i + w]
        i += w
    assert off["arm_q_target"] == pytest.approx(c.side().arm_home)
    assert off["cup_up"] == pytest.approx([0, 0, 1]) and off["goal_rel_palm"] == pytest.approx(goal.pos - m.palm_pos)
    assert off["kp_rel_goal"].reshape(4, 3) == pytest.approx(np.tile([0, 0, -0.14], (4, 1)))
    assert off["tactile"] == pytest.approx(np.tanh(np.array([0, 3.0, 0, 0, 10.0]) / 3.0))
    assert off["prev_action"] == pytest.approx(prev)
    hdgp_layout = _load_hdgp("tasks/rh_aglt_r/layout.py").actor_layout(7, 6, 5)
    assert tuple(A.ACTOR_LAYOUT) == tuple(hdgp_layout)


class _Const:
    def __init__(self, a):
        self.a, self.n = np.asarray(a, float), 0

    def forward(self, obs):
        assert obs.size == 96
        self.n += 1
        return self.a

    def reset(self):
        self.n = 0


def test_the_chain_holds_then_moves_and_needs_the_cup_to_reset(c):
    ch = N.AgltChain(c, _Const(np.r_[np.full(7, 0.5), np.ones(6)]))
    with pytest.raises(N.PourFjNodeError, match="cup"):
        ch.reset(None)
    meas = {"arm": _meas(c)}
    ch.reset(meas)
    assert ch.goal.pos == pytest.approx(meas["arm"].cup_pos + [0, 0, 0.14])
    for _ in range(c.hold_steps):
        _, _, t = ch.step(meas)
        assert t["arm"][0] == pytest.approx(c.side().arm_home)
    _, _, t = ch.step(meas)
    assert not np.allclose(t["arm"][0], c.side().arm_home)
    names, q, qd = N.aglt_target_arrays(c, t)
    assert names == tuple(c.side().arm_joints) + tuple(c.side().hand_joints) and q.size == 13 and not qd.any()


def test_start_is_refused_away_from_home_or_with_a_fallen_cup(c):
    m = _meas(c)
    assert N.aglt_start_refusals(c, {"arm": m}, 0.15) == []
    far = A.RaMeas(**{**m.__dict__, "arm_q": np.asarray(c.side().arm_home) + 0.3})
    assert "training start pose" in N.aglt_start_refusals(c, {"arm": far}, 0.15)[0]
    fallen = A.RaMeas(**{**m.__dict__, "cup_quat": np.array([np.cos(0.6), np.sin(0.6), 0, 0])})
    assert "tilt" in N.aglt_start_refusals(c, {"arm": fallen}, 0.15)[0]


def test_the_checkpoint_loads_and_acts(c):
    pytest.importorskip("torch")
    if not Path(c.checkpoint).is_file():
        pytest.skip("가중치 없음(.gitignore) — 5090 에서 복사")
    from policy_control.joint_policy import JointPolicy
    pol = JointPolicy(c, "cpu")
    ch = N.AgltChain(c, pol)
    meas = {"arm": _meas(c)}
    ch.reset(meas)
    for _ in range(12):
        obs, a, t = ch.step(meas)
    assert a.size == 13 and np.all(np.isfinite(a)) and np.all(np.isfinite(obs))


def test_a_pour_fj_contract_is_not_taken_for_rh_aglt():
    fj = POL / "both_rh_pourfj_f01" / "pour_fj_contract.json"
    assert N.family_of(fj) == "pour_fj"
    with pytest.raises(A.RhAgltError, match="schema"):
        A.load_contract(fj)
    bad = json.loads(RIGHT.read_text())
    bad["action_dim"] = 26
    with pytest.raises(A.RhAgltError):
        A.validate(A.RaContract(**{**bad, "sides": {"arm": A.RaSide(**bad["sides"]["arm"])}}))


MIRROR = POL / "right_rh_aglt_mirror_l5" / "rh_aglt_contract.json"
ARM_MIRROR_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
_M, _MN = (1.0, -1.0, 1.0), (-1.0, 1.0, -1.0)
#: hdgp scripts/tools/mirror_rh_aglt_ckpt.py actor_sign — 관측 96 칸별 좌우 부호(팔 부호 · 손 +1 · 위치 (1,−1,1) · 손바닥 x 열 (−1,1,−1))
OBS_MIRROR = (list(ARM_MIRROR_SIGN) * 3 + [1.0] * 12 + list(_M) + list(_MN) + list(_M) + list(_M) + list(_M)
              + list(_M) * 5 + list(_M) * 4 + list(_M) + [1.0] * 5 + list(ARM_MIRROR_SIGN) + [1.0] * 6)
ACT_MIRROR = list(ARM_MIRROR_SIGN) + [1.0] * 6


def test_the_mirrored_right_policy_is_the_left_policy_seen_in_a_mirror():
    """09.30 사용자 "오른팔 미러 되는지": 배포 로더로 두 체크포인트를 불러 a_R(o) = S_a · π_L(D_o · o) 를 LSTM 연속 60 스텝에서 확인."""
    pytest.importorskip("torch")
    cr, cl = A.load_contract(MIRROR), A.load_contract(LEFT)
    if not (Path(cr.checkpoint).is_file() and Path(cl.checkpoint).is_file()):
        pytest.skip("가중치 없음(.gitignore)")
    from policy_control.joint_policy import JointPolicy
    pr, pl = JointPolicy(cr, "cpu"), JointPolicy(cl, "cpu")
    assert len(OBS_MIRROR) == 96
    D, S = np.array(OBS_MIRROR), np.array(ACT_MIRROR)
    dec = A.RaDecoder(cr)
    m = _meas(cr)
    goal = A.first_goal(cr, m.cup_pos, m.cup_quat)
    rng = np.random.default_rng(3)
    prev = np.zeros(13)
    worst = 0.0
    for _ in range(60):
        o = A.build_obs(cr, m, dec, goal, prev) + rng.normal(0, 0.05, 96)     # 배포 관측 + 흔들기(LSTM 상태가 쌓인다)
        a_r, a_l = pr.forward(o), pl.forward(D * o)
        worst = max(worst, float(np.abs(a_r - S * a_l).max()))
        prev = np.clip(a_r, -1, 1)
        dec.step(a_r, active=True)
    assert worst < 1e-4, worst

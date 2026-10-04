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
    y = 0.16 if c.side().side == "left" else -0.16          # 그 팔의 소환 박스 안, 정착고(첫 목표가 목표 박스로 잘리지 않게)
    meas = {"arm": _meas(c, cup=(0.35, y, 0.264865))}
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


def test_the_cup_geometry_follows_the_object_like_the_training_env():
    """hdgp rh_aglt_env_cfg.resolve_cfg 와 같은 규칙 — shaker 만 cup_scale 을 받고 원기둥은 실물 크기."""
    assert A.cup_geometry({"object_name": "shaker", "cup_scale": 0.65}) == pytest.approx((0.056875, 0.059865))
    assert A.cup_geometry({"cup_scale": 0.65}) == pytest.approx((0.056875, 0.059865))        # 키 없는 옛 런 = shaker
    assert A.cup_geometry({"object_name": "cyl60", "cup_scale": 0.65}) == pytest.approx((0.085, 0.085))
    with pytest.raises(A.RhAgltError, match="object_name"):
        A.cup_geometry({"object_name": "mug", "cup_scale": 0.65})


@pytest.mark.parametrize("pid", ["right_rh_aglt_cyl60g", "left_rh_aglt_cyl60gmir"])
def test_the_cylinder_contracts_carry_the_cylinder_not_the_stale_shaker_dump(pid):
    """10.04: train.py 는 hydra 가 object_name=cyl60 을 덮은 뒤 · env 가 resolve_cfg 를 다시 부르기 전에 env.yaml 을 덤프한다.
    덤프의 파생 값(반높이 0.0569 · 원점 높이 0.0599)은 기본 shaker × 0.65 값이고 학습 env 는 cyl60(0.085 · 0.085)으로 돌았다 —
    계약이 덤프를 믿으면 키포인트가 축으로 2.8 cm 짧고 목표 박스가 2.5 cm 낮다."""
    run = POL / pid
    env = yaml.unsafe_load((run / "params" / "env.yaml").read_text())
    assert env["object_name"] == "cyl60" and env["cup_half_height"] == pytest.approx(0.056875)     # 덤프는 낡았다
    c = A.load_contract(run / "rh_aglt_contract.json")
    assert c.cup_half_height == pytest.approx(0.085)
    z0, zb = float(env["table_surface_z"]) + 0.085, env["goal_box_z_range"]
    assert (c.goal_box_min[2], c.goal_box_max[2]) == pytest.approx((z0 + zb[0], z0 + zb[1]))


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


class _St:
    """sources.RobotState 의 필요한 칸만."""
    def __init__(self, s, tip=(0, 0, 0, 0, 0)):
        self.stale, self.missing = (), ()
        self.arm_q, self.arm_qd = np.asarray(s.arm_home, float), np.zeros(7)
        self.ee_names, self.ee_q = tuple(s.hand_joints), np.asarray(s.hand_open, float)
        self.tip_force = np.asarray(tip, float)
        self.stamps = {"arm": 0.0}
        self.joint_force = None


class _Fk:
    def __init__(self, palm):
        self.palm = np.asarray(palm, float)

    def palm_pose(self, arm_q, hand_q):
        from types import SimpleNamespace
        return SimpleNamespace(palm_pos=self.palm, palm_quat=np.array([1.0, 0, 0, 0]), tips=np.zeros(15))


def test_the_cup_can_be_chosen_after_the_palm_and_touch_are_known(c):
    """10.04 쥔 뒤 FK: 측정이 손바닥 FK · 촉각을 먼저 구하고 컵 고르는 함수에 넘긴다(cup_attach 가 쓴다)."""
    s = c.side()
    seen = {}

    def pick(palm_pos, palm_R, tact, st):
        seen.update(palm=palm_pos, R=palm_R, tact=tact, stamp=st.stamps["arm"])
        return palm_pos + np.array([0.0, 0.0, -0.04]), np.array([1.0, 0, 0, 0])

    raw = N._side_raw(s, _St(s, tip=(1.5, 0, 2.0, 0, 0)), list(s.arm_joints), _Fk([0.3, -0.1, 0.3]), pick, "arm")
    assert seen["palm"] == pytest.approx([0.3, -0.1, 0.3]) and seen["R"] == pytest.approx(np.eye(3))
    assert seen["tact"] == pytest.approx([1.5, 0, 2.0, 0, 0])
    assert raw["cup_pos"] == pytest.approx([0.3, -0.1, 0.26])
    with pytest.raises(N.PourFjNodeError, match="컵"):
        N._side_raw(s, _St(s), list(s.arm_joints), _Fk([0.3, -0.1, 0.3]), lambda *_: None, "arm")


def test_a_grasped_cup_follows_the_palm_through_the_measurement_path(c):
    """쥔 채 0.29 s 늦은 FP++ 프레임이 오면 그 프레임 시각의 손바닥으로 붙고, 이후에는 튀는 FP++ 대신 손바닥을 따른다."""
    from policy_control.cup_attach import AttachCfg, CupAttach, grasp_flag
    s = c.side()
    est = CupAttach(AttachCfg())
    cup0 = np.array([0.30, -0.10, 0.26])

    def step(t, palm, tip, cup_live):
        st = _St(s, tip=tip)
        st.stamps = {"arm": t}
        f = lambda pp, pR, tt, stt: est.step(grasp_flag(tt, 1.0), stt.stamps["arm"], pp, pR, cup_live)  # noqa: E731
        return N._side_raw(s, st, list(s.arm_joints), _Fk(palm), f, "arm")

    for k in range(40):                                   # 쥔 채로 손은 그대로, 마지막 프레임은 0.29 s 전 것
        t = k / 60
        raw = step(t, [0.30, -0.10, 0.30], (1.5, 1.2, 0, 0, 0), (cup0, np.array([1.0, 0, 0, 0]), t - 0.29))
    assert est.source == "attached" and raw["cup_pos"] == pytest.approx(cup0)
    raw = step(41 / 60, [0.30, -0.10, 0.40], (1.5, 1.2, 0, 0, 0), (np.array([0.9, 0.9, 0.9]), np.array([1.0, 0, 0, 0]), 0.3))
    assert raw["cup_pos"] == pytest.approx([0.30, -0.10, 0.36])       # 손과 함께 10 cm 위, 튄 FP++ 무시


def test_the_node_feeds_joint_forces_into_the_grasp_signal(c):
    """노드의 _cup_for: 관절 힘(엄지 첫마디 파지)만으로도 판정되고, stale 이면 쓰지 않는다."""
    from types import SimpleNamespace
    from policy_control.cup_attach import AttachCfg, CupAttach
    s = c.side()
    est = CupAttach(AttachCfg(attach_after_s=0.0))
    live = (np.array([0.30, -0.10, 0.26]), np.array([1.0, 0, 0, 0]), 0.03)   # 파지 시작(0.02) 뒤에 찍힌 프레임
    fake = SimpleNamespace(attach={"arm": est}, attach_cfg=AttachCfg(attach_after_s=0.0), _cup=lambda r: live,
                           get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)),
                           get_logger=lambda: SimpleNamespace(info=lambda *_a, **_k: None))
    st = _St(s, tip=(0.1, 1.5, 0, 0, 0))                     # 엄지 손끝은 0.1 N
    st.stamps = {"arm": 0.01}
    st.joint_force, st.stale = np.array([[450.0], [0.0], [400.0], [0.0], [0.0], [0.0]]), ("joint_force",)
    pick = N.PourFjNode._cup_for(fake, "arm", True)
    pick(np.array([0.3, -0.1, 0.3]), np.eye(3), np.asarray(st.tip_force), st)
    assert est.source == "live"                               # stale 한 관절 힘은 안 쓴다
    st.stale = ()
    st.stamps = {"arm": 0.02}
    pick(np.array([0.3, -0.1, 0.3]), np.eye(3), np.asarray(st.tip_force), st)
    assert est.source == "attached"


def test_a_place_run_is_not_taken_for_rh_aglt_even_with_the_same_dimensions():
    """rh_place 는 관측 96 · 행동 13 · rh56f1 프로필로 rh_aglt 와 같지만 목표(홀더 자리) · 시작(인계 뱅크)이 다르다 —
    rh_aglt 계약을 만들면 콘솔이 aglt 자리에 그대로 내보인다(10.04)."""
    env = A.read_env(POL / "right_rh_aglt_cyl60g" / "params" / "env.yaml")
    assert A.is_rh_aglt_run(env)
    assert not A.is_rh_aglt_run({**env, "target_holders": (1, 2)})
    place = POL / "right_rh_place_i09" / "params" / "env.yaml"
    if place.is_file():
        assert not A.is_rh_aglt_run(A.read_env(place))

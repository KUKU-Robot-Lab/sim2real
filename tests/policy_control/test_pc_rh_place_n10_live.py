"""★11.02 oneshot 세팅 실기 연결(사용자 지시, sim_e2e 경유) — newton_10_r · newton_09_l 놓기 런을 배포 계약으로.

학습이 n07 과 다른 세 가지를 빌더 · 사슬이 받는다:
  1. cup_obs_mode 'true'(actor 컵 = sim 참값) → 계약 cup_obs "live": 실기는 FP++ 생방송, 끊기면 손바닥 FK ⊗ 붙인 자세(10.21 규칙).
  2. 물체 1종 · one-hot 없음(object_id_obs False) → object_geometry = {물체: 기하}, select_object 는 그 이름만 받는다(10.30 place_lab).
  3. 뱅크가 홀더를 덮는 런(occ_from_bank) → holder_range · target_holders 는 뱅크 holder_pos ± bank_layout_shift.
덤프 observation_space 97 은 cfg 기본(96 + 1)이고 체크포인트는 92(91 + 래치) — 빌더가 base 로 읽는다.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from types import SimpleNamespace
import numpy as np
import pytest

from policy_control import rh_aglt as A
from policy_control import rh_place as P

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy/policies/rh56f1/place"
IDS = ("right_n10_e1000", "right_n10_e4000", "left_n09_e3800")


def _contract(pid: str) -> P.PlaceContract:
    f = POL / pid / "rh_place_contract.json"
    if not f.is_file():
        pytest.skip(f"{pid} 계약 없음")
    return P.load_contract(f)


def _env(**kw) -> dict:
    return {"object_id_obs": False, "object_names": (), "object_name": "source240_real", "cup_scale": 0.65, "observation_space": 97, **kw}


# ---------------------------------------------------------------- 빌더 · 계약
def test_true_runs_are_place_runs_and_fpp_runs_are_not():
    base = {"target_holders": (1, 2), "hold_steps": 0, "profile_name": "rh56f1_v2_right"}
    assert P.is_rh_place_run({**base, "cup_obs_mode": "attached"})
    assert P.is_rh_place_run({**base, "cup_obs_mode": "true"})
    assert not P.is_rh_place_run({**base, "cup_obs_mode": "fpp"})
    assert P.CUP_OBS_MODES == {"attached": "attached", "true": "live"}


def test_single_object_without_onehot_is_named_and_sized():
    assert A.single_object_of(_env()) == "source240_real"
    assert A.single_object_of(_env(object_names=("shaker_c_real",), object_name="shaker_c_real")) == "shaker_c_real"
    assert A.single_object_of(_env(object_id_obs=True, object_names=("source240_real", "shaker_c_real"))) is None
    assert A.single_object_of(_env(object_name="cyl60_nope")) is None          # 기하 표에 없는 물체는 모른다


def test_the_stale_default_dump_is_read_as_the_layout_width():
    # 촉각 없음 91, 래치 1: 덤프 97(= 96 + 1 cfg 기본) → 91, 체크포인트 92 = 91 + 1
    assert A._base_obs_dim(_env(observation_space=97), 91, [], 1) == 91
    assert A._base_obs_dim(_env(observation_space=92), 91, [], 1) == 91
    assert A._base_obs_dim(_env(observation_space=96), 96, [], 0) == 96         # i09(촉각 있음) 그대로
    assert A._base_obs_dim(_env(observation_space=101), 91, ["a", "b", "c"], 1) == 94   # one-hot 런은 base + K
    assert A._base_obs_dim(_env(observation_space=123), 91, [], 1) == 123       # 모르는 폭은 그대로 두어 체크포인트 검사가 잡는다


def test_bank_holder_range_uses_the_bank_targets(tmp_path):
    hp = np.zeros((4, 3, 3))
    hp[:, 1, :2] = [[0.37, -0.05], [0.39, -0.04], [0.38, -0.06], [0.385, -0.045]]
    hp[:, 2, :2] = [[0.37, -0.25], [0.39, -0.24], [0.38, -0.26], [0.385, -0.245]]
    b = tmp_path / "bank.npz"
    np.savez(b, holder_pos=hp, target=np.array([1, 2, 1, 2]))
    rng, t = P.bank_holder_range(b, [0], 0.02)
    assert t == [1, 2]
    assert rng == {"x": [0.35, 0.41], "y": [-0.28, -0.02]}
    np.savez(b, holder_pos=hp)                                  # target 이 없으면 준 목표
    assert P.bank_holder_range(b, [2], 0.0)[0] == {"x": [0.37, 0.39], "y": [-0.26, -0.24]}


def test_find_bank_prefers_the_policy_folder_copy(tmp_path):
    (tmp_path / "bank").mkdir()
    (tmp_path / "bank" / "bank_right_train.npz").write_bytes(b"x")
    env = {"start_bank_path": "/home/usr/nowhere/bank_right_train.npz"}       # 학습 호스트 경로 — 여기 없다
    assert P.find_bank(env, tmp_path) == tmp_path / "bank" / "bank_right_train.npz"
    with pytest.raises(P.RhPlaceError, match="뱅크"):
        P.find_bank({"start_bank_path": "/home/usr/nowhere/other.npz"}, tmp_path)


@pytest.mark.parametrize("pid", IDS)
def test_the_built_contracts_are_92_live_one_object_from_the_bank(pid):
    c = _contract(pid)
    obj = "shaker_c_real" if pid.startswith("left") else "source240_real"
    assert c.obs_dim == 92 and A.obs_dim_of(c) == 92 and c.action_dim == 13
    assert c.cup_obs == "live" and c.latch_s2r and c.attach_source == "bundle" and c.ready_carry
    assert list(c.extra_actor_obs) == ["latch"] and c.neighbor_obs == "" and not c.latch_est
    assert c.latch_obs == {"dz": 0.04, "dxy": 0.012, "tilt_deg": 8.0, "bottom_xy": True}
    assert c.object_names == [] and list(c.object_geometry) == [obj] and c.active_object == ""
    assert c.cup_half_height == pytest.approx(0.065 if obj == "shaker_c_real" else 0.085)
    assert c.target_holders == ([0] if pid.startswith("left") else [1, 2])
    assert c.holder_range and c.holder_range["x"][0] < 0.38 < c.holder_range["x"][1]
    assert any(n.startswith("★홀더 범위 = 뱅크") for n in c.notes) and any(n.startswith("★actor 컵 = FP++") for n in c.notes)


def test_select_object_accepts_only_the_trained_object():
    c = _contract("right_n10_e1000")
    assert A.select_object(c, "source240_real").active_object == "source240_real"
    assert A.select_object(c, None).active_object == "source240_real"
    with pytest.raises(A.RhAgltError, match="source240_real"):
        A.select_object(c, "shaker_c_real")


def test_validate_refuses_bad_cup_obs_and_a_second_geometry():
    c = _contract("right_n10_e1000")
    with pytest.raises(P.RhPlaceError, match="cup_obs"):
        P.validate(replace(c, cup_obs="true"))
    with pytest.raises(P.RhPlaceError, match="latch_s2r"):
        P.validate(replace(c, cup_obs="live", latch_s2r=False, release_timed_steps=0, ready_carry={}))
    with pytest.raises(A.RhAgltError, match="물체 하나만"):
        A.validate(replace(c, object_geometry={**c.object_geometry, "shaker_c_real": c.object_geometry["source240_real"]}))
    with pytest.raises(A.RhAgltError, match="active_object"):
        A.validate(replace(c, active_object="shaker_c_real"))


# ---------------------------------------------------------------- 사슬: live 컵
class _Chain:
    """PlaceChain.cup_obs 만 — 붙인 자세 rel · 계약 cup_obs · live_cup 셈."""
    def __init__(self, cup_obs: str):
        self.c = replace(_contract("right_n10_e1000"), cup_obs=cup_obs)
        self.rel = (np.array([0.0, 0.0, 0.10]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.cup_src = {"live": 0, "fk": 0, "rejected": 0}
        self.live_cup = None

    cup_obs = P.PlaceChain.cup_obs
    cup_from_palm = P.PlaceChain.cup_from_palm


def test_live_contract_uses_the_fresh_fpp_cup_and_falls_back_to_the_palm():
    ch = _Chain("live")
    palm, R = np.array([0.3, -0.1, 0.4]), np.eye(3)
    ch.live_cup = (np.array([0.31, -0.12, 0.49]), np.array([0.0, 0.0, 0.0, 1.0]))
    pos, quat = ch.cup_obs(palm, R)
    assert np.allclose(pos, [0.31, -0.12, 0.49]) and np.allclose(quat, [0, 0, 0, 1])
    ch.live_cup = None                                          # 끊김 · 가림 → FK ⊗ 붙인 자세
    pos, quat = ch.cup_obs(palm, R)
    assert np.allclose(pos, [0.3, -0.1, 0.5]) and np.allclose(quat, [1, 0, 0, 0])
    assert ch.cup_src == {"live": 1, "fk": 1, "rejected": 0}


def test_a_live_cup_far_from_the_palm_is_dropped_for_the_fk_cup():
    """★11.04 실기 131817: tilt 뒤 FP++ 가 쥔 컵을 놓치고 탁자 아래 자세를 냈다 — 손바닥 FK 에서 5 cm 넘게 멀면 FK."""
    ch = _Chain("live")
    palm, R = np.array([0.38, -0.06, 0.40]), np.eye(3)
    ch.live_cup = (np.array([0.739, -0.118, -0.141]), np.array([0.0, 0.0, 0.0, 1.0]))
    pos, quat = ch.cup_obs(palm, R)
    assert np.allclose(pos, [0.38, -0.06, 0.50]) and np.allclose(quat, [1, 0, 0, 0])
    ch.live_cup = (np.array([0.38 + 0.9 * P.LIVE_FK_TOL_M, -0.06, 0.50]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(ch.cup_obs(palm, R)[0], [0.38 + 0.9 * P.LIVE_FK_TOL_M, -0.06, 0.50])
    assert ch.cup_src == {"live": 1, "fk": 0, "rejected": 1}


def test_attached_contract_ignores_any_live_cup():
    ch = _Chain("attached")
    ch.live_cup = (np.array([9.0, 9.0, 9.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    pos, _ = ch.cup_obs(np.array([0.3, -0.1, 0.4]), np.eye(3))
    assert np.allclose(pos, [0.3, -0.1, 0.5]) and ch.cup_src == {"live": 0, "fk": 0, "rejected": 0}


def test_the_episodes_point_at_the_new_policies():
    import yaml
    for tag in ("e1000", "e4000"):
        ep = yaml.safe_load((REPO / "config/episodes" / f"cocktail_1c_{tag}.yaml").read_text(encoding="utf-8"))
        assert ep["policies"]["rh_place_r"]["policy"] == f"rh56f1/place/right_n10_{tag}"
        assert ep["policies"]["rh_place_l"]["policy"] == "rh56f1/place/left_n09_e3800"
        steps = {s["id"]: s for s in ep["sequence"]}
        assert steps["place_blue"]["target_holder"] == "HOLDER_1" and steps["place_pink"]["target_holder"] == "HOLDER_2"
        assert steps["place_shaker"]["target_holder"] == "HOLDER_0" and steps["place_shaker"]["policy_object"] == "shaker_c_real"
        assert ep["objects"]["SRC_PINK"]["region"] == {"x": [0.26, 0.305], "y": [-0.365, -0.330]}


# ---------------------------------------------------------------- 에피소드 계획 검사(episode_spec)
def test_plan_check_accepts_the_single_trained_object_and_refuses_another(tmp_path):
    """★11.02 fake 리허설에서 걸림: object_names 없는 물체 1종 계약은 단계 policy_object 가 그 이름(또는 빈 값)이면 통과."""
    from types import SimpleNamespace
    from policy_control import episode_spec as S
    ep = S.load(REPO / "config/episodes/cocktail_1c_e1000.yaml")
    pdir = tmp_path / "left_n09_e3800"
    pdir.mkdir()
    (pdir / "rh_place_contract.json").write_text('{"object_names": [], "object_geometry": {"shaker_c_real": {}}}', encoding="utf-8")
    entry = SimpleNamespace(path=pdir, contract="rh_place_contract.json")
    assert S._object_problem(ep, "rh_place_l", entry) == ""
    (pdir / "rh_place_contract.json").write_text('{"object_names": [], "object_geometry": {"cyl60": {}}}', encoding="utf-8")
    assert "shaker_c_real" in S._object_problem(ep, "rh_place_l", entry)
    (pdir / "rh_place_contract.json").write_text('{"object_names": [], "object_geometry": {}}', encoding="utf-8")
    assert "단일 물체 정책" in S._object_problem(ep, "rh_place_l", entry)


def test_single_object_needs_a_known_radius():
    """옛 런(right_i09 cyl60): 반지름을 모르는 물체는 1종 기하로 싣지 않는다(빌더가 전처럼 기하 없이 짓는다)."""
    assert A.single_object_of(_env(object_name="cyl60")) is None
    assert A.single_object_of(_env(object_name="source240_real")) == "source240_real"


def _node_self(live_param: bool, fresh):
    ch = _Chain("live")
    me = SimpleNamespace(chain=ch, _tin=False, contract=SimpleNamespace(cup_obs="live"), _fresh_cup=lambda role: fresh,
                         get_parameter=lambda n: SimpleNamespace(value=live_param))
    return me, ch


def test_place_node_feeds_only_the_bound_cup_unless_live_is_asked_for():
    """★11.04 사용자 "place 는 묶은 값만": 노드 기본은 FP++ live 를 사슬에 넣지 않는다(손바닥 FK ⊗ 묶음). 파라미터로만 켠다."""
    from policy_control.rh_place_node import PlaceNode
    fresh = (np.array([0.31, -0.12, 0.49]), np.array([0.0, 0.0, 0.0, 1.0]), 0.0)
    palm, R = np.array([0.3, -0.1, 0.4]), np.eye(3)
    me, ch = _node_self(False, fresh)
    pos, _ = PlaceNode._cup_for(me, "src", True)(palm, R, None, None)
    assert np.allclose(pos, [0.3, -0.1, 0.5]) and ch.cup_src == {"live": 0, "fk": 1, "rejected": 0}
    me, ch = _node_self(True, fresh)
    pos, _ = PlaceNode._cup_for(me, "src", True)(palm, R, None, None)
    assert np.allclose(pos, [0.31, -0.12, 0.49]) and ch.cup_src == {"live": 1, "fk": 0, "rejected": 0}


def test_place_node_ends_when_policy_and_measured_fingers_are_open_for_three_ticks():
    """★11.11 사용자 "손가락 다 피면 에피소드 끝": 정책 손 목표 · 실측 손가락이 둘 다 ≥ open_min_rad 로 OPENED_END_TICKS 연속이면 stop.
    sim 1111b: 실측 0.12 rad(컵을 쥔 채)에서 끝내면 안 된다."""
    import time
    from policy_control import rh_place_node as N
    ends, whys, log = [], [], SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)
    ch = SimpleNamespace(settle_t=5, latch=False, latch_t=0, settling=False, done=False, open_min_rad=0.15, opened=0.0,
                         opened_meas=0.0, latch_cup="blue", outcome=None, released_cup=None, as_dict=lambda: {})
    me = SimpleNamespace(contract=SimpleNamespace(latch_s2r=True), _carry_t0=None, chain=ch, max_s=99.0,
                         _t_start=time.monotonic(), _opened_n=0, get_logger=lambda: log,
                         _end=lambda kind, why: (ends.append(kind), whys.append(why)))
    seq = [(0.2, 0.12), (0.2, 0.12), (0.2, 0.12), (0.2, 0.12),   # 실측 0.12 rad 가 이어져도 끝내지 않는다(sim 1111b)
           (0.2, 0.15), (0.1, 0.5), (0.2, 0.15), (0.2, 0.15), (0.2, 0.15)]   # (정책, 실측) rad
    out = []
    for o, om in seq:
        ch.opened, ch.opened_meas = o, om
        out.append(N.PlaceNode._after_step(me))
    assert out == [False] * 8 + [True]   # 실측 미달 · 정책 미달 틱에서 셈이 0 으로, 0.15 rad 3 틱이면 stop
    assert ends == ["stop"] and ch.outcome == "opened_before_latch" and ch.released_cup == "blue"
    assert not whys[0].startswith("placed") and "래치 전" in whys[0]   # M4 — 래치 전 손 폄을 놓음으로 단정하지 않는다

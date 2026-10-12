"""★10.12 rh_aglt 다물체 한 정책(rg3) — 관측 96 + 물체 one-hot 3(맨 끝) · 물체별 컵 기하 · 물체를 모르면 거부 · 에피소드 단계 물체.

사용자 10.12: grasping 최종(right_rg3mft · left_rg3)을 sim2real 에 등록. 실기에서는 물체가 단계마다 고정이라 에피소드 단계 설정
(policy_object)으로 물체를 넣는다. hdgp 참조: tasks/rh_aglt_r/{rh_aglt_env.py `_actor_parts`, obj_geometry.py, goals.py}."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_control import episode_spec as S
from policy_control import pour_fj_node as N
from policy_control import rh_aglt as A
from policy_control import rh_aglt_goals as G

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy" / "policies"
RG3 = {"right": "rh56f1/aglt/right_rg3mft", "left": "rh56f1/aglt/left_rg3"}
MD5 = {"right": "89a0180df28dc2d63fdc3ee32ae137d6", "left": "da6106478d1cbcf42b5af0154c6e4419"}
NAMES = ["source240_real", "source200_real", "shaker_c_real"]
#: hdgp 다물체 코드가 있는 곳(main 병합 전에는 사다리 worktree) — 없으면 그 비교만 건너뛴다
HDGP_RH = [Path.home() / "rl_ws/hdgp/source/openarm/openarm/agnostic/tasks/rh_aglt_r",
           Path("/tmp/wt_ladder/source/openarm/openarm/agnostic/tasks/rh_aglt_r")]


def _hdgp_file(name: str) -> Path:
    for d in HDGP_RH:
        geo = d / "obj_geometry.py"
        if (d / name).is_file() and geo.is_file() and "def object_geometry" in geo.read_text():
            return d / name
    pytest.skip(f"hdgp 다물체 코드 없음: {name}")


def _contract(side: str) -> A.RaContract:
    return A.load_contract(POL / RG3[side] / "rh_aglt_contract.json")


@pytest.fixture(scope="module", params=sorted(RG3))
def c(request):
    return _contract(request.param)


def _meas(c, cup=None) -> A.RaMeas:
    s = c.side()
    y = 0.16 if s.side == "left" else -0.16
    return A.RaMeas(arm_q=np.asarray(s.arm_home), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, s.hand_open)),
                    palm_pos=np.array([0.2, y, 0.3]), palm_R=np.eye(3), tips=np.tile([0.25, y, 0.28], (5, 1)),
                    cup_pos=np.asarray(cup if cup is not None else (0.35, y, 0.29), float), cup_quat=np.array([1.0, 0, 0, 0]),
                    tactile_n=np.zeros(5))


# ---------------------------------------------------------------- 계약
def test_the_rg3_contracts_are_99_with_the_three_objects_in_one_hot_order(c):
    env = yaml.unsafe_load((POL / RG3[c.side().side] / "params" / "env.yaml").read_text())
    assert env["observation_space"] == 96                          # 덤프는 해석 전 값 — 실제 actor 는 99
    assert (c.obs_dim, c.action_dim) == (99, 13) and A.obs_dim_of(c) == 99
    assert c.object_names == NAMES == list(dict.fromkeys(env["object_names"]))
    assert c.checkpoint_md5 == MD5[c.side().side] and c.hand_command == "admittance"
    assert c.goal_offset == pytest.approx([0, 0, 0.10]) and c.goal_tol == pytest.approx(0.03)     # 컵 위 10 cm · 3 cm
    assert c.goal_success_steps == 120 and c.goal_force_consecutive                               # 2 s 연속(60 Hz)
    assert A.actor_layout(c)[-1] == ("object_id", 3) and A.actor_layout(c)[:-1] == A.ACTOR_LAYOUT
    assert c.active_object == ""                                   # 파일 계약에는 물체가 정해져 있지 않다


def test_the_object_geometry_table_is_the_hdgp_formula():
    for side in RG3:
        geo = _contract(side).object_geometry
        assert geo["source240_real"] == geo["source200_real"] == pytest.approx(
            {"origin_offset_z": 0.085, "mouth_z": 0.085, "bottom_z": -0.085, "radius": 0.030, "half_height": 0.085})
        assert geo["shaker_c_real"] == pytest.approx(
            {"origin_offset_z": 0.065, "mouth_z": 0.065, "bottom_z": -0.065, "radius": 0.0354, "half_height": 0.065})


def test_the_object_table_is_the_hdgp_object_table():
    """hdgp OBJECTS(*_real unit) · obj_geometry.object_geometry 공식과 같다 — Isaac 없이 읽을 수 있는 부분만."""
    path = _hdgp_file("obj_geometry.py")
    spec = importlib.util.spec_from_file_location("_hdgp_obj_geometry", path)
    og = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = og
    spec.loader.exec_module(og)
    assert tuple(og.GEOM_FIELDS) == A.GEOM_FIELDS
    cfg = (path.parent / "rh_aglt_env_cfg.py").read_text()
    for k in NAMES:
        m = re.search(r'"%s"\s*:\s*\{[^}]*?"unit"\s*:\s*(\{[^}]*\})' % k, cfg) or re.search(r'"%s".*?"unit"\s*:\s*(\{[^}]*\})' % k, cfg, re.S)
        assert m, k
        unit = yaml.safe_load(m.group(1).replace('"', "'"))
        assert unit == pytest.approx({**A.CUP_OBJECTS[k][0], "outer_r": A.CUP_OUTER_R[k]})
        ref = og.object_geometry({"unit": unit, "scaled": A.CUP_OBJECTS[k][1]}, 0.65)
        assert A.object_geometry(k, 0.65) == pytest.approx(ref)


def test_a_bad_object_table_is_refused(c):
    with pytest.raises(A.RhAgltError, match="obs/action"):
        A.with_run(c, obs_dim=96)
    with pytest.raises(A.RhAgltError, match="중복"):
        A.with_run(c, object_names=NAMES + NAMES[:1], obs_dim=100)
    with pytest.raises(A.RhAgltError, match="object_geometry"):
        A.with_run(c, object_geometry={k: v for k, v in c.object_geometry.items() if k != "shaker_c_real"})
    with pytest.raises(A.RhAgltError, match="active_object"):
        A.with_run(c, active_object="cyl60")


def test_the_contract_rebuilds_from_the_run_with_99(c, tmp_path):
    """계약 빌더가 env.yaml(observation_space 96 덤프)에서 99 · 물체 표를 다시 만든다 — 커밋한 계약과 경로 말고 같다."""
    import json
    import subprocess
    run = POL / RG3[c.side().side]
    if not (run / "nn" / Path(c.checkpoint).name).is_file():
        pytest.skip("가중치 없음(.gitignore)")
    tool = REPO / "deploy/policy_control/tools/build_rh_aglt_contract.py"
    hdgp = Path.home() / "rl_ws/hdgp"
    if not hdgp.is_dir():
        pytest.skip("hdgp 없음(프로필 · URDF)")
    out = tmp_path / "c.json"
    r = subprocess.run([sys.executable, str(tool), "--run", str(run), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    new, old = json.loads(out.read_text()), json.loads((run / "rh_aglt_contract.json").read_text())
    for d in (new, old):
        d.pop("run_dir"), d.pop("checkpoint")
    assert new == old


# ---------------------------------------------------------------- 물체 고르기 · 관측
@pytest.mark.parametrize("name", NAMES)
def test_the_one_hot_is_the_last_three_and_follows_the_object(c, name):
    cc = A.select_object(c, name)
    m = _meas(cc)
    dec = A.RaDecoder(cc)
    goal = A.first_goal(cc, m.cup_pos, m.cup_quat)
    obs = A.build_obs(cc, m, dec, goal, np.zeros(13))
    assert obs.size == 99
    want = np.zeros(3)
    want[NAMES.index(name)] = 1.0
    assert obs[-3:] == pytest.approx(want)
    base = A.build_obs(A.with_run(cc, object_names=[], object_geometry={}, active_object="", obs_dim=96), m, dec, goal,
                       np.zeros(13))
    assert obs[:96] == pytest.approx(base)                          # 앞 96 은 단일 물체 관측과 같은 순서 · 값


def test_the_cup_geometry_and_goal_box_follow_the_object(c):
    src, shk = A.select_object(c, "source240_real"), A.select_object(c, "shaker_c_real")
    assert (src.cup_half_height, shk.cup_half_height) == pytest.approx((0.085, 0.065))
    assert src.goal_box_min == pytest.approx(c.goal_box_min) and src.goal_box_max == pytest.approx(c.goal_box_max)
    # hdgp GoalState box_z_shift = 물체 원점 − 첫 물체 원점(0.065 − 0.085): 목표 박스 높이만 내려간다
    assert shk.goal_box_min == pytest.approx([*c.goal_box_min[:2], c.goal_box_min[2] - 0.02])
    assert shk.goal_box_max == pytest.approx([*c.goal_box_max[:2], c.goal_box_max[2] - 0.02])
    assert shk.goal_offset == pytest.approx(c.goal_offset)        # 첫 목표 = 컵 원점 + 10 cm(물체 무관)
    kp_s = A.keypoints(src, np.zeros(3), np.array([1.0, 0, 0, 0]))
    kp_k = A.keypoints(shk, np.zeros(3), np.array([1.0, 0, 0, 0]))
    assert np.abs(kp_s).max() == pytest.approx(0.085) and np.abs(kp_k).max() == pytest.approx(0.065)


@pytest.mark.parametrize("name", [None, "", "cyl60", "shaker_c"])
def test_an_unknown_or_missing_object_is_refused_not_zero_filled(c, name):
    with pytest.raises(A.RhAgltError, match="모른다"):
        A.select_object(c, name)
    m = _meas(c)
    with pytest.raises(A.RhAgltError, match="one-hot"):            # 파일 계약 그대로(물체 안 고름) 관측을 만들면 거부
        A.build_obs(c, m, A.RaDecoder(c), A.first_goal(c, m.cup_pos, m.cup_quat), np.zeros(13))


def test_single_object_contracts_are_unchanged_and_refuse_an_object():
    old = A.load_contract(POL / "rh56f1/aglt/right_s240" / "rh_aglt_contract.json")
    assert old.object_names == [] and A.obs_dim_of(old) == 96 and A.actor_layout(old) == A.ACTOR_LAYOUT
    assert A.select_object(old, None) is old
    with pytest.raises(A.RhAgltError, match="단일 물체"):
        A.select_object(old, "source240_real")
    m = _meas(old)
    obs = A.build_obs(old, m, A.RaDecoder(old), A.first_goal(old, m.cup_pos, m.cup_quat), np.zeros(13))
    assert obs.size == 96


def test_success_counts_consecutive_steps_when_the_run_did(c):
    cc = A.select_object(c, "source240_real")
    cup, q = np.array([0.3, 0.0, 0.3]), np.array([1.0, 0, 0, 0])
    book = G.GoalBook.start(cc, cup, q)
    goal = book.current.copy()                                     # 목표 = 컵이 갈 자리(컵 키포인트 vs 목표 키포인트)
    for _ in range(cc.goal_success_steps - 1):
        assert not book.step(goal, q, np.full(5, 50.0), is_grasped=True)
    book.step(cup, q, np.full(5, 50.0), is_grasped=True)       # 한 번 벗어나면 0 부터(10.11 D6)
    assert book.near_steps == 0
    for _ in range(cc.goal_success_steps - 1):
        assert not book.step(goal, q, np.full(5, 50.0), is_grasped=True)
    assert book.step(goal, q, np.full(5, 50.0), is_grasped=True)


# ---------------------------------------------------------------- 체인(노드 안) · 가중치
class _Const:
    def __init__(self):
        self.seen = []

    def forward(self, obs):
        self.seen.append(np.asarray(obs).copy())
        return np.zeros(13)

    def reset(self):
        pass


def test_the_chain_picks_the_object_at_reset_and_refuses_without_one(c):
    pol = _Const()
    ch = N.AgltChain(c, pol)
    meas = {"arm": _meas(c)}
    with pytest.raises(A.RhAgltError, match="모른다"):
        ch.reset(meas)
    ch.object_name = "shaker_c_real"
    ch.reset(meas)
    assert ch.c.active_object == "shaker_c_real" and ch.c.cup_half_height == pytest.approx(0.065)
    for _ in range(c.hold_steps + 2):
        ch.step(meas)
    assert pol.seen and all(o.size == 99 and o[-3:].tolist() == [0, 0, 1] for o in pol.seen)
    ch.object_name = "source200_real"                               # 다음 에피소드 다른 물체 — reset 이 다시 고른다
    ch.reset(meas)
    pol.seen.clear()
    for _ in range(c.hold_steps + 2):
        ch.step(meas)
    assert all(o[-3:].tolist() == [0, 1, 0] for o in pol.seen) and ch.c.cup_half_height == pytest.approx(0.085)


def _local(c) -> A.RaContract:
    """계약은 공유 트리 경로를 싣는다 — 이 저장소(worktree 포함) 안 사본으로 돌린다."""
    run = POL / RG3[c.side().side]
    ckpt = run / "nn" / Path(c.checkpoint).name
    if not ckpt.is_file():
        pytest.skip("가중치 없음(.gitignore) — our_source/policy 에서 복사")
    return A.with_run(c, run_dir=str(run), checkpoint=str(ckpt))


def test_the_checkpoint_input_is_99_and_its_md5_is_the_handed_over_md5(c):
    torch = pytest.importorskip("torch")
    import hashlib
    cl = _local(c)
    assert hashlib.md5(Path(cl.checkpoint).read_bytes()).hexdigest() == MD5[c.side().side]
    sd = torch.load(cl.checkpoint, map_location="cpu", weights_only=False)["model"]
    first = [v for k, v in sd.items() if k.endswith("rnn.weight_ih_l0")]
    assert first and first[0].shape[1] == 99                         # actor 첫 층(LSTM) 입력 폭
    assert sd["running_mean_std.running_mean"].shape == (99,)


@pytest.mark.parametrize("name", NAMES)
def test_the_policy_acts_on_cpu_for_every_object(c, name):
    pytest.importorskip("torch")
    from policy_control.joint_policy import JointPolicy
    cl = _local(c)
    ch = N.AgltChain(cl, JointPolicy(cl, "cpu"))
    ch.object_name = name
    meas = {"arm": _meas(cl)}
    ch.reset(meas)
    for _ in range(cl.hold_steps + 5):
        obs, a, _t = ch.step(meas)
    assert obs.size == 99 and a.size == 13 and np.all(np.isfinite(a))


# ---------------------------------------------------------------- 에피소드 단계 물체
def _raw(obj, policy="rh56f1/aglt/right_rg3mft", **job):
    raw = yaml.safe_load((REPO / "config/episodes/pick_place_right.yaml").read_text())
    raw["policies"]["rh_aglt_r"]["policy"] = policy
    for n in raw["sequence"]:
        if n["id"] == "pick_cup" and obj is not None:
            n["policy_object"] = obj
        if n["id"] == "place_cup":
            n.update(job)
    return raw


def test_an_episode_step_carries_the_policy_object_to_the_plan():
    from policy_control.episode_fake import FakeExecutor
    from policy_control.episode_runner import SUCCESS, EpisodeManager
    ep = S.parse(_raw("source240_real"))
    seen = {}

    class Ex(FakeExecutor):
        def run_policies(self, node, plans, world):
            seen[node.id] = plans
            return super().run_policies(node, plans, world)

    assert EpisodeManager(ep, Ex(ep), approve=lambda *_: True).run() == SUCCESS
    assert seen["pick_cup"][0].policy_object == "source240_real" and seen["place_cup"][0].policy_object is None


def test_a_policy_object_on_a_non_grasp_step_is_refused():
    with pytest.raises(S.EpisodeSpecError, match="policy_object"):
        S.parse(_raw(None, policy_object="source240_real"))


def test_availability_needs_a_known_object_for_the_multi_object_policy():
    from policy_control import policy_registry as R
    entries = {e.id: e for e in R.scan(POL, deep=False)}
    assert "rh56f1/aglt/right_rg3mft" in entries
    ok = S.availability(S.parse(_raw("source240_real")), entries)
    missing = S.availability(S.parse(_raw(None)), entries)
    wrong = S.availability(S.parse(_raw("cyl60")), entries)
    assert "policy_object" not in ok["rh_aglt_r"] and "다물체" not in ok["rh_aglt_r"]
    assert "다물체" in missing["rh_aglt_r"] and "cyl60" in wrong["rh_aglt_r"]
    single = S.availability(S.parse(_raw("source240_real", policy="rh56f1/aglt/right_s240")), entries)
    assert "단일 물체" in single["rh_aglt_r"]

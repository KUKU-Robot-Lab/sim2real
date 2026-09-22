"""plan_home_path.py — 차렷 -> 홈 오프라인 경로 계획기 (ROS·GPU 없음).

시간 매개화가 pd ramp_speed(0.1 rad/s)를 지키는지, npz 가 replay_to_pd.load_frames 로 읽히는지,
충돌 검사기가 뻔한 충돌(손을 테이블 상판에 박은 자세)을 잡고 저장된 경로는 통과시키는지, 관절한계를 본다.
mujoco·trimesh 가 없는 파이썬(시스템 python3 등)에서는 충돌 부분만 skip 한다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SIM2REAL = Path(__file__).resolve().parents[2]
TOOLS = SIM2REAL / "deploy/policy_control/tools"
NPZ = SIM2REAL / "logs/policy_control/home_path_right.npz"

pytestmark = pytest.mark.unit


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("plan_home_path")
R = _load("replay_to_pd")
HOME = np.array([0.2667, 0.4487, 0.4923, 0.7184, -0.046, 0.6496, 0.4762])


def test_ramp_speed_is_read_from_pd_config():
    assert P.read_ramp_speed(P.PD_CONFIG_DEFAULT) == pytest.approx(0.1)


def test_time_parametrization_respects_speed_and_endpoints():
    path = np.array([np.zeros(7), [0.1, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0], HOME])
    frames = P.time_parametrize(path, vmax=0.1, dt=0.02, ramp=1.0)
    speed = np.abs(np.diff(frames, axis=0)) / 0.02
    assert speed.max() <= 0.1 + 1e-9
    assert np.allclose(frames[0], path[0]) and np.allclose(frames[-1], path[-1])
    for wp in path:                                   # 정지-출발: 꼭짓점을 정확히 지난다
        assert np.abs(frames - wp).max(axis=1).min() < 1e-12
    # 시작·끝은 부드럽다: 첫·마지막 스텝 속도는 최고 속도의 5 % 미만
    assert speed[0].max() < 0.005 and speed[-1].max() < 0.005


def test_short_segment_lowers_peak_speed():
    s = P.segment_profile(0.01, vmax=0.1, ramp=1.0, dt=0.02)
    assert s[0] == 0.0 and s[-1] == pytest.approx(0.01)
    assert np.diff(s).max() / 0.02 <= 0.1


def test_saved_npz_is_readable_by_replay_to_pd():
    if not NPZ.exists():
        pytest.skip(f"{NPZ} 없음 — plan_home_path.py 를 먼저 돌린다")
    d = np.load(NPZ)
    joints = [str(j) for j in d["meta_joints"]]
    args = SimpleNamespace(npz=NPZ, npz_key=None, env=0, joints=joints, dt=0.5)
    frames, dt = R.load_frames(args)
    assert frames.shape[1] == 7 and dt == pytest.approx(0.02)
    assert np.allclose(frames[0], d["meta_start"]) and np.allclose(frames[-1], d["meta_goal"])
    assert np.abs(np.diff(frames, axis=0)).max() / dt <= 0.1 + 1e-9
    for key in ("meta_min_clearance", "meta_worst_pair", "meta_method", "meta_contract_sha1", "meta_urdf_sha1"):
        assert key in d
    # 되짚기(--reverse)도 된다: 홈에 서 있으면 진입 램프 간극 0
    plan, _, n_ramp = R.build_plan(frames, start=frames[-1].copy(), pub_dt=dt, reverse=True)
    assert np.allclose(plan[-1], frames[0])


def test_saved_npz_within_profile_joint_limits():
    if not NPZ.exists():
        pytest.skip("npz 없음")
    pytest.importorskip("yaml")
    W = _load("home_path_world")
    lim = W.load_profile_limits(W.PROFILE_DEFAULT)
    d = np.load(NPZ)
    joints = [str(j) for j in d["meta_joints"]]
    lo = np.array([lim[j][0] for j in joints])
    hi = np.array([lim[j][1] for j in joints])
    q = d["arm_target"]
    assert (q >= lo - 1e-9).all() and (q <= hi + 1e-9).all()
    assert lim["r_aj_4"][0] == 0.0 and lim["r_aj_6"] == pytest.approx((-0.785398, 0.785398))


@pytest.fixture(scope="module")
def checker():
    pytest.importorskip("mujoco")
    pytest.importorskip("trimesh")
    W = _load("home_path_world")
    contract = P.load_contract(W.CONTRACT_DEFAULT)
    world = W.build_world(W.WorldSpec())
    lim = W.load_profile_limits(W.PROFILE_DEFAULT)
    lo = np.array([lim[j][0] for j in world.moving_joints])
    hi = np.array([lim[j][1] for j in world.moving_joints])
    scenes = P.build_scenes(contract, "right", "both", "contract")
    return P.Checker(world, scenes, 0.02, np.zeros(7), (lo, hi))


def test_table_geometry_comes_from_env_usda(checker):
    top = [b for b in checker.w.table_boxes if b["hi"][2] > 0.2]
    assert len(top) == 1
    assert top[0]["hi"][2] == pytest.approx(0.205, abs=1e-6)   # env.yaml table_surface_z


def test_checker_detects_hand_pushed_into_table(checker):
    # 팔을 앞으로 들어 손을 상판 높이로 — 계약 홈에서 어깨를 내려 손을 상판 속으로
    q = HOME.copy()
    q[3] = 0.0                                          # 팔꿈치를 펴면 손이 상판(z 0.205) 아래로 내려간다
    v = checker.check(q)
    assert not v.ok
    assert any(checker.w.is_world(n) for n in v.worst_pair)


def test_goal_pose_is_clear(checker):
    v = checker.check(HOME)
    assert v.ok and v.min_clear >= 0.02


def test_saved_path_passes_dense_check(checker):
    if not NPZ.exists():
        pytest.skip("npz 없음")
    d = np.load(NPZ)
    if float(d["meta_margin"]) != 0.02 or bool(d["meta_with_cup"]) or str(d["meta_hand_start"]) != "contract":
        pytest.skip("기본 설정으로 만든 npz 가 아니다")
    rep = checker.check_path(np.asarray(d["meta_waypoints"]))
    assert rep["ok"], rep["fails"][:3]
    assert rep["min_clear"] >= 0.02


def test_straight_line_to_home_hits_table(checker):
    """goto_home 직선은 차렷에서 상판 앞 모서리를 친다(이 도구를 만든 이유)."""
    rep = checker.check_path(np.stack([np.zeros(7), HOME]))
    assert not rep["ok"]
    assert any("table_8" in f["pair"] for f in rep["fails"])


def test_abduction_box_caps_the_sideways_swing_and_keeps_start_and_goal_inside():
    # 09.22 사용자: "옆으로 가지 말고 … j1,4 를 동시에" — j2(옆 벌림) 상한, j3 묶음. 좌팔은 부호 반대.
    import numpy as np
    lo, hi = np.full(7, -3.0), np.full(7, 3.0)
    start, goal = np.zeros(7), np.array([-1.2, 0.67, 0.19, 1.73, 0.69, 0.04, 0.95])
    lo2, hi2 = P.abduction_box(lo, hi, "right", 0.9, start, goal)
    assert hi2[1] == 0.9 and lo2[1] == -3.0 and (lo2[2], hi2[2]) == (-0.3, 0.6)
    assert np.all(lo2 <= np.minimum(start, goal)) and np.all(hi2 >= np.maximum(start, goal))
    lg = np.array([-0.36, -0.64, 0.03, 0.43, -0.27, -0.58, -0.73])
    lo3, hi3 = P.abduction_box(lo, hi, "left", 0.9, start, lg)
    assert lo3[1] == -0.9 and hi3[1] == 3.0 and (lo3[2], hi3[2]) == (-0.6, 0.3)
    wide = np.array([0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0])                            # 목표가 상한 밖이면 목표까지는 넓힌다
    assert P.abduction_box(lo, hi, "right", 0.9, start, wide)[1][1] == 1.2

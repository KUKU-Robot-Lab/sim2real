"""plan_home_path.py — 차렷 -> 홈 오프라인 경로 계획기 (ROS·GPU 없음).

시간 매개화가 pd ramp_speed(설정값, 09.23 부터 0.2 rad/s)를 지키는지, npz 가 replay_to_pd.load_frames 로 읽히는지,
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
SHORT_CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m-short_bi_rl/deploy_contract.json"   # tools/home_path_world.CONTRACT_DEFAULT
# 11.08 사용자가 지운 파일 — 없으면 skip
gone_SHORT_CONTRACT = pytest.mark.skipif(not SHORT_CONTRACT.exists(), reason="deleted by user 11.08: logs/policy/asset_openarm_dg5f-m-short_bi_rl/deploy_contract.json")
NPZ = SIM2REAL / "deploy/policy_control/paths/home_right.npz"   # 실기가 실제로 재생하는 저장 경로

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
    import yaml
    want = yaml.safe_load(P.PD_CONFIG_DEFAULT.read_text())["ramp_speed"]
    got = P.read_ramp_speed(P.PD_CONFIG_DEFAULT)
    assert got == pytest.approx(want) and 0.0 < got <= 0.5        # 0.5 rad/s 넘는 값은 도구가 거부한다


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
    assert np.abs(np.diff(frames, axis=0)).max() / dt <= P.read_ramp_speed(P.PD_CONFIG_DEFAULT) + 1e-9
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
    if not SHORT_CONTRACT.exists():
        pytest.skip("deleted by user 11.08: logs/policy/asset_openarm_dg5f-m-short_bi_rl/deploy_contract.json")
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
    top = [b for b in checker.w.table_boxes if b["hi"][2] > 0.2 and b["name"].startswith("table")]
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


@gone_SHORT_CONTRACT
def test_saved_path_passes_dense_check():
    """저장 경로를 **그 경로 자신의 메타**(손 자세 · 반대팔 · 여유)로 다시 조밀 검사한다.

    세계(테이블 · 뒤 박스 · 옆 벽)는 기본값이므로, 나중에 장애물이 바뀌면 이 테스트가 먼저 깨진다.
    """
    if not NPZ.exists():
        pytest.skip("npz 없음")
    pytest.importorskip("mujoco")
    d = np.load(NPZ)
    W = P.W
    side = "right" if str(d["meta_joints"][0]).startswith("r_") else "left"
    contract = P.load_contract(W.CONTRACT_DEFAULT)
    #: 손 봉투 구로 계획한 경로는 손 관절값이 세계에 없다 — 계획기와 같은 구로 세계를 짓는다(09.23).
    sphere = float(d["meta_hand_sphere"]) if "meta_hand_sphere" in d else float("nan")
    sphere = None if sphere != sphere else sphere
    hand_start = str(d["meta_hand_start"])
    scenes = P.build_scenes(contract, side, str(d["meta_other_arm"]),
                            "contract" if hand_start == "sphere" else hand_start,
                            P.parse_hand_q(str(d["meta_hand_q"]) or None))
    # Checker 가 excluded_pairs 를 쓰므로 세계를 따로 짓는다
    world = W.build_world(W.WorldSpec(side=side, hand_sphere=sphere))
    lim = W.load_profile_limits(W.PROFILE_DEFAULT)
    lo = np.array([lim[j][0] for j in world.moving_joints])
    hi = np.array([lim[j][1] for j in world.moving_joints])
    chk = P.Checker(world, scenes, float(d["meta_margin"]), np.asarray(d["meta_start"]), (lo, hi))
    assert [P.fmt_pair(k) for k in chk.always_contact] == [str(s) for s in d["meta_always_contact"]]
    rep = chk.check_path(np.asarray(d["meta_waypoints"]))
    assert rep["ok"], rep["fails"][:3]
    assert rep["min_clear"] >= float(d["meta_margin"]) - 1e-6


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


def test_the_back_box_and_side_walls_are_in_the_world_by_default():
    # 09.23 사용자: 로봇 원점 기준 뒤 고정 박스 x -1.0~-0.25 · y ±0.45 · z 0~0.22, 그리고 옆 벽 y = ±0.45.
    import numpy as np
    W = P.W
    boxes = {b["name"]: b for b in W.wall_boxes()}
    assert set(boxes) == {"back_box", "wall_y_neg", "wall_y_pos"}
    b = boxes["back_box"]
    assert b["lo"] == pytest.approx([-1.0, -0.45, 0.0]) and b["hi"] == pytest.approx([-0.25, 0.45, 0.22])
    assert boxes["wall_y_pos"]["lo"][1] == pytest.approx(0.45) and boxes["wall_y_neg"]["hi"][1] == pytest.approx(-0.45)
    assert np.all(boxes["wall_y_pos"]["hi"][2] > 1.0) and np.all(boxes["wall_y_pos"]["lo"][2] < 0.0)
    assert W.wall_boxes({}, None) == []


# ---- 10.13 실기: 정책이 멈춘 자리의 손목 j7 1.5959 가 한계 1.5708 밖이라 rehome 이 거부됐다
LIM = {f"r_aj_{i}": (-1.5708, 1.5708) for i in range(1, 8)}
JOINTS = list(LIM)


def test_a_start_just_past_the_limit_is_clipped_with_a_note():
    q = np.zeros(7); q[6] = 1.5959
    got, notes = P.clip_start_into_limits(q, JOINTS, LIM, 0.04)
    assert got[6] == pytest.approx(1.5708) and len(notes) == 1 and "r_aj_7" in notes[0]
    assert q[6] == 1.5959                                   # 입력은 그대로(복사본을 고친다)


def test_a_start_far_past_the_limit_is_left_for_the_limit_check_to_refuse():
    q = np.zeros(7); q[6] = 1.70; q[0] = -1.60
    got, notes = P.clip_start_into_limits(q, JOINTS, LIM, 0.04)
    assert got[6] == 1.70 and got[0] == pytest.approx(-1.5708) and len(notes) == 1


def test_zero_slack_changes_nothing():
    q = np.zeros(7); q[6] = 1.58
    got, notes = P.clip_start_into_limits(q, JOINTS, LIM, 0.0)
    assert got[6] == 1.58 and notes == []


def test_rehome_passes_a_slack_smaller_than_the_path_start_check():
    RH = _load("plan_rehome")
    q = {f"r_aj_{i}": 0.1 for i in range(1, 8)} | {"r_hj_1": 0.2}
    argv = RH.planner_argv("right", q, Path("/tmp/x.npz"), "rh56f1")
    slack = float(argv[argv.index("--start-slack") + 1])
    assert 0.0 < slack < 0.05 and slack == RH.START_SLACK


# ---- 11.11 홀더 원기둥(rehome 이 홀더 1 에 꽂힌 파랑 윗테를 스침, root_cause/cocktail_1111/q2) ----
HOLDERS_YAML = SIM2REAL / "config/cup_holder_poses_5090_v1.yaml"
RH_ENV = SIM2REAL / "deploy/policy_control/config/home_path/rh56f1_world_env.yaml"
ACCIDENT = SIM2REAL / "logs/runs/261110/episode/184028_cocktail_1c_e4000/rehome_right_home_r_after_blue.npz"


def test_holder_cylinders_cover_a_seated_cup_rim():
    import home_path_world as W
    objs = [o for o in P.holder_objects(HOLDERS_YAML, W.load_env_yaml(RH_ENV)) if "radius" in o]
    assert [o["name"] for o in objs] == ["cup_holder_0", "cup_holder_1", "cup_holder_2"]
    h1 = objs[1]
    # 홀더 1 에 꽂힌 파랑: 원점 (0.3731, -0.0563, 0.2930), 반지름 30 mm, 윗테 = 원점 + 0.155 = 0.448
    assert np.hypot(0.3731 - h1["center"][0], -0.0563 - h1["center"][1]) + 0.030 < h1["radius"]
    assert h1["center"][2] + h1["half_height"] > 0.448 + 0.01
    assert h1["center"][2] - h1["half_height"] <= 0.235 - 0.05 + 1e-9


def test_holders_refuse_a_world_not_in_base_link():
    env = {"robot_cfg": {"init_state": {"pos": [0.0, 0.0, 0.1], "rot": [1.0, 0.0, 0.0, 0.0]}}}
    with pytest.raises(SystemExit):
        P.holder_objects(HOLDERS_YAML, env)


def test_rehome_puts_holders_in_the_world_by_default_and_can_turn_them_off():
    RH = _load("plan_rehome")
    q = {f"r_aj_{i}": 0.1 for i in range(1, 8)} | {"r_hj_1": 0.2}
    on = RH.planner_argv("right", q, Path("/tmp/x.npz"), "rh56f1")
    assert on[on.index("--holders") + 1] == str(RH.HOLDERS) and RH.HOLDERS.exists()
    assert "--holders" not in RH.planner_argv("right", q, Path("/tmp/x.npz"), "rh56f1", None)


@pytest.mark.skipif(not ACCIDENT.exists(), reason="11.10 사고 rehome npz 없음")
def test_accident_straight_line_is_refused_only_with_holders():
    """11.10 184028: 홀더 없는 세계는 직선을 통과시켰고(실기에서 손목이 파랑 윗테를 스침) 홀더 원기둥은 거절한다."""
    pytest.importorskip("mujoco")
    pytest.importorskip("trimesh")
    RH = _load("plan_rehome")
    rp = np.load(ACCIDENT, allow_pickle=True)
    q = {f"r_aj_{i + 1}": float(v) for i, v in enumerate(rp["meta_start"])}
    q |= {k: float(v) for k, v in (kv.split("=") for kv in str(rp["meta_hand_q"]).split(","))}
    argv = RH.planner_argv("right", q, Path("/tmp/unused.npz"), "rh56f1")[2:] + ["--check-only"]
    assert P.main(argv) == 1
    i = argv.index("--holders")
    assert P.main(argv[:i] + argv[i + 2:]) == 0


# ---- 11.11 rehome 세계에 탁자 위 컵(episode_ros → plan_rehome --table-cups) + 홀더 몸체 상자
def test_rehome_table_cups_takes_snapshot_then_live_then_region_and_skips_held_or_seated():
    from types import SimpleNamespace
    from policy_control.episode_ros import rehome_table_cups
    from policy_control.episode_world import World
    objs = {n: {"topic": f"/objects/{r}/pose", "region": {"x": [0.26, 0.305], "y": [-0.365, -0.33]}}
            for n, r in (("PINK", "source240_pink"), ("BLUE", "source240_blue"), ("SHAKER", "shaker_c"))}
    ep = SimpleNamespace(objects=objs)
    w = World(objects={"PINK": {"at": "table", "pos": (0.3, -0.34, 0.29)}, "BLUE": {"at": "holder:HOLDER_1"},
                       "SHAKER": {"at": "left_hand"}})
    cups, notes = rehome_table_cups(ep, w, {}, {})
    assert [c["name"] for c in cups] == ["PINK"] and cups[0]["pos"] == [0.3, -0.34, 0.29] and "spread" not in cups[0]
    pink = World(objects={"PINK": {"at": "table"}})
    cups, _ = rehome_table_cups(ep, pink, {}, {"PINK": [((0.29, -0.35, 0.29), None, 100.0)]}, now=102.0)
    assert cups[0]["pos"] == [0.29, -0.35, 0.29]                                  # snapshot 없음 → 신선한 live 마지막
    cups, notes = rehome_table_cups(ep, pink, {}, {})
    assert cups[0]["pos"] == [0.2825, -0.3475] and "spread" not in cups[0] and "region" in notes[0]


def test_rehome_table_cups_ignores_stale_live_and_refuses_an_unknown_cup_without_pose():
    """★11.12 리뷰 H2 · H1: 5 s 넘은 live 는 자세 없음, unknown 컵이 자세가 없으면 이름을 담아 거부."""
    from types import SimpleNamespace
    from policy_control.episode_ros import REHOME_LIVE_MAX_AGE_S, RehomeCupUnknown, rehome_table_cups
    from policy_control.episode_world import World
    ep = SimpleNamespace(objects={"PINK": {"topic": "/objects/source240_pink/pose",
                                           "region": {"x": [0.26, 0.305], "y": [-0.365, -0.33]}}})
    old = {"PINK": [((0.1, -0.1, 0.29), None, 100.0)]}
    now = 100.0 + REHOME_LIVE_MAX_AGE_S + 0.5
    cups, notes = rehome_table_cups(ep, World(objects={"PINK": {"at": "table"}}), {}, old, now=now)
    assert cups[0]["pos"] == [0.2825, -0.3475] and any("낡음" in n for n in notes)  # table → region 으로 대신
    with pytest.raises(RehomeCupUnknown, match="PINK"):
        rehome_table_cups(ep, World(objects={"PINK": {"at": "unknown"}}), {}, old, now=now)
    cups, _ = rehome_table_cups(ep, World(objects={"PINK": {"at": "unknown"}}), {}, old, now=101.0)
    assert cups[0]["pos"] == [0.1, -0.1, 0.29]                                    # 신선하면 unknown 도 live 로


def test_rehome_table_cups_notes_a_non_object_topic_and_unknown_object_is_an_error():
    from types import SimpleNamespace
    from policy_control.episode_ros import rehome_table_cups
    from policy_control.episode_world import World
    ep = SimpleNamespace(objects={"CUP": {"topic": "/episode/cup"}})
    cups, notes = rehome_table_cups(ep, World(objects={"CUP": {"at": "table"}}), {}, {})
    assert cups == [] and any("CUP" in n and "실물 토픽 아님" in n for n in notes)            # 리뷰 M3
    with pytest.raises(SystemExit, match="no_such_cup"):
        P.table_cup_objects([{"name": "X", "object": "no_such_cup", "pos": [0.3, -0.3]}], P.W.load_env_yaml(RH_ENV))


def test_rehome_adds_the_measured_other_arm_scene():
    """★11.12: 반대팔 실측이 있으면 home · zero 에 실측 장면을 더한다(184028 s3), 없으면 home · zero 만."""
    import json
    RH = _load("plan_rehome")
    q = {f"r_aj_{i}": 0.1 for i in range(1, 8)} | {"r_hj_1": 0.2}
    assert not any(a.startswith("--other-q") for a in RH.planner_argv("right", q, Path("/tmp/x.npz"), "rh56f1"))
    left = {f"l_aj_{i}": -0.1 * i for i in range(1, 8)}
    argv = RH.planner_argv("right", q | left, Path("/tmp/x.npz"), "rh56f1")
    (oq,) = [a for a in argv if a.startswith("--other-q=")]
    assert P.parse_hand_q(oq.split("=", 1)[1]) == pytest.approx(left)
    path = SIM2REAL / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
    if not path.exists():
        pytest.skip("rh56f1 계약 없음")
    contract = json.loads(path.read_text())
    base = P.build_scenes(contract, "right", "both", "contract")
    more = P.build_scenes(contract, "right", "both", "contract", None, left)
    assert len(more) == len(base) + 1 and all(more[-1][k] == pytest.approx(v) for k, v in left.items())


def test_table_cups_and_holder_base_enter_the_planning_world():
    env = P.W.load_env_yaml(RH_ENV)
    (c,) = P.table_cup_objects([{"name": "PINK", "object": "source240_pink", "pos": [0.30, -0.34]}], env)
    assert c["name"].startswith("cup") and 0.044 < c["radius"] < 0.047                         # 30 mm + 15 mm
    assert c["center"][2] - c["half_height"] == pytest.approx(float(env["table_surface_z"]), abs=1e-3)
    hs = P.holder_objects(HOLDERS_YAML, env)
    bases = [o for o in hs if o["name"].endswith("_base")]
    assert len(bases) == len(hs) // 2 and all(max(o["half"][:2]) > P.HOLDER_CYL_R for o in bases)
    world = P.W.build_world(P.W.WorldSpec(urdf=P.W.URDF_DEFAULT, env_yaml=RH_ENV, side="right",
                                          objects=hs + (c,)))
    names = {world.model.body(i).name for i in range(world.model.nbody)}
    assert {"cup_tbl_PINK", bases[0]["name"]} <= names and world.is_world(bases[0]["name"])

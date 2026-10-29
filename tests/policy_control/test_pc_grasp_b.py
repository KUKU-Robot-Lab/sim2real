"""★10.19 1-B — 정책 없이 제어 모듈만: ready_carry 접근(plan_reach) → [확인] 닫기(hand_grasp) → 수직 10 cm · 3 s(plan_lift).

사용자 결정(10.19): 쥐는 자리 = tilt 기준 자리 · 접근 = 앞에서 곧게 · 승인 = 닫기 전만 · 왼손 / 오른손 독립 실행.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from policy_control import episode_spec as S
from policy_control import hdgp_pin as H
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_runner import SUCCESS, EpisodeManager

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "deploy" / "policy_control" / "tools"
ASSET = REPO / "deploy" / "policy_control" / "config" / "tilt" / "base_arc6"
HOME = yaml.safe_load((REPO / "deploy/policy_control/config/homes/rh56f1_aglt.yaml").read_text())
# 10.19 실기 컵 자리(FP++ 원점, base) — 오른손 src240b 접근 카드 bag · 왼손 shaker sim_traj 계획
CUPS = {"right": ("source240b_blue", [0.244, -0.116, 0.288]), "left": ("shaker_c_orange", [0.259, 0.078, 0.262])}


def _tool(name):
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pins_hold() -> bool:
    try:
        cfg = H.load_policy(ASSET)
        H.verify(H.hdgp_dir(), cfg["hdgp"]["md5"])
        H.verify_refs(ASSET, cfg.get("ref") or {})
        return True
    except (H.PinError, OSError):
        return False


needs_hdgp = pytest.mark.skipif(not _pins_hold(), reason="hdgp 핀 · 기준 파일이 이 호스트에서 맞지 않는다")


# ---------------------------------------------------------------- 에피소드 형식
@pytest.mark.parametrize("side", ["right", "left"])
def test_each_arm_has_its_own_policy_free_episode(side):
    ep = S.load(REPO / f"config/episodes/grasp_b_{side}.yaml")
    assert S.episode_sides(ep) == [side] and not ep.policies
    assert [n.id for n in ep.nodes] == ["go_home_start", "scene", "open_hand", "reach", "confirm_close", "grasp", "lift", "done"]
    assert ep.trajectories["reach"].object == "CUP" and not ep.trajectories["reach"].cups
    assert ep.trajectories["lift"].cups and ep.nodes[5].hand_contract == f"config/hand/rh56f1_{side}_hand.json"   # ★10.20 1-B 손 계약 고정본


def _raw(side="right"):
    return yaml.safe_load((REPO / f"config/episodes/grasp_b_{side}.yaml").read_text())


@pytest.mark.parametrize("patch, why", [
    ({"reach": {"kind": "reach", "sides": ["right", "left"], "asset": "tilt/base_arc6", "cups": False, "object": "CUP"}}, "한 팔"),
    ({"reach": {"kind": "reach", "sides": ["right"], "asset": "tilt/base_arc6", "cups": False}}, "object"),
    ({"reach": {"kind": "reach", "sides": ["right"], "asset": "tilt/base_arc6", "cups": True, "object": "CUP"}}, "빈손"),
    ({"lift": {"kind": "lift", "sides": ["right"], "asset": "tilt/base_arc6", "object": "CUP"}}, "쥔 손"),
    ({"go_home": {"kind": "rehome", "sides": ["right"], "object": "CUP"}}, "object 는"),
    ({"reach": {"kind": "reach", "sides": ["right"], "asset": "tilt/base_arc6", "cups": False, "object": "BOWL"}}, "objects 에 없다"),
])
def test_bad_reach_and_lift_are_refused(patch, why):
    d = _raw()
    d["trajectories"].update(patch)
    with pytest.raises(S.EpisodeSpecError, match=why):
        S.parse(d, path="x.yaml")


def test_one_confirm_is_enough_only_right_after_the_robot_reached():
    d = _raw()
    ok = S.parse(d, path="x.yaml")
    assert ok.nodes[4].type == "confirm"
    d["sequence"] = [n for n in d["sequence"] if n["id"] != "reach"]          # 사람이 컵을 대는 경우 = 확인 두 번(경로 B)
    with pytest.raises(S.EpisodeSpecError, match="confirm 두 개"):
        S.parse(d, path="x.yaml")
    d = _raw()
    d["sequence"] = [n for n in d["sequence"] if n["id"] != "confirm_close"]    # 닫기 전 확인은 언제나
    with pytest.raises(S.EpisodeSpecError, match="닫기 전"):
        S.parse(d, path="x.yaml")


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_fake_run_ends_holding_the_cup_after_one_confirmation(side):
    ep = S.load(REPO / f"config/episodes/grasp_b_{side}.yaml")
    ex = FakeExecutor(ep)
    asked = []
    m = EpisodeManager(ep, ex, approve=lambda what, why: asked.append(what) or True, episode_id="t")
    m.run()
    assert m.status == SUCCESS and m.world.hand(side) == "CUP"
    assert [c[2] for c in ex.calls if c[0] == "trajectory"] == ["go_home", "reach", "lift"]
    assert sum("confirm" in str(w) for w in asked) <= 1


# ---------------------------------------------------------------- 실행기(ROS 없이)
def _ros_self(tmp_path, kind: str, holds: str, rcs: list[int], frames=None, fresh=None):
    from policy_control.episode_ros import RosExecutor
    calls = []

    def run(argv):
        calls.append(argv)
        return rcs[len(calls) - 1], "tail"
    ep = S.load(REPO / "config/episodes/grasp_b_right.yaml")
    seq = iter(fresh or [])
    me = SimpleNamespace(_wait_pd_tracking=lambda side: True, _wait_arm_still=lambda sides: {"arm_still": None}, _pd_view=lambda side: {}, run_dir=tmp_path, _run=run,
                         cancel=threading.Event(), ep=ep, _lock=threading.Lock(), _frames={"CUP": frames or []},
                         _fresh_object=lambda name, wait_s=2.0: (0.244, -0.116, 0.288),   # locate snapshot(그대로)
                         _live_object=lambda name, wait_s=2.0: next(seq, None))
    me._still_object = lambda name, wait_s=0.2: RosExecutor._still_object(me, name, wait_s)
    world = SimpleNamespace(hand=lambda side: holds)
    return (lambda: RosExecutor._run_planned(me, SimpleNamespace(id=kind), ep.trajectories[kind], world)), calls


def _still(n=8):
    import time
    now = time.monotonic()
    return [((0.244, -0.116, 0.288), (1.0, 0.0, 0.0, 0.0), now - 0.05 * k) for k in range(n)]


def test_reach_plans_from_the_still_cup_and_replays_seven_joints(tmp_path):
    go, calls = _ros_self(tmp_path, "reach", "EMPTY", [0, 0], frames=_still())
    r = go()
    assert r.status == "completed" and len(calls) == 2
    pl, rp = calls
    assert Path(pl[1]).name == "plan_reach.py" and pl[pl.index("--object") + 1] == "source240b_blue"
    assert pl[pl.index("--side") + 1] == "right" and [float(v) for v in pl[pl.index("--cup") + 1].split(",")] == [0.244, -0.116, 0.288]
    assert pl[pl.index("--quat") + 1].startswith("1.0")
    assert Path(rp[1]).name == "replay_to_pd.py" and rp[rp.index("--npz") + 1].endswith("reach_cmd.npz")
    assert rp[rp.index("--joints") + 1] == ",".join(f"r_aj_{i}" for i in range(1, 8)) and "--execute" in rp


def test_reach_without_a_still_cup_refuses_without_moving(tmp_path):
    go, calls = _ros_self(tmp_path, "reach", "EMPTY", [], frames=[])
    r = go()
    assert r.status == "refused" and "컵 자세" in r.reason and calls == []


def test_reach_with_a_cup_already_in_hand_refuses(tmp_path):
    go, calls = _ros_self(tmp_path, "reach", "CUP", [], frames=_still())
    assert go().status == "refused" and calls == []


def test_a_failed_reach_plan_never_replays(tmp_path):
    go, calls = _ros_self(tmp_path, "reach", "EMPTY", [2], frames=_still())
    r = go()
    assert r.status == "refused" and len(calls) == 1


@pytest.mark.parametrize("after_z, status", [(0.378, "completed"), (0.30, "aborted")])
def test_lift_checks_that_the_cup_rose(tmp_path, after_z, status):
    go, calls = _ros_self(tmp_path, "lift", "CUP", [0, 0], fresh=[(0.244, -0.116, 0.288), (0.244, -0.116, after_z)])
    r = go()
    assert r.status == status and Path(calls[0][1]).name == "plan_lift.py" and calls[0][calls[0].index("--side") + 1] == "right"
    assert r.signals["cup_rise_m"] == pytest.approx(after_z - 0.288, abs=1e-4)


def test_live_topic_sits_next_to_the_snapshot_topic():
    from policy_control.episode_ros import live_topic
    assert live_topic("/objects/shaker_c_orange/pose") == "/objects/shaker_c_orange/live_pose"
    assert live_topic("/episode/left/objects/CUP/relay") is None


def test_lift_without_fpp_completes_but_says_the_rise_is_unjudged(tmp_path):
    go, _calls = _ros_self(tmp_path, "lift", "CUP", [0, 0], fresh=[None, None])
    r = go()
    assert r.status == "completed" and "판정 보류" in r.reason and r.signals["cup_rise_m"] is None


def test_real_object_name_comes_from_the_pose_topic():
    from policy_control.episode_ros import real_object_name
    assert real_object_name("/objects/source240b_blue/pose") == "source240b_blue"
    assert real_object_name("/episode/objects/CUP/pose") == "" and real_object_name("") == ""


# ---------------------------------------------------------------- 계획 도구(순수)
def test_the_fpp_origin_is_moved_so_the_bottoms_meet():
    R = _tool("plan_reach")
    assert R.cup_origin_d4([0.3, 0.0, 0.262], 0.065)[2] == pytest.approx(0.282)     # shaker_c 원점 65 mm → 85 mm
    assert R.cup_origin_d4([0.3, 0.0, 0.288], 0.085)[2] == pytest.approx(0.288)
    assert R.quat_tilt_deg([1, 0, 0, 0]) == pytest.approx(0.0)
    assert R.quat_tilt_deg([np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0]) == pytest.approx(20.0, abs=1e-6)


def test_palm_targets_point_the_fingers_along_the_requested_azimuth():
    R = _tool("plan_reach")
    T_to = np.eye(4)
    T_to[:3, 3] = [-0.011, 0.041, 0.032]                       # 오른손 기준 자리(손바닥 좌표 컵 원점)
    T_to[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)           # 아무 회전(det +1)
    for az in (0.0, 0.7, -1.2):
        Rm, p, a = R.palm_targets(T_to, np.array([0.3, -0.1, 0.29]), az)
        f = Rm @ np.array([0.0, 1.0, 0.0])
        assert np.degrees(np.arctan2(f[1], f[0])) == pytest.approx(np.degrees(az), abs=1e-6)
        assert a == pytest.approx([np.cos(az), np.sin(az), 0.0])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rm, p
        cup = T @ T_to
        assert cup[:3, 3] == pytest.approx([0.3, -0.1, 0.29])   # 손 안 가상 컵 = 실물 컵
        assert cup[:3, 2] == pytest.approx([0.0, 0.0, 1.0])     # 가상 컵 축이 선다


def test_cup_surface_distance_is_negative_inside():
    R = _tool("plan_reach")
    d = R.cup_surface_dist(np.array([[0.0, 0.0, 0.1], [0.05, 0.0, 0.1], [0.03, 0.0, 0.3], [0.0, 0.0, -0.01]]), [0, 0], 0.0, 0.2, 0.03)
    assert d[0] < 0 and d[1] == pytest.approx(0.02) and d[2] == pytest.approx(0.1) and d[3] == pytest.approx(0.01)


# ---------------------------------------------------------------- 계획 도구(오프라인 — 홈 출발, 오늘 컵 자리)
@needs_hdgp
@pytest.mark.parametrize("side", ["right", "left"])
def test_reach_from_home_to_todays_cup_passes_every_check(tmp_path, side):
    R = _tool("plan_reach")
    obj, cup = CUPS[side]
    ok, meta = R.plan(side, np.array(cup), np.array([1.0, 0, 0, 0]), obj, tmp_path, {s: HOME[s] for s in ("right", "left")},
                      H.hdgp_dir())
    assert ok, meta["problems"]
    z = np.load(tmp_path / "reach_cmd.npz")
    q = z["arm_target"]
    assert list(z["joint_names"]) == [f"{side[0]}_aj_{i}" for i in range(1, 8)]
    assert np.abs(q[0] - HOME[side]).max() < 1e-3 and np.abs(np.diff(q, axis=0)).max() * 60 <= 2.0
    a, d = meta["approach"], meta["deep_in"]
    # 10.20 aglt_new 사전 파지 + ready_carry 이송(걸리면 관절 공간 · RRT) — 계획이 로봇 PC 에서 몇 초 안(옛 직선 IK 탐색 50 s)
    assert meta["plan_wall_s"] < 20.0
    assert a["method"] in ("ready_carry", "joint_line", "rrt") and a["arrived"] and a["r_mult"] == 2.5
    assert a["h_m"] in R.PRE_H_M and a["vmax_rad_s"] <= 0.72 + 1e-6
    # 10.20 사용자 "테이블, 자기 충돌, 관절 제약": 홈 경로 세계 충돌 2 cm · 컵 1 cm · 관절 여유 5°(URDF · 프로필)
    assert a["collision"]["ok"] and a["margin_min_deg"] >= 5.0 and a["cup_clear_mm"] >= 10.0 and a["hand_table_mm"] >= 15.0
    assert d["ok"] and d["collision"]["ok"] and d["line_dev_mm"] <= 3.0 and d["path_pos_dev_mm"] <= 3.0
    assert d["path_ang_dev_deg"] <= 2.0 and d["rot_deg"] <= 30.0
    assert d["margin_min_deg"] >= 5.0 and d["in_cup_clear_mm"] >= 3.0 and d["vmax_rad_s"] <= 0.72
    assert z["n_carry"] == a["steps"] + 1 and len(q) == a["steps"] + d["n"]
    assert np.abs(np.diff(q, axis=0)).max() * 60 <= 0.72 + 1e-3, "이송 끝 · 들어가기 시작 이음매도 매끄럽다(float32)"
    # ★10.20 실기: 손바닥 ↔ 컵 겉면 수평 거리는 shaker 깊이(GRASP_PALM_GAP_M) 이하 — src240 은 그만큼 더 들어간다
    assert d["palm_gap_mm"] <= R.GRASP_PALM_GAP_M * 1000 + 0.5 and d["deeper_mm"] >= 0.0
    # 손 안 가상 컵(tilt 기준 자리)이 끝에서 실물 컵 원점(바닥 맞춤)에 온다 — 더 들어간 만큼만 수평으로 어긋난다
    P = R.H.load_ready_carry(R.H.hdgp_dir())
    import torch
    chn = P.chain.Chain.from_urdf(P.chain.default_urdf(), f"{side[0]}_hl_palm_sensor", R.ARMS[side])
    ref = dict(np.load(ASSET / "ref" / "single734_t120.npz", allow_pickle=False))
    T_to = R.cup_in_palm(P, chn, ref, side)[0].double().numpy()
    T = chn.fk(torch.tensor(q[-1:], dtype=torch.float32))[0].double().numpy() @ T_to
    off = T[:3, 3] - np.asarray(meta["cup_origin_d4"])
    assert abs(off[2]) < 2e-3 and T[2, 2] > np.cos(np.radians(3))
    assert np.linalg.norm(off[:2]) == pytest.approx(d["deeper_mm"] / 1000, abs=2e-3)


def test_deepen_moves_the_palm_toward_the_cup_axis_only_when_too_far():
    R = _tool("plan_reach")
    cup = np.array([0.3, -0.1, 0.25])
    p, extra = R.deepen(np.array([0.3, -0.1 + 0.052, 0.27]), cup, 0.030)          # src240: 22 mm → 16 mm
    assert extra == pytest.approx(0.006) and p == pytest.approx([0.3, -0.1 + 0.046, 0.27])
    p, extra = R.deepen(np.array([0.3 + 0.050, -0.1, 0.27]), cup, 0.0354)         # shaker 14.6 mm: 그대로
    assert extra == 0.0 and p == pytest.approx([0.35, -0.1, 0.27])


@needs_hdgp
def test_reach_refuses_a_tipped_cup_and_a_cup_out_of_reach(tmp_path):
    R = _tool("plan_reach")
    q = {s: HOME[s] for s in ("right", "left")}
    ok, meta = R.plan("right", np.array([0.244, -0.116, 0.288]), np.array([np.cos(0.15), np.sin(0.15), 0, 0]), "source240b_blue",
                      tmp_path / "a", q, H.hdgp_dir())
    assert not ok and "기울었다" in meta["problems"][0] and not (tmp_path / "a" / "reach_cmd.npz").exists()
    ok, meta = R.plan("right", np.array([0.75, -0.116, 0.288]), None, "source240b_blue", tmp_path / "b", q, H.hdgp_dir())
    assert not ok and (tmp_path / "b" / "check.json").is_file() and not (tmp_path / "b" / "reach_cmd.npz").exists()


@needs_hdgp
@pytest.mark.parametrize("side", ["right", "left"])
def test_lift_goes_straight_up_ten_cm_and_holds_three_seconds(tmp_path, side):
    R, L = _tool("plan_reach"), _tool("plan_lift")
    obj, cup = CUPS[side]
    ok, _ = R.plan(side, np.array(cup), None, obj, tmp_path / "r", {s: HOME[s] for s in ("right", "left")}, H.hdgp_dir())
    assert ok
    q = {s: HOME[s] for s in ("right", "left")}
    q[side] = np.load(tmp_path / "r" / "reach_cmd.npz")["arm_target"][-1].tolist()
    ok, meta = L.plan(side, tmp_path / "l", q, H.hdgp_dir())
    assert ok and meta["rise_m"] == pytest.approx(0.10, abs=1e-3) and meta["line_dev_mm"] <= 3.0 and meta["margin_min_deg"] >= 5.0
    assert meta["collision"]["ok"] and meta["profile_margin_min_deg"] >= 5.0       # 10.20 테이블 · 자기 충돌 · 관절 제약
    a = np.load(tmp_path / "l" / "lift_cmd.npz")["arm_target"]
    assert np.abs(a[0] - q[side]).max() < 1e-4 and np.allclose(a[-180:], a[-1]) and len(a) == 181 + 180


def test_ready_approach_values_are_the_pinned_hdgp_cfg_defaults():
    """배포 사본(config/ready_approach/rh56f1.yaml)이 hdgp rh_aglt_{r,l} cfg 의 ready_* 기본값과 같다(jitter 만 배포 0).
    다르면 aglt_new 가 값을 바꾼 것 — diff 를 보고 배포 사본 · md5 핀을 사람이 고친다."""
    import ast
    cfg = yaml.safe_load((REPO / "deploy/policy_control/config/ready_approach/rh56f1.yaml").read_text())
    root = H.hdgp_dir() / "source/openarm/openarm/agnostic/tasks"
    if not root.is_dir():
        pytest.skip("hdgp 없음")
    for side, pkg in (("right", "rh_aglt_r"), ("left", "rh_aglt_l")):
        tree = ast.parse((root / pkg / "rh_aglt_env_cfg.py").read_text())
        d = {st.target.id: ast.literal_eval(st.value) for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef) and node.name == "RhAgltEnvCfg"
             for st in node.body if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
             and st.target.id.startswith("ready_")}
        for k, v in cfg[side].items():
            if k == "ready_pre_phi_jitter_deg":
                assert v == 0.0
                continue
            want = list(d[k]) if isinstance(d[k], tuple) else d[k]
            assert v == want, f"{side} {k}: 배포 {v} ≠ hdgp {want}"



@needs_hdgp
@pytest.mark.parametrize("side,y", [("right", 0.0), ("right", -0.06), ("left", 0.0), ("left", 0.06)])
def test_center_line_cups_pass_every_condition(tmp_path, side, y):
    """10.20 aglt_new 격자: 중심선 가까운 컵(|y| < 0.08)은 Mover 이동 중 손목 j6 · j2 가 한계에 붙었다. 조건을 지키는 경로를
    찾는다(관절 공간 · RRT · 방위 후보) — 충돌 · 관절 여유 · 속도 모두."""
    R = _tool("plan_reach")
    obj, cup = CUPS[side]
    ok, meta = R.plan(side, np.array([0.25, y, cup[2]]), None, obj, tmp_path, {s: HOME[s] for s in ("right", "left")},
                      H.hdgp_dir())
    assert ok, meta["problems"]
    a, d = meta["approach"], meta["deep_in"]
    assert a["collision"]["ok"] and d["collision"]["ok"] and min(a["margin_min_deg"], d["margin_min_deg"]) >= 5.0
    assert meta["vmax_rad_s"] <= 0.72 + 1e-3


@needs_hdgp
def test_a_cup_by_the_shoulder_is_refused_with_reasons_per_candidate(tmp_path):
    R = _tool("plan_reach")
    ok, meta = R.plan("right", np.array([0.12, -0.10, 0.288]), None, "source240b_blue", tmp_path,
                      {s: HOME[s] for s in ("right", "left")}, H.hdgp_dir())
    assert not ok and not (tmp_path / "reach_cmd.npz").exists()
    assert "조건을 지키는 접근이 없다" in meta["problems"][0]
    assert len(meta["tries"]) == len(R.PRE_H_M) * len(R.PHI_OFFSETS_DEG) and all(t.get("why") for t in meta["tries"])


@needs_hdgp
def test_world_distances_come_from_vertices_not_mujoco_mpr():
    """10.20: MuJoCo 는 얇은 상판(1 cm)과 볼록 조각을 detect margin 안에서 MPR 로 재 수평 '관통'(−254 mm)을 냈다 —
    꼭짓점 최저가 상판 13.6 mm 위였다. 세계 쌍은 꼭짓점으로 다시 잰다."""
    import importlib
    _tool("plan_reach")                                         # tools 를 sys.path 에
    W = importlib.import_module("reach_world")
    rw = W.ReachWorld("left", H.hdgp_dir(), np.array(HOME["left"]), np.array(HOME["right"]), None)
    q = np.array([-0.028, -0.195, 0.381, 0.221, -0.053, -0.185, -1.361])   # 그날 왼손 shaker 사전 파지(높이 0)
    d = rw.chk.raw(q)
    v = rw.chk._verts("l_hl_pinky_sensor")
    k = next(k for k in d if "table_8" in k and "l_hl_pinky_sensor" in k)
    assert d[k] == pytest.approx(float(v[:, 2].min()) - 0.205, abs=1e-6) and 0.01 < d[k] < 0.02

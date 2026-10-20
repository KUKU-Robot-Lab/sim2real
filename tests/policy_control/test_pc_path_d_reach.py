"""★10.20 저녁 경로 D = 1-B 접근(ready_carry, 사전 파지 자리에서 멈춤) → rcg03 정책 노드.

사용자 "grasp_b 는 그대로 두고, grasp_d 를 1-b 접근에 rcg03 으로 교체". 옛 D(ap05 접근 → 같은 노드에서 rcg03 넘김)는 실기 10.20 bag
두 회에서 ap05 가 멈춘 방위가 rcg03 학습 시작(aglt_new ReadyApproach 사전 파지)과 오른손 −25° · 왼손 +15° 달랐다.
plan_reach --stop pregrasp 는 같은 사전 파지 계산(값 · hdgp 코드)으로 거기서 멈춘다 — rcg03 노드의 시작 검사(사전 파지 3 cm)를 지난다.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from policy_control import episode_spec as S
from policy_control import hdgp_pin as H
from policy_control import rh_aglt as A

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "deploy" / "policy_control" / "tools"
ASSET = REPO / "deploy" / "policy_control" / "config" / "tilt" / "base_arc6"
HOME = yaml.safe_load((REPO / "deploy/policy_control/config/homes/rh56f1_aglt.yaml").read_text())
# 10.20 실기 경로 D bag 두 회의 컵 자리(FP++ 원점, base) — 오른손 source240b · 왼손 shaker
CUPS = {"right": ("source240b_blue", "source240_real", [0.2711, -0.1339, 0.2878]),
        "left": ("shaker_c_orange", "shaker_c_real", [0.2641, 0.0753, 0.2635])}


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
def test_path_d_reaches_the_pregrasp_then_starts_rcg03(side):
    from policy_control import policy_registry as R
    ep = S.load(REPO / "config" / "episodes" / f"grasp_d_{side}.yaml")
    assert [n.id for n in ep.nodes] == ["go_home_start", "scene", "open_hand", "reach", "grasp_lift", "done"]
    tr = ep.trajectories["reach"]
    assert (tr.kind, tr.sides, tr.cups, tr.object, tr.stop) == ("reach", (side,), False, "CUP", "pregrasp")
    (b,) = ep.policies.values()
    assert b.policy == f"rh56f1/aglt/{side}_rcg03" and b.handoff is None and b.approach_grasp is None
    entries = {e.id: e for e in R.scan(REPO / "deploy" / "policies")}
    assert not any(S.availability(ep, entries).values())
    c = A.load_contract(REPO / "deploy" / "policies" / b.policy / "rh_aglt_contract.json")
    assert c.arm_start == "ready", "사전 파지 자리에서 시작하는 정책만 이 자리에 쓴다"
    # 1-B 는 그대로(손 안 컵 자리까지 → 확인 → 닫기 → 들기)
    assert S.load(REPO / "config" / "episodes" / f"grasp_b_{side}.yaml").trajectories["reach"].stop == "grasp"


def test_tilt_base_d_grasps_like_grasp_d():
    ep = S.load(REPO / "config" / "episodes" / "tilt_base_d.yaml")
    ids = [n.id for n in ep.nodes]
    for s, side, obj in (("r", "right", "SRC"), ("l", "left", "RCV")):
        i = ids.index(f"grasp_{side}")
        assert ids[i - 2: i] == [f"open_hand_{s}", f"reach_{s}"]
        tr = ep.trajectories[f"reach_{s}"]
        assert (tr.kind, tr.sides, tr.object, tr.stop) == ("reach", (side,), obj, "pregrasp")
        b = ep.policies[f"rh_aglt_{s}"]
        assert b.policy == f"rh56f1/aglt/{side}_rcg03" and b.handoff is None


@pytest.mark.parametrize("patch", [{"lift": {"stop": "pregrasp"}}, {"reach": {"stop": "deep"}}])
def test_reach_stop_is_only_for_reach_and_must_be_known(patch):
    raw = yaml.safe_load((REPO / "config" / "episodes" / "grasp_b_right.yaml").read_text())
    raw = copy.deepcopy(raw)
    for name, kv in patch.items():
        raw["trajectories"][name].update(kv)
    with pytest.raises(S.EpisodeSpecError, match="stop"):
        S.parse(raw)


# ---------------------------------------------------------------- 실행기 → 계획 도구 인자
def _plan_argv(tmp_path, episode: str) -> list:
    from policy_control.episode_ros import RosExecutor
    import time
    calls = []

    def run(argv):
        calls.append(argv)
        return 0, "tail"
    ep = S.load(REPO / "config" / "episodes" / f"{episode}.yaml")
    now = time.monotonic()
    frames = [((0.27, -0.13, 0.288), (1.0, 0.0, 0.0, 0.0), now - 0.05 * k) for k in range(8)]
    me = SimpleNamespace(_wait_pd_tracking=lambda side: True, _pd_view=lambda side: {}, run_dir=tmp_path, _run=run,
                         cancel=threading.Event(), ep=ep, _lock=threading.Lock(), _frames={"CUP": frames},
                         _live_object=lambda name, wait_s=2.0: None)
    me._still_object = lambda name, wait_s=0.2: RosExecutor._still_object(me, name, wait_s)
    r = RosExecutor._run_planned(me, SimpleNamespace(id="reach"), ep.trajectories["reach"], SimpleNamespace(hand=lambda s: "EMPTY"))
    assert r.status == "completed" and len(calls) == 2
    return calls[0]


@pytest.mark.parametrize("episode,stop", [("grasp_d_right", "pregrasp"), ("grasp_b_right", "grasp")])
def test_the_runner_passes_the_reach_stop_to_the_planner(tmp_path, episode, stop):
    argv = _plan_argv(tmp_path, episode)
    assert Path(argv[1]).name == "plan_reach.py" and argv[argv.index("--stop") + 1] == stop


# ---------------------------------------------------------------- 미션
def test_the_mission_starts_rcg03_itself_after_the_reach():
    from policy_control import episode_runner as RUN
    m = yaml.safe_load((REPO / "config" / "mission_rh56f1_control.yaml").read_text())
    text = json.dumps(m, ensure_ascii=False)
    assert "handoff_contract:=" not in text and "approach_grasp:=handoff" not in text
    for side in ("right", "left"):
        ep = S.load(REPO / "config" / "episodes" / f"grasp_d_{side}.yaml")
        assert RUN._node_max_s(next(iter(ep.policies.values()))) == 15.0
        node = next(c["argv"] for c in m["run"][f"episode_grasp_d_{side}"] if "rh_aglt_node.py" in " ".join(c.get("argv", [])))
        assert f"contract:={{repo}}/deploy/policies/rh56f1/aglt/{side}_rcg03/rh_aglt_contract.json" in node
        assert "max_episode_s:=15.0" in node and "goal_max:=1" in node and "freeze_rule:=lift_hold" in node


# ---------------------------------------------------------------- 계획 도구(오프라인 — 홈 출발, 10.20 실기 경로 D 컵 자리)
@needs_hdgp
@pytest.mark.parametrize("side", ["right", "left"])
def test_pregrasp_reach_ends_where_rcg03_starts(tmp_path, side):
    R = _tool("plan_reach")
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK
    obj, real, cup = CUPS[side]
    ok, meta = R.plan(side, np.array(cup), np.array([1.0, 0, 0, 0]), obj, tmp_path, {s: HOME[s] for s in ("right", "left")},
                      H.hdgp_dir(), stop="pregrasp")
    assert ok, meta["problems"]
    a = meta["approach"]
    assert meta["stop"] == "pregrasp" and meta["deep_in"].get("skipped") and "q_grasp" not in meta
    q = np.load(tmp_path / "reach_cmd.npz")["arm_target"]
    assert len(q) == a["steps"] + 1, "들어가기 없이 이송 끝(사전 파지)에서 끝난다"
    assert np.abs(q[0] - HOME[side]).max() < 1e-3 and np.abs(np.diff(q, axis=0)).max() * 60 <= 0.72 + 1e-3
    assert a["collision"]["ok"] and a["margin_min_deg"] >= 5.0 and a["cup_clear_mm"] >= 10.0
    c = A.select_object(A.load_contract(REPO / "deploy" / "policies" / f"rh56f1/aglt/{side}_rcg03" / "rh_aglt_contract.json"), real)
    s = c.side()
    fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
    palm = np.asarray(fk.palm_pose(q[-1], np.zeros(6)).palm_pos, float)
    rr, out_h, out_r = A.ready_pose_error(c, cup, palm)
    pn = __import__("policy_control.pour_fj_node", fromlist=["READY_POSE_TOL_M"])
    assert out_r <= pn.READY_POSE_TOL_M and out_h <= pn.READY_POSE_TOL_M, (rr, out_h, out_r)
    assert abs(rr - 2.5) < 0.2 and a["h_m"] in R.PRE_H_M
    if side == "right":                      # 오른손은 높이 0 · 회귀 방위 그대로 = sim 사전 파지 IK(홈 가지)와 같은 자세
        assert a["h_m"] == 0.0 and a["phi_offset_deg"] == 0 and a["method"] == "ready_carry"
        sim_q = [-0.209, 0.182, 0.01, 0.684, -0.044, 0.223, 1.115]   # hdgp ready_approach plan_posture, q_ref = 홈(10.20 CPU 재현)
        assert np.abs(q[-1] - sim_q).max() < 0.06

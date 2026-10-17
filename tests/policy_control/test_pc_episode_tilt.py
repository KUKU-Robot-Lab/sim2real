"""10.17 tilt 기본 궤적 경로 A(docs/PLAN_GRASP_TILT_EVAL_2026-10-17.md): 에피소드 형식 carry · tilt, 실행기의 거부 조건,
오프라인 계획 도구(plan_carry · plan_tilt, hdgp md5 고정)."""
from __future__ import annotations

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
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_runner import SUCCESS, EpisodeManager

REPO = Path(__file__).resolve().parents[2]
EPISODE = REPO / "config" / "episodes" / "tilt_base_a.yaml"
ASSET = REPO / "deploy" / "policy_control" / "config" / "tilt" / "base_arc6"
TOOLS = REPO / "deploy" / "policy_control" / "tools"
# 붓기 시작 자세(rh56f1_pour_start.yaml) = D4 기준 출발
Q_START = yaml.safe_load((REPO / "deploy/policy_control/config/homes/rh56f1_pour_start.yaml").read_text())


def _raw(**traj):
    d = yaml.safe_load(EPISODE.read_text(encoding="utf-8"))
    d["trajectories"]["tilt"] = {**d["trajectories"]["tilt"], **traj}
    return d


def test_the_episode_parses_with_two_arm_carry_and_tilt():
    ep = S.load(EPISODE)
    t, c = ep.trajectories["tilt"], ep.trajectories["to_tilt_start"]
    assert (t.kind, t.sides, t.asset, t.rate_scale) == ("tilt", ("right", "left"), "tilt/base_arc6", 1.0)
    assert (c.kind, c.asset) == ("carry", "tilt/base_arc6")
    assert [n.id for n in ep.nodes] == ["go_home_start", "scene", "grasp_right", "grasp_left", "to_tilt_start", "tilt", "done"]


@pytest.mark.parametrize("bad", [{"sides": ["right"]}, {"asset": "tilt/nope"}, {"asset": ""}, {"rate_scale": 0.0},
                                 {"rate_scale": 1.5}])
def test_bad_tilt_trajectories_are_refused(bad):
    with pytest.raises(S.EpisodeSpecError):
        S.parse(_raw(**bad), path=str(EPISODE))


def test_asset_is_only_for_planned_kinds():
    d = yaml.safe_load(EPISODE.read_text(encoding="utf-8"))
    d["trajectories"]["go_home"]["asset"] = "tilt/base_arc6"
    with pytest.raises(S.EpisodeSpecError):
        S.parse(d, path=str(EPISODE))


def test_the_fake_run_ends_at_the_setting_pose_with_both_cups_in_hand():
    ep = S.load(EPISODE)
    ex = FakeExecutor(ep)
    m = EpisodeManager(ep, ex, approve=lambda what, why: True, episode_id="t")
    m.run()
    assert m.status == SUCCESS and m.world.hand("right") == "SRC" and m.world.hand("left") == "RCV"
    assert [c[2] for c in ex.calls if c[0] == "trajectory"] == ["go_home", "to_tilt_start", "tilt"]


def _ros_self(tmp_path, holds: dict, rcs: list[int]):
    """RosExecutor._run_planned 를 ROS 없이 — pd 는 TRACKING, 도구 실행은 rcs 순서로 돌려준다."""
    from policy_control.episode_ros import RosExecutor
    calls = []

    def run(argv):
        calls.append(argv)
        return rcs[len(calls) - 1], "tail"
    me = SimpleNamespace(_wait_pd_tracking=lambda side: True, _pd_view=lambda side: {}, run_dir=tmp_path, _run=run,
                         cancel=threading.Event())
    world = SimpleNamespace(hand=lambda side: holds[side])
    traj = S.load(EPISODE).trajectories["tilt"]
    return (lambda: RosExecutor._run_planned(me, SimpleNamespace(id="tilt"), traj, world)), calls


def test_planned_trajectory_refuses_an_empty_hand_without_moving(tmp_path):
    go, calls = _ros_self(tmp_path, {"right": "SRC", "left": "EMPTY"}, [])
    r = go()
    assert r.status == "refused" and calls == []


def test_a_failed_plan_check_refuses_and_never_replays(tmp_path):
    go, calls = _ros_self(tmp_path, {"right": "SRC", "left": "RCV"}, [2])
    r = go()
    assert r.status == "refused" and len(calls) == 1 and Path(calls[0][1]).name == "plan_tilt.py"


def test_a_good_plan_replays_fourteen_joints_with_the_start_gap_guard(tmp_path):
    go, calls = _ros_self(tmp_path, {"right": "SRC", "left": "RCV"}, [0, 0])
    r = go()
    assert r.status == "completed" and len(calls) == 2
    rp = calls[1]
    assert Path(rp[1]).name == "replay_to_pd.py" and rp[rp.index("--npz") + 1].endswith("tilt_cmd.npz")
    assert len(rp[rp.index("--joints") + 1].split(",")) == 14
    assert float(rp[rp.index("--max-start-gap") + 1]) == 0.05 and "--execute" in rp


def test_pins_refuse_a_changed_hdgp_file(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    H.verify(tmp_path, {"a.py": H.md5(tmp_path / "a.py")})
    with pytest.raises(H.PinError):
        H.verify(tmp_path, {"a.py": "0" * 32})
    with pytest.raises(H.PinError):
        H.verify(tmp_path, {"missing.py": "0" * 32})


def _tool(name):
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
        H.verify_refs(ASSET, cfg["ref"])
        return True
    except H.PinError:
        return False


needs_hdgp = pytest.mark.skipif(not _pins_hold(), reason="hdgp 핀 · 기준 파일이 이 호스트에서 맞지 않는다")


@needs_hdgp
def test_plan_tilt_from_the_setting_pose_passes_every_check(tmp_path):
    (tmp_path / "q.json").write_text(json.dumps({"right": Q_START["right"], "left": Q_START["left"]}))
    rc = _tool("plan_tilt").main(["--q-json", str(tmp_path / "q.json"), "--out", str(tmp_path / "o")])
    chk = json.loads((tmp_path / "o" / "check.json").read_text())
    assert rc == 0 and chk["ok"] and chk["checks"]["rcv_cup_table_clear_m"]["value"] >= 0.01
    z = np.load(tmp_path / "o" / "tilt_cmd.npz")
    assert z["arm_target"].shape[1] == 14 and np.abs(z["arm_target"][0] - (Q_START["right"] + Q_START["left"])).max() < 1e-3


@needs_hdgp
def test_plan_carry_reaches_the_setting_pose_exactly_from_a_lift_pose(tmp_path):
    q = {"right": [-0.5814, 0.2908, -0.0098, 1.6084, 0.1172, 0.1432, 0.6816],
         "left": [0.5879, -0.4043, -0.0718, 1.7017, -0.0581, -0.2657, -0.5506]}
    q = {s: (np.asarray(v) + np.array([0.08, -0.05, 0.04, 0.12, 0.0, -0.06, 0.03])).tolist() for s, v in q.items()}
    (tmp_path / "q.json").write_text(json.dumps(q))
    rc = _tool("plan_carry").main(["--q-json", str(tmp_path / "q.json"), "--out", str(tmp_path / "o")])
    chk = json.loads((tmp_path / "o" / "check.json").read_text())
    assert rc == 0 and chk["ok"], chk["checks"]
    z = np.load(tmp_path / "o" / "carry_cmd.npz")["arm_target"]
    assert np.abs(z[0] - (q["right"] + q["left"])).max() < 1e-5
    assert np.abs(z[-1] - (Q_START["right"] + Q_START["left"])).max() < 0.05

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


def test_the_episode_parses_with_one_arm_carries_and_a_two_arm_tilt():
    ep = S.load(EPISODE)
    t = ep.trajectories["tilt"]
    assert (t.kind, t.sides, t.asset, t.rate_scale) == ("tilt", ("right", "left"), "tilt/base_arc6", 1.0)
    # 계획 경로 A "ready_carry(한 팔씩)" — 오른팔 → 왼팔
    assert [(ep.trajectories[k].kind, ep.trajectories[k].sides, ep.trajectories[k].cups)
            for k in ("to_tilt_start_r", "to_tilt_start_l")] == [("carry", ("right",), True), ("carry", ("left",), True)]
    assert [n.id for n in ep.nodes] == ["go_home_start", "scene", "grasp_right", "grasp_left", "to_tilt_start_r",
                                        "to_tilt_start_l", "tilt", "done"]


@pytest.mark.parametrize("bad", [{"sides": ["right"]}, {"asset": "tilt/nope"}, {"asset": ""}, {"rate_scale": 0.0},
                                 {"rate_scale": 1.5}, {"cups": False}, {"cups": "yes"}])
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
    assert [c[2] for c in ex.calls if c[0] == "trajectory"] == ["go_home", "to_tilt_start_r", "to_tilt_start_l", "tilt"]


def _ros_self(tmp_path, holds: dict, rcs: list[int]):
    """RosExecutor._run_planned 를 ROS 없이 — pd 는 TRACKING, 도구 실행은 rcs 순서로 돌려준다."""
    from policy_control.episode_ros import RosExecutor
    calls = []

    def run(argv):
        calls.append(argv)
        return rcs[len(calls) - 1], "tail"
    me = SimpleNamespace(_tin_rec=None, ep=SimpleNamespace(tin_fallback=""), _wait_pd_tracking=lambda side: True, _wait_arm_still=lambda sides: {"arm_still": None}, _pd_view=lambda side: {}, run_dir=tmp_path, _run=run,
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


@needs_hdgp
def test_plan_carry_empty_hand_gets_no_virtual_cup(tmp_path):
    # ★11.15 실기 칵테일: 왼손 shaker 이송 계획 때 오른손은 빈손으로 파랑에 접근 중 — 가상 컵 때문에 2.97 cm 로 거부됐다
    q = {"right": [-0.15736, 0.29088, -0.44308, 1.17247, 0.02728, 0.21839, 0.92756],
         "left": [0.416, -0.05818, -0.06504, 1.21633, 0.00858, -0.20161, -0.88941]}
    (tmp_path / "q.json").write_text(json.dumps(q))
    base = ["--q-json", str(tmp_path / "q.json"), "--sides", "left"]
    assert _tool("plan_carry").main(base + ["--out", str(tmp_path / "a")]) == 2
    assert json.loads((tmp_path / "a" / "check.json").read_text())["arms_clear"]["min_cup_gap_m"] < 0.03
    assert _tool("plan_carry").main(base + ["--out", str(tmp_path / "b"), "--empty", "right"]) == 0
    arms = json.loads((tmp_path / "b" / "check.json").read_text())["arms_clear"]
    assert arms["ok"] and arms["min_cup_gap_m"] is None and arms["min_arm_point_m"] >= 0.10
    assert arms["min_cup_arm_m"] >= 0.05          # 든 컵 ↔ 빈손 팔은 여전히 본다(실제 12.9 cm)


@needs_hdgp
def test_plan_carry_one_arm_writes_only_that_arm_and_home_start_reaches_the_setting(tmp_path):
    # 경로 B: 빈손으로 홈(rh_aglt 시작 자세)에서 붓기 시작 자세까지 — 한 팔씩
    home = yaml.safe_load((REPO / "deploy/policy_control/config/homes/rh56f1_aglt.yaml").read_text())
    q = {s: [float(v) for v in home[s]] for s in ("right", "left")}
    (tmp_path / "q.json").write_text(json.dumps(q))
    rc = _tool("plan_carry").main(["--q-json", str(tmp_path / "q.json"), "--out", str(tmp_path / "o"), "--sides", "left"])
    chk = json.loads((tmp_path / "o" / "check.json").read_text())
    assert rc == 0 and chk["ok"] and chk["sides"] == ["left"] and list(chk["checks"]) == ["left"]
    assert chk["arms_clear"]["ok"] and chk["arms_clear"]["min_cup_gap_m"] >= 0.03     # 오른팔은 홈에 선 채
    assert "window" not in chk["arms_clear"]      # ★11.18 한 팔씩이면 다른 팔은 서 있다 — ±창 검사 없음(옛 결과 그대로)
    z = np.load(tmp_path / "o" / "carry_cmd.npz")
    assert z["arm_target"].shape[1] == 7 and [str(n) for n in z["joint_names"]] == [f"l_aj_{i}" for i in range(1, 8)]
    assert np.abs(z["arm_target"][-1] - np.asarray(Q_START["left"])).max() < 0.05
    assert _tool("plan_carry").main(["--q-json", str(tmp_path / "q.json"), "--out", str(tmp_path / "p"), "--sides", "up"]) == 1


@needs_hdgp
def test_plan_carry_both_arms_from_the_141213_grasp_ends_passes_the_same_time_and_window_checks(tmp_path):
    # ★11.18 실기 141213(cocktail_1c_e4000) 두 팔 잡기 끝 실측 — 함께 계획(--sides right,left)이 같은 시각 · ±0.5 s 창을 다 통과한다
    q = {"right": [-0.036049, 0.645266, -0.411421, 0.881399, -0.116159, 0.569352, 1.261349],
         "left": [0.517853, -0.061227, -0.047875, 1.176661, 0.053216, -0.205425, -1.070230]}
    (tmp_path / "q.json").write_text(json.dumps(q))
    rc = _tool("plan_carry").main(["--q-json", str(tmp_path / "q.json"), "--out", str(tmp_path / "o"), "--sides", "right,left"])
    chk = json.loads((tmp_path / "o" / "check.json").read_text())
    arms = chk["arms_clear"]
    assert rc == 0 and chk["ok"] and arms["ok"] and arms["empty"] == []
    assert arms["window"]["lag_s"] == 0.5 and arms["window"]["ok"]
    for k, least in (("min_arm_point_m", 0.10), ("min_cup_gap_m", 0.03)):   # 창을 넓힐수록 최소 거리는 줄기만 한다
        assert arms[k] >= arms["window"][k] >= least        # 판정: 같은 시각 · ±창 둘 다 문턱 위
        assert arms["any_time"][k] <= arms["window"][k]     # any_time 은 기록만 — 문턱과 견주지 않는다
    assert np.load(tmp_path / "o" / "carry_cmd.npz")["arm_target"].shape[1] == 14


def test_arms_clear_refuses_a_pass_that_only_clears_at_the_same_time(monkeypatch):
    """★11.18 함께 계획 판정(hdgp 없이 합성 점) — 같은 시각 검사는 통과해도 ±창 안에서 걸리면 ok 가 False 다."""
    import torch
    pc = _tool("plan_carry")
    t = torch.linspace(0.0, 1.0, 61)[:, None, None]             # 60 Hz 1 s, 팔마다 점 하나 (N,1,3)
    x = torch.tensor([1.0, 0.0, 0.0])
    arm = {"right": t * x, "left": (t - 0.2) * x}                # 왼팔이 오른팔이 지난 자리를 0.2 s 뒤에 지난다
    cup = {"right": t * x + 5.0, "left": t * x - 5.0}            # 컵은 늘 멀다 — 걸리는 것은 팔 점뿐
    monkeypatch.setattr(pc, "arm_points", lambda ch, path: path)
    monkeypatch.setattr(pc, "cup_points", lambda P, ch, path, T_to, side: cup[side])
    none = dict.fromkeys(pc.ARMS)
    same = pc.arms_clear(None, none, arm, none)
    out = pc.arms_clear(None, none, arm, none, lag=round(pc.SYNC_WINDOW_S * pc.RATE_HZ))
    assert same["ok"] and "window" not in same                   # 같은 시각만 보면 통과(0.2 m)
    assert out["min_arm_point_m"] == same["min_arm_point_m"] >= pc.ARM_POINT_GAP_M
    assert out["window"]["min_arm_point_m"] < pc.ARM_POINT_GAP_M and out["window"]["min_cup_gap_m"] >= pc.CUP_GAP_M
    assert out["window"]["ok"] is False and out["ok"] is False   # ±창에서 팔 점만 걸렸다 — 함께 계획 거부


def test_arm_still_needs_a_full_quiet_window():
    """★10.30 계획 전 팔 정지 판정 — 창(0.3 s)을 덮는 실측이 문턱(0.003 rad) 안에 머물러야 멈춤."""
    from policy_control.episode_ros import ARM_STILL_RAD, arm_still
    j = [f"r_aj_{i}" for i in range(1, 8)]
    still = [(0.01 * k, {n: 0.5 + 0.0005 * (k % 2) for n in j}) for k in range(40)]
    assert arm_still(still, j)[0]
    assert arm_still(still[:10], j) == (False, None)                              # 창보다 짧다
    moving = [(0.01 * k, {**{n: 0.5 for n in j}, "r_aj_6": 0.5 + 0.02 * 0.01 * k}) for k in range(40)]   # 0.02 rad/s
    ok, span = arm_still(moving, j)
    assert not ok and span > ARM_STILL_RAD
    assert arm_still(still, j + ["l_aj_1"]) == (False, None)                      # 다른 팔 실측이 없다


def test_planned_trajectory_waits_for_the_arm_before_planning(tmp_path):
    """계획 도구보다 정지 대기가 먼저 — 그 노드 팔(sides)로 기다린다."""
    from policy_control.episode_ros import RosExecutor
    order = []

    def run(argv):
        order.append(Path(argv[1]).name)
        return 0, "tail"
    me = SimpleNamespace(_tin_rec=None, ep=SimpleNamespace(tin_fallback=""), _wait_pd_tracking=lambda side: True, _pd_view=lambda side: {}, run_dir=tmp_path, _run=run,
                         cancel=threading.Event(), _wait_arm_still=lambda sides: order.append(("still", tuple(sides))) or {"arm_still": True})
    traj = S.load(EPISODE).trajectories["tilt"]
    r = RosExecutor._run_planned(me, SimpleNamespace(id="tilt"), traj, SimpleNamespace(hand=lambda side: "SRC"))
    assert r.status == "completed"
    assert order[0] == ("still", tuple(s for s in ("right", "left") if s in traj.sides)) and len(order) == 3

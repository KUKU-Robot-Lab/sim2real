"""★10.19 사용자 "좌우 동시 실행 · control 탭 tilt 버튼" — 팔마다 에피소드 실행기 · 정책 카드끼리 목 · 인지 같이 쓰기 ·
한 팔 멈춤은 그 팔 실행기만 · tilt_now(세팅 자세로) 는 rehome 처럼 · tilt_pour 는 두 팔 pd 가 붙든(TRACKING) 때만."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import yaml

from s2r_console.actions import _Job, runner_of
from test_console_actions import rg3_workspace, setup_actions


# ── 콘솔: 실행기 고르기 ─────────────────────────────────────────────────
def _says(console, runner, *, age_s=0.0, **status):
    from s2r_console.feed import line

    s = console.session
    body = {"node": runner, "phase": "READY", "ok": True, "reasons": [], "episode": "pick_place_right",
            "next_action": "go_home_start", "busy": False, **status}
    with s.feed_lock:
        s.feed.ingest(line("hello", domain=98, nodes=["obs", "pd", runner]))
        s.feed.ingest(line("status", node=runner, data=body))
        s.feed._seen_at[runner] -= age_s


@pytest.fixture()
def episode_console(console, monkeypatch, tmp_path):
    from s2r_console import console as mod

    noop = tmp_path / "noop_episode_cmd.py"
    noop.write_text("import sys; sys.exit(0)\n")
    monkeypatch.setattr(mod, "_EPISODE_CMD", noop)
    console.open("t_fake", operator="pytest")
    return console


def test_runner_views_and_primary_are_pure():
    from s2r_console.console import RUNNER_NODES, primary_runner, runner_views

    assert RUNNER_NODES == ("episode_runner", "episode_runner_right", "episode_runner_left")
    obs = SimpleNamespace(status={"episode_runner_left": {"busy": True}, "episode_runner_right": {"busy": False},
                                  "episode_runner": {"busy": False}},
                          age_s={"episode_runner_left": 0.1, "episode_runner_right": 0.1, "episode_runner": 99.0})
    views = runner_views(obs)
    assert list(views) == list(RUNNER_NODES) and views["episode_runner"]["stale"]
    assert primary_runner(views) == "episode_runner_left"           # 신선 · 실행 중이 먼저
    views["episode_runner_left"]["busy"] = False
    assert primary_runner(views) == "episode_runner_right"          # 그다음 기본 → 오른팔 → 왼팔(기본은 늦다)
    assert primary_runner({}) is None


def test_bridge_listens_to_every_runner():
    from s2r_console import console as mod

    prof = SimpleNamespace(domain=97, status_nodes=("pd_right",), stack=None, diagram=None)
    argv = mod.bridge_argv(prof, None)
    nodes = argv[argv.index("--nodes") + 1:]
    assert {"episode_runner", "episode_runner_right", "episode_runner_left"} <= set(nodes)


def test_episode_commands_go_to_the_chosen_runner(episode_console):
    from s2r_console.console import ConsoleError

    _says(episode_console, "episode_runner_right")
    _says(episode_console, "episode_runner_left", episode="pick_place_left", next_action="go_home_start")   # rg3 는 FP++ 검사가 따로
    with pytest.raises(ConsoleError, match="모르는 에피소드 실행기"):
        episode_console.episode("next", operator="op", typed="go_home_start", runner="episode_runner_both")
    episode_console.episode("run", operator="op", typed="episode:pick_place_left", runner="episode_runner_left")
    procs = {p["key"]: p for p in episode_console.snapshot()["session"]["procs"]}
    argv = procs["episode#run#episode_runner_left"]["argv"]
    assert argv[argv.index("--runner") + 1] == "episode_runner_left"
    assert argv[argv.index("--approve") + 1] == "episode:pick_place_left"
    session = episode_console.snapshot()["session"]
    assert [r["runner"] for r in session["episode_runners"]] == ["episode_runner_right", "episode_runner_left"]


def test_stop_without_a_runner_stops_every_live_runner(episode_console):
    _says(episode_console, "episode_runner_right", busy=True)
    _says(episode_console, "episode_runner_left", busy=True)
    episode_console.episode("stop", operator="op")
    keys = {p["key"] for p in episode_console.snapshot()["session"]["procs"]}
    assert {"episode#stop#episode_runner_right", "episode#stop#episode_runner_left"} <= keys
    episode_console.quick("episode_stop", client="pytest")          # 정지 바도 실행기 전부
    keys = {p["key"] for p in episode_console.snapshot()["session"]["procs"]}
    assert {"episode#stop#episode_runner_right", "episode#stop#episode_runner_left"} <= keys


# ── 관리 작업: 자원 · 정지 · tilt_now ────────────────────────────────────
def _obs(monkeypatch, console, *, right="TRACKING", left="TRACKING", runners=()):
    status = {"pd_right": {"execute": True, "phase": right}, "pd_left": {"execute": True, "phase": left},
              **{r: {"episode": r, "phase": "RUNNING"} for r in runners}}
    obs = SimpleNamespace(status=status, age_s={k: 0 for k in status}, joints_age_s=0, joint_ages={},
                          rosgraph=None, rosgraph_age_s=None)
    monkeypatch.setattr(console.session.feed, "observed", lambda: obs)
    return obs


def test_policy_cards_on_different_arms_share_head_and_perception(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {"episode_grasp_d_right": [{"argv": ["true"]}],
                                           "episode_grasp_d_left": [{"argv": ["true"]}]})
    rg3_workspace(console)
    _obs(monkeypatch, console)
    left = _Job("rg3_left_card", "policy", "op", {"left", "head", "perception"})
    m._jobs[left.id] = left
    item = m.describe("rg3_chosen")                                   # 오른팔 카드
    assert not [r for r in item["reasons"] if "실행 중" in r], item["reasons"]
    same = _Job("rg3_other_right", "policy", "op", {"right", "head", "perception"})
    m._jobs[same.id] = same
    reasons = m.describe("rg3_chosen")["reasons"]
    assert any("rg3_other_right 실행 중 — 같은 대상(right)" in r for r in reasons), reasons


def test_one_arm_preempt_stops_only_that_arms_runner(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {"preflight": [{"argv": ["true"]}]})
    s = console.session
    for side in ("right", "left"):
        s.supervisor.spawn(f"episode_grasp_d_{side}#5", stage=f"episode_grasp_d_{side}", note="runner",
                           argv=["sleep", "30"], background=True, manual=False)
    _obs(monkeypatch, console, runners=(runner_of("right"), runner_of("left")))
    monkeypatch.setattr(console, "_pd_sides", lambda *_: ["right", "left"])
    calls = []
    monkeypatch.setattr(m, "_trigger", lambda _, service, side, **kw: calls.append((service, side)))
    monkeypatch.setattr(m, "_confirm_hold", lambda *args: None)
    monkeypatch.setattr(m, "_run_commands", lambda _s, sid, cmds: calls.append((sid, list(cmds[0].argv))))
    m._preempt(s, {"right"})
    stops = [c for c in calls if str(c[0]).startswith("action_episode_stop")]
    assert [c[0] for c in stops] == ["action_episode_stop_right"]
    assert stops[0][1][stops[0][1].index("--runner") + 1] == "episode_runner_right"
    assert ("episode/stop", "right") in calls and ("episode/stop", "left") not in calls
    assert not s.supervisor.is_alive("episode_grasp_d_right#5")
    assert s.supervisor.is_alive("episode_grasp_d_left#5")
    s.supervisor.stop(["episode_grasp_d_left#5"])


def _wait(m):
    end = time.monotonic() + 5
    while m.view().get("active") and time.monotonic() < end:
        time.sleep(.01)
    return m.view()


def test_tilt_pour_is_a_two_arm_control_that_needs_both_pds_holding(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {"tilt_pour": [{"argv": ["true"], "manual": True}, {"argv": ["true"]}]})
    _obs(monkeypatch, console, left="IDLE")
    item = next(a for a in m.catalog() if a["id"] == "tilt_pour")
    assert item["group"] == "control" and not item["enabled"]
    assert any("left PD 상태 IDLE" in r and "TRACKING" in r for r in item["reasons"]), item["reasons"]
    assert m._resources(console.session, "tilt_pour", "control", {}) == frozenset({"right", "left"})
    _obs(monkeypatch, console)
    item = m.describe("tilt_pour")
    assert item["enabled"], item["reasons"]
    m.execute("tilt_pour", operator="test")
    assert _wait(m)["outcome"] == "DONE"


def test_tilt_now_goes_to_the_setting_pose_like_rehome(console, tiny_repo, monkeypatch):
    """세팅 자세로 가는 것 — rehome 처럼 pd 가 IDLE 이어도 된다(그 자리에서 engage), 두 팔을 다 쓴다."""
    m = setup_actions(console, tiny_repo, {"tilt_now": [{"argv": ["true"], "manual": True}, {"argv": ["true"]}]})
    _obs(monkeypatch, console, right="IDLE", left="IDLE")
    item = next(a for a in m.catalog() if a["id"] == "tilt_now")
    assert item["group"] == "control" and item["enabled"], item["reasons"]
    assert "세팅 자세" in item["title"]
    assert m._resources(console.session, "tilt_now", "control", {}) == frozenset({"right", "left"})
    _obs(monkeypatch, console, left="HOLD")
    assert not m.describe("tilt_now")["enabled"]
    _obs(monkeypatch, console, left="IDLE")
    m.execute("tilt_now", operator="test")
    assert _wait(m)["outcome"] == "DONE"


def test_rh56f1_mission_runs_one_arm_episodes_on_their_own_runner():
    import yaml
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    m = yaml.safe_load((repo / "config/mission_rh56f1_control.yaml").read_text())
    for side in ("right", "left"):
        cmds = m["run"][f"episode_grasp_d_{side}"]
        runner = next(c["argv"] for c in cmds if any("episode_runner_node.py" in a for a in c.get("argv", [])))
        assert f"__node:=episode_runner_{side}" in runner
        check = next(c["argv"][-1] for c in cmds if "ros2 node list" in " ".join(c.get("argv", [])))
        assert f"episode_runner_{side}" in check and "|episode_runner)" not in check
        rec = next(c["argv"][-1] for c in cmds if "rh56f1_record.sh" in " ".join(c.get("argv", [])) and "start" in " ".join(c["argv"]))
        assert f"/policy_control/status/episode_runner_{side}" in rec and f"/episode/{side}/objects/CUP/pose" in rec
        assert f"HAND_SIDES='{side}'" in rec
    tilt = m["run"]["episode_tilt_base_b"]
    runner = next(c["argv"] for c in tilt if any("episode_runner_node.py" in a for a in c.get("argv", [])))
    assert not any(a.startswith("__node:=") for a in runner)          # 두 팔 에피소드는 기본 실행기
    now = m["run"]["tilt_now"]                                       # 세팅 자세로 — 확인 → 두 팔 engage(IDLE 이면) → 팔마다 계획 · 재생
    assert now[0].get("manual") and [" ".join(c["argv"]) for c in now[1:3]] == [
        f"python3 {{repo}}/deploy/policy_control/tools/episode_ctl.py --side {s} --only pd_engage --skip-engaged --hold-s 2"
        for s in ("right", "left")]
    assert "tilt_now.py" in " ".join(now[3]["argv"]) and now[3]["execute_args"] == ["--execute"]
    pour = m["run"]["tilt_pour"]
    assert pour[0].get("manual") and "tilt_pour.py" in " ".join(pour[1]["argv"]) and pour[1]["execute_args"] == ["--execute"]
    for sid in ("tilt_now", "tilt_pour"):
        stage = next(s for s in m["stages"] if s["id"] == sid)
        assert stage["lane"] == "both" and set(stage["needs"]) == {"pd_arm_right", "pd_arm_left"}


# ── 정책 추가: 작업별 정책만 ────────────────────────────────────────────
def test_policy_task_of():
    from s2r_console.workspace import task_of

    assert task_of("rh56f1/aglt/right_approach_rg3wg") == "approaching"
    assert task_of("rh56f1/aglt/left_neo03") == "grasping"
    assert task_of("rh56f1/aglt/right_rg3wo") == "grasping"
    assert task_of("rh56f1/place/right_i09") == "placing"
    assert task_of("rh56f1/pour_fj/both_f01") == "pouring"
    assert task_of("dg5f_m/pour_fab/both_i24") == "pouring"
    assert task_of("dg5f_m/cup_grasp/left_i14") == "grasping"


def test_choose_policy_lists_only_that_task():
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "deploy/s2r_console/s2r_console/web/console.js").read_text()
    body = js[js.index("function choosePolicy(task)"):js.index("let videoAttached")]
    assert ".filter(p => p.task === task)" in body


def test_approach_policies_are_runnable_as_path_c_cards():
    """★10.19 사용자: 경로 C 접근 정책 카드에 실행 버튼이 없었다(배치만 지원) — 접근 카드는 grasp_c 에피소드로 돈다.
    ★10.20 지금 접근 정책(ap05)은 사전 파지에서 멈춘다 — 경로 C 에피소드와 이음 자세가 달라 policy.yaml 이 실행을 막고 이유를 보인다."""
    from pathlib import Path

    from s2r_console.actions import card_stage
    from s2r_console.workspace import Workspace

    repo = Path(__file__).resolve().parents[2]
    w = Workspace.__new__(Workspace)
    w.repo, w.robot_id = repo, "rh56f1"
    approach = [p for p in w._catalog({}) if p["task"] == "approaching"]
    assert {p["side"] for p in approach} == {"right", "left"}
    for p in approach:
        gated = (yaml.safe_load(Path(p["file"]).read_text()).get("deploy") or {}).get("execution") is False
        assert p["execution_supported"] == (not gated)
        assert bool(p["execution_reason"]) == gated and ("사전 파지" in p["execution_reason"] if gated else True)
    assert card_stage({"task": "approaching", "side": "left"}) == "episode_grasp_c_left"
    assert card_stage({"task": "grasping", "side": "right"}) == "episode_grasp_d_right"   # ★10.20 경로 A 폐지 → D
    js = (repo / "deploy/s2r_console/s2r_console/web/console.js").read_text()
    assert "경로 C 실행 연결" in js and "function scriptGrip(a)" in js

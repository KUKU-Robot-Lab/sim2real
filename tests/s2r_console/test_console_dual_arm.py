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


def test_place_fixed_is_a_one_arm_control_that_needs_that_pd_holding(console, tiny_repo, monkeypatch):
    """★11.18 고정 놓기 — id 끝 팔(left)의 pd 만 보고 자원으로 잡는다 · 컵을 쥔 채라 TRACKING 이어야."""
    sid = "place_fixed_shaker_all_left"
    m = setup_actions(console, tiny_repo, {sid: [{"argv": ["true"], "manual": True}, {"argv": ["true"]}]})
    _obs(monkeypatch, console, right="IDLE", left="IDLE")
    item = next(a for a in m.catalog() if a["id"] == sid)
    assert item["group"] == "control" and not item["enabled"]
    assert any("left PD 상태 IDLE" in r and "TRACKING" in r for r in item["reasons"]), item["reasons"]
    assert not any(r.startswith("right") for r in item["reasons"]), item["reasons"]
    assert m._resources(console.session, sid, "control", {}) == frozenset({"left"})
    _obs(monkeypatch, console, right="IDLE")
    assert m.describe(sid)["enabled"]


def test_cap_student_buttons_and_policy_stop_reach_its_nodes(console, tiny_repo, monkeypatch):
    """★11.18 cap 학생 — control 탭 버튼 셋 · 학생 단계 노드는 정책 정지 대상(목 드라이버 제외) · stage/next 는 세우지 않는다."""
    from s2r_console.actions import _preempts
    caps = ("cap_now", "cap_student_right", "cap_next_right", "cap_end_right")
    m = setup_actions(console, tiny_repo, {sid: [{"argv": ["true"]}] for sid in caps})
    _obs(monkeypatch, console)
    items = {a["id"]: a for a in m.catalog()}
    assert all(items[sid]["group"] == "control" for sid in caps)
    s = console.session
    assert m._resources(s, "cap_now", "control", {}) == frozenset({"right", "left"})
    # 왼손이 쉐이커(리시버)를 쥔 채 — rehome_left 같은 왼팔 동작과 겹치지 않게 left 도 잡는다(11.18 리뷰)
    for sid in ("cap_student_right", "cap_next_right", "cap_end_right"):
        assert m._resources(s, sid, "control", {}) == frozenset({"right", "left", "head", "perception"}), sid
    assert m.describe("cap_student_right")["enabled"]
    _obs(monkeypatch, console, left="IDLE")         # 왼팔 pd 가 쉐이커 자세를 붙들고 있지 않다
    assert any("left PD 상태 IDLE" in r for r in m.describe("cap_student_right")["reasons"])
    _obs(monkeypatch, console, right="HOLD")        # 정리(cap_end)는 pd 가 HOLD 여도 돈다
    assert m.describe("cap_end_right")["enabled"] and not m.describe("cap_student_right")["enabled"]
    _obs(monkeypatch, console)
    assert _preempts("cap_student_right", "control") and not _preempts("cap_next_right", "control")
    # 단계 탭에서 cap 학생 단계가 실패 · 중단하면 정책 단계처럼 정리한다 — episode/stop 은 cap 노드 ns(right_cap)로
    from s2r_console.console import POLICY_STAGE_PREFIXES, failure_actions
    assert "cap_student_right".startswith(POLICY_STAGE_PREFIXES)
    assert failure_actions("cap_student_right", ["right"]) == [("episode/stop", "right_cap"), ("pd/hold", "right")]
    s.supervisor.spawn("cap_student_right#4", stage="cap_student_right", note="node", argv=["sleep", "30"],
                       background=True, manual=False)
    s.supervisor.spawn("cap_student_right#2", stage="cap_student_right", note="drv",
                       argv=["bash", "-c", "sleep 30", "head_joint_publisher.py"], background=True, manual=False)
    try:
        assert s.supervisor.is_alive("cap_student_right#2")
        assert m._policy_targets(s) == ["cap_student_right#4"]
    finally:
        s.supervisor.stop(["cap_student_right#4", "cap_student_right#2"])


def _cap(console, tiny_repo, monkeypatch, **commands):
    """★11.18 리뷰 — cap 단계 + 서비스 호출 기록(_trigger · 실패 정리 trigger.py)."""
    import subprocess

    from s2r_console import console as console_mod
    base = {sid: [{"argv": ["true"]}] for sid in ("cap_now", "cap_student_right", "cap_next_right", "cap_end_right",
                                                  "head_home", "tilt_now", "rehome_right", "shutdown")}
    m = setup_actions(console, tiny_repo, {**base, **commands})
    _obs(monkeypatch, console)
    calls = []
    monkeypatch.setattr(m, "_trigger", lambda s, svc, side, **kw: calls.append((svc, side)))
    monkeypatch.setattr(m, "_kill_stray_policies", lambda *a: [])
    real_run = subprocess.run

    def fake_run(argv, *a, **kw):           # 실패 정리(console._fail_safe)의 trigger.py 만 가로챈다
        if any(str(x).endswith("trigger.py") for x in argv):
            calls.append((argv[2], argv[4]))
            return subprocess.CompletedProcess(argv, 0, "ok\n", "")
        return real_run(argv, *a, **kw)
    monkeypatch.setattr(console_mod.subprocess, "run", fake_run)
    return m, calls


def test_a_failed_cap_student_stops_its_episode_and_background_and_points_to_cap_end(console, tiny_repo, monkeypatch):
    """★11.18 리뷰 H1: cap 학생 단계는 _run 으로 돈다 — 실패하면 episode/stop(right_cap) · 그 단계 배경 정지 · cap_end 안내."""
    m, calls = _cap(console, tiny_repo, monkeypatch, cap_student_right=[
        {"argv": ["bash", "-c", "sleep 30", "head_joint_publisher.py"], "background": True},
        {"argv": ["sleep", "30"], "background": True}, {"argv": ["false"]}])
    m.execute("cap_student_right", operator="test")
    v = _wait(m)
    s = console.session
    assert v["outcome"] == "FAILED" and v["note"].startswith("cap_student_right:"), v["note"]   # 원래 사유가 앞에
    assert "cap_end_right 로 기본 드라이버와 머리 자세를 되돌릴 것" in v["note"]
    assert ("episode/stop", "right_cap") in calls and ("pd/hold", "right") in calls
    assert not any(p.stage == "cap_student_right" for p in s.supervisor.alive())


@pytest.mark.parametrize("phase", ["HOLD", "stale"])
def test_cap_end_runs_even_when_the_pd_is_held_or_stale(console, tiny_repo, monkeypatch, phase):
    """★11.18 리뷰 H3: cap_end_right 는 강제 선점 — pd HOLD · stale 로 붙들기가 실패해도 경고로 두고 정지 · 복귀를 돈다."""
    from s2r_console.console import ConsoleError
    m, _ = _cap(console, tiny_repo, monkeypatch)
    s = console.session
    obs = _obs(monkeypatch, console, right="HOLD")
    if phase == "stale":
        obs.age_s["pd_right"] = 1e9

    def refuse(s_, svc, side, **kw):
        if svc == "pd/hold":
            raise ConsoleError(f"{side} pd {phase}")
    monkeypatch.setattr(m, "_trigger", refuse)
    s.supervisor.spawn("cap_student_right#1", stage="cap_student_right", note="node", argv=["sleep", "30"],
                       background=True, manual=False)
    assert m.describe("cap_end_right")["enabled"], m.describe("cap_end_right")["reasons"]
    m.execute("cap_end_right", operator="test")
    assert _wait(m)["outcome"] == "DONE", phase
    assert not s.supervisor.is_alive("cap_student_right#1")


def test_live_cap_background_blocks_other_actions_but_not_cap_stop_or_recovery(console, tiny_repo, monkeypatch):
    """★11.18 리뷰 M4: cap 학생 배경이 살아 있으면 cap · 정지 · shutdown · 복귀 말고는 cap_end_right 먼저."""
    m, _ = _cap(console, tiny_repo, monkeypatch)
    s = console.session
    assert m.describe("head_home")["enabled"] and m.describe("tilt_now")["enabled"]
    s.supervisor.spawn("cap_student_right#2", stage="cap_student_right", note="drv",
                       argv=["bash", "-c", "sleep 30", "head_joint_publisher.py"], background=True, manual=False)
    try:
        for sid in ("head_home", "tilt_now"):
            assert any("cap_end_right" in r for r in m.describe(sid)["reasons"]), sid
        for sid in ("cap_next_right", "cap_end_right", "rehome_right", "shutdown", "policy_stop"):
            assert not any("cap_end_right" in r for r in m.describe(sid)["reasons"]), sid
    finally:
        s.supervisor.stop(["cap_student_right#2"])
    assert m.describe("head_home")["enabled"]


def test_a_refused_cap_next_does_not_hold_the_arms(console, tiny_repo, monkeypatch):
    """★11.18 리뷰 M6: stage/next 거부(rc 1)는 실패로 보이되 pd/hold 를 보내지 않는다."""
    m, calls = _cap(console, tiny_repo, monkeypatch,
                    cap_next_right=[{"argv": ["bash", "-c", "echo 이송이 아직 안 끝났다; exit 1"]}])
    m.execute("cap_next_right", operator="test")
    v = _wait(m)
    assert v["outcome"] == "FAILED" and not [c for c in calls if c[0] == "pd/hold"], calls


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


def test_policy_card_placement_is_removed_from_the_screen():
    """★10.21 사용자 "카드 배치(←→↑↓)는 일단 제거" — 화면에 카드 추가 · 옮기기 · 지우기가 없다(실험 목록이 대신 실행한다)."""
    from pathlib import Path

    web = Path(__file__).resolve().parents[2] / "deploy/s2r_console/s2r_console/web"
    js, html = (web / "console.js").read_text(), (web / "index.html").read_text()
    for gone in ("policy-move", "policy-add-card", "policy-choose", "policy-reset", "function choosePolicy"):
        assert gone not in js
    assert 'id="policy-cards"' not in html and 'id="policy-add"' not in html


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
    # ★10.21 카드 배치를 화면에서 뺐다 — 경로 C 는 실험 목록의 고정 작업 ctl_grasp_c_<팔>, 스크립트 쥐기 단계는 그 실행 칸에 보인다
    assert "function scriptGrip(a)" in js and "actionState(a) + scriptGrip(a)" in js

"""상황판 에피소드 명령(10.04) — 실행기가 물을 이름을 그대로 입력해야 next · run 이 나가고, 정지는 언제나."""
from __future__ import annotations

import pytest


def _runner_says(console, *, age_s=0.0, **status):
    from s2r_console.feed import line

    s = console.session
    body = {"node": "episode_runner", "phase": "READY", "ok": True, "reasons": [], "episode": "pick_place_right",
            "next_action": "pick_cup", "busy": False, **status}
    with s.feed_lock:
        s.feed.ingest(line("hello", domain=98, nodes=["obs", "pd", "episode_runner"]))
        s.feed.ingest(line("status", node="episode_runner", data=body))
        s.feed._seen_at["episode_runner"] -= age_s


@pytest.fixture()
def episode_console(console, monkeypatch, tmp_path):
    from s2r_console import console as mod

    noop = tmp_path / "noop_episode_cmd.py"
    noop.write_text("import sys; sys.exit(0)\n")
    monkeypatch.setattr(mod, "_EPISODE_CMD", noop)
    console.open("t_fake", operator="pytest")
    return console


def test_next_needs_the_exact_node_name(episode_console):
    from s2r_console.console import ConsoleError

    _runner_says(episode_console)
    with pytest.raises(ConsoleError, match="확인 입력"):
        episode_console.episode("next", operator="op", typed="place_cup")
    episode_console.episode("next", operator="op", typed="pick_cup")
    procs = {p["key"]: p for p in episode_console.snapshot()["session"]["procs"]}
    assert "episode#next" in procs and "--approve" in procs["episode#next"]["argv"]
    assert procs["episode#next"]["argv"][procs["episode#next"]["argv"].index("--approve") + 1] == "pick_cup"


def test_run_needs_the_episode_name_and_is_refused_while_busy(episode_console):
    from s2r_console.console import ConsoleError

    _runner_says(episode_console, busy=True, node="pick_cup")
    with pytest.raises(ConsoleError, match="돌리는 중"):
        episode_console.episode("run", operator="op", typed="episode:pick_place_right")
    _runner_says(episode_console)
    with pytest.raises(ConsoleError, match="확인 입력"):
        episode_console.episode("run", operator="op", typed="pick_place_right")
    episode_console.episode("run", operator="op", typed="episode:pick_place_right")


def test_a_silent_runner_is_refused_and_stop_needs_no_name(episode_console):
    from s2r_console.console import ConsoleError

    with pytest.raises(ConsoleError, match="상태가 안 온다"):
        episode_console.episode("stop", operator="op")
    _runner_says(episode_console, age_s=30.0)
    with pytest.raises(ConsoleError, match="상태가 안 온다"):
        episode_console.episode("next", operator="op", typed="pick_cup")
    _runner_says(episode_console, busy=True, node="pick_cup")
    episode_console.episode("stop", operator="op")                         # 돌고 있어도 멈춤은 나간다


def test_the_bridge_listens_to_the_episode_runner_status():
    from s2r_console import console as mod
    from types import SimpleNamespace
    prof = SimpleNamespace(domain=97, status_nodes=("pd_right",), stack=None, diagram=None)
    argv = mod.bridge_argv(prof, None)
    i = argv.index("--nodes")
    assert argv[i + 1:i + 3] == ["pd_right", "episode_runner"]


def test_the_stop_bar_also_stops_the_episode_runner(episode_console):
    """정지 바의 '에피소드 정지'가 정책 노드만 멈추면 연속 실행 중인 실행기가 실패 → 복구(되돌아가기 동작)로 이어 간다."""
    _runner_says(episode_console, busy=True, node="pick_cup")
    episode_console.quick("episode_stop", client="pytest")
    keys = {p["key"] for p in episode_console.snapshot()["session"]["procs"]}
    assert "episode#stop" in keys

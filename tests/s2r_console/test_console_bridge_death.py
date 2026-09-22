"""브리지가 죽었을 때 운영자가 보는 것 — `rc=1` 만으로는 할 수 있는 일이 없다.

2026-09-21 실제로 겪은 것: `PYTHONPATH=… python -m s2r_console` 로 띄워 setup.bash 의 경로를 덮어썼고,
브리지는 rclpy 를 못 찾아 3 초마다 죽었으며, 화면에는 "브리지 종료: rc=1" 만 쌓였다.
"""
from __future__ import annotations

import pytest

from s2r_console.console import death_reason, restart_delay
from s2r_console.console_state import OFFLINE, Observed, derive
from s2r_console.feed import Feed, line

RCLPY_TRACE = """Traceback (most recent call last):
  File "bridge.py", line 49, in main
    import rclpy
ModuleNotFoundError: No module named 'rclpy'
"""


def test_a_missing_rclpy_says_what_to_do():
    why = death_reason(1, RCLPY_TRACE)
    assert "rc=1" in why and "rclpy" in why and "setup.bash" in why


def test_any_other_death_carries_the_last_log_line():
    assert death_reason(1, "warming up\n\nRuntimeError: boom\n\n") == "rc=1 · RuntimeError: boom"


def test_an_empty_log_is_just_the_return_code():
    assert death_reason(-9, "") == "rc=-9"


def test_a_long_last_line_is_cut():
    assert len(death_reason(1, "x" * 5000)) < 220


@pytest.mark.parametrize(("n", "want"), [(0, 3.0), (1, 3.0), (2, 6.0), (3, 12.0), (4, 24.0), (5, 30.0), (50, 30.0)])
def test_restart_backs_off_and_is_capped(n, want):
    assert restart_delay(n) == want


def test_repeated_deaths_for_one_reason_are_one_event():
    f = Feed(("pd",), expect_domain=97)
    for _ in range(14):
        f.bridge_died("rc=1 · boom")
    assert [e["text"] for e in f.events] == ["브리지 종료 ×14: rc=1 · boom"]


def test_a_different_reason_is_a_new_event():
    f = Feed(("pd",), expect_domain=97)
    f.bridge_died("rc=1 · boom")
    f.bridge_died("rc=1 · boom")
    f.bridge_died("rc=1 · other")
    assert [e["text"] for e in f.events] == ["브리지 종료 ×2: rc=1 · boom", "브리지 종료: rc=1 · other"]


def test_a_reconnect_between_two_deaths_keeps_both():
    f = Feed(("pd",), expect_domain=97)
    f.bridge_died("rc=1 · boom")
    f.ingest(line("hello", domain=97, nodes=["pd"]))
    f.bridge_died("rc=1 · boom")
    assert [e["kind"] for e in f.events] == ["bridge", "bridge", "bridge"]
    assert "×" not in f.events[-1]["text"]


def test_the_offline_banner_says_why():
    f = Feed(("pd",), expect_domain=97)
    f.bridge_died("rc=1 · boom")
    b = derive(f.observed(), expected_nodes=("pd",))
    assert b.state == OFFLINE and b.reasons == ("ROS 브리지가 떠 있지 않다", "rc=1 · boom")


def test_the_reason_is_dropped_once_the_bridge_is_back():
    now = [0.0]
    f = Feed(("pd",), expect_domain=97, clock=lambda: now[0])
    f.bridge_died("rc=1 · boom")
    f.ingest(line("hello", domain=97, nodes=["pd"]))
    assert f.observed().bridge_down_why is None


def test_offline_without_a_known_reason_is_unchanged():
    assert derive(Observed(), expected_nodes=("pd",)).reasons == ("ROS 브리지가 떠 있지 않다",)

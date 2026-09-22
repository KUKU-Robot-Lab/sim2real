"""배너 파생 — 위에서부터 먼저 걸리는 것이 이긴다."""
from __future__ import annotations

import pytest

import s2r_console._paths  # noqa: F401
from s2r_console.console_state import Observed, derive

NODES = ("obs", "pd")


def obs(pd=None, other=None, *, up=True, age=0.1, faults=(), pd_age=None):
    status, ages = {}, {}
    if pd is not None:
        status["pd"], ages["pd"] = pd, age if pd_age is None else pd_age
    if other is not None:
        status["obs"], ages["obs"] = other, age
    return Observed(bridge_up=up, status=status, age_s=ages, bridge_faults=faults)


PD_IDLE = {"phase": "IDLE", "ok": True, "reasons": [], "execute": False, "estop": False}


def state(o):
    return derive(o, expected_nodes=NODES).state


def test_no_bridge_is_offline_whatever_else_is_known():
    assert state(obs(pd=PD_IDLE, up=False)) == "OFFLINE"


def test_no_status_at_all_is_offline_not_idle():
    assert state(obs()) == "OFFLINE"


def test_stale_status_counts_as_not_seen():
    assert state(obs(pd=PD_IDLE, age=5.0)) == "OFFLINE"


def test_bridge_fault_wins_over_node_state():
    b = derive(obs(pd=PD_IDLE, faults=("도메인이 다르다",)), expected_nodes=NODES)
    assert (b.state, b.reasons) == ("FAULT", ("도메인이 다르다",))


def test_estop_beats_hold():
    pd = {**PD_IDLE, "phase": "HOLD", "estop": True, "reasons": ["estop"]}
    assert state(obs(pd=pd)) == "ESTOP"


def test_hold_carries_pd_reasons_verbatim():
    b = derive(obs(pd={**PD_IDLE, "phase": "HOLD", "reasons": ["target stale 0.31 s"]}), expected_nodes=NODES)
    assert b.state == "HOLD" and b.reasons == ("pd: target stale 0.31 s",)


def test_a_not_ok_node_is_fault():
    b = derive(obs(pd=PD_IDLE, other={"phase": "idle", "ok": False, "reasons": ["소스 없음"]}), expected_nodes=NODES)
    assert b.state == "FAULT" and b.reasons == ("obs: 소스 없음",)


@pytest.mark.parametrize(("pd_phase", "other_phase", "want"), [
    ("RELEASING", "running", "STOPPING"),
    ("TRACKING", "running", "RUNNING"),
    ("TRACKING", "idle", "ARMED"),
    ("RAMPING", "idle", "ARMED"),
    ("IDLE", "idle", "IDLE"),
])
def test_phase_table(pd_phase, other_phase, want):
    assert state(obs(pd={**PD_IDLE, "phase": pd_phase}, other={"phase": other_phase, "ok": True})) == want


def test_idle_lists_the_nodes_that_are_missing():
    b = derive(obs(pd=PD_IDLE), expected_nodes=NODES)
    assert b.state == "IDLE" and b.reasons == ("obs: status 없음",)


@pytest.mark.parametrize(("execute", "want"), [(True, True), (False, False)])
def test_armed_for_real_mirrors_pd_execute(execute, want):
    assert derive(obs(pd={**PD_IDLE, "execute": execute}), expected_nodes=NODES).armed_for_real is want


def test_armed_for_real_is_unknown_without_pd():
    assert derive(obs(other={"phase": "idle", "ok": True}), expected_nodes=NODES).armed_for_real is None

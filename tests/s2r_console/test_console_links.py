"""연결 사슬 — 브리지 → 입력 → 정책 → 실기 → 가드 가 지금 이어져 있는지를 status 에서 파생한다."""
from __future__ import annotations

import s2r_console._paths  # noqa: F401
from s2r_console.console_state import Observed
from s2r_console.links import TONE, chain

NODES = ("pour_node", "pd", "pour_guard")

INPUTS = [{"name": "src:arm", "state": "live", "age_ms": 4.0},
          {"name": "src:cup", "state": "live", "age_ms": 30.0},
          {"name": "src:force", "state": "off", "age_ms": None},
          {"name": "fill", "state": "held", "age_ms": 9000.0}]
POUR = {"phase": "running", "seq": 12, "ok": True, "reasons": [], "inputs": INPUTS}
ARM = {"state_age_ms": {"arm": 3.0, "ee": 5.0}, "state_stale_ms": 100.0}
PD = {"phase": "TRACKING", "seq": 11, "ok": True, "reasons": [], "execute": True, "estop": False,
      "arms": {"left": ARM, "right": ARM}}
GUARD = {"ok": True, "reasons": [], "latched": False, "armed": True}


def observed(status=None, *, up=True, ages=None, faults=()):
    status = {"pour_node": POUR, "pd": PD, "pour_guard": GUARD} if status is None else status
    ages = {n: 0.1 for n in status} | (ages or {})
    return Observed(bridge_up=up, status=status, age_s=ages, bridge_faults=faults)


def blocks(o, *, nodes=NODES, cls="fake"):
    return {b["id"]: b for b in chain(o, expected_nodes=nodes, domain=97, domain_class=cls)}


def rows(block):
    return {r["name"]: r for r in block["rows"]}


def test_blocks_come_in_signal_order():
    out = chain(observed(), expected_nodes=NODES, domain=97, domain_class="fake")
    assert [b["id"] for b in out] == ["bridge", "inputs", "policy", "robot", "guard"]


def test_healthy_chain_is_live_everywhere():
    assert {b["state"] for b in blocks(observed()).values()} == {"live"}


def test_dead_bridge_makes_everything_downstream_unknown_not_live():
    b = blocks(observed(up=False))
    assert b["bridge"]["state"] == "down"
    assert {b[k]["state"] for k in ("inputs", "policy", "robot", "guard")} == {"unknown"}


def test_bridge_fault_is_shown_on_the_bridge_block():
    b = blocks(observed(faults=("도메인 0 에 붙었다",)))
    assert b["bridge"]["state"] == "fault"
    assert "도메인 0" in b["bridge"]["detail"]


def test_input_rows_are_the_reporting_nodes_own_words():
    r = rows(blocks(observed())["inputs"])
    assert r["src:arm"]["state"] == "live" and r["src:arm"]["age_ms"] == 4.0


def test_inputs_block_names_who_reported_once_not_on_every_row():
    b = blocks(observed())["inputs"]
    assert "pour_node" in b["detail"]
    assert {r["note"] for r in b["rows"]} == {""}


def test_off_and_held_inputs_do_not_degrade_the_block():
    assert blocks(observed())["inputs"]["state"] == "live"


def test_worst_input_names_the_block():
    bad = [*INPUTS, {"name": "rcv:cup", "state": "stale", "age_ms": 900.0},
           {"name": "rcv:arm", "state": "missing", "age_ms": None}]
    b = blocks(observed({"pour_node": {**POUR, "inputs": bad}, "pd": PD, "pour_guard": GUARD}))
    assert b["inputs"]["state"] == "missing"


def test_inputs_are_unknown_when_their_reporter_went_quiet():
    b = blocks(observed(ages={"pour_node": 5.0}))["inputs"]
    assert b["state"] == "unknown" and b["rows"] == []
    assert "pour_node" in b["detail"]


def test_inputs_are_unknown_before_any_node_reports_them():
    b = blocks(observed({"pd": PD}))["inputs"]
    assert b["state"] == "unknown" and b["detail"]


def test_an_input_state_we_do_not_know_is_not_passed_off_as_live():
    odd = [{"name": "x", "state": "sparkly", "age_ms": 1.0}]
    b = blocks(observed({"pour_node": {**POUR, "inputs": odd}, "pd": PD, "pour_guard": GUARD}))
    assert rows(b["inputs"])["x"]["state"] == "unknown"


def test_policy_block_lists_every_non_robot_non_guard_node():
    b = blocks(observed({"obs": POUR, "policy": POUR, "pd": PD}), nodes=("obs", "policy", "fabric", "pd"))
    assert set(rows(b["policy"])) == {"obs", "policy", "fabric"}
    assert rows(b["policy"])["fabric"]["state"] == "missing"
    assert b["policy"]["state"] == "missing"
    assert "guard" not in b


def test_policy_row_tells_phase_and_seq():
    note = rows(blocks(observed())["policy"])["pour_node"]["note"]
    assert "running" in note and "12" in note


def test_a_node_that_says_not_ok_is_a_fault_with_its_reason():
    sick = {**POUR, "ok": False, "reasons": ["inputs missing ['src:cup']"]}
    r = rows(blocks(observed({"pour_node": sick, "pd": PD, "pour_guard": GUARD}))["policy"])["pour_node"]
    assert r["state"] == "fault" and "src:cup" in r["note"]


def test_a_quiet_node_is_stale_whatever_it_last_said():
    r = rows(blocks(observed(ages={"pour_node": 9.0}))["policy"])["pour_node"]
    assert r["state"] == "stale"


def test_robot_block_shows_each_arms_state_stream():
    slow = {"state_age_ms": {"arm": 250.0, "ee": None}, "state_stale_ms": 100.0}
    b = blocks(observed({"pour_node": POUR, "pd": {**PD, "arms": {"left": ARM, "right": slow}}, "pour_guard": GUARD}))
    r = rows(b["robot"])
    assert r["left:arm"]["state"] == "live"
    assert r["right:arm"]["state"] == "stale" and r["right:ee"]["state"] == "missing"
    assert b["robot"]["state"] == "missing"


def test_robot_block_says_whether_commands_really_go_out():
    assert "명령 나감" in blocks(observed())["robot"]["detail"]
    dry = blocks(observed({"pour_node": POUR, "pd": {**PD, "execute": False}, "pour_guard": GUARD}))
    assert "dry-run" in dry["robot"]["detail"]


def test_robot_block_is_titled_by_domain_class():
    assert "fake" in blocks(observed(), cls="fake")["robot"]["title"]
    assert "실기" in blocks(observed(), cls="real")["robot"]["title"]


def test_estop_is_a_fault_on_the_robot_block():
    b = blocks(observed({"pour_node": POUR, "pd": {**PD, "estop": True}, "pour_guard": GUARD}))["robot"]
    assert b["state"] == "fault" and "ESTOP" in b["detail"]


def test_latched_guard_is_a_fault():
    g = {**GUARD, "ok": False, "latched": True, "reasons": ["src cup moved 0.08 m"]}
    b = blocks(observed({"pour_node": POUR, "pd": PD, "pour_guard": g}))["guard"]
    assert b["state"] == "fault" and "0.08" in rows(b)["pour_guard"]["note"]


def test_every_state_has_a_tone():
    states = {b["state"] for b in blocks(observed(up=False)).values()} | {b["state"] for b in blocks(observed()).values()}
    assert states <= set(TONE)
    for b in blocks(observed()).values():
        assert b["tone"] == TONE[b["state"]]
        assert all(r["tone"] == TONE[r["state"]] for r in b["rows"])


def _pd_with_target(target):
    arm = {**ARM, "target": target}
    return {"pour_node": POUR, "pd": {**PD, "arms": {"left": arm, "right": arm}}, "pour_guard": GUARD}


def test_robot_block_says_whose_target_pd_is_following():
    ext = rows(blocks(observed(_pd_with_target("external")))["robot"])["left:목표"]
    assert ext["state"] == "live" and "체인" in ext["note"]
    own = rows(blocks(observed(_pd_with_target("internal")))["robot"])["left:목표"]
    assert own["state"] == "held" and "내부" in own["note"]
    none = rows(blocks(observed(_pd_with_target(None)))["robot"])["left:목표"]
    assert none["state"] == "off"


def test_pd_holding_its_own_target_does_not_degrade_the_robot_block():
    assert blocks(observed(_pd_with_target("internal")))["robot"]["state"] == "live"


def test_a_held_value_does_not_wear_the_same_lamp_as_a_flowing_one():
    # 보유(래치 값·내부 유지 목표)는 고장이 아니지만 흐르는 것도 아니다. 초록이면 fabric 을 끈
    # fake 에서 "정책 출력이 로봇에 닿는다" 로 읽힌다 (09.21 화면 확인).
    assert TONE["held"] != TONE["live"]
    assert TONE["held"] not in ("warn", "bad")

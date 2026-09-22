"""로봇 스택 블록 — policy_control 바깥(robot_control 등)의 노드·토픽·컨트롤러가 붙어 있는가."""
from __future__ import annotations

from pathlib import Path

import pytest

import s2r_console._paths  # noqa: F401
from s2r_console.bridge import TopicMeter
from s2r_console.console_state import Observed
from s2r_console.ctl_probe import summarize
from s2r_console.feed import Feed, line
from s2r_console.links import chain
from s2r_console.profiles import ProfileError, Stack, StackManager, StackTopic, parse, scan

SIM2REAL = Path(__file__).resolve().parents[2]
NODES = ("pour_node", "pd")
CM = "/controller_manager"
STACK = Stack(title="로봇 스택", nodes=("/controller_manager", "/robot_state_publisher"),
              topics=(StackTopic("/joint_states", stale_ms=500.0), StackTopic("/head/joint_states")),
              managers=(StackManager(CM, active=("joint_state_broadcaster",)),))
TOPICS = {"/joint_states": {"pubs": 1, "n": 100, "age_ms": 8.0},
          "/head/joint_states": {"pubs": 1, "n": 20, "age_ms": 40.0}}
CTL = {CM: {"ok": True, "controllers": [
    {"name": "joint_state_broadcaster", "state": "active", "type": "jsb"},
    {"name": "left_joint_trajectory_controller", "state": "active", "type": "jtc"},
    {"name": "left_forward_position_controller", "state": "inactive", "type": "fwd"}]}}


def observed(*, up=True, graph=STACK.nodes, topics=None, topics_age=0.5, ctl=None, ctl_age=0.5, probe_why=None):
    ctl = CTL if ctl is None else ctl
    return Observed(bridge_up=up, graph=tuple(graph), topics=TOPICS if topics is None else topics,
                    topics_age_s=topics_age, controllers=ctl, controllers_age_s={m: ctl_age for m in ctl},
                    probe_down_why=probe_why)


def stack_rows(o, stack=STACK):
    out = chain(o, expected_nodes=NODES, domain=97, domain_class="fake", stack=stack)
    block = next(b for b in out if b["id"] == "stack")
    return block, {r["name"]: r for r in block["rows"]}


# ── 블록 ────────────────────────────────────────────────────────────────
def test_stack_block_sits_right_after_the_bridge_and_only_when_declared():
    ids = [b["id"] for b in chain(observed(), expected_nodes=NODES, domain=97, domain_class="fake", stack=STACK)]
    assert ids[:3] == ["bridge", "stack", "inputs"]
    assert "stack" not in [b["id"] for b in chain(observed(), expected_nodes=NODES, domain=97, domain_class="fake")]


def test_healthy_stack_is_live():
    block, r = stack_rows(observed())
    assert block["state"] == "live" and block["title"] == "로봇 스택"
    assert r["/controller_manager"]["state"] == "live"
    assert r["/joint_states"]["state"] == "live" and "100 Hz" in r["/joint_states"]["note"]
    assert r["joint_state_broadcaster"]["state"] == "live"


def test_dead_bridge_makes_the_stack_unknown():
    block, _ = stack_rows(observed(up=False))
    assert block["state"] == "unknown" and block["rows"] == []


def test_a_node_missing_from_the_graph_is_missing():
    block, r = stack_rows(observed(graph=("/controller_manager",)))
    assert r["/robot_state_publisher"]["state"] == "missing" and block["state"] == "missing"


@pytest.mark.parametrize(("report", "state"), [
    ({"pubs": 0, "n": 0, "age_ms": None}, "missing"),      # 아무도 안 낸다
    ({"pubs": 1, "n": 0, "age_ms": None}, "missing"),      # 내는 쪽은 있는데 한 번도 안 왔다
    ({"pubs": 1, "n": 0, "age_ms": 900.0}, "stale"),       # 오다가 끊겼다 (문턱 500)
    ({"pubs": 1, "n": 100, "age_ms": 499.0}, "live"),
])
def test_topic_row_follows_publisher_count_and_age(report, state):
    _, r = stack_rows(observed(topics={**TOPICS, "/joint_states": report}))
    assert r["/joint_states"]["state"] == state


def test_a_topic_with_more_than_one_publisher_says_so():
    # 09.21 fake 플랜트 실측: tip_forces_xyz 를 두 노드가 같이 내고 있었다 - 60 Hz 가 120 Hz 로 보였다.
    # 판정은 바꾸지 않는다(값이 오고 있다). 몇 명이 내는지만 숨기지 않는다.
    _, one = stack_rows(observed())
    _, two = stack_rows(observed(topics={**TOPICS, "/joint_states": {"pubs": 2, "n": 120, "age_ms": 3.0}}))
    assert "발행자" not in one["/joint_states"]["note"]
    assert two["/joint_states"]["state"] == "live"
    assert two["/joint_states"]["note"] == "120 Hz · 발행자 2"


def test_topics_are_unknown_when_the_bridge_does_not_report_them():
    for o in (observed(topics={}, topics_age=None), observed(topics_age=10.0)):
        _, r = stack_rows(o)
        assert r["/joint_states"]["state"] == "unknown"


def test_a_required_controller_that_is_not_active_is_a_fault():
    ctl = {CM: {"ok": True, "controllers": [{"name": "joint_state_broadcaster", "state": "inactive"}]}}
    block, r = stack_rows(observed(ctl=ctl))
    assert r["joint_state_broadcaster"]["state"] == "fault" and block["state"] == "fault"


def test_a_required_controller_that_is_not_loaded_is_missing():
    _, r = stack_rows(observed(ctl={CM: {"ok": True, "controllers": []}}))
    assert r["joint_state_broadcaster"]["state"] == "missing"


def test_other_controllers_show_their_state_without_being_judged():
    _, r = stack_rows(observed())
    assert r["left_joint_trajectory_controller"]["state"] == "live"
    assert r["left_forward_position_controller"]["state"] == "off"
    assert "inactive" in r["left_forward_position_controller"]["note"]


def test_a_manager_that_does_not_answer_is_missing_with_its_reason():
    _, r = stack_rows(observed(ctl={CM: {"ok": False, "reason": "서비스가 없다"}}))
    assert r[CM + " 컨트롤러"]["state"] == "missing" and "서비스가 없다" in r[CM + " 컨트롤러"]["note"]
    assert "joint_state_broadcaster" not in r


def test_a_silent_probe_is_unknown_and_says_why():
    for o in (observed(ctl={}, probe_why="rc=1"), observed(ctl_age=30.0, probe_why="rc=1")):
        _, r = stack_rows(o)
        assert r[CM + " 컨트롤러"]["state"] == "unknown" and "rc=1" in r[CM + " 컨트롤러"]["note"]


def test_namespaced_manager_prefixes_its_controllers():
    hand = "/dg5f_right/controller_manager"
    stack = Stack(title="s", nodes=(), topics=(), managers=(StackManager(hand, active=("dg5f_right_controller",)),))
    ctl = {hand: {"ok": True, "controllers": [{"name": "dg5f_right_controller", "state": "active"}]}}
    _, r = stack_rows(observed(ctl=ctl), stack)
    assert r["/dg5f_right/dg5f_right_controller"]["state"] == "live"


# ── 브리지 계측 ─────────────────────────────────────────────────────────
def test_topic_meter_counts_per_interval_and_keeps_the_last_arrival():
    m = TopicMeter(["/a", "/b"])
    m.hit("/a", 10.0)
    m.hit("/a", 10.5)
    first = m.report(11.0, pubs={"/a": 1, "/b": 0})
    assert first["/a"] == {"pubs": 1, "n": 2, "age_ms": 500.0}
    assert first["/b"] == {"pubs": 0, "n": 0, "age_ms": None}
    second = m.report(12.0, pubs={"/a": 1, "/b": 0})
    assert second["/a"] == {"pubs": 1, "n": 0, "age_ms": 1500.0}      # 세는 것은 구간마다 비운다, 나이는 이어진다


def test_probe_summary_keeps_name_state_type_only():
    class C:
        name, state, type, claimed_interfaces = "jsb", "active", "t", ["x"]
    assert summarize([C()]) == [{"name": "jsb", "state": "active", "type": "t"}]


# ── feed ────────────────────────────────────────────────────────────────
class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_feed_carries_graph_topics_and_controllers_into_observed():
    clock = Clock()
    f = Feed(NODES, expect_domain=97, clock=clock)
    f.ingest(line("hello", domain=97, nodes=list(NODES)))
    f.ingest(line("beat", graph=["/pd_node"]))
    f.ingest(line("topics", data=TOPICS))
    f.ingest(line("controllers", manager=CM, ok=True, controllers=CTL[CM]["controllers"]))
    clock.t = 1.5
    o = f.observed()
    assert o.graph == ("/pd_node",) and o.topics == TOPICS and o.topics_age_s == 1.5
    assert o.controllers[CM]["ok"] is True and o.controllers_age_s[CM] == 1.5


def test_feed_logs_a_controller_switch_once():
    f = Feed(NODES, expect_domain=97, clock=Clock())
    before = CTL[CM]["controllers"]
    after = [{**c, "state": "inactive" if "trajectory" in c["name"] else "active"} for c in before]
    for ctl in (before, before, after, after):
        f.ingest(line("controllers", manager=CM, ok=True, controllers=ctl))
    texts = [e["text"] for e in f.events if e["kind"] == "stack"]
    assert len(texts) == 2 and "left_forward_position_controller" in texts[1]


def test_probe_death_is_reported_until_it_answers_again():
    f = Feed(NODES, expect_domain=97, clock=Clock())
    f.probe_died("rc=1")
    f.probe_died("rc=1")
    assert f.observed().probe_down_why == "rc=1"
    assert len([e for e in f.events if e["kind"] == "stack"]) == 1
    f.ingest(line("controllers", manager=CM, ok=False, reason="서비스가 없다"))
    assert f.observed().probe_down_why is None


# ── 프로파일 ────────────────────────────────────────────────────────────
def raw(**over):
    base = {"schema": "s2r_console/profile/v1", "id": "p", "mission": "m.yaml",
            "domain": {"id": 97, "class": "fake"}, "status_nodes": ["obs", "pd"]}
    return {**base, **over}


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "m.yaml").write_text("name: x\n")
    return tmp_path


def test_profile_without_stack_has_none(repo):
    assert parse(raw(), path=repo / "p.yaml", repo=repo).stack is None


def test_profile_stack_is_parsed(repo):
    p = parse(raw(stack={"title": "t", "nodes": ["/controller_manager"],
                         "topics": [{"name": "/joint_states", "stale_ms": 500}, "/head/joint_states"],
                         "managers": [{"name": CM, "active": ["joint_state_broadcaster"]}]}),
              path=repo / "p.yaml", repo=repo)
    assert p.stack == Stack(title="t", nodes=("/controller_manager",),
                            topics=(StackTopic("/joint_states", stale_ms=500.0), StackTopic("/head/joint_states")),
                            managers=(StackManager(CM, active=("joint_state_broadcaster",)),))
    assert p.as_dict()["stack"]["topics"][0] == {"name": "/joint_states", "stale_ms": 500.0}


@pytest.mark.parametrize("stack", [
    {"nodez": []},                                    # 모르는 키
    {"topics": [{"name": "joint_states"}]},           # 절대 이름이 아니다
    {"topics": [{"name": "/a", "stale": 1}]},         # 모르는 키
    {"managers": [{"active": ["x"]}]},                # name 없음
    {"nodes": "controller_manager"},                  # 목록이 아니다
])
def test_malformed_stack_is_a_load_error(repo, stack):
    with pytest.raises(ProfileError):
        parse(raw(stack=stack), path=repo / "p.yaml", repo=repo)


def test_shipped_profiles_declare_a_stack():
    good, bad = scan(SIM2REAL / "s2r_console" / "profiles", repo=SIM2REAL)
    assert bad == {}
    assert all(p.stack is not None and p.stack.topics for p in good)


# ── 콘솔이 자식에게 넘기는 인자 ─────────────────────────────────────────
def _profile(name: str):
    from s2r_console.profiles import scan
    good, bad = scan(SIM2REAL / "s2r_console" / "profiles", repo=SIM2REAL)
    assert not bad
    return next(p for p in good if p.id == name)


def test_bridge_argv_carries_the_declared_stack_topics():
    from s2r_console.console import bridge_argv
    argv = bridge_argv(_profile("pour_i18_fake"))
    assert argv[argv.index("--domain") + 1] == "97"
    topics = argv[argv.index("--topics") + 1:]
    assert "/joint_states" in topics and "/dg5f_left/tip_forces_xyz" in topics


def test_bridge_argv_has_no_topic_flags_when_nothing_declares_topics():
    from dataclasses import replace
    from s2r_console.console import bridge_argv
    argv = bridge_argv(replace(_profile("pour_i18_fake"), stack=None, diagram=None))
    assert "--topics" not in argv and "--watch" not in argv


def test_probe_argv_is_none_unless_managers_are_declared():
    from dataclasses import replace
    from s2r_console.console import probe_argv
    p = _profile("pour_i18_fake")
    assert probe_argv(replace(p, stack=None)) is None
    assert probe_argv(replace(p, stack=replace(p.stack, managers=()))) is None
    argv = probe_argv(p)
    assert argv[argv.index("--managers") + 1:] == ["/controller_manager"]
    assert "s2r_console.ctl_probe" in argv


# ── 그림을 위한 계측: 누가 내고 누가 받는가 ─────────────────────────────
def test_topic_meter_reports_who_publishes_and_who_listens():
    m = TopicMeter(["/a"], watch=["/w"])
    m.hit("/a", 1.0)
    ends = {"/a": {"pub_nodes": ["/p"], "sub_nodes": ["/s"]}, "/w": {"pub_nodes": ["/p"], "sub_nodes": []}}
    rep = m.report(2.0, pubs={"/a": 1, "/w": 1}, ends=ends)
    assert rep["/a"] == {"pubs": 1, "n": 1, "age_ms": 1000.0, "pub_nodes": ["/p"], "sub_nodes": ["/s"]}
    # 지켜보기만 하는 토픽은 구독하지 않는다 — 센 것이 없으니 0 이 아니라 None 이다
    assert rep["/w"] == {"pubs": 1, "n": None, "age_ms": None, "pub_nodes": ["/p"], "sub_nodes": []}


def test_bridge_argv_meters_the_diagram_wires_and_only_watches_the_rest():
    from s2r_console.console import bridge_argv, diagram_of, mission_units
    profile = _profile("pour_i18_fake")
    argv = bridge_argv(profile, diagram_of(profile, mission_units(profile, repo=SIM2REAL), repo=SIM2REAL))
    topics = argv[argv.index("--topics") + 1:argv.index("--watch")]
    watch = argv[argv.index("--watch") + 1:]
    assert "/policy_control/joint_target" in topics and "/joint_states" in topics
    assert len(topics) == len(set(topics))                          # 스택과 그림이 같은 토픽을 말해도 한 번만 구독한다
    assert "/policy_control/pour/fill_level" in watch and not set(watch) & set(topics)


def test_the_bridge_listens_to_the_perception_launcher_only_when_the_picture_has_one(tmp_path):
    from s2r_console.console import bridge_argv, diagram_of, mission_units
    from s2r_console.profiles import scan
    repo = Path(__file__).resolve().parents[2]
    profiles = {p.id: p for p in scan(repo / "s2r_console/profiles", repo=repo)[0]}
    pour = profiles["pour_i18_fake"]
    argv = bridge_argv(pour, diagram_of(pour, mission_units(pour, repo=repo), repo=repo))
    assert "--perception" not in argv                      # fake 미션에는 인지 런처가 없다

    from s2r_console.diagram_spec import parse_diagram
    with_launcher = parse_diagram({
        "boxes": [{"id": "perception", "title": "인지 런처", "col": 0, "ros": ["/perception_launcher"],
                   "host": "vision-3090"},
                  {"id": "camera", "title": "카메라", "col": 1, "host": "vision-3090"}],
        "wires": [{"from": "perception", "to": "camera", "topic": "/perception/status", "meter": False}],
    }, path=Path("p.yaml"))
    argv = bridge_argv(pour, with_launcher)
    assert argv[argv.index("--perception") + 1] == "/perception/status"

"""연결 그림 — 상자(노드)와 전선(토픽)의 지금 상태를 브리지가 본 것에서 파생한다."""
from __future__ import annotations

from pathlib import Path

import pytest

import s2r_console._paths  # noqa: F401
from s2r_console.console_state import Observed
from s2r_console.diagram import build
from s2r_console.diagram_spec import parse_diagram
from s2r_console.links import TONE

SPEC = parse_diagram({
    "boxes": [
        {"id": "arm", "title": "팔", "col": 0, "unit": "plant#0"},
        {"id": "cup", "title": "컵", "col": 0, "unit": "plant#1"},
        {"id": "operator", "title": "운영자 입력", "col": 0},
        {"id": "pour_node", "title": "정책", "col": 1, "status": "pour_node", "ros": ["/pour_node"], "unit": "chain#0"},
        {"id": "pd", "title": "pd", "col": 2, "status": "pd", "ros": ["/pd_node"], "unit": "pd_load#0"},
        {"id": "arm_cmd", "title": "팔 구동", "col": 3, "ros": ["/fake_arm_bridge"], "unit": "plant#0",
         "manager": "/controller_manager"},
    ],
    "wires": [
        {"from": "arm", "to": "pour_node", "topic": "/joint_states", "inputs": ["src:arm"], "stale_ms": 500},
        {"from": "cup", "to": "pour_node", "topic": "/objects/cup/pose", "inputs": ["src:cup"], "stale_ms": 500},
        {"from": "operator", "to": "pour_node", "topic": "/fill", "inputs": ["fill"], "on_demand": True},
        {"from": "pour_node", "to": "pd", "topic": "/policy_control/joint_target", "stale_ms": 500},
        {"from": "pour_node", "to": "pd", "topic": "/policy_control/episode", "meter": False},
        {"from": "pd", "to": "arm_cmd", "topic": "/left_forward_position_controller/commands", "stale_ms": 500},
    ],
}, path=Path("p.yaml"))

INPUTS = [{"name": "src:arm", "state": "live", "age_ms": 4.0}, {"name": "src:cup", "state": "live", "age_ms": 30.0},
          {"name": "fill", "state": "held", "age_ms": 9000.0}]
POUR = {"phase": "running", "seq": 12, "ok": True, "reasons": [], "inputs": INPUTS}
PD = {"phase": "TRACKING", "seq": 11, "ok": True, "reasons": [], "execute": True, "estop": False,
      "arms": {"left": {"state_age_ms": {"arm": 3.0}, "state_stale_ms": 100.0, "target": "external"}}}


def topic(pubs=("/x",), subs=(), n=100, age=5.0):
    return {"pubs": len(pubs), "n": n, "age_ms": age, "pub_nodes": list(pubs), "sub_nodes": list(subs)}


TOPICS = {
    "/joint_states": topic(("/fake_arm_bridge",), ("/pour_node", "/pd_node")),
    "/objects/cup/pose": topic(("/fake_cup_pose_pub",), ("/pour_node",), n=30),
    "/fill": topic((), ("/pour_node",), n=None, age=None),
    "/policy_control/joint_target": topic(("/pour_node",), ("/pd_node",), n=60),
    "/policy_control/episode": topic(("/pour_node",), ("/pd_node",), n=None, age=None),
    "/left_forward_position_controller/commands": topic(("/pd_node",), ("/fake_arm_bridge",), n=250),
}
GRAPH = ("/fake_arm_bridge", "/pour_node", "/pd_node")
CTL = {"/controller_manager": {"ok": True, "controllers": [
    {"name": "left_forward_position_controller", "state": "active"},
    {"name": "left_joint_trajectory_controller", "state": "inactive"}]}}
UP = {"alive": True, "rc": None, "started": True, "stopped": False}
UNITS = {k: {"key": k, **UP} for k in ("plant#0", "plant#1", "chain#0", "pd_load#0")}


def observed(*, up=True, status=None, topics=None, topics_age=0.5, graph=GRAPH, ages=None):
    status = {"pour_node": POUR, "pd": PD} if status is None else status
    return Observed(bridge_up=up, status=status, age_s={n: 0.1 for n in status} | (ages or {}), graph=tuple(graph),
                    topics=TOPICS if topics is None else topics, topics_age_s=topics_age,
                    controllers=CTL, controllers_age_s={m: 0.5 for m in CTL})


def view(o=None, units=None):
    out = build(observed() if o is None else o, SPEC, units=UNITS if units is None else units)
    boxes = {b["id"]: b for col in out["cols"] for b in col}
    ports = {p["topic"]: p for b in boxes.values() for p in b["ports"]}
    return out, boxes, ports


# ── 모양 ────────────────────────────────────────────────────────────────
def test_boxes_are_grouped_into_columns_in_declared_order():
    out, _, _ = view()
    assert [[b["id"] for b in col] for col in out["cols"]] == [["arm", "cup", "operator"], ["pour_node"], ["pd"], ["arm_cmd"]]


def test_each_wire_lands_on_a_port_of_its_target_box():
    out, boxes, _ = view()
    assert [p["topic"] for p in boxes["pour_node"]["ports"]] == ["/joint_states", "/objects/cup/pose", "/fill"]
    assert [(w["from"], w["to"]) for w in out["wires"]][:2] == [("arm", "pour_node"), ("cup", "pour_node")]
    assert {w["id"] for w in out["wires"]} == {p["id"] for b in boxes.values() for p in b["ports"]}


def test_every_state_carries_its_tone():
    out, boxes, ports = view()
    for item in [*boxes.values(), *ports.values(), *out["wires"]]:
        assert item["tone"] == TONE[item["state"]]


# ── 전선 ────────────────────────────────────────────────────────────────
def test_a_healthy_metered_wire_is_live_flows_and_says_its_rate():
    out, _, ports = view()
    assert ports["/joint_states"]["state"] == "live" and ports["/joint_states"]["text"] == "100 Hz"
    wire = next(w for w in out["wires"] if w["id"] == ports["/joint_states"]["id"])
    assert wire["flow"] is True


def test_an_unmetered_wire_is_connected_but_does_not_pretend_to_flow():
    out, _, ports = view()
    p = ports["/policy_control/episode"]
    assert p["state"] == "live" and "Hz" not in p["text"]
    assert next(w for w in out["wires"] if w["id"] == p["id"])["flow"] is False


def test_dead_bridge_makes_everything_unknown():
    out, boxes, ports = view(observed(up=False))
    assert {b["state"] for b in boxes.values()} == {"unknown"}
    assert {p["state"] for p in ports.values()} == {"unknown"}
    assert not any(w["flow"] for w in out["wires"])


def test_stale_topic_report_is_unknown_not_live():
    _, _, ports = view(observed(topics_age=10.0))
    assert ports["/joint_states"]["state"] == "unknown"


@pytest.mark.parametrize(("rep", "state", "needle"), [
    (topic((), ("/pour_node",), n=0, age=None), "missing", "발행자"),          # 받는 쪽은 기다리는데 아무도 안 낸다
    (topic(("/fake_arm_bridge",), (), n=0, age=None), "off", "받는 쪽"),        # 내는 쪽만 있다
    (topic((), (), n=0, age=None), "off", "양쪽"),
    (topic(("/fake_arm_bridge",), ("/pour_node",), n=0, age=None), "missing", "온 적"),
    (topic(("/fake_arm_bridge",), ("/pour_node",), n=0, age=900.0), "stale", "500"),
])
def test_wire_state_follows_publisher_subscriber_and_age(rep, state, needle):
    _, _, ports = view(observed(topics={**TOPICS, "/joint_states": rep}))
    assert ports["/joint_states"]["state"] == state and needle in ports["/joint_states"]["note"]


def test_the_receivers_own_word_wins_when_it_is_worse_than_the_topic():
    # 토픽은 흐르는데 받는 노드가 stale 이라고 한다 (QoS 불일치·콜백 막힘) — 받는 쪽 말을 믿는다.
    sick = {**POUR, "inputs": [{"name": "src:arm", "state": "stale", "age_ms": 800.0}, *INPUTS[1:]]}
    _, _, ports = view(observed(status={"pour_node": sick, "pd": PD}))
    assert ports["/joint_states"]["state"] == "stale" and "pour_node" in ports["/joint_states"]["note"]


def test_an_on_demand_wire_is_judged_by_the_receiver_not_by_the_missing_publisher():
    _, _, ports = view()
    assert ports["/fill"]["state"] == "held"
    empty = {**POUR, "inputs": [r for r in INPUTS if r["name"] != "fill"] + [{"name": "fill", "state": "off", "age_ms": None}]}
    _, _, ports = view(observed(status={"pour_node": empty, "pd": PD}))
    assert ports["/fill"]["state"] == "off"


# ── 상자 ────────────────────────────────────────────────────────────────
def test_status_boxes_say_what_the_node_says():
    _, boxes, _ = view()
    assert boxes["pour_node"]["state"] == "live" and "running" in boxes["pour_node"]["detail"]
    assert "명령 나감" in boxes["pd"]["detail"]
    assert any("체인 목표" in ln["text"] for ln in boxes["pd"]["lines"])


def test_a_chain_running_without_fabric_says_it_sends_no_targets():
    # use_fabric:=false 면 pour_node 는 joint_target 을 내지 않는다 — pd 로 가는 전선이 빨간 이유를 상자가 말해야 한다.
    _, boxes, _ = view(observed(status={"pour_node": {**POUR, "use_fabric": False}, "pd": PD}))
    assert any("fabric" in ln["text"] and ln["tone"] == "warn" for ln in boxes["pour_node"]["lines"])
    _, boxes, _ = view(observed(status={"pour_node": {**POUR, "use_fabric": True}, "pd": PD}))
    assert not boxes["pour_node"]["lines"]


def test_estop_makes_the_pd_box_a_fault():
    _, boxes, _ = view(observed(status={"pour_node": POUR, "pd": {**PD, "estop": True}}))
    assert boxes["pd"]["state"] == "fault" and "ESTOP" in boxes["pd"]["detail"]


def test_a_source_box_is_live_when_its_topics_have_publishers():
    _, boxes, _ = view()
    assert boxes["arm"]["state"] == "live" and boxes["cup"]["state"] == "live"


def test_a_source_that_was_never_started_is_off_not_a_fault():
    never = {**UNITS, "plant#1": {"key": "plant#1", "alive": False, "rc": None, "started": False, "stopped": False}}
    _, boxes, _ = view(observed(topics={**TOPICS, "/objects/cup/pose": topic((), ("/pour_node",), n=0, age=None)}), never)
    assert boxes["cup"]["state"] == "off"


def test_a_source_the_operator_switched_off_is_off():
    off = {**UNITS, "plant#1": {"key": "plant#1", "alive": False, "rc": -15, "started": True, "stopped": True}}
    _, boxes, _ = view(observed(topics={**TOPICS, "/objects/cup/pose": topic((), ("/pour_node",), n=0, age=None)}), off)
    assert boxes["cup"]["state"] == "off"


def test_a_source_whose_process_died_is_down_with_its_exit_code():
    dead = {**UNITS, "plant#1": {"key": "plant#1", "alive": False, "rc": 1, "started": True, "stopped": False}}
    _, boxes, _ = view(observed(topics={**TOPICS, "/objects/cup/pose": topic((), ("/pour_node",), n=0, age=None)}), dead)
    assert boxes["cup"]["state"] == "down" and "rc=1" in boxes["cup"]["detail"]


def test_a_running_source_that_publishes_nothing_is_missing():
    _, boxes, _ = view(observed(topics={**TOPICS, "/objects/cup/pose": topic((), ("/pour_node",), n=0, age=None)}))
    assert boxes["cup"]["state"] == "missing"


def test_a_status_box_whose_process_died_is_down():
    dead = {**UNITS, "chain#0": {"key": "chain#0", "alive": False, "rc": 1, "started": True, "stopped": False}}
    _, boxes, _ = view(observed(status={"pd": PD}), dead)
    assert boxes["pour_node"]["state"] == "down"


def test_a_sink_box_is_judged_by_its_ros_nodes_and_lists_its_controllers():
    _, boxes, _ = view()
    assert boxes["arm_cmd"]["state"] == "live"
    assert any("left_forward_position" in ln["text"] for ln in boxes["arm_cmd"]["lines"])
    _, boxes, _ = view(observed(graph=("/pour_node", "/pd_node")))
    assert boxes["arm_cmd"]["state"] == "missing"


def test_a_box_with_nothing_to_judge_follows_its_on_demand_wire():
    _, boxes, _ = view()
    assert boxes["operator"]["state"] == "held"


def test_boxes_carry_their_unit_and_say_who_shares_it():
    _, boxes, _ = view()
    assert boxes["arm"]["unit"]["key"] == "plant#0" and boxes["arm"]["shares"] == ["팔 구동"]
    assert boxes["operator"]["unit"] is None
    _, boxes, _ = view(units={})
    assert boxes["arm"]["unit"] == {"key": "plant#0", "error": "미션에 없는 명령이다"}


def test_summary_counts_a_broken_wire_even_when_every_box_is_up():
    # 09.21 화면: 상자는 전부 초록인데 pour_node→pd 전선이 빨갰고, 머리말은 "전부 이어짐" 이라고 했다.
    cut = {**TOPICS, "/policy_control/joint_target": topic(("/pour_node",), ("/pd_node",), n=0, age=None)}
    out, boxes, _ = view(observed(topics=cut))
    assert {b["tone"] for b in boxes.values()} <= {"ok", "mute"}
    assert out["summary"]["tone"] == "bad" and "정책 → pd" in out["summary"]["text"]


def test_controllers_are_folded_into_two_short_lines():
    _, boxes, _ = view()
    texts = [ln["text"] for ln in boxes["arm_cmd"]["lines"]]
    assert texts == ["active: left_forward_position", "inactive: left_joint_trajectory"]


def test_summary_names_the_broken_boxes():
    out, _, _ = view()
    assert out["summary"] == {"tone": "ok", "text": "전부 이어짐"}
    out, _, _ = view(observed(graph=("/pour_node", "/pd_node")))
    assert out["summary"]["tone"] == "bad" and "팔 구동" in out["summary"]["text"]


# ── 자동 생성이 쓰는 전선 속성 ──────────────────────────────────────────
def _spec(**wire):
    return parse_diagram({
        "boxes": [{"id": "pour_node", "title": "정책", "col": 0, "status": "pour_node", "ros": ["/pour_node"]},
                  {"id": "pd", "title": "pd", "col": 1, "status": "pd", "ros": ["/pd_node"], "stages": ["PD 법칙", "발행"]},
                  {"id": "drive", "title": "팔 구동", "col": 2, "ros": ["/controller_manager"]}],
        "wires": [{"from": "pour_node", "to": "pd", "topic": "/policy_control/joint_target", "stale_ms": 500, "episodic": True},
                  {"from": "pd", "to": "drive", "topic": "/left_forward_position_controller/commands", **wire}],
    }, path=Path("p.yaml"))


def _ports(spec, o):
    out = build(o, spec, units={})
    return out, {p["topic"]: p for col in out["cols"] for b in col for p in b["ports"]}


def test_a_controller_is_heard_on_its_own_node_not_on_the_manager():
    cmd = "/left_forward_position_controller/commands"
    topics = {**TOPICS, cmd: topic(("/pd_node",), ("/left_forward_position_controller",), n=250)}
    _, deaf = _ports(_spec(stale_ms=500), observed(topics=topics))
    _, heard = _ports(_spec(stale_ms=500, heard_by=["/left_forward_position_controller"]), observed(topics=topics))
    assert deaf[cmd]["state"] == "off" and heard[cmd]["state"] == "live"


def test_a_muted_wire_is_off_with_its_reason_and_never_a_break():
    cmd = "/left_forward_position_controller/commands"
    topics = {**TOPICS, cmd: topic((), ("/left_forward_position_controller",), n=0, age=None)}
    out, ports = _ports(_spec(heard_by=["/left_forward_position_controller"], muted="pd 가 execute:=false 다"),
                        observed(topics=topics))
    assert ports[cmd]["state"] == "off" and "execute" in ports[cmd]["note"]
    assert cmd not in out["summary"]["text"]


def test_an_episodic_wire_rests_between_episodes():
    quiet = {**TOPICS, "/policy_control/joint_target": topic(("/pour_node",), ("/pd_node",), n=0, age=9000.0)}
    idle = {"pour_node": {**POUR, "phase": "idle"}, "pd": PD}
    _, resting = _ports(_spec(), observed(status=idle, topics=quiet))
    _, broken = _ports(_spec(), observed(topics=quiet))                       # running 인데 안 온다 — 진짜 끊김
    assert resting["/policy_control/joint_target"]["state"] == "held"
    assert "에피소드" in resting["/policy_control/joint_target"]["note"]
    assert broken["/policy_control/joint_target"]["state"] == "stale"


def test_an_episodic_wire_is_not_excused_when_its_source_went_quiet():
    quiet = {**TOPICS, "/policy_control/joint_target": topic(("/pour_node",), ("/pd_node",), n=0, age=9000.0)}
    idle = {"pour_node": {**POUR, "phase": "idle"}, "pd": PD}
    _, ports = _ports(_spec(), observed(status=idle, topics=quiet, ages={"pour_node": 30.0}))
    assert ports["/policy_control/joint_target"]["state"] == "stale"


def test_boxes_carry_their_internal_stages():
    out = build(observed(), _spec(), units={})
    boxes = {b["id"]: b for col in out["cols"] for b in col}
    assert boxes["pd"]["stages"] == ["PD 법칙", "발행"] and boxes["drive"]["stages"] == []


# ── 그림 밖 연결 ────────────────────────────────────────────────────────
def test_connections_the_picture_does_not_declare_are_listed_beside_it():
    graph = {"nodes": ["/pour_node", "/pd_node", "/rogue"],
             "topics": [{"name": "/policy_control/joint_target", "type": "sensor_msgs/msg/JointState", "pubs": ["/pour_node"], "subs": ["/pd_node"]},
                        {"name": "/extra", "type": "std_msgs/msg/String", "pubs": ["/rogue"], "subs": ["/pd_node"]},
                        {"name": "/policy_control/status/pd", "type": "std_msgs/msg/String", "pubs": ["/pd_node"], "subs": []}],
             "edges": []}
    o = Observed(bridge_up=True, status={"pour_node": POUR, "pd": PD}, age_s={"pour_node": 0.1, "pd": 0.1}, graph=GRAPH,
                 topics=TOPICS, topics_age_s=0.5, rosgraph=graph, rosgraph_age_s=1.0)
    extra = build(o, _spec(), units={})["extra"]
    assert [t["name"] for t in extra["topics"]] == ["/extra"]                 # 그림에 있는 토픽 · status 배관은 뺀다
    assert extra["topics"][0]["pubs"] == ["/rogue"] and extra["nodes"] == ["/rogue"]


def test_no_observed_graph_means_no_extra_list_rather_than_an_empty_one():
    assert build(observed(), _spec(), units={})["extra"] is None


def test_chain_topics_are_labelled_without_their_common_prefix():
    # 좁은 상자에서는 "policy_contr…" 로 잘려 obs·action·joint_target 이 구별되지 않는다.
    _, _, ports = view()
    assert ports["/policy_control/joint_target"]["label"] == "joint_target"
    assert ports["/left_forward_position_controller/commands"]["label"] == "left_forward_position_controller/commands"
    assert ports["/objects/cup/pose"]["label"] == "cup/pose"


# ── 초록이라고 말할 자격 ────────────────────────────────────────────────
def _drive_spec(**wire):
    """구동 상자가 controller_manager 를 갖는 그림 — 실기 생성 결과와 같은 모양."""
    return parse_diagram({
        "boxes": [{"id": "pour_node", "title": "정책", "col": 0, "status": "pour_node", "ros": ["/pour_node"]},
                  {"id": "pd", "title": "pd", "col": 1, "status": "pd", "ros": ["/pd_node"]},
                  {"id": "drive", "title": "팔 구동", "col": 2, "ros": ["/controller_manager"],
                   "manager": "/controller_manager"}],
        "wires": [{"from": "pour_node", "to": "pd", "topic": "/policy_control/joint_target", "stale_ms": 500},
                  {"from": "pd", "to": "drive", "topic": "/left_forward_position_controller/commands",
                   "heard_by": ["/left_forward_position_controller"], "stale_ms": 500, **wire}],
    }, path=Path("p.yaml"))


CMD = "/left_forward_position_controller/commands"
_ALL_UP = (*GRAPH, "/controller_manager")


def _drive_obs(**kw):
    topics = {**TOPICS, CMD: topic(("/pd_node",), ("/left_forward_position_controller",), n=250)}
    return observed(graph=_ALL_UP, topics=topics, **kw)


def test_summary_does_not_call_it_connected_while_a_wire_is_unknown():
    # 상자가 전부 초록이어도 브리지가 보고하지 않은 전선이 있으면 "전부 이어짐" 은 거짓이다.
    blind = {k: v for k, v in TOPICS.items() if k != CMD}
    out, _ = _ports(_drive_spec(), observed(graph=_ALL_UP, topics=blind))
    assert out["summary"]["tone"] == "warn"
    assert "pd → 팔 구동" in out["summary"]["text"]


def test_summary_stays_green_when_the_dark_wire_is_intentionally_muted():
    silent = {**TOPICS, CMD: topic((), ("/left_forward_position_controller",), n=0, age=None)}
    out, ports = _ports(_drive_spec(muted="pd 가 execute:=false 로 떠 있다"),
                        observed(graph=_ALL_UP, topics=silent))
    assert ports[CMD]["state"] == "off"
    assert out["summary"] == {"tone": "ok", "text": "전부 이어짐"}


def test_a_live_command_wire_into_an_inactive_controller_is_a_fault():
    # 구독은 configure 에서 생긴다 — inactive 컨트롤러도 구독은 한다. 명령은 버려진다.
    ctl = {"/controller_manager": {"ok": True, "controllers": [
        {"name": "left_forward_position_controller", "state": "inactive"}]}}
    out, ports = _ports(_drive_spec(), _drive_obs())
    assert ports[CMD]["state"] == "live"
    o = _drive_obs()
    _, ports = _ports(_drive_spec(), Observed(**{**o.__dict__, "controllers": ctl}))
    assert ports[CMD]["state"] == "fault" and "active" in ports[CMD]["note"]
    assert out["summary"]["tone"] == "ok"


def test_an_unreachable_controller_probe_leaves_the_wire_but_says_so():
    o = _drive_obs()
    _, ports = _ports(_drive_spec(), Observed(**{**o.__dict__, "controllers_age_s": {"/controller_manager": 99.0}}))
    assert ports[CMD]["state"] == "live" and "모른다" in ports[CMD]["note"]


def test_a_fake_listener_is_not_judged_as_a_ros2_control_controller():
    # fake 플랜트에서 heard_by 는 컨트롤러가 아니라 상자 제 노드다 — 컨트롤러 조회가 없다고 사유를 붙이면 잡음이다.
    spec = parse_diagram({
        "boxes": [{"id": "pd", "title": "pd", "col": 0, "status": "pd", "ros": ["/pd_node"]},
                  {"id": "drive", "title": "팔 구동", "col": 1, "ros": ["/fake_arm_bridge"],
                   "manager": "/controller_manager"}],
        "wires": [{"from": "pd", "to": "drive", "topic": CMD, "heard_by": ["/fake_arm_bridge"], "stale_ms": 500}],
    }, path=Path("p.yaml"))
    topics = {**TOPICS, CMD: topic(("/pd_node",), ("/fake_arm_bridge",), n=250)}
    o = observed(graph=("/pd_node", "/fake_arm_bridge"), topics=topics)
    _, ports = _ports(spec, Observed(**{**o.__dict__, "controllers": {}, "controllers_age_s": {}}))
    assert ports[CMD]["state"] == "live" and ports[CMD]["note"] == ""


def test_a_flowing_wire_beats_the_missions_claim_that_pd_is_muted():
    # muted 는 미션 argv 의 *선언*이다. 밖에서 execute:=true 로 다시 띄우면 실기가 움직이는데 그림은 "무발행" 이라고 했다.
    live = {**TOPICS, CMD: topic(("/pd_node",), ("/left_forward_position_controller",), n=250)}
    _, ports = _ports(_drive_spec(muted="pd 가 execute:=false 로 떠 있다"), observed(graph=_ALL_UP, topics=live))
    assert ports[CMD]["state"] == "live" and "흐르고 있다" in ports[CMD]["note"]


def test_a_manual_unit_never_paints_a_box_as_not_started():
    # 콘솔이 띄우지 않는 명령(수동 bringup)은 "아직 켜지지 않았다" 를 말할 근거가 없다 — 실기 구동 상자의 끝이 늘 회색이었다.
    spec = parse_diagram({
        "boxes": [{"id": "pd", "title": "pd", "col": 0, "status": "pd", "ros": ["/pd_node"]},
                  {"id": "drive", "title": "팔 구동", "col": 1, "manager": "/controller_manager", "unit": "bringup#1"}],
        "wires": [{"from": "pd", "to": "drive", "topic": CMD, "heard_by": ["/left_forward_position_controller"],
                   "stale_ms": 500}],
    }, path=Path("p.yaml"))
    manual = {"bringup#1": {"key": "bringup#1", "kind": "manual", "alive": False, "started": False, "stopped": False}}
    out = build(_drive_obs(), spec, units=manual)
    drive = next(b for col in out["cols"] for b in col if b["id"] == "drive")
    assert drive["state"] == "live", drive["detail"]      # controller_manager 가 답했다 — 그것이 증거다


def test_a_drive_box_without_a_manager_answer_stays_unknown():
    spec = parse_diagram({
        "boxes": [{"id": "pd", "title": "pd", "col": 0, "status": "pd", "ros": ["/pd_node"]},
                  {"id": "drive", "title": "팔 구동", "col": 1, "manager": "/controller_manager", "unit": "bringup#1"}],
        "wires": [{"from": "pd", "to": "drive", "topic": CMD, "heard_by": ["/left_forward_position_controller"],
                   "stale_ms": 500}],
    }, path=Path("p.yaml"))
    manual = {"bringup#1": {"key": "bringup#1", "kind": "manual", "alive": False, "started": False, "stopped": False}}
    o = _drive_obs()
    out = build(Observed(**{**o.__dict__, "controllers": {}, "controllers_age_s": {}}), spec, units=manual)
    drive = next(b for col in out["cols"] for b in col if b["id"] == "drive")
    assert drive["state"] == "unknown"


def test_a_declared_mute_only_loses_when_the_wire_really_flows():
    # fake 손은 JTC 를 받지 않는다(받는 쪽 사유). 내는 쪽이 있다고 해서 그 선언이 틀린 것이 아니다.
    deaf = {**TOPICS, CMD: topic(("/pd_node",), (), n=250)}
    _, ports = _ports(_drive_spec(muted="fake 손은 JTC 를 받지 않는다"), observed(graph=_ALL_UP, topics=deaf))
    assert ports[CMD]["state"] == "off" and ports[CMD]["note"] == "fake 손은 JTC 를 받지 않는다"


def test_when_the_topic_report_is_missing_pd_own_word_beats_the_declaration():
    # 토픽 보고가 없을 때는 선언대로 두되, pd 가 스스로 "명령 나감" 이라고 하면 그 선언은 낡은 것이다.
    blind = {k: v for k, v in TOPICS.items() if k != CMD}
    spec = _drive_spec(muted="pd 가 execute:=false 로 떠 있다")
    off = _ports(spec, observed(graph=_ALL_UP, topics=blind, status={"pour_node": POUR, "pd": {**PD, "execute": False}}))
    on = _ports(spec, observed(graph=_ALL_UP, topics=blind, status={"pour_node": POUR, "pd": {**PD, "execute": True}}))
    assert off[1][CMD]["state"] == "off"
    assert on[1][CMD]["state"] == "unknown"                  # 무발행이라고 우기지 않는다 — 모른다고 한다


def test_the_outside_list_is_not_shown_when_the_bridge_is_gone():
    o = observed(up=False)
    out = build(Observed(**{**o.__dict__, "rosgraph": {"topics": [{"name": "/x", "pubs": ["/a"], "subs": []}]},
                            "rosgraph_age_s": 1.0}), SPEC, units=UNITS)
    assert out["extra"] is None


# ── 인지: 진실원천은 vision-3090 을 보는 런처다 ────────────────────────
PERCEPT_SPEC = parse_diagram({
    "boxes": [{"id": "perception", "title": "인지 런처 · vision-3090", "col": 0, "ros": ["/perception_launcher"],
               "unit": "percept#0", "host": "vision-3090"},
              {"id": "camera", "title": "카메라 (RealSense)", "col": 1, "host": "vision-3090"},
              {"id": "fpp_cup_big_s100", "title": "FPP 추적 · cup_big_s100", "col": 2, "host": "vision-3090"},
              {"id": "object_pose", "title": "object_pose_node", "col": 3, "ros": ["/object_pose_node"]}],
    "wires": [{"from": "camera", "to": "fpp_cup_big_s100", "topic": "/camera/camera/color/image_raw", "meter": False},
              {"from": "fpp_cup_big_s100", "to": "object_pose",
               "topic": "/perception_plus_plus/cup_big_s100/pose", "stale_ms": 500}],
}, path=Path("p.yaml"))
PERCEPT_UNITS = {"percept#0": {"key": "percept#0", "kind": "background", **UP}}


def _percept(payload, age=1.0, topics=None):
    o = observed(graph=("/perception_launcher", "/object_pose_node"), topics=topics or {})
    return Observed(**{**o.__dict__, "perception": payload, "perception_age_s": age})


def _percept_boxes(payload, **kw):
    out = build(_percept(payload, **kw), PERCEPT_SPEC, units=PERCEPT_UNITS)
    return {b["id"]: b for col in out["cols"] for b in col}


RUNNING_PERCEPT = {"camera_up": True, "camera_hz": 30.0, "viewer": False, "busy": False, "error": None,
                   "objects": {"cup_big_s100": {"container": "Up 3 minutes", "pose_age_s": 0.2}}}


def test_the_camera_on_the_vision_pc_is_judged_by_the_launchers_word():
    live = _percept_boxes(RUNNING_PERCEPT)
    assert live["camera"]["state"] == "live" and "30" in live["camera"]["detail"]
    down = _percept_boxes({**RUNNING_PERCEPT, "camera_up": False})
    assert down["camera"]["state"] == "off" and "vision-3090" in down["camera"]["detail"]


def test_a_tracker_without_its_container_is_off_not_unknown():
    gone = _percept_boxes({**RUNNING_PERCEPT, "objects": {"cup_big_s100": {"container": None, "pose_age_s": None}}})
    assert gone["fpp_cup_big_s100"]["state"] == "off" and "컨테이너" in gone["fpp_cup_big_s100"]["detail"]
    up = _percept_boxes(RUNNING_PERCEPT)
    assert up["fpp_cup_big_s100"]["state"] == "live"


def test_a_remote_error_makes_the_launcher_box_a_fault():
    bad = _percept_boxes({**RUNNING_PERCEPT, "error": "ssh: connect to host vision-3090 port 22: No route to host"})
    assert bad["perception"]["state"] == "fault" and "ssh" in bad["perception"]["detail"]


def test_without_the_launcher_the_perception_boxes_say_they_are_unknown_not_fine():
    blind = _percept_boxes(None)
    assert blind["camera"]["state"] in ("unknown", "missing") and blind["fpp_cup_big_s100"]["state"] in ("unknown", "missing")

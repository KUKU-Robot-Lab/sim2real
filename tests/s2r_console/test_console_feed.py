"""브리지 봉투 → Observed."""
from __future__ import annotations

import json

from s2r_console.bridge import domain_refusal
from s2r_console.feed import BEAT_STALE_S, Feed, line


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def feed(domain=97):
    clock = Clock()
    return Feed(("obs", "pd"), expect_domain=domain, clock=clock), clock


def test_hello_brings_the_bridge_up_and_silence_takes_it_down():
    f, clock = feed()
    assert not f.bridge_up()
    f.ingest(line("hello", domain=97, nodes=["obs", "pd"]))
    assert f.bridge_up() and f.observed().bridge_faults == ()
    clock.t = BEAT_STALE_S + 0.1
    assert not f.bridge_up()
    f.ingest(line("beat", graph=["/a"]))
    assert f.bridge_up() and f.graph == ("/a",)


def test_a_bridge_on_the_wrong_domain_is_a_fault():
    f, _ = feed(domain=97)
    f.ingest(line("hello", domain=126, nodes=[]))
    assert "126" in f.observed().bridge_faults[0]


def test_a_later_good_hello_clears_the_domain_fault():
    f, _ = feed()
    f.ingest(line("hello", domain=5, nodes=[]))
    f.ingest(line("hello", domain=97, nodes=[]))
    assert f.observed().bridge_faults == ()


def test_status_is_kept_per_node_with_its_age():
    f, clock = feed()
    f.ingest(line("status", node="pd", data={"phase": "IDLE", "seq": -1}))
    clock.t = 1.5
    o = f.observed()
    assert o.status["pd"]["phase"] == "IDLE" and o.age_s["pd"] == 1.5 and "obs" not in o.status


def test_hold_is_one_event_per_distinct_reason_not_one_per_tick():
    f, _ = feed()
    hold = {"phase": "HOLD", "reasons": ["target stale"]}
    for _ in range(100):
        f.ingest(line("status", node="pd", data=hold))
    f.ingest(line("status", node="pd", data={"phase": "HOLD", "reasons": ["joint limit"]}))
    f.ingest(line("status", node="pd", data={"phase": "TRACKING", "reasons": []}))
    texts = [e["text"] for e in f.events if e["kind"] == "hold"]
    assert texts == ["pd HOLD — target stale", "pd HOLD — joint limit", "pd HOLD 해제 → TRACKING"]


def test_episode_events_are_logged_with_reasons():
    f, _ = feed()
    f.ingest(line("episode", data={"episode": 3, "event": "abort", "reasons": ["tilt 130 deg"]}))
    assert f.observed().episode["event"] == "abort"
    assert f.events[-1]["text"] == "#3 abort — tilt 130 deg"


def test_noise_on_the_pipe_is_counted_not_fatal():
    f, _ = feed()
    for junk in ("", "   ", "[WARN] rclpy something", '{"no_ch": 1}', "[1, 2]"):
        f.ingest(junk)
    assert f.bad_lines == 3 and not f.bridge_up()


def test_bridge_death_is_an_event_and_takes_the_bridge_down():
    f, _ = feed()
    f.ingest(line("hello", domain=97, nodes=[]))
    f.bridge_died("rc=1")
    assert not f.bridge_up() and f.domain is None and "rc=1" in f.events[-1]["text"]


def test_bridge_refuses_before_rclpy_when_the_env_domain_is_wrong():
    assert domain_refusal("97", 97) is None
    assert "비어" in domain_refusal(None, 97) and "비어" in domain_refusal("", 97)
    assert "126" in domain_refusal("126", 97)


def test_the_observed_ros_graph_is_kept_with_its_age():
    f, clock = feed()
    assert f.observed().rosgraph is None and f.observed().rosgraph_age_s is None
    graph = {"nodes": ["/a", "/b"], "topics": [], "edges": [{"from": "/a", "to": "/b", "topic": "/x"}]}
    f.ingest(line("rosgraph", data=graph))
    clock.t = 2.0
    got = f.observed()
    assert got.rosgraph == graph and got.rosgraph_age_s == 2.0


def test_a_malformed_ros_graph_is_ignored_not_stored():
    f, _ = feed()
    f.ingest(line("rosgraph", data=["not", "a", "mapping"]))
    assert f.observed().rosgraph is None


def test_the_perception_status_from_vision_3090_reaches_the_snapshot():
    # 카메라·FP++ 컨테이너는 저 PC 에 있다 — 런처가 1 Hz 로 내는 이 한 줄이 유일한 진실원천이다.
    f, clock = feed()
    f.ingest(json.dumps({"ch": "perception", "data": {
        "camera_up": True, "camera_hz": 30.0, "viewer": False, "busy": False, "error": None,
        "objects": {"cup_big_s100": {"container": "Up 3 minutes", "pose_age_s": 0.2}}}}))
    o = f.observed()
    assert o.perception["camera_up"] is True and o.perception_age_s == 0.0
    clock.t = 4.0
    assert f.observed().perception_age_s == 4.0

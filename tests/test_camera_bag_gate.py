"""camera_bag_recorder --gate: 에피소드 runner 상태로 scene 진입부터 done(끝)까지만 쓴다."""
from __future__ import annotations

import json

import camera_bag_recorder as C

NODES = [{"id": "go_home_start"}, {"id": "scene"}, {"id": "grasp"}, {"id": "done"}]


def st(status: str, runs: dict, eid: str = "20261121_100000", nodes=NODES) -> str:
    return json.dumps({"status": status, "episode_id": eid, "runs": runs, "nodes": nodes})


def test_closed_before_scene_even_while_earlier_node_runs():
    g = C.EpisodeGate("scene")
    g.update(st("READY", {}))
    assert not g.open
    g.update(st("RUNNING", {"go_home_start": {"state": "running"}}))
    assert not g.open and not g.finished


def test_opens_at_scene_and_finishes_at_terminal_status():
    g = C.EpisodeGate("scene")
    g.update(st("RUNNING", {"go_home_start": {"state": "done"}, "scene": {"state": "running"}}))
    assert g.open
    g.update(st("READY", {"scene": {"state": "done"}}))          # 구분 실행 사이 대기도 에피소드 안
    assert g.open
    for end in ("SUCCESS", "FAILURE", "STOPPED"):
        h = C.EpisodeGate("scene")
        h.update(st("RUNNING", {"scene": {"state": "running"}}))
        h.update(st(end, {"scene": {"state": "done"}, "done": {"state": "done"}}))
        assert h.finished and not h.open, end


def test_finished_status_before_scene_does_not_open():
    """기록기가 지난 회의 SUCCESS 상태를 받으면 열지 않는다(새 에피소드를 기다린다)."""
    g = C.EpisodeGate("scene")
    g.update(st("SUCCESS", {"scene": {"state": "done"}, "done": {"state": "done"}}, eid="old"))
    assert not g.open and not g.finished
    g.update(st("RUNNING", {"scene": {"state": "running"}}, eid="new"))
    assert g.open


def test_new_episode_id_after_open_finishes():
    g = C.EpisodeGate("scene")
    g.update(st("RUNNING", {"scene": {"state": "running"}}, eid="a"))
    g.update(st("READY", {}, eid="b"))                                  # [새 에피소드] — 이 회는 끝
    assert g.finished and not g.open


def test_episode_without_start_node_opens_at_first_node():
    nodes = [{"id": "go_home"}, {"id": "done"}]
    g = C.EpisodeGate("scene")
    g.update(st("READY", {}, nodes=nodes))
    assert not g.open
    g.update(st("RUNNING", {"go_home": {"state": "running"}}, nodes=nodes))
    assert g.open


def test_bad_json_is_ignored():
    g = C.EpisodeGate("scene")
    g.update("not json")
    assert not g.open and not g.finished

"""관측 그래프 — 브리지가 본 (토픽 → 내는 노드 · 받는 노드) 를 노드-토픽-노드 연결로 바꾼다.

정책을 모른다. 도메인에 떠 있는 것은 무엇이든 그대로 나온다.
"""
from __future__ import annotations

from s2r_console.rosgraph import is_noise_node, is_noise_topic, observe

ENDS = {
    "/objects/cup/pose": {"pub_nodes": ["/object_pose_node"], "sub_nodes": ["/obs_node"]},
    "/policy_control/obs": {"pub_nodes": ["/obs_node"], "sub_nodes": ["/policy_node", "/fabric_node"]},
}
TYPES = {"/objects/cup/pose": ["geometry_msgs/msg/PoseStamped"], "/policy_control/obs": ["std_msgs/msg/Float32MultiArray"]}


def test_every_publisher_subscriber_pair_is_one_edge():
    got = observe(ENDS, TYPES)
    assert {(e["from"], e["to"], e["topic"]) for e in got["edges"]} == {
        ("/object_pose_node", "/obs_node", "/objects/cup/pose"),
        ("/obs_node", "/policy_node", "/policy_control/obs"),
        ("/obs_node", "/fabric_node", "/policy_control/obs"),
    }


def test_nodes_are_everyone_on_either_end():
    assert observe(ENDS, TYPES)["nodes"] == ["/fabric_node", "/object_pose_node", "/obs_node", "/policy_node"]


def test_a_topic_carries_its_type_and_both_ends():
    topic = next(t for t in observe(ENDS, TYPES)["topics"] if t["name"] == "/objects/cup/pose")
    assert topic == {"name": "/objects/cup/pose", "type": "geometry_msgs/msg/PoseStamped",
                     "pubs": ["/object_pose_node"], "subs": ["/obs_node"]}


def test_a_topic_nobody_publishes_is_kept_as_dangling_not_dropped():
    got = observe({"/objects/cup/pose": {"pub_nodes": [], "sub_nodes": ["/obs_node"]}}, TYPES)
    assert got["edges"] == []
    assert got["topics"][0]["pubs"] == [] and got["topics"][0]["subs"] == ["/obs_node"]
    assert got["nodes"] == ["/obs_node"]


def test_a_topic_without_a_known_type_says_so():
    got = observe({"/x": {"pub_nodes": ["/a"], "sub_nodes": ["/b"]}}, {})
    assert got["topics"][0]["type"] is None


def test_ros_plumbing_is_not_a_connection():
    assert is_noise_topic("/rosout") and is_noise_topic("/parameter_events")
    assert is_noise_topic("/fabric_node/_action/status")
    assert not is_noise_topic("/policy_control/obs")


def test_cli_daemons_and_the_bridge_itself_are_not_nodes():
    assert is_noise_node("/_ros2cli_daemon_97_abc") and is_noise_node("/s2r_console_bridge")
    assert is_noise_node("/launch_ros_4146209")
    assert not is_noise_node("/pd_node")


def test_noise_is_dropped_from_every_list():
    ends = {**ENDS, "/rosout": {"pub_nodes": ["/obs_node"], "sub_nodes": []},
            "/policy_control/obs": {"pub_nodes": ["/obs_node"], "sub_nodes": ["/policy_node", "/_ros2cli_1"]}}
    got = observe(ends, TYPES)
    assert "/rosout" not in [t["name"] for t in got["topics"]]
    assert "/_ros2cli_1" not in got["nodes"]
    assert all(e["to"] != "/_ros2cli_1" for e in got["edges"])


def test_a_node_talking_to_itself_is_not_an_edge():
    got = observe({"/x": {"pub_nodes": ["/a"], "sub_nodes": ["/a", "/b"]}}, {})
    assert [(e["from"], e["to"]) for e in got["edges"]] == [("/a", "/b")]


def test_the_graph_is_sent_when_it_changes_or_when_it_is_due():
    from s2r_console.rosgraph import REPEAT_S, is_due

    assert is_due(None, "k", None, 0.0)
    assert not is_due("k", "k", 0.0, REPEAT_S - 0.1)
    assert is_due("k", "k", 0.0, REPEAT_S)
    assert is_due("k", "other", 0.0, 0.1)


def test_the_change_key_ignores_order_but_not_content():
    from s2r_console.rosgraph import change_key

    one = observe(ENDS, TYPES)
    same = observe(dict(reversed(list(ENDS.items()))), TYPES)
    less = observe({"/objects/cup/pose": ENDS["/objects/cup/pose"]}, TYPES)
    assert change_key(one) == change_key(same) != change_key(less)

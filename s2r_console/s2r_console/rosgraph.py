"""관측 그래프 — 브리지가 본 (토픽 → 내는 노드 · 받는 노드) 를 노드-토픽-노드 연결로 바꾼다. 순수.

정책을 모른다. 프로파일 그림(`diagram_spec`)은 "있어야 하는 것" 이고 이것은 "지금 있는 것" 이다.
둘을 겹치면 정책을 바꿔도 그림이 따라온다 — 도메인에 뜬 노드는 선언이 없어도 그대로 나온다.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

#: ROS 자체 배관 — 연결로 그리면 모든 노드가 서로 이어져 보인다
NOISE_TOPICS = ("/rosout", "/parameter_events", "/clock")
NOISE_TOPIC_PARTS = ("/_action/", "/transition_event")
#: 도구가 띄우는 노드 — 스택이 아니다
NOISE_NODE_PREFIXES = ("/_ros2cli", "/launch_ros_", "/s2r_console_bridge", "/rqt_", "/rviz")


def is_noise_topic(name: str) -> bool:
    return name in NOISE_TOPICS or any(part in name for part in NOISE_TOPIC_PARTS)


def is_noise_node(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in NOISE_NODE_PREFIXES)


def _clean(names) -> list[str]:
    return sorted({str(n) for n in names or () if not is_noise_node(str(n))})


def _first_type(types: Mapping, topic: str) -> str | None:
    found = types.get(topic)
    if not found:
        return None
    return str(found) if isinstance(found, str) else str(found[0])


def observe(ends: Mapping[str, Mapping], types: Mapping) -> dict:
    """`ends` = 토픽 → {"pub_nodes", "sub_nodes"}. `types` = 토픽 → 타입 이름(들).

    돌려주는 것: nodes(정렬) · topics(타입·양끝) · edges(내는 노드 → 받는 노드, 토픽별).
    내는 쪽이 없는 토픽은 버리지 않는다 — 끊긴 자리가 바로 보여야 한다.
    """
    topics = [
        {"name": name, "type": _first_type(types, name),
         "pubs": _clean(row.get("pub_nodes")), "subs": _clean(row.get("sub_nodes"))}
        for name, row in sorted(ends.items()) if not is_noise_topic(name)
    ]
    edges = [
        {"from": pub, "to": sub, "topic": topic["name"]}
        for topic in topics for pub in topic["pubs"] for sub in topic["subs"] if pub != sub
    ]
    nodes = sorted({n for topic in topics for n in (*topic["pubs"], *topic["subs"])})
    return {"nodes": nodes, "topics": topics, "edges": edges}


#: 바뀌지 않아도 이만큼마다 다시 보낸다 — 콘솔이 "언제 본 그림인가" 를 알 수 있어야 한다
REPEAT_S = 5.0


def change_key(graph: Mapping) -> str:
    """같은 연결이면 같은 값. `observe` 가 정렬해서 내므로 그대로 직렬화하면 된다."""
    return json.dumps(graph, sort_keys=True, ensure_ascii=False)


def is_due(sent_key: str | None, key: str, sent_at: float | None, now: float) -> bool:
    return sent_key != key or sent_at is None or now - sent_at >= REPEAT_S

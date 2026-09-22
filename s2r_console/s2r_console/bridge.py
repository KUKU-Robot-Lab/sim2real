"""ROS 브리지 — **구독만** 하는 별도 프로세스. 본 것을 stdout 에 NDJSON 으로 낸다(`feed.py`).

구조로 보장하는 것:
  · 우리 코드의 publisher 0, service client 0, rosout·parameter 서비스도 끈다 → 이 프로세스는 명령을 내지 않는다.
    (rclpy 가 노드마다 `/parameter_events` publisher 를 만드는 것은 못 막는다 — 기동 때 use_sim_time 선언 이벤트
     한 건이 나간다. 로봇에 가는 토픽은 아니지만 "DDS 에 아무것도 안 쓴다" 는 아니다.)
  · `--topics` 의 스택 토픽은 **raw** 로 받아 세기만 한다(역직렬화 없음). 타입은 그래프에서 읽는다.
  · `--watch` 의 토픽은 구독도 하지 않는다 — 그래프에서 누가 내고 누가 받는지만 읽는다.
  · 도메인 전체의 연결(`rosgraph`)도 그래프 조회만으로 만든다 — 정책·프로파일을 몰라도 뜬 것은 다 보인다.
  · env 의 `ROS_DOMAIN_ID` 가 `--domain` 과 다르면 rclpy 를 init 하기 **전에** 거부한다.
  · 콜백은 `json.loads` → 한 줄 쓰기뿐이다. 무거운 일은 콘솔 프로세스가 한다.

콘솔(API) 프로세스는 rclpy 를 import 하지 않는다 — DDS 참가자가 아니면 도메인을 잘못 골라도 쓸 수 없다.

    python3 -m s2r_console.bridge --domain 97 --nodes pour_node pd pour_guard --topics /joint_states
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence

from .feed import line
from .rosgraph import change_key, is_due, is_noise_topic, observe

NS = "/policy_control"


def _emit(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def domain_refusal(env_value: str | None, want: int) -> str | None:
    """왜 뜨면 안 되는가. 떠도 되면 None. 순수."""
    if env_value is None or env_value == "":
        return f"ROS_DOMAIN_ID 가 비어 있다 — 프로파일 도메인 {want} 을 명시해서 띄워야 한다"
    if env_value != str(want):
        return f"env ROS_DOMAIN_ID={env_value} 가 프로파일 도메인 {want} 과 다르다"
    return None


class TopicMeter:
    """토픽별 도착 수와 마지막 도착 시각. 순수 — 시각은 받는다. `report` 가 구간 수를 비운다."""

    def __init__(self, topics: Sequence[str], watch: Sequence[str] = ()) -> None:
        self.topics = tuple(topics)
        #: 구독하지 않고 그래프만 보는 토픽 — 센 것이 없으므로 n·age_ms 는 0 이 아니라 None 이다
        self.watch = tuple(w for w in watch if w not in self.topics)
        self._n = {t: 0 for t in self.topics}
        self._last: dict[str, float] = {}

    @property
    def all(self) -> tuple[str, ...]:
        return self.topics + self.watch

    def hit(self, topic: str, now: float) -> None:
        self._n[topic] += 1
        self._last[topic] = now

    def report(self, now: float, *, pubs: Mapping[str, int], why: Mapping[str, str] | None = None,
               ends: Mapping[str, Mapping] | None = None) -> dict:
        """`ends` = 토픽 → {"pub_nodes", "sub_nodes"} (노드 전체 이름). 그림이 "받는 쪽이 붙어 있는가" 를 본다."""
        out = {}
        for t in self.all:
            last = self._last.get(t)
            row = {"pubs": int(pubs.get(t, 0)), "n": self._n.get(t),
                   "age_ms": None if last is None else round((now - last) * 1e3, 1)}
            row = {**row, **ends[t]} if ends and t in ends else row
            out[t] = {**row, "why": why[t]} if why and t in why else row
        self._n = {t: 0 for t in self.topics}
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", type=int, required=True)
    ap.add_argument("--nodes", nargs="+", required=True)
    ap.add_argument("--topics", nargs="*", default=[])
    ap.add_argument("--watch", nargs="*", default=[])
    ap.add_argument("--perception", default="", help="인지 런처 상태 토픽(std_msgs/String JSON) — 그림에 인지 상자가 있을 때만")
    args = ap.parse_args(argv)

    refusal = domain_refusal(os.environ.get("ROS_DOMAIN_ID"), args.domain)
    if refusal:
        _emit(line("fault", reason=refusal))
        return 2

    import rclpy  # noqa: PLC0415 — 거부 검사 뒤에 올린다
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String

    rclpy.init(domain_id=args.domain)
    node = rclpy.create_node("s2r_console_bridge", enable_rosout=False, start_parameter_services=False)

    def on_status(name: str):
        def cb(msg: String) -> None:
            try:
                _emit(line("status", node=name, data=json.loads(msg.data)))
            except ValueError:
                pass
        return cb

    def on_episode(msg: String) -> None:
        try:
            _emit(line("episode", data=json.loads(msg.data)))
        except ValueError:
            pass

    def on_perception(msg: String) -> None:
        # vision-3090 의 카메라·FP++ 컨테이너 상태. 런처가 ssh 로 읽어 1 Hz 로 낸다 — 여기서는 받기만 한다.
        try:
            _emit(line("perception", data=json.loads(msg.data)))
        except ValueError:
            pass

    meter = TopicMeter(args.topics, args.watch)
    me = node.get_fully_qualified_name()

    def ends(topic: str) -> dict:
        """누가 내고 누가 받는가. 이 브리지 자신의 구독은 뺀다 — 세려고 붙은 것이지 받는 쪽이 아니다."""
        full = lambda e: e.node_namespace.rstrip("/") + "/" + e.node_name  # noqa: E731
        return {"pub_nodes": sorted({full(e) for e in node.get_publishers_info_by_topic(topic)}),
                "sub_nodes": sorted({full(e) for e in node.get_subscriptions_info_by_topic(topic)} - {me})}
    subscribed: set[str] = set()
    sent: dict = {"key": None, "at": None}

    def send_rosgraph(types: Mapping[str, Sequence[str]]) -> None:
        """도메인에 보이는 모든 토픽의 양끝. 구독하지 않는다 — 그래프 조회뿐이다."""
        graph = observe({t: ends(t) for t in types if not is_noise_topic(t)}, types)
        key, now = change_key(graph), time.monotonic()
        if is_due(sent["key"], key, sent["at"], now):
            sent.update(key=key, at=now)
            _emit(line("rosgraph", data=graph))
    why: dict[str, str] = {}

    def on_raw(topic: str):
        def cb(_: bytes) -> None:
            meter.hit(topic, time.monotonic())
        return cb

    def subscribe_new() -> None:
        """타입은 publisher 가 뜬 뒤에야 그래프에 보인다 — 보이는 대로 붙는다."""
        types = dict(node.get_topic_names_and_types())
        for t in meter.topics:
            if t in subscribed or not types.get(t):
                continue
            try:
                node.create_subscription(get_message(types[t][0]), t, on_raw(t), qos_profile_sensor_data, raw=True)
            except (ImportError, AttributeError, ValueError) as exc:
                why[t] = f"타입 {types[t][0]} 을 못 올린다: {exc}"
                continue
            subscribed.add(t)
            why.pop(t, None)

    def beat() -> None:
        names = sorted({(ns.rstrip("/") + "/" + n) for n, ns in node.get_node_names_and_namespaces()})
        _emit(line("beat", graph=names))
        send_rosgraph(dict(node.get_topic_names_and_types()))
        if meter.all:
            subscribe_new()
            pubs = {t: node.count_publishers(t) for t in meter.all}
            _emit(line("topics", data=meter.report(time.monotonic(), pubs=pubs, why=why,
                                                   ends={t: ends(t) for t in meter.all})))

    chain = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    for n in args.nodes:
        node.create_subscription(String, f"{NS}/status/{n}", on_status(n), chain)
    node.create_subscription(String, f"{NS}/episode", on_episode, latched)
    if args.perception:
        node.create_subscription(String, args.perception, on_perception, chain)
    node.create_timer(1.0, beat)

    _emit(line("hello", domain=args.domain, nodes=list(args.nodes)))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, BrokenPipeError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

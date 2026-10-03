"""rclpy 노드 CPU 줄이기 — 10.03 실측(cProfile): 파이썬 노드 CPU 의 대부분이 executor 가 깨어날 때마다 wait set 을
엔티티 하나하나로 다시 짜는 비용(`_wait_for_ready_callbacks` 66 %)이었다. 깨어나는 횟수 × 엔티티 수에 비례한다.

여기서 줄이는 것(동작은 같다):
  · 발행자 · 구독자마다 붙는 기본 QoS 이벤트 핸들러(incompatible QoS 경고용) — 엔티티마다 하나씩 wait set 에 들어간다
  · 파라미터 서비스 6 개(describe/get/set …) — 이 노드들은 시작할 때 파라미터를 읽기만 하고 런타임에 바꾸지 않는다

    class PdNode(LeanNodeMixin, Node):
        def __init__(...):
            super().__init__(NAME, **lean_node_kwargs(), context=context, ...)
"""
from __future__ import annotations


def lean_node_kwargs() -> dict:
    """Node.__init__ 에 넘길 인자 — 파라미터 서비스 끔(--ros-args -p 로 주는 값은 그대로 읽힌다)."""
    return {"start_parameter_services": False}


class LeanNodeMixin:
    """create_publisher / create_subscription 에 기본 QoS 이벤트 핸들러를 붙이지 않는다(호출자가 따로 주면 그대로)."""

    def create_publisher(self, *args, **kwargs):
        if "event_callbacks" not in kwargs:
            from rclpy.qos_event import PublisherEventCallbacks
            kwargs["event_callbacks"] = PublisherEventCallbacks(use_default_callbacks=False)
        return super().create_publisher(*args, **kwargs)

    def create_subscription(self, *args, **kwargs):
        if "event_callbacks" not in kwargs:
            from rclpy.qos_event import SubscriptionEventCallbacks
            kwargs["event_callbacks"] = SubscriptionEventCallbacks(use_default_callbacks=False)
        return super().create_subscription(*args, **kwargs)


def make_lean_node(name: str, **kwargs):
    """함수형 노드용 — `Node(name)` 대신. rclpy 를 부를 때만 import 한다."""
    from rclpy.node import Node

    class _LeanNode(LeanNodeMixin, Node):
        pass

    return _LeanNode(name, **{**lean_node_kwargs(), **kwargs})

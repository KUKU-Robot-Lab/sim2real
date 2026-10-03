"""상태 토픽을 실행기 없이 틱에서 꺼낸다 — rclpy 실행기가 메시지마다 깨어나는 비용을 없앤다.

10.04 실측(실기 bag 을 fake 도메인에서 재생 · py-spy): pd 프로세스 CPU 의 대부분이 메시지가 올 때마다 executor 가
wait set 을 다시 짜는 비용(`_wait_for_ready_callbacks`, 메인 스레드 시간의 약 84 %)이었고 제어 계산(틱 · 명령 · 상태)은
약 6 % 였다. 팔 250 Hz + 손 250 Hz + 온도 10 Hz 를 실행기에서 빼면 초당 깨어나는 횟수가 그만큼 준다.

방법: 실행기에 넣지 않는 노드(`make_poll_node`)에 raw 구독을 만들고, 틱마다 `take_last` 로 쌓인 것을 비우며 마지막 것만
쓴다. DDS 리더는 스핀과 상관없이 받아 둔다(QoS depth 만큼, 오래된 것부터 버림). 받은 시각은 rmw 가 적은
received_timestamp(시스템 시계)를 단조 시계로 옮긴다 — 낡음 판정이 예전(콜백에서 time.monotonic())과 같은 뜻이다.
★rclpy Humble 의 `Subscription.handle.take_message` 를 쓴다(executor 가 쓰는 것과 같은 호출). rclpy 를 올리면 여기부터 본다.
"""
from __future__ import annotations

import time


def make_poll_node(name: str, context=None):
    """실행기에 넣지 않을 노드 — 전역 인자(`-r __node:=…` · 파라미터)를 받지 않아 본 노드와 이름이 겹치지 않는다."""
    from .lean_node import make_lean_node

    return make_lean_node(name, context=context, use_global_arguments=False)


def _noop(_msg) -> None:
    """실행기에 안 들어가므로 불리지 않는다."""


def poll_subscription(node, msg_type, topic: str, qos):
    """raw(직렬 바이트) 구독 — 틱에서 take_last 로 꺼낸다."""
    return node.create_subscription(msg_type, topic, _noop, qos, raw=True)


def received_to_monotonic(received_ns: int, now_mono: float | None = None, now_wall_ns: int | None = None) -> float:
    """rmw received_timestamp(시스템 시계 ns) → time.monotonic() 기준 초. 0 · 미래 값은 '지금'으로 본다. 순수."""
    now_mono = time.monotonic() if now_mono is None else now_mono
    now_wall_ns = time.time_ns() if now_wall_ns is None else now_wall_ns
    if received_ns <= 0:
        return now_mono
    age = (now_wall_ns - int(received_ns)) / 1e9
    return now_mono - age if age > 0 else now_mono


def take_all(sub) -> list[tuple[bytes, float]]:
    """쌓인 메시지를 모두(오래된 것부터) — 한 토픽에 여러 출처가 섞여 오는 명령(양팔 joint_target)용."""
    out = []
    while True:
        with sub.handle:
            got = sub.handle.take_message(sub.msg_type, sub.raw)
        if got is None:
            return out
        data, info = got
        received = info.get("received_timestamp", 0) if isinstance(info, dict) else 0
        out.append((data, received_to_monotonic(received)))


def take_last(sub) -> tuple[bytes, float] | None:
    """쌓인 메시지를 모두 비우고 마지막 것(직렬 바이트, 받은 시각 — 단조 초)을 돌려준다. 없으면 None."""
    last = None
    while True:
        with sub.handle:
            got = sub.handle.take_message(sub.msg_type, sub.raw)
        if got is None:
            break
        last = got
    if last is None:
        return None
    data, info = last
    received = info.get("received_timestamp", 0) if isinstance(info, dict) else 0
    return data, received_to_monotonic(received)

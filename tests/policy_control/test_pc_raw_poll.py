"""raw_poll — 상태 토픽을 실행기 없이 틱에서 꺼낸다(10.04 pd CPU). 받은 시각 변환은 순수, take_last 는 실제 rclpy(도메인 96 · localhost)."""
from __future__ import annotations

import os
import time

import pytest

from policy_control import raw_poll as R


def test_received_time_maps_to_the_monotonic_clock():
    assert R.received_to_monotonic(9_000_000_000, now_mono=100.0, now_wall_ns=10_000_000_000) == pytest.approx(99.0)
    assert R.received_to_monotonic(0, now_mono=100.0, now_wall_ns=10_000_000_000) == 100.0          # 정보 없음 = 지금
    assert R.received_to_monotonic(11_000_000_000, now_mono=100.0, now_wall_ns=10_000_000_000) == 100.0  # 미래 = 지금


@pytest.fixture
def ros_ctx(monkeypatch):
    rclpy = pytest.importorskip("rclpy")
    monkeypatch.setenv("ROS_DOMAIN_ID", "96")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    ctx = rclpy.Context()
    rclpy.init(context=ctx)
    yield ctx
    rclpy.shutdown(context=ctx)


def test_take_last_drains_the_queue_and_keeps_only_the_newest(ros_ctx):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    pub_node = rclpy.create_node("raw_poll_test_pub", context=ros_ctx)
    poll = R.make_poll_node("raw_poll_test_poll", context=ros_ctx)
    try:
        topic = f"/raw_poll_test_{os.getpid()}"
        pub = pub_node.create_publisher(JointState, topic, qos_profile_sensor_data)
        sub = R.poll_subscription(poll, JointState, topic, qos_profile_sensor_data)
        deadline = time.monotonic() + 5.0
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert R.take_last(sub) is None                                  # 아직 아무것도 없다
        t_before = time.monotonic()
        for i in range(4):
            pub.publish(JointState(name=["a"], position=[float(i)]))
            time.sleep(0.003)
        time.sleep(0.1)
        got = R.take_last(sub)
        assert got is not None
        data, t_recv = got
        assert deserialize_message(data, JointState).position[0] == 3.0   # 마지막 것
        assert t_before - 0.05 <= t_recv <= time.monotonic()               # 받은 시각 = 실제로 받은 때(단조 시계)
        assert R.take_last(sub) is None                                    # 다 비웠다
    finally:
        poll.destroy_node()
        pub_node.destroy_node()


def test_the_poll_node_ignores_global_remaps_so_names_do_not_collide(ros_ctx):
    poll = R.make_poll_node("raw_poll_name_check", context=ros_ctx)
    try:
        assert poll.get_name() == "raw_poll_name_check"
    finally:
        poll.destroy_node()


def test_take_all_keeps_every_message_in_order_for_mixed_sources(ros_ctx):
    """양팔 목표가 한 토픽에 섞여 온다 — 틱 사이 두 팔 것이 모두 와도 하나도 버리지 않는다."""
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    pub_node = rclpy.create_node("raw_poll_all_pub", context=ros_ctx)
    poll = R.make_poll_node("raw_poll_all_poll", context=ros_ctx)
    try:
        topic = f"/raw_poll_all_{os.getpid()}"
        pub = pub_node.create_publisher(JointState, topic, qos)
        sub = R.poll_subscription(poll, JointState, topic, qos)
        deadline = time.monotonic() + 5.0
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        for side in ("r", "l", "r", "l"):
            pub.publish(JointState(name=[f"{side}_aj_1"], position=[1.0]))
        time.sleep(0.1)
        got = R.take_all(sub)
        assert [deserialize_message(d, JointState).name[0] for d, _ in got] == ["r_aj_1", "l_aj_1", "r_aj_1", "l_aj_1"]
        assert all(t1 <= t2 + 1e-6 for (_, t1), (_, t2) in zip(got, got[1:]))
        assert R.take_all(sub) == []
    finally:
        poll.destroy_node()
        pub_node.destroy_node()

#!/usr/bin/env python3
"""실기 관절 상태 -> Isaac 뷰어 UDP 중계. **구독만 한다**(로봇 명령 발행 0).

ROS 2 Humble(py3.10) 쪽 프로세스. Isaac Sim(py3.11) 과 섞이지 않게 UDP 로만 넘긴다.

    source /opt/ros/humble/setup.bash
    ROS_DOMAIN_ID=126 python3 joint_state_relay.py --port 47811

- 구독: /joint_states(팔) · /dg5f_right/joint_states · /dg5f_left/joint_states · /head/joint_states(있으면)
- 이름 사상: robot_control 프로필 `openarm_tesollo.yaml` 의 source->canonical·sign (joint_map.py)
- 송신: 127.0.0.1:<port> 로 `{t, names, positions}` JSON 을 --rate Hz(기본 30)
- ROS_DOMAIN_ID 가 비었거나 0 이면 거부한다(기본 도메인에서 엉뚱한 그래프에 붙지 않게).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_map  # noqa: E402
import packet  # noqa: E402

DEFAULT_TOPICS = ("/joint_states", "/dg5f_right/joint_states", "/dg5f_left/joint_states", "/head/joint_states")
#: rclpy(Humble) Node 가 **항상** 만드는 내부 발행자. 로봇 명령 토픽이 아니다.
#: /rosout 은 enable_rosout=False 로 끈다. /parameter_events 는 Humble rclpy 가 조건 없이 만든다.
ALLOWED_INTERNAL_PUBLISHERS = frozenset({"/parameter_events"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
STATUS_PERIOD_S = 5.0


def require_ros_domain(environ: Mapping[str, str]) -> int:
    """ROS_DOMAIN_ID 가 1 이상 정수일 때만 통과. 비었거나 0 이면 SystemExit."""
    raw = environ.get("ROS_DOMAIN_ID", "").strip()
    if not raw:
        raise SystemExit("[relay] ROS_DOMAIN_ID 가 비어 있다 — 실기 도메인(예: 126)을 명시할 것. 거부.")
    try:
        dom = int(raw)
    except ValueError:
        raise SystemExit(f"[relay] ROS_DOMAIN_ID={raw!r} 가 정수가 아니다. 거부.") from None
    if dom <= 0:
        raise SystemExit(f"[relay] ROS_DOMAIN_ID={dom} — 0(기본 도메인)/음수는 거부한다.")
    return dom


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=packet.DEFAULT_HOST, help="UDP 목적지(루프백만 허용)")
    ap.add_argument("--port", type=int, default=packet.DEFAULT_PORT)
    ap.add_argument("--rate", type=float, default=30.0, help="송신 Hz")
    ap.add_argument("--profile", default=str(joint_map.DEFAULT_PROFILE), help="robot_control 관절 프로필 yaml")
    ap.add_argument("--topics", nargs="+", default=list(DEFAULT_TOPICS), help="구독할 JointState 토픽")
    ap.add_argument("--stale_sec", type=float, default=1.0,
                    help="이 시간보다 오래 갱신 없는 토픽의 관절은 송신에서 뺀다(뷰어는 마지막 값 유지)")
    args = ap.parse_args(argv)
    if args.host not in LOOPBACK_HOSTS:
        ap.error(f"--host 는 루프백만 허용: {sorted(LOOPBACK_HOSTS)}")
    if not (1.0 <= args.rate <= 200.0):
        ap.error("--rate 는 1~200 Hz")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    domain = require_ros_domain(os.environ)
    table = joint_map.load_profile_table(args.profile)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    # BEST_EFFORT 구독은 RELIABLE/BEST_EFFORT 발행자 모두와 맞는다(발행 측에 영향 없음).
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

    class Relay(Node):
        def __init__(self) -> None:
            super().__init__("isaac_viewer_joint_relay", enable_rosout=False, start_parameter_services=False)
            self._sock = socket.socket(socket.AF_INET6 if args.host == "::1" else socket.AF_INET, socket.SOCK_DGRAM)
            self._dest = (args.host, args.port)
            self._per_topic: dict[str, tuple[float, Mapping[str, float]]] = {}
            self._unknown_seen: set[str] = set()
            self._sent = 0
            self._msgs = {t: 0 for t in args.topics}
            self._last_status = time.monotonic()
            for topic in args.topics:
                self.create_subscription(JointState, topic, lambda m, t=topic: self._on_js(t, m), qos)
            self.create_timer(1.0 / args.rate, self._tick)

        def _on_js(self, topic: str, msg) -> None:
            try:
                res = joint_map.map_joint_state(list(msg.name), list(msg.position), table)
            except ValueError as exc:
                self.get_logger().warning(f"{topic}: {exc}")
                return
            new_unknown = set(res.unknown) - self._unknown_seen
            if new_unknown:
                self._unknown_seen |= new_unknown
                self.get_logger().warning(f"{topic}: 프로필에 없는 관절 이름 무시 {sorted(new_unknown)}")
            if res.non_finite:
                self.get_logger().warning(f"{topic}: NaN/inf 위치 무시 {list(res.non_finite)}")
            self._msgs[topic] += 1
            prev = self._per_topic.get(topic, (0.0, {}))[1]
            self._per_topic = {**self._per_topic, topic: (time.monotonic(), joint_map.merge(prev, res.values))}

        def _tick(self) -> None:
            now = time.monotonic()
            joints: dict[str, float] = {}
            for _topic, (stamp, vals) in self._per_topic.items():
                if now - stamp <= args.stale_sec:
                    joints.update(vals)
            if joints:
                try:
                    self._sock.sendto(packet.encode(time.time(), joints), self._dest)
                    self._sent += 1
                except (OSError, ValueError) as exc:
                    self.get_logger().warning(f"UDP 송신 실패: {exc}")
            if now - self._last_status >= STATUS_PERIOD_S:
                self._last_status = now
                ages = {t: (f"{now - s:.2f}s" if t in self._per_topic else "-")
                        for t, s in ((t, self._per_topic.get(t, (0.0, None))[0]) for t in args.topics)}
                self.get_logger().info(f"sent={self._sent} joints={len(joints)} msgs={self._msgs} age={ages}")

        def assert_subscribe_only(self) -> None:
            pubs = {name for name, _types in self.get_publisher_names_and_types_by_node(self.get_name(),
                                                                                      self.get_namespace())}
            extra = pubs - ALLOWED_INTERNAL_PUBLISHERS
            if extra:
                raise RuntimeError(f"[relay] 구독 전용 노드인데 발행자가 있다: {sorted(extra)}")
            self.get_logger().info(f"발행자 점검 OK — 로봇 토픽 발행 0 (rclpy 내부 {sorted(pubs)} 만)")

    rclpy.init()
    node = Relay()
    try:
        # 그래프 조회가 자기 자신을 보려면 discovery 가 한 바퀴 돌아야 한다.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.assert_subscribe_only()
        node.get_logger().info(f"ROS_DOMAIN_ID={domain} · 구독 {args.topics} -> udp://{args.host}:{args.port} "
                               f"@{args.rate:.0f}Hz · 프로필 {args.profile} ({len(table)} 관절)")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

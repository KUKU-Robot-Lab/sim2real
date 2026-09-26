#!/usr/bin/env python3
"""로봇 PC — vision-3090 의 fpp_pose_tx 가 보낸 UDP 를 받아 FP++ 자세를 ROS 로 다시 낸다. 형식은 scripts/fpp_udp.py.

    python3 fpp_pose_rx.py [--port 51126]

  /perception_plus_plus/<물체>/pose  (geometry_msgs/PoseStamped, 카메라 프레임) — FP++ 가 내던 토픽 그대로.
                                       object_pose_node 가 base_link 로 바꾼다(바뀐 것 없음).
  /perception/camera_hz              (std_msgs/Float32, 2 Hz) — heartbeat 가 끊기면 0. 인지 런처가 읽는다.

형식이 틀리거나 모르는 물체 · 옛 seq 인 패킷은 **내지 않고** 센다 — 이 자세는 정책 입력이다.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fpp_udp as U  # noqa: E402
from object_registry import input_topic, load_registry  # noqa: E402

REPORT_S = 10.0
RECV_TIMEOUT_S = 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--port", type=int, default=U.PORT)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    names = load_registry().names()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from std_msgs.msg import Float32

    class PoseRx(Node):
        def __init__(self) -> None:
            super().__init__("fpp_pose_rx")
            self.pubs = {n: self.create_publisher(PoseStamped, input_topic(n), 10) for n in names}
            self.hz_pub = self.create_publisher(Float32, U.CAMERA_HZ_TOPIC, 10)
            self.gate = U.SeqGate()
            self.lock = threading.Lock()
            self.heartbeat: tuple[float, float] | None = None
            self.published = {n: 0 for n in names}
            self.rejected = 0
            self.last_reject = ""
            self.sender = ""
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((args.bind, args.port))
            self.sock.settimeout(RECV_TIMEOUT_S)
            self.running = True
            threading.Thread(target=self._loop, name="fpp-udp-rx", daemon=True).start()
            self.create_timer(U.HEARTBEAT_S, self._publish_hz)
            self.create_timer(REPORT_S, self._report)
            self.get_logger().info(f"← udp {args.bind}:{args.port} · 물체 {', '.join(names)}")

        def _loop(self) -> None:
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(U.MAX_BYTES + 1)
                except socket.timeout:
                    continue
                except OSError:
                    return                                  # 소켓이 닫혔다(종료)
                self._handle(data, addr)

        def _handle(self, data: bytes, addr) -> None:
            try:
                packet = U.decode(data, names)
            except ValueError as exc:
                with self.lock:
                    self.rejected += 1
                    self.last_reject = str(exc)[:120]
                return
            with self.lock:
                self.sender = f"{addr[0]}:{addr[1]}"
                key = "hb" if isinstance(packet, U.Heartbeat) else packet.name
                if not self.gate.accept(key, packet.seq):
                    return
                if isinstance(packet, U.Heartbeat):
                    self.heartbeat = (time.monotonic(), packet.camera_hz)
                    return
                self.published[packet.name] += 1
            msg = PoseStamped()
            msg.header.frame_id = packet.frame
            msg.header.stamp.sec, msg.header.stamp.nanosec = packet.stamp
            (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) = packet.p
            (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
             msg.pose.orientation.w) = packet.q
            self.pubs[packet.name].publish(msg)

        def _publish_hz(self) -> None:
            with self.lock:
                hz = U.camera_hz_at(self.heartbeat, time.monotonic())
            self.hz_pub.publish(Float32(data=float(hz)))

        def _report(self) -> None:
            with self.lock:
                pub = " · ".join(f"{n} {c}" for n, c in self.published.items() if c)
                hz = U.camera_hz_at(self.heartbeat, time.monotonic())
                line = (f"송신 {self.sender or '없음'} · 카메라 {hz:.1f} Hz · 냄 {pub or '0'} · 버림 {self.rejected}"
                        f"{' (' + self.last_reject + ')' if self.last_reject else ''}")
            self.get_logger().info(line)

        def close(self) -> None:
            self.running = False
            self.sock.close()

    rclpy.init()
    node = PoseRx()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

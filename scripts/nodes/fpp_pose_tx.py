#!/usr/bin/env python3
"""vision-3090 — FP++ 자세(카메라 프레임)와 카메라 fps 를 로봇 PC 로 UDP 로 보낸다. 형식은 scripts/fpp_udp.py.

    ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=126 python3 fpp_pose_tx.py --dest 100.103.21.126
    (보통은 scripts/vision/pose_tx_up.sh 가 띄운다 — 인지 런처가 ssh 로 부른다)

카메라 · FP++ 와 같은 localhost 전용 DDS 안에서 구독하고, 밖으로는 UDP 만 낸다. 소켓은 non-blocking 이다 —
링크가 막히면 그 패킷을 버리고 센다(카메라 · FP++ 를 기다리게 하지 않는다. 09.26 에 막힌 것이 그것이었다).
레지스트리의 물체 전부를 구독한다 — 어떤 FP++ 컨테이너가 떠 있든 이 송신기는 하나로 충분하다.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fpp_udp as U  # noqa: E402
from object_registry import input_topic, load_registry  # noqa: E402

CAMERA_INFO = "/camera/camera/color/camera_info"
REPORT_S = 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--dest", required=True, help="로봇 PC 주소(런처가 ssh 접속 주소로 넘긴다)")
    ap.add_argument("--port", type=int, default=U.PORT)
    args = ap.parse_args()
    names = load_registry().names()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo

    class PoseTx(Node):
        def __init__(self) -> None:
            super().__init__("fpp_pose_tx")
            self.dest = (args.dest, args.port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.seq = 0
            self.sent = {n: 0 for n in names}
            self.dropped = 0
            self.last_error = ""
            self.camera = U.RateMeter()
            for n in names:
                self.create_subscription(PoseStamped, input_topic(n), lambda m, n=n: self._on_pose(n, m), 10)
            self.create_subscription(CameraInfo, CAMERA_INFO, lambda _m: self.camera.tick(time.monotonic()),
                                     qos_profile_sensor_data)
            self.create_timer(U.HEARTBEAT_S, self._heartbeat)
            self.create_timer(REPORT_S, self._report)
            self.get_logger().info(f"→ udp {args.dest}:{args.port} · 물체 {', '.join(names)}")

        def _send(self, packet) -> bool:
            try:
                self.sock.sendto(U.encode(packet), self.dest)
                return True
            except (BlockingIOError, OSError) as exc:          # 링크가 막히거나 끊겼다 — 기다리지 않고 버린다
                self.dropped += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                return False

        def _next(self) -> int:
            self.seq += 1
            return self.seq

        def _on_pose(self, name: str, msg) -> None:
            p, q = msg.pose.position, msg.pose.orientation
            packet = U.PosePacket(seq=self._next(), name=name, frame=msg.header.frame_id or "camera",
                                  stamp=(int(msg.header.stamp.sec), int(msg.header.stamp.nanosec)),
                                  p=(p.x, p.y, p.z), q=(q.x, q.y, q.z, q.w))
            if self._send(packet):
                self.sent[name] += 1

        def _heartbeat(self) -> None:
            self._send(U.Heartbeat(seq=self._next(), camera_hz=self.camera.hz(time.monotonic())))

        def _report(self) -> None:
            sent = " · ".join(f"{n} {c}" for n, c in self.sent.items() if c)
            self.get_logger().info(f"카메라 {self.camera.hz(time.monotonic()):.1f} Hz · 보냄 {sent or '0'} · "
                                   f"버림 {self.dropped}{' (' + self.last_error + ')' if self.last_error else ''}")

    rclpy.init()
    node = PoseTx()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

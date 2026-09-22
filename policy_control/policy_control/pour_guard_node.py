"""pour 안전망 노드 — 두 컵 포즈만 보고 위험하면 `episode/abort` 를 부른다.

판정은 전부 `pour_guard.GuardCore` 가 한다. 이 파일은 구독과 서비스 호출만 하는 껍질이다.

**발행자를 하나도 만들지 않는다.** status 한 줄만 내고, 로봇을 움직이는 경로는 `episode/abort`
Trigger 하나뿐이다 — 그 서비스는 에피소드를 끝낼 뿐 새 지령을 내지 않는다. 그래서 이 노드는
`execute` 와 무관하게 언제나 켜 두어도 된다(오히려 켜 두는 것이 목적이다).

    ros2 run policy_control pour_guard_node --ros-args \\
        -p src_cup_topic:=/objects/cup_src/pose -p rcv_cup_topic:=/objects/cup_rcv/pose
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):          # `python pour_guard_node.py` — pour_node 와 같은 전문
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import _paths  # noqa: F401,E402  (side effect: sibling trees on sys.path)
from policy_control.pour_guard import GuardCore, GuardInput, GuardLimits  # noqa: E402

NS = "/policy_control"
NODE = "pour_guard"

try:
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except ImportError:                                  # 테스트는 코어만 본다
    Node = object                                    # type: ignore[misc,assignment]


class PourGuardNode(Node):                           # pragma: no cover - 배선 계층
    def __init__(self, core: GuardCore | None = None) -> None:
        super().__init__(NODE)
        for name, default in (("src_cup_topic", ""), ("rcv_cup_topic", ""), ("rate_hz", 20.0),
                              ("tilt_abort_deg", GuardLimits().tilt_abort_deg),
                              ("cross_margin_m", GuardLimits().cross_margin_m),
                              ("palm_min_dist_m", GuardLimits().palm_min_dist_m),
                              ("stale_sec", GuardLimits().stale_s),
                              ("abort", True)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        topics = {"src": str(p("src_cup_topic")), "rcv": str(p("rcv_cup_topic"))}
        if not all(topics.values()):
            raise ValueError("src_cup_topic 과 rcv_cup_topic 이 필요하다 (PoseStamped, base_link)")

        self.core = core or GuardCore(GuardLimits(
            tilt_abort_deg=float(p("tilt_abort_deg")), cross_margin_m=float(p("cross_margin_m")),
            palm_min_dist_m=float(p("palm_min_dist_m")), stale_s=float(p("stale_sec"))))
        self._abort_enabled = bool(p("abort"))
        self._cups: dict[str, tuple[float, tuple, tuple]] = {}
        self._fired = False

        self._pub_status = self.create_publisher(String, f"{NS}/status/{NODE}", QoSProfile(depth=10))
        for role, topic in topics.items():
            self.create_subscription(PoseStamped, topic, self._cup_cb(role), qos_profile_sensor_data)
        self.create_subscription(String, f"{NS}/episode", self._on_episode, QoSProfile(depth=10))
        self._abort = self.create_client(Trigger, f"{NS}/episode/abort")
        self.create_timer(1.0 / max(1.0, float(p("rate_hz"))), self._tick)
        self.get_logger().info(
            f"pour_guard up · cups {topics} · tilt>{self.core.limits.tilt_abort_deg:.0f}deg · "
            f"cross>{self.core.limits.cross_margin_m * 100:.0f}cm · abort={self._abort_enabled}")

    def _cup_cb(self, role: str):
        def cb(msg):
            q = msg.pose.orientation
            # 컵의 위쪽 축 = 회전행렬 3열. 쿼터니언에서 그 열만 뽑는다.
            up = (2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x),
                  1 - 2 * (q.x * q.x + q.y * q.y))
            pos = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            self._cups[role] = (time.monotonic(), pos, up)
        return cb

    def _on_episode(self, msg) -> None:
        try:
            event = json.loads(msg.data).get("event")
        except json.JSONDecodeError:
            return
        if event == "reset":
            self.core.reset()
            self._fired = False

    def _tick(self) -> None:
        now = time.monotonic()
        src, rcv = self._cups.get("src"), self._cups.get("rcv")
        g = GuardInput(t=now,
                       src_cup_pos=src[1] if src else None, src_cup_up=src[2] if src else None,
                       rcv_cup_pos=rcv[1] if rcv else None,
                       src_cup_t=src[0] if src else None, rcv_cup_t=rcv[0] if rcv else None)
        reasons = self.core.check(g)
        self._pub_status.publish(String(data=json.dumps(
            {"node": NODE, "ok": not reasons, "reasons": reasons, "latched": self.core.latched,
             "armed": self._abort_enabled, "t_ns": self.get_clock().now().nanoseconds},
            ensure_ascii=False)))
        if reasons and self._abort_enabled and not self._fired:
            self._fired = True                      # 에피소드당 한 번 — 서비스 폭주를 막는다
            self.get_logger().error("guard abort: " + "; ".join(reasons))
            if self._abort.service_is_ready():
                self._abort.call_async(Trigger.Request())
            else:
                self.get_logger().error("episode/abort 가 없다 — 사람이 멈춰야 한다")


def main(argv=None) -> None:                         # pragma: no cover
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=argv)
    node = PourGuardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # SIGINT/SIGTERM 은 정상 종료다
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":                           # pragma: no cover
    main()

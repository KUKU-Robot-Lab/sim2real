"""RH56F1 손 상태 → rad JointState · 손끝 힘. 구독 · 발행만(명령을 내지 않는다).

09.29 사용자: rh56f1 제어 연결. 벤더 드라이버(robot_control inspire_rh56f1)는 각도 레지스터(0.1°, 슬롯 순 새끼부터)만 낸다.
pd · 정책 · 콘솔은 rad JointState 를 읽는다 — 이 노드가 변환표(config/rh56f1_hand_map.yaml)로 바꾼다.

  구독  /hand_<side>/angle_actual (rh56f1_interfaces/GetAngleAct1)  ·  /hand_<side>/touch_data (TouchData1)
  발행  /hand_<side>/joint_states (sensor_msgs/JointState, canonical 이름, 구동 6 + 종속 6(mimic), rad · rad/s)
        /hand_<side>/tip_forces   (std_msgs/Float64MultiArray, 5, sim 순 엄지 → 새끼, N — √(법선² + 접선²), 단위 실측 전)

    python3 deploy/policy_control/policy_control/rh56f1_state_node.py --side right
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):              # 파일 경로로 띄울 때(미션 명령) — colcon 진입점을 새로 만들지 않는다
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import rh56f1_map  # noqa: E402

#: 속도 = 위치 차분의 지수 평균 — 드라이버 50 Hz 에서 잡음 줄이기(α 가 클수록 최근 값)
VEL_ALPHA = 0.3


@dataclass
class HandStateCore:
    """레지스터 → (이름, 위치, 속도). 순수 — ROS 없이 테스트한다."""

    hmap: rh56f1_map.HandMap
    side: str
    alpha: float = VEL_ALPHA
    _last: tuple[float, np.ndarray] | None = field(default=None, init=False)
    _vel: np.ndarray | None = field(default=None, init=False)

    def names(self) -> list[str]:
        p = self.side[0]
        order = list(self.hmap.joint_order) + [m[0] for m in self.hmap.mimic]
        return [f"{p}_hj_{n}" for n in order]

    def on_angle(self, reg: Sequence[float], t: float) -> tuple[list[str], np.ndarray, np.ndarray]:
        q6 = self.hmap.to_rad(reg, self.side)                     # 이 손의 보정(09.30 스윕)
        full = self.hmap.with_mimic(q6)
        pos = np.array([full[n.split("_hj_", 1)[1]] for n in self.names()])
        if self._last is not None and t > self._last[0]:
            v = (pos - self._last[1]) / (t - self._last[0])
            self._vel = v if self._vel is None else self.alpha * v + (1.0 - self.alpha) * self._vel
        self._last = (t, pos)
        vel = np.zeros_like(pos) if self._vel is None else self._vel
        return self.names(), pos, vel

    def tip_forces(self, finger_forces: Sequence[float], tangentials: Sequence[float] | None = None) -> np.ndarray:
        return self.hmap.touch_sim_order(finger_forces, tangentials)


def main(argv: list[str] | None = None) -> int:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from rh56f1_interfaces.msg import GetAngleAct1, TouchData1

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--map", default=str(rh56f1_map.DEFAULT_PATH), help="변환표 yaml")
    args, _ = ap.parse_known_args(argv)
    core = HandStateCore(rh56f1_map.load(args.map), args.side)
    ns = f"/hand_{args.side}"

    rclpy.init()
    node = Node(f"rh56f1_state_{args.side}")
    js_pub = node.create_publisher(JointState, f"{ns}/joint_states", 10)
    tip_pub = node.create_publisher(Float64MultiArray, f"{ns}/tip_forces", 10)

    def on_angle(msg) -> None:
        stamp = node.get_clock().now()
        names, pos, vel = core.on_angle(list(msg.joint_values), stamp.nanoseconds * 1e-9)
        out = JointState()
        out.header.stamp = stamp.to_msg()
        out.name, out.position, out.velocity = names, pos.tolist(), vel.tolist()
        js_pub.publish(out)

    def on_touch(msg) -> None:
        tip_pub.publish(Float64MultiArray(data=core.tip_forces(list(msg.finger_forces), list(msg.finger_tangentials)).tolist()))

    node.create_subscription(GetAngleAct1, f"{ns}/angle_actual", on_angle, 10)
    node.create_subscription(TouchData1, f"{ns}/touch_data", on_touch, 10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

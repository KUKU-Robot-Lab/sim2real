#!/usr/bin/env python3
"""RH56F1 손 브리지: Isaac Sim 명령 ↔ inspire_control_ros2 드라이버.

기존 `bridge_node`(OpenArm 팔 + Tesollo 손)와 별개로, RH56F1 손만 담당한다.
RH56F1 세팅에서는 `bridge_node`(팔) + 이 노드(양손)를 함께 띄운다.

흐름:
    /isaacsim/{side}_hand_cmd (Float64MultiArray[6], 라디안, SIM_DRIVE_SUFFIXES 순서)
        → 레지스터 변환 → SetAngle1 → /hand_{side}/angle_set
    /hand_{side}/angle_actual (GetAngleAct1, 레지스터 int[6])
        → 라디안 변환 → JointState(canonical *_hj_*) → merged_hand_states_topic

`rh56f1_interfaces`(SetAngle1/GetAngleAct1)는 지연 import한다. 없으면 상태/명령
채널은 비활성화되지만 노드는 계속 돈다.
"""

from __future__ import annotations

from collections import OrderedDict

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

from isaacsim_bridge.rh56f1_hand import (
    HAND_DRIVE_DOF,
    HandCalibration,
    default_hand_calibration,
    hand_calibration_from_config,
)

try:  # 드라이버 인터페이스 패키지는 inspire_ws 빌드 후에만 존재한다.
    from rh56f1_interfaces.msg import GetAngleAct1, SetAngle1

    _INTERFACES_AVAILABLE = True
except ImportError:  # pragma: no cover - 빌드 환경 의존
    GetAngleAct1 = None
    SetAngle1 = None
    _INTERFACES_AVAILABLE = False


_SIDE_PREFIX = {"right": "r_hj_", "left": "l_hj_"}
_DEFAULT_HAND_ID = {"right": 2, "left": 1}


def _load_calibration(side: str, config_path: str) -> tuple[HandCalibration, int]:
    """config yaml에서 side 손의 캘리브레이션과 hand_id를 읽는다.

    yaml이 없거나 해당 side가 없으면 기본값을 쓴다.
    반환: (HandCalibration, hand_id).
    """
    prefix = _SIDE_PREFIX[side]
    if config_path:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        entry = data.get(side)
        if entry and entry.get("joints"):
            cal = hand_calibration_from_config(prefix, entry["joints"])
            return cal, int(entry.get("hand_id", _DEFAULT_HAND_ID[side]))
    return default_hand_calibration(prefix), _DEFAULT_HAND_ID[side]


class _HandChannel:
    """한 손의 명령/상태 배선."""

    def __init__(self, node: "Rh56f1HandBridgeNode", side: str, config_path: str) -> None:
        self._node = node
        self._side = side
        self._cal, self._hand_id = _load_calibration(side, config_path)

        self._cmd_pub = None
        if _INTERFACES_AVAILABLE:
            self._cmd_pub = node.create_publisher(
                SetAngle1, f"/hand_{side}/angle_set", 10
            )
            node.create_subscription(
                GetAngleAct1, f"/hand_{side}/angle_actual", self._state_cb, 10
            )

        node.create_subscription(
            Float64MultiArray, f"/isaacsim/{side}_hand_cmd", self._cmd_cb, 10
        )

    def _cmd_cb(self, msg: Float64MultiArray) -> None:
        if self._node.emergency_stop_active:
            return
        if self._cmd_pub is None:
            self._node.get_logger().warning(
                f"{self._side}_hand_cmd received but rh56f1_interfaces is unavailable; dropped"
            )
            return
        rads = list(msg.data)
        if len(rads) != HAND_DRIVE_DOF:
            self._node.get_logger().warning(
                f"{self._side}_hand_cmd expected {HAND_DRIVE_DOF} values, got {len(rads)}"
            )
            return
        regs = self._cal.sim_to_registers(rads)
        out = SetAngle1()
        out.header.stamp = self._node.get_clock().now().to_msg()
        out.hand_id = self._hand_id
        out.joint_values = [int(v) for v in regs]
        self._cmd_pub.publish(out)

    def _state_cb(self, msg) -> None:
        regs = list(msg.joint_values)
        if len(regs) != HAND_DRIVE_DOF:
            return
        self._node.update_hand_state(self._cal.registers_to_sim(regs))


class Rh56f1HandBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("rh56f1_hand_bridge")

        self.declare_parameter("hands", ["right", "left"])
        self.declare_parameter("calibration_config", "")
        self.declare_parameter("merged_hand_states_topic", "/rh56f1/joint_states")

        self.emergency_stop_active = False
        self._state_cache: OrderedDict[str, float] = OrderedDict()

        self._state_pub = self.create_publisher(
            JointState, self.get_parameter("merged_hand_states_topic").value, 10
        )
        self.create_subscription(
            Bool, "/isaacsim/emergency_stop", self._emergency_stop_cb, 10
        )

        config_path = self.get_parameter("calibration_config").value
        hands = list(self.get_parameter("hands").value)
        self._channels = [_HandChannel(self, side, config_path) for side in hands]

        if not _INTERFACES_AVAILABLE:
            self.get_logger().warning(
                "rh56f1_interfaces not found; command/state to the driver are disabled. "
                "Build sim2real/vendor/inspire_ws and source its install."
            )
        self.get_logger().info(
            f"RH56F1 hand bridge ready for hands={hands}. "
            "Input: /isaacsim/{side}_hand_cmd (Float64MultiArray[6], rad)."
        )

    def _emergency_stop_cb(self, msg: Bool) -> None:
        self.emergency_stop_active = bool(msg.data)

    def update_hand_state(self, joint_to_rad: dict[str, float]) -> None:
        for name, rad in joint_to_rad.items():
            self._state_cache[name] = rad
        merged = JointState()
        merged.header.stamp = self.get_clock().now().to_msg()
        merged.name = list(self._state_cache.keys())
        merged.position = [self._state_cache[n] for n in merged.name]
        self._state_pub.publish(merged)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Rh56f1HandBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

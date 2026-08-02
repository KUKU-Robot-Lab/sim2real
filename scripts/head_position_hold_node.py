#!/usr/bin/env python3
"""ROS 2 node for unwrapped signed absolute-position hold of two DYNAMIXEL XC330 motors.

Angle frame:
  -180.0 deg -> unwrapped encoder tick 0
     0.0 deg -> raw encoder tick ~2048
  +180.0 deg -> unwrapped encoder tick 4095
  +215.0 deg -> unwrapped encoder tick >4095, same physical position as -145.0 deg

This is intentionally different from `head_goto_hold.py`, which uses a
URDF joint frame. This node is for unwrapped signed absolute DYNAMIXEL encoder
positioning with Extended Position Mode, so a range like 145..215 can move
monotonically across the encoder seam.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


TICK_MAX = 4095
PROTOCOL = 2.0

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

OP_MODE_EXTENDED_POSITION = 4
TORQUE_OFF = 0
TORQUE_ON = 1

DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "head_dynamixel_calibration.yaml"
)


def deg_to_tick(deg: float) -> int:
    """Convert unwrapped signed encoder degrees to an extended-position tick."""
    return round((float(deg) + 180.0) / 360.0 * TICK_MAX)


def tick_to_deg(tick: int) -> float:
    """Convert an extended-position tick to unwrapped signed absolute degrees."""
    return int(tick) / TICK_MAX * 360.0 - 180.0


def parse_ids(text: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(ids) != 2:
        raise ValueError(f"ids must contain exactly two motor ids: {text!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"motor ids must be unique: {ids}")
    for dxl_id in ids:
        if not 0 <= dxl_id <= 252:
            raise ValueError(f"motor id must be in 0..252: {dxl_id}")
    return ids


def parse_angle_targets(ids: tuple[int, ...], text: str) -> dict[int, float]:
    angles = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(angles) != len(ids):
        raise ValueError(
            f"angles_deg count must match ids count: ids={ids}, angles={text!r}"
        )
    return dict(zip(ids, angles))


def unwrap_calibrated_range(min_deg: float, max_deg: float) -> tuple[float, float]:
    """Return the directed min->max arc as monotonically increasing degrees."""
    start = float(min_deg)
    end = float(max_deg)
    if end < start:
        end += 360.0
    if end - start > 360.0:
        raise ValueError(
            f"calibrated range must fit within one directed turn: "
            f"{min_deg:.1f}..{max_deg:.1f}"
        )
    return start, end


def unwrap_target_into_range(target_deg: float, min_deg: float, max_deg: float) -> float:
    start, end = unwrap_calibrated_range(min_deg, max_deg)
    target = float(target_deg)
    while target < start:
        target += 360.0
    while target > end:
        target -= 360.0
    if not start <= target <= end:
        raise ValueError(
            f"target {target_deg:+.1f}deg is outside calibrated range "
            f"{min_deg:.1f}..{max_deg:.1f}"
        )
    return target


def unwrap_range_copy_for_current(
    current_deg: float, min_deg: float, max_deg: float
) -> tuple[float, float]:
    start, end = unwrap_calibrated_range(min_deg, max_deg)
    current = float(current_deg)
    for turn in range(-20, 21):
        copy_start = start + 360.0 * turn
        copy_end = end + 360.0 * turn
        if copy_start <= current <= copy_end:
            return copy_start, copy_end
    turn = round((current - start) / 360.0)
    return start + 360.0 * turn, end + 360.0 * turn


def unwrap_target_for_current(
    current_deg: float, target_deg: float, min_deg: float, max_deg: float
) -> float:
    copy_start, copy_end = unwrap_range_copy_for_current(
        current_deg, min_deg, max_deg
    )
    target = float(target_deg)
    while target < copy_start:
        target += 360.0
    while target > copy_end:
        target -= 360.0
    if not copy_start <= target <= copy_end:
        raise ValueError(
            f"target {target_deg:+.1f}deg is outside current directed range "
            f"{copy_start:.1f}..{copy_end:.1f}"
        )
    return target


@dataclass(frozen=True)
class MotorCalibration:
    name: str
    dxl_id: int
    min_deg: float
    max_deg: float
    inverted: bool


@dataclass(frozen=True)
class CalibrationFile:
    port: str
    baud: int
    motors: dict[str, MotorCalibration]


def motor_deg_from_center_offset(calibration: MotorCalibration, offset_deg: float) -> float:
    """Validate a signed absolute encoder target within the calibrated range."""
    try:
        motor_deg = unwrap_target_into_range(
            float(offset_deg),
            calibration.min_deg,
            calibration.max_deg,
        )
    except ValueError as exc:
        raise ValueError(
            f"{calibration.name} command {offset_deg:+.1f}deg is outside "
            f"calibrated range {calibration.min_deg:.1f}..{calibration.max_deg:.1f}"
        ) from exc
    deg_to_tick(motor_deg)
    return motor_deg


def parse_offsets(names: tuple[str, ...], text: str) -> dict[str, float]:
    offsets = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(offsets) != len(names):
        raise ValueError(
            f"offsets_deg count must match motor count: names={names}, offsets={text!r}"
        )
    return dict(zip(names, offsets))


def load_calibration_yaml(path: Path) -> CalibrationFile:
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"calibration file must contain a mapping: {path}")
    motors_data = data.get("motors")
    if not isinstance(motors_data, dict):
        raise ValueError(f"calibration file must contain motors mapping: {path}")

    motors: dict[str, MotorCalibration] = {}
    for name, raw in motors_data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"motor {name!r} must contain a mapping")
        motors[str(name)] = MotorCalibration(
            name=str(name),
            dxl_id=int(raw["id"]),
            min_deg=float(raw["min_deg"]),
            max_deg=float(raw["max_deg"]),
            inverted=bool(raw.get("inverted", False)),
        )
    return CalibrationFile(
        port=str(data.get("port", "/dev/ttyUSB0")),
        baud=int(data.get("baud", 1_000_000)),
        motors=motors,
    )


def targets_from_calibration_offsets(
    calibration: CalibrationFile,
    offsets_deg: str,
) -> dict[int, float]:
    names = tuple(calibration.motors)
    offsets = parse_offsets(names, offsets_deg)
    return {
        calibration.motors[name].dxl_id: motor_deg_from_center_offset(
            calibration.motors[name], offsets[name]
        )
        for name in names
    }


@dataclass(frozen=True)
class HoldConfig:
    port: str
    baud: int
    targets_deg: dict[int, float]
    ranges_deg: dict[int, tuple[float, float]] | None
    hold: bool
    profile_acceleration: int
    profile_velocity: int
    tolerance_deg: float
    timeout_s: float


class DynamixelPositionController:
    """Small SDK adapter for absolute position hold."""

    def __init__(self, port_name: str, baud: int):
        from dynamixel_sdk import PacketHandler, PortHandler

        self.port_name = port_name
        self.port = PortHandler(port_name)
        self.packet = PacketHandler(PROTOCOL)
        if not self.port.openPort():
            raise RuntimeError(f"failed to open port: {port_name}")
        try:
            if not self.port.setBaudRate(baud):
                raise RuntimeError(f"failed to set baud: {baud}")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.port.closePort()

    def _check(self, comm_result: int, packet_error: int, context: str) -> None:
        from dynamixel_sdk import COMM_SUCCESS

        if comm_result != COMM_SUCCESS:
            raise RuntimeError(f"{context}: {self.packet.getTxRxResult(comm_result)}")
        if packet_error:
            raise RuntimeError(f"{context}: {self.packet.getRxPacketError(packet_error)}")

    def ping(self, dxl_id: int) -> None:
        _model, comm, err = self.packet.ping(self.port, dxl_id)
        self._check(comm, err, f"ping id={dxl_id}")

    def write1(self, dxl_id: int, address: int, value: int, label: str) -> None:
        comm, err = self.packet.write1ByteTxRx(self.port, dxl_id, address, value)
        self._check(comm, err, f"{label} id={dxl_id}")

    def write4(self, dxl_id: int, address: int, value: int, label: str) -> None:
        comm, err = self.packet.write4ByteTxRx(self.port, dxl_id, address, value)
        self._check(comm, err, f"{label} id={dxl_id}")

    def read_present_tick(self, dxl_id: int) -> int:
        value, comm, err = self.packet.read4ByteTxRx(
            self.port, dxl_id, ADDR_PRESENT_POSITION
        )
        self._check(comm, err, f"present position id={dxl_id}")
        return int(value)

    def configure_position_mode(
        self, dxl_id: int, profile_acceleration: int, profile_velocity: int
    ) -> None:
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
        self.write1(
            dxl_id,
            ADDR_OPERATING_MODE,
            OP_MODE_EXTENDED_POSITION,
            "operating mode",
        )
        self.write4(
            dxl_id,
            ADDR_PROFILE_ACCELERATION,
            int(profile_acceleration),
            "profile acceleration",
        )
        self.write4(
            dxl_id,
            ADDR_PROFILE_VELOCITY,
            int(profile_velocity),
            "profile velocity",
        )
        self.write4(
            dxl_id,
            ADDR_GOAL_POSITION,
            self.read_present_tick(dxl_id),
            "seed goal position",
        )
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON, "torque on")

    def write_goal_tick(self, dxl_id: int, tick: int) -> None:
        self.write4(dxl_id, ADDR_GOAL_POSITION, tick, "goal position")

    def torque_off(self, dxl_id: int) -> None:
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")


def goto_absolute_hold(config: HoldConfig, controller: DynamixelPositionController) -> bool:
    ids = tuple(config.targets_deg)
    tolerance_tick = max(1, deg_to_tick(config.tolerance_deg) - deg_to_tick(0.0))

    for dxl_id in ids:
        controller.ping(dxl_id)
    for dxl_id in ids:
        controller.configure_position_mode(
            dxl_id, config.profile_acceleration, config.profile_velocity
        )
    target_ticks = {}
    for dxl_id, angle in config.targets_deg.items():
        if config.ranges_deg and dxl_id in config.ranges_deg:
            min_deg, max_deg = config.ranges_deg[dxl_id]
            current_deg = tick_to_deg(controller.read_present_tick(dxl_id))
            angle = unwrap_target_for_current(current_deg, angle, min_deg, max_deg)
        target_ticks[dxl_id] = deg_to_tick(angle)
    for dxl_id, tick in target_ticks.items():
        controller.write_goal_tick(dxl_id, tick)

    deadline = time.time() + config.timeout_s
    reached = False
    while time.time() < deadline:
        if all(
            abs(controller.read_present_tick(dxl_id) - tick) <= tolerance_tick
            for dxl_id, tick in target_ticks.items()
        ):
            reached = True
            break
        time.sleep(0.05)

    if not config.hold:
        for dxl_id in ids:
            controller.torque_off(dxl_id)
    return reached


def main() -> None:
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("head_position_hold_node")
    node.declare_parameter("port", "/dev/ttyUSB0")
    node.declare_parameter("baud", 1_000_000)
    node.declare_parameter("ids", "1,2")
    node.declare_parameter("angles_deg", "0,0")
    node.declare_parameter("calibration_file", "")
    node.declare_parameter("offsets_deg", "0,0")
    node.declare_parameter("hold", True)
    node.declare_parameter("profile_acceleration", 20)
    node.declare_parameter("profile_velocity", 50)
    node.declare_parameter("tolerance_deg", 1.0)
    node.declare_parameter("timeout", 5.0)

    controller = None
    try:
        calibration_file = str(node.get_parameter("calibration_file").value)
        if calibration_file:
            calibration = load_calibration_yaml(Path(calibration_file))
            names = tuple(calibration.motors)
            offsets = parse_offsets(names, str(node.get_parameter("offsets_deg").value))
            targets = {}
            ranges = {}
            for name in names:
                motor = calibration.motors[name]
                targets[motor.dxl_id] = motor_deg_from_center_offset(
                    motor, offsets[name]
                )
                ranges[motor.dxl_id] = (motor.min_deg, motor.max_deg)
            port = calibration.port
            baud = calibration.baud
        else:
            ids = parse_ids(node.get_parameter("ids").value)
            targets = parse_angle_targets(ids, node.get_parameter("angles_deg").value)
            ranges = None
            port = str(node.get_parameter("port").value)
            baud = int(node.get_parameter("baud").value)
        config = HoldConfig(
            port=port,
            baud=baud,
            targets_deg=targets,
            ranges_deg=ranges,
            hold=bool(node.get_parameter("hold").value),
            profile_acceleration=int(node.get_parameter("profile_acceleration").value),
            profile_velocity=int(node.get_parameter("profile_velocity").value),
            tolerance_deg=float(node.get_parameter("tolerance_deg").value),
            timeout_s=float(node.get_parameter("timeout").value),
        )
        controller = DynamixelPositionController(config.port, config.baud)
        reached = goto_absolute_hold(config, controller)
        for dxl_id, angle in targets.items():
            node.get_logger().info(
                f"id={dxl_id} target={angle:.1f}deg tick={deg_to_tick(angle)}"
            )
        if reached:
            if config.hold:
                node.get_logger().info("target reached; torque remains enabled")
            else:
                node.get_logger().info("target reached; torque disabled because hold=false")
        else:
            node.get_logger().warning("target timeout; torque state preserved")
        rclpy.spin(node)
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        if controller is not None:
            controller.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

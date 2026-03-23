"""Real Tesollo fingertip force/torque protocol helpers.

This module is intentionally standalone so it can be reused from:
    - Isaac Sim Script Editor prototypes
    - ROS 2 bridge nodes
    - Isaac Lab debug/export code
"""

from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class TesolloFTProtocol:
    sensor_code: int = 0x05
    force_limit_n: float = 30.0
    torque_limit_nm: float = 250.0
    force_resolution_n: float = 0.1
    torque_resolution_nm: float = 0.1
    byte_order: str = "big"
    data_order: tuple[str, ...] = ("fx", "fy", "fz", "tx", "ty", "tz")

    @property
    def force_scale(self) -> float:
        return 1.0 / self.force_resolution_n

    @property
    def torque_scale(self) -> float:
        return 1.0 / self.torque_resolution_nm


DEFAULT_PROTOCOL = TesolloFTProtocol()


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def quantize_force(value_n: float, protocol: TesolloFTProtocol = DEFAULT_PROTOCOL) -> int:
    return int(round(clamp(value_n, protocol.force_limit_n) * protocol.force_scale))


def quantize_torque(value_nm: float, protocol: TesolloFTProtocol = DEFAULT_PROTOCOL) -> int:
    return int(round(clamp(value_nm, protocol.torque_limit_nm) * protocol.torque_scale))


def pack_wrench_be_i16(
    fx: float,
    fy: float,
    fz: float,
    tx: float,
    ty: float,
    tz: float,
    protocol: TesolloFTProtocol = DEFAULT_PROTOCOL,
) -> bytes:
    """Pack one F/T sample as big-endian signed 16-bit integers.

    If your real interface uses a different integer width, change the format
    string here and keep the quantization logic unchanged.
    """

    if protocol.byte_order != "big":
        raise ValueError(f"Unsupported byte order: {protocol.byte_order}")

    return struct.pack(
        ">6h",
        quantize_force(fx, protocol),
        quantize_force(fy, protocol),
        quantize_force(fz, protocol),
        quantize_torque(tx, protocol),
        quantize_torque(ty, protocol),
        quantize_torque(tz, protocol),
    )

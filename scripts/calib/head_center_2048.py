#!/usr/bin/env python3
"""목(pan/tilt) 다이나믹셀을 공식 중앙(틱 2048)에 세우는 재조립용 도구.

규약(2026-11-15 사용자 결정):
  - 영점 = 다이나믹셀 공식 중앙 틱 2048. Homing Offset 0. 각도 = (틱 - 2048) * 360/4096.
  - 방향 = 공식 기본(Drive Mode 0) — 혼 쪽에서 볼 때 반시계가 +.
  - 모터를 2048 에 토크로 잡아 둔 채 브라켓/혼을 다시 끼워 카메라가 정면·수평을 보게 맞춘다.

단계 (기본은 읽기만 한다):
    python head_center_2048.py                       # 상태 읽기만
    python head_center_2048.py --zero-offset         # 토크 off 확인 후 Homing Offset 0 쓰기 (움직임 없음)
    python head_center_2048.py --hold pan            # pan 을 천천히 2048 로 보내고 토크 유지
    python head_center_2048.py --hold tilt           # tilt 도 같은 방식 (브라켓을 뗀 뒤에)
    python head_center_2048.py --trim                # 토크 유지한 채 I 게인으로 정확히 2048
    python head_center_2048.py --release             # 두 모터 토크 off
"""

from __future__ import annotations

import argparse
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT763P8T-if00-port0"
BAUD = 1_000_000
IDS = {"pan": 1, "tilt": 2}
CENTER = 2048
TICKS_PER_TURN = 4096

ADDR_DRIVE_MODE = 10
ADDR_OPERATING_MODE = 11
ADDR_HOMING_OFFSET = 20
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_I_GAIN = 82
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_MOVING = 122
ADDR_PRESENT_POSITION = 132

OP_MODE_POSITION = 3
PROFILE_ACCELERATION = 5   # 느리게 (단위 214.577 rev/min^2)
PROFILE_VELOCITY = 15      # 약 3.4 rpm (단위 0.229 rpm)
ARRIVE_TOL_TICK = 1
# ★11.15 실측: pan(ID 1) Present Position 은 2048 을 보고하지 않는다 — 목표 2046/2047 → 2047, 2048 → 2049
#   (위·아래 접근 같음, I 게인 400~4000 무관 = 컨트롤러 오차 0). 그래서 판정은 '목표 레지스터 = 2048' 이고 읽기값은 ±1 틱 허용.
POSITION_I_GAIN = 400       # config/head_home_rh56f1.yaml 과 같은 값 (RAM — 전원 끄면 사라짐)
TIMEOUT_S = 20.0


def signed32(value: int) -> int:
    return value - (1 << 32) if value & 0x8000_0000 else value


def deg(tick: int) -> float:
    return round((tick - CENTER) * 360.0 / TICKS_PER_TURN, 2)


class Bus:
    def __init__(self) -> None:
        self.port = PortHandler(PORT)
        if not self.port.openPort() or not self.port.setBaudRate(BAUD):
            raise SystemExit(f"포트를 못 열었다: {PORT}")
        self.pk = PacketHandler(2.0)

    def _check(self, what: str, result: int, error: int) -> None:
        if result != COMM_SUCCESS:
            raise SystemExit(f"{what}: {self.pk.getTxRxResult(result)}")
        if error:
            raise SystemExit(f"{what}: {self.pk.getRxPacketError(error)}")

    def read(self, dxl_id: int, addr: int, size: int) -> int:
        fn = {1: self.pk.read1ByteTxRx, 2: self.pk.read2ByteTxRx, 4: self.pk.read4ByteTxRx}[size]
        value, result, error = fn(self.port, dxl_id, addr)
        self._check(f"읽기 id={dxl_id} addr={addr}", result, error)
        return signed32(value) if size == 4 else value

    def write(self, dxl_id: int, addr: int, size: int, value: int) -> None:
        fn = {1: self.pk.write1ByteTxRx, 2: self.pk.write2ByteTxRx, 4: self.pk.write4ByteTxRx}[size]
        raw = value & 0xFFFF_FFFF if size == 4 else value
        result, error = fn(self.port, dxl_id, addr, raw)
        self._check(f"쓰기 id={dxl_id} addr={addr} value={value}", result, error)

    def close(self) -> None:
        self.port.closePort()


def status(bus: Bus) -> dict[str, dict[str, int | float]]:
    out = {}
    for name, dxl_id in IDS.items():
        present = bus.read(dxl_id, ADDR_PRESENT_POSITION, 4)
        out[name] = {
            "drive_mode": bus.read(dxl_id, ADDR_DRIVE_MODE, 1),
            "op_mode": bus.read(dxl_id, ADDR_OPERATING_MODE, 1),
            "homing_offset": bus.read(dxl_id, ADDR_HOMING_OFFSET, 4),
            "torque": bus.read(dxl_id, ADDR_TORQUE_ENABLE, 1),
            "i_gain": bus.read(dxl_id, ADDR_POSITION_I_GAIN, 2),
            "present": present,
            "present_deg": deg(present),
        }
    return out


def zero_offset(bus: Bus) -> None:
    for name, dxl_id in IDS.items():
        if bus.read(dxl_id, ADDR_TORQUE_ENABLE, 1):
            raise SystemExit(f"{name}: 토크가 켜져 있다 — EEPROM 은 토크 off 에서만 쓴다 (--release 먼저)")
    for name, dxl_id in IDS.items():
        before = bus.read(dxl_id, ADDR_PRESENT_POSITION, 4)
        bus.write(dxl_id, ADDR_HOMING_OFFSET, 4, 0)
        time.sleep(0.05)
        offset = bus.read(dxl_id, ADDR_HOMING_OFFSET, 4)
        after = bus.read(dxl_id, ADDR_PRESENT_POSITION, 4)
        if offset != 0:
            raise SystemExit(f"{name}: Homing Offset 읽기값 {offset} ≠ 0")
        print(f"{name}: offset → 0, 현재 틱 {before} → {after} ({deg(after):+.2f}°)")


def hold(bus: Bus, name: str) -> None:
    dxl_id = IDS[name]
    if bus.read(dxl_id, ADDR_HOMING_OFFSET, 4) != 0:
        raise SystemExit(f"{name}: Homing Offset 이 0 이 아니다 — --zero-offset 먼저")
    start = bus.read(dxl_id, ADDR_PRESENT_POSITION, 4)
    print(f"{name}: 시작 틱 {start} ({deg(start):+.2f}°) → 목표 {CENTER}, 이동 {abs(start - CENTER) * 360 / TICKS_PER_TURN:.1f}°")
    bus.write(dxl_id, ADDR_TORQUE_ENABLE, 1, 0)
    if bus.read(dxl_id, ADDR_OPERATING_MODE, 1) != OP_MODE_POSITION:
        bus.write(dxl_id, ADDR_OPERATING_MODE, 1, OP_MODE_POSITION)
    bus.write(dxl_id, ADDR_POSITION_I_GAIN, 2, POSITION_I_GAIN)  # 모드 쓰기 뒤에 (모드가 게인을 리셋)
    bus.write(dxl_id, ADDR_PROFILE_ACCELERATION, 4, PROFILE_ACCELERATION)
    bus.write(dxl_id, ADDR_PROFILE_VELOCITY, 4, PROFILE_VELOCITY)
    bus.write(dxl_id, ADDR_GOAL_POSITION, 4, start)  # 토크 on 순간 튀지 않게
    bus.write(dxl_id, ADDR_TORQUE_ENABLE, 1, 1)
    bus.write(dxl_id, ADDR_GOAL_POSITION, 4, CENTER)
    deadline = time.monotonic() + TIMEOUT_S
    present = start
    while time.monotonic() < deadline:
        time.sleep(0.2)
        present = bus.read(dxl_id, ADDR_PRESENT_POSITION, 4)
        if abs(present - CENTER) <= ARRIVE_TOL_TICK and not bus.read(dxl_id, ADDR_MOVING, 1):
            break
    ok = abs(present - CENTER) <= ARRIVE_TOL_TICK
    print(f"{name}: 현재 틱 {present} ({deg(present):+.2f}°) — {'도착, 토크 유지 중' if ok else '도착 못 함 (토크는 켜 둠)'}")
    if not ok:
        sys.exit(1)


def trim(bus: Bus) -> None:
    """토크를 켠 채로 I 게인만 넣어 2048 에 정확히 정착시킨다 (모드·토크는 건드리지 않음)."""
    for name, dxl_id in IDS.items():
        if not bus.read(dxl_id, ADDR_TORQUE_ENABLE, 1):
            raise SystemExit(f"{name}: 토크 off — --hold {name} 먼저")
        bus.write(dxl_id, ADDR_POSITION_I_GAIN, 2, POSITION_I_GAIN)
        bus.write(dxl_id, ADDR_GOAL_POSITION, 4, CENTER)
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(0.5)
        present = {n: bus.read(i, ADDR_PRESENT_POSITION, 4) for n, i in IDS.items()}
        if all(abs(v - CENTER) <= ARRIVE_TOL_TICK for v in present.values()):
            break
    stable = []
    for _ in range(10):  # 정착 후 2 초 동안 흔들림 확인
        time.sleep(0.2)
        stable.append(tuple(bus.read(i, ADDR_PRESENT_POSITION, 4) for i in IDS.values()))
    print("2 초 샘플 (pan, tilt):", sorted(set(stable)))
    if any(abs(v - CENTER) > ARRIVE_TOL_TICK for row in stable for v in row):
        sys.exit(1)


def release(bus: Bus) -> None:
    for name, dxl_id in IDS.items():
        bus.write(dxl_id, ADDR_TORQUE_ENABLE, 1, 0)
        print(f"{name}: 토크 off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--zero-offset", action="store_true", help="Homing Offset 0 쓰기 (토크 off, 움직임 없음)")
    group.add_argument("--hold", choices=sorted(IDS), help="이 모터를 2048 로 보내고 토크 유지")
    group.add_argument("--trim", action="store_true", help="토크 유지한 채 I 게인으로 정확히 2048 정착")
    group.add_argument("--release", action="store_true", help="두 모터 토크 off")
    args = ap.parse_args()
    bus = Bus()
    try:
        if args.zero_offset:
            zero_offset(bus)
        elif args.hold:
            hold(bus, args.hold)
        elif args.trim:
            trim(bus)
        elif args.release:
            release(bus)
        for name, values in status(bus).items():
            print(name, values)
    finally:
        bus.close()


if __name__ == "__main__":
    main()

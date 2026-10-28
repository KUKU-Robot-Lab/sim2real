import math

import pytest

from head_goto_hold import (
    rad_to_tick, tick_to_rad, clamp_tick, to_rad, validate_within_limits, parse_ids,
    RES_RAD_PER_TICK, OFFSET_RAD, TICK_MAX, PAN_LIMIT_RAD, TILT_LIMIT_RAD,
    ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE,
    ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION, OP_MODE_POSITION,
)


def test_center_tick_is_zero_rad():
    assert rad_to_tick(0.0) == 2048
    assert tick_to_rad(2048) == pytest.approx(0.0, abs=RES_RAD_PER_TICK)


def test_full_range_endpoints():
    assert rad_to_tick(OFFSET_RAD) == 0                 # -π → tick 0
    assert rad_to_tick(-OFFSET_RAD) == TICK_MAX         # +π → clamp 4095


def test_roundtrip_within_resolution():
    for deg in (-90, -30, 0, 15, 30, 89):
        r = math.radians(deg)
        assert tick_to_rad(rad_to_tick(r)) == pytest.approx(r, abs=RES_RAD_PER_TICK)


def test_clamp_bounds():
    assert clamp_tick(-100) == 0
    assert clamp_tick(999999) == TICK_MAX
    assert clamp_tick(2048) == 2048


def test_tilt_30deg_maps_above_center():
    # tilt=+30° → 중앙(2048)보다 큰 tick (부호 규약 sanity)
    assert rad_to_tick(math.radians(30)) > 2048


def test_deg_is_default_unit():
    assert to_rad(30.0, use_deg=True) == pytest.approx(math.radians(30))
    assert to_rad(0.5236, use_deg=False) == pytest.approx(0.5236)


def test_calibration_pose_within_limits():
    # 캘리브 기본 자세 pan=0°, tilt=30° 는 한계 안
    validate_within_limits(0.0, math.radians(30))


def test_reject_raw_30_interpreted_as_radians():
    # ★CRITICAL 회귀 가드: 30을 라디안으로 해석(=1718°)하면 반드시 거부
    with pytest.raises(ValueError):
        validate_within_limits(0.0, 30.0)
    with pytest.raises(ValueError):
        validate_within_limits(30.0, 0.0)


def test_limits_are_90deg():
    assert PAN_LIMIT_RAD == pytest.approx(math.radians(90), abs=1e-4)
    assert TILT_LIMIT_RAD == pytest.approx(math.radians(90), abs=1e-4)


def test_parse_ids():
    assert parse_ids("1,2") == (1, 2)
    for bad in ("1", "1,1", "1,2,3", "a,b"):
        with pytest.raises(ValueError):
            parse_ids(bad)


def test_control_table_addresses():
    # XC330 컨트롤 테이블 (repo/dynamixel_hardware_interface xc330_m288.model 근거)
    assert (ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE) == (11, 64)
    assert (ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION) == (116, 132)
    assert OP_MODE_POSITION == 3


# ---------- 부호 있는 읽기 · 토크를 켜기 전 역회전 검사 (2026-10-28) ----------

def test_head_bus_reads_present_and_offset_as_signed():
    from head_goto_hold import HeadBus

    class Packet:
        def read4ByteTxRx(self, port, dxl_id, address):
            return {132: 0xFFFF_F9A1, 20: 0xFFFF_FC00}[address], 0, 0   # −1631 · −1024

    bus = object.__new__(HeadBus)
    bus.port, bus.packet = None, Packet()
    assert bus.read_present(2) == -1631 and bus.read4_signed(2, 20) == -1024


class _FakeBus:
    """제어표 4바이트 칸만 아는 가짜 목 버스 — 쓰기는 순서대로 기록한다."""

    def __init__(self, table):
        self.table, self.writes = table, []

    def ping(self, dxl_id):
        return 1240

    def read4_signed(self, dxl_id, address):
        return self.table[dxl_id][address]

    def read_present(self, dxl_id):
        return self.table[dxl_id][132]

    def set_position_mode(self, dxl_id):
        self.writes.append(("mode", dxl_id))

    def write_goal(self, dxl_id, tick):
        self.writes.append(("goal", dxl_id, tick))

    def torque(self, dxl_id, on):
        self.writes.append(("torque", dxl_id, on))

    def close(self):
        pass


_PAN_AT_CENTER = {132: 2048, 20: 0, 52: 0, 48: 4095}


def test_goto_hold_refuses_before_any_write_when_the_tilt_is_past_the_boundary(monkeypatch):
    import head_goto_hold as g
    bus = _FakeBus({1: _PAN_AT_CENTER, 2: {132: 3100, 20: -1024, 52: 0, 48: 4095}})
    monkeypatch.setattr(g, "HeadBus", lambda port, baud: bus)
    with pytest.raises(g.HeadReachError):
        g.goto_hold("p", 1, 1, 2, 0.0, math.radians(30), hold=True, tolerance_tick=10, timeout_s=0.0)
    assert bus.writes == []


def test_goto_hold_seeds_the_goal_with_where_the_motor_reads_after_torque_on(monkeypatch):
    """예전에는 −1631 을 부호 없이 읽어 4095 로 잘라 seed 했다. 위치 모드가 다시 잡을 2465 여야 한다."""
    import head_goto_hold as g
    bus = _FakeBus({1: _PAN_AT_CENTER, 2: {132: -1631, 20: -1024, 52: 0, 48: 4095}})
    monkeypatch.setattr(g, "HeadBus", lambda port, baud: bus)
    g.goto_hold("p", 1, 1, 2, 0.0, math.radians(30), hold=True, tolerance_tick=10, timeout_s=0.0)
    assert bus.writes.index(("goal", 2, 2465)) < bus.writes.index(("torque", 2, True))

"""openarm_zero_mfr --shift-joint: 관절 하나만 영점을 옮긴다. 가짜 모터, CAN 없음."""
import importlib
import sys
import types

import numpy as np
import pytest


class Motor:
    def __init__(self, bus, i):
        self.bus, self.i = bus, i

    def get_position(self):
        return self.bus.q[self.i] - self.bus.zero[self.i]


class Arm:
    def __init__(self, bus):
        self.bus = bus

    def get_motors(self):
        return [Motor(self.bus, i) for i in range(7)]

    def mit_control_one(self, i, p):
        if self.bus.enabled:
            self.bus.q[i] = p.q + self.bus.zero[i]   # 위치 지령을 그대로 따라간다


class OpenArm:
    def __init__(self, bus):
        self.bus, self.arm = bus, Arm(bus)

    def recv_all(self):
        pass

    def refresh_all(self):
        pass

    def disable_all(self):
        self.bus.enabled = False
        self.bus.q[5] += self.bus.sag   # 끄는 순간 j6 처짐


class Bus:
    def __init__(self, sag=0.0):
        self.q = np.deg2rad([1.0, -2.0, 3.0, 20.0, 5.0, 0.4, -1.0])   # 물리 각(옛 영점 기준과 같게 시작)
        self.zero = np.zeros(7)
        self.enabled, self.sag, self.zero_ids = True, np.deg2rad(sag), []


@pytest.fixture
def mod(monkeypatch):
    fake = types.SimpleNamespace(MITParam=lambda kp, kd, q, dq, tau: types.SimpleNamespace(q=q))
    monkeypatch.setitem(sys.modules, 'openarm_can', fake)
    m = importlib.import_module('openarm_zero_mfr')
    m = importlib.reload(m)
    monkeypatch.setattr(m.time, 'sleep', lambda s: None)
    return m


def run(mod, monkeypatch, sag):
    bus = Bus(sag)
    oa = OpenArm(bus)

    def zero_one(iface, can_id):
        bus.zero_ids.append(can_id)
        bus.zero[can_id - 1] = bus.q[can_id - 1]

    monkeypatch.setattr(mod, 'send_set_zero_one', zero_one)
    before = [m.get_position() for m in oa.arm.get_motors()]
    return bus, oa, before


def test_shift_moves_only_that_joint(mod, monkeypatch):
    bus, oa, before = run(mod, monkeypatch, sag=0.1)
    done = mod.shift_one(oa, oa.arm, 5, np.deg2rad(-6.1), 'can1')
    assert bus.zero_ids == [0x06]
    assert np.rad2deg(done) == pytest.approx(-6.1, abs=0.15)
    # 같은 물리 자세에서 새 읽기 = 옛 읽기 + 옮긴 양
    phys = bus.q[5]
    assert np.rad2deg(phys - bus.zero[5]) - np.rad2deg(phys) == pytest.approx(np.rad2deg(done), abs=1e-9)
    assert np.all(bus.zero[[0, 1, 2, 3, 4, 6]] == 0)


def test_refuses_when_wrist_sags_after_disable(mod, monkeypatch):
    bus, oa, _ = run(mod, monkeypatch, sag=1.0)
    with pytest.raises(RuntimeError, match='zero NOT written'):
        mod.shift_one(oa, oa.arm, 5, np.deg2rad(-6.1), 'can1')
    assert bus.zero_ids == []


def test_refuses_large_move_without_can(mod, monkeypatch):
    bus, oa, _ = run(mod, monkeypatch, sag=0.0)
    bus.q[5] = np.deg2rad(40.0)                       # 차렷이 아닌 자리에서 출발
    with pytest.raises(RuntimeError, match='이동량'):
        mod.shift_one(oa, oa.arm, 5, np.deg2rad(-6.1), 'can1')
    assert bus.zero_ids == [] and bus.q[5] == pytest.approx(np.deg2rad(40.0))


def test_set_zero_frame_bytes(mod):
    f = mod.set_zero_frame(0x06)
    assert len(f) == 16 and f[:4] == (6).to_bytes(4, 'little') and f[4] == 8 and f[8:] == b'\xff' * 7 + b'\xfe'


def test_recent_shift_blocks_repeat_within_window(mod, tmp_path):
    h = tmp_path / 'h.jsonl'
    mod.record_shift('left_arm', 6, -6.1, -6.09, path=h)
    assert mod.recent_shift('left_arm', 6, path=h)['done_deg'] == pytest.approx(-6.09)
    assert mod.recent_shift('right_arm', 6, path=h) is None
    assert mod.recent_shift('left_arm', 6, now=__import__('time').time() + mod.SHIFT_REPEAT_S + 1, path=h) is None

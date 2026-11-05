"""목 드라이버 연속 목표 모드(11.05) — 가짜 버스, 시리얼을 열지 않는다."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import head_motion as hm  # noqa: E402
from head_driver_core import HeadDriver  # noqa: E402
from head_home import ADDR_GOAL_POSITION, ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE, HeadHome  # noqa: E402

DT = 0.05


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class Bus:
    """goal 레지스터를 따라가는 가짜 모터. 쓴 레지스터를 모두 기록한다."""

    def __init__(self, pan=0.0, tilt=20.0, torque=1, mode=3):
        self.deg = {1: pan, 2: tilt}
        self.goal = dict(self.deg)
        self.reg = {ADDR_TORQUE_ENABLE: torque, ADDR_OPERATING_MODE: mode}
        self.writes = []

    def read_present_tick(self, dxl_id):
        return round((self.deg[dxl_id] + 180.0) * 4095 / 360.0)

    def read4_signed(self, dxl_id, address, label):
        """토크를 켜기 전 검사가 읽는 칸 — Homing Offset 0 · 위치 한계 0~4095."""
        return {132: self.read_present_tick(dxl_id), 20: 0, 52: 0, 48: 4095}[address]

    def read1(self, dxl_id, address, label):
        return self.reg[address]

    def write4(self, dxl_id, address, value, label):
        self.writes.append(address)
        if address == ADDR_GOAL_POSITION:
            self.goal[dxl_id] = value * 360.0 / 4095 - 180.0

    def write1(self, dxl_id, address, value, label):
        self.writes.append(address)

    write2 = write1

    def step(self):
        self.deg = dict(self.goal)


def _driver(bus, clock, max_tilt=90.0):
    out = {"angles": [], "status": [], "log": []}
    config = HeadHome("p", 1_000_000, {1: 0.0, 2: 20.0}, {1: "pan", 2: "tilt"}, 400, 3, 20, 50)

    def apply(controller, cfg, dxl_id):
        bus.goal[dxl_id] = cfg.targets_deg[dxl_id]

    driver = HeadDriver(bus, config, pan_id=1, tilt_id=2,
                        publish_angles=lambda p, t, s: out["angles"].append((p, t, s)),
                        publish_status=out["status"].append, clock=clock, max_tilt_deg=max_tilt, apply=apply,
                        period_s=DT, publish_stream_log=lambda text: out["log"].append(json.loads(text)))
    return driver, out


def _stream(driver, seq, pan, tilt):
    driver.on_stream(json.dumps({"seq": seq, "pan_deg": pan, "tilt_deg": tilt}))


def _run(driver, bus, clock, n, target=None):
    for i in range(n):
        if target is not None:
            _stream(driver, i, *target)
        clock.t += DT
        driver.tick()
        bus.step()


def _last(out):
    return json.loads(out["status"][-1])


# ---------- 클램프(순수) ----------

def test_step_clamps_range_then_velocity_then_acceleration():
    cmd, vel, path, vels = 0.0, 0.0, [], [0.0]
    for _ in range(200):
        cmd, vel, why = hm.stream_step(cmd, vel, 50.0, -10.0, 30.0, 60.0, 300.0, DT)
        path.append(cmd)
        vels.append(vel)
    assert why == "range" and path[-1] == pytest.approx(30.0) and max(path) <= 30.0  # 범위 · 넘어가지 않음
    assert max(abs(v) for v in vels) <= 60.0 + 1e-9
    assert max(abs(b - a) for a, b in zip(vels, vels[1:])) <= 300.0 * DT + 1e-9


def test_step_never_breaks_limits_with_jumping_targets():
    import random
    rng = random.Random(0)
    for _ in range(200):
        cmd, vel, target = 0.0, 0.0, 0.0
        for _ in range(200):
            if rng.random() < 0.1:
                target = rng.uniform(-80.0, 80.0)
            new, new_vel, _ = hm.stream_step(cmd, vel, target, -45.0, 30.0, 60.0, 300.0, DT)
            assert -45.0 <= new <= 30.0 and abs(new_vel) <= 60.0 + 1e-9 and abs(new_vel - vel) <= 300.0 * DT + 1e-9
            cmd, vel = new, new_vel


def test_step_inside_range_reports_rate_until_reached():
    cmd, vel, why = hm.stream_step(0.0, 0.0, 10.0, -45.0, 30.0, 60.0, 300.0, DT)
    assert why == "rate" and cmd == pytest.approx(300.0 * DT * DT)
    cmd, vel, why = hm.stream_step(10.0, 0.0, 10.0, -45.0, 30.0, 60.0, 300.0, DT)
    assert (cmd, vel, why) == (10.0, 0.0, "")


def test_limits_intersect_request_range_and_droop():
    lim = hm.stream_limits(90.0, 109.9)
    assert lim.pan == (-45.0, 30.0) and lim.tilt == (-2.0, 82.0) and not lim.problem()
    assert hm.stream_limits(90.0, 85.0).tilt[1] == pytest.approx(75.0)  # 처짐 하한 − 10
    assert "tilt" in hm.stream_limits(hm.MAX_TILT_DEG, None).problem()  # 기본 --max-tilt -10 이면 학생 범위와 안 겹친다


def test_bad_stream_message_is_rejected():
    for text in ("x", '{"seq": 1, "pan_deg": "a", "tilt_deg": 0}', '{"seq": 1, "pan_deg": NaN, "tilt_deg": 0}'):
        with pytest.raises(hm.BadRequest):
            hm.parse_stream(text)


# ---------- 모드 ----------

def test_stream_enters_writes_only_goal_and_keeps_publishing():
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 60, target=(10.0, 30.0))
    assert driver.streaming and _last(out)["state"] == "accepted"
    assert set(bus.writes) == {ADDR_GOAL_POSITION}  # 토크 · 모드 · 게인 · 프로파일은 안 썼다
    assert len(out["angles"]) == 60 and len(out["log"]) == 60
    assert bus.deg[1] == pytest.approx(10.0, abs=0.1) and bus.deg[2] == pytest.approx(30.0, abs=0.1)
    rec = out["log"][-1]
    assert rec["seq"] == 59 and rec["read_t0"] <= rec["read_t1"] <= rec["write_t"] and set(rec["clamp"]) == {"pan", "tilt"}


def test_stream_clamps_out_of_range_target():
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 100, target=(-90.0, 30.0))
    assert bus.deg[1] == pytest.approx(-45.0, abs=0.1) and out["log"][-1]["clamp"]["pan"] == "range"


def test_stale_target_leaves_stream_and_keeps_last_goal():
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 5, target=(10.0, 30.0))
    _run(driver, bus, clock, round(hm.STREAM_STALE_S / DT) + 1)  # 목표가 끊겼다
    assert not driver.streaming and _last(out)["state"] == "done" and "stale" in _last(out)["reason"]
    goal, writes = dict(bus.goal), len(bus.writes)
    _run(driver, bus, clock, 20)
    assert bus.goal == goal and len(bus.writes) == writes  # 마지막 goal 그대로, 더 쓰지 않는다
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 0.0, 20.0)))
    assert _last(out)["state"] == "accepted"  # 한 번 이동 모드로 돌아왔다


def test_move_request_during_stream_is_rejected_busy():
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 3, target=(5.0, 25.0))
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 0.0, 20.0)))
    st = _last(out)
    assert st["id"] == "m1" and st["state"] == "rejected" and "stream" in st["reason"] and driver.streaming


def test_stream_during_move_is_ignored_and_does_not_start_afterwards():
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 5.0, 25.0)))
    _stream(driver, 1, -10.0, 10.0)
    assert _last(out)["id"] == hm.STREAM_ID and "이동 중" in _last(out)["reason"]
    _run(driver, bus, clock, 60)
    assert not driver.moving and not driver.streaming and bus.goal == {1: 5.0, 2: 25.0}


@pytest.mark.parametrize("bus_kwargs, word", [({"torque": 0}, "토크"), ({"mode": 4}, "모드"), ({"tilt": 100.0}, "범위")])
def test_stream_entry_refused_unless_torque_mode_and_pose_ok(bus_kwargs, word):
    bus, clock = Bus(**bus_kwargs), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 3, target=(0.0, 20.0))
    assert not driver.streaming and bus.writes == [] and word in _last(out)["reason"]


def test_default_max_tilt_refuses_stream():
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock, max_tilt=hm.MAX_TILT_DEG)
    _run(driver, bus, clock, 3, target=(0.0, 20.0))
    assert not driver.streaming and bus.writes == [] and "범위가 비었다" in _last(out)["reason"]


def test_joint_stamp_is_middle_of_the_read():
    bus, clock = Bus(), Clock()

    class SlowBus(Bus):
        def read_present_tick(self, dxl_id):
            clock.t += 0.004
            return super().read_present_tick(dxl_id)

    bus = SlowBus()
    driver, out = _driver(bus, clock)
    t0 = clock.t
    driver.tick()
    assert out["angles"][-1][2] == pytest.approx(t0 + 0.004)
    assert math.isclose(out["angles"][-1][0], 0.0, abs_tol=0.1)


# ---------- 리뷰 반영(11.05) ----------

HOSTILE = ['{"seq": Infinity, "pan_deg": 0, "tilt_deg": 0}', '{"seq": 1, "pan_deg": 1%s, "tilt_deg": 0}' % ("0" * 400),
           "[" * 100000, '{"seq": true, "pan_deg": 0, "tilt_deg": 0}', '{"seq": 1.5, "pan_deg": 0, "tilt_deg": 0}',
           '{"seq": "1", "pan_deg": 0, "tilt_deg": 0}', "[1, 2]"]


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_stream_text_is_rejected_not_raised(text):
    with pytest.raises(hm.BadRequest):
        hm.parse_stream(text)
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    driver.on_stream(text)
    assert _last(out)["state"] == "rejected" and not driver.streaming


@pytest.mark.parametrize("text", ["[" * 100000, '{"id": "a", "pan_deg": 1%s, "tilt_deg": 0}' % ("0" * 400)])
def test_hostile_move_text_is_rejected_not_raised(text):
    with pytest.raises(hm.BadRequest):
        hm.parse_request(text)
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    driver.on_request(text)
    assert _last(out)["state"] == "rejected"


class TiltWriteFails(Bus):
    def write4(self, dxl_id, address, value, label):
        if dxl_id == 2:
            raise OSError("tilt 쓰기 실패")
        super().write4(dxl_id, address, value, label)


def test_partial_write_failure_reports_cooldown_and_once():
    bus, clock = TiltWriteFails(), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 1, target=(10.0, 30.0))
    st = _last(out)
    assert st["state"] == "failed" and st["exit"] == "fail" and "pan 만 썼다" in st["reason"] and not driver.streaming
    assert driver._goal[1] == pytest.approx(bus.goal[1], abs=0.1) and 2 not in driver._goal  # 실제로 쓴 값만
    _run(driver, bus, clock, 10, target=(10.0, 30.0))  # 0.5 s — 쿨다운 안
    assert not driver.streaming and sum(json.loads(x)["state"] == "failed" for x in out["status"]) == 1
    assert "들어가지 않는다" in _last(out)["reason"]
    clock.t += hm.STREAM_FAIL_COOLDOWN_S
    _run(driver, bus, clock, 1, target=(10.0, 30.0))  # 쿨다운 뒤 다시 들어가 같은 사유로 실패
    assert sum(json.loads(x)["state"] == "failed" for x in out["status"]) == 1  # 같은 사유는 한 번만
    assert json.loads(out["status"][-1])["state"] == "accepted"


def test_irregular_tick_intervals_keep_speed_in_real_time():
    import random
    rng = random.Random(1)
    bus, clock = Bus(pan=0.0), Clock()
    driver, out = _driver(bus, clock)
    prev_goal, prev_t = None, None
    for i in range(80):
        _stream(driver, i, 30.0 if i < 40 else -45.0, 20.0)
        clock.t += rng.choice([0.005, 0.02, 0.05, 0.08, 0.3])
        driver.tick()
        bus.step()
        rec = out["log"][-1]
        if prev_goal is not None:
            dt = rec["read_t1"] - prev_t
            assert abs(rec["goal_pan_deg"] - prev_goal) <= 60.0 * min(dt, 0.1) + 1e-9  # 실제 경과 시간 기준, 긴 정지 뒤에도 0.1 s 몫까지
        prev_goal, prev_t = rec["goal_pan_deg"], rec["read_t1"]
    assert driver.streaming


def test_read_failures_in_a_row_leave_stream():
    class Flaky(Bus):
        broken = False

        def read_present_tick(self, dxl_id):
            if self.broken:
                raise OSError("no status")
            return super().read_present_tick(dxl_id)

    bus, clock = Flaky(), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 3, target=(5.0, 25.0))
    bus.broken = True
    _run(driver, bus, clock, hm.STREAM_MAX_READ_FAILS - 1, target=(5.0, 25.0))
    assert driver.streaming
    _run(driver, bus, clock, 1, target=(5.0, 25.0))
    st = _last(out)
    assert not driver.streaming and st["state"] == "failed" and st["exit"] == "fail" and "읽기" in st["reason"]


def test_move_accept_clears_pending_stream_target():
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    _stream(driver, 1, 10.0, 30.0)
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 0.0, 20.0)))
    assert _last(out)["state"] == "accepted" and driver._target is None


def test_stuck_head_leaves_stream_and_holds_where_it_is():
    class Stuck(Bus):
        def step(self):
            pass

    bus, clock = Stuck(), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 60, target=(30.0, 20.0))
    failed = [json.loads(x) for x in out["status"] if json.loads(x)["state"] == "failed"]
    assert not driver.streaming and len(failed) == 1 and "추종 오차" in failed[0]["reason"]
    assert bus.goal[1] == pytest.approx(0.0, abs=0.1) and bus.goal[2] == pytest.approx(20.0, abs=0.1)  # 읽은 자리


def test_positive_goal_bias_lowers_the_upper_limit():
    from dataclasses import replace

    from head_driver_core import stream_limits_for

    config = HeadHome("p", 1_000_000, {1: 0.0, 2: 20.0}, {1: "pan", 2: "tilt"}, 400, 3, 20, 50)
    base = stream_limits_for(config, 1, 2, 90.0)
    lim = stream_limits_for(replace(config, goal_bias_tick={1: 19, 2: -3}), 1, 2, 90.0)
    assert lim.pan[1] == pytest.approx(base.pan[1] - 19 * 360 / 4095, abs=1e-6) and lim.tilt == base.tilt


def test_step_pulls_cmd_into_range_and_stale_exit_is_marked():
    cmd, _, _ = hm.stream_step(50.0, 0.0, 50.0, -10.0, 30.0, 60.0, 300.0, DT)
    assert cmd == 30.0
    bus, clock = Bus(), Clock()
    driver, out = _driver(bus, clock)
    _run(driver, bus, clock, 2, target=(1.0, 21.0))
    _run(driver, bus, clock, 12)
    assert _last(out)["exit"] == "stale" and _last(out)["state"] == "done"

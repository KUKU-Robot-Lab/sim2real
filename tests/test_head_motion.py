"""목 이동 요청 규약·정착 판정·드라이버 상태기계 — 순수부 테스트(ROS·하드웨어 무관)."""
import json
import math

import pytest

import head_motion as hm
from head_driver_core import HeadDriver
from head_home import HeadHome


# ---------- 요청 규약 ----------

def test_request_roundtrip_and_home():
    req = hm.parse_request(hm.encode_request(hm.MoveRequest("a", 20.0, -30.0)))
    assert (req.request_id, req.pan_deg, req.tilt_deg, req.is_home) == ("a", 20.0, -30.0, False)
    assert hm.parse_request(hm.encode_request(hm.MoveRequest("h", None, None))).is_home


@pytest.mark.parametrize("pan,tilt", [(0, -5), (0, 10), (200, -20), (0, float("nan")), (None, -20), (0, None)])
def test_bad_requests_are_rejected(pan, tilt):
    with pytest.raises(hm.BadRequest):
        hm.parse_request(json.dumps({"id": "x", "pan_deg": pan, "tilt_deg": tilt}))


def test_pan_is_free_around_the_circle_and_tilt_up_is_free():
    hm.parse_request(json.dumps({"id": "x", "pan_deg": -180, "tilt_deg": -90}))
    hm.parse_request(json.dumps({"id": "x", "pan_deg": 180, "tilt_deg": -10}))


def test_garbage_is_rejected():
    for text in ("not json", "{}", json.dumps({"pan_deg": 0})):
        with pytest.raises(hm.BadRequest):
            hm.parse_request(text)


# ---------- 정착 판정 ----------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _run(tracker, clock, samples, dt=0.05):
    result = None
    for pan, tilt in samples:
        clock.t += dt
        tracker.add(pan, tilt)
        result = tracker.poll()
        if result is not None:
            return result
    return None


def test_settles_when_still_and_on_target():
    clock = Clock()
    tracker = hm.SettleTracker(20.0, -30.0, clock)
    moving = [(20.0 * i / 15, -20.0 - 10.0 * i / 15) for i in range(15)]
    result = _run(tracker, clock, moving + [(20.0, -30.0)] * 40)
    assert result is not None and result.ok
    assert result.pan_deg == pytest.approx(20.0, abs=0.05) and result.tilt_deg == pytest.approx(-30.0, abs=0.05)


def test_not_settled_while_still_moving():
    clock = Clock()
    tracker = hm.SettleTracker(20.0, -30.0, clock)
    assert _run(tracker, clock, [(i * 0.5, -20.0) for i in range(30)]) is None


def test_still_but_off_target_fails_with_reason():
    clock = Clock()
    tracker = hm.SettleTracker(20.0, -30.0, clock)
    result = _run(tracker, clock, [(0.0, -20.0)] * 40)
    assert result is not None and not result.ok and "떨어져" in result.reason


def test_timeout_and_read_failures_end_the_move():
    clock = Clock()
    tracker = hm.SettleTracker(20.0, -30.0, clock, timeout_s=3.0)
    result = _run(tracker, clock, [(i * 0.3, -20.0) for i in range(100)])
    assert result is not None and not result.ok and "정착하지 못했다" in result.reason
    tracker = hm.SettleTracker(0.0, -20.0, Clock(), max_read_failures=3)
    for _ in range(4):
        tracker.add_failure()
    assert not tracker.poll().ok


def test_result_lines_match_head_home_format():
    import re

    status = {"ok": True, "pan_deg": 20.04, "tilt_deg": -29.98, "pan_sigma_deg": 0.01, "tilt_sigma_deg": 0.0,
              "target_pan_deg": 20.0, "target_tilt_deg": -30.0}
    line = re.compile(r"^\s*(pan|tilt):\s*목표\s*([+-][\d.]+)°\s*·\s*실제\s*([+-][\d.]+)°\s*·\s*오차\s*([+-][\d.]+)°\s*·\s*σ\s*([\d.]+)°.*$")
    lines = hm.format_result_lines(status)
    assert all(line.match(x) and "✓" in x for x in lines)


# ---------- 드라이버 상태기계 ----------

class FakeBus:
    """목표로 일정 속도로 움직이는 가짜 모터. 읽기와 쓰기가 서로 다른 스레드에서 오면 안 되는 것을 흉내 내지는 않는다."""

    def __init__(self):
        self.pan, self.tilt = 0.0, -20.0
        self.goal = (0.0, -20.0)

    def step(self, deg=4.0):
        self.pan += max(-deg, min(deg, self.goal[0] - self.pan))
        self.tilt += max(-deg, min(deg, self.goal[1] - self.tilt))

    def read_present_tick(self, dxl_id):
        deg = self.pan if dxl_id == 1 else self.tilt
        return round((deg + 180.0) * 4095 / 360.0)

    def read4_signed(self, dxl_id, address, label):
        """토크를 켜기 전 검사가 읽는 칸 — Homing Offset 0 · 위치 한계 0~4095 인 모터."""
        deg = self.pan if dxl_id == 1 else self.tilt
        return {132: round((deg + 180.0) * 4095 / 360.0), 20: 0, 52: 0, 48: 4095}[address]


def _driver(bus, clock, max_tilt=-10.0):
    published, statuses = [], []

    def apply(controller, config, dxl_id):
        bus.goal = (config.targets_deg[1], config.targets_deg[2]) if dxl_id == 2 else bus.goal

    config = HeadHome("p", 1_000_000, {1: 0.0, 2: -20.0}, {1: "pan", 2: "tilt"}, 400, 3, 20, 50)
    driver = HeadDriver(bus, config, pan_id=1, tilt_id=2, publish_angles=lambda p, t: published.append((p, t)),
                        publish_status=statuses.append, clock=clock, max_tilt_deg=max_tilt, apply=apply)
    return driver, published, statuses


def _states(statuses):
    return [json.loads(s)["state"] for s in statuses]


def test_angles_keep_publishing_during_a_move_and_move_completes():
    bus, clock = FakeBus(), Clock()
    driver, published, statuses = _driver(bus, clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 20.0, -30.0)))
    assert _states(statuses) == ["accepted"] and driver.moving
    for _ in range(80):
        clock.t += 0.05
        bus.step()
        driver.tick()
    assert _states(statuses) == ["accepted", "done"] and not driver.moving
    assert len(published) == 80  # 이동 중에도 매 틱 발행
    done = json.loads(statuses[-1])
    assert done["ok"] and done["pan_deg"] == pytest.approx(20.0, abs=0.1)


def test_second_request_during_a_move_is_rejected_busy():
    bus, clock = FakeBus(), Clock()
    driver, _, statuses = _driver(bus, clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 20.0, -30.0)))
    driver.on_request(hm.encode_request(hm.MoveRequest("m2", -20.0, -30.0)))
    rejected = json.loads(statuses[-1])
    assert rejected["state"] == "rejected" and rejected["id"] == "m2" and "busy" in rejected["reason"]
    assert bus.goal == (20.0, -30.0)  # 두 번째 요청이 목표를 덮어쓰지 않았다


def test_out_of_range_request_never_reaches_the_motors():
    bus, clock = FakeBus(), Clock()
    driver, _, statuses = _driver(bus, clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("up", 0.0, -5.0)))
    assert json.loads(statuses[-1])["state"] == "rejected" and bus.goal == (0.0, -20.0) and not driver.moving


def test_home_request_goes_to_config_pose():
    bus, clock = FakeBus(), Clock()
    bus.pan, bus.tilt = 30.0, -30.0
    driver, _, statuses = _driver(bus, clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("home", None, None)))
    assert bus.goal == (0.0, -20.0)


def test_read_failure_keeps_the_driver_alive_and_ends_a_move_with_failure():
    class Broken(FakeBus):
        def read_present_tick(self, dxl_id):
            raise RuntimeError("Incorrect status packet")

    bus, clock = Broken(), Clock()
    driver, published, statuses = _driver(bus, clock)
    driver.tick()  # 이동 중이 아니면 조용히 건너뛴다
    assert published == []
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 20.0, -30.0)))
    for _ in range(25):
        clock.t += 0.05
        driver.tick()
    last = json.loads(statuses[-1])
    assert last["state"] == "failed" and not driver.moving and math.isnan(last["pan_deg"])


def test_a_tilt_past_the_wrap_boundary_moves_neither_motor():
    """10.28 RH 머리: 고개가 경계(홈보다 18° 더 숙인 곳) 너머면 위치 모드 토크가 반대쪽으로 ~340° 돈다 — 둘 다 쓰지 않고
    실패로 알린다."""
    class PastEdge(FakeBus):
        def read4_signed(self, dxl_id, address, label):
            if dxl_id == 2:
                return {132: 3100, 20: -1024, 52: 0, 48: 4095}[address]
            return super().read4_signed(dxl_id, address, label)

    applied, statuses = [], []
    config = HeadHome("p", 1_000_000, {1: 0.0, 2: 71.8}, {1: "pan", 2: "tilt"}, 400, 3, 20, 50)
    driver = HeadDriver(PastEdge(), config, pan_id=1, tilt_id=2, publish_angles=lambda p, t: None,
                        publish_status=statuses.append, clock=Clock(), apply=lambda c, g, i: applied.append(i))
    driver.on_request(hm.encode_request(hm.MoveRequest("home", None, None)))
    last = json.loads(statuses[-1])
    assert last["state"] == "failed" and "tilt" in last["reason"] and "경계" in last["reason"]
    assert applied == [] and not driver.moving


def test_a_read_failure_before_torque_on_refuses_the_move():
    class Deaf(FakeBus):
        def read4_signed(self, dxl_id, address, label):
            raise RuntimeError("Incorrect status packet")

    bus, clock = Deaf(), Clock()
    driver, _, statuses = _driver(bus, clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 20.0, -30.0)))
    assert json.loads(statuses[-1])["state"] == "failed" and bus.goal == (0.0, -20.0) and not driver.moving


def test_driver_trims_a_resting_offset_with_feedback_and_then_succeeds():
    """정지했는데 0.5° 어긋나면 목표를 오차만큼 옮겨 다시 맞춘다(마찰·불감대 보정)."""
    class Sticky(FakeBus):
        offset = 0.5  # 목표에 못 미치는 정상상태 오차

        def step(self, deg=4.0):
            goal = (self.goal[0] - self.offset, self.goal[1])
            self.pan += max(-deg, min(deg, goal[0] - self.pan))
            self.tilt += max(-deg, min(deg, goal[1] - self.tilt))

        def write4(self, dxl_id, addr, tick, label):
            # 트림 목표 쓰기: 틱 → 도로 되돌려 목표에 반영
            deg = tick * 360.0 / 4095 - 180.0
            self.goal = (deg, self.goal[1]) if dxl_id == 1 else (self.goal[0], deg)

    bus, clock = Sticky(), Clock()
    driver, _, statuses = _driver(bus, clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 10.0, -20.0)))
    for _ in range(300):
        clock.t += 0.05
        bus.step()
        driver.tick()
    last = json.loads(statuses[-1])
    assert last["state"] == "done" and last["ok"] and abs(last["pan_deg"] - 10.0) <= 0.3
    assert driver._trims >= 1


def test_trim_is_bounded_and_big_errors_are_not_trimmed():
    bus, clock = FakeBus(), Clock()
    driver, _, statuses = _driver(bus, clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("m1", 20.0, -30.0)))
    bus.goal = (5.0, -30.0)  # 모터가 엉뚱한 곳에 서게(오차 15° > 상한)
    for _ in range(60):
        clock.t += 0.05
        bus.step()
        driver.tick()
    assert json.loads(statuses[-1])["state"] == "failed" and driver._trims == 0


class OffsetBus:
    """RH56F1 머리처럼 목표 레지스터보다 일정 틱 큰 자리에 서는 모터(10.28 실측 pan +19 · tilt +3) — 진짜 apply_one 을 받는다."""
    STOPS_PAST = {1: 19, 2: 3}

    def __init__(self, pan=2033, tilt=2864):
        self.tick = {1: pan, 2: tilt}
        self.goal = dict(self.tick)

    def write1(self, dxl_id, address, value, label):
        if address == 116:                       # Goal Position
            self.goal[dxl_id] = int(value)

    write2 = write4 = write1

    def read4_signed(self, dxl_id, address, label):
        return {132: self.tick[dxl_id], 20: 0, 52: 0, 48: 4095}[address]

    def read_present_tick(self, dxl_id):
        return self.tick[dxl_id]

    def step(self, ticks=40):
        for i in self.tick:
            self.tick[i] += max(-ticks, min(ticks, self.goal[i] + self.STOPS_PAST[i] - self.tick[i]))


CAMSET_YAML = """
port: p
baud: 1000000
motors:
  pan: {id: 1, deg: -19.56, goal_bias_tick: -19}
  tilt: {id: 2, deg: 67.473, goal_bias_tick: -3}
position_i_gain: 400
operating_mode: 3
profile_acceleration: 20
profile_velocity: 50
"""


@pytest.mark.parametrize("bias, state", [(True, "done"), (False, "failed")])
def test_a_head_that_stops_past_every_goal_reaches_home_only_with_its_goal_bias(tmp_path, bias, state):
    """10.28 RH 머리는 목표보다 pan 19 · tilt 3 틱 큰 자리에 선다 — pan 1.67° 는 보정 상한(1°)을 넘어 트림으로 못 메운다.
    설정의 goal_bias_tick 만큼 빼서 쓰면 기준 자세(camset 1825/2815)에 그대로 선다."""
    from head_home import load_head_home
    text = CAMSET_YAML if bias else CAMSET_YAML.replace(", goal_bias_tick: -19", "").replace(", goal_bias_tick: -3", "")
    path = tmp_path / "head.yaml"
    path.write_text(text, encoding="utf-8")
    bus, clock, statuses = OffsetBus(), Clock(), []
    driver = HeadDriver(bus, load_head_home(path), pan_id=1, tilt_id=2, publish_angles=lambda p, t: None,
                        publish_status=statuses.append, clock=clock)
    driver.on_request(hm.encode_request(hm.MoveRequest("home", None, None)))
    for _ in range(400):
        clock.t += 0.05
        bus.step()
        driver.tick()
        if not driver.moving:
            break
    assert json.loads(statuses[-1])["state"] == state
    if bias:
        assert (bus.tick[1], bus.tick[2]) == (1825, 2815)

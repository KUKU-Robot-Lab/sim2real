"""head_home 의 설정 검증 로직 테스트."""

import pytest

from head_home import HeadHome, load_head_home


def _write(tmp_path, text):
    p = tmp_path / "head_home.yaml"
    p.write_text(text, encoding="utf-8")
    return p


GOOD = """
port: /dev/ttyUSB0
baud: 1000000
motors:
  pan: {id: 1, deg: 0.0}
  tilt: {id: 2, deg: -20.0}
position_i_gain: 400
operating_mode: 3
profile_acceleration: 20
profile_velocity: 50
"""


def test_loads_canonical_config(tmp_path):
    cfg = load_head_home(_write(tmp_path, GOOD))
    assert isinstance(cfg, HeadHome)
    assert cfg.baud == 1_000_000
    assert cfg.position_i_gain == 400
    assert cfg.targets_deg == {1: 0.0, 2: -20.0}
    assert cfg.names == {1: "pan", 2: "tilt"}


def test_rejects_missing_field(tmp_path):
    with pytest.raises(ValueError, match="position_i_gain"):
        load_head_home(_write(tmp_path, GOOD.replace("position_i_gain: 400", "")))


def test_rejects_duplicate_ids(tmp_path):
    bad = GOOD.replace("tilt: {id: 2,", "tilt: {id: 1,")
    with pytest.raises(ValueError, match="중복"):
        load_head_home(_write(tmp_path, bad))


def test_rejects_out_of_range_angle(tmp_path):
    bad = GOOD.replace("deg: -20.0", "deg: -400.0")
    with pytest.raises(ValueError, match="각도"):
        load_head_home(_write(tmp_path, bad))


def test_rejects_bad_i_gain(tmp_path):
    bad = GOOD.replace("position_i_gain: 400", "position_i_gain: 99999")
    with pytest.raises(ValueError, match="게인"):
        load_head_home(_write(tmp_path, bad))


# ---------- 쓰기 순서 (2026-09-01 에 두 번 물린 곳) ----------

# 제어표의 4바이트 칸(부호 있는 값): Present Position 132 · Homing Offset 20 · Min/Max Position Limit 52/48
AT_GOOD_PAN = {132: 2048, 20: 0, 52: 0, 48: 4095}     # GOOD pan 0° 자리
AT_GOOD_TILT = {132: 1820, 20: 0, 52: 0, 48: 4095}    # GOOD tilt −20° 자리


class RecordingController:
    """쓰기 순서를 기록하는 가짜 컨트롤러. 읽기는 제어표 칸에서 준다(기본: GOOD 자세에 서 있는 모터)."""

    def __init__(self, table=None):
        self.writes: list[tuple[str, int, int]] = []
        self.table = table or {1: AT_GOOD_PAN, 2: AT_GOOD_TILT}
        self.closed = False

    def write1(self, dxl_id, address, value, label):
        self.writes.append((label, address, value))

    write2 = write4 = write1

    def read4_signed(self, dxl_id, address, label):
        return self.table[dxl_id][address]

    def ping(self, dxl_id):
        pass

    def close(self):
        self.closed = True


def _addresses(recorder):
    return [label for label, _a, _v in recorder.writes]


def _apply(tmp_path):
    from head_home import apply_one, load_head_home
    cfg = load_head_home(_write(tmp_path, GOOD))
    rec = RecordingController()
    apply_one(rec, cfg, 2)
    return rec


def test_operating_mode_written_before_gain(tmp_path):
    """모드를 쓰면 게인이 리셋된다 — 게인이 뒤여야 한다."""
    order = _addresses(_apply(tmp_path))
    assert order.index("operating mode") < order.index("position i gain")


def test_goal_written_after_torque_on(tmp_path):
    """토크 0→1 에서 펌웨어가 Goal Position 을 Present 로 덮어쓴다 — 목표가 뒤여야 한다."""
    order = _addresses(_apply(tmp_path))
    assert order.index("torque on") < order.index("goal")


def test_torque_off_comes_first(tmp_path):
    """Operating Mode 는 EEPROM 이라 토크가 켜져 있으면 쓰기가 거부된다."""
    order = _addresses(_apply(tmp_path))
    assert order[0] == "torque off"


# ---------- 토크를 켜기 전 역회전 검사 (2026-10-28 RH 머리 tilt) ----------

RH = HeadHome("p", 1_000_000, {1: -1.27, 2: 71.8}, {1: "pan", 2: "tilt"}, 400, 3, 20, 50)
RH_PAN = {132: 1981, 20: 1024, 52: 0, 48: 4095}
TURN_LOW_TILT = {132: -1631, 20: -1024, 52: 0, 48: 4095}  # 10.28 토크 꺼짐 — 한 바퀴 낮게 셌다(실제 2465 = 홈보다 35° 들림)
PAST_EDGE_TILT = {132: 3100, 20: -1024, 52: 0, 48: 4095}  # 경계(3071 = 홈보다 18° 더 숙인 곳) 너머로 숙인 고개


def test_a_tilt_past_the_boundary_is_refused_before_any_write():
    from head_home import HeadReachError, apply_one
    rec = RecordingController({1: RH_PAN, 2: PAST_EDGE_TILT})
    with pytest.raises(HeadReachError, match="tilt"):
        apply_one(rec, RH, 2)
    assert rec.writes == []


def test_the_1028_one_turn_low_tilt_is_applied_and_goes_home():
    from head_home import apply_one
    rec = RecordingController({1: RH_PAN, 2: TURN_LOW_TILT})
    apply_one(rec, RH, 2)
    assert rec.writes[-1] == ("goal", 116, 2864)


def test_here_tick_is_where_the_motor_reads_after_torque_on():
    """head_pose_check 가 '지금 자리'를 목표로 쓸 때 — 토크 꺼짐 값(−1631)을 쓰면 한 바퀴를 돈다."""
    from head_home import here_tick
    assert here_tick(RecordingController({1: RH_PAN, 2: TURN_LOW_TILT}), RH, 2) == 2465


def test_cli_checks_both_motors_before_writing_to_either(tmp_path, monkeypatch, capsys):
    """pan 만 움직이고 tilt 에서 멈추지 않는다 — 하나라도 걸리면 아무것도 쓰지 않는다."""
    import head_home
    rec = RecordingController({1: RH_PAN, 2: PAST_EDGE_TILT})
    monkeypatch.setattr(head_home, "CompliantController", lambda *_: rec)
    cfg = _write(tmp_path, GOOD.replace("deg: 0.0", "deg: -1.27").replace("deg: -20.0", "deg: 71.8"))
    assert head_home.main(["--config", str(cfg), "--execute"]) == 1
    assert rec.writes == []
    assert "경계" in capsys.readouterr().err


def test_release_never_changes_goal_or_gains(tmp_path, monkeypatch):
    import head_home
    cfg = _write(tmp_path, GOOD)
    rec = RecordingController()
    rec.closed = False
    rec.close = lambda: setattr(rec, 'closed', True)
    monkeypatch.setattr(head_home, 'CompliantController', lambda *_: rec)
    assert head_home.main(['--config', str(cfg), '--release', '--execute']) == 0
    assert rec.closed
    assert rec.writes == [('torque off', head_home.ADDR_TORQUE_ENABLE, head_home.TORQUE_OFF)] * 2


def test_release_without_execute_never_opens_a_serial_port(tmp_path, monkeypatch):
    import head_home
    cfg = _write(tmp_path, GOOD)
    def forbidden(*_):
        raise AssertionError('dry run must not open serial')
    monkeypatch.setattr(head_home, 'CompliantController', forbidden)
    assert head_home.main(['--config', str(cfg), '--release']) == 0


# ---------- 목표 보정 goal_bias_tick (2026-10-28 RH 머리) ----------
# RH56F1 머리는 목표 레지스터보다 일정 틱 큰 자리에 선다 — 펌웨어 궤적 끝(Position Trajectory)부터 목표와 다르다
# (10.28 실측 pan +19 · tilt +3: 1805→1824 · 1815→1834 · 1806→1825 · 2807→2810 · 2812→2815). 설정의 goal_bias_tick 을
# 더해 쓰면 목표 그 자리에 선다. 판정(verify · 드라이버)은 원래 목표 기준이다.
RH_CAMSET = """
port: /dev/ttyUSB0
baud: 1000000
motors:
  pan: {id: 1, deg: -19.56, goal_bias_tick: -19}
  tilt: {id: 2, deg: 67.473, goal_bias_tick: -3}
position_i_gain: 400
operating_mode: 3
profile_acceleration: 20
profile_velocity: 50
"""
RH_AT_1028 = {1: {132: 1982, 20: 1024, 52: 0, 48: 4095}, 2: {132: 2465, 20: -1024, 52: 0, 48: 4095}}


def test_goal_bias_is_read_per_motor_and_is_zero_when_absent(tmp_path):
    from head_home import goal_tick
    from head_position_hold_node import deg_to_tick
    cfg = load_head_home(_write(tmp_path, RH_CAMSET))
    assert (goal_tick(cfg, 1), goal_tick(cfg, 2)) == (1825 - 19, 2815 - 3)
    plain = load_head_home(_write(tmp_path, GOOD))
    assert goal_tick(plain, 2) == deg_to_tick(-20.0)


def test_a_goal_bias_beyond_60_ticks_is_refused(tmp_path):
    with pytest.raises(ValueError, match="goal_bias_tick"):
        load_head_home(_write(tmp_path, RH_CAMSET.replace("goal_bias_tick: -19", "goal_bias_tick: -61")))


def test_apply_writes_the_goal_with_the_bias_so_the_head_stops_on_target(tmp_path):
    from head_home import apply_one
    cfg = load_head_home(_write(tmp_path, RH_CAMSET))
    rec = RecordingController(RH_AT_1028)
    apply_one(rec, cfg, 1)
    apply_one(rec, cfg, 2)
    assert [v for label, _a, v in rec.writes if label == "goal"] == [1806, 2812]

"""XC330 위치의 바퀴 수 — 부호 있는 읽기 · 위치 모드가 다시 맞출 값 · 토크를 켜기 전 역회전 검사.

숫자는 2026-10-28 RH 머리 실측이다: tilt Homing Offset −1024 · 홈 71.8°(2864 틱) · 토크가 꺼진 채 −1631 → 재부팅 뒤 2465.
pan 은 offset +1024 · 홈 −1.27°(2033 틱).
"""
import pytest

from head_turns import HeadReachError, signed32, torque_on_starts, torque_on_tick

TILT_HOME, PAN_HOME = 2864, 2033


def _motor(present, offset, lo=0, hi=4095):
    """제어표의 4바이트 칸(부호 있는 값): Present Position 132 · Homing Offset 20 · Min/Max Position Limit 52/48."""
    return {132: present, 20: offset, 52: lo, 48: hi}


def _read4(table):
    return lambda dxl_id, address: table[dxl_id][address]


@pytest.mark.parametrize("raw, want", [
    (0xFFFF_F9A1, -1631),          # 10.28 tilt — SDK 는 4294965665 로 준다(각도로 바꾸면 3.8억 도)
    (-1631, -1631),                # 이미 부호 있는 값은 그대로
    (2465, 2465),
    (0x7FFF_FFFF, 2_147_483_647),
    (0x8000_0000, -2_147_483_648),
])
def test_signed32_reads_a_4_byte_register_as_signed(raw, want):
    assert signed32(raw) == want


@pytest.mark.parametrize("present, offset, want", [
    (-1631, -1024, 2465),   # 10.28 재부팅 실측
    (2465, -1024, 2465),    # 이미 한 바퀴 안이면 그대로
    (3100, -1024, -996),    # tilt 경계(3071 = 홈보다 18° 더 숙인 곳) 너머로 숙인 고개 → 반대쪽 끝
    (1981, 1024, 1981),     # pan 은 1024~5119 안이면 그대로
    (1000, 1024, 5096),     # pan 경계(1024 = 홈 오른쪽 89°) 너머 → 위쪽 끝
])
def test_position_mode_rebases_to_one_turn_plus_the_homing_offset(present, offset, want):
    assert torque_on_tick(present, offset, operating_mode=3) == want


@pytest.mark.parametrize("mode", [4, 5])
def test_multi_turn_modes_keep_the_count_when_torque_turns_on(mode):
    assert torque_on_tick(-1631, -1024, operating_mode=mode) == -1631


def test_the_1028_one_turn_low_tilt_passes_and_starts_where_the_motor_will_read():
    assert torque_on_starts(_read4({2: _motor(-1631, -1024)}), {2: TILT_HOME}) == {2: 2465}


def test_a_tilt_bent_past_the_boundary_is_refused_with_the_boundary_angle():
    # 3100 = 홈보다 21° 더 숙인 자리: 토크를 켜면 −996 으로 잡혀 목표까지 더 숙이는 쪽으로 339° 를 돈다.
    # 경계는 목표 +18°(틱이 커지는 쪽 = 숙임)
    with pytest.raises(HeadReachError, match=r"tilt: .*-996.*339°.*목표 \+18°"):
        torque_on_starts(_read4({2: _motor(3100, -1024)}), {2: TILT_HOME}, names={2: "tilt"})


def test_a_rebase_inside_the_limits_but_more_than_half_a_turn_away_is_refused():
    # 4200 → 104 틱(범위 0~4095 안)이지만 목표 2864 까지 2760 틱 = 243°
    with pytest.raises(HeadReachError, match="243°"):
        torque_on_starts(_read4({2: _motor(4200, -1024)}), {2: TILT_HOME})


def test_a_rebase_outside_the_position_limits_is_refused():
    # pan 이 경계 너머 → 5096 틱(Max Position Limit 4095 밖)
    with pytest.raises(HeadReachError, match=r"5096.*0~4095"):
        torque_on_starts(_read4({1: _motor(1000, 1024)}), {1: PAN_HOME})


def test_half_a_turn_is_the_line_between_the_short_and_the_long_way():
    read4 = _read4({1: _motor(PAN_HOME, 1024)})
    assert torque_on_starts(read4, {1: PAN_HOME + 1934}) == {1: PAN_HOME}     # 170° — 짧은 쪽
    with pytest.raises(HeadReachError):
        torque_on_starts(read4, {1: PAN_HOME + 2049})                         # 180.1° — 긴 쪽


def test_one_bad_motor_refuses_the_whole_set_and_only_it_is_named():
    read4 = _read4({1: _motor(1981, 1024), 2: _motor(3100, -1024)})
    with pytest.raises(HeadReachError) as exc:
        torque_on_starts(read4, {1: PAN_HOME, 2: TILT_HOME}, names={1: "pan", 2: "tilt"})
    assert "tilt" in str(exc.value) and "pan" not in str(exc.value)


def test_a_multi_turn_mode_refuses_an_off_by_one_turn_count():
    # 모드 5 는 토크를 켜도 다시 맞추지 않는다 — −1631 에서 2864 까지 395° 를 돈다
    with pytest.raises(HeadReachError, match="395°"):
        torque_on_starts(_read4({2: _motor(-1631, -1024)}), {2: TILT_HOME}, operating_mode=5)

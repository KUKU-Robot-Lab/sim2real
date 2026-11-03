"""XC330 위치의 바퀴 수 — 부호 있는 읽기 · 위치 모드가 다시 맞출 값 · 토크를 켜기 전 역회전 검사.

제조사 매뉴얼(XC330-M288-T · Present Position 132 · Homing Offset 20):
- 토크가 꺼져 있으면 Present Position 은 바퀴 수까지 이어 센다 — 부호 있는 4바이트라 음수가 된다
  (SDK read4ByteTxRx 는 부호 없이 돌려준다 → `signed32`).
- 위치 제어 모드(3)는 토크를 켤 때 · 모드를 3 으로 바꿀 때 · 전원 · 재부팅 때 '한 바퀴 안 값(0~4095) + Homing Offset'
  으로 다시 맞춘다. offset 은 숫자만 옮긴다 — 경계(offset+4095 ↔ offset)의 물리 위치는 그대로다.
- 확장 위치(4) · 전류 기반 위치(5)는 토크를 켜도 다시 맞추지 않는다.

2026-10-28 RH 머리 tilt(offset −1024 · 홈 2864): 틱이 커지면 숙인다(10.28 머리 카메라로 확인). 경계 3071 은 홈보다 18° 더
숙인 곳(카메라가 바로 아래)이다. 전원이 들어올 때 고개가 경계 너머(더 숙인 쪽)에 있으면 한 바퀴 낮게 잡힌다(10.28: 그 뒤
토크가 꺼진 채 들려 −1631 로 셌고, 재부팅하니 2465 = 홈보다 35° 들린 자리). 고개가 경계 너머에 있을 때 위치 모드로 토크를
켜면 −1024~0 으로 다시 잡혀 홈까지 더 숙이는 쪽으로 ~340° 를 돈다. `torque_on_starts` 가 토크를 켜기 전에 이것을 거른다 —
읽기만 하고 쓰지 않는다.
"""

from __future__ import annotations

from typing import Callable, Mapping

TICKS_PER_TURN = 4096
OP_MODE_POSITION = 3
ADDR_HOMING_OFFSET = 20
ADDR_MAX_POSITION_LIMIT = 48
ADDR_MIN_POSITION_LIMIT = 52
ADDR_PRESENT_POSITION = 132
MAX_TRAVEL_DEG = 180.0  # 이보다 길면 짧은 쪽이 아니라 반대쪽으로 도는 것이다
DROOP_MARGIN_DEG = 5.0  # 기계 하한 실측값 너머로 이만큼까지는 처진 자리로 본다(인코더 · 실측 오차)


class HeadReachError(RuntimeError):
    """토크를 켜면 목이 반대쪽으로 거의 한 바퀴 돌 자리 — 아무것도 쓰지 않고 멈춘다."""


def signed32(value: int) -> int:
    """제어표의 4바이트 칸을 부호 있는 값으로 읽는다 — 이미 부호 있는 값은 그대로다."""
    value = int(value) & 0xFFFF_FFFF
    return value - (1 << 32) if value & 0x8000_0000 else value


def torque_on_tick(present_tick: int, homing_offset: int, operating_mode: int = OP_MODE_POSITION) -> int:
    """토크를 켠 뒤 모터가 읽을 위치. 위치 모드는 한 바퀴 안 값 + offset 으로 다시 맞추고, 4 · 5 는 그대로 둔다."""
    if int(operating_mode) != OP_MODE_POSITION:
        return int(present_tick)
    offset = int(homing_offset)
    return (int(present_tick) - offset) % TICKS_PER_TURN + offset


def torque_on_starts(read4: Callable[[int, int], int], goals: Mapping[int, int], *,
                     operating_mode: int = OP_MODE_POSITION, names: Mapping[int, str] | None = None,
                     max_travel_deg: float = MAX_TRAVEL_DEG) -> dict[int, int]:
    """토크를 켜기 전에 모든 모터를 읽어 확인하고, 토크를 켠 뒤 모터가 읽을 위치를 돌려준다.

    `read4(id, 주소)` 는 4바이트 칸을 부호 있는 값으로 준다. 목표까지 반 바퀴를 넘게 돌거나(반대쪽으로 도는 것),
    위치 모드가 다시 맞춘 위치가 위치 한계 밖인 모터가 하나라도 있으면 HeadReachError — 걸린 모터만 이름을 담는다.
    """
    names = names or {}
    starts: dict[int, int] = {}
    problems: list[str] = []
    for dxl_id, goal in goals.items():
        present = read4(dxl_id, ADDR_PRESENT_POSITION)
        offset = read4(dxl_id, ADDR_HOMING_OFFSET)
        limits = (read4(dxl_id, ADDR_MIN_POSITION_LIMIT), read4(dxl_id, ADDR_MAX_POSITION_LIMIT))
        start = torque_on_tick(present, offset, operating_mode)
        problem = _problem(names.get(dxl_id, f"id {dxl_id}"), present, offset, start, int(goal),
                           int(operating_mode), limits, max_travel_deg)
        if problem:
            problems.append(problem)
        starts[dxl_id] = start
    if problems:
        raise HeadReachError(" / ".join(problems))
    return starts


def _problem(name: str, present: int, offset: int, start: int, goal: int, operating_mode: int,
             limits: tuple[int, int], max_travel_deg: float) -> str | None:
    travel = abs(goal - start) * 360.0 / TICKS_PER_TURN
    if operating_mode != OP_MODE_POSITION:
        if travel <= max_travel_deg:
            return None
        return (f"{name}: 모드 {operating_mode} 는 토크를 켜도 바퀴 수를 다시 맞추지 않는다 — 지금 {present} 에서 "
                f"목표 {goal} 까지 {travel:.0f}° 를 돈다(한도 {max_travel_deg:.0f}°). 토크가 꺼진 채 모터를 재부팅하면"
                f"(움직이지 않는다) 한 바퀴 안 값으로 돌아온다")
    lo, hi = limits
    inside = lo <= start <= hi
    if inside and travel <= max_travel_deg:
        return None
    outside = "" if inside else f" · 위치 한계 {lo}~{hi} 밖"
    return (f"{name}: 위치 모드로 토크를 켜면 {start} 틱으로 다시 잡혀(지금 {present} · offset {offset:+d}) "
            f"목표 {goal} 까지 {travel:.0f}° 를 돈다(한도 {max_travel_deg:.0f}°{outside}) — 고개가 경계"
            f"(목표 {_edge_deg(goal, offset):+.0f}°) 너머에 있다. 토크가 꺼진 채 고개를 손으로 목표 쪽으로 옮긴 뒤 다시 실행")


def _edge_deg(goal: int, offset: int) -> float:
    """목표에서 가장 가까운 위치 모드 경계까지의 각 — + 는 틱이 커지는 쪽."""
    up, down = offset + TICKS_PER_TURN - goal, goal - offset
    return (up if up <= down else -down) * 360.0 / TICKS_PER_TURN


def physical_tick(tick: int, droop_limit_tick: int, *, margin_deg: float = DROOP_MARGIN_DEG) -> int:
    """어느 모드 · 어느 바퀴로 센 값이든 실제 자리 하나로 — 고개 범위가 한 바퀴보다 작아 (하한+여유 − 4095 ~ 하한+여유) 창이 유일하다."""
    hi = int(droop_limit_tick) + round(margin_deg * TICKS_PER_TURN / 360.0)
    lo = hi - (TICKS_PER_TURN - 1)
    return (int(tick) - lo) % TICKS_PER_TURN + lo


def droop_tick(present: int, offset: int, goal: int, droop_limit_tick: int, *,
               margin_deg: float = DROOP_MARGIN_DEG, max_travel_deg: float = MAX_TRAVEL_DEG) -> int | None:
    """위치 모드 경계 너머로 처진 고개의 펼친 실제 위치(경계 ~ 기계 하한 + 여유, 목표까지 반 바퀴 미만) — 아니면 None.

    고개가 움직이는 범위가 한 바퀴보다 훨씬 작아 한 바퀴 안 값으로 실제 자리가 하나로 정해진다. 처지면 틱이 커진다
    (10.28 머리 카메라로 확인) — 위치 모드가 다시 맞춘 값(offset ~ offset+4095)에 한 바퀴를 더하면 경계 너머 자리다.
    """
    unwrapped = torque_on_tick(present, offset) + TICKS_PER_TURN  # 늘 경계(offset+4096) 이상
    in_zone = unwrapped <= droop_limit_tick + round(margin_deg * TICKS_PER_TURN / 360.0)
    short = abs(unwrapped - int(goal)) * 360.0 / TICKS_PER_TURN < max_travel_deg
    return unwrapped if in_zone and short else None

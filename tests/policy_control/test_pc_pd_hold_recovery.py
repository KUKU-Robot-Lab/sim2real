"""워치독 HOLD 는 목표가 돌아오면 풀린다. 다른 HOLD 는 사람이 푼다 (ROS 없음).

2026-09-07 우팔 실기에서 HOLD 가 종점인 설계가 두 번 물렸다:

① engage 와 목표 스트림 시작 사이에 20 초 공백이 생겨 **워치독 HOLD**. 그 뒤 104 초짜리
   자세 이동 재생이 목표 **2612 개를 전부 보냈고 pd 는 다 받았는데**(status seq 2611)
   세트포인트가 얼어 있어 팔이 1 mm 도 안 움직였다. 실패 신호도 없었다.
② preset 유지 중 **발열 HOLD**(r_aj_7). pd 로는 팔을 내릴 수조차 없어, 팔을 든 채
   release 하고 옛 JTC 경로로 내려야 했다.

규칙: **워치독만** 자동 복귀한다(조건이 사라졌으므로). 추종오차·관절한계·τ·발열·estop 은
사람이 판단할 사유라 남는다. 어느 쪽이든 HOLD 중에 목표가 들어오면 **무시하고 있다는 사실이
status 에 보여야 한다** — 오늘은 그 신호가 없어서 재생이 통째로 헛돌았다.
"""
from __future__ import annotations

import pytest

from policy_control import pd_state as S

pytestmark = pytest.mark.unit


def _held(reason: str) -> S.FsmState:
    st = S.initial_fsm(release_zero_ticks=5)
    st = S.transition(st, "engage")
    return S.transition(st, "fault", reason)


def test_watchdog_hold_clears_when_a_fresh_target_arrives():
    st = _held("watchdog: target stale 0.257 s > 0.25 s")
    out = S.transition(st, "target_fresh")
    assert out.phase is S.Phase.RAMPING, "워치독 사유가 사라졌으면 다시 세트포인트를 전진시켜야 한다"
    assert out.hold_reason is None


def test_tracking_error_hold_does_not_clear_itself():
    st = _held("tracking error 0.612 rad > 0.5")
    out = S.transition(st, "target_fresh")
    assert out.phase is S.Phase.HOLD, "사람이 판단할 사유는 목표가 와도 안 풀린다"
    assert out.hold_reason == st.hold_reason


def test_thermal_hold_does_not_clear_itself():
    st = _held("thermal r_aj_7: effort above threshold too long")
    assert S.transition(st, "target_fresh").phase is S.Phase.HOLD


def test_mixed_reasons_keep_the_hold():
    st = _held("watchdog: target stale 0.257 s > 0.25 s")
    st = S.transition(st, "fault", "thermal r_aj_7: effort above threshold too long")
    out = S.transition(st, "target_fresh")
    assert out.phase is S.Phase.HOLD, "워치독이 섞여 있어도 다른 사유가 있으면 남는다"


def test_target_fresh_is_a_no_op_outside_hold():
    st = S.transition(S.initial_fsm(release_zero_ticks=5), "engage")
    assert S.transition(st, "target_fresh") == st


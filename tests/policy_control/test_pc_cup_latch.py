"""단독 aglt 단계의 컵 붙잡기(cup_latch) — reset 때 FP++ 컵을 잡아 두고, 그 에피소드 동안은 FP++ 가 끊겨도 그 값(10.04 실기)."""
from __future__ import annotations

import time
from types import SimpleNamespace as NS

import pytest

pour_fj_node = pytest.importorskip("policy_control.pour_fj_node")


def test_a_latched_cup_survives_an_fpp_dropout():
    pos, quat = (0.25, -0.20, 0.29), (1.0, 0.0, 0.0, 0.0)
    old = (time.monotonic() - 3.0, (0.0, 0.0, 0.0), quat, 0.0)                     # FP++ 가 3 s 끊겼다(손이 가림)
    me = NS(cup_latch=True, cup_hold_last=False, _latched={"arm": (pos, quat, 1.0)}, cups={"arm": old})
    assert pour_fj_node.PourFjNode._cup(me, "arm") == (pos, quat, 1.0)


def test_without_the_latch_a_quiet_fpp_still_means_no_cup():
    old = (time.monotonic() - pour_fj_node.CUP_STALE_S - 0.1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 0.0)
    me = NS(cup_latch=False, cup_hold_last=False, _latched={}, cups={"arm": old})
    assert pour_fj_node.PourFjNode._cup(me, "arm") is None
    me = NS(cup_latch=True, cup_hold_last=False, _latched={}, cups={"arm": old})   # 아직 안 잡았으면 FP++ 규칙 그대로
    assert pour_fj_node.PourFjNode._cup(me, "arm") is None


def _bound(me):
    me._fresh_cup = lambda role: pour_fj_node.PourFjNode._fresh_cup(me, role)
    return me


def test_hold_last_keeps_the_last_live_cup_through_an_occlusion():
    """10.12 사용자(A안): 쥘 때까지 FP++ live — 손이 가려 끊기면 마지막 값(sim FP++ 모델의 ZOH 와 같다)."""
    pos, quat = (0.25, -0.20, 0.29), (1.0, 0.0, 0.0, 0.0)
    old = (time.monotonic() - 2.3, pos, quat, 7.0)                                 # 10.09 실측 최대 끊김 2.3 s
    me = _bound(NS(cup_latch=False, cup_hold_last=True, _latched={}, cups={"arm": old}))
    assert pour_fj_node.PourFjNode._cup(me, "arm") == (pos, quat, 7.0)             # stamp 는 그 프레임 그대로(붙이기 규칙이 쓴다)
    assert pour_fj_node.PourFjNode._fresh_cup(me, "arm") is None                   # reset 은 신선한 값만 받는다
    me = _bound(NS(cup_latch=False, cup_hold_last=True, _latched={}, cups={}))
    assert pour_fj_node.PourFjNode._cup(me, "arm") is None                         # 한 번도 못 받았으면 없다

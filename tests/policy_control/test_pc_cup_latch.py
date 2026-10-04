"""단독 aglt 단계의 컵 붙잡기(cup_latch) — reset 때 FP++ 컵을 잡아 두고, 그 에피소드 동안은 FP++ 가 끊겨도 그 값(10.04 실기)."""
from __future__ import annotations

import time
from types import SimpleNamespace as NS

import pytest

pour_fj_node = pytest.importorskip("policy_control.pour_fj_node")


def test_a_latched_cup_survives_an_fpp_dropout():
    pos, quat = (0.25, -0.20, 0.29), (1.0, 0.0, 0.0, 0.0)
    old = (time.monotonic() - 3.0, (0.0, 0.0, 0.0), quat, 0.0)                     # FP++ 가 3 s 끊겼다(손이 가림)
    me = NS(cup_latch=True, _latched={"arm": (pos, quat, 1.0)}, cups={"arm": old})
    assert pour_fj_node.PourFjNode._cup(me, "arm") == (pos, quat, 1.0)


def test_without_the_latch_a_quiet_fpp_still_means_no_cup():
    old = (time.monotonic() - pour_fj_node.CUP_STALE_S - 0.1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 0.0)
    me = NS(cup_latch=False, _latched={}, cups={"arm": old})
    assert pour_fj_node.PourFjNode._cup(me, "arm") is None
    me = NS(cup_latch=True, _latched={}, cups={"arm": old})                         # 아직 안 잡았으면 FP++ 규칙 그대로
    assert pour_fj_node.PourFjNode._cup(me, "arm") is None

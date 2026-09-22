"""pour 안전망 코어 — 정책과 무관하게 실기에서 항상 켜 두는 두 가지 감시.

sim 에서는 두 팔이 한 articulation 이라 자가충돌이 처리되지만 실기에서는 팔마다 fabric 이
따로 돌아 서로를 모른다. 그래서 이 판정은 정책 바깥에 있어야 하고, 정책이 바뀌어도 남아야 한다.
"""
from __future__ import annotations

import math

import pytest

from policy_control.pour_guard import (GuardCore, GuardInput, GuardLimits, limits_from_contract,
                                       tilt_deg)

UP = (0.0, 0.0, 1.0)                                    # 똑바로 선 컵
SRC = (0.30, -0.16, 0.25)                               # 오른쪽(소스)
RCV = (0.30, 0.16, 0.25)                                # 왼쪽(수신)


def _up_at(deg: float):
    return (math.sin(math.radians(deg)), 0.0, math.cos(math.radians(deg)))


def ok_input(**over):
    base = dict(t=10.0, src_cup_pos=SRC, src_cup_up=UP, rcv_cup_pos=RCV,
                src_palm=(0.30, -0.16, 0.30), rcv_palm=(0.30, 0.16, 0.30),
                src_cup_t=10.0, rcv_cup_t=10.0)
    base.update(over)
    return GuardInput(**base)


def test_tilt_deg_is_the_angle_from_world_up():
    assert tilt_deg((0, 0, 1)) == pytest.approx(0.0)
    assert tilt_deg((1, 0, 0)) == pytest.approx(90.0)
    assert tilt_deg((0, 0, -1)) == pytest.approx(180.0)


def test_tilt_deg_survives_a_slightly_unnormalised_axis():
    assert tilt_deg((0.0, 0.0, 1.0000001)) == pytest.approx(0.0)


def test_a_normal_pose_raises_nothing():
    assert GuardCore().check(ok_input()) == []


def test_over_tilt_aborts():
    r = GuardCore().check(ok_input(src_cup_up=_up_at(140.0)))
    assert len(r) == 1 and "tilt" in r[0]


def test_pouring_tilt_below_the_limit_is_allowed():
    """붓기는 100~110도에서 일어난다 — 그걸 막으면 과제가 불가능해진다."""
    assert GuardCore().check(ok_input(src_cup_up=_up_at(110.0))) == []


def test_crossing_the_centre_line_aborts():
    r = GuardCore().check(ok_input(src_cup_pos=(0.30, 0.05, 0.25)))
    assert len(r) == 1 and "centre line" in r[0]


def test_a_small_wobble_across_zero_does_not_abort():
    """포즈 추정 떨림으로 에피소드가 끊기면 안 된다 — 마진 안쪽은 통과."""
    assert GuardCore().check(ok_input(src_cup_pos=(0.30, 0.01, 0.25))) == []


def test_the_centre_line_follows_the_receiver_side():
    """좌우가 뒤바뀐 배치에서도 같은 규칙이 서야 한다."""
    r = GuardCore().check(ok_input(src_cup_pos=(0.30, -0.05, 0.25), rcv_cup_pos=(0.30, -0.16, 0.25),
                                   src_palm=None, rcv_palm=None))
    assert len(r) == 1 and "centre line" in r[0]


def test_palms_too_close_aborts():
    r = GuardCore().check(ok_input(src_palm=(0.30, -0.02, 0.30), rcv_palm=(0.30, 0.02, 0.30)))
    assert any("palms" in x for x in r)


def test_missing_inputs_are_simply_not_judged():
    """알 수 없는 것을 위반으로 세지 않는다 — 거짓 중단이 진짜 중단보다 비싸다."""
    assert GuardCore().check(GuardInput(t=1.0)) == []


def test_a_stale_cup_pose_is_reported():
    r = GuardCore().check(ok_input(t=11.0, src_cup_t=10.0))
    assert any("stale" in x for x in r)


def test_a_fresh_pose_within_the_window_is_not_stale():
    assert GuardCore().check(ok_input(t=10.3, src_cup_t=10.0)) == []


def test_reasons_latch_until_reset():
    g = GuardCore()
    g.check(ok_input(src_cup_up=_up_at(140.0)))
    assert g.check(ok_input()) == []          # 이번 tick 은 깨끗하고
    assert g.latched and "tilt" in g.latched[0]   # 걸렸던 사실은 남는다
    g.reset()
    assert g.latched == []


def test_the_same_reason_is_not_latched_twice():
    g = GuardCore()
    for _ in range(3):
        g.check(ok_input(src_cup_up=_up_at(140.0)))
    assert len(g.latched) == 1


def test_limits_can_be_tightened():
    g = GuardCore(GuardLimits(tilt_abort_deg=95.0))
    assert g.check(ok_input(src_cup_up=_up_at(110.0)))


def test_limits_from_contract_falls_back_when_the_contract_is_silent():
    class Bare:
        pass

    lim = limits_from_contract(Bare())
    assert lim.tilt_abort_deg == GuardLimits().tilt_abort_deg


def test_limits_from_contract_prefers_the_contract():
    class WithLimits:
        src_tilt_abort_deg = 100.0
        palm_min_dist_m = 0.2

    lim = limits_from_contract(WithLimits())
    assert lim.tilt_abort_deg == 100.0 and lim.palm_min_dist_m == 0.2


def test_several_violations_are_all_reported():
    r = GuardCore().check(ok_input(src_cup_up=_up_at(150.0), src_cup_pos=(0.30, 0.10, 0.25)))
    assert len(r) == 2


def test_the_node_file_runs_as_a_script():
    """pour_fake_run.sh 는 이 파일을 경로로 띄운다. 상대 import 였을 때 기동 즉시 죽었고
    (run6 guard.log: ImportError), 체인 판정은 그걸 말하지 않았다."""
    import runpy
    from pathlib import Path

    node = Path(__file__).resolve().parents[2] / "policy_control" / "policy_control" / "pour_guard_node.py"
    ns = runpy.run_path(str(node), run_name="not_main")
    assert "PourGuardNode" in ns and callable(ns["main"])

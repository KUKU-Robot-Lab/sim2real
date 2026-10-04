"""실패한 실기 단계 뒤 자동 pd 해제는 팔을 지정한다(10.04 실기: 쪽 없이 불러 늘 실패했다)."""
from __future__ import annotations

from types import SimpleNamespace as NS

from mission_core import Lane
from s2r_console.console import Console


def _fake(lanes, sides=(), phases=None):
    """_release_sides 가 보는 것만 가진 콘솔 · 세션."""
    phases = phases or {}
    feed = NS(observed=lambda: NS())
    s = NS(mission=NS(lanes=lanes), feed_lock=_NoLock(), feed=feed, supervisor=NS(table=lambda: []))
    me = NS(_lane=Console._lane, _pd_sides=lambda _s, _obs: list(sides),
            _pd_phase=lambda _s, _obs, _procs, side="": phases.get(side, "TRACKING"))
    return me, s


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


LANES = [Lane(id="rig", title="장비"), Lane(id="arm_right", title="오른팔", side="right"),
         Lane(id="arm_left", title="왼팔", side="left")]


def test_a_one_arm_stage_releases_that_arm():
    me, s = _fake(LANES, sides=("right", "left"))
    assert Console._release_sides(me, s, NS(lane="arm_right")) == ["right"]
    assert Console._release_sides(me, s, NS(lane="arm_left")) == ["left"]


def test_a_shared_stage_releases_every_pd_that_holds_an_arm():
    me, s = _fake(LANES, sides=("right", "left"), phases={"left": "IDLE"})
    assert Console._release_sides(me, s, NS(lane="rig")) == ["right"]

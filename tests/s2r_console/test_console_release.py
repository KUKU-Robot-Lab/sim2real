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


def test_a_failed_real_stage_freezes_the_arm_instead_of_releasing_it():
    """★10.09 실기: 실패한 실기 단계 뒤 자동 pd 해제마다 팔이 중력에 0.05~0.1 rad 처진 뒤 JTC 가 잡아, 왼팔이 네 번에 0.56 rad
    밀렸다. 이제 해제하지 않는다 — 정책 단계면 그 팔 에피소드를 먼저 정지하고(pd 가 그 자리를 붙들고 늦게 온 목표를 버린다),
    그다음 pd 붙들기. 해제(pd/release)는 나오지 않는다."""
    from s2r_console.console import failure_actions
    assert failure_actions("policy_aglt_left", ["left"]) == [("episode/stop", "left"), ("pd/hold", "left")]
    assert failure_actions("rehome_right", ["right"]) == [("pd/hold", "right")]
    acts = failure_actions("episode_pick_place_right", ["right", "left"])
    assert ("pd/release", "right") not in acts and acts[-1] == ("pd/hold", "left")
    assert all(svc != "pd/release" for svc, _ in acts)

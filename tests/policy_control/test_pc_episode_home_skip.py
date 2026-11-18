"""★11.04 go_home_start 빠르게 — 이미 홈 · 정지면 계획 · 재생 · goto_home 을 건너뛰고, 손끝 영점은 두 팔 한 번에."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from policy_control.episode_ros import HOME_SKIP_RAD, RosExecutor, at_home, home_arm

HOME = home_arm("rh56f1", "right")


def _frames(q: list[float], side: str = "right", n: int = 40, dt: float = 0.01, jitter: float = 0.0):
    return [(k * dt, {f"{side[0]}_aj_{i + 1}": v + (jitter if k % 2 else 0.0) for i, v in enumerate(q)}) for k in range(n)]


def test_rh56f1_home_comes_from_the_rehome_contract():
    assert HOME is not None and len(HOME) == 7 and len(home_arm("rh56f1", "left")) == 7
    assert home_arm("dg5f", "right") is None                     # 계약을 안 넘기는 로봇은 건너뛰지 않는다


def test_at_home_needs_close_and_still():
    assert at_home(_frames(HOME), "right", HOME)[0]
    near = [v + 0.9 * HOME_SKIP_RAD for v in HOME]
    assert at_home(_frames(near), "right", HOME)[0]
    far = list(HOME)
    far[3] += 2 * HOME_SKIP_RAD
    ok, err = at_home(_frames(far), "right", HOME)
    assert not ok and err == pytest.approx(2 * HOME_SKIP_RAD)
    assert not at_home(_frames(HOME, jitter=0.01), "right", HOME)[0]      # 아직 움직인다
    assert not at_home(_frames(HOME, n=5), "right", HOME)[0]              # 창(0.3 s)을 못 덮는다
    assert at_home([], "right", HOME) == (False, None) and at_home(_frames(HOME), "right", None) == (False, None)


def _me(tmp_path, frames):
    calls, many = [], []
    me = SimpleNamespace(_wait_pd_tracking=lambda side: True, _pd_view=lambda side: {}, run_dir=tmp_path, robot="rh56f1",
                         _run=lambda argv: (calls.append(argv), (0, "tail"))[1], _run_many=lambda argvs: many.extend(argvs) or [0] * len(argvs),
                         _log_line=lambda text: None, cancel=threading.Event(), _lock=threading.Lock(), _arm=frames,
                         contract="c.json", ep=SimpleNamespace(objects={}, holder_poses="config/h.yaml"), _relay={}, _live={},
                         holders={}, _place_seen={})
    return me, calls, many


def test_rehome_at_home_moves_nothing_and_zeroes_both_tips_at_once(tmp_path):
    left = home_arm("rh56f1", "left")
    fr = [(t, {**a, **b}) for (t, a), (_, b) in zip(_frames(HOME), _frames(left, side="left"))]
    me, calls, many = _me(tmp_path, fr)
    traj = SimpleNamespace(kind="go_home", sides=("right", "left"))
    world = SimpleNamespace(hand=lambda side: "EMPTY")
    r = RosExecutor.run_trajectory(me, SimpleNamespace(id="go_home_start"), traj, world)
    assert r.status == "completed" and calls == [] and r.signals["home_skipped"] == ["right", "left"]
    assert [a[a.index("--side") + 1] for a in many] == ["right", "left"] and all("hand/tip_zero" in a for a in many)


def test_rehome_away_from_home_still_plans_and_moves(tmp_path):
    far = list(HOME)
    far[1] += 0.3
    me, calls, many = _me(tmp_path, _frames(far))
    traj = SimpleNamespace(kind="go_home", sides=("right",))
    r = RosExecutor.run_trajectory(me, SimpleNamespace(id="go_home_start"), traj, SimpleNamespace(hand=lambda side: "EMPTY"))
    assert r.status == "completed" and [a[1].rsplit("/", 1)[-1] for a in calls] == \
        ["plan_rehome.py", "check_path_start.py", "replay_to_pd.py", "episode_ctl.py"] and len(many) == 1
    plan = calls[0]                                       # 리뷰 M2: 에피소드 홀더 파일을 넘긴다
    assert plan[plan.index("--holders") + 1].endswith("config/h.yaml")


def test_rehome_without_a_lost_cup_moves_home_or_refuses_when_switched_off(tmp_path, monkeypatch):
    """★11.18 사용자 "컵이 밖으로 떨어지는 거랑 별개로 홈 자세 복귀": 다 기다려도 자리를 모르는 놓기 실패 컵은 계획 세계에서
    빼고 복귀한다(사유 · signals 에 컵 이름). REHOME_WITHOUT_LOST_CUP=False 면 ★11.12 리뷰 H1 대로 움직이지 않고 refused."""
    import policy_control.episode_ros as er
    from policy_control.episode_world import World
    monkeypatch.setattr(er, "REHOME_WAIT_LIVE_S", 0.0)
    far = list(HOME)
    far[1] += 0.3
    world = World(objects={"BLUE": {"at": "unknown"}}, hands={"right": "EMPTY", "left": "EMPTY"})
    node, traj = SimpleNamespace(id="go_home_r"), SimpleNamespace(kind="go_home", sides=("right",))
    me, calls, many = _me(tmp_path, _frames(far))
    me.ep = SimpleNamespace(objects={"BLUE": {"topic": "/objects/source240_blue/pose"}}, holder_poses="")
    r = RosExecutor.run_trajectory(me, node, traj, world)
    assert r.status == "completed" and "BLUE" in r.reason and "BLUE" in r.signals["lost_cup_skipped"][0]
    plan = next(a for a in calls if "plan_rehome.py" in a[1])
    assert "--table-cups" not in plan
    monkeypatch.setattr(er, "REHOME_WITHOUT_LOST_CUP", False)
    me, calls, many = _me(tmp_path, _frames(far))
    me.ep = SimpleNamespace(objects={"BLUE": {"topic": "/objects/source240_blue/pose"}}, holder_poses="")
    r = RosExecutor.run_trajectory(me, node, traj, world)
    assert r.status == "refused" and "BLUE" in r.reason and calls == []


def test_rehome_moves_with_a_lost_cup_that_sat_still_in_its_target_holder(tmp_path, monkeypatch):
    """★11.18 놓기 실패 컵 live 가 낡았어도 마지막 live 가 목표 홀더 받침에 멈춰 있었으면 받침 자리로 보고 움직인다 —
    계획 세계(--table-cups)에 그 자리가 들어가고 runner 사유 · signals 에 대체가 남는다."""
    import time
    import policy_control.episode_ros as er
    from policy_control.episode_world import World
    monkeypatch.setattr(er, "REHOME_WAIT_LIVE_S", 0.0)
    far = list(HOME)
    far[1] += 0.3
    me, calls, many = _me(tmp_path, _frames(far))
    me.ep = SimpleNamespace(objects={"SRC_PINK": {"topic": "/objects/source240_pink/pose"}}, holder_poses="")
    me.holders = {2: ((0.3762, -0.2762, 0.2350), 0.03)}
    me._place_seen = {"SRC_PINK": 2}
    t = time.monotonic() - 30.0
    me._live = {"SRC_PINK": [((0.3766, -0.2766, 0.292), None, t + 0.1 * k) for k in range(5)]}
    world = World(objects={"SRC_PINK": {"at": "unknown"}}, hands={"right": "EMPTY", "left": "EMPTY"})
    r = RosExecutor.run_trajectory(me, SimpleNamespace(id="home_r_after_pink"), SimpleNamespace(kind="go_home", sides=("right",)), world)
    assert r.status == "completed" and "SRC_PINK live 낡음 → 홀더 2 받침 자리로 봄" in r.reason and r.signals["lost_cup_seat"]
    plan = next(a for a in calls if "plan_rehome.py" in a[1])
    assert '"pos": [0.3762, -0.2762' in plan[plan.index("--table-cups") + 1]

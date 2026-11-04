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
                         contract="c.json")
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

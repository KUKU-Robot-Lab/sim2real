"""창 모드(`python3 -m s2r_console --window`) — 창을 닫는 것이 곧 콘솔을 내리는 것이라 닫기 전에 묻는 조건을 잠근다.

브라우저 탭은 닫아도 서버가 남았다. 창은 닫는 순간 콘솔이 끝나고 자식이 정리되므로,
진행 중인 레인이나 살아 있는 자식이 있으면 확인을 받아야 한다. GUI 자체는 띄우지 않는다.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from s2r_console import __main__ as M

pytestmark = pytest.mark.unit


def _console(*, real=False, busy=(), live=()):
    runners = {lane: SimpleNamespace(active=lane in busy) for lane in ("arm_right", "arm_left", "head")}
    sup = SimpleNamespace(alive=lambda: [SimpleNamespace(key=k) for k in live])
    session = SimpleNamespace(runners=runners, supervisor=sup, profile=SimpleNamespace(is_real=real))
    return SimpleNamespace(session=session)


def test_nothing_open_closes_without_asking():
    assert M.close_warning(SimpleNamespace(session=None)) == ""
    assert M.close_warning(_console()) == ""


def test_a_running_lane_or_a_live_child_asks_first():
    msg = M.close_warning(_console(busy=("arm_right",), live=("pd_arm_right#1",)))
    assert "arm_right" in msg and "pd_arm_right#1" in msg
    assert "모두 정지" in msg                                   # fake: 전부 내린다


def test_the_real_robot_warning_says_the_drivers_stay():
    msg = M.close_warning(_console(real=True, live=("drivers#0",)))
    assert "드라이버는 남기고" in msg


def test_window_mode_refuses_before_starting_anything_when_webkit_is_missing(monkeypatch, capsys):
    from s2r_console import window

    def missing():
        raise ImportError("WebKit2 GI 바인딩이 없다")

    monkeypatch.setattr(window, "load", missing)
    monkeypatch.setattr(M, "Console", lambda **_: pytest.fail("콘솔을 만들기 전에 거부해야 한다"))
    assert M.main(["--window", "--no-bridge"]) == 2
    assert "창을 열 수 없다" in capsys.readouterr().err

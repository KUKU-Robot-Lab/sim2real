"""★11.11 stop 뒤 끝 status(outcome) 를 기다려 판정 — 7094e55 sim 에서 opened_before_latch 가 빠지던 race."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from policy_control.episode_ros import RosExecutor


def _me(status):
    warn = []
    me = SimpleNamespace(_lock=threading.Lock(), _status=status,
                         node=SimpleNamespace(get_logger=lambda: SimpleNamespace(warning=warn.append)))
    return me, warn


def test_waits_for_the_end_status_published_after_stop():
    me, warn = _me({"k": {"phase": "running", "episode": 3}})
    t = threading.Timer(0.05, lambda: me._status.__setitem__("k", {"phase": "done", "episode": 3, "outcome": "opened_before_latch"}))
    t.start()
    st = RosExecutor._status_after_end(me, "k", 3, wait_s=1.0)
    assert st["outcome"] == "opened_before_latch" and not warn


def test_old_episode_end_status_is_not_taken():
    me, warn = _me({"k": {"phase": "done", "episode": 2, "outcome": "episode_end"}})
    assert RosExecutor._status_after_end(me, "k", 3, wait_s=0.05) is None and warn


def test_returns_none_and_warns_when_no_end_status():
    me, warn = _me({"k": {"phase": "running", "episode": 3}})
    assert RosExecutor._status_after_end(me, "k", 3, wait_s=0.05) is None and len(warn) == 1

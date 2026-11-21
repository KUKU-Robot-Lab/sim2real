"""11.10 place_m_left 멈춤 대응 — 목 정지 때 깊이-목 짝짓기 완화 · reset 깊이 나이 최대 1 s 대기."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control import place_student as PS

S = 1_000_000_000


def _head(pans, t0=10 * S, dt=S // 20):
    h = PS.HeadState()
    for i, pan in enumerate(pans):
        h.on_state(t0 + i * dt, PS.HEAD_JOINTS, [pan, 0.0], now_ns=t0 + i * dt)
    return h, t0 + (len(pans) - 1) * dt


def test_still_head_pairs_late_frame():
    """(a) 목이 멈춰 있으면 마지막 상태보다 HEAD_PAIR_S 넘게 늦은 깊이도 마지막 상태로 짝짓고 셈한다."""
    h, t_last = _head([0.2] * 10)
    stamp = t_last + int(0.25 * S)
    q = h.at(stamp, now_ns=t_last + int(0.1 * S))
    np.testing.assert_allclose(q, h.latest(t_last))
    assert h.relaxed == 1 and h.dt_ms == pytest.approx(250.0)


def test_moving_head_rejects_late_frame():
    """(b) 목이 움직이는 중(> 0.1°)이면 지금처럼 거부한다."""
    h, t_last = _head(np.linspace(0.0, 0.02, 10))          # 0.02 rad ≈ 1.1°
    with pytest.raises(PS.PlaceInputStale, match="움직이는 중"):
        h.at(t_last + int(0.25 * S), now_ns=t_last + int(0.1 * S))
    assert h.relaxed == 0


def test_still_head_does_not_relax_early_frame():
    """완화는 깊이가 마지막 목 상태보다 늦을 때만 — 첫 상태보다 이른 깊이는 그대로 거부."""
    h, t_last = _head([0.2] * 10)
    with pytest.raises(PS.PlaceInputStale, match="이르다"):
        h.at(10 * S - int(0.3 * S), now_ns=t_last)
    assert h.relaxed == 0


def test_still_but_stale_last_state_rejects():
    """마지막 목 상태가 HEAD_MAX_AGE_S 넘게 낡았으면 정지여도 완화하지 않는다."""
    h, t_last = _head([0.2] * 10)
    with pytest.raises(PS.PlaceInputStale, match="전"):
        h.at(t_last + int(0.25 * S), now_ns=t_last + int((PS.HEAD_MAX_AGE_S + 0.05) * S))
    assert h.relaxed == 0


class _Sub:
    """take_message 를 n 번째 호출부터 msg 로 돌려주는 가짜 구독."""

    def __init__(self, msg=None, after=0):
        self.msg, self.after, self.calls = msg, after, 0
        self.msg_type, self.raw = object, False
        self.handle = self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def take_message(self, msg_type, raw):
        self.calls += 1
        return (self.msg, None) if self.msg is not None and self.calls > self.after else None


def _node(monkeypatch, sub):
    N = pytest.importorskip("policy_control.rh_place_student_node")
    monkeypatch.setattr(N, "DEPTH_START_WAIT_S", 0.3)        # 시험 시간만 줄인다(판정 논리는 같다)
    sp = SimpleNamespace(frame=SimpleNamespace(stamp_ns=time.time_ns() - int(0.574 * S)), max_depth_age_s=0.2, now_ns=0)
    node = N.PlaceStudentNode.__new__(N.PlaceStudentNode)
    node._policy, node.logs = sp, []
    node.get_logger = lambda: SimpleNamespace(info=node.logs.append)
    node._pumps = ((sub, lambda m: setattr(sp, "frame", SimpleNamespace(stamp_ns=time.time_ns()))),)
    return N, node, sp


def test_reset_waits_for_new_depth(monkeypatch):
    """(c) 시작 때 깊이가 낡았어도 대기 안에 새 프레임이 오면 통과한다."""
    sub, other = _Sub(msg="depth", after=3), _Sub()
    N, node, sp = _node(monkeypatch, sub)
    node._pumps = (*node._pumps, (other, lambda m: None))    # 마스크 · 목 구독도 같이 꺼낸다
    t0 = time.monotonic()
    now = node._wait_depth()
    assert time.monotonic() - t0 < N.DEPTH_START_WAIT_S
    assert PS.age_refusal((now - sp.frame.stamp_ns) * 1e-9, sp.max_depth_age_s, "놓기 깊이") is None
    assert "새 프레임" in node.logs[-1] and other.calls == sub.calls >= 4


def test_reset_still_stops_without_new_depth(monkeypatch):
    """(d) 대기 안에 새 프레임이 없으면 지금처럼 '놓기 깊이 나이 … > 0.2 s' 로 reset 거부."""
    N, node, sp = _node(monkeypatch, _Sub())
    sp.masks = SimpleNamespace(alive=lambda now: None)
    sp.head = SimpleNamespace(latest=lambda now: np.zeros(2))
    sp.q0, sp.cfg, sp.meta = np.zeros(2), {"mask": {"yolo": None}}, {"depth_pre": None}
    node._mask_cfg, node.holder_id, node._holder_yaw, node._frame_err = {}, 1, {1: 0.0}, ""
    monkeypatch.setattr(N.SM, "check_mask_config", lambda *a: None)
    t0 = time.monotonic()
    with pytest.raises(PS.PlaceInputStale, match=r"놓기 깊이 프레임 놓기 깊이 나이 .* > 0.2 s"):
        node._reset_chain({})
    assert time.monotonic() - t0 >= N.DEPTH_START_WAIT_S
    assert "못 받음" in node.logs[-1]

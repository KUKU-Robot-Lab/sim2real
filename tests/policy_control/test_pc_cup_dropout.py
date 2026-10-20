"""10.19 FP++ 컵 끊김 처리(cup_dropout) — 잡기 전 멈춤 · 쥔 상태 손바닥 따라가기 · 놓친 뒤 멈춤."""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from policy_control.cup_dropout import CupDropout, DropoutCfg, DropoutStop

Q = np.array([1.0, 0.0, 0.0, 0.0])
CUP = (np.array([0.30, -0.10, 0.26]), Q)
CFG = DropoutCfg(stop_s=1.0)


class Clock:
    """정책 틱(60 Hz) — now(monotonic) · 팔 stamp 를 같이 민다. 컵 메시지는 받은 시각 · 영상 stamp(0.29 s 늦게 찍힘)."""

    def __init__(self, d: CupDropout) -> None:
        self.d, self.t, self.got = d, 100.0, None

    def frame(self, pos=None, quat=Q, lag=0.29):
        self.got = (self.t, np.asarray(CUP[0] if pos is None else pos, float), quat, round(self.t - lag, 6))

    def tick(self, palm, grasped=False, R=np.eye(3), n=1, frames=False):
        out = None
        for _ in range(n):
            self.t += 1 / 60
            if frames:
                self.frame()
            cup = None if self.got is None else (self.got[1], self.got[2], self.got[3])
            out = self.d.step(self.t, self.got, grasped, self.t, np.asarray(palm, float), R, cup)
        return out


def test_config_is_off_by_default_and_checks_its_order():
    assert not DropoutCfg().enabled and CFG.enabled
    with pytest.raises(ValueError):
        DropoutCfg(stop_s=1.0, follow_s=1.5)            # 따라가기 · 멈춤 한도는 잡기 전 한도보다 짧아야
    with pytest.raises(ValueError):
        DropoutCfg(stop_s=3.0, near_max_s=2.5)
    with pytest.raises(ValueError):
        DropoutCfg(stop_s=1.0, hold_steps=0)


def test_before_the_grasp_a_far_dropout_stops_after_one_second():
    d = CupDropout(CFG)
    c = Clock(d)
    c.frame()
    far = [0.30, -0.45, 0.40]                            # 손바닥–컵 약 38 cm
    c.tick(far, n=31)
    assert d.mode == "hold"                              # 0.5 s 넘음 — 마지막 값(cup_hold_last 그대로)
    c.tick(far, n=28)
    with pytest.raises(DropoutStop, match="잡기 전 · 손바닥–컵 3[5-9] cm"):
        c.tick(far, n=2)


def test_near_the_hand_it_waits_like_the_training_occlusion_then_stops():
    d = CupDropout(CFG)
    c = Clock(d)
    c.frame()
    near = [0.30, -0.10, 0.36]                           # 10 cm 위(손이 컵을 가린다)
    assert c.tick(near, n=120)[0] == pytest.approx(CUP[0])   # 2.0 s — 마지막 값으로 계속
    assert d.mode == "hold"
    with pytest.raises(DropoutStop, match="손 근처 2.5 s"):
        c.tick(near, n=40)


def test_a_new_frame_resets_the_clock_but_a_republished_stamp_does_not():
    d = CupDropout(CFG)
    c = Clock(d)
    c.frame()
    far = [0.30, -0.45, 0.40]
    c.tick(far, n=50)
    c.frame()                                            # 새 프레임
    c.tick(far, n=50)
    assert d.mode == "hold" and d.age_s < 1.0
    stamp = c.got[3]
    c.got = (c.t, c.got[1], Q, stamp)                    # 같은 영상을 다시 받음 — 새 프레임이 아니다
    with pytest.raises(DropoutStop):
        c.tick(far, n=20)


def test_grasp_flicker_is_not_a_grasp_or_a_release():
    """실기 10.19 왼손: 쥐기 직전 쥔 신호가 1~2 틱씩 깜빡였다(3.61~4.25 s)."""
    d = CupDropout(CFG)
    c = Clock(d)
    c.frame()
    palm = [0.30, -0.10, 0.31]
    for _ in range(20):
        c.tick(palm, grasped=True, n=2)
        c.tick(palm, grasped=False, n=2)
    assert not d.held and not d.released and d.mode == "hold"    # 끊긴 지 1.3 s 지만 손 근처 — 잡기 전 규칙


def test_while_held_the_cup_follows_the_palm_from_the_last_frame():
    d = CupDropout(CFG)
    c = Clock(d)
    palm = np.array([0.30, -0.10, 0.30])
    c.frame(pos=palm + [0, 0, -0.04])
    for _ in range(40):                                  # 쥔 채로 위로 0.3 m/s — 프레임은 매 틱(0.29 s 늦게 찍힘)
        palm = palm + [0, 0, 0.005]
        c.tick(palm, grasped=True, frames=False)
        z_then = palm[2] - 0.29 * 0.3                    # 찍힌 시각의 손바닥 높이
        c.got = (c.t, np.array([0.30, -0.10, z_then - 0.04]), Q, round(c.t - 0.29, 6))   # 컵 = 손바닥 4 cm 아래
    assert d.held and d.mode == "live"
    for _ in range(40):                                  # 끊김(카메라 밖) — 계속 든다
        palm = palm + [0, 0, 0.005]
        out = c.tick(palm, grasped=True)
    assert d.mode == "follow" and d.follows == 1
    assert out[0] == pytest.approx(palm + [0, 0, -0.04], abs=2e-3)
    c.frame(pos=[0.31, -0.10, 0.50])                     # 다시 보임 — FP++ 로
    out = c.tick(palm, grasped=True)
    assert d.mode == "live" and out[0] == pytest.approx([0.31, -0.10, 0.50])


def test_a_frame_from_before_the_grasp_pairs_with_the_palm_at_grasp_time():
    """마지막 프레임이 쥐기 전 것이면 컵은 그때 서 있었다 — 쥔 시각 손바닥과 짝짓는다(cup_attach static 과 같은 규칙)."""
    d = CupDropout(CFG)
    c = Clock(d)
    c.frame(lag=0.0)                                     # 서 있는 컵, 손바닥은 아직 6 cm 위로 다가오는 중
    palm = np.array([0.30, -0.10, 0.36])
    c.tick(palm, n=5)
    palm = np.array([0.30, -0.10, 0.30])                 # 내려와 쥠(이후 프레임 없음)
    c.tick(palm, grasped=True, n=30)
    assert d.mode == "follow"
    out = c.tick(palm + [0, 0, 0.10], grasped=True)      # 10 cm 듦
    assert out[0] == pytest.approx(CUP[0] + [0, 0, 0.10], abs=1e-6)


def test_after_the_grasp_is_lost_a_dropout_stops_at_once():
    """실기 10.19 19:09 오른손: 쥠이 풀리고(5.71 s) FP++ 가 끊겼다(5.73 s) — 노드는 공중 마지막 값을 쫓아 팔을 비틀었다."""
    d = CupDropout(CFG)
    c = Clock(d)
    palm = [0.30, -0.10, 0.36]
    c.tick(palm, grasped=True, n=30, frames=True)
    assert d.held
    c.tick(palm, grasped=False, n=14, frames=True)
    assert d.held                                        # 15 틱 전에는 아직 쥔 상태
    c.tick(palm, grasped=False, n=1, frames=True)
    assert d.released and not d.held and d.mode == "live"     # 보이는 동안은 정책을 계속(다시 쥘 수 있다)
    c.tick(palm, grasped=False, n=30)
    with pytest.raises(DropoutStop, match="쥠이 풀린 뒤"):
        c.tick(palm, grasped=False, n=1)
    assert "쥠이 풀린 뒤" in d.as_dict()["stop"]


def test_new_episode_forgets_everything():
    d = CupDropout(CFG)
    c = Clock(d)
    c.tick([0.3, -0.1, 0.36], grasped=True, n=30, frames=True)
    d.new_episode()
    assert d.as_dict() == {"mode": "live", "age_s": 0.0, "held": False, "released": False, "follows": 0, "stop": None}


def test_the_node_ends_the_episode_with_abort_on_a_dropout_stop():
    pour_fj_node = pytest.importorskip("policy_control.pour_fj_node")
    ended, status = [], []

    def measure(advance=False):
        raise DropoutStop("컵 위치 끊김 0.6 s — 쥠이 풀린 뒤 · 멈춤")

    me = NS(_drain=lambda: None, book=NS(phase="running"), chain=NS(), _measure=measure, cup_fresh={},
            fam=NS(errors=(ValueError,)), _end=lambda kind, why: ended.append((kind, why)),
            _status=lambda ok, reasons, extra=None: status.append((ok, reasons)))
    pour_fj_node.PourFjNode._on_tick(me)
    assert ended == [("abort", "컵 위치 끊김 0.6 s — 쥠이 풀린 뒤 · 멈춤")] and status[-1][0] is False

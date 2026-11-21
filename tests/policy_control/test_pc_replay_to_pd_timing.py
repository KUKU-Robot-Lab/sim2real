"""replay_to_pd 발행 시각표 (ROS 없음).

11.17 실측: rate_scale 0.5 가 발행 주기를 dt/0.5(30 Hz 계단)로 늦췄고, sleep(pub_dt) 누적으로 간격 p95 42 ms.
이제 발행은 기록 dt 고정 · 시간축만 1/rate_scale 배(rh_replay.resample) · 절대 시각 스케줄이다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]


def _tool():
    path = SIM2REAL / "deploy/policy_control/tools/replay_to_pd.py"
    spec = importlib.util.spec_from_file_location("replay_to_pd_timing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _tool()
DT = 1.0 / 60.0
FRAMES = np.cumsum(np.random.default_rng(0).normal(0.0, 0.01, (37, 7)), axis=0)


def test_rate_one_is_identity():
    out = T.stretch(FRAMES, DT, 1.0)
    assert out.shape == FRAMES.shape and np.array_equal(out, FRAMES)
    plan, vel, n_ramp = T.build_plan(out, FRAMES[0], DT, reverse=False)
    assert n_ramp == 1 and np.array_equal(plan[1:], FRAMES)            # 시작 = 첫 프레임 → 램프 1 프레임
    assert np.allclose(vel[2:], np.diff(FRAMES, axis=0) / DT)


def test_rate_half_doubles_frames_same_path():
    out = T.stretch(FRAMES, DT, 0.5)
    assert len(out) == 2 * (len(FRAMES) - 1) + 1
    assert np.array_equal(out[0], FRAMES[0]) and np.array_equal(out[-1], FRAMES[-1])
    assert np.allclose(out[::2], FRAMES)                               # 원본 프레임은 짝수 칸에 그대로
    assert np.allclose(out[1::2], (FRAMES[:-1] + FRAMES[1:]) / 2)      # 사이는 선형 중점
    # 속도 전향은 늘어난 시간에 맞는다 — 원본 속도의 절반
    _, vel, n_ramp = T.build_plan(out, FRAMES[0], DT, reverse=False)
    v_rec = np.diff(FRAMES, axis=0) / DT
    assert np.allclose(vel[n_ramp + 1::2], v_rec / 2)


def test_resample_time_is_monotonic():
    """resample 의 원본 프레임 위치(시간축)가 단조 증가하고 끝을 정확히 찍는다."""
    _, pos = T.resample(FRAMES, DT, 0.3, DT)
    assert np.all(np.diff(pos) > 0) and pos[0] == 0.0 and pos[-1] == len(FRAMES) - 1


def test_schedule_wait_absolute():
    t0, dt = 100.0, 0.02
    assert T.schedule_wait(t0, 3, dt, 100.05) == pytest.approx(0.01)
    assert T.schedule_wait(t0, 3, dt, 100.07) == 0.0                   # 늦으면 자지 않는다
    # 매 스텝 5 ms 늦게 깨어나도 목표 시각은 t0 + k·dt — 지연이 쌓이지 않는다
    now, wake = t0, []
    for k in range(1, 50):
        now += T.schedule_wait(t0, k, dt, now) + 0.005
        wake.append(now)
    assert max(w - (t0 + (k + 1) * dt) for k, w in enumerate(wake)) == pytest.approx(0.005)


# ── 첫 프레임 전 구독 대기 · 발행 지연 판정 (11.17 리뷰) ──

@pytest.mark.parametrize("names, sides, missing", [
    ({"pd_node_left"}, ["left"], []),
    ({"pd_node_right"}, ["left"], ["left"]),                           # 다른 팔 pd 는 안 친다
    ({"pd_node_left", "pd_node_right"}, ["right", "left"], []),
    ({"pd_node_left"}, ["right", "left"], ["right"]),
    ({"pd_node"}, ["right", "left"], []),                              # 양팔 pd 하나(pd_node.node_name)
    ({"pd_node"}, ["left"], []),
    ({"/pd_node_left", "/rosbag2_recorder"}, ["left"], []),            # 앞 '/' 는 떼고 비교
    ({"rosbag2_recorder", "console"}, ["left"], ["left"]),
    (set(), ["right", "left"], ["right", "left"]),
    ({"/pd_node_poll_right"}, ["right"], []),                          # 실제 구독 노드(pd_node.py:240)
    ({"pd_node_poll_left"}, ["right", "left"], ["right"]),
    ({"pd_node_poll_left_right"}, ["right", "left"], []),              # 양팔 pd 하나
    ({"pd_node_poll_right"}, ["left"], ["left"]),
])
def test_missing_pd(names, sides, missing):
    assert T.missing_pd(names, sides) == missing


def test_need_subs_counts_pd_nodes():
    assert T.need_subs(set(), ["left"], 1) == 1
    assert T.need_subs({"pd_node_left", "pd_node_right"}, ["left"], 1) == 1    # 이 재생의 pd 만
    assert T.need_subs({"/pd_node_left", "pd_node_right", "bag"}, ["right", "left"], 1) == 2
    assert T.need_subs({"pd_node_left"}, ["left"], 3) == 3
    assert T.need_subs({"pd_node_poll_left", "pd_node_poll_right"}, ["right", "left"], 1) == 2
    assert T.need_subs({"pd_node_poll_left_right", "x"}, ["right", "left"], 1) == 1


def _clock(step):
    t = [0.0]
    def clock():
        return t[0]
    def spin():
        t[0] += step
    return clock, spin


def test_wait_stable_needs_consecutive():
    clock, spin = _clock(0.05)
    seq = iter([True, True, False, True, True, True, True, True])
    assert T.wait_stable(lambda: next(seq), spin, 5, 3.0, clock) is True
    assert clock() == pytest.approx(0.4)                               # 끊긴 뒤 다시 5회 → 8번 spin


def test_wait_stable_timeout():
    clock, spin = _clock(0.05)
    assert T.wait_stable(lambda: False, spin, 5, 1.0, clock) is False
    clock, spin = _clock(0.05)
    flip = iter([True, False] * 100)                                   # 깜빡이면 통과 못 한다
    assert T.wait_stable(lambda: next(flip), spin, 5, 1.0, clock) is False


def test_late_state():
    dt = 1.0 / 60.0
    assert T.late_state(0.0, dt, 0.1) == "ok"
    assert T.late_state(2 * dt, dt, 0.1) == "ok"                       # 경계는 늦음 아님
    assert T.late_state(2 * dt + 1e-4, dt, 0.1) == "late"
    assert T.late_state(0.1, dt, 0.1) == "late"
    assert T.late_state(0.1001, dt, 0.1) == "resync"


# ── 한 번 늦음은 재기준으로 넘기고, 크거나 잦으면 abort (11.18 tilt_comp_ab 020453 프레임 19 314 ms) ──

def _sched(stalls, n=60, dt=DT, **kw):
    """가짜 시계로 run_schedule. stalls = {프레임: 그 프레임 직전 멈춘 시간 [s]}. 돌려줌 = (결과, 보낸 (k, 시각))."""
    t, sent = [0.0], []
    def clock():
        return t[0]
    def sleep(s):
        t[0] += s
    def send(k):
        sent.append((k, t[0]))
        t[0] += stalls.get(k + 1, 0.0)            # 다음 프레임 전에 멈춤
    kw = {"max_late": 0.1, "abort_late": 1.0, "max_resync": 3, **kw}
    return T.run_schedule(n, dt, send=send, clock=clock, sleep=sleep, **kw), sent


def test_one_stall_resyncs_without_burst():
    st, sent = _sched({19: 0.3})
    assert st["aborted"] is None and st["n_resync"] == 1 and len(sent) == 60
    assert st["shift"] == pytest.approx(0.3 - DT) and st["worst"] == pytest.approx(0.3 - DT)
    gaps = np.diff([s for _, s in sent])
    assert gaps[18] == pytest.approx(0.3)                                # 18 → 19 사이 멈춤 한 번
    assert np.allclose(np.delete(gaps, 18), DT)                         # 그 뒤 간격 pub_dt — 몰아 보내기 없음
    assert [k for k, _ in sent] == list(range(60))                      # 프레임을 건너뛰지도 않는다


def test_big_stall_aborts():
    st, sent = _sched({19: 1.2})
    assert st["aborted"] and "--abort-late" in st["aborted"] and len(sent) == 19 and st["n_resync"] == 0


def test_too_many_resyncs_abort():
    st, sent = _sched({10: 0.3, 20: 0.3, 30: 0.3, 40: 0.3})
    assert st["n_resync"] == 3 and "--max-resync" in st["aborted"] and len(sent) == 40

"""센서 단계 판정(signals) — 합성 손 기록으로 규칙 · 노드 연결 · 러너 신호 결합을 본다. 값은 11.11~11.16 실기 bag 분포 근처."""
import json
from pathlib import Path

import pytest

from policy_control import episode_spec as S
from vlm_supervisor import signals as G
from vlm_supervisor.bench import replay_bags as RB

EP = S.load(Path(__file__).resolve().parents[2] / "config" / "episodes" / "cocktail_1c_e4000.yaml")
CFG = G.load_cfg()
OPEN = [1740.0] * 6


def hand(segments, hz=50.0):
    """segments = [(길이 s, 각도 6, 명령 6, 힘 6, 촉각 합)] 를 이어 붙인 손 기록."""
    h, t = G.HandBuffer(), 0.0
    for dur, ang, tgt, force, tip in segments:
        h.feed_target(t, tgt)
        for _ in range(int(dur * hz)):
            h.feed_angle(t, ang)
            h.feed_force(t, force)
            h.feed_tip(t, [tip / 5] * 5)
            t += 1 / hz
    return h, t


def closed(clos):
    return [1740 - clos * 840] * 4 + [1200, 900]


HELD = (3.0, [x + 40 for x in closed(0.5)], closed(0.55), [600, 600, 600, 600, 1700, 50], 15.0)      # 막혀 gap 40
EMPTY = (3.0, closed(0.45), closed(0.45), [60, 60, 60, 60, 230, 10], 0.5)                            # 명령까지 닫힘
WEAK_BLOCKED = (3.0, [x + 24 for x in closed(0.34)], closed(0.37), [50, 50, 50, 50, 220, 10], 0.7)  # 놓친 뒤 남은 gap


def test_grasp_held_ok_and_empty_fail():
    h, t = hand([(0.5, OPEN, OPEN, [0] * 6, 0.0), HELD])
    v = G.judge("grasp", h, 0.0, t, CFG)
    assert v.label == G.OK and "쥠 힘" in v.reasons[0]
    h, t = hand([(0.5, OPEN, OPEN, [0] * 6, 0.0), EMPTY])
    assert G.judge("grasp", h, 0.0, t, CFG).label == G.FAIL


def test_low_force_but_blocked_is_unsure_not_fail():
    h, t = hand([WEAK_BLOCKED])
    assert G.judge("grasp", h, 0.0, t, CFG).label == G.UNSURE
    assert G.judge("hold", h, 0.0, t, CFG).label == G.UNSURE


def test_empty_hand_during_hold_is_unsure():
    h, t = hand([EMPTY])
    assert G.judge("hold", h, 0.0, t, CFG).label == G.UNSURE


def test_grip_loss_inside_stage_fails():
    h, t = hand([HELD, WEAK_BLOCKED])            # 힘 4100 g → 400 g, 손가락은 그대로 막힌 자리
    v = G.judge("grasp", h, 0.0, t, CFG)
    assert v.label == G.FAIL and any("쥠 상실" in r for r in v.reasons)
    assert v.feats["grip"]["gmax"] > 4000


def test_object_far_from_palm_fails_hold_even_with_weak_hand():
    h, t = hand([WEAK_BLOCKED])
    o = G.ObjectBuffer()
    for k in range(int(t * 30)):
        o.feed("source240_blue", k / 30, (0.40, -0.10, 0.43 if k < 30 else 0.29))
    v = G.judge("hold", h, 0.0, t, CFG, objs=o, palm=(0.40, -0.10, 0.42), obj_name="source240_blue")
    assert v.label == G.FAIL and v.feats["palm_d"] > 0.12


def test_object_lifted_near_palm_rescues_unsure_grasp():
    h, t = hand([WEAK_BLOCKED])
    o = G.ObjectBuffer()
    for k in range(int(t * 30)):
        o.feed("source240_blue", k / 30, (0.4, -0.1, 0.29 + 0.1 * k / (t * 30)))
    v = G.judge("grasp", h, 0.0, t, CFG, objs=o, palm=(0.4, -0.1, 0.44), obj_name="source240_blue")
    assert v.label == G.OK


def test_release():
    h, t = hand([HELD, (2.0, OPEN, OPEN, [-30, 0, 0, 0, 13, 0], 0.2)])
    assert G.judge("release", h, 0.0, t, CFG).label == G.OK
    h, t = hand([HELD])
    assert G.judge("release", h, 0.0, t, CFG).label == G.FAIL
    h, t = hand([(2.0, closed(0.15), closed(0.15), [20, 20, 20, 20, 218, 0], 0.1)])
    assert G.judge("release", h, 0.0, t, CFG).label == G.UNSURE


def test_no_data_is_unsure_and_minus_one_target_keeps_axis():
    assert G.judge("grasp", G.HandBuffer(), 0.0, 1.0, CFG).label == G.UNSURE
    h = G.HandBuffer()
    h.feed_target(0.0, closed(0.5))
    h.feed_target(0.1, [-1] * 6)
    h.feed_angle(0.2, closed(0.5))
    h.feed_force(0.2, [0] * 6)
    assert G.snap(h, 0.3).gap == pytest.approx(0.0)


def test_plan_checks_cocktail():
    nodes = {n.id: n for n in EP.nodes}
    held = {"right": None, "left": None}
    lanes = G.plan_checks(EP, nodes["grasp_to_tilt_start"], held)
    assert {(c.side, c.kind, c.obj) for c in lanes} == {("right", "grasp", "source240_blue"), ("left", "grasp", "shaker_c_orange")}
    assert G.plan_checks(EP, nodes["tilt_blue"], held) == []                      # 쥔 것이 없으면 이송 검사 없음
    held = {"right": "source240_blue", "left": "shaker_c_orange"}
    assert {(c.side, c.kind) for c in G.plan_checks(EP, nodes["tilt_blue"], held)} == {("right", "hold"), ("left", "hold")}
    pb = G.plan_checks(EP, nodes["place_blue"], held)
    assert [(c.side, c.kind, c.obj) for c in pb] == [("right", "release", "source240_blue")]


def test_hub_tracks_held_and_combine_only_lowers():
    hub = G.SensorHub(EP, CFG)
    hub.hand["right"], t = hand([HELD])
    hub.hand["left"], _ = hand([EMPTY])
    nodes = {n.id: n for n in EP.nodes}
    vs = hub(nodes["grasp_to_tilt_start"], 0.0, t)
    assert {v.feats["side"]: v.label for v in vs} == {"right": G.OK, "left": G.FAIL}
    assert hub.held == {"right": "source240_blue", "left": None}
    sig = {"node": "x", "ok": True, "reason": ""}
    out = G.combine(sig, vs)
    assert out["ok"] is False and out["reason"].startswith("센서:") and sig["ok"] is True
    json.dumps(out)                                                              # 기록 줄로 쓸 수 있다
    assert G.combine({"ok": False, "reason": "r"}, [G.Verdict(G.OK, ("x",))])["ok"] is False
    assert G.combine({"ok": True}, [G.Verdict(G.UNSURE, ("x",))])["ok"] is True


def test_replay_helpers():
    assert RB.score(G.OK, "ok") == "맞음" and RB.score(G.FAIL, "ok") == "틀림"
    assert RB.score(G.UNSURE, "fail") == "애매" and RB.score(G.OK, "") == "라벨없음"
    ev = [{"event": "enter", "state_id": "a", "timestamp": "2026-11-11T15:00:00.000"},
          {"event": "exit", "state_id": "a", "executor_type": "policy", "timestamp": "2026-11-11T15:00:05.000",
           "end_time": "2026-11-11T15:00:05.000"}]
    (sid, t0, t1, _), = RB.stages(ev)
    assert sid == "a" and t1 - t0 == pytest.approx(5.0)


# ---------------------------------------------------------------- 실기에서 조용히 틀리는 경로(11.18 리뷰)
def test_force_topic_cut_before_stage_end_is_no_grip_feature():
    h, t = hand([(2.0, *HELD[1:])])                    # 힘 기록 0~2 s 뿐
    v = G.judge("grasp", h, 0.0, 3.0, CFG)             # 전에는 ZeroDivisionError
    assert v.feats["grip"] is None
    assert G.grip_drop(h, 0.0, t + CFG["force_stale_s"] + 0.05, stale_s=CFG["force_stale_s"]) is None
    assert G.grip_drop(h, 0.0, t, stale_s=CFG["force_stale_s"]) is not None


def test_stale_force_in_snap_is_unsure_not_held():
    """힘 토픽만 끊기고 각도는 계속 올 때 — 끊기기 전 쥔 힘 평균으로 '쥠'이라 하지 않는다(전에는 1 s 까지 옛 값)."""
    h, t = hand([(2.0, *HELD[1:])])
    for k in range(int(0.4 * 50)):                     # 각도만 0.4 s 더(> force_stale_s 0.2)
        h.feed_angle(t + k / 50, HELD[1])
    t1 = t + 0.4
    assert G.snap(h, t1, CFG["window_s"], CFG["force_stale_s"]) is None
    assert G.judge("grasp", h, 0.0, t1, CFG).label == G.UNSURE
    assert G.snap(h, t, CFG["window_s"], CFG["force_stale_s"]) is not None


def test_non_finite_sample_is_dropped_and_counted():
    h = G.HandBuffer()
    for k in range(50):
        h.feed_force(k / 50, [float("nan") if k == 9 else 600.0] * 6)
        h.feed_angle(k / 50, closed(0.5))
    h.feed_angle(1.0, [float("inf")] * 6)
    dr = G.grip_drop(h, 0.0, 49 / 50)
    assert dr["gend"] == pytest.approx(3000.0) and h.force.dropped == 1 and h.angle.dropped == 1


def test_no_name_no_palm_ignores_objects():
    h, t = hand([WEAK_BLOCKED])
    o = G.ObjectBuffer()
    for k in range(int(t * 30)):
        o.feed("a_other", k / 30, (0.3, 0.0, 0.29))
        o.feed("b_cup", k / 30, (0.4, -0.1, 0.29 + 0.1 * k / (t * 30)))
    v = G.judge("grasp", h, 0.0, t, CFG, objs=o)
    assert v.label == G.UNSURE and "obj" not in v.feats      # 전에는 a_other(제자리)로 실패


def test_hub_clock_mismatch_raises_but_empty_buffer_is_unsure():
    nodes = {n.id: n for n in EP.nodes}
    hub = G.SensorHub(EP, CFG)
    assert all(v.label == G.UNSURE for v in hub(nodes["grasp_to_tilt_start"], 0.0, 5.0))
    hub.hand["right"], t = hand([HELD])
    for t1 in (t + 1e6, t - 1e6):                             # 러너 monotonic 과 ROS 시각이 섞였을 때
        with pytest.raises(G.ClockSkewError):
            hub(nodes["grasp_to_tilt_start"], t1 - 3.0, t1)


def test_hub_rejects_cfg_missing_keys():
    bad = json.loads(json.dumps(CFG))
    del bad["grasp"]["drop_end_max"], bad["window_s"]
    with pytest.raises(ValueError, match="drop_end_max") as e:
        G.SensorHub(EP, bad)
    assert "window_s" in str(e.value)


def test_feed_and_read_from_two_threads():
    """ROS 콜백 스레드가 넣는 동안 러너 스레드가 읽는다 — 시각과 값의 짝이 어긋나거나 예외가 나면 안 된다."""
    import sys
    import threading
    import time
    s, stop, bad = G._Series(32), threading.Event(), []
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def feed():
        k = 0
        while not stop.is_set():
            k += 1
            s.add(k, [k])
    th = threading.Thread(target=feed, daemon=True)
    th.start()
    end = time.monotonic() + 0.5
    try:
        while time.monotonic() < end and len(bad) < 3:
            n = s.t[-1] if s.t else 0
            bad += [r for r in s.window(n - 20, n + 50) if r[0] != r[1][0]]
            v = s.last_before(n, 1e9)
            if v is not None and v[0] > n:
                bad.append((n, v))
    finally:
        stop.set()
        th.join(1)
        sys.setswitchinterval(old)
    assert not bad, bad[:3]

"""10.13 A안 FP++ live 거르기 — sim(hdgp FppGate)과 같은 규칙인지."""
import numpy as np
import pytest

from policy_control.fpp_gate import FppGate, GateCfg

Q = (1.0, 0.0, 0.0, 0.0)
CFG = GateCfg(jump_m=0.02, dz_m=0.02, radius_m=0.20, max_hold_s=2.5)


def _f(x, y, z, k):
    return ((x, y, z), Q, float(k))


def test_off_passes_every_frame():
    g = FppGate(GateCfg())
    g.step(_f(0.3, -0.2, 0.29, 1), 0.0, (0.3, -0.2, 0.35), True)
    assert g.step(_f(0.45, -0.2, 0.23, 2), 0.1, (0.3, -0.2, 0.35), True)[0] == (0.45, -0.2, 0.23)


def test_rejects_the_121158_jump_near_the_hand_and_keeps_the_last_frame():
    g, palm = FppGate(CFG), (0.32, -0.20, 0.34)
    first = _f(0.259, -0.150, 0.294, 1)
    g.step(first, 0.0, palm, True)
    for k, (x, y, z) in enumerate([(0.271, -0.125, 0.281), (0.287, -0.082, 0.242), (0.308, -0.054, 0.280)], start=2):
        assert g.step(_f(x, y, z, k), 0.1 * k, palm, True) == first
    assert g.rejects == 3


def test_rejects_a_frame_below_the_table_even_if_small():
    g, palm = FppGate(CFG), (0.30, -0.20, 0.33)
    g.step(_f(0.30, -0.20, 0.290, 1), 0.0, palm, True)
    assert g.step(_f(0.30, -0.20, 0.265, 2), 0.1, palm, True)[0][2] == 0.290


def test_accepts_small_motion_far_hand_after_grasp_and_after_max_hold():
    g = FppGate(CFG)
    near, far = (0.30, -0.20, 0.33), (0.30, 0.20, 0.50)
    g.step(_f(0.30, -0.20, 0.29, 1), 0.0, near, True)
    assert g.step(_f(0.31, -0.20, 0.29, 2), 0.1, near, True)[0][0] == pytest.approx(0.31)   # 1 cm 밀림은 받는다
    assert g.step(_f(0.45, -0.20, 0.29, 3), 0.2, far, True)[0][0] == pytest.approx(0.45)    # 손이 멀다
    assert g.step(_f(0.30, -0.20, 0.29, 4), 0.3, (0.44, -0.2, 0.33), False)[0][0] == pytest.approx(0.30)  # 쥔 뒤
    for k in range(5, 10):
        assert g.step(_f(0.50, -0.20, 0.29, k), 0.3 + 0.1 * (k - 4), (0.31, -0.2, 0.33), True)[0][0] == pytest.approx(0.30)
    assert g.step(_f(0.50, -0.20, 0.29, 10), 2.9, (0.31, -0.2, 0.33), True)[0][0] == pytest.approx(0.50)   # 2.5 s 지나면 받는다


def test_same_stamp_is_not_judged_twice_and_new_episode_clears():
    g, palm = FppGate(CFG), (0.30, -0.20, 0.33)
    a = _f(0.30, -0.20, 0.29, 1)
    g.step(a, 0.0, palm, True)
    assert g.step(_f(0.40, -0.20, 0.29, 2), 0.1, palm, True) == a
    assert g.step(_f(0.40, -0.20, 0.29, 2), 0.2, palm, True) == a and g.rejects == 1
    g.new_episode()
    assert g.step(_f(0.40, -0.20, 0.29, 3), 0.3, palm, True)[0][0] == pytest.approx(0.40) and g.rejects == 0


def test_bad_cfg_is_refused():
    with pytest.raises(ValueError):
        GateCfg(jump_m=0.02, radius_m=0.0)
    with pytest.raises(ValueError):
        GateCfg(jump_m=float("nan"))


def test_holding_is_true_only_while_the_latest_new_frame_was_rejected():
    """10.14 실기 204216: 컵이 들린 채 떨어지자 이 거르기가 20 프레임을 버리고 들린 자리를 붙들었다 — 노드가 그 동안
    성공을 세지 않도록 '지금 마지막 값을 붙들고 있다'를 알려 준다."""
    g, palm = FppGate(CFG), (0.29, -0.20, 0.40)
    assert not g.holding
    g.step(_f(0.306, -0.180, 0.384, 1), 0.0, palm, True)
    assert not g.holding
    g.step(_f(0.310, -0.194, 0.328, 2), 0.1, palm, True)          # 들린 컵이 5.6 cm 떨어짐 = 튐으로 버림
    assert g.holding and g.rejects == 1
    g.step(_f(0.310, -0.194, 0.328, 2), 0.2, palm, True)          # 같은 프레임 — 판정도 상태도 그대로
    assert g.holding
    g.step(_f(0.314, -0.156, 0.290, 3), 2.6, palm, True)          # 2.5 s 지나 받음
    assert not g.holding
    g.step(_f(0.40, -0.156, 0.290, 4), 2.7, palm, True)
    g.new_episode()
    assert not g.holding


# ---------------------------------------------------------------- ★11.03 기준점 거르기(anchor_m)
ANCHOR = GateCfg(jump_m=0.02, dz_m=0.02, radius_m=0.20, max_hold_s=2.5, anchor_m=0.03)


def test_anchor_stops_the_15_03_slow_slide_and_keeps_rejecting_after_max_hold():
    """실기 15:03 오른손: 프레임당 2 cm 미만으로 +y 3.7 cm 미끄러짐(jump_m 통과) → 14 cm 튐 · z −6 cm → 2.5 s 뒤에도 틀린 값."""
    g, palm = FppGate(ANCHOR), (0.30, -0.05, 0.30)
    first = _f(0.282, -0.146, 0.292, 1)
    g.step(first, 0.0, palm, True)
    seen = [g.step(_f(0.282, -0.146 + 0.0074 * k, 0.292, 1 + k), 0.1 * k, palm, True)[0] for k in range(1, 6)]   # 0.74 cm 씩
    assert [round(p[1], 4) for p in seen[:4]] == [-0.1386, -0.1312, -0.1238, -0.1164]   # 3 cm 안은 받는다
    assert seen[4] == first[0]                                                          # 3.7 cm = 기준점으로 되돌림
    assert g.step(_f(0.30, -0.006, 0.23, 7), 0.7, palm, True) == first                  # 14 cm 튐 · z −6 cm
    assert g.step(_f(0.295, -0.005, 0.29, 8), 3.5, palm, True) == first                 # max_hold_s 지나도 버린다
    assert g.holding and g.anchor_rejects == 3 and g.as_dict()["anchor_rejects"] == 3
    assert g.peek(None) == first


def test_anchor_does_not_apply_far_from_the_hand_after_grasp_or_when_off():
    g = FppGate(ANCHOR)
    g.step(_f(0.30, -0.20, 0.29, 1), 0.0, (0.30, -0.20, 0.50), True)        # 손 21 cm
    assert g.step(_f(0.30, -0.15, 0.29, 2), 0.1, (0.30, -0.20, 0.55), True)[0][1] == pytest.approx(-0.15)   # 손이 멀다
    assert g.step(_f(0.30, -0.20, 0.29, 3), 0.2, (0.30, -0.20, 0.33), False)[0][1] == pytest.approx(-0.20)  # 쥔 뒤
    assert g.step(_f(0.30, -0.10, 0.39, 4), 0.3, (0.30, -0.20, 0.33), False)[0][1] == pytest.approx(-0.10)
    assert g.anchor_rejects == 0
    g = FppGate(CFG)                                                            # anchor_m 0 = 옛 규칙 그대로
    g.step(_f(0.30, -0.20, 0.29, 1), 0.0, (0.30, -0.20, 0.33), True)
    for k in range(1, 5):
        g.step(_f(0.30, -0.20 + 0.01 * k, 0.29, 1 + k), 0.1 * k, (0.30, -0.20, 0.33), True)
    assert g.last[0][1] == pytest.approx(-0.16) and g.anchor_rejects == 0


def test_anchor_needs_the_jump_gate():
    with pytest.raises(ValueError):
        GateCfg(anchor_m=0.03)
    with pytest.raises(ValueError):
        GateCfg(jump_m=0.02, anchor_m=-0.01)

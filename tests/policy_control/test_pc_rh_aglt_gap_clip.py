"""★11.10 E 팔 목표 틈 상한(GripAssist.arm_gap_clip_rad, 배포 전용) — 끄면 이전과 같은 q*, 켜면 |q* − 실측 q| ≤ δ.

근거: sim2real logs/runs/261110/root_cause/accum(분홍 접근에서 실측 팔이 막힌 동안 q* 가 쌓여 명령 손끝이 탁자 아래 32 mm)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from policy_control import pour_fj_node as N
from policy_control import rh_aglt as A

LEFT = Path(__file__).resolve().parent / "fixtures" / "rh_aglt_left_i05_contract.json"


def _run(dec, acts, qs, pass_q=True):
    out = []
    hq = np.asarray(dec.c.side().hand_open, float)
    for a, q in zip(acts, qs):
        kw = {"arm_q": q} if pass_q else {}
        out.append(dec.step(a, active=True, hand_q=hq, tactile_n=np.zeros(5), **kw)["arm"][0].copy())
    return np.array(out)


def _case(n=200, seed=0):
    c = A.load_contract(LEFT)
    rng = np.random.default_rng(seed)
    acts = np.clip(rng.normal(0.4, 0.5, (n, 13)), -1, 1)            # 한쪽으로 미는 행동 → q* 누적
    home = np.asarray(c.side().arm_home, float)
    qs = home + rng.normal(0, 0.002, (n, 7))                          # 실측 팔은 막혀 홈 근처
    return c, acts, qs


def test_off_is_identical_with_or_without_arm_q():
    c, acts, qs = _case()
    ref = _run(A.RaDecoder(c), acts, qs, pass_q=False)
    off = _run(A.RaDecoder(c, A.GripAssist()), acts, qs, pass_q=True)
    assert np.array_equal(ref, off)


def test_on_bounds_gap_and_matches_when_inside():
    c, acts, qs = _case()
    d = np.radians(3.0)
    dec = A.RaDecoder(c, A.GripAssist(arm_gap_clip_rad=(d,) * 7))
    tg = _run(dec, acts, qs)
    assert np.all(np.abs(tg - qs) <= d + 1e-12)
    assert np.abs(_run(A.RaDecoder(c), acts, qs) - qs).max() > 3 * d   # 끄면 실제로 그 이상 쌓인다(시험이 의미 있음)
    # 실측이 목표를 따라오면(틈 < δ) 켜도 끈 것과 같다
    ref = A.RaDecoder(c); ref_out = []
    on = A.RaDecoder(c, A.GripAssist(arm_gap_clip_rad=(1.0 * 0.5,) * 7)); on_out = []
    hq = np.asarray(c.side().hand_open, float)
    for a in acts[:50] * 0.1:
        q = ref.state["arm"].arm_target.copy()
        ref_out.append(ref.step(a, active=True, hand_q=hq, tactile_n=np.zeros(5))["arm"][0])
        on_out.append(on.step(a, active=True, hand_q=hq, tactile_n=np.zeros(5), arm_q=q)["arm"][0])
    assert np.allclose(ref_out, on_out)


def test_on_requires_arm_q_and_validates():
    c, acts, qs = _case(n=1)
    dec = A.RaDecoder(c, A.GripAssist(arm_gap_clip_rad=(0.05,) * 7))
    with pytest.raises(A.RhAgltError):
        _run(dec, acts, qs, pass_q=False)
    for bad in ((0.05,) * 6, (0.0,) * 7, (0.6,) * 7, (float("nan"),) * 7):
        with pytest.raises(A.RhAgltError):
            A.GripAssist(arm_gap_clip_rad=bad)


def test_inactive_ignores_clip():
    c, acts, qs = _case(n=5)
    dec = A.RaDecoder(c, A.GripAssist(arm_gap_clip_rad=(0.01,) * 7))
    hq = np.asarray(c.side().hand_open, float)
    out = dec.step(acts[0], active=False, hand_q=hq, tactile_n=np.zeros(5))["arm"][0]   # 대기 중엔 arm_q 없어도 된다
    assert np.array_equal(out, dec._idle)


def test_node_param_parse():
    assert N.gap_clip_rad("") == ()
    assert N.gap_clip_rad(" 3 ") == pytest.approx((np.radians(3.0),) * 7)
    assert N.gap_clip_rad("1,2,3,4,5,6,7") == pytest.approx(tuple(np.radians([1, 2, 3, 4, 5, 6, 7])))
    for bad in ("1,2", "a"):
        with pytest.raises(A.RhAgltError):
            N.gap_clip_rad(bad)
    assert len(replace(A.GripAssist(), arm_gap_clip_rad=N.gap_clip_rad("3")).arm_gap_clip_rad) == 7

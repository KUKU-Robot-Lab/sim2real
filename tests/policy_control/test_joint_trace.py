"""실기 joint 기록 요약 — 합성 스트림으로 지연 · 추종 오차 · 끊김 · 포화 셈을 잠근다."""
from __future__ import annotations

import json

import numpy as np
import pytest

from policy_control.joint_trace import render, summarize

pytestmark = pytest.mark.unit
ARM = [f"l_aj_{i}" for i in range(1, 8)]
HAND = ["l_hj_index_2"]


def _rec(lag=0.1, hz=60.0, T=4.0, drop=()):
    t = np.arange(0.0, T, 1.0 / hz)
    t = np.delete(t, list(drop))
    q = np.stack([0.3 * np.sin(2.0 * t + k) for k in range(7)], axis=1)
    tm = np.arange(0.0, T + 0.5, 0.002)
    qm = np.stack([np.interp(tm - lag, t, q[:, k]) for k in range(7)], axis=1)       # 실측 = 목표를 lag 만큼 늦게
    hand_t = np.stack([0.5 * np.ones_like(t)], axis=1)
    act = np.zeros((len(t), 8))
    act[:, 0] = 1.2                                                                   # 팔 첫 차원만 포화
    jn = [json.dumps({"phase": "running", "proc_ms": 5.0, "obj_source": "live"}) for _ in t]
    return {
        "meta_policy_hz": hz, "jn_t": t, "jn_json": np.array(jn),
        "tgt_t": t, "tgt_names": np.array(ARM + HAND), "tgt_q": np.concatenate([q, hand_t], axis=1),
        "arm_t": tm, "arm_names": np.array(ARM), "arm_q": qm,
        "hand_t": tm, "hand_names": np.array(HAND), "hand_q": np.full((len(tm), 1), 0.45),
        "app_t": np.zeros(0), "app_names": np.array([]), "app_q": np.zeros((0, 0)),
        "act_t": t, "act": act,
    }


def test_the_lag_that_best_explains_the_arm_is_found_and_removes_the_error():
    s = summarize(_rec(lag=0.1))
    assert s["arm"].lag_s == pytest.approx(0.1, abs=0.011)
    assert s["arm"].rms_lag.max() < 0.01 < s["arm"].rms0.max()


def test_a_constant_hand_offset_shows_up_as_the_tracking_error():
    s = summarize(_rec())
    assert s["hand"].rms0[0] == pytest.approx(0.05, abs=1e-9)


def test_gaps_rate_saturation_and_object_source_are_counted():
    s = summarize(_rec(drop=range(60, 66)))
    assert s["gaps_over_2dt"] == 1 and s["gap_max_ms"] == pytest.approx(7000 / 60, abs=1.0)
    assert s["action_sat"]["arm"] == pytest.approx(1 / 7) and s["action_sat"]["hand"] == 0.0
    assert s["obj_source"] == {"live": len(_rec(drop=range(60, 66))["tgt_t"])}
    assert "가장 잘 맞는 지연" in render(s)

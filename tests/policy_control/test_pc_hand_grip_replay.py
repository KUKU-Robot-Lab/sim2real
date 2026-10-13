"""10.13 hand_grip_replay — 실기 bag 손 목표 재생 계획(순수 부분)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"))
import hand_grip_replay as R  # noqa: E402

EV = [(1.0, {"event": "stop"}), (10.0, {"event": "start"}), (15.0, {"event": "stop"}),
      (20.0, {"event": "start"}), (25.26, {"event": "stop"}), (26.0, {"event": "stop"})]


def test_episode_window_pairs_start_with_next_stop():
    assert R.episode_window(EV, 1) == (10.0, 15.0)
    assert R.episode_window(EV, 2) == (20.0, 25.26)
    with pytest.raises(ValueError):
        R.episode_window(EV, 3)


def test_plan_uses_target_before_from_as_lead_and_relative_times():
    st = np.arange(19.0, 26.0, 1 / 60)
    vals = np.tile(np.arange(6.0), (len(st), 1)) + (st - 19.0)[:, None]
    p = R.make_plan(st, vals, 1, 20.0, 25.26, 3.5, None)
    i0 = np.searchsorted(st, 23.5, "right") - 1
    assert np.allclose(p.lead, vals[i0])
    assert p.times[0] > 0 and p.times[-1] <= 25.26 - 23.5 + 1e-9
    assert np.all(np.diff(p.times) > 0) and p.targets.shape == (len(p.times), 6)
    assert p.t_to == pytest.approx(5.26)


@pytest.mark.parametrize("t_from,t_to", [(5.0, 4.0), (-1.0, None), (1.0, 9.0)])
def test_plan_rejects_bad_window(t_from, t_to):
    st = np.arange(19.0, 26.0, 1 / 60)
    with pytest.raises(ValueError):
        R.make_plan(st, np.ones((len(st), 6)), 1, 20.0, 25.26, t_from, t_to)


def test_plan_rejects_non_finite_and_bad_shape():
    st = np.arange(19.0, 26.0, 1 / 60)
    v = np.ones((len(st), 6)); v[300, 2] = np.nan
    with pytest.raises(ValueError):
        R.make_plan(st, v, 1, 20.0, 25.26, 0.0, None)
    with pytest.raises(ValueError):
        R.make_plan(st, np.ones((len(st), 5)), 1, 20.0, 25.26, 0.0, None)


def test_over_current_needs_the_whole_span_above_limit():
    h = [(0.0 + k * 0.01, 800.0) for k in range(40)]
    assert R.over_current(h, 0.39, 750.0, 0.3)
    h2 = h[:30] + [(0.30, 600.0)] + h[31:]                 # 창 안에 한 번 내려감 → 아님
    assert not R.over_current(h2, 0.39, 750.0, 0.3)
    assert not R.over_current(h[:10], 0.09, 750.0, 0.3)     # 아직 0.3 s 를 다 못 봤다
    assert not R.over_current([], 1.0, 750.0, 0.3)

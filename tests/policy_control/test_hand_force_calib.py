"""hand_force_calib 계획 로직(실기 없이)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"))
import hand_force_calib as HC  # noqa: E402


def _cfg(**kw):
    a = dict(finger="index", step_reg=5, dwell_s=1.0, contact_g=120.0, past="0,10,20,40,80", hold_s=5.0,
             abort_g=1300.0, abort_ma=750.0, abort_s=0.3)
    a.update(kw)
    return HC.make_cfg(**a)


def test_index_is_slot_3_and_thumb_rotation_is_not_calibrated():
    assert _cfg().slot == 3
    with pytest.raises(HC.CalibError):
        _cfg(finger="thumb_1")


@pytest.mark.parametrize("past", ["", "10,0", "-5,10", "a"])
def test_bad_past_is_refused(past):
    with pytest.raises((HC.CalibError, ValueError)):
        _cfg(past=past)


def test_approach_closes_by_step_and_stops_at_the_closed_end():
    assert HC.approach_next(1740, 5, "index") == 1735
    assert HC.approach_next(904, 5, "index") is None
    assert HC.approach_next(1105, 5, "thumb_2") == 1100
    assert HC.approach_next(1104, 5, "thumb_2") is None


def test_press_targets_go_deeper_and_clip_without_repeats():
    assert HC.press_targets(1500, (0, 10, 20, 40, 80), "index") == [1500, 1490, 1480, 1460, 1420]
    assert HC.press_targets(930, (0, 10, 20, 40, 80), "index") == [930, 920, 910, 900]


def test_touched_uses_the_rest_offset():
    assert not HC.touched(200.0, 91.0, 120.0)
    assert HC.touched(212.0, 91.0, 120.0)


def test_over_for_needs_the_whole_span():
    h = [(0.0, 800.0), (0.1, 800.0), (0.2, 800.0)]
    assert not HC.over_for(h, 0.2, 750.0, 0.3)
    h.append((0.31, 800.0))
    assert HC.over_for(h, 0.31, 750.0, 0.3)
    h.append((0.32, 700.0))
    assert not HC.over_for(h, 0.32, 750.0, 0.3)


def test_dry_run_prints_the_plan_without_ros(capsys):
    assert HC.main(["--side", "right", "--finger", "middle"]) == 0
    out = capsys.readouterr().out
    assert "middle(슬롯 2)" in out and "계획만" in out

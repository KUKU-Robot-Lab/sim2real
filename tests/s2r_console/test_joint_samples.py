"""Joint freshness is independent for each driver, even in merged reports."""
import pytest
from s2r_console import bridge


def test_arm_updates_do_not_refresh_an_unheard_hand():
    samples = bridge.JointSamples()
    samples.update('r_hj_index_1', {0: 0.4}, now=10.0)
    samples.update('r_aj_1', {0: 0.2, 1: 0.0}, now=14.0)
    values, ages = samples.report(now=14.1)
    assert values['r_hj_index_1'] == [0.4, None, None, None, None]
    assert ages['r_hj_index_1'] == pytest.approx(4.1)
    assert ages['r_aj_1'] == pytest.approx(0.1)

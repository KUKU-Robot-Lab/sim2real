"""The sensor guard never treats malformed or thresholded samples as free space."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "stationary_contact_guard",
    Path(__file__).resolve().parents[1]
    / "deploy/policy_control/tools/stationary_contact_guard.py",
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def healthy():
    return {"op": True, "node_ok": True, "stopping": False,
            "al_code": 0, "error": [0] * 6}


def test_free_sample_requires_valid_driver_and_sensor_dimensions():
    assert guard.assess([0] * 6, [0] * 6, [0] * 10, healthy()) is None
    assert guard.assess([0] * 6, [0] * 6, [0] * 10, {}) == "hand_driver_fault"
    assert guard.assess([0] * 5, [0] * 6, [0] * 10, healthy()) == "invalid_sensor_shape"


@pytest.mark.parametrize("field,value", [("error", []), ("op", "false"), ("node_ok", None)])
def test_malformed_driver_is_not_free(field, value):
    status = healthy()
    status[field] = value
    assert guard.assess([0] * 6, [0] * 6, [0] * 10, status) == "hand_driver_fault"


def test_both_force_directions_and_touch_at_threshold_trip():
    for force in (20, -20):
        assert guard.assess([force] * 6, [0] * 6, [0] * 10, healthy()) == "joint_force_change_20g"
    assert guard.assess([0] * 6, [0] * 6, [20] * 10, healthy()) == "finger_touch_20_raw_counts"
    assert guard.assess([float("nan")] * 6, [0] * 6, [0] * 10, healthy()) == "nonfinite_sensor"


def test_no_touch_mode_ignores_missing_tip_sensor():
    """11.10 사용자: 마찰 sweep 엔 손끝 센서가 불필요 — touch 없이도 관절 힘 · 드라이버로만 판정."""
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "deploy/policy_control/tools/stationary_contact_guard.py"
    spec = importlib.util.spec_from_file_location("guard_nt", path)
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    ok = {"op": True, "node_ok": True, "stopping": False, "al_code": 0, "error": [0] * 6}
    assert g.assess([0.0] * 6, [0.0] * 6, None, ok) is None
    assert g.assess([25.0] + [0.0] * 5, [0.0] * 6, None, ok) == "joint_force_change_20g"

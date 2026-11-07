from policy_control.stationary_fake import simulate


def test_load_compensation_improves_and_keeps_position_fixed():
    before = simulate("constant_load", False)
    after = simulate("constant_load", True)
    assert (
        after["summary"]["tip_error_median_mm"]
        < before["summary"]["tip_error_median_mm"]
    )
    assert all(row["motor_position"] == [0.0] * 7 for row in after["rows"])


def test_saturation_cannot_be_called_success_and_delay_is_excluded():
    assert not simulate("saturation", True)["summary"]["pass"]
    assert not simulate("delay", True)["summary"]["eligible"]

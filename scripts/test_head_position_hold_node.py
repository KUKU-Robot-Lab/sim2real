import pytest

from head_position_hold_node import (
    MotorCalibration,
    TICK_MAX,
    deg_to_tick,
    load_calibration_yaml,
    motor_deg_from_center_offset,
    parse_angle_targets,
    parse_ids,
    tick_to_deg,
)


def test_absolute_degrees_map_to_encoder_ticks():
    assert deg_to_tick(0.0) == 0
    assert deg_to_tick(90.0) == 1024
    assert deg_to_tick(180.0) == 2048
    assert deg_to_tick(270.0) == 3071
    assert deg_to_tick(360.0) == TICK_MAX


def test_tick_to_degrees_uses_absolute_encoder_frame():
    assert tick_to_deg(0) == pytest.approx(0.0)
    assert tick_to_deg(2048) == pytest.approx(180.04, abs=0.05)
    assert tick_to_deg(TICK_MAX) == pytest.approx(360.0)


def test_rejects_angles_outside_absolute_range():
    for angle in (-0.1, 360.1):
        with pytest.raises(ValueError):
            deg_to_tick(angle)


def test_parse_ids_requires_unique_motor_ids():
    assert parse_ids("1,2") == (1, 2)
    with pytest.raises(ValueError):
        parse_ids("1")
    with pytest.raises(ValueError):
        parse_ids("1,1")


def test_parse_angle_targets_pairs_ids_with_absolute_degrees():
    assert parse_angle_targets((1, 2), "45,180") == {1: 45.0, 2: 180.0}
    with pytest.raises(ValueError):
        parse_angle_targets((1, 2), "45")
    with pytest.raises(ValueError):
        parse_angle_targets((1, 2), "45,361")


def test_center_offset_uses_midpoint_as_zero():
    pan = MotorCalibration("pan", 1, 45.0, 135.0, False)
    tilt = MotorCalibration("tilt", 2, 105.0, 325.0, False)

    assert motor_deg_from_center_offset(pan, 0.0) == pytest.approx(90.0)
    assert motor_deg_from_center_offset(pan, -30.0) == pytest.approx(60.0)
    assert motor_deg_from_center_offset(pan, 30.0) == pytest.approx(120.0)
    assert motor_deg_from_center_offset(tilt, 0.0) == pytest.approx(215.0)


def test_center_offset_respects_inverted_direction():
    calibration = MotorCalibration("tilt", 2, 105.0, 325.0, True)

    assert motor_deg_from_center_offset(calibration, 30.0) == pytest.approx(185.0)
    assert motor_deg_from_center_offset(calibration, -30.0) == pytest.approx(245.0)


def test_center_offset_rejects_commands_outside_calibrated_range():
    calibration = MotorCalibration("pan", 1, 45.0, 135.0, False)

    with pytest.raises(ValueError):
        motor_deg_from_center_offset(calibration, 46.0)
    with pytest.raises(ValueError):
        motor_deg_from_center_offset(calibration, -46.0)


def test_load_calibration_yaml_reads_saved_ui_format(tmp_path):
    path = tmp_path / "head_dynamixel_calibration.yaml"
    path.write_text(
        "port: /dev/ttyUSB0\n"
        "baud: 1000000\n"
        "motors:\n"
        "  pan:\n"
        "    id: 1\n"
        "    min_deg: 45.0\n"
        "    max_deg: 135.0\n"
        "    inverted: false\n"
        "  tilt:\n"
        "    id: 2\n"
        "    min_deg: 105.0\n"
        "    max_deg: 325.0\n"
        "    inverted: true\n"
    )

    config = load_calibration_yaml(path)

    assert config.port == "/dev/ttyUSB0"
    assert config.baud == 1_000_000
    assert config.motors == {
        "pan": MotorCalibration("pan", 1, 45.0, 135.0, False),
        "tilt": MotorCalibration("tilt", 2, 105.0, 325.0, True),
    }

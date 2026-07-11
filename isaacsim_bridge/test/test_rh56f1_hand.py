"""RH56F1 손 변환 로직 검증. ROS 없이 순수 pytest로 돈다."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaacsim_bridge.rh56f1_hand import (  # noqa: E402
    HAND_DRIVE_DOF,
    SIM_DRIVE_SUFFIXES,
    DriveCalibration,
    HandCalibration,
    default_hand_calibration,
    hand_calibration_from_config,
)


def test_command_order_matches_control_joint_order():
    cal = default_hand_calibration("r_hj_")
    assert cal.sim_joint_order == tuple(f"r_hj_{s}" for s in SIM_DRIVE_SUFFIXES)


def test_rad_endpoints_map_to_register_endpoints():
    # index 손가락: rad 0 -> reg 900, rad 1.5286 -> reg 1740.
    cal = DriveCalibration("r_hj_index_1", 3, 0.0, 1.5285593588966337, 900, 1740)
    assert cal.rad_to_reg(0.0) == 900
    assert cal.rad_to_reg(1.5285593588966337) == 1740
    assert cal.rad_to_reg(0.7642796794483169) == pytest.approx(1320, abs=1)


def test_round_trip_is_near_identity():
    cal = default_hand_calibration("r_hj_")
    rads = [0.3, 0.2, 0.8, 0.9, 1.0, 1.1]
    regs = cal.sim_to_registers(rads)
    back = cal.registers_to_sim(regs)
    for name, rad in zip(cal.sim_joint_order, rads):
        # 레지스터 정수화 때문에 완전 일치는 아니지만 오차는 작다.
        assert back[name] == pytest.approx(rad, abs=0.01)


def test_sim_to_registers_places_values_by_driver_slot():
    cal = default_hand_calibration("r_hj_")
    # 명령 순서: thumb_1, thumb_2, index_1, middle_1, ring_1, pinky_1
    # 슬롯:      5,       4,       3,       2,        1,      0
    rads = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    regs = cal.sim_to_registers(rads)
    assert len(regs) == HAND_DRIVE_DOF
    # 각 관절이 하한(rad 0)이면 그 관절의 reg_lo가 해당 슬롯에 들어가야 한다.
    assert regs[3] == 900  # index -> 슬롯 3
    assert regs[4] == 1100  # thumb_2 -> 슬롯 4
    assert regs[5] == 600  # thumb_1 -> 슬롯 5


def test_out_of_range_angle_is_clamped():
    cal = DriveCalibration("r_hj_index_1", 3, 0.0, 1.5285593588966337, 900, 1740)
    assert cal.rad_to_reg(-1.0) == 900
    assert cal.rad_to_reg(99.0) == 1740


def test_reversed_register_direction_is_supported():
    # reg_lo > reg_hi: 각도 증가에 레지스터 감소.
    cal = DriveCalibration("r_hj_index_1", 3, 0.0, 1.5, 1740, 900)
    assert cal.rad_to_reg(0.0) == 1740
    assert cal.rad_to_reg(1.5) == 900
    # 클램프도 뒤집힌 경계로 동작.
    assert cal.rad_to_reg(99.0) == 900


def test_degenerate_ranges_are_rejected():
    with pytest.raises(ValueError):
        DriveCalibration("j", 0, 0.0, 0.0, 900, 1740)
    with pytest.raises(ValueError):
        DriveCalibration("j", 0, 0.0, 1.0, 900, 900)
    with pytest.raises(ValueError):
        DriveCalibration("j", 9, 0.0, 1.0, 900, 1740)


def test_hand_needs_a_full_slot_permutation():
    dup = tuple(
        DriveCalibration(f"j{i}", 0, 0.0, 1.0, 900, 1740) for i in range(HAND_DRIVE_DOF)
    )
    with pytest.raises(ValueError, match="permutation"):
        HandCalibration(joints=dup)


def test_wrong_length_command_is_rejected():
    cal = default_hand_calibration("r_hj_")
    with pytest.raises(ValueError):
        cal.sim_to_registers([0.0, 0.0])
    with pytest.raises(ValueError):
        cal.registers_to_sim([0, 0, 0])


def test_config_builder_follows_entry_order():
    entries = [
        {"suffix": "index_1", "driver_index": 0, "rad_lo": 0.0, "rad_hi": 1.5, "reg_lo": 900, "reg_hi": 1740},
        {"suffix": "middle_1", "driver_index": 1, "rad_lo": 0.0, "rad_hi": 1.5, "reg_lo": 900, "reg_hi": 1740},
        {"suffix": "ring_1", "driver_index": 2, "rad_lo": 0.0, "rad_hi": 1.5, "reg_lo": 900, "reg_hi": 1740},
        {"suffix": "pinky_1", "driver_index": 3, "rad_lo": 0.0, "rad_hi": 1.5, "reg_lo": 900, "reg_hi": 1740},
        {"suffix": "thumb_2", "driver_index": 4, "rad_lo": 0.0, "rad_hi": 0.47, "reg_lo": 1100, "reg_hi": 1350},
        {"suffix": "thumb_1", "driver_index": 5, "rad_lo": 0.0, "rad_hi": 2.09, "reg_lo": 600, "reg_hi": 1800},
    ]
    cal = hand_calibration_from_config("l_hj_", entries)
    assert cal.sim_joint_order[0] == "l_hj_index_1"
    assert cal.sim_to_registers([0.0] * 6)[0] == 900

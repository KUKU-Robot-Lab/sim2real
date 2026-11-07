"""Experiment gains must match the actual launch description, never just YAML."""

import importlib.util
from pathlib import Path

import pytest

from policy_control import pd_gains as G


def description():
    blocks = []
    for side in ("left", "right"):
        params = "".join(
            f'<param name="{kind}{i}">{value}</param>'
            for kind, value in [("kp", 70), ("kd", 2)]
            for i in range(1, 8)
        )
        blocks.append(
            f'<ros2_control name="openarm_{side}_hardware_interface"><hardware>'
            f"<plugin>openarm_hardware/OpenArmHW</plugin>{params}</hardware></ros2_control>"
        )
    return '<robot name="test">' + "".join(blocks) + "</robot>"


def test_live_description_verification_and_evidence():
    evidence = G.verify_runtime_gains(description(), "right", [70] * 7, [2] * 7)
    assert evidence["kp"] == [70] * 7
    assert len(evidence["robot_description_sha256"]) == 64
    assert evidence["joint_order"] == [f"r_aj_{i}" for i in range(1, 8)]
    with pytest.raises(G.GainsError, match="mismatch"):
        G.verify_runtime_gains(description(), "right", [140] * 7, [2] * 7)


@pytest.mark.parametrize(
    "xml", ["not xml", "<robot/>", description().replace("70", "nan")]
)
def test_live_description_fails_closed(xml):
    with pytest.raises(G.GainsError):
        G.verify_runtime_gains(xml, "right", [70] * 7, [2] * 7)


def test_experiment_override_requires_explicit_profile(tmp_path):
    with pytest.raises(G.GainsError):
        G.GainsBlock(
            tmp_path / "base.yaml", True, arm_gains_override=tmp_path / "override.yaml"
        )
    block = G.GainsBlock(
        tmp_path / "base.yaml",
        True,
        experiment_profile="stationary_tracking",
        arm_gains_override=tmp_path / "override.yaml",
    )
    assert block.require_live_description
    assert not G.GainsBlock(tmp_path / "base.yaml", False).require_live_description


def helper():
    path = (
        Path(__file__).resolve().parents[3]
        / "robot_control/ros_ws/src/openarm_ros2/openarm_bringup/launch/arm_gains.py"
    )
    spec = importlib.util.spec_from_file_location("launch_arm_gains", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bringup_default_byte_identical_and_override_right_only(tmp_path):
    h = helper()
    xml = description()
    assert h.apply_arm_gains_override(xml, "") == xml
    file = tmp_path / "gains.yaml"
    file.write_text("\n".join(f"joint{i}: {{kp: 140, kd: 3}}" for i in range(1, 8)))
    changed = h.apply_arm_gains_override(xml, str(file))
    assert G.verify_runtime_gains(changed, "left", [70] * 7, [2] * 7)
    assert G.verify_runtime_gains(changed, "right", [140] * 7, [3] * 7)


@pytest.mark.parametrize("value", ["nan", "-1", "501", "true"])
def test_bringup_rejects_bad_gains(tmp_path, value):
    file = tmp_path / "gains.yaml"
    file.write_text("\n".join(f"joint{i}: {{kp: {value}, kd: 2}}" for i in range(1, 8)))
    with pytest.raises(ValueError):
        helper().apply_arm_gains_override(description(), str(file))

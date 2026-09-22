#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from copy import deepcopy

from isaacsim_bridge.joint_tuning_report import _build_report, _load_stats


LEFT_ARM_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

RIGHT_ARM_JOINTS = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

RIGHT_HAND_JOINTS = [
    "rj_dg_1_1",
    "rj_dg_1_2",
    "rj_dg_1_3",
    "rj_dg_1_4",
    "rj_dg_2_1",
    "rj_dg_2_2",
    "rj_dg_2_3",
    "rj_dg_2_4",
    "rj_dg_3_1",
    "rj_dg_3_2",
    "rj_dg_3_3",
    "rj_dg_3_4",
    "rj_dg_4_1",
    "rj_dg_4_2",
    "rj_dg_4_3",
    "rj_dg_4_4",
    "rj_dg_5_1",
    "rj_dg_5_2",
    "rj_dg_5_3",
    "rj_dg_5_4",
]

LEFT_GRIPPER_JOINT = "openarm_left_finger_joint1"

DEFAULT_ROBOT_ROOT = "/openarm_dual_modular"


def _default_drive_config() -> dict[str, object]:
    joints: dict[str, dict[str, object]] = {}

    for joint_name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
        joints[joint_name] = {
            "drive_type": "angular",
            "group": "arm",
            "stiffness": 150000.0,
            "damping": 15000.0,
            "recommended_offset_delta": 0.0,
        }

    for joint_name in RIGHT_HAND_JOINTS:
        joints[joint_name] = {
            "drive_type": "angular",
            "group": "hand",
            "stiffness": 40000.0,
            "damping": 4000.0,
            "recommended_offset_delta": 0.0,
        }

    joints[LEFT_GRIPPER_JOINT] = {
        "drive_type": "linear",
        "group": "gripper",
        "stiffness": 40000.0,
        "damping": 4000.0,
        "recommended_offset_delta": 0.0,
    }

    return {
        "robot_root": DEFAULT_ROBOT_ROOT,
        "joints": joints,
    }


def _load_drive_config(path: str) -> dict[str, object]:
    if not path:
        return _default_drive_config()

    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    config = _default_drive_config()
    if isinstance(loaded, dict):
        if isinstance(loaded.get("robot_root"), str):
            config["robot_root"] = loaded["robot_root"]

        loaded_joints = loaded.get("joints", {})
        if isinstance(loaded_joints, dict):
            for joint_name, joint_config in loaded_joints.items():
                if not isinstance(joint_config, dict):
                    continue
                config["joints"].setdefault(joint_name, {})
                config["joints"][joint_name].update(joint_config)

    return config


def _clamp_by_group(group: str, stiffness: float, damping: float) -> tuple[float, float]:
    limits = {
        "arm": (1000.0, 500000.0, 100.0, 100000.0),
        "hand": (500.0, 150000.0, 50.0, 30000.0),
        "gripper": (500.0, 150000.0, 50.0, 30000.0),
    }
    min_stiffness, max_stiffness, min_damping, max_damping = limits.get(
        group, (100.0, 1_000_000.0, 10.0, 200_000.0)
    )
    next_stiffness = min(max(stiffness, min_stiffness), max_stiffness)
    next_damping = min(max(damping, min_damping), max_damping)
    return next_stiffness, next_damping


def _compute_next_drive_config(
    report: dict[str, object],
    current_config: dict[str, object],
) -> dict[str, object]:
    next_config = deepcopy(current_config)
    joints = next_config["joints"]
    assert isinstance(joints, dict)

    report_joints = report.get("joints", {})
    assert isinstance(report_joints, dict)

    for joint_name, joint_config in joints.items():
        if not isinstance(joint_config, dict):
            continue

        summary = report_joints.get(joint_name)
        if not isinstance(summary, dict):
            joint_config["adjustment_reasons"] = ["no data for joint; keep current values"]
            continue

        stiffness = float(joint_config.get("stiffness", 0.0))
        damping = float(joint_config.get("damping", 0.0))
        group = str(joint_config.get("group", "unknown"))

        mean_pos = float(summary.get("mean_position_error", 0.0))
        rmse_pos = float(summary.get("rmse_position_error", 0.0))
        rmse_vel = float(summary.get("rmse_velocity_error", 0.0))
        samples = int(summary.get("samples", 0))

        reasons: list[str] = []
        stiffness_scale = 1.0
        damping_scale = 1.0

        if samples < 20:
            reasons.append("sample count is low; adjustment confidence is low")

        if abs(mean_pos) > 0.03:
            joint_config["recommended_offset_delta"] = -mean_pos
            reasons.append(
                f"apply offset correction of {-mean_pos:.6f} before or alongside gain tuning"
            )
        else:
            joint_config["recommended_offset_delta"] = 0.0

        if rmse_pos > 0.12 and rmse_vel < 0.4:
            stiffness_scale *= 1.15
            reasons.append("high position error without strong velocity mismatch -> stiffness +15%")
        elif rmse_pos > 0.06 and rmse_vel < 0.3:
            stiffness_scale *= 1.08
            reasons.append("moderate position lag -> stiffness +8%")

        if rmse_vel > 0.6:
            damping_scale *= 1.15
            reasons.append("high velocity mismatch -> damping +15%")
        elif rmse_vel > 0.35:
            damping_scale *= 1.08
            reasons.append("moderate velocity mismatch -> damping +8%")

        if rmse_pos < 0.02 and rmse_vel < 0.1 and samples >= 20:
            reasons.append("tracking already tight -> keep gains unchanged")

        next_stiffness = stiffness * stiffness_scale
        next_damping = damping * damping_scale
        next_stiffness, next_damping = _clamp_by_group(group, next_stiffness, next_damping)

        joint_config["stiffness"] = next_stiffness
        joint_config["damping"] = next_damping
        joint_config["adjustment_reasons"] = reasons or ["no strong signal; keep current values"]

    next_config["report_snapshot"] = {
        "joint_count": report.get("joint_count", 0),
        "total_samples": report.get("total_samples", 0),
        "global_rmse_position_error": report.get("global_rmse_position_error", 0.0),
        "global_rmse_velocity_error": report.get("global_rmse_velocity_error", 0.0),
    }
    return next_config


def _print_next_config(config: dict[str, object], report: dict[str, object]) -> None:
    print(
        "Cycle:"
        f" joints={report.get('joint_count', 0)}"
        f" samples={report.get('total_samples', 0)}"
        f" rmse_pos={float(report.get('global_rmse_position_error', 0.0)):.5f}"
        f" rmse_vel={float(report.get('global_rmse_velocity_error', 0.0)):.5f}"
    )

    joints = config.get("joints", {})
    assert isinstance(joints, dict)
    for joint_name, joint_config in joints.items():
        if not isinstance(joint_config, dict):
            continue
        reasons = joint_config.get("adjustment_reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        print(
            f"{joint_name}:"
            f" stiffness={float(joint_config.get('stiffness', 0.0)):.3f}"
            f" damping={float(joint_config.get('damping', 0.0)):.3f}"
            f" offset_delta={float(joint_config.get('recommended_offset_delta', 0.0)):.6f}"
        )
        for reason in reasons:
            print(f"  - {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one automatic tuning cycle from recorded joint error CSV."
    )
    parser.add_argument(
        "--input-csv",
        default="/tmp/isaacsim_joint_error.csv",
        help="CSV from joint_error_recorder",
    )
    parser.add_argument(
        "--input-drive-config",
        default="",
        help="Optional previous drive config JSON. Defaults to built-in baseline.",
    )
    parser.add_argument(
        "--output-report",
        default="/tmp/isaacsim_joint_tuning_report.json",
        help="JSON report snapshot output path",
    )
    parser.add_argument(
        "--output-drive-config",
        default="/tmp/isaacsim_next_joint_drive_config.json",
        help="Next Isaac Sim drive config JSON output path",
    )
    args = parser.parse_args()

    report = _build_report(_load_stats(args.input_csv))
    current_config = _load_drive_config(args.input_drive_config)
    next_config = _compute_next_drive_config(report, current_config)

    with open(args.output_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with open(args.output_drive_config, "w", encoding="utf-8") as handle:
        json.dump(next_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _print_next_config(next_config, report)
    print(f"Wrote report JSON: {args.output_report}")
    print(f"Wrote next drive config JSON: {args.output_drive_config}")


if __name__ == "__main__":
    main()

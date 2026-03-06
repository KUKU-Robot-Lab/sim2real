#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass


@dataclass
class JointStats:
    count: int = 0
    pos_sum: float = 0.0
    pos_sq_sum: float = 0.0
    vel_sum: float = 0.0
    vel_sq_sum: float = 0.0
    effort_sum: float = 0.0
    effort_sq_sum: float = 0.0
    max_abs_pos: float = 0.0
    max_abs_vel: float = 0.0
    max_abs_effort: float = 0.0

    def add(self, position_error: float, velocity_error: float, effort_error: float) -> None:
        self.count += 1
        self.pos_sum += position_error
        self.pos_sq_sum += position_error * position_error
        self.vel_sum += velocity_error
        self.vel_sq_sum += velocity_error * velocity_error
        self.effort_sum += effort_error
        self.effort_sq_sum += effort_error * effort_error
        self.max_abs_pos = max(self.max_abs_pos, abs(position_error))
        self.max_abs_vel = max(self.max_abs_vel, abs(velocity_error))
        self.max_abs_effort = max(self.max_abs_effort, abs(effort_error))

    def to_summary(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "samples": 0,
                "mean_position_error": 0.0,
                "rmse_position_error": 0.0,
                "max_abs_position_error": 0.0,
                "mean_velocity_error": 0.0,
                "rmse_velocity_error": 0.0,
                "max_abs_velocity_error": 0.0,
                "mean_effort_error": 0.0,
                "rmse_effort_error": 0.0,
                "max_abs_effort_error": 0.0,
            }

        return {
            "samples": self.count,
            "mean_position_error": self.pos_sum / self.count,
            "rmse_position_error": math.sqrt(self.pos_sq_sum / self.count),
            "max_abs_position_error": self.max_abs_pos,
            "mean_velocity_error": self.vel_sum / self.count,
            "rmse_velocity_error": math.sqrt(self.vel_sq_sum / self.count),
            "max_abs_velocity_error": self.max_abs_vel,
            "mean_effort_error": self.effort_sum / self.count,
            "rmse_effort_error": math.sqrt(self.effort_sq_sum / self.count),
            "max_abs_effort_error": self.max_abs_effort,
        }


def _load_stats(path: str) -> dict[str, JointStats]:
    stats: dict[str, JointStats] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            joint_name = (row.get("joint") or "").strip()
            if not joint_name:
                continue

            try:
                position_error = float(row["position_error"])
                velocity_error = float(row["velocity_error"])
                effort_error = float(row["effort_error"])
            except (KeyError, ValueError):
                continue

            joint_stats = stats.setdefault(joint_name, JointStats())
            joint_stats.add(position_error, velocity_error, effort_error)

    return stats


def _recommendations(summary: dict[str, float]) -> list[str]:
    notes: list[str] = []

    mean_pos = abs(summary["mean_position_error"])
    rmse_pos = summary["rmse_position_error"]
    rmse_vel = summary["rmse_velocity_error"]
    max_pos = summary["max_abs_position_error"]

    if summary["samples"] < 20:
        notes.append("more data needed: sample count is low")

    if mean_pos > 0.08 and rmse_pos < 0.12:
        notes.append("check joint zero offset, sign, or scale before gain tuning")

    if max_pos > 0.2:
        notes.append("large peak position error: verify joint limits and command scaling")

    if rmse_pos > 0.12 and rmse_vel < 0.4:
        notes.append("tracking is too soft: increase stiffness/kp by 10-20%")
    elif rmse_pos < 0.03 and rmse_vel < 0.15:
        notes.append("tracking looks stable: keep stiffness unchanged for now")

    if rmse_vel > 0.6:
        notes.append("velocity mismatch is high: increase damping/kd by 10-20%")
    elif rmse_vel < 0.2 and rmse_pos > 0.08:
        notes.append("response is slow without oscillation: stiffness likely too low")

    if mean_pos > 0.03 and rmse_vel > 0.6:
        notes.append("possible lag or friction mismatch: tune damping with lower command speed")

    if not notes:
        notes.append("no strong heuristic signal; inspect commanded trajectory and joint mapping")

    return notes


def _build_report(stats: dict[str, JointStats]) -> dict[str, object]:
    joint_reports: dict[str, dict[str, object]] = {}

    global_position_sq_sum = 0.0
    global_velocity_sq_sum = 0.0
    total_samples = 0

    for joint_name in sorted(stats):
        summary = stats[joint_name].to_summary()
        joint_reports[joint_name] = {
            **summary,
            "recommendations": _recommendations(summary),
        }
        global_position_sq_sum += stats[joint_name].pos_sq_sum
        global_velocity_sq_sum += stats[joint_name].vel_sq_sum
        total_samples += stats[joint_name].count

    report = {
        "joint_count": len(joint_reports),
        "total_samples": total_samples,
        "global_rmse_position_error": (
            math.sqrt(global_position_sq_sum / total_samples) if total_samples else 0.0
        ),
        "global_rmse_velocity_error": (
            math.sqrt(global_velocity_sq_sum / total_samples) if total_samples else 0.0
        ),
        "joints": joint_reports,
    }
    return report


def _print_report(report: dict[str, object]) -> None:
    print(
        "Global:"
        f" joints={report['joint_count']}"
        f" samples={report['total_samples']}"
        f" rmse_pos={report['global_rmse_position_error']:.5f}"
        f" rmse_vel={report['global_rmse_velocity_error']:.5f}"
    )

    joints = report["joints"]
    assert isinstance(joints, dict)
    for joint_name, joint_data in joints.items():
        assert isinstance(joint_data, dict)
        print(
            f"{joint_name}:"
            f" samples={joint_data['samples']}"
            f" mean_pos={joint_data['mean_position_error']:.5f}"
            f" rmse_pos={joint_data['rmse_position_error']:.5f}"
            f" rmse_vel={joint_data['rmse_velocity_error']:.5f}"
            f" max_pos={joint_data['max_abs_position_error']:.5f}"
        )
        recommendations = joint_data["recommendations"]
        assert isinstance(recommendations, list)
        for note in recommendations:
            print(f"  - {note}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze sim-vs-real joint error CSV and generate tuning hints."
    )
    parser.add_argument(
        "--input",
        default="/tmp/isaacsim_joint_error.csv",
        help="Path to CSV produced by joint_error_recorder",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON report output path",
    )
    args = parser.parse_args()

    stats = _load_stats(args.input)
    report = _build_report(stats)
    _print_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Wrote JSON report: {args.output}")


if __name__ == "__main__":
    main()

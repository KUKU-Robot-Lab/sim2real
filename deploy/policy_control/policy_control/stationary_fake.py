"""CPU-only fault-injection fixture. Toy dynamics do not establish real stability."""

import json
from pathlib import Path

import numpy as np

from .pd_law import PdCommand
from .pd_stationary import StationaryRuntime
from . import stationary_tracking as S

# DM MIT velocity resolution seen on the real arm [rad/s] (j1/j2, j3/j4, j5~j7).
VEL_STEP = np.array([0.022, 0.022, 0.0049, 0.0049, 0.0147, 0.0147, 0.0147])


def quantize_velocity(v):
    step = VEL_STEP[: len(v)]
    return np.round(np.asarray(v) / step) * step


CASES = (
    "constant_load",
    "payload_bias",
    "friction_quantization",
    "saturation",
    "delay",
)


def simulate(case, enabled):
    if case not in CASES:
        raise ValueError(case)
    dt, kp = 0.01, np.ones(7) * 70.0
    load = np.ones(7) * (0.6 if case == "saturation" else 0.2)
    limit = np.ones(7) * (0.1 if case == "saturation" else 7.0)
    config = {"integral_enabled": enabled, "ki": (kp / 5).tolist()}
    controller = StationaryRuntime(
        config, kp=kp, kd=np.ones(7) * 8.0, torque_limit=limit, blend_sec=1.0
    )
    q, velocity = -load / kp, np.zeros(7)
    command = PdCommand(np.zeros(7), np.zeros(7), np.zeros(7), (), False)
    rows = []
    for i in range(1501):
        measured = (
            np.round(q / 0.00038) * 0.00038
            if case == "friction_quantization"
            else q.copy()
        )
        delayed = case == "delay" and 500 <= i < 510
        if case == "payload_bias":
            from dataclasses import replace

            command = replace(command, tau=0.15 * np.cos(measured))
        cmd, row = controller.apply(
            command,
            q_meas=measured,
            qd_meas=quantize_velocity(velocity),
            target=np.zeros(7),
            target_velocity=np.zeros(7),
            phase="TRACKING",
            contact=False,
            stale=delayed,
            episode=1,
            dt=dt,
            gravity=command.tau,
            request=np.zeros(7),
        )
        torque = kp * (-measured) - 8.0 * velocity + cmd.tau
        torque = np.clip(torque, -limit, limit)
        friction = (
            0.08 * np.tanh(velocity / 0.01) if case == "friction_quantization" else 0.0
        )
        actual_load = 0.35 * np.cos(q) if case == "payload_bias" else load
        velocity += (torque - actual_load - friction) * dt
        q += velocity * dt
        row.update(
            stamp=i * dt,
            pd_target=[0.0] * 7,
            hand_names=[],
            measured_hand=[],
            temperature=[30.0] * 7,
            torque_feedback=torque.tolist(),
        )
        rows.append(row)

    def fake_fk(q, hand):
        return np.array([q[0] * 0.5, q[1] * 0.5, q[2] * 0.5])

    summary, t, tip = S.analyze_samples(rows, fake_fk, path_end=0.0)
    summary.update(
        injected_condition=case,
        compensation_enabled=enabled,
        source="toy CPU plant; injected cause is known, not inferred from telemetry",
        real_hardware_stability_proven=False,
    )
    return {"summary": summary, "rows": rows, "t": t, "tip": tip, "config": config}


def simulate_static_friction(fs, *, coulomb=0.6, ki=0.0, seconds=8.0):
    """4-joint toy arm with stiction: stuck while |PD + ff| <= coulomb.

    Starts at rest, stuck at 0.9·coulomb/kp off target (alternating sign) like
    the real arm after a move; returns measured error [deg] per tick.
    Unit inertia, 120 Hz, 0.00038 rad encoder step.
    """
    dt, kp, kd = 1.0 / 120.0, np.array([70.0, 70.0, 70.0, 60.0]), np.ones(4) * 8.0
    config = {
        "fs": [fs] * 4,
        "integral_enabled": ki > 0,
        "ki": [ki] * 4,
        "integral_limit": [1.0] * 4,  # the 7-joint defaults for j1~j4
        "integral_rate": [0.2] * 4,
    }
    controller = StationaryRuntime(
        config, kp=kp, kd=kd, torque_limit=np.ones(4) * 7.0, blend_sec=1.0
    )
    q = 0.9 * coulomb / kp * np.array([1.0, -1.0, 1.0, -1.0])
    velocity = np.zeros(4)
    command = PdCommand(np.zeros(4), np.zeros(4), np.zeros(4), (), False)
    errors = []
    for _ in range(int(seconds / dt)):
        measured = np.round(q / 0.00038) * 0.00038
        cmd, _row = controller.apply(
            command, q_meas=measured, qd_meas=quantize_velocity(velocity), target=np.zeros(4),
            target_velocity=np.zeros(4), phase="TRACKING", contact=False,
            stale=False, episode=1, dt=dt, gravity=np.zeros(4), request=np.zeros(4),
        )
        torque = kp * (-measured) - kd * velocity + cmd.tau
        stuck = (velocity == 0.0) & (np.abs(torque) <= coulomb)
        direction = np.where(velocity != 0.0, np.sign(velocity), np.sign(torque))
        new_velocity = velocity + (torque - coulomb * direction) * dt
        # Kinetic friction stops the joint; it cannot push it backwards.
        new_velocity[stuck | (np.sign(new_velocity) == -direction)] = 0.0
        velocity = new_velocity
        q = q + velocity * dt
        errors.append(np.rad2deg(-measured))
    return np.asarray(errors)


def simulate_move(config, move_deg, *, f_small=0.3, f_large=0.8, split_deg=2.0,
                  offset=None, seconds=8.0):
    """4-joint toy arm moving move_deg onto a held target with stiction that grows
    with travel: coulomb f_small until the joint has moved split_deg, then f_large.

    Returns measured error [deg] per tick (120 Hz, unit inertia, 0.00038 rad
    encoder step). offset: constant load torque per joint (e.g. the j1 offset).
    """
    dt, kp, kd = 1.0 / 120.0, np.array([70.0, 70.0, 70.0, 60.0]), np.ones(4) * 8.0
    controller = StationaryRuntime(
        config, kp=kp, kd=kd, torque_limit=np.ones(4) * 7.0, blend_sec=1.0
    )
    load = np.zeros(4) if offset is None else np.asarray(offset, dtype=float)
    q = np.deg2rad(move_deg) * np.array([1.0, -1.0, 1.0, -1.0])
    velocity, travelled = np.zeros(4), np.zeros(4)
    command = PdCommand(np.zeros(4), np.zeros(4), np.zeros(4), (), False)
    errors = []
    for _ in range(int(seconds / dt)):
        measured = np.round(q / 0.00038) * 0.00038
        cmd, _row = controller.apply(
            command, q_meas=measured, qd_meas=quantize_velocity(velocity), target=np.zeros(4),
            target_velocity=np.zeros(4), phase="TRACKING", contact=False,
            stale=False, episode=1, dt=dt, gravity=np.zeros(4), request=np.zeros(4),
        )
        coulomb = np.where(np.rad2deg(travelled) > split_deg, f_large, f_small)
        torque = kp * (-measured) - kd * velocity + cmd.tau - load
        stuck = (velocity == 0.0) & (np.abs(torque) <= coulomb)
        direction = np.where(velocity != 0.0, np.sign(velocity), np.sign(torque))
        new_velocity = velocity + (torque - coulomb * direction) * dt
        new_velocity[stuck | (np.sign(new_velocity) == -direction)] = 0.0
        velocity = new_velocity
        q = q + velocity * dt
        travelled += np.abs(velocity) * dt
        errors.append(np.rad2deg(-measured))
    return np.asarray(errors)


def move_metrics(errors, dt=1.0 / 120.0, band_deg=0.2):
    """Per joint: final |error|, time it enters band_deg for good (nan = never),
    overshoot past the target [deg], and error sign flips (chatter)."""
    start = np.sign(errors[0])
    out_band = np.abs(errors) >= band_deg
    settle = [
        np.nan if out_band[-1, j] else (np.flatnonzero(out_band[:, j])[-1] + 1) * dt
        if out_band[:, j].any() else 0.0
        for j in range(errors.shape[1])
    ]
    signs = np.sign(errors)
    flips = [int(np.sum(np.diff(s[s != 0]) != 0)) for s in signs.T]
    return {
        "final_deg": np.abs(errors[-1]).round(3).tolist(),
        "settle_s": [round(float(x), 3) for x in settle],
        "overshoot_deg": np.maximum(-(errors * start).min(axis=0), 0.0).round(3).tolist(),
        "flips": flips,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    comparisons = []
    for case in CASES:
        for enabled in (False, True):
            result = simulate(case, enabled)
            out = args.out / f"{case}_{'on' if enabled else 'off'}"
            S.new_run(
                out,
                {"model": Path(__file__)},
                {
                    "fake": True,
                    "case": case,
                    "compensation": result["config"],
                    "episode": 1,
                },
                repo=root,
            )
            with (out / "events.jsonl").open("x") as stream:
                for row in result["rows"]:
                    stream.write(
                        json.dumps(
                            {
                                "topic": "/policy_control/pd_right/diagnostics",
                                "recv_wall_ns": int(row["stamp"] * 1e9),
                                "recv_mono_ns": int(row["stamp"] * 1e9),
                                "message": {"data": json.dumps(row)},
                            }
                        )
                        + "\n"
                    )
            S.write_report(out, result["summary"], result["t"], result["tip"])
            comparisons.append(result["summary"])
    (args.out / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "injected_condition",
                        "compensation_enabled",
                        "eligible",
                        "pass",
                        "tip_error_median_mm",
                    )
                }
                for r in comparisons
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

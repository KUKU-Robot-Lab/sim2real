"""CPU-only fault-injection fixture. Toy dynamics do not establish real stability."""

import json
from pathlib import Path

import numpy as np

from .pd_law import PdCommand
from .pd_stationary import StationaryRuntime
from . import stationary_tracking as S

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
            qd_meas=velocity,
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

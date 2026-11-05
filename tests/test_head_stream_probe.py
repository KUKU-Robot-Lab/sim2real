import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import head_stream_probe as hp  # noqa: E402

PAN0, TILT0 = -19.56, 67.473
DELAY, TAU = 0.050, 0.100


def _fopdt(t_u, u, t_out, y0, dt=1e-4):
    """정답 1차계: 0.1 ms 격자, 입력 ZOH, 순수 지연 DELAY."""
    grid = np.arange(t_u[0], t_out[-1] + dt, dt)
    uu = u[np.clip(np.searchsorted(t_u, grid - DELAY, side="right") - 1, 0, None)]
    a, y, out = 1 - np.exp(-dt / TAU), y0, np.empty(len(grid))
    for i, v in enumerate(uu):
        out[i] = y
        y += a * (v - y)
    return np.interp(t_out, grid, out)


def _synthetic(folder: Path) -> None:
    plan = hp.build_plan(PAN0, TILT0)
    t0 = 1000.0
    tc = np.array([t0 + p[0] for p in plan])
    stamps = np.arange(t0 - 0.5, tc[-1] + 1.0, 1 / 20) + 0.013  # 20 Hz, 목표와 어긋난 위상
    rng = np.random.default_rng(0)
    meas = {a: _fopdt(tc, np.array([p[i] for p in plan]), stamps, v) + rng.normal(0, 0.02, len(stamps))
            for a, i, v in (("pan", 2, PAN0), ("tilt", 3, TILT0))}
    with open(folder / "targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_send", "seq", "seg", "pan_deg", "tilt_deg"])
        w.writerows((t, k, p[1], p[2], p[3]) for k, (t, p) in enumerate(zip(tc, plan)))
    with open(folder / "joint_states.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp", "recv_t", "pan_deg", "tilt_deg"])
        w.writerows(zip(stamps, stamps + 0.002, meas["pan"], meas["tilt"]))
    with open(folder / "stream_log.jsonl", "w") as f:
        for k, (t, p) in enumerate(zip(tc, plan)):
            f.write(json.dumps({"seq": k, "recv_t": t + 0.001, "write_t": t + 0.005, "goal_pan_deg": p[2],
                                "goal_tilt_deg": p[3], "clamp": {"pan": "rate" if k % 10 == 0 else "", "tilt": ""}}) + "\n")


def test_analyze_recovers_delay_and_tau(tmp_path):
    _synthetic(tmp_path)
    s = hp.analyze(tmp_path)
    for axis in ("pan", "tilt"):
        fit = s["fopdt"][axis]["target"]
        assert abs(fit["delay_ms"] - DELAY * 1e3) <= 10, fit
        assert abs(fit["tau_s"] - TAU) <= 0.010, fit
        goal = s["fopdt"][axis]["goal"]  # goal 은 목표보다 5 ms 늦게 쓰였다
        assert abs(goal["delay_ms"] - (DELAY * 1e3 - 5)) <= 10, goal
    steps = {r["seg"]: r for r in s["steps"]}
    assert set(steps) >= {"pan+5", "back1", "pan-5", "tilt+5", "tilt-5"}
    assert steps["tilt+5"]["axis"] == "tilt"
    # 0.2° 넘는 시각 = 지연 + τ·ln(1/(1−0.04)) ≈ 54 ms → 그 뒤 첫 stamp(20 Hz), 상승 10→90 % = τ·ln 9 ≈ 220 ms
    assert 54 <= steps["pan+5"]["deadtime_ms"] <= 54 + 50
    assert abs(steps["pan+5"]["rise_ms"] - 220) <= 20
    assert abs(steps["pan+5"]["settle_err_deg"]) < 0.1
    assert s["still"][0]["pan_std_deg"] < 0.05
    sine = next(r for r in s["tracking"] if r["seg"] == "pan_sine")
    # 사인 지연(보낸 ZOH 목표 기준) = 지연 + atan(ωτ)/ω ≈ 50 + 97 = 147 ms
    assert abs(sine["lag_ms"] - 147) <= 15, sine
    assert s["stream_log"]["clamp_any_frac"] == 0.1
    assert abs(s["stream_log"]["recv_to_write_ms"]["p50"] - 4.0) < 0.01
    assert (tmp_path / "summary.json").exists()


def test_plan_out_of_stream_range_rejected():
    assert hp.plan_problem(hp.build_plan(PAN0, TILT0)) == ""
    assert "pan" in hp.plan_problem(hp.build_plan(PAN0, TILT0, amp=2.0))  # 삼각파 −49° < −45°
    assert "tilt" in hp.plan_problem(hp.build_plan(PAN0, 80.0))


def test_start_pose_far_from_camset_rejected():
    assert hp.start_problem(PAN0 + 1.9, TILT0 - 1.9, PAN0, TILT0) == ""
    assert hp.start_problem(PAN0, TILT0 + 2.1, PAN0, TILT0) != ""

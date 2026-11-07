import numpy as np
import pytest
from policy_control import stationary_tracking as s


def test_alignment_uses_names_and_refuses_old_sample():
    idx, gap = s.nearest_indices([0.0, 0.04], [0.001, 0.02, 0.07])
    assert idx.tolist() == [0, 0, -1]
    assert gap[-1] == pytest.approx(0.03)
    assert s.named_values(
        {"name": ["b", "a"], "position": [2, 1]}, ["a", "b"]
    ).tolist() == [1, 2]
    with pytest.raises(ValueError):
        s.named_values({"name": ["a", "a"], "position": [1, 2]}, ["a"])


def test_settling_requires_full_contiguous_hold_and_joint_diagnostics():
    t = np.arange(0, 15.001, 0.01)
    err = np.where(t < 2, 2.0, 0.5)
    q = np.zeros((len(t), 7))
    result = s.evaluate(t, err, q, q, path_end=0.0)
    assert result["pass"] and result["settling_s"] == 2.0
    assert not s.evaluate(t[t < 11], err[t < 11], q[t < 11], q[t < 11], path_end=0.0)[
        "pass"
    ]
    keep = (t < 6) | (t > 6.1)
    assert not s.evaluate(t[keep], err[keep], q[keep], q[keep], path_end=0.0)["pass"]
    q[:, 1] = np.radians(0.2)
    result = s.evaluate(t, err, np.zeros_like(q), q, path_end=0.0)
    assert result["tracking_pass"] and not result["pass"]


def test_invalid_contact_or_missing_state_excludes_pass():
    t = np.arange(0, 15.001, 0.01)
    q = np.zeros((len(t), 7))
    for why in [
        "contact",
        "stale",
        "protection",
        "target_changed",
        "missing_required_state",
    ]:
        r = s.evaluate(t, np.zeros(len(t)), q, q, path_end=0.0, invalid_reasons=[why])
        assert r["pass"] is False and r["eligible"] is False
    q[500, 0] = np.nan
    assert not s.evaluate(t, np.zeros(len(t)), q, q, path_end=0.0)["eligible"]


def test_prepare_split_paths_limits_and_exclusions():
    poses = [
        {"id": f"p{i}", "split": "fit" if i < 3 else "holdout", "q": [0.0] * 7}
        for i in range(6)
    ]

    def smooth(path):
        return np.linspace(path[0], path[-1], 100)

    plan = s.prepare_trials(
        poses,
        np.full(7, -0.03),
        np.full(7, 0.1),
        smooth,
        lambda path: {"ok": True},
        stage="A",
    )
    assert len(plan["trials"]) == 63
    assert len(plan["excluded"]) == 63
    assert all(t["split"] == "fit" for t in plan["trials"])
    assert all(t["hold_s"] == 15 for t in plan["trials"])
    with pytest.raises(ValueError):
        s.prepare_trials(
            poses[:5], np.full(7, -1), np.ones(7), smooth, lambda p: {"ok": True}
        )


def test_manifest_hashes_inputs_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    source.write_text("x")
    out = tmp_path / "run"
    m = s.new_run(out, {"urdf": source}, {"stage": "A"}, repo=tmp_path)
    assert len(m["inputs"]["urdf"]["sha256"]) == 64 and m["commands_sent"] == 0
    with pytest.raises(FileExistsError):
        s.new_run(out, {}, {}, repo=tmp_path)


def test_candidate_selection_prefers_low_gain_then_rms():
    rows = [
        {"pass": True, "kp_multiplier": [k] * 7, "additional_torque_rms": r}
        for k, r in [(2, 0.1), (1.25, 0.4), (1.25, 0.2)]
    ]
    assert s.select_candidate(rows) == rows[2]


def test_legacy_samples_report_unmatched_and_never_infer_a_pass():
    rows = []
    for topic, time, names in [
        ("/policy_control/pd_right/applied", 1.0, list(s.JOINTS)),
        ("/joint_states", 1.001, [f"openarm_right_joint{i}" for i in range(1, 8)]),
        ("/hand_right/joint_states", 1.0, ["r_hj_index_1"]),
    ]:
        rows.append(
            {
                "topic": topic,
                "recv_wall_ns": int(time * 1e9),
                "header_stamp_ns": int(time * 1e9),
                "message": {"name": names, "position": [0.0] * len(names)},
            }
        )
    samples, info = s.aligned_samples(rows)
    assert info["legacy"] and info["matched"] == 1
    assert samples[0]["contact"] is None
    assert samples[0]["user_target"] is None
    rows[1]["header_stamp_ns"] = int(1.03e9)
    samples, info = s.aligned_samples(rows)
    assert info["matched"] == 0 and info["unmatched_fraction"] == 1.0


def test_stage_blocks_enforce_one_variable_and_integral_limits():
    s.validate_stage("A", [70.0] * 7, [2.0] * 7, [70.0] * 7, [2.0] * 7, None)
    with pytest.raises(ValueError):
        s.validate_stage(
            "A",
            [70.0] * 7,
            [2.0] * 7,
            [70.0] * 7,
            [2.0] * 7,
            {"integral_enabled": True},
        )
    with pytest.raises(ValueError):
        s.validate_stage("B", [140.0] * 7, [2.0] * 7, [70.0] * 7, [2.0] * 7, None)
    s.validate_stage("B", [140.0] + [70.0] * 6, [2.0] * 7, [70.0] * 7, [2.0] * 7, None)


def test_capture_pose_needs_only_measured_states_not_a_pd_command():
    rows = [
        {
            "topic": "/joint_states",
            "recv_wall_ns": 1000000000,
            "header_stamp_ns": 1000000000,
            "message": {
                "name": [f"openarm_right_joint{i}" for i in range(1, 8)],
                "position": [0.1] * 7,
            },
        },
        {
            "topic": "/hand_right/joint_states",
            "recv_wall_ns": 1001000000,
            "header_stamp_ns": 1001000000,
            "message": {"name": ["r_hj_index_1"], "position": [0.2]},
        },
    ]
    pose = s.capture_pose(rows)
    assert pose["q"] == [0.1] * 7 and pose["hand"] == {"r_hj_index_1": 0.2}
    rows[-1]["header_stamp_ns"] = 1100000000
    with pytest.raises(ValueError):
        s.capture_pose(rows)


def test_live_gate_rejects_nonfinite_or_missing_required_diagnostics():
    d = {
        "contact": False,
        "stale": False,
        "protected": False,
        "temperature": [30.0] * 7,
        "torque_feedback": [0.0] * 7,
        "measured": [0.0] * 7,
        "velocity": [0.0] * 7,
    }
    s.validate_live_diagnostics(d)
    for field in ("temperature", "torque_feedback", "measured", "velocity"):
        bad = dict(d, **{field: [float("nan")] * 7})
        with pytest.raises(ValueError):
            s.validate_live_diagnostics(bad)
    with pytest.raises(ValueError):
        s.validate_live_diagnostics({k: v for k, v in d.items() if k != "protected"})


def test_planned_endpoint_uses_declared_target_despite_one_tick_transport_delay():
    rows = [
        {
            "stamp": float(t),
            "user_target": [0.0001] * 7 if t == 0 else [0.0] * 7,
            "measured": [0.0] * 7,
            "hand_names": [],
            "measured_hand": [],
            "contact": False,
            "stale": False,
            "protected": False,
            "temperature": [30.0] * 7,
            "torque_feedback": [0.0] * 7,
        }
        for t in np.arange(0, 15.001, 0.01)
    ]
    summary, _, _ = s.analyze_samples(
        rows, lambda q, h: q[:3], path_end=0.0, expected_target=[0.0] * 7
    )
    assert summary["pass"]
    rows[100]["user_target"] = [0.01] * 7
    summary, _, _ = s.analyze_samples(
        rows, lambda q, h: q[:3], path_end=0.0, expected_target=[0.0] * 7
    )
    assert not summary["eligible"] and "target_changed" in summary["excluded_reasons"]

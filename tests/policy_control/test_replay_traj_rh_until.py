"""replay_traj_rh --until: 구간 라벨 끝(포함)까지 q · cup · segments 를 자르고, 없는 라벨은 거부."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"))
import replay_traj_rh as RT  # noqa: E402


def test_until_cuts_at_segment_end_and_rejects_unknown():
    segs = tuple({"label": k, "start": a, "end": b, "hand_command": "position"}
                 for k, a, b in (("approach", 0, 3), ("descend", 4, 6), ("release", 7, 9)))
    traj = RT.R.ReplayTraj(side="right", names=RT.R.expected_names("right"), q=np.arange(10 * 13.0).reshape(10, 13),
                           step_dt=1 / 60, segments=segs, arm_src="arm_q", hand_src="hand_q_cmd", cup=np.zeros((10, 7)))
    t = RT.until(traj, "descend")
    assert t.n_frames == 7 and t.cup.shape[0] == 7 and [s["label"] for s in t.segments] == ["approach", "descend"]
    assert np.array_equal(t.q, traj.q[:7])
    with pytest.raises(SystemExit):
        RT.until(traj, "pour")


def test_hand_start_applied_missing_exits_without_measured_fallback():
    arm = [f"openarm_right_joint{i}" for i in range(1, 8)]
    hand = list(RT.R.expected_names("right")[7:])
    arm_m, hand_m = dict.fromkeys(arm, 0.0), dict.fromkeys(hand, 0.5)
    with pytest.raises(SystemExit, match="손 applied"):
        RT.start_hand(arm_m, hand_m, {}, arm, hand, "applied", "TRACKING")
    assert RT.start_hand(arm_m, hand_m, {}, arm, hand, "measured", "TRACKING") is hand_m
    applied = dict.fromkeys(hand, 1.0)
    assert RT.start_hand(arm_m, hand_m, applied, arm, hand, "applied", "TRACKING") is applied

"""Bounded local fallback IK tests; synthetic kinematics, no ROS."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/policy_control/tools"))
from bounded_jog_ik import solve  # noqa: E402


class RedundantKin:
    def tip(self, q, h=None):
        T = np.eye(4)
        T[:3, 3] = [q[0] + 2 * q[6] ** 2, q[1], q[2]]
        T[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return T


def problem():
    kin = RedundantKin()
    q = np.zeros(7)
    lo, hi = np.full(7, -.5), np.full(7, .5)
    hi[0] = .01
    target = kin.tip(q)
    target[:3, 3] = [.08, .02, -.03]
    target[:3, :3] = Rotation.from_rotvec([.03, -.02, .01]).as_matrix()
    return kin, q, target, lo, hi


def test_redundancy_finds_exact_pose_without_relaxing_limits():
    kin, q0, target, lo, hi = problem()
    # DLS at q6=0 would move q0 to .08, outside its .01 upper limit.
    q, why = solve(kin, q0, target, lo, hi)
    assert q is not None, why
    assert np.all(q >= lo) and np.all(q <= hi)
    assert np.max(np.abs(q-q0)) <= .3
    actual = kin.tip(q)
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) <= .0002
    assert np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()) <= np.radians(.1)
    assert abs(q[6]) > .15


def test_unreachable_target_is_not_clipped_into_fake_success():
    kin, q0, target, lo, hi = problem()
    target[0, 3] = 1
    q, why = solve(kin, q0, target, lo, hi)
    assert q is None and why


def test_local_continuity_is_a_hard_bound_even_with_wide_joint_limits():
    kin, q0, target, lo, hi = problem()
    target[1, 3] = .4
    q, why = solve(kin, q0, target, lo, hi)
    assert q is None and why


@pytest.mark.parametrize("which", ["nan_seed", "invalid_limits", "bad_pose"])
def test_invalid_inputs_rejected(which):
    kin, q0, target, lo, hi = problem()
    if which == "nan_seed":
        q0[0] = np.nan
    elif which == "invalid_limits":
        hi[0] = lo[0]
    else:
        target[0, 0] = 2
    with pytest.raises(ValueError):
        solve(kin, q0, target, lo, hi)

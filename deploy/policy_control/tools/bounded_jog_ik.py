"""Bounded local pose IK fallback. No orientation/limit relaxation or ROS.

Use after the normal fast IK fails, outside the high-rate feedback loop.
Deterministic nullspace starts escape a limit-bound or singular initial solve.
A successful endpoint still requires the caller's full interpolated path checks.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

POS_TOL, ANG_TOL, JUMP = .0002, np.radians(.1), .3


def solve(kin, q0, Tg, lo, hi, h=None, budget_s=.06):
    """Return (local feasible q, '') or (None, reason), keeping exact 6D pose.

    ``lo`` and ``hi`` must already contain the caller's safety margins.
    No seed or fitted endpoint is returned without checking both hard bounds
    and final position/orientation residuals independently of optimizer status.
    """
    q0, lo, hi = [np.asarray(x, dtype=float) for x in (q0, lo, hi)]
    Tg = np.asarray(Tg, dtype=float)
    if (q0.ndim != 1 or q0.size < 6 or lo.shape != q0.shape or hi.shape != q0.shape
            or not all(np.all(np.isfinite(x)) for x in (q0, lo, hi, Tg))
            or np.any(lo >= hi) or Tg.shape != (4, 4)):
        raise ValueError("finite matching joint vectors and a 4x4 target are required")
    R = Tg[:3, :3]
    if (not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-6)
            or not np.allclose(Tg[3], [0, 0, 0, 1])):
        raise ValueError("target must be a rigid pose")
    if h is not None and not np.all(np.isfinite(np.asarray(h, dtype=float))):
        raise ValueError("hand joints must be finite")
    lower, upper = np.maximum(lo, q0 - JUMP), np.minimum(hi, q0 + JUMP)
    if np.any(lower >= upper):
        return None, "bounded IK: no local interval within joint limits"

    deadline = time.monotonic() + budget_s

    def error(q):
        if time.monotonic() >= deadline:
            raise TimeoutError
        T = kin.tip(q, h)
        return np.r_[Tg[:3, 3] - T[:3, 3], Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec()]

    scale = np.array([POS_TOL] * 3 + [ANG_TOL] * 3)
    seed = np.clip(q0, lower, upper)
    try:
        J = np.column_stack([(error(seed + np.eye(q0.size)[i] * 1e-6) - error(seed)) / 1e-6
                             for i in range(q0.size)])
    except TimeoutError:
        return None, "bounded IK: planning time budget exceeded"
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    null = Vt[-1]
    seeds = [seed] + [np.clip(seed + sign * .20 * null, lower, upper) for sign in (1, -1)]
    candidates, best_error = [], None
    for initial in seeds:
        try:
            fit = least_squares(lambda q: error(q) / scale, initial, bounds=(lower, upper),
                                ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=100)
            e = error(fit.x)
        except TimeoutError:
            break
        quality = (float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])))
        if best_error is None or quality[0] / POS_TOL + quality[1] / ANG_TOL < best_error[0] / POS_TOL + best_error[1] / ANG_TOL:
            best_error = quality
        if (quality[0] <= POS_TOL and quality[1] <= ANG_TOL
                and np.all(fit.x >= lower) and np.all(fit.x <= upper)):
            margin = np.min(np.minimum(fit.x - lo, hi - fit.x) / (hi - lo))
            candidates.append((float(margin), -float(np.linalg.norm(fit.x - q0)), fit.x.copy()))
    if candidates:
        return max(candidates, key=lambda x: x[:2])[2], ""
    if best_error is None:
        return None, "bounded IK: planning time budget exceeded"
    return None, (f"bounded IK: no local feasible pose (position {best_error[0]*1000:.2f} mm, "
                  f"rotation {np.degrees(best_error[1]):.2f} deg; jump <= {JUMP} rad)")

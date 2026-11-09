"""Opt-in stationary torque compensation, independent of ROS and motor I/O.

The caller supplies *full* motor PD + existing gravity + requested torque as
``base_torque``. Only ``result.addition`` goes into the feed-forward command.
Safety transitions remain the caller's responsibility: this module never moves
position targets, engages a motor, or relaxes an existing fault threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np

# Static-friction cap [N·m] on fs, fs_large and |static + integral| (2026-11-08 fit:
# j1~j4 0.41~0.78, wrist untested).
FS_MAX = 1.0
FS_JOINTS = 4


def _vector(value, n, name, *, nonnegative=False, positive=False):
    v = np.asarray(value, dtype=float)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError(f"{name} needs {n} finite values")
    if (nonnegative and np.any(v < 0)) or (positive and np.any(v <= 0)):
        raise ValueError(f"{name} must be {'> 0' if positive else '>= 0'}")
    return v


@dataclass(frozen=True)
class CompensationConfig:
    gravity_enabled: bool = False
    friction_enabled: bool = False
    integral_enabled: bool = False
    payload_parameters: tuple = (0.0, 0.0, 0.0, 0.0)
    fc: tuple = (0.0,) * 7
    fv: tuple = (0.0,) * 7
    fo: tuple = (0.0,) * 7
    vs: tuple = (0.01,) * 7
    ki: tuple = (0.0,) * 7
    # Static friction pre-push f·s(e); 0 = off. j5~j7 must stay 0.
    # fs = floor after a small move; fs_large (if any > 0) is used instead when the
    # joint moved more than move_split_deg since the last target/episode change.
    fs: tuple = (0.0,) * 7
    fs_large: tuple = (0.0,) * 7
    move_split_deg: float = 2.0
    # 0.15 deg ≈ 7 counts of the 0.00038 rad encoder step (stationary_fake), well above
    # quantization/standstill noise, so s(e) stays exactly 0 near the target (no chatter).
    fs_deadband_deg: float = 0.15
    fs_width_deg: float = 0.05
    integral_limit: tuple = (1.0,) * 4 + (0.3,) * 3
    integral_rate: tuple = (0.2,) * 4 + (0.05,) * 3
    deadband_stop_deg: float = 0.03
    deadband_resume_deg: float = 0.06
    blend_sec: float = 1.0
    # Standstill = j1~j4 measured position span over the last settle_window_sec stays
    # within settle_band_deg. Measured velocity is not used: DM MIT velocity is
    # quantized (j1/j2 step ≈ 0.022 rad/s), so |qd| < small never holds on hardware.
    settle_window_sec: float = 0.1
    settle_band_deg: float = 0.05

    def __post_init__(self):
        for key in ("gravity_enabled", "friction_enabled", "integral_enabled"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be bool")
        n = len(self.ki)
        if n <= 0:
            raise ValueError("empty joint list")
        for key in (
            "fc", "fv", "fo", "vs", "ki", "fs", "fs_large", "integral_limit", "integral_rate"
        ):
            v = _vector(
                getattr(self, key),
                n,
                key,
                nonnegative=key in ("fc", "fv", "ki", "fs", "fs_large", "integral_limit"),
                positive=key in ("vs", "integral_rate"),
            )
            object.__setattr__(self, key, tuple(v))
        object.__setattr__(
            self,
            "payload_parameters",
            tuple(_vector(self.payload_parameters, 4, "payload_parameters")),
        )
        for key in ("fs", "fs_large"):
            v = np.asarray(getattr(self, key))
            if np.any(v > FS_MAX) or np.any(v[FS_JOINTS:]):
                raise ValueError(
                    f"{key} must be <= {FS_MAX} N·m on j1~j{FS_JOINTS} and 0 elsewhere"
                )
        if self.static_enabled and np.any(np.asarray(self.integral_limit) > FS_MAX):
            raise ValueError(f"static friction needs integral_limit <= {FS_MAX} N·m")
        for key in (
            "deadband_stop_deg",
            "deadband_resume_deg",
            "blend_sec",
            "fs_deadband_deg",
            "fs_width_deg",
            "move_split_deg",
            "settle_window_sec",
            "settle_band_deg",
        ):
            if not np.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if (
            not 0 <= self.deadband_stop_deg < self.deadband_resume_deg
            or self.blend_sec <= 0
        ):
            raise ValueError("deadband requires 0 <= stop < resume; blend_sec > 0")
        if min(self.fs_deadband_deg, self.fs_width_deg, self.move_split_deg) <= 0:
            raise ValueError("fs_deadband_deg, fs_width_deg, move_split_deg must be > 0")
        if min(self.settle_window_sec, self.settle_band_deg) <= 0:
            raise ValueError("settle_window_sec and settle_band_deg must be > 0")

    @property
    def n(self):
        return len(self.ki)

    @property
    def enabled(self):
        return (
            self.gravity_enabled
            or self.friction_enabled
            or self.integral_enabled
            or self.static_enabled
        )

    @property
    def static_enabled(self):
        return any(self.fs) or any(self.fs_large)

    @classmethod
    def from_mapping(cls, mapping: Mapping | None, n=7):
        values = dict(mapping or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"unknown stationary compensation fields: {sorted(unknown)}"
            )
        if n != 7:
            for key in ("fc", "fv", "fo", "ki", "fs", "fs_large"):
                values.setdefault(key, (0.0,) * n)
            values.setdefault("vs", (0.01,) * n)
            values.setdefault("integral_limit", (0.3,) * n)
            values.setdefault("integral_rate", (0.05,) * n)
        result = cls(**values)
        if result.n != n:
            raise ValueError(f"configuration needs {n} joints")
        return result


def validate_no_duplicate_integrators(
    cfg, *, jog_integral=False, home_integral=False, droop_integral=False
):
    if cfg.integral_enabled and (jog_integral or home_integral or droop_integral):
        raise ValueError("stationary integral conflicts with jog/home/droop integral")
    # fs may run with the stationary integral (it is not an integrator, and the two
    # share the integral_limit cap), but not with another static-error compensator.
    if cfg.static_enabled and (jog_integral or home_integral or droop_integral):
        raise ValueError("static friction fs conflicts with jog/home/droop integral")
    if cfg.static_enabled and cfg.friction_enabled and (any(cfg.fc) or any(cfg.fo)):
        raise ValueError("static friction fs duplicates friction fc/fo compensation")


@dataclass(frozen=True)
class CompensationResult:
    addition: np.ndarray
    gravity: np.ndarray
    friction: np.ndarray
    integral: np.ndarray
    estimated_total: np.ndarray
    saturated: np.ndarray
    updating: np.ndarray
    static: np.ndarray


class StationaryCompensator:
    """Per-arm state; config is immutable for the lifetime of the experiment.

    New target/episode discards the learned integral with a monotone slew. The
    fade takes at least blend_sec, longer if required by integral_rate. Contact,
    stale data or stopped states freeze learning; the outer controller's blend
    and safety actions determine whether these held torques are sent.
    """

    def __init__(self, cfg: CompensationConfig):
        self.cfg = cfg
        self.integral = np.zeros(cfg.n)
        self._learning = np.ones(cfg.n, dtype=bool)
        self._fading = False
        self._fade_rate = np.zeros(cfg.n)
        self._fs_armed = False
        self._fs_gain = 0.0
        self.fs_armed = False  # diagnostics: static term armed after the last step
        self._fs_shape = np.zeros(cfg.n)
        self._moved = np.zeros(cfg.n)  # |motion| [rad] since target/episode change

    def step(
        self,
        error,
        velocity,
        base_torque,
        torque_limit,
        dt,
        *,
        settled,
        contact=False,
        stale=False,
        stopped=False,
        target_changed=False,
        episode_changed=False,
        gravity_correction=None,
        feedforward=None,
    ):
        cfg = self.cfg
        n = cfg.n
        e = _vector(error, n, "error")
        v = _vector(velocity, n, "velocity")
        base = _vector(base_torque, n, "base_torque")
        limit = _vector(torque_limit, n, "torque_limit", positive=True)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        zero = np.zeros(n)
        if not cfg.enabled:
            return CompensationResult(
                zero.copy(),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                base.copy(),
                np.zeros(n, bool),
                np.zeros(n, bool),
                zero.copy(),
            )
        gravity = zero.copy()
        if cfg.gravity_enabled:
            if gravity_correction is None:
                raise ValueError("gravity_enabled requires gravity_correction")
            gravity = _vector(gravity_correction, n, "gravity_correction").copy()
        friction = (
            (
                np.asarray(cfg.fc) * np.tanh(v / np.asarray(cfg.vs))
                + np.asarray(cfg.fv) * v
                + np.asarray(cfg.fo)
            )
            if cfg.friction_enabled
            else zero.copy()
        )
        # If the old controller is already saturated, additions may only oppose
        # that saturation. They are never forced to cancel a pre-existing fault.
        low = np.minimum(-limit - base, 0.0)
        high = np.maximum(limit - base, 0.0)
        if feedforward is not None:
            ff = _vector(feedforward, n, "feedforward")
            low = np.maximum(low, np.minimum(-limit - ff, 0.0))
            high = np.minimum(high, np.maximum(limit - ff, 0.0))
        requested_model = gravity + friction
        gravity = np.clip(gravity, low, high)
        friction = np.clip(friction, low - gravity, high - gravity)
        model = gravity + friction
        safe = not (contact or stale or stopped)
        updating = np.zeros(n, bool)
        if (target_changed or episode_changed) and not self._fading:
            self._fading = bool(np.any(self.integral))
            self._fade_rate = np.minimum(
                np.abs(self.integral) / cfg.blend_sec, cfg.integral_rate
            )
            self._learning[:] = True
        if target_changed or episode_changed:
            self._moved[:] = 0.0
        # Static term: armed once settled, kept through the breakaway motion it
        # causes, dropped on target/episode change or integral fade (slewed over
        # blend_sec with the last direction frozen) and cut at once on contact/stale/stop.
        if not safe or target_changed or episode_changed or self._fading:
            self._fs_armed = False
        elif settled:
            self._fs_armed = True
        self.fs_armed = self._fs_armed
        step = dt / cfg.blend_sec
        self._fs_gain = (
            min(self._fs_gain + step, 1.0)
            if self._fs_armed
            else max(self._fs_gain - step, 0.0)
        ) if safe else 0.0
        if not self._fs_armed:
            # Counted only until armed, so the breakaway motion fs causes is excluded.
            self._moved += np.abs(v) * dt
        if self._fs_armed:
            f = np.asarray(cfg.fs)
            if any(cfg.fs_large):
                large = self._moved > np.deg2rad(cfg.move_split_deg)
                f = np.where(large, np.asarray(cfg.fs_large), f)
            over = np.maximum(np.abs(e) - np.deg2rad(cfg.fs_deadband_deg), 0.0)
            self._fs_shape = (
                f
                * np.sign(e)
                * np.tanh(over / np.deg2rad(cfg.fs_width_deg))
            )
        cap = np.asarray(cfg.integral_limit)
        requested_static = self._fs_gain * self._fs_shape
        static = np.clip(
            requested_static,
            np.maximum(low - model, -cap),
            np.minimum(high - model, cap),
        )
        held = model + static
        # Integral bounds: torque headroom and |static + integral| <= integral_limit.
        ilow = np.maximum(low - held, -cap - static)
        ihigh = np.minimum(high - held, cap - static)
        if self._fading:
            self.integral = np.sign(self.integral) * np.maximum(
                np.abs(self.integral) - self._fade_rate * dt, 0.0
            )
            if np.max(np.abs(self.integral)) < 1e-12:
                self.integral[:] = 0.0
                self._fading = False
        elif (
            cfg.integral_enabled
            and safe
            and settled
            and not (target_changed or episode_changed)
        ):
            magnitude = np.abs(e)
            self._learning[magnitude <= np.deg2rad(cfg.deadband_stop_deg)] = False
            self._learning[magnitude >= np.deg2rad(cfg.deadband_resume_deg)] = True
            delta = np.clip(
                np.asarray(cfg.ki) * e * dt,
                -np.asarray(cfg.integral_rate) * dt,
                np.asarray(cfg.integral_rate) * dt,
            )
            delta = np.where(self._learning, delta, 0.0)
            before = self.integral.copy()
            candidate = np.clip(
                before + delta, -np.asarray(cfg.integral_limit), cfg.integral_limit
            )
            # Conditional integration: never accumulate further into saturation.
            total = base + held + candidate
            blocked = (((total > limit) | (candidate > ihigh)) & (delta > 0)) | (
                ((total < -limit) | (candidate < ilow)) & (delta < 0)
            )
            candidate = np.where(blocked, before, candidate)
            self.integral = candidate
            updating = self._learning & ~blocked & (delta != 0)
        integral = np.clip(self.integral, ilow, ihigh)
        addition = held + integral
        saturated = (
            (np.abs(requested_model - model) > 1e-12)
            | (np.abs(requested_static - static) > 1e-12)
            | (np.abs(integral - self.integral) > 1e-12)
        )
        return CompensationResult(
            addition,
            gravity,
            friction,
            integral,
            base + addition,
            saturated,
            updating,
            static,
        )


def payload_regressor(chain, q, gravity=(0.0, 0.0, -9.80665)):
    """N x 4 static regressor for [mass, mx, my, mz] in LAST ARM LINK frame.

    This matches robot_control.kinematics.with_payload; the tip transform is
    deliberately not used. Fixed hand posture is required when fitting this
    lumped correction. Other link errors remain model error, not hand mass.
    """
    frames = chain.frames(q)
    g = _vector(gravity, 3, "gravity")
    last = frames[-1]
    out = np.zeros((len(frames), 4))
    for i, (joint, frame) in enumerate(zip(chain.joints, frames)):
        axis = frame[:3, :3] @ joint.axis
        out[i, 0] = -g @ np.cross(axis, last[:3, 3] - frame[:3, 3])
        for j in range(3):
            out[i, j + 1] = -g @ np.cross(axis, last[:3, j])
    return out


@dataclass(frozen=True)
class PayloadFit:
    parameters: np.ndarray
    identifiable_basis: np.ndarray
    rank: int
    condition: float
    train_rmse: float
    validation_rmse: float
    validation_baseline_rmse: float
    accepted: bool


def fit_payload(
    train_regressor,
    train_residual,
    validation_regressor,
    validation_residual,
    *,
    rcond=1e-8,
    max_condition=1e6,
):
    """Fit ONLY training data via truncated SVD; score separate held-out poses.

    Acceptance requires improved held-out prediction and identifiable conditioning.
    A minimum-norm parameter vector represents only the identifiable subspace;
    individual mass/COM values must not be interpreted when rank < 4.
    Caller must use disjoint pose identities; no validation rows enter fitting.
    """
    a = np.asarray(train_regressor, float).reshape(-1, 4)
    b = np.asarray(validation_regressor, float).reshape(-1, 4)
    y = np.asarray(train_residual, float).reshape(-1)
    z = np.asarray(validation_residual, float).reshape(-1)
    if not (len(a) == len(y) and len(b) == len(z) and len(a) > 0 and len(b) > 0):
        raise ValueError(
            "nonempty training/held-out regressors and residuals must match"
        )
    if not all(np.isfinite(v).all() for v in (a, b, y, z)):
        raise ValueError("identification inputs must be finite")
    if not 0 < rcond < 1 or not np.isfinite(max_condition) or max_condition < 1:
        raise ValueError("invalid SVD conditioning limits")
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * rcond)) if singular[0] > 0 else 0
    basis = vt[:rank].T
    beta = basis @ ((u[:, :rank].T @ y) / singular[:rank]) if rank else np.zeros(4)
    condition = float(singular[0] / singular[rank - 1]) if rank else float("inf")
    train_rmse = float(np.sqrt(np.mean((a @ beta - y) ** 2)))
    val_rmse = float(np.sqrt(np.mean((b @ beta - z) ** 2)))
    baseline = float(np.sqrt(np.mean(z**2)))
    accepted = bool(rank and condition <= max_condition and val_rmse < baseline)
    return PayloadFit(
        beta, basis, rank, condition, train_rmse, val_rmse, baseline, accepted
    )


def fit_friction(velocity, residual, validation_velocity, validation_residual, *, vs):
    """Independent low-acceleration, matched-position residuals; no gravity inference.

    Caller splits pose identities BEFORE this fit and removes the selected gravity
    model. Bias Fo still includes direction-independent model error.
    """
    v, y, vv, yy = [
        np.asarray(x, float)
        for x in (velocity, residual, validation_velocity, validation_residual)
    ]
    n = len(vs)
    scale = _vector(vs, n, "vs", positive=True)
    if (
        v.ndim != 2
        or v.shape[1] != n
        or y.shape != v.shape
        or vv.ndim != 2
        or vv.shape[1] != n
        or yy.shape != vv.shape
    ):
        raise ValueError("friction fit needs training/validation [samples,joints]")
    if min(len(v), len(vv)) < 3 or not all(
        np.isfinite(x).all() for x in (v, y, vv, yy)
    ):
        raise ValueError("finite nonempty identification data required")
    if np.any(np.min(v, axis=0) >= 0) or np.any(np.max(v, axis=0) <= 0):
        raise ValueError("both directions required on every joint")
    coefficients, ranks, conditions, errors, baselines = [], [], [], [], []
    for j in range(n):
        a = np.column_stack([np.tanh(v[:, j] / scale[j]), v[:, j], np.ones(len(v))])
        b = np.column_stack([np.tanh(vv[:, j] / scale[j]), vv[:, j], np.ones(len(vv))])
        beta, _, rank, singular = np.linalg.lstsq(a, y[:, j], rcond=1e-8)
        coefficients.append(beta)
        ranks.append(int(rank))
        conditions.append(float(singular[0] / singular[-1]) if rank == 3 else None)
        errors.append(float(np.sqrt(np.mean((b @ beta - yy[:, j]) ** 2))))
        baselines.append(float(np.sqrt(np.mean(yy[:, j] ** 2))))
    beta = np.asarray(coefficients)
    return {
        "fc": beta[:, 0].tolist(),
        "fv": beta[:, 1].tolist(),
        "fo": beta[:, 2].tolist(),
        "vs": scale.tolist(),
        "rank": ranks,
        "condition": conditions,
        "validation_rmse": errors,
        "validation_baseline_rmse": baselines,
        "accepted": bool(
            all(r == 3 for r in ranks)
            and all(c is not None and c <= 1e6 for c in conditions)
            and np.all(beta[:, :2] >= 0)
            and np.all(np.asarray(errors) < baselines)
        ),
    }

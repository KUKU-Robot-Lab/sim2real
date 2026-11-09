"""ROS-free precision jog primitives. Limits are joint-space rad/s and rad/s².

A path has one shared clock for all joints and starts/ends at rest. Bias is
bounded position compensation, not torque or velocity feedforward. Callers
must block it on contact, stale state, saturation and unsafe paths, and avoid
integrating transient tracking lag as steady droop.
"""
from __future__ import annotations

import numpy as np


def _vector(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim != 1 or not out.size or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return out.copy()


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


class QuinticPath:
    """Synchronized quintic interpolation with bounded velocity/acceleration.

    ``sample(elapsed)`` returns independent ``(position, velocity)`` arrays.
    Time outside the path is clamped to the endpoint with velocity zero.
    Retargeting is a new path and requires a stationary start.
    """

    def __init__(self, start, goal, max_vel: float = .3, max_acc: float = .6, min_duration: float = 0):
        self.start, self.goal = _vector(start, "start"), _vector(goal, "goal")
        if self.start.shape != self.goal.shape:
            raise ValueError("start and goal shapes must match")
        max_vel = _positive(max_vel, "max_vel")
        max_acc = _positive(max_acc, "max_acc")
        min_duration = _positive(min_duration, "min_duration", zero=True)
        self.delta = self.goal - self.start
        distance = float(np.max(np.abs(self.delta)))
        self.duration = max(min_duration, 1.875 * distance / max_vel,
                            np.sqrt((10 / np.sqrt(3)) * distance / max_acc))

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        elapsed = float(elapsed)
        if not np.isfinite(elapsed):
            raise ValueError("elapsed must be finite")
        if self.duration == 0 or elapsed >= self.duration:
            return self.goal.copy(), np.zeros_like(self.goal)
        if elapsed <= 0:
            return self.start.copy(), np.zeros_like(self.start)
        u = elapsed / self.duration
        position = u ** 3 * (10 + u * (-15 + 6 * u))
        velocity = 30 * u ** 2 * (1 - u) ** 2 / self.duration
        return self.start + self.delta * position, self.delta * velocity


def bias_step(c, error, dt: float, blocked: bool = False, gain: float = 2.,
              max_rate: float = .03, max_bias: float = .12) -> np.ndarray:
    """Integrate signed desired-minus-measured error with no hidden state.

    Output clipping provides anti-windup: outward error at a cap cannot build
    additional internal state, and inward error immediately leaves the cap.
    ``blocked`` freezes the existing correction, without clearing it.
    """
    c, error = _vector(c, "c"), _vector(error, "error")
    if c.shape != error.shape:
        raise ValueError("c and error shapes must match")
    dt = _positive(dt, "dt", zero=True)
    gain = _positive(gain, "gain", zero=True)
    max_rate, max_bias = _positive(max_rate, "max_rate"), _positive(max_bias, "max_bias")
    if np.any(np.abs(c) > max_bias):
        raise ValueError("initial bias exceeds max_bias")
    if blocked or dt == 0:
        return c
    return np.clip(c + np.clip(gain * error, -max_rate, max_rate) * dt, -max_bias, max_bias)


class RetargetablePath:
    """Quintic joint path preserving the initial position/velocity/acceleration.

    Ends at ``goal`` at rest. Reversal/braking may overshoot the endpoint range;
    the caller must validate the entire path for collision and joint limits.
    Duration search is bounded at 20 s and checks polynomial extrema, including
    extrema between sampling ticks. Infeasible initial states are rejected.
    """

    def __init__(self, start, goal, velocity=None, acceleration=None,
                 max_vel: float = .3, max_acc: float = .6, min_duration: float = 0):
        self.start, self.goal = _vector(start, "start"), _vector(goal, "goal")
        self.velocity = np.zeros_like(self.start) if velocity is None else _vector(velocity, "velocity")
        self.acceleration = np.zeros_like(self.start) if acceleration is None else _vector(acceleration, "acceleration")
        if any(x.shape != self.start.shape for x in (self.goal, self.velocity, self.acceleration)):
            raise ValueError("position, velocity and acceleration shapes must match")
        max_vel, max_acc = _positive(max_vel, "max_vel"), _positive(max_acc, "max_acc")
        min_duration = _positive(min_duration, "min_duration", zero=True)
        if np.any(np.abs(self.velocity) > max_vel) or np.any(np.abs(self.acceleration) > max_acc):
            raise ValueError("initial velocity or acceleration exceeds limits")
        self.delta = self.goal - self.start
        if not (self.delta.any() or self.velocity.any() or self.acceleration.any()):
            self.duration = min_duration
            self.coefficients = np.stack([self.start] + [np.zeros_like(self.start)] * 5)
            return
        distance = float(np.max(np.abs(self.delta)))
        duration = max(min_duration, .001, 1.875 * distance / max_vel,
                       np.sqrt(10 / np.sqrt(3) * distance / max_acc),
                       2 * float(np.max(np.abs(self.velocity))) / max_acc)
        while duration <= 20:
            vt, at = self.velocity * duration, self.acceleration * duration ** 2
            self.coefficients = np.stack((self.start, vt, at / 2,
                                          10 * self.delta - 6 * vt - 1.5 * at,
                                          -15 * self.delta + 8 * vt + 1.5 * at,
                                          6 * self.delta - 3 * vt - .5 * at))
            self.duration = duration
            if self._within_limits(max_vel, max_acc):
                return
            duration *= 1.1
        raise ValueError("no feasible retarget trajectory within 20 s")

    def _within_limits(self, max_vel: float, max_acc: float) -> bool:
        for coefficients in self.coefficients.T:
            velocity = np.polynomial.polynomial.polyder(coefficients) / self.duration
            acceleration = np.polynomial.polynomial.polyder(velocity) / self.duration
            jerk = np.polynomial.polynomial.polyder(acceleration) / self.duration
            for polynomial, derivative, limit in ((velocity, acceleration, max_vel), (acceleration, jerk, max_acc)):
                roots = np.polynomial.polynomial.polyroots(derivative)
                times = [0., 1.] + [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1]
                if np.max(np.abs(np.polynomial.polynomial.polyval(times, polynomial))) > limit * (1 + 1e-12):
                    return False
        return True

    def position_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact joint-wise extrema over the path, including braking overshoot."""
        lower, upper = [], []
        for coefficients in self.coefficients.T:
            roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
            times = [0., 1.] + [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1]
            values = np.polynomial.polynomial.polyval(times, coefficients)
            lower.append(float(np.min(values)))
            upper.append(float(np.max(values)))
        return np.array(lower), np.array(upper)

    def sample_state(self, elapsed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        elapsed = float(elapsed)
        if not np.isfinite(elapsed):
            raise ValueError("elapsed must be finite")
        if elapsed < 0 or (elapsed == 0 and self.duration > 0):
            return self.start.copy(), self.velocity.copy(), self.acceleration.copy()
        if self.duration == 0 or elapsed >= self.duration:
            return self.goal.copy(), np.zeros_like(self.goal), np.zeros_like(self.goal)
        u = elapsed / self.duration
        q = np.polynomial.polynomial.polyval(u, self.coefficients)
        v = np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(self.coefficients)) / self.duration
        a = np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(self.coefficients, m=2)) / self.duration ** 2
        return q, v, a

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        q, v, _ = self.sample_state(elapsed)
        return q, v

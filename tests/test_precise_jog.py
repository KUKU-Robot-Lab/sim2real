"""Pure precision-jog trajectory/bias tests; no ROS or physical robot."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/policy_control/tools"))
from precise_jog import QuinticPath, bias_step  # noqa: E402


def test_quintic_is_synchronized_and_respects_velocity_and_acceleration():
    start = np.array([0., .1, -.5])
    goal = np.array([.4, -.2, -.5])
    path = QuinticPath(start, goal)
    times = np.linspace(0, path.duration, 10001)
    samples = [path.sample(t) for t in times]
    positions, velocities = map(np.array, zip(*samples))
    assert np.max(np.abs(velocities)) <= .3 + 1e-12
    assert np.max(np.abs(np.diff(velocities, axis=0) / np.diff(times)[:, None])) <= .6 + 1e-8
    assert np.allclose((positions[:, 0] - start[0]) / .4, (positions[:, 1] - start[1]) / -.3)
    assert np.all(positions[:, 2] == -.5)
    for t, expected in [(-1, start), (0, start), (path.duration, goal), (path.duration + 1, goal)]:
        q, qd = path.sample(t)
        assert np.array_equal(q, expected)
        assert np.array_equal(qd, np.zeros(3))


def test_quintic_honors_min_duration_and_jitter_without_state_or_aliasing():
    start = np.zeros(2)
    goal = np.array([.01, -.02])
    path = QuinticPath(start, goal, min_duration=2)
    start[:] = 99
    goal[:] = 99
    assert path.duration == 2
    rng = np.random.default_rng(17)
    for t in rng.uniform(0, 2, 100):
        q, v = path.sample(t)
        q2, v2 = path.sample(t)
        assert np.array_equal(q, q2) and np.array_equal(v, v2)
    p = QuinticPath([1, 2], [1, 2])
    assert p.duration == 0
    assert np.array_equal(p.sample(0)[0], [1, 2])


@pytest.mark.parametrize('args,kw', [(([0], [0, 1]), {}), (([np.nan], [0]), {}),
                                   (([], []), {}), (([[0]], [[1]]), {}),
                                   (([0], [1]), {'max_vel': 0}), (([0], [1]), {'max_acc': -1}),
                                   (([0], [1]), {'min_duration': -1}), (([0], [1]), {'max_vel': np.inf})])
def test_invalid_path_is_rejected(args, kw):
    with pytest.raises(ValueError):
        QuinticPath(*args, **kw)


def test_bias_caps_rate_magnitude_and_freezes_without_hidden_accumulation():
    c = np.array([.1199, -.1199, 0.])
    b = bias_step(c, [1, -1, .001], .01)
    assert np.max(np.abs(b - c)) <= .03 * .01 + 1e-15
    assert np.max(np.abs(b)) <= .12
    assert np.sign(b[2]) == 1
    for _ in range(100):
        b = bias_step(b, [1, -1, 1], .01, blocked=True)
    assert b[0] == .12 and b[1] == -.12
    assert bias_step(b, [-1, 1, 0], .01)[0] < b[0]  # no wound-up state to unwind
    assert np.array_equal(bias_step(c, [1, 1, 1], 0), c)
    assert np.array_equal(c, [.1199, -.1199, 0.])


@pytest.mark.parametrize('c,e,dt,kw', [([0], [0, 1], .01, {}), ([np.nan], [0], .01, {}),
                                     ([0], [np.inf], .01, {}), ([0], [0], -1, {}),
                                     ([0], [0], np.nan, {}), ([0], [0], .01, {'gain': -1}),
                                     ([0], [0], .01, {'max_rate': 0})])
def test_invalid_bias_input_is_rejected(c, e, dt, kw):
    with pytest.raises(ValueError):
        bias_step(c, e, dt, **kw)


def test_synthetic_sag_plant_settles_faster_with_bounded_precision_bias():
    # Synthetic 80 ms plant, 40 mrad droop; this is not a real-robot accuracy claim.
    def simulate(gain):
        dt = 1 / 120
        bias, meas = np.zeros(1), np.array([-.04])
        errors = []
        for _ in range(120 * 3):
            bias = bias_step(bias, -meas, dt, gain=gain)
            meas += (bias - .04 - meas) * dt / .08
            errors.append(abs(meas[0]))
        return np.array(errors)
    slow, precise = simulate(.5), simulate(2.)
    assert precise[-1] < .001
    assert precise[-1] < slow[-1] / 5


def test_retarget_continuity_preserves_position_velocity_acceleration():
    from precise_jog import RetargetablePath
    first = RetargetablePath([0., .1], [.2, -.1])
    q, v, a = first.sample_state(first.duration * .37)
    second = RetargetablePath(q, [.3, -.2], velocity=v, acceleration=a)
    state = second.sample_state(0)
    for initial, actual in zip((q, v, a), state):
        assert np.array_equal(initial, actual)
    assert np.linalg.norm(second.sample(.001)[0] - q) > 0
    assert np.allclose(second.sample_state(second.duration)[0], [.3, -.2])
    assert np.array_equal(second.sample_state(second.duration)[1], [0., 0.])
    assert np.array_equal(second.sample_state(second.duration)[2], [0., 0.])


@pytest.mark.parametrize('goal,velocity,acceleration', [([.2, -.1], [.08, -.02], [.01, 0]),
                                                       ([-.2, .1], [.08, -.02], [.01, 0]),
                                                       ([0., 0.], [.1, -.1], [0, 0])])
def test_retarget_reverse_and_braking_respect_limits(goal, velocity, acceleration):
    from precise_jog import RetargetablePath
    path = RetargetablePath([0., 0.], goal, velocity, acceleration)
    states = [path.sample_state(t) for t in np.linspace(0, path.duration, 3001)]
    for q, v, a in states:
        assert np.all(np.isfinite(q))
        assert np.max(np.abs(v)) <= .3 + 1e-10
        assert np.max(np.abs(a)) <= .6 + 1e-10
    q0, v0, a0 = path.sample_state(0)
    assert np.array_equal(q0, [0, 0]) and np.array_equal(v0, velocity) and np.array_equal(a0, acceleration)


@pytest.mark.parametrize('kw', [{'velocity': [.31]}, {'acceleration': [.61]}, {'velocity': [np.nan]},
                               {'acceleration': [0, 0]}, {'max_acc': 0}])
def test_retarget_invalid_initial_conditions_rejected(kw):
    from precise_jog import RetargetablePath
    with pytest.raises(ValueError):
        RetargetablePath([0], [.1], **kw)


def test_retarget_rest_zero_path_and_elapsed_validation():
    from precise_jog import RetargetablePath
    path = RetargetablePath([.1], [.1])
    assert path.duration == 0
    assert np.array_equal(path.sample(-1)[0], [.1])
    with pytest.raises(ValueError):
        path.sample(np.nan)


def test_retarget_position_bounds_include_braking_overshoot():
    from precise_jog import RetargetablePath
    path = RetargetablePath([0., 1.], [0., 1.], [.1, -.1])
    lo, hi = path.position_bounds()
    assert hi[0] > 0 and lo[1] < 1
    positions = np.array([path.sample(t)[0] for t in np.linspace(0, path.duration, 2001)])
    assert np.all(positions >= lo - 1e-12) and np.all(positions <= hi + 1e-12)
    assert np.allclose(lo, positions.min(0), atol=1e-7)
    assert np.allclose(hi, positions.max(0), atol=1e-7)

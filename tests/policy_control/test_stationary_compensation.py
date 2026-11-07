"""Pure stationary compensation tests: no ROS or hardware side effects."""

import numpy as np
import pytest

from policy_control import stationary_compensation as sc


def cfg(**changes):
    return sc.CompensationConfig.from_mapping(changes)


def tick(c, **changes):
    args = dict(
        error=np.ones(7) * 0.1,
        velocity=np.zeros(7),
        base_torque=np.zeros(7),
        torque_limit=np.ones(7) * 5,
        dt=0.1,
        settled=True,
    )
    args.update(changes)
    return c.step(**args)


def test_disabled_is_exact_zero():
    c = sc.StationaryCompensator(cfg())
    r = tick(c, base_torque=np.ones(7) * 10, gravity_correction=np.ones(7))
    assert np.array_equal(r.addition, np.zeros(7))
    assert np.array_equal(r.estimated_total, np.ones(7) * 10)


def test_integral_rate_limit_deadband_and_resume_hysteresis():
    c = sc.StationaryCompensator(cfg(integral_enabled=True, ki=[10] * 7))
    r = tick(c)
    assert r.integral == pytest.approx([0.02] * 4 + [0.005] * 3)
    held = tick(c, error=np.ones(7) * np.deg2rad(0.02)).integral
    assert tick(c, error=np.ones(7) * np.deg2rad(0.045)).integral == pytest.approx(held)
    assert np.all(tick(c, error=np.ones(7) * np.deg2rad(0.07)).integral > held)


@pytest.mark.parametrize("flag", ["contact", "stale", "stopped"])
def test_safety_events_freeze_learning(flag):
    c = sc.StationaryCompensator(cfg(integral_enabled=True, ki=[10] * 7))
    previous = tick(c).integral
    r = tick(c, **{flag: True})
    assert r.integral == pytest.approx(previous)
    assert not np.any(r.updating)


def test_moving_target_does_not_integrate():
    c = sc.StationaryCompensator(cfg(integral_enabled=True, ki=[10] * 7))
    assert not np.any(tick(c, settled=False).integral)


def test_full_pd_torque_headroom_and_antiwindup():
    c = sc.StationaryCompensator(cfg(integral_enabled=True, ki=[10] * 7))
    for _ in range(100):
        r = tick(c, base_torque=np.ones(7) * 4.99)
    assert np.all(r.estimated_total <= 5)
    assert np.all(r.integral <= 0.01 + 1e-12)
    # Opposite error can unwind even with a saturated base motor PD torque.
    for _ in range(
        3
    ):  # Wrist slew 0.005 Nm/tick must first unwind the previous +0.01 Nm.
        r = tick(c, error=-np.ones(7), base_torque=np.ones(7) * 6)
    assert np.all(r.integral < 0)
    assert np.all(r.addition <= 0)


def test_new_target_and_episode_remove_integral_over_blend_without_relearning():
    c = sc.StationaryCompensator(cfg(integral_enabled=True, ki=[10] * 7, blend_sec=1.0))
    for _ in range(5):
        before = tick(c).integral
    first = tick(c, target_changed=True).integral
    assert np.all(first >= before * 0.9 - 1e-12)
    for _ in range(9):
        last = tick(c).integral
    assert last == pytest.approx(np.zeros(7), abs=1e-12)
    tick(c)
    assert not np.allclose(tick(c, episode_changed=True).integral, before)


def test_friction_and_gravity_sum_are_limited_using_motor_torque():
    c = sc.StationaryCompensator(
        cfg(
            friction_enabled=True,
            gravity_enabled=True,
            fc=[1] * 7,
            fv=[2] * 7,
            fo=[0.2] * 7,
        )
    )
    r = tick(
        c,
        velocity=np.ones(7),
        base_torque=np.ones(7) * 4,
        gravity_correction=np.ones(7) * 0.3,
    )
    assert np.all(r.estimated_total <= 5)
    assert r.addition == pytest.approx(r.gravity + r.friction + r.integral)
    zero = tick(
        c, velocity=np.zeros(7), base_torque=np.zeros(7), gravity_correction=np.zeros(7)
    )
    assert zero.friction == pytest.approx(np.ones(7) * 0.2)


@pytest.mark.parametrize(
    "mapping",
    [
        dict(vs=[0] * 7),
        dict(ki=[-1] * 7),
        dict(integral_enabled="false"),
        dict(unknown=1),
        dict(fc=[float("nan")] * 7),
        dict(deadband_stop_deg=0.07),
        dict(blend_sec=0),
        dict(ki=[0] * 6),
    ],
)
def test_bad_configuration_is_rejected(mapping):
    with pytest.raises(ValueError):
        cfg(**mapping)


def test_duplicate_integrators_rejected_only_when_new_integral_enabled():
    sc.validate_no_duplicate_integrators(cfg(), home_integral=True)
    for flag in ("home_integral", "jog_integral", "droop_integral"):
        with pytest.raises(ValueError):
            sc.validate_no_duplicate_integrators(
                cfg(integral_enabled=True), **{flag: True}
            )


def test_payload_fit_uses_identifiable_combinations_and_heldout_predictions():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(50, 4))
    a[:, 3] = a[:, 0] * 2
    beta = np.array([0.4, 0.1, -0.2, 0.8])
    b = rng.normal(size=(20, 4))
    b[:, 3] = b[:, 0] * 2
    fit = sc.fit_payload(a, a @ beta, b, b @ beta)
    assert fit.rank == 3
    assert np.isfinite(fit.condition)
    assert fit.accepted
    assert fit.validation_rmse < 1e-12
    assert a @ fit.parameters == pytest.approx(a @ beta)
    wrong = sc.fit_payload(a, a @ beta, b, -b @ beta)
    assert not wrong.accepted


def test_payload_regressor_matches_existing_gravity_model():
    from robot_control.kinematics import Chain, Revolute, Link, with_payload

    chain = Chain(
        [Revolute("j", np.array([0.0, 1.0, 0.0]), np.zeros(3), np.eye(3), "l")],
        [Link("l", 1.0, np.array([0.1, 0.0, 0.0]))],
        np.eye(4),
    )
    q = np.array([0.3])
    mass = 0.7
    com = np.array([0.2, 0.1, 0.3])
    expected = with_payload(chain, mass, com).gravity_torque(q) - chain.gravity_torque(
        q
    )
    assert sc.payload_regressor(chain, q) @ np.r_[mass, mass * com] == pytest.approx(
        expected
    )


def test_constant_load_quantized_fake_equilibrium_improves_without_target_motion():
    c = sc.StationaryCompensator(cfg(integral_enabled=True, ki=[14] * 7))
    load = np.ones(7) * 0.2
    kp = 70.0
    integral = np.zeros(7)
    initial = load / kp
    for _ in range(1500):
        error = np.round((load - integral) / kp, 5)
        integral = tick(c, error=error, base_torque=kp * error, dt=0.01).integral
    assert np.max(np.abs((load - integral) / kp)) < np.max(initial) / 4
    # A rounded sensor can enter the deadband half an LSB before the true angle.
    assert np.max(np.abs((load - integral) / kp)) <= np.deg2rad(0.03) + 0.000005


def test_friction_fit_requires_both_directions_and_improves_heldout():
    v = np.array([-0.05, -0.025, -0.01, 0.01, 0.025, 0.05])[:, None] * np.ones((1, 7))
    torque = 0.2 * np.tanh(v / 0.01) + 0.4 * v + 0.03
    fit = sc.fit_friction(
        v,
        torque,
        v * 0.8,
        0.2 * np.tanh(v * 0.8 / 0.01) + 0.4 * v * 0.8 + 0.03,
        vs=[0.01] * 7,
    )
    assert fit["accepted"]
    assert np.asarray(fit["fc"]) == pytest.approx([0.2] * 7)
    with pytest.raises(ValueError, match="directions"):
        sc.fit_friction(abs(v), torque, v, torque, vs=[0.01] * 7)

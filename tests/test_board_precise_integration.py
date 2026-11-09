import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/policy_control/tools"))
import board_jog as B  # noqa: E402


def setup(precise=False):
    home = np.array(json.loads(B.CONTRACT.read_text())["sides"]["right"]["home_arm"])
    kin = B.Kin(B.URDF_DEFAULT, "right", B.POINT)
    lo, hi = B.arm_limits("right")
    io = B.FakeIO("right", kin, home, -1.0)
    ctl = B.Ctl(io, kin, lo, hi, home, np.array(B.POINT), False, True, None, None, precise=precise)
    io.jog = ctl.jog
    return ctl, kin, home


def test_reachable_grip_is_not_faked_zero():
    grip = B.reachable_grip(np.array(B.POINT), "right")
    assert 0.06 < grip[2] < 0.07
    assert np.allclose(B.reachable_grip(grip, "right"), grip)


def test_fixed_bias_preserves_motor_goal_despite_feedback_error():
    ctl, _, home = setup(True)
    ctl.continuous_hold = False
    ctl.jog.c = np.array([.01, -.01, .02, -.02, .01, -.03, .02])
    before = ctl.jog.pub().copy()
    for mode in ("idle", "move"):
        ctl.jog.mode = mode
        for i in range(240):
            measured = home + .02 * np.sin(i * .1)
            ctl._hold_step(10. + i * ctl.dt, ctl.dt, measured, False, False)
            assert np.array_equal(ctl.jog.pub(), before)
    assert ctl.hold_state == "보정 고정"


def test_fixed_bias_move_keeps_endpoint_after_trajectory_finishes():
    ctl, _, home = setup(True)
    ctl.continuous_hold = False
    ctl.jog.c = np.full(7, .01)
    goal = home.copy()
    goal[1] += .04
    assert ctl.jog.go(goal, False, False) is None
    for i in range(600):
        ctl.jog.tick(i * ctl.dt, ctl.dt, home, False, False)
        ctl._hold_step(10. + i * ctl.dt, ctl.dt, home, False, False)
    assert ctl.jog.mode == "idle"
    assert np.array_equal(ctl.jog.cmd, goal)
    assert np.allclose(ctl.jog.pub(), goal + .01)


def test_actual_hand_error_blocks_convergence():
    ctl, kin, home = setup()
    actual = np.array(B.POINT)
    actual[2] = 0.06
    ctl.r = {"meas": home, "hmeas": actual}
    ok, text = ctl._converged()
    assert not ok, text
    assert np.linalg.norm(kin.tip(home, actual)[:3, 3] - kin.tip(home)[:3, 3]) > .006


def test_precise_trajectory_and_active_bias_keep_target_separate():
    ctl, kin, home = setup(True)
    assert ctl.dt == 1 / 120
    assert ctl.jog.precise
    goal = home.copy()
    goal[1] += .05
    assert ctl.jog.go(goal, False, False) is None
    positions = [ctl.jog.cmd.copy()]
    for i in range(240):
        ctl.jog.tick(i * ctl.dt, ctl.dt, ctl.jog.cmd, False, False)
        positions.append(ctl.jog.cmd.copy())
    velocity = np.diff(positions, axis=0) / ctl.dt
    assert np.max(np.abs(velocity)) <= .30 + 1e-6
    assert np.max(np.abs(np.diff(velocity, axis=0) / ctl.dt)) <= .60 + 1e-3
    assert np.allclose(ctl.jog.cmd, goal)
    ctl.jog.mode = "move"
    ctl.r = {"hmeas": ctl.jog.hand.copy()}
    nominal = ctl.jog.cmd.copy()
    ctl._hold_step(10., ctl.dt, nominal - .01, False, False)
    assert np.any(ctl.jog.c > 0)
    assert np.array_equal(ctl.jog.cmd, nominal)
    c = ctl.jog.c.copy()
    ctl._hold_step(11., ctl.dt, nominal - .01, True, False)
    assert np.array_equal(ctl.jog.c, c)


def test_real_hand_feedback_corrects_tip_without_moving_nominal_goal():
    ctl, kin, home = setup(True)
    actual = ctl.jog.hand.copy()
    actual[2] = .06
    ctl.r = {"hmeas": actual}
    initial = np.linalg.norm(kin.tip(home, actual)[:3, 3] - kin.tip(home)[:3, 3])
    for i in range(600):
        measured = ctl.jog.cmd + ctl.jog.c
        ctl._hold_step(i * ctl.dt, ctl.dt, measured, False, False)
    final = np.linalg.norm(kin.tip(ctl.jog.cmd + ctl.jog.c, actual)[:3, 3] - kin.tip(home)[:3, 3])
    assert final < .001 and final < initial / 10
    assert np.array_equal(ctl.jog.cmd, home)
    c = ctl.jog.c.copy()
    ctl._hold_step(10., ctl.dt, home, False, True)
    assert np.array_equal(ctl.jog.c, c)


def test_precise_auto_approach_never_doubles_requested_cap():
    ctl, _, home = setup(True)
    goal = home.copy()
    goal[1] += .05
    assert ctl.jog.go(goal, False, False, vmax=B.JOG_VEL, precise_fast=False) is None
    samples = [ctl.jog.path.sample(t)[1] for t in np.linspace(0, ctl.jog.path.duration, 101)]
    assert np.max(np.abs(samples)) <= B.JOG_VEL + 1e-9


def test_raw_keeps_actual_hand_fk_and_loop_timing(tmp_path):
    ctl, kin, home = setup(True)
    actual = ctl.jog.hand.copy()
    actual[2] = .06
    state = {"meas": home, "hmeas": actual, "jf": np.zeros(6), "tip": np.zeros(5)}
    ctl.actual_dt = .009
    ctl._row(1., state)
    out = tmp_path / "raw.npz"
    B.save_raw(out, ctl.rows, "right")
    with np.load(out) as raw:
        assert raw["loop_dt"][0] == .009
        assert np.allclose(raw["T_tip_measured"][0], kin.tip(home, actual))
        assert np.allclose(raw["T_tip_reference"][0], kin.tip(home, ctl.jog.hand))
        assert "hold_state" in raw


def test_precise_retarget_keeps_velocity_and_acceleration():
    ctl, _, home = setup(True)
    goal = home.copy()
    goal[1] += .04
    assert ctl.jog.go(goal, False, False) is None
    for i in range(20):
        ctl.jog.tick(i * ctl.dt, ctl.dt, ctl.jog.cmd, False, False)
    before = ctl.jog.path.sample_state(ctl.jog.path_elapsed)
    goal[1] += .01
    assert ctl.jog.go(goal, False, False, retarget=True) is None
    after = ctl.jog.path.sample_state(0.)
    for a, b in zip(before, after):
        assert np.allclose(a, b, atol=1e-10)


def test_precise_retarget_checks_overshoot_against_joint_limits():
    ctl, _, home = setup(True)
    goal = home.copy()
    goal[1] += .04
    assert ctl.jog.go(goal, False, False) is None
    # Path is in motion; keep safety checks active on retargets.
    outside = home.copy()
    outside[1] = ctl.jog.hi[1] + .001
    assert ctl.jog.go(outside, False, False, retarget=True) is not None


def test_precise_contact_stop_has_no_position_target_jump():
    ctl, _, home = setup(True)
    ctl.jog.c = np.full(7, .02)
    before = ctl.jog.pub().copy()
    measured = home.copy()
    measured[1] += .05
    ctl.jog.hold(measured, 1., contact=True)
    assert np.array_equal(ctl.jog.pub(), before)
    assert ctl.jog.mode == "brake"
    positions = [before]
    for i in range(240):
        ctl.jog.tick(1 + i * ctl.dt, ctl.dt, measured, True, False)
        positions.append(ctl.jog.pub())
    assert np.max(np.abs(np.diff(positions, axis=0))) <= .3 * ctl.dt + 1e-6


def test_precise_emergency_unload_keeps_existing_release_semantics():
    ctl, _, home = setup(True)
    measured = home.copy()
    measured[1] -= .04
    ctl.jog.c[1] = .02
    ctl.jog.hold(measured, emergency=True)
    assert np.allclose(ctl.jog.cmd, measured)
    assert ctl.jog.mode == "idle"
    assert np.allclose(ctl.jog.pub(), measured + ctl.jog.c)


def test_precise_contact_brake_stops_on_force_rise():
    ctl, _, home = setup(True)
    measured = home.copy()
    measured[1] += .05
    ctl.jog.hold(measured, contact=True)
    assert ctl.jog.mode == "brake"
    ctl.jog.tick(0., ctl.dt, measured, True, False, (1., 0.))
    before = ctl.jog.cmd.copy()
    text = ctl.jog.tick(ctl.dt, ctl.dt, measured, True, False, (1. + B.FORCE_G + 1., 0.))
    assert "힘 상승" in text
    assert np.array_equal(before, ctl.jog.cmd)
    assert ctl.jog.mode == "idle"


def test_precise_ik_reserves_room_for_published_bias(monkeypatch):
    ctl, kin, home = setup(True)
    ctl.jog.c = np.full(7, .02)
    seen = {}

    def solver(kin, q0, target, lo, hi, h, bounded=False):
        seen.update(lo=lo.copy(), hi=hi.copy(), bounded=bounded)
        return q0.copy(), ""

    monkeypatch.setattr(B, "ik", solver)
    ctl.jog.plan(kin.tip(home), False, False)
    assert seen["bounded"]
    assert np.allclose(seen["lo"], np.maximum(ctl.jog.lo, ctl.jog.lo - ctl.jog.c))
    assert np.allclose(seen["hi"], np.minimum(ctl.jog.hi, ctl.jog.hi - ctl.jog.c))

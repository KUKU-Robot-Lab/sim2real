"""Static friction pre-push fs·s(e): pure tests plus a stiction toy plant. No ROS."""

from pathlib import Path

import numpy as np
import pytest

from policy_control import stationary_compensation as sc
from policy_control.pd_law import load_pd_config
from policy_control.stationary_fake import simulate, simulate_static_friction

FS = [0.5] * 4 + [0.0] * 3
CONFIG = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "config"


def comp(**changes):
    return sc.StationaryCompensator(sc.CompensationConfig.from_mapping({"fs": FS, **changes}))


def tick(c, error_deg=1.0, **changes):
    args = dict(
        error=np.ones(7) * np.deg2rad(error_deg),
        velocity=np.zeros(7),
        base_torque=np.zeros(7),
        torque_limit=np.ones(7) * 5,
        dt=0.1,
        settled=True,
    )
    args.update(changes)
    return c.step(**args)


def engaged(**changes):
    c = comp(**changes)
    for _ in range(10):  # blend_sec 1.0 at dt 0.1
        tick(c)
    return c


def test_deadband_is_exact_zero_and_continuous_above():
    c = engaged()
    assert np.array_equal(tick(c, error_deg=0.149).static, np.zeros(7))
    just_above = tick(c, error_deg=0.151).static
    assert 0 < just_above[0] < 0.5 * 0.05  # tanh(0.001/0.05) -> no step at the edge
    assert np.all(just_above[4:] == 0)


def test_sign_follows_error_and_saturates_at_fs():
    c = engaged()
    assert tick(c, error_deg=2.0).static[:4] == pytest.approx([0.5] * 4, abs=1e-6)
    assert tick(c, error_deg=-2.0).static[:4] == pytest.approx([-0.5] * 4, abs=1e-6)


def test_engagement_ramps_over_blend_sec():
    c = comp()
    assert tick(c, error_deg=2.0).static[0] == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize("mapping", [dict(fs=[1.01] + [0.0] * 6), dict(fs=[0.0] * 4 + [0.1, 0, 0]),
                                     dict(fs=[-0.1] + [0.0] * 6), dict(fs_deadband_deg=0.0)])
def test_bad_fs_rejected(mapping):
    with pytest.raises(ValueError):
        sc.CompensationConfig.from_mapping(mapping)


def test_combined_with_integral_stays_inside_integral_limit_and_headroom():
    c = engaged(integral_enabled=True, ki=[10.0] * 7)
    for _ in range(200):
        r = tick(c, error_deg=5.0)
    assert np.all(np.abs(r.static + r.integral) <= np.asarray(c.cfg.integral_limit) + 1e-12)
    # conditional integration stops one rate step (0.02) short of the shared cap
    assert r.static[0] == pytest.approx(0.5) and 0.48 <= r.integral[0] <= 0.5
    near = tick(c, error_deg=5.0, base_torque=np.ones(7) * 4.8)
    assert np.all(near.estimated_total <= 5 + 1e-12)
    assert np.all(near.saturated[:4])


def test_never_settled_gives_zero():
    c = comp()
    for _ in range(20):
        assert not np.any(tick(c, settled=False).static)


@pytest.mark.parametrize("flag", ["contact", "stale", "stopped"])
def test_safety_events_cut_immediately(flag):
    c = engaged()
    assert not np.any(tick(c, **{flag: True}).static)


def test_target_change_fades_monotonically_with_held_direction():
    c = engaged()
    out = [tick(c, error_deg=2.0, target_changed=True).static[0]]
    for _ in range(12):  # new target on the other side must not flip the push
        out.append(tick(c, error_deg=-2.0, settled=False).static[0])
    assert np.all(np.diff(out) <= 0) and min(out) >= 0
    assert out[0] < 0.5 and out[-1] == 0.0


def test_fs_duplicates_rejected_but_integral_allowed():
    cfg = sc.CompensationConfig.from_mapping({"fs": FS, "integral_enabled": True, "ki": [1.0] * 7})
    sc.validate_no_duplicate_integrators(cfg)
    for flag in ("home_integral", "jog_integral", "droop_integral"):
        with pytest.raises(ValueError):
            sc.validate_no_duplicate_integrators(
                sc.CompensationConfig.from_mapping({"fs": FS}), **{flag: True})
    with pytest.raises(ValueError):
        sc.validate_no_duplicate_integrators(sc.CompensationConfig.from_mapping(
            {"fs": FS, "friction_enabled": True, "fc": [0.1] * 7}))


def sign_flips(errors):
    flips = 0
    for column in errors.T:
        signs = np.sign(column[np.abs(column) > 0.01])
        flips += int(np.sum(signs[1:] != signs[:-1]))
    return flips


def test_fake_stiction_error_drops_below_0_2_deg_without_chatter():
    before = simulate_static_friction(0.0)
    after = simulate_static_friction(0.6)
    kp = np.array([70.0, 70.0, 70.0, 60.0])
    assert np.abs(before[-1]) == pytest.approx(np.rad2deg(0.9 * 0.6 / kp), abs=0.03)
    assert np.max(np.abs(after[-120:])) < 0.2
    assert sign_flips(after) == 0


# Frozen from 2fe2b4f (before fs existed): fs = 0 must not change any output.
GOLDEN = {
    ("constant_load", True): (1374.4129524549403, 0.1641707683918513),
    ("payload_bias", False): (1576.0435671993457, 0.14999938776051638),
    ("payload_bias", True): (2950.4602366232093, 0.31417062099351495),
    ("friction_quantization", True): (1353.4948756000103, 0.1585892000000017),
    ("delay", True): (1371.8114155809476, 0.16416178037745147),
    ("saturation", True): (0.0, 0.0),
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_fs_zero_output_unchanged(key):
    ff = np.array([r["torque_feedforward"] for r in simulate(*key)["rows"]])
    total, last = GOLDEN[key]
    assert float(np.abs(ff).sum()) == pytest.approx(total, rel=1e-12, abs=1e-12)
    assert ff[-1][:4] == pytest.approx([last] * 4, rel=1e-12, abs=1e-12)
    assert all(r["torque_static_friction"] == [0.0] * 7 for r in simulate(*key)["rows"])


@pytest.mark.parametrize("name", ["pd_rh56f1.yaml", "pd_rh56f1_exec.yaml", "pd_rh56f1_fake.yaml",
                                  "pd_rh56f1_stationary.yaml", "pd_rh56f1_stationary_left.yaml",
                                  "pd_rh56f1_stationary_left_exec.yaml",
                                  "pd_rh56f1_stationary_left_home_setup.yaml"])
def test_existing_profiles_keep_fs_off(name):
    comp_cfg = load_pd_config(CONFIG / name).stationary_compensation
    assert comp_cfg is None or not any(sc.CompensationConfig.from_mapping(comp_cfg).fs)


@pytest.mark.parametrize("name, fs_large", [
    ("pd_rh56f1_stationary_left_fric.yaml", [0.329, 0.413, 0.511, 0.539]),
    ("pd_rh56f1_stationary_right_fric.yaml", [0.287, 0.294, 0.392, 0.546]),
])
def test_fric_profiles_load_with_integral_and_scaled_fs(name, fs_large):
    text = (CONFIG / name).read_text()
    assert text.startswith("# 실기 승인 전 사용 금지")
    pd = load_pd_config(CONFIG / name)
    assert pd.execute is False
    c = sc.CompensationConfig.from_mapping(pd.stationary_compensation)
    assert c.integral_enabled and c.move_split_deg == 2.0
    # Small-move floor (right-arm data for both arms) and 0.7 x fit after large moves.
    assert list(c.fs) == pytest.approx([0.22, 0.25, 0.45, 0.38] + [0.0] * 3)
    assert list(c.fs_large) == pytest.approx(fs_large + [0.0] * 3)
    # The integral must still reach the right j1 offset (~0.42 N·m) on top of fs.
    assert c.integral_limit[0] >= 0.42 + c.fs[0]
    assert max(c.integral_limit) <= sc.FS_MAX


def two_stage(moved_deg, **changes):
    """Move moved_deg (velocity while unsettled), then settle with 1 deg error."""
    c = comp(fs_large=[0.8] * 4 + [0.0] * 3, **changes)
    for _ in range(10):
        tick(c, settled=False, velocity=np.ones(7) * np.deg2rad(moved_deg))  # 1 s
    for _ in range(10):
        r = tick(c)
    return c, r


def test_two_stage_picks_fs_large_only_after_large_move():
    _, small = two_stage(1.0)
    _, large = two_stage(3.0)
    assert small.static[:4] == pytest.approx([FS[0]] * 4, abs=1e-6)
    assert large.static[:4] == pytest.approx([0.8] * 4, abs=1e-6)
    _, split = two_stage(3.0, move_split_deg=5.0)
    assert split.static[:4] == pytest.approx([FS[0]] * 4, abs=1e-6)


def test_move_count_frozen_while_armed_and_reset_on_target_change():
    c, _ = two_stage(1.0)
    for _ in range(30):  # breakaway motion after arming must not count
        r = tick(c, velocity=np.ones(7) * np.deg2rad(1.0))
    assert r.static[:4] == pytest.approx([FS[0]] * 4, abs=1e-6)
    c, _ = two_stage(3.0)
    tick(c, target_changed=True)
    for _ in range(10):
        r = tick(c)
    assert r.static[:4] == pytest.approx([FS[0]] * 4, abs=1e-6)


@pytest.mark.parametrize("mapping", [
    {"fs_large": [1.01] + [0.0] * 6},
    {"fs_large": [0.0] * 4 + [0.1, 0.0, 0.0]},
    {"move_split_deg": 0.0},
    {"integral_limit": [1.2] + [1.0] * 3 + [0.3] * 3},
])
def test_bad_two_stage_rejected(mapping):
    with pytest.raises(ValueError):
        sc.CompensationConfig.from_mapping({"fs": FS, **mapping})


def test_fake_move_floor_two_stage_integral_reaches_band_without_chatter():
    from policy_control.stationary_fake import move_metrics, simulate_move

    full = {
        "fs": [0.22, 0.25, 0.45, 0.38], "fs_large": [0.287, 0.294, 0.392, 0.546],
        "integral_enabled": True, "ki": [14.0, 14.0, 14.0, 12.0],
        "integral_limit": [1.0] * 4, "integral_rate": [0.2] * 4,
    }
    for move in (0.4, 5.0):
        m = move_metrics(simulate_move(full, move, seconds=12.0))
        assert max(m["final_deg"]) < 0.2 and max(m["flips"]) <= 1
    off = move_metrics(simulate_move({}, 0.4))
    assert min(off["final_deg"]) > 0.2  # small-move stiction is left uncorrected

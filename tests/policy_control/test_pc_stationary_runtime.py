"""The experimental torque path must preserve motor position and existing safety."""

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control.pd_law import PdCommand, load_pd_config


def runtime(config=None):
    from policy_control.pd_stationary import StationaryRuntime

    return StationaryRuntime(
        config,
        kp=np.ones(7) * 70,
        kd=np.ones(7),
        torque_limit=np.ones(7) * 7,
        blend_sec=1.0,
    )


def tick(r, **kw):
    args = dict(
        q_meas=np.zeros(7),
        qd_meas=np.zeros(7),
        target=np.ones(7) * 0.01,
        target_velocity=np.zeros(7),
        phase="TRACKING",
        contact=False,
        stale=False,
        episode=1,
        dt=0.01,
        gravity=np.ones(7),
        request=np.zeros(7),
    )
    args.update(kw)
    cmd = PdCommand(np.ones(7) * 0.01, np.zeros(7), np.ones(7), (), False)
    return r.apply(cmd, **args)


def test_disabled_preserves_every_command_field():
    r = runtime()
    cmd, diag = tick(r)
    assert np.array_equal(cmd.q, np.ones(7) * 0.01)
    assert np.array_equal(cmd.tau, np.ones(7))
    assert diag["torque_estimated_total"] == pytest.approx([1.7] * 7)


def test_integral_changes_only_torque_and_requires_known_noncontact():
    r = runtime({"integral_enabled": True, "ki": [10.0] * 7})
    for _ in range(200):
        cmd, d = tick(r, contact=None)
    assert d["torque_integral"] == [0.0] * 7
    for _ in range(200):
        cmd, d = tick(r)
    assert np.array_equal(cmd.q, np.ones(7) * 0.01)
    assert min(d["torque_integral"]) > 0
    before = d["torque_integral"]
    for _ in range(20):
        _, d = tick(r, phase="HOLD", contact=True)
    assert d["torque_integral"] == before


def test_moving_target_cannot_integrate_and_fresh_episode_fades():
    r = runtime({"integral_enabled": True, "ki": [10.0] * 7})
    for _ in range(200):
        _, d = tick(r, target_velocity=np.ones(7) * 0.02)
    assert d["torque_integral"] == [0.0] * 7
    for _ in range(200):
        _, d = tick(r)
    prior = np.array(d["torque_integral"])
    _, d = tick(r, episode=2)
    assert np.max(np.abs(np.array(d["torque_integral"]) - prior)) <= 0.00201
    for _ in range(201):  # Wrist rate cap can require longer than the nominal blend.
        _, d = tick(r, episode=2, phase="RAMPING")
    assert d["torque_integral"] == pytest.approx([0.0] * 7)


def test_compensation_config_is_optional_and_experiment_only(tmp_path):
    import yaml

    path = (
        Path(__file__).resolve().parents[2]
        / "deploy/policy_control/config/pd_rh56f1.yaml"
    )
    # 11.08: 운영 설정은 모두 끈 블록(diagnostics 용)만 가진다. 하나라도 켜면 실험 프로필이 필요하다.
    block = load_pd_config(path).stationary_compensation
    assert block is not None and not any(block.values())
    data = yaml.safe_load(path.read_text())
    data["stationary_compensation"] = {"integral_enabled": True}
    out = tmp_path / "pd.yaml"
    out.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="stationary_tracking"):
        load_pd_config(out)


def test_feedforward_cap_is_preserved_even_when_pd_opposes_it():
    r = runtime({"friction_enabled": True, "fo": [10.0] * 7})
    cmd, d = tick(r, q_meas=np.ones(7) * 0.08)
    assert np.max(np.abs(cmd.tau)) <= 7.0


def test_continuously_changing_target_does_not_restart_integral_fade():
    r = runtime({"integral_enabled": True, "ki": [10.0] * 7})
    for _ in range(100):
        tick(r)
    for i in range(250):
        _, d = tick(r, target=np.ones(7) * (0.01 + i * 0.0001), phase="RAMPING")
    assert d["torque_integral"] == pytest.approx([0.0] * 7)


def test_new_engagement_never_reuses_previous_integral():
    r = runtime({"integral_enabled": True, "ki": [10.0] * 7})
    for _ in range(100):
        tick(r)
    r.reset()
    _, diag = tick(r, phase="RAMPING")
    assert diag["torque_integral"] == [0.0] * 7


def test_real_arm_tick_wires_same_tick_diagnostics_and_live_gains(monkeypatch):
    from policy_control import pd_arm
    from policy_control.codec import JointSample
    from policy_control.contract import load_contract
    from policy_control.sources import load_robot_cfg, load_profile

    root = Path(__file__).resolve().parents[2]
    cfg = load_pd_config(
        root / "deploy/policy_control/config/pd_rh56f1_stationary.yaml"
    )
    contract = load_contract(
        root / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
    )
    robot = load_robot_cfg(
        root / "deploy/policy_control/config/robots/rh56f1_right_fake.yaml"
    )
    written = []
    backends = SimpleNamespace(
        arm=SimpleNamespace(write=written.append),
        switch=None,
        hand=None,
        gripper=None,
        hand_gains=None,
    )
    monkeypatch.setattr(pd_arm, "build_side_backends", lambda *a, **kw: backends)
    cfg = replace(
        cfg, stationary_compensation={"integral_enabled": True, "ki": [1.0] * 7}
    )
    unit = pd_arm.ArmUnit(
        None,
        side_cfg=contract.side("right"),
        robot_cfg=robot,
        cfg=cfg,
        contract=contract,
        profile=load_profile(robot.joint_profiles),
        stage_name="reduced",
        execute=False,
    )
    assert not unit.runtime_gains_verified
    params = "".join(
        f'<param name="{kind}{i + 1}">{v}</param>'
        for kind, vals in [("kp", unit.kp), ("kd", unit.kd)]
        for i, v in enumerate(vals)
    )
    unit.on_robot_description(
        '<robot><ros2_control name="openarm_right_hardware_interface"><hardware>'
        + params
        + "</hardware></ros2_control></robot>"
    )
    assert unit.runtime_gains_verified
    q = np.array(unit.home_arm)
    n = len(unit.hand_joints)
    unit.on_joint_state(
        ["arm"],
        JointSample(
            tuple(f"openarm_right_joint{i}" for i in range(1, 8)),
            q,
            np.zeros(7),
            np.zeros(7),
            123.0,
        ),
        10.0,
    )
    unit.on_joint_state(
        ["ee"],
        JointSample(tuple(unit.hand_joints), np.zeros(n), np.zeros(n), None, 123.0),
        10.0,
    )
    unit.on_temperatures({f"openarm_right_joint{i}": 30.0 for i in range(1, 8)}, 10.0)
    unit.on_contact(False, 10.0)
    unit.stage.engage(q)
    unit.take_target(
        JointSample(tuple(unit.arm_joints), q, np.zeros(7), None, 123.0), 1, 10.0
    )
    result = unit.tick(10.0, False)
    assert result.diagnostics["arm_source_stamp"] == 123.0
    assert result.diagnostics["pd_target"] == pytest.approx(q)
    assert result.diagnostics["torque_feedforward"] == pytest.approx(written[-1].tau)
    assert result.diagnostics["temperature"] == [30.0] * 7
    unit.on_robot_description("<robot/>")
    result = unit.tick(10.01, False)
    assert result.status.phase == "HOLD"


def _runtime(config):
    import numpy as np
    from policy_control.pd_stationary import StationaryRuntime

    kp = np.ones(7) * 70.0
    return StationaryRuntime(
        config, kp=kp, kd=np.ones(7) * 8.0, torque_limit=np.ones(7) * 7.0, blend_sec=1.0
    )


def _hold(rt, q, qd, n=60, dt=1.0 / 120.0, **kw):
    import numpy as np
    from policy_control.pd_law import PdCommand

    zero = np.zeros(7)
    cmd = PdCommand(zero.copy(), zero.copy(), zero.copy(), (), False)
    args = dict(target=zero, target_velocity=zero, phase="TRACKING", contact=False,
                stale=False, episode=1, dt=dt, gravity=zero, request=zero)
    args.update(kw)
    return [rt.apply(cmd, q_meas=q(i), qd_meas=qd(i), **args)[1] for i in range(n)]


def test_settles_despite_quantized_velocity_noise():
    """Real DM velocity flickers ±1 step (0.022 rad/s on j1/j2) at standstill; the
    old |qd| < 0.005 test never settled on hardware (261109 right, 96 runs)."""
    import numpy as np

    rt = _runtime({"fs": [0.3, 0.3, 0.3, 0.3, 0, 0, 0], "integral_enabled": True,
                   "ki": [14.0] * 4 + [0.0] * 3})
    q0 = np.r_[np.deg2rad([0.5, -0.5, 0.5, -0.5]), np.zeros(3)]
    rows = _hold(rt, lambda i: q0, lambda i: np.r_[0.022 * (-1) ** i, 0.022, 0.0049, 0, 0, 0, 0])
    assert not rows[0]["settled"] and rows[0]["settle_block"] == ["not_standstill"]
    assert rows[-1]["settled"] and rows[-1]["fs_armed"] and rows[-1]["settle_block"] == []
    assert any(abs(t) > 0 for t in rows[-1]["torque_static_friction"][:4])
    assert any(rows[-1]["integral_updating"][:4])


def test_motion_over_band_blocks_settle_and_j5_j7_ignored():
    import numpy as np

    rt = _runtime({"fs": [0.3, 0.3, 0.3, 0.3, 0, 0, 0]})
    step = np.deg2rad(0.03)  # > 0.05 deg span within the 0.1 s window
    rows = _hold(rt, lambda i: np.r_[i * step, 0, 0, 0, 0, 0, 0], lambda i: np.zeros(7))
    assert not any(r["settled"] for r in rows)
    rt = _runtime({"fs": [0.3, 0.3, 0.3, 0.3, 0, 0, 0]})
    rows = _hold(rt, lambda i: np.r_[0, 0, 0, 0, i * step, 0, 0], lambda i: np.zeros(7))
    assert rows[-1]["settled"]


def test_block_reasons_recorded():
    import numpy as np

    rt = _runtime({"fs": [0.3, 0.3, 0.3, 0.3, 0, 0, 0]})
    rows = _hold(rt, lambda i: np.zeros(7), lambda i: np.zeros(7), contact=True, phase="HOLD")
    assert not rows[-1]["settled"] and not rows[-1]["fs_armed"]
    assert {"phase", "contact"} <= set(rows[-1]["settle_block"])

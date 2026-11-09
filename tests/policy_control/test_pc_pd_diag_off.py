"""11.08 사용자 승인: 운영 pd yaml 에 모두 끈 stationary_compensation 블록을 넣어 pd_<side>/diagnostics 를 켠다.

잠그는 것: 블록이 있는 설정과 없는 설정이 같은 입력 열에서 팔에 쓰는 지령(q · qd · tau · limited)을
비트 단위로 같게 낸다. 다르면 블록을 넣지 않는다.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control.pd_law import load_pd_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "deploy/policy_control/config"
CONTRACT = ROOT / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
OPERATING = ("pd_rh56f1.yaml", "pd_rh56f1_exec.yaml")


def _unit(cfg, side, monkeypatch):
    from policy_control import pd_arm
    from policy_control.contract import load_contract
    from policy_control.sources import load_profile, load_robot_cfg

    contract = load_contract(CONTRACT)
    robot = load_robot_cfg(CONFIG / f"robots/rh56f1_{side}_fake.yaml")
    written = []
    backends = SimpleNamespace(arm=SimpleNamespace(write=written.append), switch=None,
                               hand=None, gripper=None, hand_gains=None)
    monkeypatch.setattr(pd_arm, "build_side_backends", lambda *a, **kw: backends)
    unit = pd_arm.ArmUnit(None, side_cfg=contract.side(side), robot_cfg=robot, cfg=cfg,
                          contract=contract, profile=load_profile(robot.joint_profiles),
                          stage_name="reduced", execute=False)
    return unit, written


def _drive(cfg, side, monkeypatch):
    """같은 합성 입력 열: 램프 → 추종(움직이는 목표 · 처진 측정) → 접촉 → 새 에피소드 → 정지 목표."""
    from policy_control.codec import JointSample

    unit, written = _unit(cfg, side, monkeypatch)
    home = np.array(unit.home_arm, dtype=float)
    n = len(unit.hand_joints)
    rng = np.random.default_rng(1108)
    names = tuple(unit.arm_joints)
    driver = tuple(f"openarm_{side}_joint{i}" for i in range(1, 8))
    unit.stage.engage(home)
    outs, diags = [], 0
    q_meas = home.copy()
    for k in range(900):
        now = 10.0 + k * unit.dt
        q_meas = q_meas + 0.3 * (written[-1].q - q_meas) if written else q_meas
        noise = rng.normal(0.0, 2e-4, 7)
        unit.on_joint_state(["arm"], JointSample(driver, q_meas + noise, rng.normal(0, 1e-3, 7),
                                                 rng.normal(0, 0.2, 7), now), now)
        unit.on_joint_state(["ee"], JointSample(tuple(unit.hand_joints), np.zeros(n), np.zeros(n), None, now), now)
        unit.on_temperatures({j: 30.0 for j in driver}, now)
        unit.on_contact(300 <= k < 400, now)
        if k % 4 == 0:
            episode = "ep1" if k < 500 else "ep2"
            wobble = 0.0 if k >= 750 else 0.05 * np.sin(k / 40.0 + np.arange(7))
            unit.take_target(JointSample(names, home + wobble, np.zeros(7), None, now), k, now, episode=episode)
        result = unit.tick(now, False)
        diags += result.diagnostics is not None
        cmd = written[-1] if written else None
        outs.append(None if cmd is None else (cmd.q.tobytes(), cmd.qd.tobytes(), cmd.tau.tobytes(),
                                              tuple(cmd.limited), bool(cmd.effort_fault)))
    return outs, diags


@pytest.mark.parametrize("name", OPERATING)
@pytest.mark.parametrize("side", ["right", "left"])
def test_all_off_block_keeps_commanded_torque_bitwise(name, side, monkeypatch):
    after = load_pd_config(CONFIG / name)
    block = after.stationary_compensation
    assert block is not None and not any(block.get(k) for k in
                                         ("gravity_enabled", "friction_enabled", "integral_enabled"))
    before = replace(after, stationary_compensation=None)
    out_before, diag_before = _drive(before, side, monkeypatch)
    out_after, diag_after = _drive(after, side, monkeypatch)
    assert sum(o is not None for o in out_before) > 800
    assert len({o[2] for o in out_before if o}) > 100          # torque actually varies over the sequence
    assert out_after == out_before
    assert diag_before == 0 and diag_after > 800                # 블록이 하는 일 = diagnostics 발행뿐

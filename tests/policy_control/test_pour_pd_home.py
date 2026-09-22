"""pd home for the bimanual pour: the control-only contract takes its homes from the pour contract.

pd_node never reads the pour contract; it runs on a control-only DeployContract of the same asset.
Its default homes (``zero`` = attention pose, or ``run:<dir>`` = env.yaml init_state) are NOT the pour
reset pose (the env resets the arms to the hdgp profile ``arm_reset_joint_pos``), so ``pour:<json>``
copies the pour contract's per-side ``arm_reset`` / ``hand_open`` into the control-only contract.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_control import contract_assets as A
from policy_control import pour_contract as PC
from policy_control.contract import ContractError
from pour_trace_util import ASSET, contract

needs_asset = pytest.mark.skipif(not ASSET.manifest.exists(), reason="dg5f-m-short asset missing")


@pytest.fixture(scope="module")
def pour():
    return contract()


@pytest.fixture()
def pour_json(pour, tmp_path):
    path = tmp_path / "pour_contract.json"
    PC.save_contract(pour, path)
    return path


def test_arm_homes_come_from_the_pour_reset_pose(pour, pour_json):
    homes, note = A.arm_homes(f"pour:{pour_json}", ("right", "left"))
    for s in pour.sides:
        assert np.allclose(homes[s.side], s.arm_reset)
    assert "pour" in note and pour_json.name in note


def test_pour_home_differs_from_the_run_init_state(pour_json):
    from pour_trace_util import FIX

    pour_h, _ = A.arm_homes(f"pour:{pour_json}", ("right",))
    init_h, _ = A.arm_homes(f"run:{FIX}", ("right",))
    assert np.abs(np.asarray(pour_h["right"]) - np.asarray(init_h["right"])).max() > 0.1


def test_missing_pour_contract_or_side_is_refused(pour, pour_json, tmp_path):
    with pytest.raises(ContractError, match="pour"):
        A.arm_homes(f"pour:{tmp_path / 'nope.json'}", ("right", "left"))
    one = replace(pour, sides=tuple(s for s in pour.sides if s.side == "right"))
    with pytest.raises(ContractError, match="left"):
        A.pour_homes(one, ("right", "left"))


@needs_asset
def test_control_only_contract_holds_the_pour_reset_pose(pour, pour_json):
    c = A.build_asset_contract(asset=ASSET.name, home=f"pour:{pour_json}")
    assert c.control_only
    for s in pour.sides:
        side = c.sides[s.side]
        assert list(side.arm_joints) == list(s.arm_joints)
        assert np.allclose(side.home_arm, s.arm_reset)
        assert np.allclose([side.home_hand[j] for j in s.hand_joints], s.hand_open)


@needs_asset
def test_pour_home_refuses_another_asset(pour_json):
    with pytest.raises(ContractError, match="asset"):
        A.build_asset_contract(asset="openarm_dg5f-m_bi_rl", home=f"pour:{pour_json}")


@needs_asset
def test_default_homes_are_unchanged():
    c = A.build_asset_contract(asset=ASSET.name)
    assert all(v == 0.0 for s in c.sides.values() for v in s.home_arm)
    assert c.sides["right"].home_hand["r_hj_thumb_3"] == pytest.approx(-0.5)

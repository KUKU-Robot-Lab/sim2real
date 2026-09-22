"""Shared loader for the pour_i11 golden trace (sim, i11 checkpoint, 900 steps x 4 envs)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy_control.contract_assets import ASSETS
from policy_control.fk_numpy import UrdfChainFK
from policy_control.pour_build import build_pour_contract
from policy_control.pour_obs import CupPose, SideState

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "policy_control" / "pour_i11"
HDGP = Path(__file__).resolve().parents[3] / "hdgp"
ASSET = ASSETS["openarm_dg5f-m-short_bi_rl"]


def contract(run: Path | None = None):
    """기본은 i11 픽스처. `fetch_run.py` 로 받은 실제 런 디렉터리도 같은 규약이라 그대로 들어간다."""
    run = run or FIX
    return build_pour_contract(run, run / "trace_meta.json", HDGP, ASSET.urdf, ASSET.name)


def trace(run: Path | None = None):
    """골든 trace. **저장소에 커밋되지 않는다**(수백 MB) — 없으면 받는 법을 알려주고 건너뛴다."""
    run = run or FIX
    npz = run / "trace.npz"
    if not npz.is_file():
        import pytest
        pytest.skip(f"{npz} 가 없다. 받는 법: deploy/policy_control/tools/fetch_run.py "
                    f"--run <label> --checkpoint <ep:NNNN> --trace auto --out {run}")
    return np.load(npz), json.loads((run / "trace_meta.json").read_text())


def make_fk(side):
    return UrdfChainFK(ASSET.urdf, side.arm_joints, side.hand_joints, side.palm_body, side.tip_bodies)


def side_state(z, meta, side, fk, t, e, use_fk):
    """SideState from trace row (t, e). use_fk: palm/tips from URDF FK of joint_pos, else sim body poses."""
    names = meta["joint_names"]
    jp = dict(zip(names, z["joint_pos"][t, e]))
    jv = dict(zip(names, z["joint_vel"][t, e]))
    r = side.role
    pose = fk.palm_pose([jp[n] for n in side.arm_joints], [jp[n] for n in side.hand_joints])
    if use_fk:
        palm, tips = pose.palm_pos, pose.tips
    else:
        palm, tips = z[f"{r}_palm_pose"][t, e, :3], z[f"{r}_tips"][t, e]
    return SideState(jp, jv, palm, pose.extra["palm_rot"], tips,
                     CupPose(z[f"{r}_cup_pos"][t, e], z[f"{r}_cup_quat"][t, e]), z[f"{r}_syn_target"][t, e])

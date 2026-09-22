"""Build a `PourContract` from a run dump + hdgp pair profile + asset URDF + sim meta.

Nothing here is typed by hand: numbers come from env.yaml/agent.yaml, names and poses from the
hdgp robot profile of `pair_name`, hand limits from the asset URDF, and the three values that only
exist inside Isaac (PhysX DOF order, fabric-FK anchor, fab_to_env) from the sim meta json written by
`play.py --trace_steps`.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import yaml

from .pour_contract import (FAMILY, SCHEMA, FillLevelCfg, PourContract, PourFabricCfg, PourContractError,
                            PourSideCfg, file_md5, file_sha1, validate)
from .pour_profiles import load_pair

FILL_NOTE = ("No real sensor. Operator/estimator supplies how full the SOURCE cup is, 0..1, fixed for the episode "
             "(sim: settled bead mean height x2 / cup inner height, latched at hold end; "
             "hdgp pour_rules.py:38-50, pour_fabric_env.py:520-524). "
             "default=None means the node refuses to run without it.")

# pour_fabric_env.py builds SideRig(src, oppose_sign=+1) and SideRig(rcv, oppose_sign=-1); not in the dump.
OPPOSE_SIGN = {"src": 1.0, "rcv": -1.0}


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_multi_constructor("tag:yaml.org,2002:python/tuple", lambda ldr, _s, n: tuple(ldr.construct_sequence(n)))
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda _l, _s, _n: None)


def _yaml(path: Path) -> dict:
    try:
        return yaml.load(Path(path).read_text(), Loader=_Loader)
    except (OSError, yaml.YAMLError) as exc:
        raise PourContractError(f"cannot read {path}: {exc}") from exc


def is_pour_run(env_yaml: Path) -> bool:
    text = Path(env_yaml).read_text()
    return all(re.search(rf"^{k}:", text, re.M) for k in ("pair_name", "src_palm_delta_lo", "rcv_palm_delta_lo"))


def _need(env: dict, key: str):
    if key not in env or env[key] is None:
        raise PourContractError(f"env.yaml has no '{key}'")
    return env[key]


def _urdf_limits(urdf: Path, joints) -> tuple:
    import xml.etree.ElementTree as ET

    limits = {j.get("name"): j.find("limit") for j in ET.parse(urdf).getroot().iter("joint")}
    out = []
    for n in joints:
        lim = limits.get(n)
        lo = None if lim is None else lim.get("lower")
        hi = None if lim is None else lim.get("upper")
        if lo is None or hi is None:                 # upper 도 반드시 본다 — 없으면 float(None) 이 TypeError 로 터진다
            raise PourContractError(f"{urdf.name}: no limit for {n} (lower={lo!r} upper={hi!r})")
        out.append((float(lo), float(hi)))
    return tuple(out)


def _delta(env: dict, key: str) -> tuple:
    v = [float(x) for x in _need(env, key)]
    if len(v) != 6:
        raise PourContractError(f"{key} must have 6 values")
    return tuple(v[:3] + [math.radians(x) for x in v[3:]])


def _side(role: str, prefix: str, profile, env: dict, meta: dict, urdf: Path) -> PourSideCfg:
    names = meta["joint_names"]
    arm = tuple(names[i] for i in meta[f"{role}_arm_ids"])
    hand_obs = tuple(names[i] for i in meta[f"{role}_hand_ids"])
    hand_syn = tuple(names[i] for i in meta[f"{role}_syn_ids"])
    fab = tuple(names[i] for i in meta[f"{role}_fab_ids"])
    if hand_syn != tuple(profile.hand_joint_names):
        raise PourContractError(f"{role}: sim synergy order differs from profile.hand_joint_names")
    if fab != tuple(profile.fabric_joint_order):
        raise PourContractError(f"{role}: sim fabric order differs from profile.fabric_joint_order")
    if not all(re.fullmatch(profile.arm_joint_regex, n) for n in arm):
        raise PourContractError(f"{role}: arm joints {arm} do not match {profile.arm_joint_regex}")
    fingers = tuple(profile.finger_sensor_bodies.keys())
    if len(profile.contact_group_a) != 1:
        raise PourContractError(f"{role}: grip3 needs exactly one opposing finger")
    thumb = profile.contact_group_a[0]
    jf, jc, sfx = [], [], []
    for n in hand_syn:
        hit = [i for i, f in enumerate(fingers) if f"_{f}_" in n]
        if len(hit) != 1:
            raise PourContractError(f"{role}: cannot place {n} on one finger")
        s = n.rsplit("_", 1)[1]
        jf.append(hit[0])
        jc.append(int(profile.hand_channel_of_joint[s]))
        sfx.append(s)
    d = float(_need(env, "oppose_grip_delta_rad")) * OPPOSE_SIGN[role]
    grip = [profile.hand_open_pose[i] + d if (d != 0.0 and jc[i] == 1 and fingers[jf[i]] == thumb) else g
            for i, g in enumerate(profile.hand_grip_pose)]
    lim = _urdf_limits(urdf, hand_syn)
    # grip pose is NOT clamped (side_rig only reports it); only the lerped target is clamped to limits
    frz = tuple(profile.hand_freeze_suffixes)
    return PourSideCfg(
        role=role, side={"r": "right", "l": "left"}[arm[0][0]], arm_joints=arm, hand_obs_joints=hand_obs,
        hand_joints=hand_syn, palm_body=profile.palm_body, tip_bodies=tuple(profile.fingertip_bodies),
        fingers=fingers, thumb=thumb,
        grp_a=tuple(fingers.index(f) for f in profile.contact_group_a),
        grp_b=tuple(fingers.index(f) for f in profile.contact_group_b), joint_finger=tuple(jf), joint_channel=tuple(jc),
        joint_is_mid=tuple(s in frz and s == "3" for s in sfx),
        joint_is_dist=tuple(s in frz and s != "3" for s in sfx),
        arm_reset=tuple(float(v) for v in profile.arm_reset_joint_pos),
        hand_open=tuple(float(v) for v in profile.hand_open_pose), hand_grip=tuple(grip),
        hand_lo=tuple(lo for lo, _ in lim), hand_hi=tuple(h for _, h in lim),
        anchor=tuple(meta[f"{role}_anchor"]), fab_to_env=tuple(meta[f"{role}_fab_to_env"]),
        delta_lo=_delta(env, f"{prefix}_palm_delta_lo"), delta_hi=_delta(env, f"{prefix}_palm_delta_hi"),
        box_lo=tuple(profile.palm_box_min), box_hi=tuple(profile.palm_box_max),
        fabric_joint_order=fab, fabric_class=profile.fabric_class, fabric_robot_dir=profile.fabric_robot_dir,
        fabric_params=profile.fabric_params_filename)


def build_pour_contract(run_dir: Path, sim_meta: Path, hdgp_root: Path, urdf: Path, asset: str,
                        checkpoint: Path | None = None, fill_default: float | None = None) -> PourContract:
    run = Path(run_dir)
    env_yaml, agent_yaml = run / "params" / "env.yaml", run / "params" / "agent.yaml"
    if not is_pour_run(env_yaml):
        raise PourContractError(f"{env_yaml} is not a pour_fabric run dump")
    env, agent = _yaml(env_yaml), _yaml(agent_yaml)["params"]
    meta = json.loads(Path(sim_meta).read_text())
    if str(_need(env, "synergy_freeze_scope")) != "joint":
        raise PourContractError("only synergy_freeze_scope 'joint' is implemented")
    if float(_need(env, "finger_residual_scale")) != 0.0 or not bool(_need(env, "couple_four_fingers")):
        raise PourContractError("decoder implements coupled four fingers with residual 0 only")
    alpha = float(_need(env, "palm_action_ema_alpha"))
    if abs(alpha - float(meta["palm_ema_alpha"])) > 1e-12:
        raise PourContractError("sim meta palm_ema_alpha differs from env.yaml (meta from another run?)")
    policy_dt = float(env["sim"]["dt"]) * int(env["decimation"])
    if abs(policy_dt - float(meta["policy_dt"])) > 1e-9:
        raise PourContractError("sim meta policy_dt differs from env.yaml")
    pair = load_pair(Path(hdgp_root), str(_need(env, "pair_name")))
    sides = (_side("src", "src", pair.source, env, meta, Path(urdf)),
             _side("rcv", "rcv", pair.receiver, env, meta, Path(urdf)))
    clip = agent.get("env", {}).get("clip_observations")
    ckpt = Path(checkpoint) if checkpoint else None
    return validate(PourContract(
        schema=SCHEMA, family=FAMILY, task=str(agent["config"]["name"]), run_dir=str(run),
        checkpoint=str(ckpt) if ckpt else "", checkpoint_md5=file_md5(ckpt) if ckpt else "",
        env_yaml_sha1=file_sha1(env_yaml), agent_yaml_sha1=file_sha1(agent_yaml), asset=asset,
        policy_hz=1.0 / policy_dt, fabric_dt=float(_need(env, "fabrics_dt")),
        fabric_decimation=int(_need(env, "fabric_decimation")),
        obs_dim=int(_need(env, "observation_space")), action_dim=int(_need(env, "action_space")),
        mlp_units=tuple(int(u) for u in agent["network"]["mlp"]["units"]),
        normalize_input=bool(agent["config"]["normalize_input"]),
        obs_clip=None if clip is None else float(clip), action_clip=1.0,
        hold_steps=int(_need(env, "hold_steps")), palm_ema_alpha=alpha,
        cup_mouth_z=float(_need(env, "cup_mouth_z")), joint_pos_err_max=float(_need(env, "joint_pos_err_max")),
        hand_action_mode=str(_need(env, "hand_action_mode")),
        synergy_close_speed=float(_need(env, "synergy_close_speed")),
        synergy_contact_freeze=bool(_need(env, "synergy_contact_freeze")),
        contact_force_threshold=float(_need(env, "contact_force_threshold")),
        close_gate_enabled=bool(_need(env, "close_gate_enabled")),
        close_gate_radius=float(_need(env, "close_gate_radius")), close_gate_ramp=float(_need(env, "close_gate_ramp")),
        fill_level=FillLevelCfg("external", 0.0, 1.0, fill_default, FILL_NOTE),
        fabric=PourFabricCfg(
            damping=float(_need(env, "fabrics_damping_gain")),
            max_objects=int(_need(env, "fabrics_max_objects_per_env")),
            vel_ff_scale=float(_need(env, "fabric_velocity_ff_scale")),
            use_hand_repulsion=bool(_need(env, "use_hand_repulsion")),
            use_body_repulsion_pairs=bool(_need(env, "use_body_repulsion_pairs")),
            table_obstacle=bool(_need(env, "fabric_table_obstacle")),
            table_margin_xy=float(_need(env, "fabric_table_margin_xy")),
            table_thickness=float(_need(env, "fabric_table_thickness")),
            table_z=float(_need(env, "table_surface_z"))),
        sides=sides))

"""Bimanual pour observation (223) - numpy mirror of hdgp `pour_fabric_env._side_obs/_get_observations`.

obs = [src block 99][rcv block 99][rcv_cup - src_cup 3][rcv_mouth - src_mouth 3][fill 1][prev_actions 18]
side block = arm_q, arm_qd, hand_q (PhysX DOF order), palm pos, palm R col0+col1, tips - palm,
             cup - palm, tips - cup, joint_err (synergy order), cup_up.
All positions live in the robot base (= sim env-local) frame. Joint vectors enter as name->value
maps and are ordered BY NAME from the contract; slices of a driver array are never trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .pour_contract import PourContract, PourSideCfg


class PourObsError(ValueError):
    """An input needed for the observation is missing, malformed or non-finite."""


@dataclass(frozen=True)
class CupPose:
    pos: np.ndarray           # (3,) base frame
    quat: np.ndarray          # (4,) wxyz


@dataclass(frozen=True)
class SideState:
    joint_pos: Mapping        # joint name -> rad (arm + hand)
    joint_vel: Mapping        # joint name -> rad/s (arm only is read; hand velocity is not an actor input)
    palm_pos: np.ndarray      # (3,)
    palm_R: np.ndarray        # (3, 3)
    tips: np.ndarray          # (T, 3) contract tip_bodies order
    cup: CupPose              # this side's cup (src: source cup, rcv: receiver cup)
    hand_target: np.ndarray   # (J,) synergy-order joint target from the decoder


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = float(q[0]), np.asarray(q[1:], float)
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _named(values: Mapping, names, what: str) -> np.ndarray:
    missing = [n for n in names if n not in values]
    if missing:
        raise PourObsError(f"{what}: missing joints {missing}")
    return np.array([float(values[n]) for n in names])


def resolve_fill_level(c: PourContract, value: float | None) -> float:
    """External operator/estimator input. No value and no documented default -> refuse (never a silent 0)."""
    if value is None:
        if c.fill_level.default is None:
            raise PourObsError("fill_level has no sensor: provide the operator/estimator value (0..1)")
        value = c.fill_level.default
    value = float(value)
    if not np.isfinite(value) or not c.fill_level.lo <= value <= c.fill_level.hi:
        raise PourObsError(f"fill_level {value} outside [{c.fill_level.lo}, {c.fill_level.hi}]")
    return value


def mouth(c: PourContract, cup: CupPose) -> np.ndarray:
    return np.asarray(cup.pos, float) + quat_apply(cup.quat, np.array([0.0, 0.0, c.cup_mouth_z]))


def side_block(c: PourContract, side: PourSideCfg, st: SideState) -> dict:
    palm = np.asarray(st.palm_pos, float).reshape(3)
    R = np.asarray(st.palm_R, float).reshape(3, 3)
    tips = np.asarray(st.tips, float).reshape(len(side.tip_bodies), 3)
    cup_p = np.asarray(st.cup.pos, float).reshape(3)
    hand_syn = _named(st.joint_pos, side.hand_joints, f"{side.role} hand")
    err = (np.asarray(st.hand_target, float) - hand_syn) / c.joint_pos_err_max
    return {
        "arm_q": _named(st.joint_pos, side.arm_joints, f"{side.role} arm"),
        "arm_qd": _named(st.joint_vel, side.arm_joints, f"{side.role} arm vel"),
        "hand_q": _named(st.joint_pos, side.hand_obs_joints, f"{side.role} hand"),
        "palm_pos": palm,
        "palm_rot6": np.concatenate([R[:, 0], R[:, 1]]),
        "tips_rel_palm": (tips - palm).reshape(-1),
        "cup_rel_palm": cup_p - palm,
        "tips_rel_cup": (tips - cup_p).reshape(-1),
        "joint_err": np.clip(err, -1.0, 1.0),
        "cup_up": quat_apply(st.cup.quat, np.array([0.0, 0.0, 1.0])),
    }


def segment_slices(c: PourContract, states_like: dict | None = None) -> dict:
    """{"src/arm_q": slice, ..., "common/prev_actions": slice} for tests, logging and docs."""
    out, cur = {}, 0
    for s in c.sides:
        na, nh, nt = len(s.arm_joints), len(s.hand_obs_joints), len(s.tip_bodies)
        for name, w in (("arm_q", na), ("arm_qd", na), ("hand_q", nh), ("palm_pos", 3), ("palm_rot6", 6),
                        ("tips_rel_palm", 3 * nt), ("cup_rel_palm", 3), ("tips_rel_cup", 3 * nt),
                        ("joint_err", nh), ("cup_up", 3)):
            out[f"{s.role}/{name}"] = slice(cur, cur + w)
            cur += w
    for name, w in (("cup_rel", 3), ("mouth_rel", 3), ("fill_level", 1), ("prev_actions", c.action_dim)):
        out[f"common/{name}"] = slice(cur, cur + w)
        cur += w
    if cur != c.obs_dim:
        raise PourObsError(f"layout is {cur} wide, contract obs_dim is {c.obs_dim}")
    return out


def build_obs(c: PourContract, states: Mapping, fill_level: float | None, prev_actions: np.ndarray) -> np.ndarray:
    """states: {"src": SideState, "rcv": SideState}; prev_actions from `PourDecoder.prev_action_obs`."""
    parts = []
    for side in c.sides:
        if side.role not in states:
            raise PourObsError(f"missing state for '{side.role}'")
        parts.extend(side_block(c, side, states[side.role]).values())
    src, rcv = states["src"].cup, states["rcv"].cup
    pa = np.asarray(prev_actions, float).reshape(-1)
    if pa.size != c.action_dim:
        raise PourObsError(f"prev_actions has {pa.size} values, contract wants {c.action_dim}")
    parts += [np.asarray(rcv.pos, float) - np.asarray(src.pos, float), mouth(c, rcv) - mouth(c, src),
              np.array([resolve_fill_level(c, fill_level)]), pa]
    obs = np.concatenate(parts)
    if obs.size != c.obs_dim:
        raise PourObsError(f"obs is {obs.size} wide, contract wants {c.obs_dim}")
    if not np.all(np.isfinite(obs)):
        raise PourObsError("obs contains NaN/Inf")
    return obs


def clip_obs(c: PourContract, obs: np.ndarray) -> np.ndarray:
    """rl_games wrapper clip_observations, applied before input normalisation (as in training)."""
    return obs if c.obs_clip is None else np.clip(obs, -c.obs_clip, c.obs_clip)

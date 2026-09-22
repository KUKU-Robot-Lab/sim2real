"""Bimanual pour decoder - numpy mirror of hdgp pour_fabric `side_rig.py` action handling.

Per side: palm 6D = anchor + sign-split asymmetric delta (a=0 is the anchor), position boxed in
env frame; hand = grip3 (thumb opposition, thumb ch2, four-finger close) expanded to a
(finger, channel) grid, slewed per joint with close gate and contact freeze, then lerped
open->grip and clamped to joint limits. The palm EMA and the +-1 clamp live in `PourDecoder`
because the filtered palm command is also what the next observation must carry.

Everything numeric comes from `PourContract`; nothing task specific is hard coded here.
Inputs are never mutated; state is replaced, not edited in place.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pour_contract import PourContract, PourSideCfg

PALM_DIM = 6
HAND_DIM = 3


class PourDecodeError(ValueError):
    """The action or the per-step inputs do not match the contract."""


@dataclass(frozen=True)
class SideTargets:
    palm_cmd: np.ndarray      # (6,) EMA-filtered palm action (goes back into the next obs)
    palm_target: np.ndarray   # (6,) fabric-frame palm pose target [xyz, euler]
    syn_close: np.ndarray     # (J,) per-joint closure state 0..1
    hand_target: np.ndarray   # (J,) joint targets in the contract's synergy joint order


@dataclass(frozen=True)
class SideInputs:
    """Real-side measurements the hand slew needs each step (previous control step)."""

    close_gate: float         # 0..1, palm<->cup distance ramp (see `close_gate`)
    f_mid: np.ndarray         # (F,) middle-phalanx contact force per finger [N]
    f_dist: np.ndarray        # (F,) distal-phalanx contact force per finger [N]


def close_gate(palm_pos: np.ndarray, cup_pos: np.ndarray, radius: float, ramp: float, enabled: bool,
               grasped: bool = False) -> float:
    """Closing permission 0..1. `ramp` is a FRACTION of `radius`; once grasped the gate latches open (1)."""
    if not enabled or grasped:
        return 1.0
    d = float(np.linalg.norm(np.asarray(palm_pos, float) - np.asarray(cup_pos, float)))
    return float(np.clip((radius - d) / max(ramp * radius, 1e-6), 0.0, 1.0))


def grasped(side: PourSideCfg, forces: np.ndarray, threshold: float) -> bool:
    """Opposing contact groups both above threshold (hdgp SideRig.grasped). forces: (F,) per-finger [N]."""
    f = np.asarray(forces, float)
    return bool((f[list(side.grp_a)] > threshold).any() and (f[list(side.grp_b)] > threshold).any())


def palm_target(side: PourSideCfg, a6: np.ndarray, active: bool) -> np.ndarray:
    anchor = np.asarray(side.anchor, float)
    if not active:
        return anchor.copy()
    a6 = np.asarray(a6, float)
    lo, hi = np.asarray(side.delta_lo, float), np.asarray(side.delta_hi, float)
    raw = anchor + np.where(a6 >= 0.0, a6 * hi, -a6 * lo)
    f2e = np.asarray(side.fab_to_env, float)
    pos_env = np.clip(raw[:3] + f2e, side.box_lo, side.box_hi)
    return np.concatenate([pos_env - f2e, raw[3:]])


def expand_grip3(side: PourSideCfg, a3: np.ndarray) -> np.ndarray:
    """(3,) -> (F, 3) grid: fingers get a3[2] on ch1/ch2, the thumb gets a3[0], a3[1]; ch0 stays 0."""
    grid = np.zeros((len(side.fingers), 3))
    grid[:, 1] = a3[2]
    grid[:, 2] = a3[2]
    thumb = side.fingers.index(side.thumb)
    grid[thumb, 1] = a3[0]
    grid[thumb, 2] = a3[1]
    return grid


def step_close(c: PourContract, side: PourSideCfg, close: np.ndarray, a3: np.ndarray, inp: SideInputs) -> np.ndarray:
    fi = np.asarray(side.joint_finger)
    ch = np.asarray(side.joint_channel)
    cmd = 0.5 * (expand_grip3(side, a3) + 1.0)[fi, ch]
    delta = np.clip(cmd - close, -c.synergy_close_speed, c.synergy_close_speed)
    delta = np.where(delta > 0.0, delta * float(inp.close_gate), delta)
    if c.synergy_contact_freeze:
        thr = c.contact_force_threshold
        h_mid = (np.asarray(inp.f_mid, float) > thr)[fi] & np.asarray(side.joint_is_mid)
        h_dist = (np.asarray(inp.f_dist, float) > thr)[fi] & np.asarray(side.joint_is_dist)
        delta = np.where((h_mid | h_dist) & (delta > 0.0), 0.0, delta)
    return np.clip(close + delta, 0.0, 1.0)


def hand_target(side: PourSideCfg, close: np.ndarray) -> np.ndarray:
    o, g = np.asarray(side.hand_open, float), np.asarray(side.hand_grip, float)
    return np.clip(o + (g - o) * close, side.hand_lo, side.hand_hi)


class PourDecoder:
    """Stateful per-episode decoder for both sides. `step` returns new targets; nothing is edited in place."""

    def __init__(self, contract: PourContract):
        self.c = contract
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._palm_cmd = {r: np.zeros(PALM_DIM) for r in self.c.roles}
        self._close = {r: np.zeros(len(self.c.side(r).hand_joints)) for r in self.c.roles}

    @property
    def active(self) -> bool:
        return self._step >= self.c.hold_steps

    def step(self, action: np.ndarray, inputs: dict[str, SideInputs]) -> dict[str, SideTargets]:
        a = np.asarray(action, float).reshape(-1)
        if a.size != self.c.action_dim:
            raise PourDecodeError(f"action has {a.size} values, contract wants {self.c.action_dim}")
        if not np.all(np.isfinite(a)):
            raise PourDecodeError("action contains NaN/Inf")
        a = np.clip(a, -self.c.action_clip, self.c.action_clip)
        # hold phase: EMA and synergy still see the raw action; only the palm target (anchor) and gate (0) change
        active = self.active
        alpha = self.c.palm_ema_alpha
        out, new_cmd, new_close = {}, {}, {}
        for i, role in enumerate(self.c.roles):
            if role not in inputs:
                raise PourDecodeError(f"missing SideInputs for '{role}'")
            side = self.c.side(role)
            base = i * (PALM_DIM + HAND_DIM)
            cmd = alpha * a[base:base + PALM_DIM] + (1.0 - alpha) * self._palm_cmd[role]
            gate_in = inputs[role] if active else SideInputs(0.0, inputs[role].f_mid, inputs[role].f_dist)
            close = step_close(self.c, side, self._close[role], a[base + PALM_DIM:base + PALM_DIM + HAND_DIM], gate_in)
            new_cmd[role], new_close[role] = cmd, close
            out[role] = SideTargets(cmd, palm_target(side, cmd, active), close, hand_target(side, close))
        self._palm_cmd, self._close = new_cmd, new_close
        self._step += 1
        return out

    def prev_action_obs(self, raw_action: np.ndarray) -> np.ndarray:
        """prev_actions obs slot: clamped raw action with palm slots overwritten by the EMA command."""
        pa = np.clip(np.asarray(raw_action, float).reshape(-1), -self.c.action_clip, self.c.action_clip).copy()
        for i, role in enumerate(self.c.roles):
            base = i * (PALM_DIM + HAND_DIM)
            pa[base:base + PALM_DIM] = self._palm_cmd[role]
        return pa

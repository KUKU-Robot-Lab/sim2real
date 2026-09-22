"""Offline bimanual pour chain: measurement -> obs(223) -> policy -> decoder -> (fabric pair) joint targets.

Pure python/numpy; ROS nodes and tests drive the same object. Per step (= hdgp pour_fabric_env order):
  1. obs is built from the CURRENT measurement, the decoder's previous hand target and previous action slot
  2. the policy maps obs -> raw action
  3. grasp latch / close gate are computed from the same (pre-step) measurement
  4. decoder -> palm/hand targets; the optional fabric pair -> joint targets
Nothing is mutated in place; every step returns a new frozen `PourStep`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from .pour_contract import PourContract
from .pour_decoder import PourDecoder, SideInputs, close_gate, grasped, hand_target
from .pour_obs import SideState, build_obs, clip_obs, resolve_fill_level


class PourChainError(RuntimeError):
    """The chain was wired or driven inconsistently with the contract."""


class PolicyLike(Protocol):
    def forward(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class PourMeasure:
    joint_pos: Mapping                 # joint name -> rad (both arms + both hands)
    joint_vel: Mapping                 # joint name -> rad/s (arm joints are read)
    cups: Mapping                      # role -> CupPose in the robot base (= env-local) frame
    forces: Mapping | None = None      # role -> (f_mid (F,), f_dist (F,)) [N]; None = no contact sensing (zeros)
    fill_level: float | None = None    # how full the SOURCE cup is, 0..1; None -> contract default


@dataclass(frozen=True)
class PourStep:
    obs: np.ndarray
    action: np.ndarray
    active: bool
    gates: Mapping
    grasped: Mapping
    targets: Mapping                   # role -> SideTargets
    joint_targets: Mapping | None      # role -> fabric JointTarget (None without a fabric)


class RecordedPolicy:
    """Teacher-forcing backend: replays a recorded (T, action_dim) action table."""

    def __init__(self, actions: np.ndarray):
        self._a, self._i = np.asarray(actions, float), 0

    def reset(self) -> None:
        self._i = 0

    def forward(self, obs: np.ndarray) -> np.ndarray:
        a = self._a[self._i]
        self._i += 1
        return a.copy()


class PourChain:
    def __init__(self, contract: PourContract, policy: PolicyLike, fks: Mapping, fabric=None):
        missing = [r for r in contract.roles if r not in fks]
        if missing:
            raise PourChainError(f"no FK for role(s) {missing}; both sides are required")
        self.c, self.policy, self.fks, self.fabric = contract, policy, dict(fks), fabric
        self.decoder = PourDecoder(contract)
        self._ready = False

    def reset(self, home: Mapping | None = None) -> None:
        self.decoder.reset()
        self.policy.reset()
        self._hand = {s.role: hand_target(s, np.zeros(len(s.hand_joints))) for s in self.c.sides}
        self._prev = np.zeros(self.c.action_dim)
        self._prev_slot = self.decoder.prev_action_obs(self._prev)
        if self.fabric is not None:
            if home is None:
                raise PourChainError("a fabric needs the measured home joint vector per role at reset")
            self.fabric.reset(home)
        self._ready = True

    def _state(self, side, m: PourMeasure) -> SideState:
        if side.role not in m.cups:
            raise PourChainError(f"measurement has no cup pose for role {side.role}")
        pose = self.fks[side.role].palm_pose([m.joint_pos[n] for n in side.arm_joints],
                                             [m.joint_pos[n] for n in side.hand_joints])
        return SideState(m.joint_pos, m.joint_vel, pose.palm_pos, pose.extra["palm_rot"], pose.tips,
                         m.cups[side.role], self._hand[side.role])

    def _inputs(self, side, st: SideState, m: PourMeasure) -> tuple:
        n = len(side.fingers)
        f_mid, f_dist = (np.zeros(n), np.zeros(n)) if m.forces is None else m.forces[side.role]
        held = grasped(side, np.maximum(np.asarray(f_mid, float), np.asarray(f_dist, float)),
                       self.c.contact_force_threshold)
        gate = close_gate(st.palm_pos, st.cup.pos, self.c.close_gate_radius, self.c.close_gate_ramp,
                          self.c.close_gate_enabled, held)
        return SideInputs(gate, np.asarray(f_mid, float), np.asarray(f_dist, float)), held

    def step(self, m: PourMeasure) -> PourStep:
        if not self._ready:
            raise PourChainError("reset() before step()")
        c = self.c
        states = {s.role: self._state(s, m) for s in c.sides}
        fill = resolve_fill_level(c, m.fill_level)
        obs = clip_obs(c, build_obs(c, states, fill, self._prev_slot))
        action = np.asarray(self.policy.forward(obs.astype(np.float32)), float).reshape(-1)
        pairs = {s.role: self._inputs(s, states[s.role], m) for s in c.sides}
        active = self.decoder.active
        targets = self.decoder.step(action, {r: p[0] for r, p in pairs.items()})
        self._prev_slot = self.decoder.prev_action_obs(action)
        self._hand = {r: t.hand_target for r, t in targets.items()}
        joint = None
        if self.fabric is not None:
            joint = self.fabric.step({r: t.palm_target for r, t in targets.items()},
                                     {r: t.hand_target for r, t in targets.items()}, hold=not active)
        return PourStep(obs, action, active, {r: p[0].close_gate for r, p in pairs.items()},
                        {r: p[1] for r, p in pairs.items()}, targets, joint)

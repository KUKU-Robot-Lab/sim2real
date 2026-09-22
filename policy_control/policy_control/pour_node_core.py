"""ROS-free half of pour_node: latest messages -> PourMeasure, PourStep -> one joint_target for both arms.

The inbox keys everything by canonical joint name (the caller maps driver names/signs first), refuses to
produce a measurement while anything either side needs is missing or stale, and never invents a value.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .pour_chain import PourMeasure, PourStep
from .pour_contract import PourContract
from .pour_obs import CupPose, PourObsError, resolve_fill_level


class PourNodeError(RuntimeError):
    pass


def _vec(values, size: int | None, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if size is not None and arr.size != size:
        raise PourNodeError(f"{what}: expected {size} values, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise PourNodeError(f"{what}: non-finite value")
    return arr


class PourInbox:
    def __init__(self, contract: PourContract, stale_sec: float) -> None:
        if stale_sec <= 0:
            raise PourNodeError("stale_sec must be positive")
        self.c, self.stale_sec = contract, float(stale_sec)
        self._need = tuple(n for s in contract.sides for n in tuple(s.arm_joints) + tuple(s.hand_joints))
        self._joint: dict = {}      # name -> (pos, vel, t)
        self._cup: dict = {}        # role -> (CupPose, t)
        self._force: dict = {}      # role -> (f_mid, f_dist, t)
        self._fill: tuple | None = None
        self._fill_held = False

    # ------------------------------------------------------------ puts
    def put_joints(self, names: Sequence[str], position, velocity, now: float) -> None:
        if velocity is None:
            raise PourNodeError("joint message without velocity (arm_qd is an actor input)")
        pos = _vec(position, len(names), "joint position")
        vel = _vec(velocity, len(names), "joint velocity")
        self._joint = {**self._joint, **{n: (float(p), float(v), float(now)) for n, p, v in zip(names, pos, vel)}}

    def put_cup(self, role: str, pos, quat_wxyz, now: float) -> None:
        if role not in self.c.roles:
            raise PourNodeError(f"unknown role {role!r}; have {self.c.roles}")
        cup = CupPose(_vec(pos, 3, f"cup:{role} pos"), _vec(quat_wxyz, 4, f"cup:{role} quat"))
        self._cup = {**self._cup, role: (cup, float(now))}

    def put_forces(self, role: str, f_mid, f_dist, now: float) -> None:
        if role not in self.c.roles:
            raise PourNodeError(f"unknown role {role!r}; have {self.c.roles}")
        n = len(self.c.side(role).fingers)
        self._force = {**self._force, role: (_vec(f_mid, n, "f_mid"), _vec(f_dist, n, "f_dist"), float(now))}

    def hold_fill(self, held: bool) -> None:
        """While an episode runs the value is frozen (training keeps it constant after the hold)."""
        self._fill_held = bool(held)

    def put_fill(self, value: float, now: float) -> None:
        if self._fill_held:
            raise PourNodeError("fill_level is held for the running episode; publish it before start")
        v = float(value)
        if not np.isfinite(v) or not 0.0 <= v <= 1.0:
            raise PourNodeError(f"fill_level {value!r} outside [0, 1]")
        self._fill = (v, float(now))

    # ------------------------------------------------------------ measure
    def _problems(self, now: float) -> tuple[list, list]:
        missing = [n for n in self._need if n not in self._joint]
        missing += [f"cup:{r}" for r in self.c.roles if r not in self._cup]
        old = lambda t: float(now) - t > self.stale_sec  # noqa: E731
        stale = [n for n in self._need if n in self._joint and old(self._joint[n][2])]
        stale += [f"cup:{r}" for r in self.c.roles if r in self._cup and old(self._cup[r][1])]
        stale += [f"force:{r}" for r, v in self._force.items() if old(v[2])]
        # fill_level is latched: training measures it once and holds it for the episode, so an
        # operator value published once stays valid (it is never a live sensor stream here)
        return missing, stale

    def inputs(self, now: float) -> list[dict]:
        """Every input with its liveness, for the status message (the operator console shows it).

        `measure` only names what blocks a tick; this lists the healthy ones too, so a screen can
        tell "all inputs live" from "nobody is looking". A group is as old as its oldest member.
        """
        def group(name: str, stamps: Sequence, optional: bool = False) -> dict:
            if any(t is None for t in stamps):
                off = optional and all(t is None for t in stamps)
                return {"name": name, "state": "off" if off else "missing", "age_ms": None}
            age = float(now) - min(stamps)
            return {"name": name, "state": "stale" if age > self.stale_sec else "live", "age_ms": age * 1e3}

        t_joint = lambda n: self._joint[n][2] if n in self._joint else None  # noqa: E731
        rows = []
        for s in self.c.sides:
            rows.append(group(f"{s.role}:arm", [t_joint(n) for n in s.arm_joints]))
            rows.append(group(f"{s.role}:hand", [t_joint(n) for n in s.hand_joints]))
            rows.append(group(f"{s.role}:cup", [self._cup[s.role][1] if s.role in self._cup else None]))
            force = group(f"{s.role}:force", [self._force[s.role][2] if s.role in self._force else None])
            # forces are optional as a whole, but once one role reports the other must too (see measure)
            rows.append({**force, "state": "off"} if not self._force else force)
        fill = {"name": "fill", "state": "off", "age_ms": None}
        if self._fill is not None:                   # latched value: old is fine, so never "stale"
            fill = {"name": "fill", "state": "held", "age_ms": (float(now) - self._fill[1]) * 1e3}
        return rows + [fill]

    def measure(self, now: float) -> PourMeasure:
        missing, stale = self._problems(now)
        if missing or stale:
            raise PourNodeError(f"inputs missing {missing} stale {stale}")
        forces = None
        if self._force:
            absent = [r for r in self.c.roles if r not in self._force]
            if absent:
                raise PourNodeError(f"contact forces arrived for some roles only; missing {absent}")
            forces = {r: (v[0].copy(), v[1].copy()) for r, v in self._force.items()}
        return PourMeasure(
            joint_pos={n: self._joint[n][0] for n in self._need},
            joint_vel={n: self._joint[n][1] for n in self._need},
            cups={r: v[0] for r, v in self._cup.items()},
            forces=forces,
            fill_level=None if self._fill is None else self._fill[0],
        )


def fabric_home(contract: PourContract, m: PourMeasure) -> dict:
    """Measured posture per role in that side's fabric joint order (PourFabricPair.reset input)."""
    return {s.role: np.array([m.joint_pos[n] for n in s.fabric_joint_order], dtype=np.float64)
            for s in contract.sides}


def joint_target_arrays(contract: PourContract, step: PourStep) -> tuple[tuple, np.ndarray, np.ndarray]:
    """One joint_target for pd_node: per side arm (fabric q/qd) then hand (decoder target, qd 0), as the env
    sends ``fabric_q[:, :n_arm]`` to the arm and the synergy target to the hand."""
    if step.joint_targets is None:
        raise PourNodeError("no fabric output: joint_target needs the fabric step")
    names, q, qd = [], [], []
    for s in contract.sides:
        jt, hand = step.joint_targets[s.role], np.asarray(step.targets[s.role].hand_target, dtype=np.float64)
        names += list(s.arm_joints) + list(s.hand_joints)
        q += [np.asarray(jt.q_arm, dtype=np.float64), hand]
        qd += [np.asarray(jt.qd_arm, dtype=np.float64), np.zeros(hand.size)]
    q, qd = np.concatenate(q), np.concatenate(qd)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
        raise PourNodeError("non-finite joint target")
    return tuple(names), q, qd


def tip_forces_to_inputs(side, tip_names: Sequence[str], xyz) -> tuple[np.ndarray, np.ndarray]:
    """Fingertip force vectors (driver order) -> (f_mid, f_dist) in contract finger order.

    The real hand has fingertip sensors only: f_dist = |F_tip|, f_mid = 0 (no middle-phalanx sensor).
    """
    arr = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if arr.shape[0] != len(tip_names):
        raise PourNodeError(f"tip force: {arr.shape[0]} vectors for {len(tip_names)} names")
    if not np.all(np.isfinite(arr)):
        raise PourNodeError("tip force: non-finite")
    out = []
    for finger in side.fingers:
        hits = [i for i, n in enumerate(tip_names) if finger in n]
        if len(hits) != 1:
            raise PourNodeError(f"tip force: finger {finger!r} matches {len(hits)} of {list(tip_names)}")
        out.append(float(np.linalg.norm(arr[hits[0]])))
    f_dist = np.asarray(out, dtype=float)
    return np.zeros_like(f_dist), f_dist


def reset_pose_error(contract: PourContract, m: PourMeasure) -> float:
    """Max |arm joint - training reset pose| over both arms [rad]; the episode must start near it."""
    worst = 0.0
    for s in contract.sides:
        q = np.array([m.joint_pos[n] for n in s.arm_joints], dtype=float)
        worst = max(worst, float(np.abs(q - np.asarray(s.arm_reset, dtype=float)).max()))
    return worst


def start_refusals(contract: PourContract, m: PourMeasure, tol: float) -> list:
    """Why `start` must be refused (empty = go): arms away from the training reset pose, fill_level unresolved."""
    reasons = []
    err = reset_pose_error(contract, m)
    if err > tol:
        reasons.append(f"arm is {err:.3f} rad from the training reset pose (tol {tol})")
    try:
        resolve_fill_level(contract, m.fill_level)
    except PourObsError as exc:
        reasons.append(f"fill_level: {exc}")
    return reasons

"""Adapter between the existing PD command and opt-in stationary torque compensation."""

from collections import deque
from dataclasses import replace

import numpy as np

from .stationary_compensation import (
    FS_JOINTS,
    CompensationConfig,
    StationaryCompensator,
    payload_regressor,
)


class StationaryRuntime:
    def __init__(self, config, *, kp, kd, torque_limit, blend_sec, gravity_chain=None):
        self.cfg = CompensationConfig.from_mapping(config, n=len(kp))
        self.cfg = replace(self.cfg, blend_sec=blend_sec)
        self.compensator = StationaryCompensator(self.cfg)
        self.kp, self.kd = np.asarray(kp), np.asarray(kd)
        self.torque_limit = np.asarray(torque_limit)
        self.chain = gravity_chain
        if self.cfg.gravity_enabled and self.chain is None:
            raise ValueError("gravity correction requires the configured gravity chain")
        self.target = None
        self.episode = None
        self.settle_joints = min(len(kp), FS_JOINTS)
        self._window = deque()  # (elapsed [s], q[:settle_joints]) since last motion
        self._elapsed = 0.0

    def reset(self):
        """Called only before a new engagement, after the previous release blend."""
        self.compensator = StationaryCompensator(self.cfg)
        self.target = None
        self.episode = None
        self._window.clear()
        self._elapsed = 0.0

    def _standstill(self, q_meas, dt, hold):
        """Position-span standstill on j1~j4 over settle_window_sec (see config)."""
        if not hold:
            self._window.clear()
            return False
        self._elapsed += dt
        self._window.append((self._elapsed, np.asarray(q_meas[: self.settle_joints])))
        window = self.cfg.settle_window_sec
        # Keep one sample at/behind the window edge so tick jitter cannot open a gap.
        while len(self._window) > 1 and self._elapsed - self._window[1][0] >= window:
            self._window.popleft()
        if self._elapsed - self._window[0][0] < window:
            return False  # not enough history yet
        q = np.stack([s[1] for s in self._window])
        span = np.max(q.max(axis=0) - q.min(axis=0))
        return bool(span <= np.deg2rad(self.cfg.settle_band_deg))

    def apply(
        self,
        cmd,
        *,
        q_meas,
        qd_meas,
        target,
        target_velocity,
        phase,
        contact,
        stale,
        episode,
        dt,
        gravity,
        request,
        output_scale=1.0,
    ):
        changed = self.target is not None and not np.allclose(
            target, self.target, rtol=0.0, atol=1e-10
        )
        new_episode = self.episode is not None and episode != self.episode
        self.target, self.episode = np.asarray(target).copy(), episode
        block = []
        if changed or new_episode:
            block.append("target_changed")
        if np.any(np.abs(target_velocity) > 1e-8) or np.any(np.abs(cmd.q - target) > 1e-6):
            block.append("target_moving")
        if bool(cmd.limited):
            block.append("limited")
        if phase != "TRACKING":
            block.append("phase")
        if output_scale != 1.0:
            block.append("output_scale")
        standstill = self._standstill(q_meas, dt, not block)
        if not block and not standstill:
            block.append("not_standstill")
        settled = not block
        if contact is not False:
            block.append("contact")
        if stale:
            block.append("stale")
        pd = self.kp * (cmd.q - q_meas) + self.kd * (cmd.qd - qd_meas)
        delta_g = (
            payload_regressor(self.chain, q_meas) @ self.cfg.payload_parameters
            if self.cfg.gravity_enabled
            else None
        )
        result = self.compensator.step(
            cmd.q - q_meas,
            qd_meas,
            pd + cmd.tau,
            self.torque_limit,
            dt,
            settled=settled,
            contact=contact is not False,
            stale=stale,
            stopped=phase != "TRACKING",
            target_changed=changed,
            episode_changed=new_episode,
            gravity_correction=delta_g,
            feedforward=cmd.tau,
        )
        scale = float(np.clip(output_scale, 0.0, 1.0))
        addition = result.addition * scale
        out = replace(cmd, tau=cmd.tau + addition) if self.cfg.enabled else cmd
        diag = {
            "user_target": np.asarray(target).tolist(),
            "motor_position": out.q.tolist(),
            "motor_velocity": out.qd.tolist(),
            "measured": np.asarray(q_meas).tolist(),
            "velocity": np.asarray(qd_meas).tolist(),
            "torque_pd": pd.tolist(),
            "torque_gravity": np.asarray(gravity).tolist(),
            "torque_request": np.asarray(request).tolist(),
            "torque_gravity_correction": (result.gravity * scale).tolist(),
            "torque_friction": (result.friction * scale).tolist(),
            "torque_integral": (result.integral * scale).tolist(),
            "torque_static_friction": (result.static * scale).tolist(),
            "torque_feedforward": out.tau.tolist(),
            "torque_estimated_total": (pd + out.tau).tolist(),
            "compensation_saturated": result.saturated.tolist(),
            "integral_updating": result.updating.tolist(),
            "settled": settled,
            "fs_armed": bool(self.compensator.fs_armed),
            "settle_block": block,
            "target_changed": bool(changed),
            "contact": contact,
            "stale": bool(stale),
            "protected": phase not in ("TRACKING", "RAMPING"),
            "phase": phase,
            "episode": episode,
            "limited": list(cmd.limited),
        }
        return out, diag

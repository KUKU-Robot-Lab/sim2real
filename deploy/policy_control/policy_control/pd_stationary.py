"""Adapter between the existing PD command and opt-in stationary torque compensation."""

from dataclasses import replace

import numpy as np

from .stationary_compensation import (
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

    def reset(self):
        """Called only before a new engagement, after the previous release blend."""
        self.compensator = StationaryCompensator(self.cfg)
        self.target = None
        self.episode = None

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
        moving = (
            changed
            or np.any(np.abs(target_velocity) > 1e-8)
            or np.any(np.abs(cmd.q - target) > 1e-6)
            or bool(cmd.limited)
        )
        settled = (
            not moving
            and phase == "TRACKING"
            and np.max(np.abs(qd_meas)) < 0.005
            and output_scale == 1.0
        )
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
            "torque_feedforward": out.tau.tolist(),
            "torque_estimated_total": (pd + out.tau).tolist(),
            "compensation_saturated": result.saturated.tolist(),
            "integral_updating": result.updating.tolist(),
            "target_changed": bool(changed),
            "contact": contact,
            "stale": bool(stale),
            "protected": phase not in ("TRACKING", "RAMPING"),
            "phase": phase,
            "episode": episode,
            "limited": list(cmd.limited),
        }
        return out, diag

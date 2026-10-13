"""rh_place 10.13 IK 이송(carry) — hdgp hdgp_place_r1 c3009166 rh_place_env._carry_actions 와 같은 식(배포 노이즈 0).

리셋(쥔 컵 인계) 뒤 정책보다 먼저, 팔을 손바닥 DLS 로 목표 홀더 슬롯 위까지 옮긴다:
  · 컵 목표 = (홀더 xy, 홀더 원점 z − HOLDER_BASE_DZ(상판) + 물체 원점 높이 + carry_dz). sim 은 carry_noise 를 더한다(배포 0).
  · dp = 손바닥 − 지각 컵(이송 첫 틱, 세계 틀로 고정) → 손바닥 목표 = 컵 목표 + dp.
  · v = clamp(gain · (손바닥 목표 − 손바닥), ±speed) m/틱 · ω = clamp(rot_gain · 축(up × ez) · atan2(|up × ez|, up_z), ±rot_max)
    (지각 컵 축을 세계 z 로 세우는 기울기 보정만, yaw 자유).
  · dq = Jᵀ(JJᵀ + λ²I)⁻¹ [v; ω] (손바닥 6D 자코비안 · 팔 7 관절, 축마다 ±1) → 팔 행동 = clamp(dq / (k_arm · arm_ema), ±1).
  · 손 행동 = 리셋 손 목표를 행동 단위로(2(h − lo)/(hi − lo) − 1) 내내 유지.
  · 끝 = 이송 틱 수 ≥ min_steps 이고 |지각 컵 − 목표| < tol, 또는 max_steps. 그 틱까지 이송 행동, 다음 틱부터 정책.
정책 forward 는 이송 중에도 매 틱 부르고 출력만 버린다(LSTM · prev_action = 실행한 이송 행동, sim 과 같음).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

#: hdgp tasks/rh_place_r/place_geom.py:11 — 상판 → 홀더 원점
HOLDER_BASE_DZ = 0.030
#: env.yaml 키 → 계약 carry 키(이 키가 모두 덤프에 있어야 한다 — 상수로 숨은 값 금지)
ENV_KEYS = {"dz": "carry_dz", "tol": "carry_tol", "min_steps": "carry_min_steps", "max_steps": "carry_max_steps",
            "speed": "carry_speed", "gain": "carry_gain", "rot_gain": "carry_rot_gain", "rot_max": "carry_rot_max",
            "damp": "carry_dls_damp"}
JAC_EPS = 1e-6


class CarryError(ValueError):
    pass


@dataclass(frozen=True)
class CarryCfg:
    dz: float
    tol: float
    min_steps: int
    max_steps: int
    speed: float
    gain: float
    rot_gain: float
    rot_max: float
    damp: float

    @classmethod
    def from_env(cls, env: Mapping) -> "CarryCfg | None":
        if not bool(env.get("carry_enable", False)):
            return None
        missing = [v for v in ENV_KEYS.values() if v not in env]
        if missing:
            raise CarryError(f"carry_enable 인데 env.yaml 에 {missing} 가 없다 — sim 상수를 배포가 추측하지 않는다")
        return cls.from_dict({k: env[v] for k, v in ENV_KEYS.items()})

    @classmethod
    def from_dict(cls, d: Mapping) -> "CarryCfg":
        if set(d) != set(ENV_KEYS):
            raise CarryError(f"carry 키 {sorted(d)} ≠ {sorted(ENV_KEYS)}")
        c = cls(dz=float(d["dz"]), tol=float(d["tol"]), min_steps=int(d["min_steps"]), max_steps=int(d["max_steps"]),
                speed=float(d["speed"]), gain=float(d["gain"]), rot_gain=float(d["rot_gain"]),
                rot_max=float(d["rot_max"]), damp=float(d["damp"]))
        if not (c.tol > 0 and 0 <= c.min_steps <= c.max_steps and c.max_steps > 0 and c.speed > 0 and c.gain > 0
                and c.rot_gain >= 0 and c.rot_max >= 0 and c.damp > 0):
            raise CarryError(f"carry 값이 범위 밖이다: {c}")
        return c

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ENV_KEYS}


def carry_target(holder_pos: Sequence[float], origin_z: float, dz: float) -> np.ndarray:
    """sim: 홀더 xy · 상판(= 홀더 원점 − HOLDER_BASE_DZ) + 물체 원점 높이 + carry_dz."""
    h = np.asarray(holder_pos, float)
    return np.array([h[0], h[1], h[2] - HOLDER_BASE_DZ + float(origin_z) + float(dz)])


def _rotvec(R: np.ndarray) -> np.ndarray:
    """회전 행렬 → 회전 벡터(작은 각 수치 자코비안용)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * w if th < 1e-9 else th / (2.0 * np.sin(th)) * w


def palm_jacobian(pose_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], arm_q: Sequence[float],
                  eps: float = JAC_EPS) -> np.ndarray:
    """(6, n) 손바닥 기하 자코비안(위치 · 각속도, 기준 틀) — pose_fn(arm_q) → (pos, R) 의 중심 차분."""
    q = np.asarray(arm_q, float)
    J = np.zeros((6, q.size))
    for i in range(q.size):
        d = np.zeros_like(q)
        d[i] = eps
        p1, R1 = pose_fn(q + d)
        p0, R0 = pose_fn(q - d)
        J[:3, i] = (np.asarray(p1) - np.asarray(p0)) / (2.0 * eps)
        J[3:, i] = _rotvec(np.asarray(R1) @ np.asarray(R0).T) / (2.0 * eps)
    return J


def dls_step(J: np.ndarray, err: np.ndarray, lam: float, max_dq: float = 1.0) -> np.ndarray:
    """hdgp rh_aglt_r/invert_probe.dls_step: dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e, 축마다 ±max_dq."""
    JJt = J @ J.T
    x = np.linalg.solve(JJt + (lam ** 2) * np.eye(JJt.shape[0]), np.asarray(err, float))
    return np.clip(J.T @ x, -max_dq, max_dq)


def tilt_twist(cup_quat: np.ndarray, quat_apply: Callable, rot_gain: float, rot_max: float) -> np.ndarray:
    """지각 컵 축을 세계 z 로 세우는 각속도(yaw 자유) — sim 과 같은 식."""
    ez = np.array([0.0, 0.0, 1.0])
    up = np.asarray(quat_apply(cup_quat, ez), float)
    ax = np.cross(up, ez)
    sn = float(np.linalg.norm(ax))
    w = rot_gain * ax / max(sn, 1e-6) * np.arctan2(sn, up[2])
    return np.clip(w, -rot_max, rot_max)


class Carry:
    """한 에피소드의 이송 상태. step 은 (팔 행동, 이번 틱이 이송 틱이었나) — 끝난 뒤에는 (None, False)."""

    def __init__(self, cfg: CarryCfg, target: np.ndarray, hand_action: np.ndarray, k_arm: float, arm_ema: float) -> None:
        self.cfg, self.target = cfg, np.asarray(target, float)
        self.hand_action = np.asarray(hand_action, float)
        self.scale = float(k_arm) * float(arm_ema)
        self.t, self.on, self.arrived = 0, True, False
        self.dp: np.ndarray | None = None
        self.err = float("nan")

    def step(self, palm_pos: np.ndarray, cup_pos: np.ndarray, cup_quat: np.ndarray, J: np.ndarray,
             quat_apply: Callable) -> np.ndarray:
        c = self.cfg
        palm, cp = np.asarray(palm_pos, float), np.asarray(cup_pos, float)
        if self.t == 0:
            self.dp = palm - cp
        v6 = np.zeros(6)
        v6[:3] = np.clip(c.gain * (self.target + self.dp - palm), -c.speed, c.speed)
        v6[3:] = tilt_twist(cup_quat, quat_apply, c.rot_gain, c.rot_max)
        dq = dls_step(J, v6, c.damp, 1.0)
        a_arm = np.clip(dq / self.scale, -1.0, 1.0)
        self.t += 1
        self.err = float(np.linalg.norm(cp - self.target))
        self.arrived = self.t >= c.min_steps and self.err < c.tol
        if self.arrived or self.t >= c.max_steps:
            self.on = False
        return np.concatenate([a_arm, self.hand_action])

    def as_dict(self) -> dict:
        return {"on": self.on, "t": self.t, "arrived": self.arrived, "err": None if self.err != self.err else round(self.err, 4),
                "target": [round(float(v), 4) for v in self.target]}

"""tilt 구간 전용 보정(11.16): 쥔 물체 payload 중력 ff + TRACKING 적분. pd yaml `tilt_comp` 블록, 기본 꺼짐.

켜지는 조건: pd yaml 에 그 팔 블록이 있고, replay_to_pd --tilt-comp 가 낸 episode 사건 `tilt_comp_on` 을 받은 뒤
TRACKING 일 때만. stop · abort · reset 사건에서 적분을 0 으로 버리고 꺼진다(구간 끝 리셋).

    tilt_comp:
      left:                                   # 팔 이름(left/right). 없는 팔은 꺼짐
        payload: [0.10, 0.0174, 0.0167, 0.1945]   # (kg, x, y, z) 체인 마지막 링크(*_al_7) 프레임 — pd_gravity payload 와 같은 규약
        ki: [0, 0, 0, 3.0, 0, 0, 0]           # [N·m/(rad·s)] 0 = 그 관절 적분 끔. 기본 예: j4 만
        limit: [0, 0, 0, 0.6, 0, 0, 0]        # [N·m] 적분 windup 한계(관절별)
        deadband_deg: 0.15                    # |오차| 가 이보다 작으면 적분을 멈춘다(떨림 방지)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .stationary_compensation import payload_regressor

EVENT_ON = "tilt_comp_on"
PAYLOAD_KG_MAX = 0.5        # 실행 중 바꾸기(set_payload_kg) 상한 — 손에 쥘 물체 범위(셰이커 0.117 kg)
_KEYS = {"payload", "ki", "limit", "deadband_deg"}


@dataclass(frozen=True)
class TiltCompCfg:
    payload: tuple = (0.0, 0.0, 0.0, 0.0)     # (kg, x, y, z)
    ki: tuple = (0.0,) * 7
    limit: tuple = (0.0,) * 7
    deadband_deg: float = 0.15

    @classmethod
    def from_mapping(cls, raw, n: int = 7) -> "TiltCompCfg":
        unknown = set(raw) - _KEYS
        if unknown:
            raise ValueError(f"tilt_comp: unknown keys {sorted(unknown)}")
        cfg = cls(payload=tuple(float(v) for v in raw.get("payload", cls.payload)),
                  ki=tuple(float(v) for v in raw.get("ki", (0.0,) * n)),
                  limit=tuple(float(v) for v in raw.get("limit", (0.0,) * n)),
                  deadband_deg=float(raw.get("deadband_deg", cls.deadband_deg)))
        if len(cfg.payload) != 4 or not 0.0 <= cfg.payload[0] <= 1.0:
            raise ValueError(f"tilt_comp.payload must be (kg ≤ 1, x, y, z), got {cfg.payload}")
        if len(cfg.ki) != n or len(cfg.limit) != n:
            raise ValueError(f"tilt_comp ki/limit need {n} values")
        if min(cfg.ki) < 0 or min(cfg.limit) < 0 or cfg.deadband_deg < 0:
            raise ValueError("tilt_comp ki/limit/deadband_deg must be >= 0")
        if not all(np.isfinite([*cfg.payload, *cfg.ki, *cfg.limit, cfg.deadband_deg])):
            raise ValueError("tilt_comp values must be finite")
        return cfg


class TiltComp:
    """τ_add = payload_regressor(q) @ [m, m·c] + ∫ki·e dt (관절별 ±limit, deadband 안에서는 멈춤)."""

    def __init__(self, cfg: TiltCompCfg, chain):
        if chain is None and cfg.payload[0] > 0.0:
            raise ValueError("tilt_comp payload needs the model gravity chain (gravity.mode model_tau_ff)")
        self.cfg, self.chain = cfg, chain
        self._set_theta()
        self.integral = np.zeros(len(cfg.ki))
        self.active = False

    def _set_theta(self) -> None:
        m, x, y, z = self.cfg.payload
        self._theta = np.array([m, m * x, m * y, m * z])

    def set_payload_kg(self, kg: float) -> None:
        """★11.17 실행 중 payload kg 만 바꾼다(무게중심 · ki · limit 그대로). 켜진 구간(tilt_comp_on 뒤) 중에는 거부."""
        kg = float(kg)
        if self.active:
            raise ValueError("tilt_comp 켜진 구간이다 — payload 를 바꾸지 않는다")
        if not 0.0 <= kg <= PAYLOAD_KG_MAX:            # NaN 도 여기서 걸린다
            raise ValueError(f"payload {kg} kg — 0~{PAYLOAD_KG_MAX} kg 만")
        if self.chain is None and kg > 0.0:
            raise ValueError("payload 는 model 중력 체인이 있어야 한다(gravity.mode model_tau_ff)")
        self.cfg = replace(self.cfg, payload=(kg, *self.cfg.payload[1:]))
        self._set_theta()

    def reset(self) -> None:
        self.integral[:] = 0.0
        self.active = False

    def step(self, q_meas, q_target, dt: float) -> np.ndarray:
        q_meas = np.asarray(q_meas, float)
        e = np.asarray(q_target, float) - q_meas
        live = np.abs(e) > np.deg2rad(self.cfg.deadband_deg)
        lim = np.asarray(self.cfg.limit)
        self.integral = np.clip(self.integral + np.where(live, np.asarray(self.cfg.ki) * e * dt, 0.0), -lim, lim)
        grav = payload_regressor(self.chain, q_meas) @ self._theta if self._theta[0] > 0.0 else 0.0
        return grav + self.integral

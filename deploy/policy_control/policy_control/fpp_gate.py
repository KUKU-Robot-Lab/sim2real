"""10.13 쥐기 전 FP++ live 거르기(A안) — hdgp rh_aglt_r/fpp_obs.FppGate 의 배포판(컵 하나).

실기 bag 20261013_121158 · 121432: 손바닥이 컵 10~17 cm 안에 오면 FP++ live 가 프레임당 3~9 cm 씩 튀어 수평 14~15 cm ·
z −6 cm(테이블 속)까지 갔다가 1~2 s 뒤 돌아왔다. 정책은 그 자리를 쫓아 팔을 크게 움직이고 컵을 밀었다.
sim(vision-3090, rg3s ep4700, 배포와 같은 설정)에서 같은 튐을 넣으면 진짜 성공 0.89 → 0.25, 이 거르기를 넣으면 0.83.

규칙(sim 과 같은 식): 쥐기 전 · 붙기 전, 마지막으로 받은 컵이 손바닥 radius_m 안이면 새 프레임이
  마지막으로 받은 값에서 jump_m 넘게 떨어졌거나 · z 가 에피소드 첫 프레임보다 dz_m 넘게 낮으면 버리고 마지막 값을 쓴다.
  마지막으로 받은 뒤 max_hold_s 가 지나면 다음 프레임은 받는다(진짜로 밀렸을 수 있다 — 학습 가림 DR 상한 2.5 s).
마지막 값 유지는 학습 FP++ 가림 DR(손 10 cm 안에서 0.5~2.5 s 마지막 값 고정)과 같은 모양이라 정책이 본 분포 안이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateCfg:
    jump_m: float = 0.0          # > 0 이면 켬
    dz_m: float = 0.02
    radius_m: float = 0.20
    max_hold_s: float = 2.5

    def __post_init__(self) -> None:
        vals = (self.jump_m, self.dz_m, self.radius_m, self.max_hold_s)
        if not all(np.isfinite(v) and v >= 0.0 for v in vals):
            raise ValueError(f"GateCfg 값은 유한한 ≥ 0: {vals}")
        if self.jump_m > 0.0 and min(self.dz_m, self.radius_m, self.max_hold_s) <= 0.0:
            raise ValueError(f"GateCfg: 켜면(jump_m > 0) dz_m · radius_m · max_hold_s > 0: {vals}")

    @property
    def enabled(self) -> bool:
        return self.jump_m > 0.0


class FppGate:
    """컵 하나의 live 거르기. `step(live, t, palm_pos, active)` 를 달리는 정책 틱마다, `peek()` 는 판정 없이 마지막 값."""

    def __init__(self, cfg: GateCfg) -> None:
        self.cfg = cfg
        self.new_episode()

    def new_episode(self) -> None:
        self.last = None             # 마지막으로 받은 (pos, quat, stamp)
        self.z0: float | None = None
        self.t_acc: float | None = None
        self.seen_stamp = None
        self.rejects = 0
        # ★10.14 실기 204216: 마지막 새 프레임을 버려 옛 값을 붙들고 있는 중 — 노드는 이 동안 성공을 세지 않는다
        self.holding = False

    def _accept(self, live, t: float):
        self.holding = False
        self.last, self.t_acc = live, float(t)
        if self.z0 is None:
            self.z0 = float(np.asarray(live[0], float)[2])
        return live

    def step(self, live, t: float, palm_pos, active: bool):
        """live = (pos, quat, stamp) 또는 None. 반환 = 정책이 볼 live(거른 값)."""
        if live is None:
            return self.last
        stamp = live[2]
        if stamp == self.seen_stamp:              # 새 프레임 아님 — 판정은 프레임마다 한 번(sim new_frame)
            return self.last if self.last is not None else live
        self.seen_stamp = stamp
        if self.last is None or not self.cfg.enabled:
            return self._accept(live, t)
        pos = np.asarray(live[0], float)
        last = np.asarray(self.last[0], float)
        near = float(np.linalg.norm(last - np.asarray(palm_pos, float))) < self.cfg.radius_m
        z0 = float(pos[2]) if self.z0 is None else self.z0
        odd = float(np.linalg.norm(pos - last)) > self.cfg.jump_m or pos[2] < z0 - self.cfg.dz_m
        young = self.t_acc is not None and float(t) - self.t_acc < self.cfg.max_hold_s
        if active and near and odd and young:
            self.rejects += 1
            self.holding = True
            return self.last
        return self._accept(live, t)

    def peek(self, live):
        return self.last if self.last is not None else live

    def as_dict(self) -> dict:
        return {"rejects": int(self.rejects), "held_s": None if self.t_acc is None else round(float(self.t_acc), 3),
                "holding": bool(self.holding)}

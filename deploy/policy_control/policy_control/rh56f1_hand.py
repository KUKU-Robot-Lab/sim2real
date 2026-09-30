"""RH56F1 손 행동 법칙 — 정책 행동 6 → 손 관절 목표 q*(rad). 배포의 모든 RH56F1 정책이 이 한 곳을 쓴다. 순수(ROS 없음).

09.30 사용자: "rh56f1 의 핸드 액션 구조가 명확해야 — 개발한 정책들과 튜닝되고, 센서도 활성, deploy 에 연결".
학습 쪽 기준(같은 식을 두 과제가 쓴다 — hdgp 09.29 사용자 "손 액션은 pour_bi_rh 와 동일하게"):
  rh_aglt   tasks/rh_aglt_r/hand_action.py(linear_grip_raw · step_hand_target) · rh_aglt_env.py _pre_physics_step
  pour_fj   tasks/pour_fabric_mimic/side_rig.py direct_hand_targets(hand_direct_range · hand_direct_contact_freeze)

  관절(행동 슬롯 순 = 프로필 순): thumb_1(외전) · thumb_2(엄지 굽힘) · index_1 · middle_1 · ring_1 · pinky_1
    종속 6(thumb_3 · thumb_4 · *_2)은 PhysX mimic / 실기는 손 기구가 따라온다 — 행동에 없다.
  [lo, hi]  grip  : 관절별 [min(open, grip), max(open, grip)]  (open 1.57,0,0,0,0,0 · grip 1.20,0.24,1.08,1.08,0.85,0.85)
            limits: 관절 한계 전 범위                           (pour_fj f00b · f01 만)
  raw  = lo + ½(clip(a,−1,1)+1)(hi−lo)            a = 0 은 구간 가운데. thumb_1 만 a = +1 이 편 쪽(open > grip)
  ema  = α·raw + (1−α)·q*_{t−1}                    α = hand_ema(0.1)
  Δ    = clip(ema − q*_{t−1}, ±(한계 전 범위)/(full_range_s·policy_hz))   벤더 속도 "전 범위 1 s" 모사
         vel_cap_rad_s(관절별, 09.30 pour_fj 실기 맞춤)가 있으면 그 값/policy_hz 로 바꾼다
  lo   = max(lo, finger_open_floor)  네 손가락만(09.30 실기 레지스터 1740 까지만 펴짐), hi = max(hi, lo)
  동결: 그 손가락이 닿았고(촉각 > 임계, 기본 1 N) 닫는 방향이면 Δ = 0 — 펴기는 허용           (freeze = True 인 과제만)
  q*   = clip(q*_{t−1} + Δ, lo, hi)
  대기(hold_steps): hold = "open" → q* = open(rh_aglt) · "follow" → 대기 중에도 행동을 따른다(pour_fj)

실기 차이(배포에서 대신하는 것):
  · 학습의 동결 신호는 **컵만** 거른 첫마디 · 손끝 접촉력이다. 실기는 손끝 촉각(대상 무관 net)밖에 없다 — 같은 임계로 대신한다.
  · q* 는 rad → 벤더 각도 레지스터(0.1°) 변환(rh56f1_map) → pd rh56f1_angle 백엔드가 보낸다. 손 자체 속도 · 힘 한계는
    pd yaml hand_hw(speed · force) 로 드라이버에 준다 — 정책의 q* 속도 상한보다 느리면 안 된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

JOINTS = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_OF = (0, 0, 1, 2, 3, 4)                  # JOINTS → FINGERS
RANGES = ("grip", "limits")
HOLDS = ("open", "follow")


class HandActionError(ValueError):
    pass


@dataclass(frozen=True)
class HandLaw:
    """한 손의 행동 법칙 — 계약에 그대로 적는 값들."""

    q_open: tuple                 # JOINTS 순
    q_grip: tuple
    lim_lo: tuple
    lim_hi: tuple
    range_mode: str               # grip | limits
    ema: float
    full_range_s: float
    policy_hz: float
    freeze: bool
    freeze_joints: tuple          # JOINTS 순 bool — 동결 대상(프로필 hand_freeze_suffixes). 기본 전부
    freeze_threshold_n: float
    hold: str                     # open | follow
    finger_open_floor: float = 0.0    # 네 손가락 목표 하한(rad). 0 = 없음 — hdgp hand_finger_open_floor_rad
    vel_cap_rad_s: tuple = ()         # JOINTS 순 속도 상한(rad/s). 빈 값 = 전 범위/full_range_s — hdgp hand_vel_cap_rad_s

    def __post_init__(self) -> None:
        for name in ("q_open", "q_grip", "lim_lo", "lim_hi", "freeze_joints"):
            if len(getattr(self, name)) != len(JOINTS):
                raise HandActionError(f"{name} 는 {len(JOINTS)} 개")
        if self.vel_cap_rad_s and (len(self.vel_cap_rad_s) != len(JOINTS) or min(self.vel_cap_rad_s) <= 0.0):
            raise HandActionError(f"vel_cap_rad_s 는 양수 {len(JOINTS)} 개: {self.vel_cap_rad_s}")
        if self.finger_open_floor < 0.0:
            raise HandActionError(f"finger_open_floor {self.finger_open_floor} < 0")
        if self.range_mode not in RANGES or self.hold not in HOLDS:
            raise HandActionError(f"range_mode {self.range_mode!r} · hold {self.hold!r}")
        if not (0.0 < self.ema <= 1.0 and self.full_range_s > 0.0 and self.policy_hz > 0.0):
            raise HandActionError(f"ema {self.ema} · full_range_s {self.full_range_s} · policy_hz {self.policy_hz}")
        lo, hi = np.asarray(self.lim_lo), np.asarray(self.lim_hi)
        for q in (self.q_open, self.q_grip):
            if np.any(np.asarray(q) < lo - 1e-9) or np.any(np.asarray(q) > hi + 1e-9):
                raise HandActionError(f"자세 {q} 가 관절 한계 밖")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        llo, lhi = np.asarray(self.lim_lo, float), np.asarray(self.lim_hi, float)
        if self.range_mode == "limits":
            lo, hi = llo, lhi
        else:
            o, g = np.asarray(self.q_open, float), np.asarray(self.q_grip, float)
            lo, hi = np.clip(np.minimum(o, g), llo, lhi), np.clip(np.maximum(o, g), llo, lhi)
        if self.finger_open_floor > 0.0:
            finger = np.array([not j.startswith("thumb") for j in JOINTS])
            lo = np.where(finger, np.maximum(lo, self.finger_open_floor), lo)
            hi = np.maximum(hi, lo)
        return lo, hi

    def cap(self) -> np.ndarray:
        """스텝당 최대 변화 — 관절 한계 전 범위를 full_range_s 초에(vel_cap_rad_s 가 있으면 그 속도)."""
        if self.vel_cap_rad_s:
            return np.asarray(self.vel_cap_rad_s, float) / self.policy_hz
        return (np.asarray(self.lim_hi, float) - np.asarray(self.lim_lo, float)) / (self.full_range_s * self.policy_hz)

    def raw(self, a: Sequence[float]) -> np.ndarray:
        lo, hi = self.bounds()
        return lo + 0.5 * (np.clip(np.asarray(a, float), -1.0, 1.0) + 1.0) * (hi - lo)

    def touch_to_freeze(self, tactile_n: Sequence[float] | None) -> np.ndarray | None:
        """촉각 5(엄지 → 새끼, N) → 관절별 동결 여부. 촉각이 없거나 동결을 안 쓰는 과제면 None."""
        if not self.freeze or tactile_n is None:
            return None
        t = np.asarray(tactile_n, float).reshape(-1)
        if t.size != len(FINGERS):
            raise HandActionError(f"촉각은 {len(FINGERS)} 개: {t.size}")
        return (t[list(FINGER_OF)] > self.freeze_threshold_n) & np.asarray(self.freeze_joints, bool)

    def step(self, prev: np.ndarray, a: Sequence[float], *, active: bool, freeze: np.ndarray | None = None) -> np.ndarray:
        """q*_{t−1} (6) · 행동 (6) → q*_t. freeze(6 bool)는 touch_to_freeze 의 결과."""
        act = np.asarray(a, float).reshape(-1)
        if act.size != len(JOINTS) or not np.all(np.isfinite(act)):
            raise HandActionError(f"손 행동은 유한한 {len(JOINTS)} 개: {act}")
        if not active and self.hold == "open":
            return np.asarray(self.q_open, float).copy()
        prev = np.asarray(prev, float)
        lo, hi = self.bounds()
        ema = self.ema * self.raw(act) + (1.0 - self.ema) * prev
        cap = self.cap()
        delta = np.clip(ema - prev, -cap, cap)
        if freeze is not None:
            closing = delta * np.sign(np.asarray(self.q_grip, float) - np.asarray(self.q_open, float)) > 0.0
            delta = np.where(np.asarray(freeze, bool) & closing, 0.0, delta)
        return np.clip(prev + delta, lo, hi)


def tactile_obs(force_n: Sequence[float], clip_n: float, tanh_n: float) -> np.ndarray:
    """손끝 촉각 관측(엄지 → 새끼) — 학습과 같게 clip(0, clip_n) → tanh(F / tanh_n). 배포는 잡음을 넣지 않는다."""
    x = np.clip(np.asarray(force_n, float), 0.0, float(clip_n))
    return np.tanh(x / float(tanh_n)) if tanh_n > 0 else x

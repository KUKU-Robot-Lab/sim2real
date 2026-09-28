"""joint family 디코더 — 행동(팔 7 증분 + 손 19 절대) → 팔 · 손 관절 목표. 순수 numpy, 상태는 q* 두 벡터뿐.

hdgp grasp_fj_env.py(학습 커밋 d38b4346) 와 같은 식:
  팔  q_raw = clip(q*_{t-1} + k_arm·a, lo, hi);  q*_t = clip(ema·q_raw + (1−ema)·q*_{t-1}, lo, hi)   `_arm_command`
  손  raw  = lo + ½(clip(a,−1,1)+1)(hi−lo);      q*_t = clip(ema·raw + (1−ema)·q*_{t-1}, lo, hi)     `_hand_targets`
팔은 **직전 목표** 위에 쌓는다(실측이 아니다). 관측의 cmd_state · 손 액션칸은 이 q* 에서 나온다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .joint_contract import JointContract


class JointDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class JointTargets:
    arm: np.ndarray        # (n_arm,) q*  arm_joints 순
    hand: np.ndarray       # (n_hand,) q* hand_joints 순
    action: np.ndarray     # 클립한 행동(관측 action_arm 칸이 쓴다)


class JointDecoder:
    def __init__(self, contract: JointContract) -> None:
        c = contract
        self.c = c
        self._arm_lo, self._arm_hi = np.asarray(c.arm_lo, float), np.asarray(c.arm_hi, float)
        self._hand_lo, self._hand_hi = np.asarray(c.hand_lo, float), np.asarray(c.hand_hi, float)
        self.reset()

    def reset(self) -> None:
        """에피소드 시작 — q* 를 계약의 시작 자세로, 직전 행동을 0 으로(학습 리셋과 같다)."""
        self.arm = np.asarray(self.c.arm_reset, float).copy()
        self.hand = np.asarray(self.c.hand_reset, float).copy()
        self.action = np.zeros(self.c.action_dim)

    def step(self, action: np.ndarray) -> JointTargets:
        a = np.asarray(action, float).reshape(-1)
        if a.size != self.c.action_dim:
            raise JointDecodeError(f"action has {a.size} values, contract wants {self.c.action_dim}")
        if not np.all(np.isfinite(a)):
            raise JointDecodeError("action contains NaN/Inf")
        a = np.clip(a, -self.c.action_clip, self.c.action_clip)
        n = self.c.n_arm
        q_raw = np.clip(self.arm + self.c.k_arm * a[:n], self._arm_lo, self._arm_hi)
        arm = np.clip(self.c.arm_ema * q_raw + (1.0 - self.c.arm_ema) * self.arm, self._arm_lo, self._arm_hi)
        raw = self._hand_lo + 0.5 * (np.clip(a[n:], -1.0, 1.0) + 1.0) * (self._hand_hi - self._hand_lo)
        hand = np.clip(self.c.hand_ema * raw + (1.0 - self.c.hand_ema) * self.hand, self._hand_lo, self._hand_hi)
        self.arm, self.hand, self.action = arm, hand, a
        return JointTargets(arm.copy(), hand.copy(), a.copy())

    def hand_action_obs(self) -> np.ndarray:
        """관측 손 액션칸 = 2(q*−lo)/(hi−lo) − 1 (hdgp `_action_obs`)."""
        return 2.0 * (self.hand - self._hand_lo) / (self._hand_hi - self._hand_lo) - 1.0

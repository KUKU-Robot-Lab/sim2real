"""물체 자세 추정 — FP++ 가 잡는 동안은 FP++, **파지한 뒤에는 손에 붙인다**(joint family). 순수 numpy.

09.28 사용자: "cup 을 파지하고 나서는 fpp 가 제대로 작동 안 할 것으로 봄 — 대비 필요". 손이 컵을 덮으면 FP++ 는
놓치거나 엉뚱한 자세로 흐른다. 그래서
  live      FP++ 자세를 쓴다. 한 번에 jump_m 넘게 튀면 그 한 개는 버리고 직전 값을 쓴다(연속 reject_limit 번이면
            진짜로 옮겨진 것으로 보고 받는다). 쓴 자세마다 **손바닥 기준 상대 자세**를 갱신해 둔다.
  attached  손바닥–물체 원점 거리 < attach_dist_m 이고 손 닫힘 > attach_closure 가 attach_steps 스텝 이어지면
            마지막 상대 자세로 **붙인다**. 이후 물체 = 손바닥 자세 ∘ 상대 자세(에피소드 끝까지).
            붙인 뒤에도 FP++ 가 그 예측에서 refine_m 안이면 상대 자세를 그 값으로 다듬는다('attached_live') —
            일찍 붙이면 손가락이 조이는 동안 컵이 손 안에서 움직인다(trace: 다듬지 않으면 최대 2.3 cm).
            가려진 FP++ 가 흐르면(예측에서 refine_m 밖) 다듬지 않고 마지막 좋은 상대 자세를 지킨다('attached').
기본값은 학습 trace(right_m15_e800, 600 스텝)에서 잰 파지 상태다: 들어 올린 동안 손바닥–컵 원점 12.1~13.6 cm ·
닫힘 ≥ 0.56, 접근 중에는 ≥ 15 cm · 닫힘 < 0.1.
붙인 뒤 컵이 미끄러지거나 떨어지면 이 추정은 모른다 — 운영자가 본다(status 의 source 가 'attached').
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .joint_contract import JointContract
from .joint_obs import Pose, quat_to_matrix


@dataclass(frozen=True)
class AttachCfg:
    attach_dist_m: float = 0.15
    attach_closure: float = 0.5
    attach_steps: int = 5
    jump_m: float = 0.05
    reject_limit: int = 3
    refine_m: float = 0.03


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) → wxyz (w ≥ 0)."""
    m = np.asarray(R, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


class ObjectEstimator:
    def __init__(self, contract: JointContract, cfg: AttachCfg = AttachCfg()) -> None:
        self.c, self.cfg = contract, cfg
        lo, hi = np.asarray(contract.hand_lo, float), np.asarray(contract.hand_hi, float)
        self._lo, self._span = lo, hi - lo
        self._movable = (hi - lo) > 0.05                    # ±0.01 로 묶인 관절은 닫힘에 넣지 않는다
        self.reset()

    def reset(self) -> None:
        self.attached = False
        self._last: Pose | None = None
        self._rel: tuple[np.ndarray, np.ndarray] | None = None   # (p_rel, R_rel) 손바닥 프레임
        self._near = 0
        self._rejects = 0
        self.rejected_total = 0

    def closure(self, hand_qstar) -> float:
        n = (np.asarray(hand_qstar, float) - self._lo) / self._span
        return float(np.clip(n[self._movable], 0.0, 1.0).mean())

    def _accept(self, fpp: Pose | None) -> Pose | None:
        if fpp is None:
            return None
        if self._last is not None and np.linalg.norm(np.asarray(fpp.pos) - self._last.pos) > self.cfg.jump_m:
            self._rejects += 1
            self.rejected_total += 1
            if self._rejects < self.cfg.reject_limit:
                return self._last                           # 한 번 튄 값은 버린다
        self._rejects = 0
        self._last = Pose(np.asarray(fpp.pos, float).copy(), np.asarray(fpp.quat, float).copy())
        return self._last

    def update(self, fpp: Pose | None, palm_pos, palm_rot, hand_qstar) -> tuple[Pose | None, str]:
        """(물체 자세 | None, 출처 'live' · 'held' · 'attached_live' · 'attached' · 'missing')."""
        palm_pos, palm_rot = np.asarray(palm_pos, float), np.asarray(palm_rot, float)
        if self.attached:
            p_rel, R_rel = self._rel
            pred = Pose(palm_pos + palm_rot @ p_rel, matrix_to_quat(palm_rot @ R_rel))
            if fpp is not None and np.linalg.norm(np.asarray(fpp.pos, float) - pred.pos) < self.cfg.refine_m:
                meas = Pose(np.asarray(fpp.pos, float).copy(), np.asarray(fpp.quat, float).copy())
                self._rel = (palm_rot.T @ (meas.pos - palm_pos), palm_rot.T @ quat_to_matrix(meas.quat))
                self._last = meas
                return meas, "attached_live"
            return pred, "attached"
        pose = self._accept(fpp)
        if pose is None:
            return None, "missing"
        held = fpp is None or pose is not None and not np.allclose(pose.pos, np.asarray(fpp.pos, float))
        self._rel = (palm_rot.T @ (pose.pos - palm_pos), palm_rot.T @ quat_to_matrix(pose.quat))
        near = np.linalg.norm(pose.pos - palm_pos) < self.cfg.attach_dist_m
        self._near = self._near + 1 if near and self.closure(hand_qstar) > self.cfg.attach_closure else 0
        if self._near >= self.cfg.attach_steps:
            self.attached = True
        return pose, "held" if held else "live"

"""rh_aglt 배포 목표 — 사용자가 넣은 목표를 학습 목표 분포 안에서만 받고, 먼 목표는 중간 목표로 나눠 차례로 준다. 순수(ROS 없음).

10.01 사용자(T2R RH56F1 [Grasping] 세션 전달): 배포에서 목표 위치를 직접 넣는다. 정책은 목표를 관측(kp_rel_goal · goal_rel_palm)
으로만 받으므로 재학습 없이 된다. hdgp 기준: tasks/rh_aglt_{r,l}/goals.py GoalState · modules/keypoint_goal.py.

  좌표   로봇 base(= 학습 env-local). 자세는 늘 리셋 때 컵 자세(학습은 기울임 없는 목표만).
  박스   계약 goal_box_min/max(소환 박스 xy · 정착고 + goal_box_z_range) — 밖이면 거부.
  첫 목표  기본 = 리셋 때 컵 + (0, 0, first_z 가운데), 박스로 자른다. 리셋 때 컵에서 수평 ±goal_first_xy_range · 위로 goal_first_z_range. 들어오면 첫 목표를 그것으로 바꾼다.
          밖이면 첫 목표(기본 가운데)는 두고, 거기서부터 중간 목표로 잇는다.
  다음 목표  직전 달성 목표에서 축마다 ±goal_delta_distance(sample_delta_goal 의 균일 박스). 먼 목표는 직선을 n 등분
          (n = ⌈최대 축 거리 / delta⌉)해 차례로.
  달성   키포인트 최대거리 ≤ goal_tol 인 스텝 누적 ≥ goal_success_steps(연속 아님 — force_consecutive False)
          AND 지금 근처 AND 쥠. 달성하면 다음 중간 목표로 넘기고 누적을 0 으로. 줄이 비면 그 목표에 머문다.
  쥠     엄지 촉각 AND 다른 손가락 하나 이상 > grasp_threshold_n(학습 _grasp_flag: 첫마디+손끝 컵 접촉, 실기는 손끝 촉각).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from policy_control import rh_aglt as A

EPS = 1e-9


class GoalInputError(ValueError):
    pass


def has_goal_spec(c: A.RaContract) -> bool:
    return (len(c.goal_box_min) == 3 and len(c.goal_box_max) == 3 and len(c.goal_first_z_range) == 2
            and c.goal_delta_distance > 0.0 and c.goal_success_steps > 0 and c.goal_tol > 0.0)


def in_box(c: A.RaContract, p: np.ndarray) -> bool:
    return bool(np.all(p >= np.asarray(c.goal_box_min) - EPS) and np.all(p <= np.asarray(c.goal_box_max) + EPS))


def in_first_region(c: A.RaContract, p: np.ndarray, cup_reset: np.ndarray) -> bool:
    d = p - cup_reset
    lo, hi = c.goal_first_z_range
    return bool(abs(d[0]) <= c.goal_first_xy_range + EPS and abs(d[1]) <= c.goal_first_xy_range + EPS
                and lo - EPS <= d[2] <= hi + EPS)


def waypoints(start: np.ndarray, end: np.ndarray, delta: float) -> list[np.ndarray]:
    """start → end 직선을 축마다 delta 이내 간격으로 n 등분한 점들(start 제외, end 포함)."""
    d = np.asarray(end, float) - np.asarray(start, float)
    n = max(1, math.ceil(float(np.abs(d).max()) / delta - EPS))
    return [np.asarray(start, float) + d * k / n for k in range(1, n + 1)]


def grasped(c: A.RaContract, tactile_n: Sequence[float]) -> bool:
    t = np.asarray(tactile_n, float).reshape(-1)          # 엄지 → 새끼
    return bool(t[0] > c.grasp_threshold_n and np.any(t[1:] > c.grasp_threshold_n))


@dataclass
class GoalBook:
    """에피소드 하나의 목표 줄. reset 때 만든다(첫 목표 = rh_aglt.first_goal)."""

    c: A.RaContract
    cup_reset: np.ndarray
    quat: np.ndarray
    current: np.ndarray
    queue: list = field(default_factory=list)
    anchor: np.ndarray | None = None          # 직전 달성 목표 — None 이면 아직 첫 목표
    near_steps: int = 0
    successes: int = 0
    kp_dist: float = float("nan")
    target: np.ndarray | None = None           # 사용자가 마지막으로 넣은 최종 목표

    @classmethod
    def start(cls, c: A.RaContract, cup_pos: Sequence[float], cup_quat: Sequence[float]) -> "GoalBook":
        g = A.first_goal(c, cup_pos, cup_quat)
        pos = g.pos.copy()
        if has_goal_spec(c):                       # 학습 sample_first_goal 도 박스로 자른다(_clamp_box)
            pos = np.clip(pos, c.goal_box_min, c.goal_box_max)
        return cls(c=c, cup_reset=np.asarray(cup_pos, float).copy(), quat=np.asarray(cup_quat, float).copy(), current=pos)

    @property
    def goal(self) -> A.Goal:
        return A.Goal(self.current.copy(), self.quat.copy())

    def request(self, target: Sequence[float]) -> list[str]:
        """최종 목표를 받는다. 거부 사유 목록(빈 목록 = 받았다). 받으면 지금 목표 · 줄을 바꾸고 누적을 0 으로."""
        c = self.c
        if not has_goal_spec(c):
            return ["contract has no goal distribution — rebuild it with tools/build_rh_aglt_contract.py"]
        p = np.asarray(target, float).reshape(-1)
        if p.size != 3 or not np.all(np.isfinite(p)):
            return [f"goal must be 3 finite numbers, got {p.tolist()}"]
        if not in_box(c, p):
            return [f"goal {np.round(p, 3).tolist()} outside the training goal box "
                    f"{np.round(c.goal_box_min, 3).tolist()}..{np.round(c.goal_box_max, 3).tolist()}"]
        if self.anchor is None:
            if in_first_region(c, p, self.cup_reset):
                self.current, self.queue = p.copy(), []
            else:                                  # 첫 목표는 그대로, 달성한 뒤 거기서부터 잇는다
                self.queue = waypoints(self.current, p, c.goal_delta_distance)
        else:
            path = waypoints(self.anchor, p, c.goal_delta_distance)
            self.current, self.queue = path[0], path[1:]
        self.target, self.near_steps = p.copy(), 0
        return []

    def step(self, cup_pos: Sequence[float], cup_quat: Sequence[float], tactile_n: Sequence[float],
             is_grasped: bool | None = None) -> bool:
        """is_grasped 를 주면 그 판정(배포: 손끝 또는 관절 힘 — cup_attach.grasp_signal), 없으면 손끝 촉각 규칙."""
        """한 스텝 갱신. 이번 스텝에 목표를 달성했으면 True(다음 중간 목표로 넘긴 뒤)."""
        c = self.c
        kp_c = A.keypoints(c, cup_pos, cup_quat)
        kp_g = A.keypoints(c, self.current, self.quat)
        self.kp_dist = float(np.linalg.norm(kp_c - kp_g, axis=1).max())
        if not has_goal_spec(c):
            return False
        near = self.kp_dist <= c.goal_tol
        self.near_steps += int(near)
        held = grasped(c, tactile_n) if is_grasped is None else bool(is_grasped)
        if not (near and self.near_steps >= c.goal_success_steps and held):
            return False
        if self.anchor is not None and not self.queue and np.allclose(self.anchor, self.current):
            return False                           # 마지막 목표에 머무는 중 — 다시 세지 않는다
        self.successes += 1
        self.anchor = self.current.copy()
        if self.queue:
            self.current = self.queue.pop(0)
            self.near_steps = 0
        return True

    def as_dict(self) -> dict:
        r = lambda v: [round(float(x), 4) for x in v]  # noqa: E731
        return {"goal": r(self.current), "queue": len(self.queue), "target": None if self.target is None else r(self.target),
                "kp_dist": None if math.isnan(self.kp_dist) else round(self.kp_dist, 4), "near_steps": self.near_steps,
                "successes": self.successes}

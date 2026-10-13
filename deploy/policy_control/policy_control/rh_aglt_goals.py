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


FREEZE_RULES = ("success", "lift_hold")


@dataclass(frozen=True)
class FreezeRule:
    """★10.13 grasping(hdgp 9ac23c77 freeze_rule) — 정책을 멈추고 팔 · 손 목표를 고정할 때.

    success   = 기존 goal_max 성공만(기본).
    lift_hold = 관측 컵 z − 리셋 컵 z(GoalBook.cup_reset) > lift_m AND 쥔 판정이 hold_steps 틱 연속 — 한 틱 끊기면 0 부터.
                goal_max 성공과 OR(먼저 오는 쪽). 컵은 정책 입력과 같은 관측 경로(쥐기 전 FP++ live, 쥔 뒤 손바닥 FK)."""

    rule: str = "success"
    lift_m: float = 0.07
    hold_steps: int = 30

    def __post_init__(self) -> None:
        if self.rule not in FREEZE_RULES:
            raise GoalInputError(f"freeze_rule {self.rule!r} — one of {FREEZE_RULES}")
        if not (math.isfinite(self.lift_m) and self.lift_m > 0.0) or int(self.hold_steps) < 1:
            raise GoalInputError(f"freeze_lift_m > 0 and freeze_hold_steps >= 1, got {self.lift_m}, {self.hold_steps}")


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
    freeze: FreezeRule = field(default_factory=FreezeRule)
    lift_dz: float = float("nan")              # 관측 컵 z − 리셋 컵 z
    lift_steps: int = 0                        # (lift_dz > freeze.lift_m AND 쥠) 연속 틱

    @classmethod
    def start(cls, c: A.RaContract, cup_pos: Sequence[float], cup_quat: Sequence[float],
              freeze: FreezeRule | None = None) -> "GoalBook":
        g = A.first_goal(c, cup_pos, cup_quat)
        pos = g.pos.copy()
        if has_goal_spec(c):                       # 학습 sample_first_goal 도 박스로 자른다(_clamp_box)
            pos = np.clip(pos, c.goal_box_min, c.goal_box_max)
        return cls(c=c, cup_reset=np.asarray(cup_pos, float).copy(), quat=np.asarray(cup_quat, float).copy(), current=pos,
                   freeze=freeze or FreezeRule())

    @property
    def goal(self) -> A.Goal:
        return A.Goal(self.current.copy(), self.quat.copy())

    @property
    def reached(self) -> bool:
        """사용자가 넣은 최종 목표 자체를 달성했다(중간 목표 말고) — 에피소드 실행기의 aglt 끝(SETTING 도달)."""
        return (self.target is not None and self.anchor is not None and not self.queue
                and bool(np.allclose(self.anchor, self.target)))

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
        held = grasped(c, tactile_n) if is_grasped is None else bool(is_grasped)
        self.lift_dz = float(np.asarray(cup_pos, float)[2] - self.cup_reset[2])
        self.lift_steps = self.lift_steps + 1 if (self.lift_dz > self.freeze.lift_m and held) else 0
        if not has_goal_spec(c):
            return False
        near = self.kp_dist <= c.goal_tol
        self.near_steps = (self.near_steps + 1 if near else 0) if c.goal_force_consecutive else self.near_steps + int(near)
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
                "successes": self.successes, "reached": self.reached, "freeze_rule": self.freeze.rule,
                "lift_dz": None if math.isnan(self.lift_dz) else round(self.lift_dz, 4), "lift_steps": self.lift_steps}


def end_reason(goals: "GoalBook | None", stop_on_target: bool, goal_max: int) -> str | None:
    """정책 노드가 이번 스텝에 에피소드를 끝낼 사유(None = 계속).

    ★10.13 사용자 방안 A: goal_max > 0 이면 sim 처럼 목표 성공 goal_max 번에 끝낸다(rg3s 학습 goal_max=1 — 첫 목표
    = 컵 리셋 + 10 cm 를 연속 goal_success_steps 틱 유지 + 그 틱에 쥠). 정책은 성공 뒤를 본 적이 없어서 더 돌리면 팔이 흘러간다.
    stop_on_target 이면 예전처럼 실행기가 보낸 SETTING 자체를 달성했을 때 끝낸다."""
    if goals is None:
        return None
    if goal_max > 0 and goals.successes >= goal_max:
        return f"goal {goals.successes}/{goal_max} reached (sim goal_max) {np.round(goals.anchor, 3).tolist()}"
    f = goals.freeze
    if f.rule == "lift_hold" and goals.lift_steps >= f.hold_steps:
        return (f"lift_hold: cup {goals.lift_dz:+.3f} m above reset (> {f.lift_m}) and grasped "
                f"for {goals.lift_steps} ticks (sim freeze_rule lift_hold)")
    if stop_on_target and goals.reached:
        return f"goal target reached {goals.as_dict().get('target')}"
    return None


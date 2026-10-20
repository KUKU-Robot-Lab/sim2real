"""10.19 FP++ 컵 끊김 처리 — live 컵(cup_hold_last)을 쓰는 정책 노드가 컵 위치를 못 받을 때. 순수 numpy.

실기 10.19 19:09 오른손(neo03m): 컵이 손에서 빠지며 기울던 5.73 s 에 FP++ 가 끊겼다. 노드는 마지막 값(공중 5.9 cm)을
계속 정책에 주었고, 정책은 그 자리를 쫓아 팔을 비틀었다(j2 +2.2 rad). 컵이 카메라 밖으로 나가도 같은 일이 생긴다.
10.19 사용자: "실기 노드에 넣는 게 맞다".

규칙(age = 새 FP++ 프레임(영상 시각이 바뀐 것)을 마지막으로 받은 뒤 지난 시간):
  잡기 전   age > stop_s(1.0 s) 이면 멈춘다. 단 마지막 컵이 손바닥 near_m(0.20 m) 안이면 near_max_s(2.5 s)까지는 마지막 값
            (학습 FP++ 가림 DR 은 손 근처에서 0.5~2.5 s 마지막 값을 고정한다 — fpp_gate max_hold_s 와 같은 값).
  쥔 상태   age > follow_s(0.5 s) 이면 컵이 손바닥을 따라간다: (마지막 프레임 시각의 손바닥 FK)⁻¹ ∘ 마지막 컵 을 지금 손바닥 FK 에
            곱한다. 그 프레임이 쥐기 전 것이면 컵은 그때 서 있었으니 쥔 시각의 손바닥과 짝짓는다(cup_attach static 규칙).
            새 프레임이 오면 FP++ 로 돌아간다. 10.14 "위치제어 정책은 쥔 뒤에도 FP++ 만" 의 예외 — 끊긴 동안만.
  놓친 뒤   쥔 상태였다가 쥠이 풀렸는데 age > follow_s 이면 멈춘다(떨어진 컵을 못 찾는 채 정책을 돌리지 않는다).
쥔 상태 = 쥔 신호가 hold_steps(15 틱 = 250 ms) 이어진 뒤부터, release_steps(15 틱) 끊기면 풀림.
  실기 10.19 왼손: 쥐기 직전 쥔 신호가 3.61~4.25 s 동안 1~2 틱씩 깜빡였다 — 깜빡임을 쥠 · 풀림으로 보지 않는다.
시각: 손바닥 = 팔 상태 stamp, 컵 = FP++ 영상 stamp(둘 다 시스템 시계 초). age 는 받은 시각(monotonic)으로 잰다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .cup_attach import _slerp, quat_to_matrix
from .joint_object import matrix_to_quat


class DropoutStop(RuntimeError):
    """끊김 규칙이 에피소드를 멈추라고 한다 — 노드가 abort 로 끝낸다(메시지 = 사유)."""


@dataclass(frozen=True)
class DropoutCfg:
    stop_s: float = 0.0           # > 0 이면 켬 — 잡기 전 끊김 한도
    near_m: float = 0.20          # 잡기 전: 손바닥–마지막 컵 이 안이면 near_max_s 까지 기다린다
    near_max_s: float = 2.5
    follow_s: float = 0.5         # 쥔 상태 · 놓친 뒤: 이만큼 끊기면 따라가기 · 멈춤(노드 CUP_STALE_S 와 같은 값)
    hold_steps: int = 15
    release_steps: int = 15
    history_s: float = 3.0        # 손바닥 FK 기록(쥔 시각 · 프레임 시각과 짝짓기)

    def __post_init__(self) -> None:
        vals = (self.stop_s, self.near_m, self.near_max_s, self.follow_s, self.history_s)
        if not all(np.isfinite(v) and v >= 0.0 for v in vals) or int(self.hold_steps) < 1 or int(self.release_steps) < 1:
            raise ValueError(f"DropoutCfg 값은 유한한 ≥ 0 · 틱 ≥ 1: {self}")
        if self.stop_s > 0.0 and not 0.0 < self.follow_s <= self.stop_s <= self.near_max_s < self.history_s:
            raise ValueError(f"DropoutCfg: 켜면 0 < follow_s ≤ stop_s ≤ near_max_s < history_s: {self}")

    @property
    def enabled(self) -> bool:
        return self.stop_s > 0.0


class CupDropout:
    """한 팔의 컵 끊김 판정. step 은 달리는 정책 틱마다 한 번."""

    def __init__(self, cfg: DropoutCfg) -> None:
        self.cfg = cfg
        self._hist: deque = deque()
        self.new_episode()

    def new_episode(self) -> None:
        self._stamp = None                # 마지막 새 프레임의 영상 시각
        self._t_new: float | None = None  # 그 프레임을 받은 시각(monotonic)
        self.on_steps = 0                 # 쥔 신호가 이어진 틱
        self.off_steps = 0                # 쥔 상태에서 쥔 신호가 끊긴 틱
        self.held = False                 # 쥔 상태(hold_steps 이어짐)
        self.held_since: float | None = None   # 쥔 신호가 켜진 손바닥 시각(쥔 상태가 된 그 연속 구간의 처음)
        self._on_since: float | None = None
        self.released = False             # 쥔 상태였다가 풀렸다(다시 쥔 상태가 되면 False)
        self.mode = "live"                # live | hold(잡기 전 마지막 값) | follow(손바닥 FK)
        self.age_s = 0.0
        self.follows = 0                  # 이 에피소드에서 따라가기를 시작한 횟수
        self._rel = None                  # 따라가기 상대 자세 (p, R) — 새 프레임이 오면 버린다
        self.stop_reason: str | None = None
        self._hist.clear()

    def rebase(self, now: float) -> bool:
        """★10.20 경로 D 넘김 — 잡기 전 끊김 시계를 지금부터 다시 센다(마지막 값 · 영상 stamp 는 그대로). True = 다시 셌다.

        실기 10.20 좌: ap05 가 사전 파지 자리에서 기다리는 동안 손이 컵을 가려 FP++ live 가 4.17 s 에 끊겼고, 2.5 s 뒤 멈췄다.
        넘김은 손바닥이 사전 파지 고리(축에서 r_mult·R)에 서 있을 때만 일어나고, 접근 정책은 컵을 건드리지 않는 단계(학습 접근 성공
        clean) — 컵은 마지막 값 자리에 그대로다. 넘겨받은 정책은 학습 FP++ 가림 DR(손 근처 0.5~2.5 s 마지막 값)과 같은 창을 새로 받는다.
        쥔 상태 · 놓친 뒤 · 프레임을 한 번도 못 받은 경우는 바꾸지 않는다."""
        if self._t_new is None or self.held or self.released:
            return False
        self._t_new = float(now)
        self.age_s = 0.0
        return True

    # ---------------------------------------------------------------- 손바닥 기록
    def _record(self, t: float, palm_pos, palm_R) -> None:
        if self._hist and t <= self._hist[-1][0]:
            return
        self._hist.append((float(t), np.asarray(palm_pos, float).copy(), matrix_to_quat(np.asarray(palm_R, float))))
        while self._hist and self._hist[0][0] < t - self.cfg.history_s:
            self._hist.popleft()

    def _palm_at(self, t: float):
        """기록에서 시각 t 의 손바닥(pos, R). 기록보다 늦으면 마지막 것(FP++ stamp 가 팔 stamp 보다 앞설 수 있다), 이르면 None."""
        h = self._hist
        if not h or t < h[0][0]:
            return None
        if t >= h[-1][0]:
            return h[-1][1], quat_to_matrix(h[-1][2])
        for (t0, p0, q0), (t1, p1, q1) in zip(h, list(h)[1:]):
            if t0 <= t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return p0 + a * (p1 - p0), quat_to_matrix(_slerp(q0, q1, a))
        return None

    # ---------------------------------------------------------------- 판정
    def _grasp(self, grasped: bool, t_palm: float) -> None:
        if grasped:
            self._on_since = float(t_palm) if self.on_steps == 0 else self._on_since
            self.on_steps += 1
            self.off_steps = 0
            if not self.held and self.on_steps >= self.cfg.hold_steps:
                self.held, self.released, self.held_since = True, False, self._on_since
        else:
            self.on_steps = 0
            if self.held:
                self.off_steps += 1
                if self.off_steps >= self.cfg.release_steps:
                    self.held, self.released, self.off_steps = False, True, 0

    def _stop(self, why: str):
        self.stop_reason = f"컵 위치 끊김 {self.age_s:.1f} s — {why} · 멈춤"
        raise DropoutStop(self.stop_reason)

    def step(self, now: float, got, grasped: bool, t_palm: float, palm_pos, palm_R, cup):
        """got = 노드가 마지막으로 받은 (받은 시각, pos, quat, stamp) 또는 None. cup = 지금까지 고른 컵 자세.
        반환 = 정책이 볼 컵 자세(따라가기면 손바닥 FK). 멈출 때는 DropoutStop."""
        self._record(t_palm, palm_pos, palm_R)
        self._grasp(bool(grasped), t_palm)
        if got is not None and got[3] != self._stamp:
            self._stamp, self._t_new, self._rel = got[3], float(got[0]), None
        self.age_s = float("inf") if self._t_new is None else max(0.0, float(now) - self._t_new)
        c = self.cfg
        if self.held:
            if self.age_s <= c.follow_s:
                self.mode = "live"
                return cup
            if self._rel is None:
                self._rel = self._pair(got)
                if self._rel is None:
                    self._stop("쥔 상태인데 짝지을 손바닥 기록이 없다")
                self.follows += 1
            self.mode = "follow"
            R = np.asarray(palm_R, float)
            p_rel, R_rel = self._rel
            return np.asarray(palm_pos, float) + R @ p_rel, matrix_to_quat(R @ R_rel), self._stamp
        if self.released and self.age_s > c.follow_s:
            self._stop("쥠이 풀린 뒤")
        if self.age_s <= c.stop_s:
            self.mode = "live" if self.age_s <= c.follow_s else "hold"
            return cup
        last = None if got is None else np.asarray(got[1], float)
        d = float("inf") if last is None else float(np.linalg.norm(last - np.asarray(palm_pos, float)))
        if d >= c.near_m:
            self._stop(f"잡기 전 · 손바닥–컵 {d * 100:.0f} cm")
        if self.age_s > c.near_max_s:
            self._stop(f"잡기 전 · 손 근처 {c.near_max_s:g} s 넘음")
        self.mode = "hold"
        return cup

    def _pair(self, got):
        """마지막 프레임 ↔ 손바닥: 쥐기 전에 찍힌 프레임이면 쥔 시각 손바닥(컵이 서 있었다), 아니면 그 프레임 시각 손바닥."""
        if got is None:
            return None
        stamp = float(got[3])
        t_ref = stamp if self.held_since is None or stamp >= self.held_since else self.held_since
        at = self._palm_at(t_ref)
        if at is None:
            return None
        p0, R0 = at
        return R0.T @ (np.asarray(got[1], float) - p0), R0.T @ quat_to_matrix(got[2])

    @property
    def following(self) -> bool:
        return self.mode == "follow"

    def as_dict(self) -> dict:
        return {"mode": self.mode, "age_s": None if not np.isfinite(self.age_s) else round(self.age_s, 3),
                "held": self.held, "released": self.released, "follows": self.follows, "stop": self.stop_reason}

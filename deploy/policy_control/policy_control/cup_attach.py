"""컵 자세 — 파지 전에는 FP++, 쥔 뒤에는 손에 붙여 손바닥 FK 로(RH56F1 pour_fj · rh_aglt 계열). 순수 numpy.

10.04 사용자: "FPP 는 기본적으로 정지 상태만 알면 됌. 잡고나서는 cup pose 는 잡을 때 위치를 기준으로 핸드와 함께 fk 계산으로
입력 넣어주면 되니까." 규칙은 T2R RH56F1 Grasping 세션의 sim(rh_aglt, hdgp 2721a946)과 같게 둔다 — 배포와 sim 관측이
같은 순간 · 같은 방식으로 붙는다.

  grasped    엄지 손끝 > force_n 이고 검지 · 중지 · 약지 · 새끼 중 하나라도 > force_n (60 Hz 정책 스텝마다).
             sim 은 첫마디 + 손끝 링크의 컵 접촉력, 실기는 손끝 촉각(/hand_<side>/tip_forces, 엄지 → 새끼)뿐이다.
  attached   grasped 가 이어지는 동안, 파지 시작 뒤 attach_after_s(100 ms) 이상 지나서 **찍힌** FP++ 프레임이 도착하면 붙인다.
             상대 자세 = (그 프레임이 찍힌 시각의 손바닥 FK)⁻¹ ∘ 그 프레임의 컵 자세. FP++ 는 약 0.29 s 늦어서, 같은 프레임을
             '지금' 손바닥에 붙이면 쥐자마자 들어 올린 손과 어긋난다(sim 첫 기동 77 mm · trace 100 mm 넘게) — 손바닥 FK 를
             history_s 동안 기록해 두고 프레임 시각에서 보간한다. 그 전까지(대개 파지 뒤 약 0.4 s)는 FP++ 를 그대로 쓴다.
             붙인 뒤 컵 = 지금 손바닥 FK ∘ 상대 자세. FP++ 는 보지 않는다(가려져 흐르는 값을 받지 않는다 — 09.28 joint 계열 교훈).
  released   grasped 가 release_steps(15 = 250 ms) 스텝 연속 끊기면 떼고 FP++ 로 돌아간다. 다시 쥐면 처음부터.
시각: 손바닥 = 팔 상태 메시지 stamp, 컵 = FP++ 영상 stamp(RealSense). 둘 다 시스템 시계 초다.
붙인 뒤 컵이 손에서 미끄러져도 이 추정은 모른다 — status 의 cup.source 가 'attached' 인지 운영자가 본다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .joint_object import matrix_to_quat


@dataclass(frozen=True)
class AttachCfg:
    force_n: float = 1.0          # sim _grasp_flag 문턱(엄지 · 다른 손가락 각각)
    attach_after_s: float = 0.1   # 파지 시작 뒤 이만큼 지나서 찍힌 프레임부터 붙인다(sim 6 스텝)
    release_steps: int = 15       # 250 ms @ 60 Hz
    history_s: float = 0.8        # 손바닥 FK 기록 — FP++ 지연(최대 약 0.4 s)보다 넉넉히
    max_extrapolate_s: float = 0.02


def grasp_flag(tactile_n, force_n: float) -> bool:
    """손끝 법선 힘 5(엄지 → 새끼, N) → 쥐었나. 순수."""
    t = np.asarray(tactile_n, float).reshape(-1)
    if t.size < 5:
        return False
    return bool(t[0] > force_n and np.any(t[1:5] > force_n))


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


class CupAttach:
    """한 팔의 컵. step 은 정책 스텝마다 한 번(판정 · 기록이 움직인다), peek 은 상태를 바꾸지 않는다.

    live = (pos, quat, stamp_s) 또는 None — stamp 는 FP++ 영상 시각.
    """

    def __init__(self, cfg: AttachCfg = AttachCfg()) -> None:
        self.cfg = cfg
        self._hist: deque = deque()
        self.reset()

    def reset(self) -> None:
        self.attached = False
        self.grasp_since: float | None = None
        self.off_steps = 0
        self._rel_p: np.ndarray | None = None
        self._rel_R: np.ndarray | None = None
        self._hist.clear()

    @property
    def source(self) -> str:
        return "attached" if self.attached else "live"

    # ---------------------------------------------------------------- 손바닥 기록
    def record(self, t: float, palm_pos, palm_R) -> None:
        if self._hist and t <= self._hist[-1][0]:
            return                                   # 같은 메시지를 두 번 보거나 시각이 뒤로 — 버린다
        self._hist.append((float(t), np.asarray(palm_pos, float).copy(), matrix_to_quat(np.asarray(palm_R, float))))
        while self._hist and self._hist[0][0] < t - self.cfg.history_s:
            self._hist.popleft()

    def palm_at(self, t: float):
        """기록에서 시각 t 의 손바닥(pos, R) — 위치는 선형, 자세는 slerp. 기록 밖이면 None."""
        h = self._hist
        if not h or t < h[0][0]:
            return None
        if t >= h[-1][0]:
            if t - h[-1][0] > self.cfg.max_extrapolate_s:
                return None
            return h[-1][1], quat_to_matrix(h[-1][2])
        for (t0, p0, q0), (t1, p1, q1) in zip(h, list(h)[1:]):
            if t0 <= t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return p0 + a * (p1 - p0), quat_to_matrix(_slerp(q0, q1, a))
        return None

    # ---------------------------------------------------------------- 컵
    def _from_palm(self, palm_pos, palm_R) -> tuple[np.ndarray, np.ndarray]:
        R = np.asarray(palm_R, float)
        return np.asarray(palm_pos, float) + R @ self._rel_p, matrix_to_quat(R @ self._rel_R)

    def peek(self, palm_pos, palm_R, live):
        """지금 쓸 컵 자세 — 붙어 있으면 FK, 아니면 FP++(없으면 None)."""
        if self.attached:
            return self._from_palm(palm_pos, palm_R)
        return live

    def step(self, grasped: bool, t: float, palm_pos, palm_R, live):
        """정책 스텝 하나(t = 이 손바닥 FK 의 시각): 기록 · 판정 · 붙이기 · 떼기 뒤 쓸 컵 자세."""
        self.record(t, palm_pos, palm_R)
        if self.attached:
            self.off_steps = 0 if grasped else self.off_steps + 1
            if self.off_steps >= self.cfg.release_steps:
                hist = list(self._hist)
                self.reset()
                self._hist.extend(hist)                  # 기록은 이어 간다(다시 쥘 때 쓴다)
                return live
            return self._from_palm(palm_pos, palm_R)
        if not grasped:
            self.grasp_since = None
            return live
        if self.grasp_since is None:
            self.grasp_since = float(t)
        if live is not None and len(live) > 2 and live[2] is not None \
                and float(live[2]) >= self.grasp_since + self.cfg.attach_after_s:
            at = self.palm_at(float(live[2]))
            if at is not None:
                p0, R0 = at
                self._rel_p = R0.T @ (np.asarray(live[0], float) - p0)
                self._rel_R = R0.T @ quat_to_matrix(live[1])
                self.attached, self.off_steps = True, 0
                return self._from_palm(palm_pos, palm_R)
        return live

    def as_dict(self) -> dict:
        return {"source": self.source, "grasp_since": self.grasp_since, "off_steps": self.off_steps,
                "rel_pos": None if self._rel_p is None else [round(float(v), 4) for v in self._rel_p]}

"""11.04 깊이 학생(rh_aglt 잡기) 넘김 상태 기계 — hdgp student_adapters/rh_aglt.py DeployHandover(평가 설정)와 같은 규칙.

접근 시작부터 틱마다 step(손바닥 위치)를 부른다(리셋 대기 동안은 부르지 않는다, 그동안 단계 = approach).
  넘김 = 접근 T_min 뒤, 지난 still_s 창의 손바닥 평균 속도(창 동안 변위 / 창 길이) ≤ still_mps. 창 = 최근 still_s 의 틱 위치
         (approach_grasp 의 palm_win 과 같은 식). 창 안에서 움직이면 그 위치가 창을 빠져나갈 때까지(still_s) 넘기지 않는다.
  넘긴 틱부터 단계 = grasp(되돌리지 않음). 그 틱 포함 stop_s 동안 정지 — 실행 행동 = 팔 0 · 손 폄(rh_aglt.ready_hold_action),
         학생 출력은 버린다(LSTM 은 계속 돈다). 그다음 틱부터 학생 행동.
  접근 상한 max_s: on_max "handover" = 넘김과 같게(sim ap_max_steps 6 s), "fail" = 실패(배포 approach_max_s 15 s).
sim 은 판정을 행동 뒤 위치로 하고(t = 접근 스텝 − 1), 배포는 행동 전 측정으로 부르므로 n 번째 호출 = sim t = n − 1 로 맞춘다.
학습 전용(창 길이 흔들기 · r/R 일찍 넘김)은 평가 기본 0 이라 뺐다. 컵 위치(ready_tol) 조건은 없다.
"""
from __future__ import annotations

from collections import deque

import numpy as np

STOP_S = 0.5            # hdgp rh_aglt_env_cfg ap_stop_steps 30 @ 60 Hz
ON_MAX = ("handover", "fail")


class StudentHandover:
    def __init__(self, *, hz: float, max_s: float, on_max: str, tmin_s: float, still_s: float, still_mps: float,
                 stop_s: float = STOP_S) -> None:
        if on_max not in ON_MAX:
            raise ValueError(f"on_max {on_max!r} ∉ {ON_MAX}")
        if not (hz > 0 and max_s > 0 and still_s > 0 and still_mps > 0 and tmin_s >= 0 and stop_s >= 0):
            raise ValueError(f"넘김 인자 범위: hz {hz} · max_s {max_s} · tmin_s {tmin_s} · still_s {still_s} · "
                             f"still_mps {still_mps} · stop_s {stop_s}")
        if tmin_s < still_s:     # 창이 차기 전(v = inf)에는 못 넘기므로 T_min 이 창보다 짧으면 T_min 이 뜻을 잃는다
            raise ValueError(f"tmin_s {tmin_s} < still_s {still_s}")
        self.dt, self.on_max, self.still_mps = 1.0 / hz, on_max, still_mps
        self.w = max(2, round(still_s * hz))          # hdgp DeployHandover.w
        self.tmin_n, self.stop_n, self.max_n = round(tmin_s * hz), round(stop_s * hz), round(max_s * hz)
        self.reset()

    def reset(self) -> None:
        self.n = 0                  # 접근 호출 수(= 실행된 접근 행동 수)
        self.win: deque = deque(maxlen=self.w + 1)
        self.stop_left: int | None = None     # None = 아직 접근
        self.failed = False
        self.why = ""
        self.v = float("inf")       # 마지막 창 평균 속도 [m/s](기록용)

    @property
    def stage(self) -> str:
        return "approach" if self.stop_left is None else "grasp"

    def step(self, palm) -> tuple[str, bool, bool]:
        """이 틱 측정 손바닥 위치(3,) → (단계, 정지 틱인가, 실패). 정지 틱이면 실행 행동 = ready_hold_action."""
        if self.stop_left is not None:
            self.stop_left = max(0, self.stop_left - 1)
            return "grasp", self.stop_left > 0, False
        if self.failed:
            return "approach", False, True
        self.win.append(np.asarray(palm, float).copy())
        n, self.n = self.n, self.n + 1
        if len(self.win) == self.win.maxlen:
            self.v = float(np.linalg.norm(self.win[-1] - self.win[0])) / (self.w * self.dt)
        if n - 1 >= self.tmin_n and self.v <= self.still_mps:
            return self._hand(f"멈춤 {self.v:.3f} m/s · 접근 {n * self.dt:.2f} s")
        if n >= self.max_n:
            if self.on_max == "fail":
                self.failed, self.why = True, f"접근 상한 {self.max_n * self.dt:.1f} s 안에 못 멈춤(창 평균 {self.v:.3f} m/s)"
                return "approach", False, True
            return self._hand(f"접근 상한 {self.max_n * self.dt:.1f} s")
        return "approach", False, False

    def _hand(self, why: str) -> tuple[str, bool, bool]:
        self.stop_left, self.why = self.stop_n, why
        return "grasp", self.stop_n > 0, False

    def action(self, stopping: bool, student_a: np.ndarray, hold_a: np.ndarray) -> np.ndarray:
        """정지 틱이면 hold_a, 아니면 학생 행동."""
        return np.asarray(hold_a if stopping else student_a, float)

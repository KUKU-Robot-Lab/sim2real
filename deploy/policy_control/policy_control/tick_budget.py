"""정책 틱 예산 — 한 에피소드 동안 틱이 예산(1/policy_hz)을 넘긴 횟수를 센다(순수, ROS 없음).

★11.09 실기 느린 bag(161342 · 181759)은 정책 한 틱이 25 ms 로 60 Hz 예산 16.7 ms 를 넘겨 실제 37 Hz 로 돌았다.
다음 실기에서 status 한 줄로 바로 보이게 한다.
- proc: 틱 안의 계산 시간(관측 조립 + 추론 + 발행)이 예산을 넘긴 횟수
- late: 틱과 틱 사이 간격이 예산의 1.5 배를 넘긴 횟수(타이머가 밀린 것 — 계산 밖의 지연까지 잡힌다)
"""
from __future__ import annotations

from dataclasses import dataclass, field

LATE_FACTOR = 1.5


@dataclass
class TickBudget:
    budget_ms: float
    ticks: int = 0
    proc_over: int = 0
    late: int = 0
    proc_max_ms: float = 0.0
    gap_max_ms: float = 0.0
    _last_s: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.ticks = self.proc_over = self.late = 0
        self.proc_max_ms = self.gap_max_ms = 0.0
        self._last_s = None

    def record(self, t_start_s: float, proc_ms: float) -> None:
        """t_start_s = 틱이 시작한 단조 시각(s), proc_ms = 그 틱의 계산 시간."""
        self.ticks += 1
        if proc_ms > self.budget_ms:
            self.proc_over += 1
        self.proc_max_ms = max(self.proc_max_ms, proc_ms)
        if self._last_s is not None:
            gap = (t_start_s - self._last_s) * 1e3
            self.gap_max_ms = max(self.gap_max_ms, gap)
            if gap > LATE_FACTOR * self.budget_ms:
                self.late += 1
        self._last_s = t_start_s

    def as_dict(self) -> dict:
        return {"budget_ms": round(self.budget_ms, 2), "ticks": self.ticks, "proc_over": self.proc_over,
                "late": self.late, "proc_max_ms": round(self.proc_max_ms, 2), "gap_max_ms": round(self.gap_max_ms, 2)}

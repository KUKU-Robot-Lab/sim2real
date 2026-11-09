"""TickBudget — 예산 넘긴 틱 · 밀린 틱 개수(11.09 실기 37 Hz 재현)."""
from policy_control.tick_budget import TickBudget


def test_counts_over_budget_and_late_ticks():
    b = TickBudget(1e3 / 60)                       # 16.67 ms
    t = 0.0
    for proc_ms, gap_s in [(5.0, 1 / 60), (25.0, 1 / 60), (6.0, 0.027), (5.0, 1 / 60)]:
        b.record(t, proc_ms)
        t += gap_s
    d = b.as_dict()
    assert d["ticks"] == 4 and d["proc_over"] == 1 and d["late"] == 1   # 27 ms > 1.5×16.7 ms
    assert d["proc_max_ms"] == 25.0 and 26.9 < d["gap_max_ms"] < 27.1


def test_reset_clears_the_last_tick():
    b = TickBudget(16.7)
    b.record(0.0, 30.0)
    b.reset()
    b.record(10.0, 1.0)                            # 리셋 뒤 첫 틱은 간격을 안 잰다
    assert b.as_dict()["late"] == 0 and b.as_dict()["proc_over"] == 0 and b.ticks == 1

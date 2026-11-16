"""place_run_report.open_latch — 합성 손 목표 · 래치 열로 t15 · 20 틱 열림 · 래치 틱 계산 (ROS 없음)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

TOOLS = Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"


def _tool():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("place_run_report_tool", TOOLS / "place_run_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_open_latch_ticks():
    m = _tool()
    close_dir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])     # 쥐는 쪽 = +
    fingers = [2, 3, 4, 5]
    hand0 = np.full(6, 1.0)
    K = 30
    ramp = np.clip((np.arange(K) - 4) * 0.03, 0, None)         # 틱 4 부터 틱당 0.03 rad 씩 연다 → 0.15 는 틱 9
    ht = hand0 - ramp[:, None]
    ht[:, :2] = hand0[:2]                                      # 엄지는 열지 않는다(평균에서 빠짐)
    hq = hand0 - np.r_[np.zeros(3), ramp[:-3]][:, None]        # 실측은 3 틱 늦다 → 틱 12
    latch = np.zeros(K)
    latch[17:] = 1.0
    r = m.open_latch(hand0, ht, hq, close_dir, fingers, latch, thr=0.15)
    assert r["t15"] == 9 and r["t_rel"] == 12 and r["latch_tick"] == 17 and r["latch_all0"] is False
    assert r["open20"] == pytest.approx(0.03 * 15)
    r0 = m.open_latch(hand0, np.tile(hand0, (5, 1)), np.tile(hand0, (5, 1)), close_dir, fingers, np.zeros(5))
    assert r0["t15"] is None and r0["open20"] is None and r0["latch_tick"] is None and r0["latch_all0"] is True


def test_bank_compare_const_cells_not_z():
    """뱅크 std 0 칸(arm_qd 전부 0)은 z·범위 밖에서 빠지고 '뱅크 상수' 로 절대 차이만 나온다."""
    m = _tool()
    rng = np.random.default_rng(0)
    B = np.zeros((50, 4))
    B[:, :2] = rng.normal(0, 1, (50, 2))  # cup_rel_palm 처럼 퍼진 칸 2개, arm_qd 2칸은 상수 0
    x = np.array([0.1, 5.0, 0.02, -0.03])
    cells = [("cup", 0), ("cup", 1), ("arm_qd", 2), ("arm_qd", 3)]
    r = m.bank_compare(B, x, cells, {"cup", "arm_qd"})
    assert r["n_bank_const"] == 2 and r["bank_const"] == [("arm_qd", 0.03)]
    assert r["n_out"] == 1 and r["n_z3"] == 1  # cup[1] 만
    assert [n for n, _ in r["top_z"]] == ["cup", "cup"] and max(abs(v) for _, v in r["top_z"]) < 10

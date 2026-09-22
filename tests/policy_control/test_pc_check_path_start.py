"""저장 경로 재생 전 검사 — 시작점 · 엔코더 살아 있음(09.22 CAN RX 0 인데 전부 0.0)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools" / "check_path_start.py"
_s = importlib.util.spec_from_file_location("check_path_start", _P)
cps = importlib.util.module_from_spec(_s)
sys.modules["check_path_start"] = cps
_s.loader.exec_module(cps)

J = [f"r_aj_{i}" for i in range(1, 8)]


def test_start_within_tolerance_passes_and_far_start_is_refused():
    start = np.zeros(7)
    near = dict(zip(J, [0.006, -0.01, 0.0, 0.004, 0.0, 0.002, -0.003]))
    assert cps.verdict(near, J, start, 0.05) == []
    far = {**near, "r_aj_4": 0.3}
    assert any("r_aj_4" in r for r in cps.verdict(far, J, start, 0.05))


def test_all_exact_zero_means_the_encoders_are_not_read():
    zero = dict.fromkeys(J, 0.0)
    assert any("0.0" in r for r in cps.verdict(zero, J, np.zeros(7), 0.05))
    assert cps.verdict(zero, J, np.zeros(7), 0.05, allow_exact_zero=True) == []    # fake 플랜트


def test_missing_joints_are_refused():
    assert any("없음" in r for r in cps.verdict({}, J, np.zeros(7), 0.05))


def test_the_hand_must_be_in_the_pose_the_path_was_checked_with():
    # 09.22 실기: 차렷에서 손은 주먹이었다 — 경로는 그 손으로 검사했고, pd 는 팔이 움직이는 동안 손을 그대로 둔다.
    hq = "r_hj_index_2=1.62,r_hj_middle_2=1.59"
    assert cps.hand_verdict({"r_hj_index_2": 1.58, "r_hj_middle_2": 1.61}, hq, 0.2) == []
    assert any("r_hj_index_2" in r for r in cps.hand_verdict({"r_hj_index_2": 0.0, "r_hj_middle_2": 1.6}, hq, 0.2))
    assert cps.hand_verdict({}, "", 0.2) == []                                     # 손을 가정하지 않은 경로

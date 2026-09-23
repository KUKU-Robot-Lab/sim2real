"""plan_approach_to_start.py — 지금 자세 → 저장 경로 시작점 정렬 구간의 판정 규칙(ROS·MuJoCo 없음).

저장 경로가 이미 인정한 '시작 자세에서 좁은 쌍'(npz 의 meta_escape_pairs)은 그 자세보다 가까워지지만 않으면
통과시킨다. 09.23: 실기 차렷에서 r_hl_flange_adapter 가 상판에서 1.6 cm 라 1° 정렬조차 거부됐다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SIM2REAL = Path(__file__).resolve().parents[2]
TOOLS = SIM2REAL / "deploy/policy_control/tools"

pytestmark = pytest.mark.unit


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("plan_approach_to_start")
ADAPTER = ("r_hl_flange_adapter", "table_8")
THUMB = ("r_hl_thumb_2", "table_8")
SAVED = {"r_hl_flange_adapter<->table_8", "r_hl_thumb_2<->table_8"}


def _fail(pair, dist):
    return {"pair": pair, "dist": dist, "s": 0.01, "q": [0.0] * 7}


def test_a_pair_the_saved_path_already_tolerates_passes_when_it_does_not_get_closer():
    fails = [_fail(ADAPTER, 0.0161)]
    assert A.drop_saved_escape_fails(fails, {ADAPTER: 0.0161}, SAVED) == []


def test_getting_closer_than_the_saved_start_still_fails():
    fails = [_fail(ADAPTER, 0.0120)]
    assert A.drop_saved_escape_fails(fails, {ADAPTER: 0.0161}, SAVED) == fails


def test_a_pair_the_saved_path_did_not_tolerate_still_fails():
    fails = [_fail(("r_hl_index_4", "wall_y_neg"), 0.0161)]
    assert A.drop_saved_escape_fails(fails, {("r_hl_index_4", "wall_y_neg"): 0.0161}, SAVED) == fails


def test_a_pair_missing_from_the_start_pose_distances_still_fails():
    fails = [_fail(THUMB, 0.0161)]
    assert A.drop_saved_escape_fails(fails, {}, SAVED) == fails

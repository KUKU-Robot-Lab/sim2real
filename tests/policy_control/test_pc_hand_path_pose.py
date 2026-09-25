"""손 기준 자세 — 어떤 손 자세에서든 저장 경로가 검사한 그 모양으로 맞춘다 (ROS 없음, 순수 부분).

09.23 실기: 손 전원을 껐다 켜자 손가락이 다른 자세로 자리 잡아(엄지 0.9 rad 차이) 경로 시작점 검사가 막았다.
저장 경로를 다시 계획하는 대신 손을 되돌린다 — 기준 자세는 pd yaml 한 곳이고 경로 계획도 같은 값을 읽는다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

SIM2REAL = Path(__file__).resolve().parents[2]
TOOLS = SIM2REAL / "deploy/policy_control/tools"
CONFIGS = ("pd_dg5f_m_short.yaml", "pd_dg5f_m_short_exec.yaml", "pd_dg5f_m_short_fake.yaml")

pytestmark = pytest.mark.unit


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("hand_to_path_pose")


def _pose(cfg: str, side: str) -> dict:
    raw = yaml.safe_load((SIM2REAL / "deploy/policy_control/config" / cfg).read_text())
    return raw["hand_path_pose"][side]


def test_the_sweep_starts_where_the_hand_is_and_ends_at_the_base_pose():
    now, want = {"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0}
    steps = H.sweep(now, want, 4)
    assert len(steps) == 5
    assert steps[0] == pytest.approx(now) and steps[-1] == pytest.approx(want)
    assert steps[2]["a"] == pytest.approx(0.5)                       # 관절공간 직선


def test_a_joint_the_hand_does_not_report_falls_back_to_the_base_pose():
    steps = H.sweep({}, {"a": 0.7}, 2)
    assert all(s["a"] == pytest.approx(0.7) for s in steps)          # 모르는 값으로 움직이지 않는다


def test_every_short_config_carries_the_same_base_pose_for_both_hands():
    right = _pose(CONFIGS[0], "right")
    assert len(right) == 20 and len(_pose(CONFIGS[0], "left")) == 20
    for cfg in CONFIGS[1:]:
        assert _pose(cfg, "right") == right, cfg                     # 발행 사본·fake 가 같은 값을 쓴다
        assert _pose(cfg, "left") == _pose(CONFIGS[0], "left"), cfg


def test_the_base_pose_sits_inside_the_vendor_limits_with_margin():
    # 09.23 실기: 한계값 그대로 지령하면 기계 끝점으로 미는 것이다 — 손가락이 꺾였다
    prof = yaml.safe_load((SIM2REAL.parent / "robot_control/src/robot_control/profiles/openarm_tesollo.yaml").read_text())
    lim = {j["canonical"]: (j["lower"], j["upper"]) for j in prof["joints"] if "lower" in j}
    for side in ("right", "left"):
        for j, v in _pose(CONFIGS[0], side).items():
            lo, hi = lim[j]
            assert lo + 0.049 <= v <= hi - 0.049, (j, v, lim[j])


def test_the_planner_can_read_the_same_pose():
    src = (TOOLS / "plan_home_path.py").read_text()
    assert '"pd"' in src and "hand_path_pose" in src                 # --hand-start pd


C = _load("capture_hand_pose")


def test_capture_pulls_the_measured_pose_inside_the_limits():
    # 09.23 실기: 한계값 그대로 기준 자세로 삼으면 pd 가 기계 끝점으로 민다 — 손가락이 꺾였다
    lim = {"j1": (0.0, 2.0), "j2": (-2.7, 0.0), "j3": (0.0, 0.05)}
    out, moved = C.clamped({"j1": 0.0, "j2": -1.2, "j3": 0.0}, lim, 0.05)
    assert out["j1"] == pytest.approx(0.05) and out["j2"] == pytest.approx(-1.2)
    assert out["j3"] == pytest.approx(0.025)                       # 범위가 여유의 두 배보다 좁으면 가운데
    assert [m[0] for m in moved] == ["j1", "j3"]                    # 무엇을 물렸는지 말해 준다


def test_capture_writes_the_same_block_into_every_short_config(tmp_path, monkeypatch):
    pose = {"right": {"r_hj_index_2": 1.4}, "left": {"l_hj_index_2": 1.4}}
    for name in C.CONFIGS:                                          # 실제 설정을 베껴 온다(블록 모양 그대로)
        (tmp_path / name).write_text((SIM2REAL / "deploy/policy_control/config" / name).read_text())
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path)
    assert C.write_configs(pose) == list(C.CONFIGS)
    for name in C.CONFIGS:
        raw = yaml.safe_load((tmp_path / name).read_text())
        assert raw["hand_path_pose"] == pose, name
        assert raw["settle"]["tol"] == 0.01 and raw["gravity"]["mode"] == "model_tau_ff"   # 다른 곳은 그대로
        assert "09.23 실기" in (tmp_path / name).read_text()                                # 주석도 남는다


def test_a_pair_already_tight_at_both_ends_may_come_a_little_closer_but_never_touch():
    # 09.23 실기: 차렷에서 새끼손가락이 받침판 가장자리에 겹쳐 있어(모델이 보수적) 3 mm 접근에 막혔다.
    src = (TOOLS / "hand_to_path_pose.py").read_text()
    assert "SWEEP_DIP = 0.005" in src
    assert "need = max(0.0, need - SWEEP_DIP)" in src                # 관통(0) 아래로는 못 내려간다
    assert "if 0.0 < need < args.margin" in src
    # 이미 닿아 있는 쌍(차렷에서 손가락이 받침판에 얹혀 있다)은 한도 안에서만 더 깊어져도 둔다
    assert "TOUCH_SLACK = 0.010" in src and "need -= TOUCH_SLACK" in src

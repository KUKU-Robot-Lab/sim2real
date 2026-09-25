"""reset_to_rest.py — 경로 위 어디서 멈췄든 그 지점부터 되짚어 차렷으로 (ROS 없음, 순수 부분).

09.23 실기: goto_home 정착 실패로 팔이 홈 근처에 선 채 멈췄다. 되돌아오는 길은 올라온 경로를
거꾸로 가는 것뿐이다 — 관절공간 직선은 손이 지나는 곳을 계산하지 않는다(09.07 테이블).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SIM2REAL = Path(__file__).resolve().parents[2]
TOOLS = SIM2REAL / "deploy/policy_control/tools"
NPZ = SIM2REAL / "deploy/policy_control/paths/home_right.npz"

pytestmark = pytest.mark.unit


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("reset_to_rest")


def test_the_nearest_frame_is_the_one_we_stopped_at():
    frames = np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2], [1.5, 0.3]])
    assert R.nearest_frame(frames, np.array([1.02, 0.18])) == (2, pytest.approx(0.02))
    assert R.nearest_frame(frames, np.array([0.0, 0.0])) == (0, pytest.approx(0.0))
    assert R.nearest_frame(frames, np.array([1.5, 0.3]))[0] == 3


def test_a_pose_far_off_the_path_reports_its_distance():
    frames = np.array([[0.0, 0.0], [1.0, 0.0]])
    k, gap = R.nearest_frame(frames, np.array([0.5, 0.9]))
    assert gap == pytest.approx(0.9) and k in (0, 1)          # --max-gap 이 이 값을 보고 멈춘다


def test_on_the_saved_path_every_frame_finds_itself():
    if not NPZ.exists():
        pytest.skip("저장 경로 없음")
    frames = np.load(NPZ)["arm_target"]
    for i in (0, len(frames) // 3, len(frames) // 2, len(frames) - 1):
        k, gap = R.nearest_frame(frames, frames[i])
        assert k == i and gap == pytest.approx(0.0, abs=1e-12)


def test_the_trimmed_reset_path_ends_where_we_are_and_starts_at_rest():
    if not NPZ.exists():
        pytest.skip("저장 경로 없음")
    d = np.load(NPZ)
    frames = d["arm_target"]
    k = len(frames) // 2
    cut = frames[:k + 1]                                       # 도구가 쓰는 구간
    assert np.allclose(cut[0], d["meta_start"])                # 되짚기의 도착점 = 차렷
    assert np.allclose(cut[-1], frames[k])                     # 되짚기의 출발점 = 멈춘 자리
    assert np.abs(np.diff(cut, axis=0)).max() / float(d["meta_step_dt"]) <= 0.2 + 1e-9


def test_an_already_engaged_pd_is_not_engaged_again(monkeypatch):
    """정책을 돌리다 멈춘 자리에서 부르면 pd 는 이미 팔을 잡고 있다 — engage 는 거부된다.

    09.23 fake: fabric 뒤에 reset 을 부르자 'forward effort controller already active ·
    phase TRACKING is not IDLE' 로 막혔다. 잡고 있는 상태가 곧 우리가 원하는 상태다.
    """
    class Out:
        def __init__(self, text):
            self.stdout, self.returncode = text, 0

    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return Out(seen["phase"])

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    for phase in R.PD_HOLDING:
        seen["phase"] = f"{phase}\n"
        assert R.pd_holds("right") is True, phase
    for phase in ("IDLE", "", "FAULT"):
        seen["phase"] = f"{phase}\n"
        assert R.pd_holds("left") is False, phase
    assert "--read-pd" in seen["argv"]                    # 구독 전용 — 서비스는 부르지 않는다
    #: pd status 는 팔마다 따로다(09.23) — 어느 팔을 보는지 말해야 한다
    assert seen["argv"][seen["argv"].index("--side") + 1] == "left"

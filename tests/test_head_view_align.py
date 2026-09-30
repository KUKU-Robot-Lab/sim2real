"""머리 화면 정렬 도구 — 평면 수치 · 제안 offset 계산 · 저장된 5090 기준. 하드웨어 불필요."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from head_view_align import (DEFAULT_HOME, DEFAULT_REF, deg_to_tick, load_ref, proposed_homing_offset,
                             targets_from_home, view_metrics)

K = np.array([[606.6, 0.0, 320.0], [0.0, 605.7, 240.6], [0.0, 0.0, 1.0]])


def _plane_depth(pitch_deg: float, roll_deg: float, dist: float) -> np.ndarray:
    """카메라에서 수직 거리 dist 인 평면을 pitch(광축 기울기) · roll 로 본 깊이 영상."""
    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    n = np.array([-np.sin(p) * np.sin(r), np.sin(p) * np.cos(r), np.cos(p)])      # +z 쪽 법선
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    return (dist / (rays @ n)).astype(np.float32)


@pytest.mark.parametrize("pitch, roll, dist", [(18.9, 4.7, 0.614), (25.0, -3.0, 0.5), (5.0, 0.0, 0.7)])
def test_plane_metrics_recover_pitch_roll_and_distance(pitch, roll, dist):
    m = view_metrics(_plane_depth(pitch, roll, dist), K)
    assert m["pitch_deg"] == pytest.approx(pitch, abs=0.2)
    assert m["dist_m"] == pytest.approx(dist, abs=0.003)
    assert m["roll_deg"] == pytest.approx(roll, abs=0.5)
    assert m["inliers"] > 0.95


def test_the_proposed_offset_makes_this_pose_read_the_target():
    # 지금 offset 0 에서 1037 로 읽히는 자세가 목표(2048)라면 offset 1011 → 1037 + 1011 = 2048
    assert proposed_homing_offset(present=1037, current_offset=0, target=2048) == 1011
    # offset 이 이미 들어가 있으면 raw 기준으로 다시 잰다
    assert proposed_homing_offset(present=2048, current_offset=1011, target=2048) == 1011
    assert proposed_homing_offset(present=2000, current_offset=1011, target=2048) == 1059


def test_targets_are_the_head_home_angles_in_ticks():
    assert deg_to_tick(0.0) == 2048 and deg_to_tick(-20.0) == 1820
    assert targets_from_home(DEFAULT_HOME) == (2048, 1820)


@pytest.mark.skipif(not Path(DEFAULT_REF).is_file(), reason="기준 npz 없음")
def test_the_5090_reference_sees_the_table_from_the_home_pose():
    ref = load_ref(DEFAULT_REF)
    assert ref["rgb"].shape == (480, 640, 3) and (ref["pan_tick"], ref["tilt_tick"]) == (2049, 1820)
    m = view_metrics(ref["depth"], ref["K"])
    assert m["pitch_deg"] == pytest.approx(18.9, abs=1.0) and m["dist_m"] == pytest.approx(0.614, abs=0.02)


def test_signed_pitch_keeps_its_sign_across_straight_down():
    assert view_metrics(_plane_depth(10.0, 0.0, 0.6), K)["pitch_signed_deg"] == pytest.approx(10.0, abs=0.2)
    assert view_metrics(_plane_depth(-10.0, 0.0, 0.6), K)["pitch_signed_deg"] == pytest.approx(-10.0, abs=0.2)


def test_autoalign_steps_toward_zero_and_never_leave_the_window():
    from head_view_autoalign import clamp_step, within_window
    assert clamp_step(error=4.0, gain=1.0, max_step=2.0) == -2.0          # 큰 오차도 한 걸음은 2° 까지
    assert clamp_step(error=-0.5, gain=-1.0, max_step=2.0) == pytest.approx(-0.5)
    assert clamp_step(error=3.0, gain=0.0, max_step=2.0) == 0.0            # 이득을 모르면 움직이지 않는다
    lim = round(25 * 4096 / 360)
    assert within_window(5000, 2048, 25.0) == 2048 + lim and within_window(0, 2048, 25.0) == 2048 - lim

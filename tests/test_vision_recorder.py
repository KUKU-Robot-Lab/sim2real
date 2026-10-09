"""vision_recorder — 실기 런 동안 카메라 JPEG(5 Hz) · FP++ 실시간 좌표(카메라 · base)를 남긴다(10.09 사용자, 순수부).
카메라 · FP++ 는 localhost 전용 DDS 라 팔 bag 기록기가 못 본다 — 이 노드가 ROS_LOCALHOST_ONLY=1 로 돈다."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "nodes"))
import vision_recorder as V  # noqa: E402


def test_a_live_pose_row_carries_camera_and_base_positions_and_tilt():
    T = np.eye(4)
    T[:3, 3] = (0.3, 0.0, 0.9)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])                 # 카메라가 아래를 본다
    row = V.pose_row(12.5, "source200_pink", (0.0, 0.0, 0.6), (1.0, 0.0, 0.0, 0.0), T)   # 카메라 x 축 180° = 서 있음
    assert row["name"] == "source200_pink" and row["t"] == 12.5
    assert (row["bx"], row["by"], row["bz"]) == pytest.approx((0.3, 0.0, 0.3))
    assert row["tilt_deg"] == pytest.approx(0.0, abs=1e-6)


def test_frames_are_kept_at_the_asked_rate():
    gate = V.RateGate(5.0)
    kept = [t for t in np.arange(0.0, 2.0, 1 / 30) if gate.take(float(t))]
    assert 9 <= len(kept) <= 11

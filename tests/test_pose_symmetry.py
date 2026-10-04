"""pose_symmetry — 대칭축 둘레 twist 제거(swing 만 남김). numpy 만."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pose_symmetry import quat_axis_direction, quat_mul, remove_twist  # noqa: E402

Z = np.array([0.0, 0.0, 1.0])


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.r_[np.cos(h), np.sin(h) * a]


def test_pure_yaw_about_axis_becomes_identity():
    assert np.allclose(remove_twist(_rot(Z, 90), Z), [1, 0, 0, 0], atol=1e-9)
    assert np.allclose(remove_twist(_rot(Z, -170), Z), [1, 0, 0, 0], atol=1e-9)


def test_tilt_then_local_yaw_keeps_only_tilt():
    q = quat_mul(_rot([1, 0, 0], 30), _rot(Z, 45))      # swing ⊗ twist(local z)
    swing = remove_twist(q, Z)
    assert np.allclose(swing, _rot([1, 0, 0], 30), atol=1e-9)


def test_axis_direction_preserved_and_no_residual_twist():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.normal(size=4)
        q = v / np.linalg.norm(v)
        swing = remove_twist(q, Z)
        assert np.allclose(quat_axis_direction(swing, Z), quat_axis_direction(q, Z), atol=1e-9)
        assert abs(np.dot(swing[1:], Z)) < 1e-9          # z 성분 0 = z 둘레 twist 없음
        assert np.isclose(np.linalg.norm(swing), 1.0)


def test_flip_edge_case_and_other_axis():
    flip = _rot([1, 0, 0], 180)                           # w=0, z 성분 0 → twist 정의 불가 → 그대로
    assert np.allclose(remove_twist(flip, Z), flip, atol=1e-9)
    y = np.array([0.0, 1.0, 0.0])                         # cup CAD 는 Y-up
    assert np.allclose(remove_twist(_rot(y, 60), y), [1, 0, 0, 0], atol=1e-9)


def test_point_axis_up_flips_an_upside_down_cylinder_and_keeps_upright_ones():
    """10.04 arm4090: FP++ 가 cyl60 을 뒤집어 잡았다(축 기울기 179°) — 축이 아래 반구면 축에 수직한 축 둘레 180° 로 위로."""
    from pose_symmetry import point_axis_up
    up = _rot([1, 0, 0], 2.0)                              # 서 있는 컵(2° 기울기)은 그대로
    assert np.allclose(point_axis_up(up, Z), up, atol=1e-12)
    down = _rot([1, 0, 0], 179.2)                          # 뒤집힘 → 축이 위로, 기울기는 0.8°
    fixed = point_axis_up(down, Z)
    axis = quat_axis_direction(fixed, Z)
    assert axis[2] > 0 and np.degrees(np.arccos(np.clip(axis[2], -1, 1))) == pytest.approx(0.8, abs=1e-6)
    y = np.array([0.0, 1.0, 0.0])                          # 다른 대칭축(Y-up CAD): 아래를 향하면 위로
    assert quat_axis_direction(_rot([1, 0, 0], -100), y)[2] < 0
    assert quat_axis_direction(point_axis_up(_rot([1, 0, 0], -100), y), y)[2] > 0

#!/usr/bin/env python3
"""회전 대칭 물체의 pose 정규화 — 대칭축 둘레 twist 를 버리고 swing 만 남긴다.

원통(shaker·cup)은 축 둘레 회전(yaw)이 FP++ 추적기의 자유 방향이라 프레임마다 흘러간다.
q = swing ⊗ twist(축 둘레, 로컬 프레임) 로 분해해 twist 를 제거하면 축의 방향(기울기)은
그대로이고 축 둘레 회전은 항상 0 이 된다. 위치는 건드리지 않는다.

쿼터니언은 전부 wxyz. numpy 만. test_pose_symmetry.py 대상.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a ⊗ b (wxyz)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_axis_direction(q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """로컬 축 `axis` 가 q 로 회전된 방향 벡터."""
    v = np.r_[0.0, np.asarray(axis, float)]
    return quat_mul(quat_mul(q, v), quat_conj(q))[1:]


def point_axis_up(q: np.ndarray, axis: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """로컬 `axis` 가 아래 반구를 가리키면 축에 수직한 로컬 축 둘레로 180° 돌려 위로 향하게 한다(wxyz, 단위).

    위아래도 거의 대칭인 원통(cyl60)은 FP++ 가 뒤집어 잡는다(10.04 arm4090 기울기 179°). 테이블 위 컵은 서 있다는 가정에서만
    쓴다 — 90° 넘게 기운 컵(붓는 중)은 FP++ 가 아니라 손바닥 FK 로 본다. 위치는 건드리지 않는다(원점 = 중심).
    """
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    if np.dot(quat_axis_direction(q, a), np.asarray(up, float)) >= 0.0:
        return q
    perp = np.cross(a, [1.0, 0.0, 0.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(a, [0.0, 1.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    out = quat_mul(q, np.r_[0.0, perp])          # 로컬 perp 둘레 180°
    return out / np.linalg.norm(out)


def remove_twist(q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """q 에서 로컬 `axis` 둘레 twist 를 제거한 swing 쿼터니언(wxyz, 단위)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    proj = np.dot(q[1:], a) * a
    twist = np.r_[q[0], proj]
    norm = np.linalg.norm(twist)
    if norm < _EPS:
        return q                      # 축에 수직한 180° 회전 — twist 가 정의되지 않으니 그대로
    twist = twist / norm
    swing = quat_mul(q, quat_conj(twist))
    return swing / np.linalg.norm(swing)

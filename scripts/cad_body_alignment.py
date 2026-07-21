"""cup.obj(mesh) ↔ sim 컵 body 프레임 정합. 순수 numpy — ROS 불필요.

mesh 규약: Y축이 높이(측정: x/z=9cm, y=17.76cm), 원점 임의(바닥이 y=min).
body 규약: +z=위, 원점=컵 바닥 중심 (extrinsics yaml 주석).
따라서 T_cad_body = (mesh를 Y-up→Z-up 회전) 후 (바닥중심을 원점으로 이동).
"""
from __future__ import annotations
import numpy as np


def mesh_aabb(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo.copy()
    with open(obj_path) as f:
        for ln in f:
            if ln.startswith("v "):
                xyz = np.array([float(v) for v in ln.split()[1:4]])
                lo = np.minimum(lo, xyz)
                hi = np.maximum(hi, xyz)
    return lo, hi


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def cad_to_body_yup_to_zup(
    aabb_min: np.ndarray, aabb_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(pos_xyz, quat_wxyz): mesh(Y-up) → body(Z-up, 원점=바닥중심)."""
    # Y-up→Z-up: x축 기준 +90° 회전 (Y→Z, Z→-Y)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    quat = np.array([c, s, 0.0, 0.0])  # wxyz, Rx(+90°)
    # mesh 바닥면 중심 (x/z 중앙, y=min)
    bottom_center = np.array([(aabb_min[0] + aabb_max[0]) / 2,
                              aabb_min[1],
                              (aabb_min[2] + aabb_max[2]) / 2])
    # 회전 후 그 점이 원점에 오도록 평행이동: pos = -R·bottom_center
    def rot(q, v):
        qv = np.array([0.0, *v])
        qc = np.array([q[0], -q[1], -q[2], -q[3]])
        return _quat_mul(_quat_mul(q, qv), qc)[1:]
    pos = -rot(quat, bottom_center)
    return pos, quat

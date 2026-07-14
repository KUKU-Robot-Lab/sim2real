#!/usr/bin/env python3
"""OpenArm 오른팔 palm pose FK (pos + quat wxyz). numpy만 사용.

pour 라이브 s2r의 (6.3) 경로: pour 단계에서 소스 컵 pose를 비전 없이
`컵 = palm FK ∘ grasp offset`으로 추정한다. 이 모듈이 palm FK를 담당한다.

URDF 체인 상수는 hdgp `scripts/tools/openarm_fk.py`와 동일해야 하며,
`test_palm_fk.py`가 hdgp 모듈을 직접 import해 출력(위치·축방향)을 대조한다
(drift-guard). 위치 캘리브레이션 오프셋도 동일 방식(sim 실측 기준점)이다.

프레임: 반환 pose는 sim world = 로봇 base. quat은 Isaac Lab 관례 wxyz.
orientation 기준은 palm_link (Fabrics가 추종하는 프레임), 위치는 palm_center.
"""

from __future__ import annotations

import math

import numpy as np


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


def _make_T(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rot(*rpy)
    T[:3, 3] = xyz
    return T


def _rot_axis(axis: np.ndarray, q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + s * K + (1 - c) * (K @ K)


# --- openarm_tesollo.urdf 관절 체인 (hdgp openarm_fk.py와 동일, drift-guard 감시) ---
ARM_JOINTS = [
    ([0.0, -0.0935, 0.6979998], [1.5708, 0.0, 0.0], [0, 0, 1]),          # j1
    ([-0.0301, 0.0, 0.06], [-1.5707963, 0.0, 3.1415927], [1, 0, 0]),     # j2
    ([-0.0301, 0.0, 0.06625], [0.0, 0.0, 3.1415927], [0, 0, 1]),         # j3
    ([0.0, 0.0315, 0.15375], [0.0, 0.0, 0.0], [0, 1, 0]),                # j4
    ([0.0, -0.0315, 0.0955], [0.0, 0.0, 0.0], [0, 0, 1]),                # j5
    ([0.0375, 0.0, 0.1205], [0.0, 0.0, 0.0], [1, 0, 0]),                 # j6
    ([-0.0375, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 1, 0]),                   # j7
]
T_PALM_LINK = _make_T([0.0, 0.0000003, 0.1333695], [0.0, 0.0, -1.5707964])
T_PALM_CENTER = _make_T([0.0, 0.03, 0.04], [0.0, 0.0, 0.0])

# 위치 캘리브레이션 (hdgp와 동일): FK_raw(Q_REF) - SIM_POS_REF
Q_REF = [0.5, 0.5, -0.6, 0.7, 0.0, 0.0, 1.0]
SIM_POS_REF = np.array([0.281, -0.270, 0.424])


def _fk_T_palm(q7) -> np.ndarray:
    """base → palm_link 4×4 변환."""
    q = list(q7)
    if len(q) != 7:
        raise ValueError(f"q must have 7 elements, got {len(q)}")
    T = np.eye(4)
    for (xyz, rpy, axis), qi in zip(ARM_JOINTS, q):
        T = T @ _make_T(xyz, rpy)
        R = np.eye(4)
        R[:3, :3] = _rot_axis(np.array(axis, dtype=float), qi)
        T = T @ R
    return T @ T_PALM_LINK


def _compute_offset() -> np.ndarray:
    T = _fk_T_palm(Q_REF)
    palm_center_raw = (T @ T_PALM_CENTER)[:3, 3]
    return palm_center_raw - SIM_POS_REF


_OFFSET = _compute_offset()


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3×3 회전행렬 → 쿼터니언 wxyz (Shepperd)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def palm_pose(q7, apply_offset: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """팔 관절 7개 → (palm_center 위치(3), palm_link 자세 quat wxyz(4))."""
    T = _fk_T_palm(q7)
    pos = (T @ T_PALM_CENTER)[:3, 3]
    if apply_offset:
        pos = pos - _OFFSET
    return pos, rot_to_quat_wxyz(T[:3, :3])


def quat_inverse_wxyz(q: np.ndarray) -> np.ndarray:
    """단위 쿼터니언 역원 (conjugate)."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def grasp_offset_from_snapshot(
    palm_pos: np.ndarray,
    palm_quat: np.ndarray,
    cup_pos: np.ndarray,
    cup_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """잡는 순간의 palm pose + 컵 pose(비전) 스냅샷 → grasp offset.

    offset = T_palm⁻¹ ∘ T_cup (palm 프레임에서 본 컵). 이후 pour 단계에서
    `pour_obs_builder.compose_pose(palm_pos, palm_quat, *offset)`로 컵 pose 복원.
    """
    from pour_obs_builder import quat_mul  # 순환 없음: builder는 palm_fk에 무의존

    q_inv = quat_inverse_wxyz(palm_quat)
    from pour_obs_geometry import quat_apply

    rel_pos = quat_apply(q_inv, np.asarray(cup_pos, dtype=np.float64) - np.asarray(palm_pos, dtype=np.float64))
    rel_quat = quat_mul(q_inv, np.asarray(cup_quat, dtype=np.float64))
    return rel_pos, rel_quat


def extract_joints(
    names: list[str], values: list[float], order: tuple[str, ...] | list[str]
) -> np.ndarray:
    """JointState(name, value) → 지정한 관절 순서의 배열. 누락 시 즉시 에러.

    /isaacsim/joint_states 처럼 여러 소스가 병합된 메시지에서 canonical 순서로
    proprio를 뽑을 때 사용. 조용한 0-채움은 obs를 오염시키므로 하지 않는다.
    """
    index = {n: i for i, n in enumerate(names)}
    missing = [j for j in order if j not in index]
    if missing:
        raise KeyError(f"joints missing from state: {missing}")
    return np.array([values[index[j]] for j in order], dtype=np.float64)

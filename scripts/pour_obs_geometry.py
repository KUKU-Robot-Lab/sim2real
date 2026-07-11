#!/usr/bin/env python3
"""pour 정책 obs 중 '컵 pose 파생' 항목을 실물에서 재현하기 위한 순수 지오메트리.

pour actor obs(55D)에서 로봇 엔코더로 못 채우는 것은 두 컵(소스/타깃)의 6-DOF
pose에서 나오는 4개 항목뿐이다:

    pour_point_to_opening (3), source_pour_axis (3), source_up_axis (3), target_up_axis (3)

이 모듈은 hdgp `tesollo/right/pour_v1` env의 지오메트리(pour_right_env.py 1463~1640)를
Isaac/torch 없이 numpy로 그대로 포팅한다. 상수는 pour_v1 preset에서 가져왔고,
`tests/test_pour_obs_geometry.py`가 실제 preset 값과 drift 나는지 감시한다.

좌표계: 모든 pose는 sim world = 로봇 base 프레임. 쿼터니언은 Isaac Lab 관례 wxyz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- pour_v1 preset에서 가져온 컵 body-frame 상수 (drift는 테스트가 감시) ---
SOURCE_CUP_POUR_POINT_POS_B = (0.0, 0.0, 0.100)  # 컵 중앙→림(입구) 오프셋
TARGET_CUP_OPENING_POS_B = (0.0, 0.0, 0.100)
SOURCE_CUP_POUR_AXIS_B = (1.0, 0.0, 0.0)
SOURCE_CUP_UP_AXIS_B = (0.0, 0.0, 1.0)
TARGET_CUP_UP_AXIS_B = (0.0, 0.0, 1.0)
SOURCE_OUTER_RADIUS = 0.045
POUR_POINT_DYN_LO = 0.15  # ≈45°
POUR_POINT_DYN_HI = 0.30  # ≈67°


def quat_apply(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Isaac Lab quat_apply (wxyz)로 벡터 회전. quat/vec은 (...,4)/(...,3)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    qvec = q[..., 1:4]
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)


@dataclass(frozen=True)
class CupGeometry:
    """두 컵의 body-frame 지오메트리 + pour_point 블렌드 파라미터."""

    source_pour_point_pos_b: tuple[float, float, float] = SOURCE_CUP_POUR_POINT_POS_B
    target_opening_pos_b: tuple[float, float, float] = TARGET_CUP_OPENING_POS_B
    source_pour_axis_b: tuple[float, float, float] = SOURCE_CUP_POUR_AXIS_B
    source_up_axis_b: tuple[float, float, float] = SOURCE_CUP_UP_AXIS_B
    target_up_axis_b: tuple[float, float, float] = TARGET_CUP_UP_AXIS_B
    source_outer_radius: float = SOURCE_OUTER_RADIUS
    pour_point_dyn_lo: float = POUR_POINT_DYN_LO
    pour_point_dyn_hi: float = POUR_POINT_DYN_HI


def _normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def target_opening_w(
    target_pos_w: np.ndarray, target_quat_w: np.ndarray, geom: CupGeometry
) -> np.ndarray:
    return np.asarray(target_pos_w, dtype=np.float64) + quat_apply(
        target_quat_w, np.array(geom.target_opening_pos_b)
    )


def source_pour_point_w(
    source_pos_w: np.ndarray,
    source_quat_w: np.ndarray,
    target_opening: np.ndarray,
    geom: CupGeometry,
) -> np.ndarray:
    """소스 컵의 최하단 배출점(pour_v1 env 1463~1628 포팅).

    림 중심에서 '가장 낮은 림' 방향으로 컵 반경만큼 이동한 점. xy 방향은 이송(정적,
    타깃 방향)↔붓기(동적, 중력수직 배출구)를 tilt 깊이 smoothstep으로 blend한다.
    """
    source_pos_w = np.asarray(source_pos_w, dtype=np.float64)
    rim_center = source_pos_w + quat_apply(
        source_quat_w, np.array(geom.source_pour_point_pos_b)
    )
    cup_up = quat_apply(source_quat_w, np.array(geom.source_up_axis_b))

    world_down = np.zeros_like(cup_up)
    world_down[..., 2] = -1.0
    dot = np.sum(world_down * cup_up, axis=-1, keepdims=True)
    gravity_perp = world_down - dot * cup_up
    gravity_perp_hat = _normalize(gravity_perp)

    perp_xy_mag = np.linalg.norm(gravity_perp_hat[..., :2], axis=-1, keepdims=True)

    pour_dir_xy = np.asarray(target_opening)[..., :2] - rim_center[..., :2]
    static_dir_hat = _normalize(pour_dir_xy)
    dynamic_dir_hat = gravity_perp_hat[..., :2] / np.maximum(perp_xy_mag, 1e-6)

    su_dot = np.clip(cup_up[..., 2], -1.0, 1.0)
    tilt_amt = np.clip((1.0 - su_dot) / 2.0, 0.0, 1.0)
    span = max(geom.pour_point_dyn_hi - geom.pour_point_dyn_lo, 1e-6)
    dyn_t = np.clip((tilt_amt - geom.pour_point_dyn_lo) / span, 0.0, 1.0)
    dyn_w = (dyn_t * dyn_t * (3.0 - 2.0 * dyn_t))[..., None]  # smoothstep

    blended = (1.0 - dyn_w) * static_dir_hat + dyn_w * dynamic_dir_hat
    pour_dir_hat = _normalize(blended)

    pp_xy = rim_center[..., :2] + geom.source_outer_radius * perp_xy_mag * pour_dir_hat
    pp_z = rim_center[..., 2:3] + geom.source_outer_radius * gravity_perp_hat[..., 2:3]
    return np.concatenate([pp_xy, pp_z], axis=-1)


def vision_obs_terms(
    source_pos_w: np.ndarray,
    source_quat_w: np.ndarray,
    target_pos_w: np.ndarray,
    target_quat_w: np.ndarray,
    geom: CupGeometry | None = None,
) -> dict[str, np.ndarray]:
    """두 컵 6-DOF pose → pour actor obs의 컵 파생 4항목(총 12값).

    반환 dict: pour_point_to_opening(3), source_pour_axis(3),
    source_up_axis(3), target_up_axis(3). obs 조립 순서와 동일.
    """
    geom = geom or CupGeometry()
    opening = target_opening_w(target_pos_w, target_quat_w, geom)
    pour_point = source_pour_point_w(source_pos_w, source_quat_w, opening, geom)
    return {
        "pour_point_to_opening": opening - pour_point,
        "source_pour_axis": quat_apply(source_quat_w, np.array(geom.source_pour_axis_b)),
        "source_up_axis": quat_apply(source_quat_w, np.array(geom.source_up_axis_b)),
        "target_up_axis": quat_apply(target_quat_w, np.array(geom.target_up_axis_b)),
    }


def vision_obs_vector(
    source_pos_w: np.ndarray,
    source_quat_w: np.ndarray,
    target_pos_w: np.ndarray,
    target_quat_w: np.ndarray,
    geom: CupGeometry | None = None,
) -> np.ndarray:
    """vision_obs_terms를 actor obs 순서대로 이어붙인 12-벡터."""
    t = vision_obs_terms(source_pos_w, source_quat_w, target_pos_w, target_quat_w, geom)
    return np.concatenate(
        [
            t["pour_point_to_opening"],
            t["source_pour_axis"],
            t["source_up_axis"],
            t["target_up_axis"],
        ],
        axis=-1,
    )

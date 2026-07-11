#!/usr/bin/env python3
"""pour_v1 actor obs(55D)를 실물 입력으로 조립하는 순수 로직 (numpy, Isaac 불필요).

obs 레이아웃 (tesollo/right/pour_v1, pour_right_env.py `_get_observations`):

    arm_joint_pos          7    로봇 팔 엔코더
    arm_joint_vel          7    로봇 팔 엔코더
    finger_grasp_progress  5    손 엔코더 → open/grasp 사이 진행도
    left_arm_joint_pos     9    좌팔+그리퍼 엔코더 (한팔 세팅이면 0)
    left_arm_joint_vel     9    좌팔+그리퍼 엔코더 (한팔 세팅이면 0)
    pour_point_to_opening  3    컵 pose 파생 (pour_obs_geometry)
    source_pour_axis       3    컵 pose 파생
    source_up_axis         3    컵 pose 파생
    target_up_axis         3    컵 pose 파생
    last_palm_actions      6    직전 action[:6]
    ------------------------------------------------
    합계                   55

(6.3) pour 단계에서 소스 컵 pose는 손 FK ∘ grasp offset으로 얻는다 → `compose_pose`.
컵 pose → 비전 obs 4항목은 `pour_obs_geometry.vision_obs_vector`가 담당한다.
"""

from __future__ import annotations

import numpy as np

from pour_obs_geometry import CupGeometry, quat_apply, vision_obs_vector

ACTOR_OBS_DIM = 55
NUM_FINGERTIPS = 5
NUM_HAND_DOF = 20
NUM_PALM_ACTION = 6

# pour_v1 preset (drift는 test가 감시). 손가락당 4관절 × 5손가락 = 20.
HAND_APPROACH_POSE = (
    0.0, -1.57, -0.5, 0.0,   # thumb
    0.0, 0.0, 0.0, 0.0,      # index
    0.0, 0.0, 0.0, 0.0,      # middle
    0.0, 0.0, 0.0, 0.0,      # ring
    0.0, 0.0, 0.0, 0.0,      # pinky
)
HAND_GRASP_POSE = (
    0.0, -1.57, 1.5, 1.5,    # thumb
    0.0, 1.6, 1.5, 1.5,      # index
    0.0, 1.6, 1.5, 1.5,      # middle
    0.0, 1.6, 1.5, 1.5,      # ring
    0.0, 0.0, 1.5, 1.5,      # pinky
)


def finger_grasp_progress(
    finger_pos_20: np.ndarray,
    approach: tuple[float, ...] = HAND_APPROACH_POSE,
    grasp: tuple[float, ...] = HAND_GRASP_POSE,
) -> np.ndarray:
    """20-DOF 손 관절 → 손가락별 grasp 진행도 (5,), [0,1].

    관절별 (pos-approach)/(grasp-approach)를 clamp 후, 손가락당 유효 관절만 평균.
    (env `_finger_grasp_progress` 포팅.)
    """
    pos = np.asarray(finger_pos_20, dtype=np.float64)
    if pos.shape[-1] != NUM_HAND_DOF:
        raise ValueError(f"expected {NUM_HAND_DOF} finger joints, got {pos.shape[-1]}")
    a = np.asarray(approach, dtype=np.float64)
    g = np.asarray(grasp, dtype=np.float64)
    delta = g - a
    valid = np.abs(delta) > 1e-6
    denom = np.where(valid, delta, 1.0)
    progress = np.clip((pos - a) / denom, 0.0, 1.0) * valid
    progress_5x4 = progress.reshape(*progress.shape[:-1], NUM_FINGERTIPS, 4)
    valid_counts = np.clip(valid.reshape(NUM_FINGERTIPS, 4).sum(axis=-1), 1, None)
    return progress_5x4.sum(axis=-1) / valid_counts


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """쿼터니언 곱 q1 ⊗ q2 (wxyz)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def compose_pose(
    parent_pos: np.ndarray,
    parent_quat: np.ndarray,
    child_pos: np.ndarray,
    child_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """T_base_child = T_base_parent ∘ T_parent_child.

    FK 컵 pose용: parent=palm(base 프레임), child=grasp offset(palm 프레임에서 본 컵).
    반환 (pos_w, quat_w).
    """
    pos = np.asarray(parent_pos, dtype=np.float64) + quat_apply(parent_quat, child_pos)
    quat = quat_mul(np.asarray(parent_quat, dtype=np.float64), np.asarray(child_quat, dtype=np.float64))
    return pos, quat


def assemble_actor_obs(
    arm_joint_pos: np.ndarray,       # (7,)
    arm_joint_vel: np.ndarray,       # (7,)
    finger_joint_pos: np.ndarray,    # (20,)
    left_arm_joint_pos: np.ndarray,  # (9,)  한팔이면 zeros
    left_arm_joint_vel: np.ndarray,  # (9,)
    source_cup_pos: np.ndarray,      # (3,)
    source_cup_quat: np.ndarray,     # (4,) wxyz
    target_cup_pos: np.ndarray,      # (3,)
    target_cup_quat: np.ndarray,     # (4,) wxyz
    last_palm_actions: np.ndarray,   # (6,)
    geom: CupGeometry | None = None,
) -> np.ndarray:
    """실물 입력 → pour_v1 actor obs (55,). 입력은 canonical 순서라고 가정."""
    arm_p = np.asarray(arm_joint_pos, dtype=np.float64).reshape(-1)
    arm_v = np.asarray(arm_joint_vel, dtype=np.float64).reshape(-1)
    la_p = np.asarray(left_arm_joint_pos, dtype=np.float64).reshape(-1)
    la_v = np.asarray(left_arm_joint_vel, dtype=np.float64).reshape(-1)
    palm_a = np.asarray(last_palm_actions, dtype=np.float64).reshape(-1)
    for name, arr, n in [
        ("arm_joint_pos", arm_p, 7), ("arm_joint_vel", arm_v, 7),
        ("left_arm_joint_pos", la_p, 9), ("left_arm_joint_vel", la_v, 9),
        ("last_palm_actions", palm_a, NUM_PALM_ACTION),
    ]:
        if arr.shape[0] != n:
            raise ValueError(f"{name} expected {n}, got {arr.shape[0]}")

    grasp_prog = finger_grasp_progress(finger_joint_pos)          # (5,)
    vision = vision_obs_vector(                                    # (12,)
        source_cup_pos, source_cup_quat, target_cup_pos, target_cup_quat, geom
    )
    obs = np.concatenate([arm_p, arm_v, grasp_prog, la_p, la_v, vision, palm_a])
    if obs.shape[0] != ACTOR_OBS_DIM:
        raise RuntimeError(f"actor obs dim {obs.shape[0]} != {ACTOR_OBS_DIM}")
    return obs

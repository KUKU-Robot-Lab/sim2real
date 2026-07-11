#!/usr/bin/env python3
"""pour_v1 action(12D) → palm_link pose target 디코더. numpy 순수, Isaac 불필요.

hdgp `tesollo/right/pour_v1` env의 `_pre_physics_step` 비-warmstart 경로를,
**기본 config**(pour_action_mode="b_trajectory", pour_approach_pivot="palm",
pour_spout_z_lock=True, pour_orient_release=True) 기준으로 포팅했다.
상수 기본값은 `test_pour_action_decoder.py`의 drift-guard가 env_cfg와 대조한다.

입력(매 스텝):
    action(12,) ∈[-1,1]  — [0:6] palm(xyz + spin/β/ortho), [6] nullspace α, [7:12] hand
    소스 컵 pose, 타깃 컵 pose (base 프레임; pour 단계에선 palm FK ∘ grasp offset)
    palm_center 위치·palm_link quat (FK)

출력:
    palm_pose_target(7,) = [pos(3), quat xyzw(4)] — Fabrics set_features("quaternion")용
    nullspace α(EMA), hand action(5,) — 호출측이 hand lerp/Fabrics default_config에 사용

스텝 간 상태(PourDecoderState): EMA action, pour_ready 래치, spout offset 동결.
env과 동일하게 매 정책 스텝마다 decode()를 호출해야 상태가 정합한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pour_obs_builder import quat_mul
from pour_obs_geometry import CupGeometry, quat_apply, source_pour_point_w, target_opening_w

# --- pour_v1 env_cfg 기본값 (drift-guard 감시) ---
PALM_DELTA_XYZ = 0.03
PALM_DELTA_ROT_DEG = 15.0
EMA_ACTION_ALPHA = 0.7
TILT_GATE_XY_NEAR = 0.06
TILT_GATE_XY_FAR = 0.25
BETA_ACTION_INDEX = 4
BETA_TARGET_TILT_AMOUNT = 0.854
BETA_TILT_KP = 3.0
BETA_TILT_MAX_STEP = 0.06
POUR_Z_MARGIN = 0.03
TARGET_INNER_RADIUS = 0.041
POUR_CORRIDOR_XY_MARGIN = 0.015
POUR_CORRIDOR_Z_MIN = -0.02
POUR_CORRIDOR_Z_MAX = 0.12
POUR_CORRIDOR_SCALE = 20.0
READY_LATCH_THRESHOLD = 0.60
MAX_POSE_ANGLE = 45.0
PALM_EE_OFFSET_LOCAL = (0.028, 0.0, 0.04)  # palm_link→palm_ee (URDF palm_link_to_ee)

_D = math.pi / 180.0
# pour_v1 preset palm workspace (palm_ee 기준 위치 박스 + 각도 범위)
PALM_POSE_MINS = (
    -0.30, -0.55, 0.10,
    (90.0 - MAX_POSE_ANGLE) * _D, (0.0 - MAX_POSE_ANGLE) * _D, (90.0 - MAX_POSE_ANGLE) * _D,
)
PALM_POSE_MAXS = (
    0.65, 0.25, 0.68,
    (90.0 + MAX_POSE_ANGLE) * _D, (0.0 + MAX_POSE_ANGLE) * _D, (90.0 + MAX_POSE_ANGLE) * _D,
)


def _safe_normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    fb = fallback / max(np.linalg.norm(fallback), 1e-6)
    return vec / max(n, 1e-6) if n > 1e-6 else fb


def _scale(a: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """[-1,1] → [lo,hi] 선형."""
    return 0.5 * (a + 1.0) * (hi - lo) + lo


def quat_from_rotvec_wxyz(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    half = angle / 2.0
    return np.concatenate([[math.cos(half)], math.sin(half) * axis])


def pour_corridor_score(
    pour_point: np.ndarray, target_opening: np.ndarray
) -> float:
    """corridor 내부=1, 밖 초과거리에 exp penalty (env utils 포팅)."""
    radius = TARGET_INNER_RADIUS + POUR_CORRIDOR_XY_MARGIN
    delta = np.asarray(pour_point) - np.asarray(target_opening)
    xy_excess = max(float(np.hypot(delta[0], delta[1])) - radius, 0.0)
    z_rel = float(delta[2])
    z_excess = max(POUR_CORRIDOR_Z_MIN - z_rel, 0.0) + max(z_rel - POUR_CORRIDOR_Z_MAX, 0.0)
    return math.exp(-POUR_CORRIDOR_SCALE * math.hypot(xy_excess, z_excess))


def build_cup_local_tilt_rotvec(
    delta_local: np.ndarray,
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    geom: CupGeometry,
) -> np.ndarray:
    """[spin, tilt-toward, ortho] 로컬 명령 → world rotvec (env 포팅)."""
    pp = np.asarray(source_pos) + quat_apply(source_quat, np.array(geom.source_pour_point_pos_b))
    opening = target_opening_w(target_pos, target_quat, geom)
    cup_up = quat_apply(source_quat, np.array(geom.source_up_axis_b))
    cup_pour = quat_apply(source_quat, np.array(geom.source_pour_axis_b))

    mouth_delta = opening - pp
    target_dir = mouth_delta - np.dot(mouth_delta, cup_up) * cup_up
    pour_plane = cup_pour - np.dot(cup_pour, cup_up) * cup_up
    target_dir = _safe_normalize(target_dir, pour_plane)

    tilt_toward = _safe_normalize(np.cross(target_dir, cup_up), np.cross(pour_plane, cup_up))
    tilt_ortho = _safe_normalize(np.cross(cup_up, tilt_toward), pour_plane)
    spin = _safe_normalize(cup_up, np.array([0.0, 0.0, 1.0]))

    return delta_local[0] * spin + delta_local[1] * tilt_toward + delta_local[2] * tilt_ortho


@dataclass
class PourDecoderState:
    """스텝 간 유지되는 디코더 상태. 에피소드 시작 시 새로 만든다."""

    ema_palm: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ema_null: float = 0.0
    pour_ready_latched: bool = False
    spout_offset_body: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(frozen=True)
class PalmTarget:
    """decode() 결과."""

    pos: np.ndarray          # (3,) palm_link 위치 target
    quat_xyzw: np.ndarray    # (4,) palm_link 자세 target (Fabrics "quaternion" 형식)
    null_alpha: float        # EMA nullspace α ∈[-1,1]
    hand_action: np.ndarray  # (5,) per-finger lerp 명령 (원본 그대로)
    ready: bool              # pour_ready 래치 상태 (진단)


def decode(
    action: np.ndarray,
    state: PourDecoderState,
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    palm_center_pos: np.ndarray,
    palm_quat_wxyz: np.ndarray,
    geom: CupGeometry | None = None,
) -> PalmTarget:
    """action 12D → palm_link pose target. state는 in-place 갱신."""
    geom = geom or CupGeometry()
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] != 12:
        raise ValueError(f"expected 12D action, got {action.shape[0]}")
    palm_action = np.clip(action[:6], -1.0, 1.0)
    alpha_action = float(np.clip(action[6], -1.0, 1.0))
    hand_action = np.clip(action[7:12], -1.0, 1.0)

    # EMA smoothing (env과 동일: raw는 로깅용, Fabrics에는 EMA 전달)
    state.ema_palm = EMA_ACTION_ALPHA * palm_action + (1.0 - EMA_ACTION_ALPHA) * state.ema_palm
    state.ema_null = EMA_ACTION_ALPHA * alpha_action + (1.0 - EMA_ACTION_ALPHA) * state.ema_null

    # 컵 지오메트리 (env-local = base 프레임)
    opening = target_opening_w(target_pos, target_quat, geom)
    pp = source_pour_point_w(source_pos, source_quat, opening, geom)
    mouth_xy_distance = float(np.hypot(*(opening[:2] - pp[:2])))

    # ready 래치 (corridor score) + spout offset 동결
    if not state.pour_ready_latched:
        state.pour_ready_latched = (
            pour_corridor_score(pp, opening) >= READY_LATCH_THRESHOLD
        )

    # delta 스케일 + tilt gate
    d_rad = math.radians(PALM_DELTA_ROT_DEG)
    lo = np.array([-PALM_DELTA_XYZ] * 3 + [-d_rad] * 3)
    hi = -lo
    delta = _scale(state.ema_palm, lo, hi)
    gate_den = max(TILT_GATE_XY_FAR - TILT_GATE_XY_NEAR, 1e-6)
    tilt_gate = float(np.clip((TILT_GATE_XY_FAR - mouth_xy_distance) / gate_den, 0.0, 1.0))

    # β-trajectory: action[4]를 목표 tilt_amount로 해석 → 피드백 스텝
    su = quat_apply(source_quat, np.array(geom.source_up_axis_b))
    tu = quat_apply(target_quat, np.array(geom.target_up_axis_b))
    rim_antiparallel = float(np.clip(np.dot(su, tu), -1.0, 1.0))
    beta = float(np.clip(state.ema_palm[BETA_ACTION_INDEX] * 0.5 + 0.5, 0.0, 1.0))
    target_ta = beta * BETA_TARGET_TILT_AMOUNT
    cur_ta = float(np.clip((1.0 - rim_antiparallel) / 2.0, 0.0, 1.0))
    delta[4] = float(np.clip(BETA_TILT_KP * (target_ta - cur_ta), -BETA_TILT_MAX_STEP, BETA_TILT_MAX_STEP))
    delta[3:6] *= tilt_gate

    rotvec_world = build_cup_local_tilt_rotvec(
        delta[3:6], source_pos, source_quat, target_pos, target_quat, geom
    )
    delta_quat_wxyz = quat_from_rotvec_wxyz(rotvec_world)

    mins = np.array(PALM_POSE_MINS[:3])
    maxs = np.array(PALM_POSE_MAXS[:3])
    ee_off = np.array(PALM_EE_OFFSET_LOCAL)
    rim_rel = pp - palm_center_pos

    # --- approach 경로 (pivot="palm" + z_lock) ---
    spout_z_target = opening[2] + POUR_Z_MARGIN
    palm_ee = palm_center_pos + delta[:3]
    # z_lock: 주둥이 z를 target 입구 + margin으로 강제 → palm z로 환산
    palm_ee[2] = spout_z_target - quat_apply(delta_quat_wxyz, rim_rel)[2]
    palm_ee = np.clip(palm_ee, mins, maxs)
    # orientation: current palm quat(wxyz→xyzw)에 world delta 좌곱
    tgt_quat_wxyz = quat_mul(delta_quat_wxyz, np.asarray(palm_quat_wxyz, dtype=np.float64))
    # palm_ee → palm_link 역변환 (orientation 공유, origin만 R·offset 차감)
    pos = palm_ee - quat_apply(tgt_quat_wxyz, ee_off)
    quat_xyzw = tgt_quat_wxyz[[1, 2, 3, 0]]

    # --- ready 후: orientation release + spout hold (B-light) ---
    R_cur = np.asarray(palm_quat_wxyz, dtype=np.float64)
    off_now = quat_apply(np.array([R_cur[0], -R_cur[1], -R_cur[2], -R_cur[3]]), rim_rel)
    if not state.pour_ready_latched:
        state.spout_offset_body = off_now
    else:
        spout_target = pp + delta[:3]
        spout_target[2] = spout_z_target
        palm_ee_bl = spout_target - quat_apply(R_cur, state.spout_offset_body)
        palm_ee_bl = np.clip(palm_ee_bl, mins, maxs)
        pos = palm_ee_bl - quat_apply(R_cur, ee_off)
        quat_xyzw = R_cur[[1, 2, 3, 0]]

    return PalmTarget(
        pos=pos,
        quat_xyzw=quat_xyzw,
        null_alpha=float(state.ema_null),
        hand_action=hand_action,
        ready=state.pour_ready_latched,
    )

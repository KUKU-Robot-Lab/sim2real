"""joint family 관측 — 측정 + FK + 디코더 상태 + 물체 · 목표 자세 → 133 벡터. 순수 numpy.

hdgp fj_kp_env.py `_get_observations`(학습 커밋 d38b4346) 의 `clean` 벡터와 같은 순서 · 같은 식이다.
학습에만 있는 것(관절 · 물체 노이즈, 0..2 스텝 관측 지연, 물체 0..9 스텝 지연)은 넣지 않는다.

  arm_q · arm_qd        측정 팔 관절 (arm_joints 순)
  hand_q · hand_qd      측정 손 관절 (**hand_obs_order** 순 — 시뮬레이터 관절 순)
  palm_pos              palm 원점, 로봇 base(= env-local) 프레임
  palm_ax               palm 회전행렬의 0열 · 1열 (base 축)
  tips_rel_palm         손끝 − palm (base 축, 손끝 순서 = tip_bodies)
  cmd_state             팔 q* (디코더)
  kp_rel_palm           물체 키포인트 4개 − palm
  kp_rel_goal           물체 키포인트 − 목표 키포인트
  action_arm            직전 행동의 팔 7 (클립)
  action_hand           직전 손 q* 정규화 2(q*−lo)/(hi−lo)−1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .joint_contract import JointContract, obs_layout


class JointObsError(ValueError):
    pass


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray        # (3,) base 프레임
    quat: np.ndarray       # (4,) wxyz


@dataclass(frozen=True)
class JointState:
    joint_pos: Mapping     # 이름 → rad (팔 7 + 손 hand_joints 전부)
    joint_vel: Mapping     # 이름 → rad/s
    palm_pos: np.ndarray   # (3,)
    palm_rot: np.ndarray   # (3, 3)
    tips: np.ndarray       # (n_tips, 3), tip_bodies 순
    obj: Pose
    goal: Pose


def quat_to_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(n) or n < 1e-9:
        raise JointObsError("zero or non-finite quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def keypoints(c: JointContract, pose: Pose) -> np.ndarray:
    """(4, 3) = pos + R · (half_height · unit) — hdgp keypoint_goal.keypoints_world."""
    off = np.asarray(c.keypoint_axial_unit, float) * float(c.keypoint_half_height)
    return np.asarray(pose.pos, float)[None, :] + off @ quat_to_matrix(pose.quat).T


def _joints(values: Mapping, names, what: str) -> np.ndarray:
    missing = [n for n in names if n not in values]
    if missing:
        raise JointObsError(f"{what}: missing {missing}")
    return np.array([float(values[n]) for n in names])


def build_obs(c: JointContract, s: JointState, cmd_arm: np.ndarray, action_arm: np.ndarray,
              action_hand: np.ndarray) -> np.ndarray:
    palm = np.asarray(s.palm_pos, float)
    R = np.asarray(s.palm_rot, float)
    tips = np.asarray(s.tips, float)
    if tips.shape != (len(c.tip_bodies), 3):
        raise JointObsError(f"tips shape {tips.shape} != ({len(c.tip_bodies)}, 3)")
    kp_obj = keypoints(c, s.obj)
    parts = {
        "arm_q": _joints(s.joint_pos, c.arm_joints, "arm_q"),
        "arm_qd": _joints(s.joint_vel, c.arm_joints, "arm_qd"),
        "hand_q": _joints(s.joint_pos, c.hand_obs_order, "hand_q"),
        "hand_qd": _joints(s.joint_vel, c.hand_obs_order, "hand_qd"),
        "palm_pos": palm,
        "palm_ax": np.concatenate([R[:, 0], R[:, 1]]),
        "tips_rel_palm": (tips - palm[None, :]).reshape(-1),
        "cmd_state": np.asarray(cmd_arm, float),
        "kp_rel_palm": (kp_obj - palm[None, :]).reshape(-1),
        "kp_rel_goal": (kp_obj - keypoints(c, s.goal)).reshape(-1),
        "action_arm": np.asarray(action_arm, float),
        "action_hand": np.asarray(action_hand, float),
    }
    out = []
    for name, width in obs_layout(c.n_arm, c.n_hand, len(c.tip_bodies)):
        v = np.asarray(parts[name], float).reshape(-1)
        if v.size != width:
            raise JointObsError(f"obs segment {name}: {v.size} values, want {width}")
        out.append(v)
    obs = np.concatenate(out)
    if obs.size != c.obs_dim or not np.all(np.isfinite(obs)):
        raise JointObsError(f"obs must be {c.obs_dim} finite values, got {obs.size}")
    return obs


def clip_obs(c: JointContract, obs: np.ndarray) -> np.ndarray:
    """rl_games 래퍼의 clip_observations — 네트워크(정규화) 앞에서 적용한다(학습과 같다)."""
    if c.obs_clip is None:
        return np.asarray(obs, float)
    return np.clip(np.asarray(obs, float), -c.obs_clip, c.obs_clip)


def first_goal(c: JointContract, obj: Pose, offset=None) -> Pose:
    """첫 목표 = 리셋 때 물체 위치 + offset(기본 = 계약의 학습 분포 가운데), 목표 상자로 clip, 자세는 리셋 때 물체 자세.

    학습(keypoint_goal.py `sample_first_goal`)은 가로 ±goal_first_xy_range · 높이 goal_first_z_range 에서 뽑는다.
    배포는 한 점을 쓴다 — 가운데가 기본이고 운영자가 offset 으로 바꿀 수 있다.
    """
    off = np.asarray(c.goal_offset if offset is None else offset, float)
    pos = np.clip(np.asarray(obj.pos, float) + off, c.goal_box_lo, c.goal_box_hi)
    return Pose(pos, np.asarray(obj.quat, float).copy())       # hdgp: goal_quat = settled_quat(기울임 0)

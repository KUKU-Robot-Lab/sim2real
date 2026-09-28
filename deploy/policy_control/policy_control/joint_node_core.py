"""joint_node 의 ROS 없는 절반: 소스 스냅샷 → JointMeasure, JointStep → pd 가 받는 joint_target 한 벌.

측정은 없는 값을 지어내지 않는다 — 팔 · 손 · 물체 중 하나라도 없거나 stale 이면 거부한다.
joint_target 은 팔 7 + 손 hand_joints + 용접 관절(용접 각도)이다. 실기 손에는 용접 관절이 있고 드라이버는
20 관절을 한꺼번에 받으므로, 학습에서 움직이지 않은 그 관절을 그 각도로 붙잡아 보낸다.
"""
from __future__ import annotations

import numpy as np

from .joint_chain import JointMeasure, JointStep
from .joint_contract import JointContract
from .joint_obs import Pose


class JointNodeError(RuntimeError):
    pass


def measure_from_state(c: JointContract, st, arm_names) -> JointMeasure:
    """sources.RobotState(한 팔) → JointMeasure. 결손 · stale · 속도 없음 · 물체 없음이면 JointNodeError."""
    bad = set(st.stale) | set(st.missing)
    for need in ("arm", "ee", "object"):
        if need in bad:
            raise JointNodeError(f"source {need!r} is {'stale' if need in st.stale else 'missing'}")
    if st.arm_q is None or st.arm_qd is None:
        raise JointNodeError("arm source has no position/velocity (arm_qd is an actor input)")
    if st.ee_q is None or st.ee_qd is None:
        raise JointNodeError("hand source has no position/velocity (hand_qd is an actor input)")
    if st.object_pos is None or st.object_quat is None:
        raise JointNodeError("no object pose")
    pos = {**dict(zip(arm_names, st.arm_q)), **dict(zip(st.ee_names, st.ee_q))}
    vel = {**dict(zip(arm_names, st.arm_qd)), **dict(zip(st.ee_names, st.ee_qd))}
    need = list(c.arm_joints) + list(c.hand_joints)
    missing = [n for n in need if n not in pos]
    if missing:
        raise JointNodeError(f"joints not in the sources: {missing}")
    if not all(np.isfinite(pos[n]) and np.isfinite(vel[n]) for n in need):
        raise JointNodeError("non-finite joint value")
    return JointMeasure({n: float(pos[n]) for n in pos}, {n: float(vel[n]) for n in vel},
                        Pose(np.asarray(st.object_pos, float).copy(), np.asarray(st.object_quat, float).copy()))


def joint_target_arrays(c: JointContract, step: JointStep) -> tuple[tuple, np.ndarray, np.ndarray]:
    """pd_node 가 받는 한 벌: 팔 q* · 손 q* · 용접 관절(고정 각), 속도 목표는 0(학습도 위치 목표만)."""
    names = tuple(c.arm_joints) + tuple(c.hand_joints) + tuple(c.welded)
    q = np.concatenate([step.targets.arm, step.targets.hand, np.asarray(list(c.welded.values()), float)])
    if not np.all(np.isfinite(q)):
        raise JointNodeError("non-finite joint target")
    return names, q, np.zeros(q.size)


def reset_pose_error(c: JointContract, m: JointMeasure) -> float:
    q = np.array([m.joint_pos[n] for n in c.arm_joints], float)
    return float(np.abs(q - np.asarray(c.arm_reset, float)).max())


def start_refusals(c: JointContract, m: JointMeasure, tol: float) -> list:
    """start 를 거부할 이유(빈 목록 = 시작). 팔이 학습 시작 자세에서 멀면 거부 — 정책은 그 자세에서만 출발해 봤다."""
    err = reset_pose_error(c, m)
    return [f"arm is {err:.3f} rad from the training reset pose (tol {tol})"] if err > tol else []

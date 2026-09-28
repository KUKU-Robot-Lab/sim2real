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
    for need in ("arm", "ee"):
        if need in bad:
            raise JointNodeError(f"source {need!r} is {'stale' if need in st.stale else 'missing'}")
    if st.arm_q is None or st.arm_qd is None:
        raise JointNodeError("arm source has no position/velocity (arm_qd is an actor input)")
    if st.ee_q is None or st.ee_qd is None:
        raise JointNodeError("hand source has no position/velocity (hand_qd is an actor input)")
    # 물체는 없거나 stale 이어도 측정은 낸다(None) — 파지 뒤에는 손에 붙인 추정을 쓰고, 그 전에는 체인이 거부한다
    fresh = "object" not in bad and st.object_pos is not None and st.object_quat is not None
    obj = Pose(np.asarray(st.object_pos, float).copy(), np.asarray(st.object_quat, float).copy()) if fresh else None
    pos = {**dict(zip(arm_names, st.arm_q)), **dict(zip(st.ee_names, st.ee_q))}
    vel = {**dict(zip(arm_names, st.arm_qd)), **dict(zip(st.ee_names, st.ee_qd))}
    need = list(c.arm_joints) + list(c.hand_joints)
    missing = [n for n in need if n not in pos]
    if missing:
        raise JointNodeError(f"joints not in the sources: {missing}")
    if not all(np.isfinite(pos[n]) and np.isfinite(vel[n]) for n in need):
        raise JointNodeError("non-finite joint value")
    return JointMeasure({n: float(pos[n]) for n in pos}, {n: float(vel[n]) for n in vel}, obj)


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


START_TILT_MAX_DEG = 15.0   # 학습 초기 컵은 똑바로 서 있다(fj reset 기울기 0) — FP++ 잡음 · 테이블 기울기 여유
START_MOVE_MAX_M = 0.02     # 리셋(목표를 정한 때)과 시작 사이 선 컵이 이만큼 흐르면 FP++ 추종이 불안정한 것


def object_start_refusals(obj: Pose | None, reset_obj: Pose | None, max_tilt_deg: float = START_TILT_MAX_DEG,
                          max_move_m: float = START_MOVE_MAX_M) -> list:
    """리셋 · 시작 때 FP++ 가 선 컵을 제대로 보고 있는지(빈 목록 = 통과). 09.28 사용자: "로봇을 리셋할 때 fpp 가 제대로
    cup 을 추종하고 있는지 확인해야 하는데 그 과정이 없음" — 2회차는 FP++ 가 선 컵을 174° 로 뒤집어 본 채 시작했다.
    기울기는 학습과 같다(물체 z 축과 세계 z 의 각, fj_core_env._get_dones). reset_obj 가 None 이면 흐름은 보지 않는다."""
    from .joint_obs import quat_to_matrix
    if obj is None:
        return ["no fresh FP++ object pose — check that FP++ is tracking the cup"]
    tilt = float(np.degrees(np.arccos(np.clip(quat_to_matrix(obj.quat)[2, 2], -1.0, 1.0))))
    out = []
    if tilt > max_tilt_deg:
        out.append(f"FP++ says the cup is at tilt {tilt:.1f} deg (> {max_tilt_deg:.0f}) — a standing cup is expected; "
                   f"re-register FP++")
    if reset_obj is not None:
        moved = float(np.linalg.norm(np.asarray(obj.pos, float) - np.asarray(reset_obj.pos, float)))
        if moved > max_move_m:
            out.append(f"FP++ cup pose moved {moved * 100:.1f} cm since reset (> {max_move_m * 100:.0f}) — "
                       f"tracking is unstable or the cup was moved; reset again")
    return out


def keypoint_goal_dist(c: JointContract, obj: Pose, goal: Pose) -> float:
    """학습의 목표 도달 거리 — 물체 · 목표 키포인트(4 개) 사이 거리의 최대(hdgp keypoint_max_dist)."""
    from .joint_obs import keypoints
    return float(np.linalg.norm(keypoints(c, obj) - keypoints(c, goal), axis=1).max())


class EpisodeEnd:
    """정책 에피소드를 스스로 끝낼 때를 정한다 — 09.28 사용자: "목표에 이송(리프트)하고 에피소드가 끝나면 정책은 끝난 것.
    이후에 홈자세로 돌아와서 다시 액션 실행". 학습 규칙 그대로:
      도달  키포인트 최대 거리 ≤ tol 이 연속 steps 스텝(env goal_success_steps · force_consecutive, tol 은 그 체크포인트가
            학습된 값 — 커리큘럼 상태는 체크포인트에 없어 tfevents task/tol 에서 읽어 넘긴다)
      시간  running 이 max_s 초(env episode_length_s)
    0 이하는 그 조건을 끈다. 순수 — 시각은 부르는 쪽이 준다.
    """

    def __init__(self, tol: float = 0.0, steps: int = 10, max_s: float = 0.0) -> None:
        self.tol, self.steps, self.max_s = float(tol), max(int(steps), 1), float(max_s)
        self.reset(0.0)

    def reset(self, t_start: float) -> None:
        self.t_start, self.near = float(t_start), 0

    def update(self, dist: float | None, t: float) -> str | None:
        """끝낼 이유(없으면 None). dist 가 None(물체 없음)이면 도달 연속을 끊는다."""
        if self.tol > 0:
            self.near = self.near + 1 if dist is not None and dist <= self.tol else 0
            if self.near >= self.steps:
                return f"goal reached (keypoint dist {dist:.3f} <= {self.tol:.3f} m for {self.near} steps)"
        if self.max_s > 0 and t - self.t_start >= self.max_s:
            return f"episode time {t - self.t_start:.1f} s >= {self.max_s:.1f} s"
        return None

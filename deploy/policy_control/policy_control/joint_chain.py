"""joint family 체인(ROS 없음): 측정 → obs(133) → 정책 → 디코더 → 팔 · 손 관절 목표.

한 스텝(= hdgp 학습 순서): 관측은 **지금** 측정 + 디코더의 직전 q* · 직전 행동으로 만들고, 정책이 행동을 내고,
디코더가 새 q* 를 낸다. 목표 자세는 리셋 때 한 번 정한다(물체 위치 + offset). 매 스텝 새 frozen `JointStep` 을 돌려준다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from .joint_contract import JointContract
from .joint_decoder import JointDecoder, JointTargets
from .joint_object import AttachCfg, ObjectEstimator
from .joint_obs import JointState, Pose, build_obs, clip_obs, first_goal


class JointChainError(RuntimeError):
    pass


class PolicyLike(Protocol):
    def forward(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class JointMeasure:
    joint_pos: Mapping      # 이름 → rad (팔 + 손 hand_joints; 용접 관절은 있어도 쓰지 않는다)
    joint_vel: Mapping
    obj: Pose | None        # 로봇 base 프레임 FP++ 자세. None = 지금 없다(가림 · stale) — 붙인 뒤에는 필요 없다


@dataclass(frozen=True)
class JointStep:
    obs: np.ndarray
    action: np.ndarray
    targets: JointTargets
    goal: Pose
    obj: Pose               # 관측에 쓴 물체 자세
    obj_source: str         # live · held · attached


class RecordedPolicy:
    """Teacher-forcing backend: replays a recorded (T, action_dim) action table."""

    def __init__(self, actions: np.ndarray):
        self._a, self._i = np.asarray(actions, float), 0

    def reset(self) -> None:
        self._i = 0

    def forward(self, obs: np.ndarray) -> np.ndarray:
        a = self._a[self._i]
        self._i += 1
        return a.copy()


class JointChain:
    def __init__(self, contract: JointContract, policy: PolicyLike, fk, attach: AttachCfg = AttachCfg()):
        self.c, self.policy, self.fk = contract, policy, fk
        self.decoder = JointDecoder(contract)
        self.objects = ObjectEstimator(contract, attach)
        self.goal: Pose | None = None

    def reset(self, obj: Pose | None, goal_offset=None) -> Pose:
        """에피소드 시작: q* · 직전 행동 · LSTM 을 0/시작 자세로, 목표를 지금 물체 위치에서 정한다(FP++ 가 보여야 한다)."""
        if obj is None:
            raise JointChainError("reset needs a live object pose (FP++) to set the goal")
        self.decoder.reset()
        self.policy.reset()
        self.objects.reset()
        self.goal = first_goal(self.c, obj, goal_offset)
        return self.goal

    def state(self, m: JointMeasure) -> tuple[JointState, str]:
        c = self.c
        arm = [m.joint_pos[n] for n in c.arm_joints]
        hand = [m.joint_pos[n] for n in self.fk.hand_joints]
        pose = self.fk.palm_pose(arm, hand)
        obj, src = self.objects.update(m.obj, pose.palm_pos, pose.extra["palm_rot"], self.decoder.hand)
        if obj is None:
            raise JointChainError("no object pose: FP++ missing/stale and the object is not attached to the hand")
        return JointState(m.joint_pos, m.joint_vel, pose.palm_pos, pose.extra["palm_rot"], pose.tips, obj,
                          self.goal), src

    def step(self, m: JointMeasure) -> JointStep:
        if self.goal is None:
            raise JointChainError("reset() before step()")
        d = self.decoder
        st, src = self.state(m)
        obs = clip_obs(self.c, build_obs(self.c, st, d.arm, d.action[:self.c.n_arm], d.hand_action_obs()))
        action = np.asarray(self.policy.forward(obs.astype(np.float32)), float).reshape(-1)
        targets = d.step(action)
        return JointStep(obs, action, targets, self.goal, st.obj, src)

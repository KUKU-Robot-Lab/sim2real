"""joint family 계약 — 팔 관절 증분 + 손 관절 절대 목표를 내는 정책(hdgp grasp_fj · cup_pick 계열, fabric 없음).

    policy_control/joint_contract/v1   ·   family "joint_direct"

기존 deploy_contract/v2 의 세 family 는 모두 palm 6D 를 fabric 에 넘긴다. 이 계열은 정책이 **관절 목표를 직접** 낸다:
  팔 7  q_raw = clip(q*_{t-1} + k_arm·a, lo, hi) → q*_t = clip(ema·q_raw + (1−ema)·q*_{t-1}, lo, hi)
  손 19 raw = lo + ½(a+1)(hi−lo)                   → q*_t = clip(ema·raw + (1−ema)·q*_{t-1}, lo, hi)
(hdgp grasp_fj_env.py `_arm_command` · `_hand_targets`, 학습 커밋에서 읽은 식.)

관측(133)은 `joint_obs.SEGMENTS` 순서다. 손 관측(hand_q · hand_qd)은 **시뮬레이터 관절 순서**이고 손 행동·목표는
프로필 순서라 둘을 따로 적는다(`hand_obs_order` · `hand_joints`). 시뮬레이터 순서는 학습 자산에서 실측해야 한다 —
어디서 왔는지를 `hand_obs_order_source` 에 적고, 가정한 값이면 `verified` 로 올리지 않는다.

`welded` 는 학습 자산에서 용접(fixed)된 관절과 그 각도다. 실기 손에는 그 관절이 있으므로 그 값으로 붙잡아 보낸다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "policy_control/joint_contract/v1"
FAMILY = "joint_direct"
N_KEYPOINTS = 4


class JointContractError(ValueError):
    pass


@dataclass(frozen=True)
class JointContract:
    schema: str
    family: str
    task: str
    run_dir: str
    checkpoint: str
    checkpoint_md5: str
    env_yaml_sha1: str
    agent_yaml_sha1: str
    asset: str                 # 배포(pd · FK) 자산 — 실기 손과 같은 관절 수
    train_asset: str           # 학습 자산(용접 관절이 있을 수 있다)
    side: str
    policy_hz: float
    obs_dim: int
    action_dim: int
    obs_clip: float | None     # rl_games 래퍼 clip_observations — 네트워크 앞에서 적용
    action_clip: float
    recurrent: bool
    arm_joints: tuple
    arm_lo: tuple
    arm_hi: tuple
    arm_reset: tuple           # 에피소드 시작 팔 자세(= q* 시드)
    k_arm: float
    arm_ema: float
    hand_joints: tuple         # 행동 · 목표 · 관측 액션칸 순서(프로필 순)
    hand_lo: tuple
    hand_hi: tuple
    hand_reset: tuple          # 손 시작 자세(= q* 시드), hand_joints 순
    hand_ema: float
    hand_obs_order: tuple      # 관측 hand_q · hand_qd 칸 순서(시뮬레이터 관절 순)
    hand_obs_order_source: str
    welded: dict               # 이름 → 고정 각도 [rad]
    palm_body: str
    tip_bodies: tuple
    keypoint_half_height: float
    keypoint_axial_unit: tuple  # (4, 3)
    goal_offset: tuple         # 첫 목표 = 리셋 때 물체 위치 + 이 값(학습 분포의 가운데)
    goal_box_lo: tuple
    goal_box_hi: tuple
    notes: tuple = field(default_factory=tuple)

    @property
    def n_arm(self) -> int:
        return len(self.arm_joints)

    @property
    def n_hand(self) -> int:
        return len(self.hand_joints)

    @property
    def assumed_order(self) -> bool:
        return self.hand_obs_order_source.startswith("assumed")


def obs_layout(n_arm: int, n_hand: int, n_tips: int) -> tuple:
    """(이름, 폭) — hdgp fj_kp_env `_get_observations` 의 cat 순서."""
    return (("arm_q", n_arm), ("arm_qd", n_arm), ("hand_q", n_hand), ("hand_qd", n_hand), ("palm_pos", 3),
            ("palm_ax", 6), ("tips_rel_palm", 3 * n_tips), ("cmd_state", n_arm), ("kp_rel_palm", 3 * N_KEYPOINTS),
            ("kp_rel_goal", 3 * N_KEYPOINTS), ("action_arm", n_arm), ("action_hand", n_hand))


def _finite(values, what: str) -> None:
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise JointContractError(f"{what}: non-finite value")


def validate(c: JointContract) -> JointContract:
    if c.schema != SCHEMA or c.family != FAMILY:
        raise JointContractError(f"schema/family {c.schema}/{c.family} != {SCHEMA}/{FAMILY}")
    if c.side not in ("right", "left"):
        raise JointContractError(f"side {c.side!r}")
    na, nh = c.n_arm, c.n_hand
    for name, n in (("arm_lo", na), ("arm_hi", na), ("arm_reset", na), ("hand_lo", nh), ("hand_hi", nh),
                    ("hand_reset", nh), ("goal_offset", 3), ("goal_box_lo", 3), ("goal_box_hi", 3)):
        v = getattr(c, name)
        if len(v) != n:
            raise JointContractError(f"{name} has {len(v)} values, want {n}")
        _finite(v, name)
    if sorted(c.hand_obs_order) != sorted(c.hand_joints):
        raise JointContractError("hand_obs_order must be a permutation of hand_joints")
    if set(c.welded) & set(c.hand_joints):
        raise JointContractError(f"welded joints are also action joints: {set(c.welded) & set(c.hand_joints)}")
    for lo, hi, n in zip(c.arm_lo + c.hand_lo, c.arm_hi + c.hand_hi, c.arm_joints + c.hand_joints):
        if not hi > lo:
            raise JointContractError(f"{n}: empty range [{lo}, {hi}]")
    for q, lo, hi, n in zip(c.arm_reset + c.hand_reset, c.arm_lo + c.hand_lo, c.arm_hi + c.hand_hi,
                            c.arm_joints + c.hand_joints):
        if not lo - 1e-9 <= q <= hi + 1e-9:
            raise JointContractError(f"{n}: reset {q} outside [{lo}, {hi}]")
    if not (0.0 < c.arm_ema <= 1.0 and 0.0 < c.hand_ema <= 1.0 and c.k_arm > 0.0):
        raise JointContractError(f"gains k_arm {c.k_arm} · ema {c.arm_ema}/{c.hand_ema} out of range")
    if len(c.tip_bodies) < 1 or c.keypoint_half_height <= 0.0 or len(c.keypoint_axial_unit) != N_KEYPOINTS:
        raise JointContractError("tip bodies / keypoints malformed")
    want_obs = sum(w for _, w in obs_layout(na, nh, len(c.tip_bodies)))
    if c.obs_dim != want_obs or c.action_dim != na + nh:
        raise JointContractError(f"dims {c.obs_dim}/{c.action_dim} != layout {want_obs}/{na + nh}")
    if not all(lo <= hi for lo, hi in zip(c.goal_box_lo, c.goal_box_hi)):
        raise JointContractError("goal box inverted")
    if c.policy_hz <= 0.0 or c.action_clip <= 0.0 or (c.obs_clip is not None and c.obs_clip <= 0.0):
        raise JointContractError("policy_hz / clips must be positive")
    return c


def _tuplify(v):
    if isinstance(v, list):
        return tuple(_tuplify(x) for x in v)
    return v


def to_dict(c: JointContract) -> dict:
    return dataclasses.asdict(c)


def from_dict(raw: dict) -> JointContract:
    try:
        body = {k: (v if k == "welded" else _tuplify(v)) for k, v in raw.items()}
        return validate(JointContract(**body))
    except TypeError as exc:
        raise JointContractError(f"malformed joint contract: {exc}") from exc


def save_contract(c: JointContract, path: Path) -> None:
    Path(path).write_text(json.dumps(to_dict(validate(c)), indent=1, ensure_ascii=False) + "\n")


def load_contract(path: Path) -> JointContract:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise JointContractError(f"cannot read {path}: {exc}") from exc
    return from_dict(raw)


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

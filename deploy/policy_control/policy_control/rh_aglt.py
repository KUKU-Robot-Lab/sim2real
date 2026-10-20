"""rh_aglt — RH56F1 한 팔 먼 출발 접근 · 파지 · 들기 · 이송(hdgp `open-rh_{r,l}_aglt`) 배포 계약 · 디코더 · 관측. 순수(ROS 없음).

09.30 사용자: RH56F1 정책들을 deploy 에 연결. hdgp 기준 코드(09.30): tasks/rh_aglt_r/{rh_aglt_env.py _pre_physics_step ·
_actor_parts, hand_action.py, arm_action.py, layout.py actor_layout, goals.py}, modules/keypoint_goal.py.
rh_aglt_env.py 는 좌우가 바이트 단위로 같다(hdgp test_rh_aglt_contract 가 잠금).

행동 13 = 팔 7 · 손 6, clamp ±1.
  팔  q_raw = clip(q* + k·a, lo, hi) → q* = clip(α·q_raw + (1−α)·q*, lo, hi)          (arm_action.step_arm_increment)
      대기(hold_steps) 동안 q* = 시작 자세(arm_start_q)
  손  policy_control/rh56f1_hand.py HandLaw — grip 범위 · EMA 0.1 · 전 범위 1 s · 1 N 동결 · 대기 중 편 손(hold "open")
관측 96(layout.actor_layout 순):
  arm_q 7 · arm_qd 7 · arm q* 7 · hand_q 6(★프로필 순 — hand_t = 이름으로 찾은 hand_ids) · hand_err 6((q*−q)/1.2, ±1) ·
  palm 3 · palm R 열0+열1 6 · cup−palm 3 · cup_up 3 · tips−cup 15 · (컵 kp − 목표 kp) 12 · 목표−palm 3 · 촉각 5 · 직전 행동 13
★10.12 다물체 한 정책(object_names · per_env_geometry · object_id_obs): 관측 96 뒤에 물체 one-hot K(object_names 의 서로 다른
키 순서, 잡음 없음 — hdgp rh_aglt_env._actor_parts 맨 끝) → 99. 컵 반높이 · 목표 박스 높이는 물체마다(select_object).
위치는 로봇 base(= 학습 env-local, 로봇이 원점) 기준. 목표: 리셋 때 컵 + (0, 0, goal_first_z 가운데), 자세 = 리셋 때 컵 자세
(학습은 xy ±5 cm · z 0.10~0.18 균등 — 배포는 가운데). 10.01 사용자: 목표를 직접 넣는다 — rh_aglt_goals.py(학습 목표 분포
검사 · 먼 목표는 goal_delta_distance 이내 중간 목표로 나눠 차례로, 성공 = 키포인트 최대거리 ≤ tol_floor 누적 10 스텝 + 쥠).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from policy_control import hand_block as HB
from policy_control import rh56f1_hand as RH
from policy_control.pour_fj import _urdf_limits, read_env

SCHEMA = "policy_control/rh_aglt_contract/v1"
ROLES = ("arm",)
TASK_PREFIX = "open-rh_{}_aglt"
#: modules/keypoint_goal.KEYPOINT_AXIAL_UNIT — 물체 프레임 축 위 네 점(× half_height)
KEYPOINT_AXIAL_UNIT = ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0 / 3.0), (0.0, 0.0, -1.0 / 3.0))
ACTOR_LAYOUT = (("arm_q", 7), ("arm_qd", 7), ("arm_q_target", 7), ("hand_q", 6), ("hand_err", 6), ("palm_pos", 3),
                ("palm_rot", 6), ("cup_rel_palm", 3), ("cup_up", 3), ("tips_rel_cup", 15), ("kp_rel_goal", 12),
                ("goal_rel_palm", 3), ("tactile", 5), ("prev_action", 13))


class RhAgltError(ValueError):
    pass


@dataclass(frozen=True)
class RaSide:
    role: str
    side: str
    arm_joints: list
    arm_home: list
    arm_lo: list
    arm_hi: list
    hand_joints: list           # 프로필 순 = 행동 · 관측 순(rh_aglt 는 이름으로 찾는다 — PhysX 순 문제가 없다)
    hand_open: list
    hand_grip: list
    hand_lim_lo: list
    hand_lim_hi: list
    palm_body: str
    tip_bodies: list            # 엄지 → 새끼


@dataclass(frozen=True)
class RaContract:
    schema: str
    task: str
    run_dir: str
    checkpoint: str
    checkpoint_md5: str
    env_yaml_sha1: str
    agent_yaml_sha1: str
    asset: str
    profile: str
    policy_hz: float
    episode_s: float
    obs_dim: int
    action_dim: int
    obs_clip: float
    action_clip: float
    recurrent: bool
    hold_steps: int
    k_arm: float
    arm_ema: float
    hand_ema: float
    hand_full_range_s: float
    freeze_threshold_n: float
    joint_err_norm: float
    tactile_clip_n: float
    tactile_tanh_n: float
    cup_half_height: float
    goal_offset: list           # 리셋 때 컵 위치에 더한다(학습 첫 목표 분포의 가운데)
    hand_obs_order_source: str
    sides: dict                 # "arm" → RaSide
    notes: list = field(default_factory=list)
    # ★10.01 목표 입력(rh_aglt_goals) — 학습 목표 분포. 빈 값 = 이 필드 전 계약(목표 입력을 거부한다, 다시 빌드)
    goal_box_min: list = field(default_factory=list)     # base 절대 박스 = 소환 박스 xy ± margin · z 정착고 + goal_box_z_range
    goal_box_max: list = field(default_factory=list)
    goal_first_xy_range: float = 0.0                     # 첫 목표: 리셋 때 컵에서 수평 ±, 위로 goal_first_z_range
    goal_first_z_range: list = field(default_factory=list)
    goal_delta_distance: float = 0.0                     # 다음 목표: 직전 목표에서 축마다 ±
    goal_success_steps: int = 0                          # 키포인트 최대거리 ≤ goal_tol 인 스텝 누적(연속 아님)
    goal_tol: float = 0.0                                # 학습 종점 tol_floor
    grasp_threshold_n: float = 0.0                       # 쥠 = 엄지 AND 다른 손가락 > 이 힘(contact_force_threshold)
    # ★10.08 손 명령 입력 — 학습 env hand_adm_enable 이면 "admittance": pd 가 손 목표를 /hand_<s>/angle_target(마스터의 손가락별
    #   어드민턴스, robot_control components/rh56f1.yaml control.admittance)으로 보낸다. "position" = angle_set(이 필드 전 계약 전부).
    hand_command: str = "position"
    hand_admittance: dict = field(default_factory=dict)  # 학습 값 k_g_per_rad · f_max_g · tau_contact_s · rate_rad_s · dr_frac(빈 값 = 위치)
    # ★10.12 다물체 한 정책(hdgp object_id_obs) — object_names = 서로 다른 키, 순서 = one-hot id. 빈 값 = 물체 관측 없음(이 필드 전 계약)
    object_names: list = field(default_factory=list)
    object_geometry: dict = field(default_factory=dict)  # 이름 → {origin_offset_z, mouth_z, bottom_z, radius, half_height}(hdgp obj_geometry)
    goal_force_consecutive: bool = False                 # 10.11 D6: 목표 근처 스텝을 연속으로만 센다(벗어나면 0)
    active_object: str = ""                              # select_object 결과(계약 파일에는 비어 있다 — 에피소드 reset 때 고른다)
    # ★10.12 hdgp actor_tactile_obs(3a1f897f) — False 면 손끝 촉각 5 칸이 actor 에서 빠진다(sim 원위 마디 전체 합력 대 실기
    #   손끝 패드라 분포가 다르다 · rg3 실기 재현 차이의 25~30 %). True = 이 필드 전 계약 전부.
    actor_tactile_obs: bool = True
    # ★10.12 막힘 신호 쥠 멈춤 · 쥔 판정(hdgp hand_block_enable — grasping 확정 정의, policy_control/hand_block.py).
    #   빈 값 = 이 필드 전 계약(손끝 촉각 > freeze_threshold_n 멈춤 · grasp_signal 쥔 판정).
    hand_block: dict = field(default_factory=dict)       # {g_rad, eps_rad, n_ticks, lag_ticks, axes}
    grasp_rule: str = ""                                 # "thumb_flex_and_any_finger"(hand_block 과 같이)
    hand_lead_cap_rad: float = 0.0                       # 학습 sim 의 손 목표 앞섬 상한 — 실기 pd HAND_MAX_LEAD 와 같아야 한다
    # ★10.13 쥔 판정 유지(hdgp grasp_hold_rule, hand_block.GraspHoldNp). 빈 값 = 막힘 쥔 판정 그대로(10.12 계약).
    grasp_hold_rule: str = ""                            # "block_follow_open"
    grasp_release_open_rad: list = field(default_factory=list)   # (엄지굽힘, 검지, 중지, 약지, 새끼) [rad]
    grasp_release_steps: int = 0
    # ★10.14 쥔 규칙 latched_axes_at_least(hdgp hand_grasp_judge pos · opp) — 판정 축 · k · 물체-손바닥 거리. 빈 값 = 옛 규칙.
    grasp_judge_axes: list = field(default_factory=list)  # 계약 axes 이름(thumb_flex · index · middle · ring · pinky)
    grasp_min_axes: int = 0
    grasp_near_m: float = 0.0                            # > 0 이면 쥔 판정 = 손 판정 AND |물체 원점 − 손바닥| < 이 값(hdgp hand_grasp_near_m)
    # ★10.12 계열 고유 관측 칸(폭 1씩, 물체 one-hot 뒤) — rh_place 관측 래치("latch", hdgp extra_actor_obs). 값은 체인이 준다.
    extra_actor_obs: list = field(default_factory=list)
    # ★10.19 팔 들림 래치(hdgp arm_hold_rise_m — tasks/rh_aglt_l/arm_action.py update_arm_latch · step_arm_hold,
    #   rh_aglt_env.py :957-966 · :1355-1358, 51094780). 관측 컵 z 가 처음 높이(정책이 처음 움직이는 스텝의 관측 컵 z)보다
    #   arm_hold_rise_m 위가 되는 스텝에 직전 팔 목표를 래치 자세로 둔다 → 그 뒤 팔 목표 = EMA(직전, 래치 + k_arm·a)(증분 누적 없음).
    #   래치 중 들림 < arm_hold_release_m 이 arm_hold_release_steps 연속이면 풀고 증분으로 돌아간다(다시 오르면 새로 래치).
    #   0 = 끔(이 필드 전 계약 전부).
    arm_hold_rise_m: float = 0.0
    arm_hold_release_m: float = 0.0
    arm_hold_release_steps: int = 30
    # ★10.20 학습 시작 자세(hdgp aglt_start, 10.19 d43340d11) — "home" = 홈에서 정책(이 필드 전 계약 전부),
    #   "ready" = ReadyApproach 사전 파지 자세(컵 축 r_mult·R · 높이 h)까지 스크립트로 간 뒤 정책(rcg_02 · rcg_03).
    #   ready 면 노드는 홈 대신 사전 파지 자리 근처에서만 reset 을 받고(pour_fj_node.aglt_start_refusals), 디코더는 실측 팔에서
    #   시작하며 대기(hold_steps) 동안 관측 직전 행동 = (팔 0 · 손 폄)(ready_hold_action, sim 기록 행동과 같다).
    arm_start: str = "home"
    #   사전 파지 손바닥 원점 = 물체 원점 + r_mult × 반지름 × (수평 바깥) + h × 위 — 학습 범위 [lo, hi](hdgp ready_pre_*)
    ready_r_mult: list = field(default_factory=list)
    ready_h_m: list = field(default_factory=list)

    def side(self, role: str = "arm") -> RaSide:
        return self.sides[role]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


#: 손 명령 입력 — pd 가 이 값을 에피소드 reset 때 받아 손 토픽을 고른다(pd_backends.HAND_COMMANDS 와 같다)
HAND_COMMANDS = ("position", "admittance")
#: hdgp rh_aglt_env_cfg hand_adm_* → 계약 hand_admittance 키
ADM_KEYS = {"hand_adm_k_g_per_rad": "k_g_per_rad", "hand_adm_f_max_g": "f_max_g", "hand_adm_tau_contact_s": "tau_contact_s",
            "hand_adm_rate_rad_s": "rate_rad_s", "hand_adm_dr_frac": "dr_frac"}


def hand_command_of(env: Mapping) -> tuple[str, dict]:
    """학습 env → (손 명령 입력, 학습 어드민턴스 값). hand_adm_enable 이 없거나 거짓이면 위치 제어."""
    if not bool(env.get("hand_adm_enable", False)):
        return "position", {}
    missing = [k for k in ADM_KEYS if k != "hand_adm_dr_frac" and k not in env]
    if missing:
        raise RhAgltError(f"hand_adm_enable 인데 {missing} 이 env.yaml 에 없다")
    return "admittance", {v: float(env.get(k, 0.0)) for k, v in ADM_KEYS.items()}


def admittance_mismatch(train: Mapping, real: Mapping, reg_per_rad: float, *, rel_tol: float = 0.02) -> list[str]:
    """학습 어드민턴스(rad · g) ↔ 실기 드라이버 어드민턴스(레지스터 · g, robot_control control.admittance) — 다른 항목 설명.
    reg_per_rad = 네 손가락 레지스터/rad(변환표). 학습 DR(±dr_frac)은 허용 폭이 아니다 — 명목값이 같아야 한다."""
    pairs = (("k_g_per_rad", float(real["k_g_per_reg"]) * reg_per_rad), ("f_max_g", float(real["f_max_g"])),
             ("tau_contact_s", float(real["tau_contact_s"])), ("rate_rad_s", float(real["rate_reg_s"]) / reg_per_rad))
    out = []
    for key, r in pairs:
        t = float(train[key])
        if abs(t - r) > rel_tol * max(abs(t), abs(r), 1e-9):
            out.append(f"{key}: 학습 {t:g} ≠ 실기 {r:g}")
    return out


def hand_law(c: RaContract) -> RH.HandLaw:
    s = c.side()
    return RH.HandLaw(q_open=tuple(s.hand_open), q_grip=tuple(s.hand_grip), lim_lo=tuple(s.hand_lim_lo),
                      lim_hi=tuple(s.hand_lim_hi), range_mode="grip", ema=c.hand_ema, full_range_s=c.hand_full_range_s,
                      policy_hz=c.policy_hz, freeze=True, freeze_joints=(True,) * 6,
                      freeze_threshold_n=c.freeze_threshold_n, hold="open")


def obs_dim_of(c: RaContract | None = None, *, tactile: bool | None = None) -> int:
    """관측 폭 — 96(촉각 없으면 91) + 물체 one-hot K(object_names 가 있을 때)."""
    return sum(w for _, w in actor_layout(c, tactile=tactile))


def actor_layout(c: RaContract | None = None, *, tactile: bool | None = None) -> tuple:
    """hdgp layout.actor_layout 과 같은 순서 — tactile 은 계약 actor_tactile_obs(없으면 True)."""
    tac = (c.actor_tactile_obs if c is not None else True) if tactile is None else bool(tactile)
    base = ACTOR_LAYOUT if tac else tuple(x for x in ACTOR_LAYOUT if x[0] != "tactile")
    base = base + ((("object_id", len(c.object_names)),) if c is not None and c.object_names else ())
    return base + (tuple((str(n), 1) for n in c.extra_actor_obs) if c is not None else ())


#: 학습 덤프가 가질 수 있는 actor 폭(촉각 있음 · 없음) — 물체 one-hot 은 따로 더한다
ACTOR_BASE_DIMS = (obs_dim_of(tactile=True), obs_dim_of(tactile=False))


#: hdgp rh_aglt_env._make_ready_approach 가 받는 aglt_start 값
ARM_STARTS = ("home", "ready")


def ready_hold_action(c: RaContract) -> np.ndarray:
    """arm_start ready 의 대기 중 기록 행동 = (팔 0, 손 폄) — hdgp _a_open_rdy 와 같은 식(q_open 을 [min, max](open, grip) 로)."""
    lo, hi = hand_law(c).bounds()
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    a_open = np.clip(2.0 * (np.asarray(c.side().hand_open, float) - lo) / span - 1.0, -1.0, 1.0)
    return np.concatenate([np.zeros(len(c.side().arm_joints)), a_open])


def ready_pose_error(c: RaContract, cup_pos: Sequence[float], palm_pos: Sequence[float]) -> tuple[float, float, float]:
    """사전 파지 자리 오차 — (수평 거리 / 반지름 = r/R, 높이 − h 범위 밖 거리[m], 수평 거리 − r 범위 밖 거리[m]). 범위 안이면 0."""
    if len(c.ready_r_mult) != 2 or len(c.ready_h_m) != 2:
        raise RhAgltError("ready_pose_error: 계약에 사전 파지 범위(ready_r_mult · ready_h_m)가 없다 — 학습 env 에 ready_pre_* 가 없던 정책")
    if not c.active_object:
        raise RhAgltError("ready_pose_error: 물체를 먼저 고른다(select_object) — 반지름이 물체마다 다르다")
    radius = float(c.object_geometry[c.active_object]["radius"])
    d = np.asarray(palm_pos, float) - np.asarray(cup_pos, float)
    horiz, dz = float(np.hypot(d[0], d[1])), float(d[2])
    r_lo, r_hi = (radius * float(v) for v in c.ready_r_mult)
    out_r = max(r_lo - horiz, horiz - r_hi, 0.0)
    out_h = max(float(c.ready_h_m[0]) - dz, dz - float(c.ready_h_m[1]), 0.0)
    return horiz / radius, out_h, out_r


def validate(c: RaContract) -> None:
    if c.action_dim != 13 or c.obs_dim != obs_dim_of(c):
        raise RhAgltError(f"obs/action {c.obs_dim}/{c.action_dim} ≠ 레이아웃 {obs_dim_of(c)}/13")
    s = c.side()
    if len(s.arm_joints) != 7 or len(s.hand_joints) != 6 or len(s.tip_bodies) != 5:
        raise RhAgltError("팔 7 · 손 6 · 손끝 5")
    if [n.split("_hj_")[1] for n in s.hand_joints] != list(RH.JOINTS):
        raise RhAgltError(f"손 관절 {s.hand_joints} — RH56F1 프로필 순서가 아니다")
    if not all(lo <= h <= hi for lo, h, hi in zip(s.arm_lo, s.arm_home, s.arm_hi)):
        raise RhAgltError("arm_home 이 관절 한계 밖")
    if not (c.k_arm > 0 and 0 < c.arm_ema <= 1 and c.cup_half_height > 0 and c.joint_err_norm > 0):
        raise RhAgltError("k_arm · arm_ema · cup_half_height · joint_err_norm")
    if c.arm_start not in ARM_STARTS:
        raise RhAgltError(f"arm_start {c.arm_start!r} 는 {ARM_STARTS}(hdgp aglt_start)")
    if c.arm_start == "ready" or c.ready_r_mult or c.ready_h_m:
        if not c.object_geometry:
            raise RhAgltError("사전 파지 범위는 물체 반지름(object_geometry)이 있어야 쓴다 — 사전 파지 거리 = r_mult × 반지름")
        if not (len(c.ready_r_mult) == 2 and len(c.ready_h_m) == 2 and 1.0 < c.ready_r_mult[0] <= c.ready_r_mult[1]
                and c.ready_h_m[0] <= c.ready_h_m[1]):
            raise RhAgltError(f"arm_start ready: ready_r_mult {c.ready_r_mult} · ready_h_m {c.ready_h_m} 는 [lo, hi](r 은 1 초과)")
    hand_law(c)                                    # 손 자세 · 한계 검사
    if c.hand_command not in HAND_COMMANDS:
        raise RhAgltError(f"hand_command {c.hand_command!r} — {HAND_COMMANDS}")
    if (c.hand_command == "admittance") != bool(c.hand_admittance):
        raise RhAgltError("hand_command admittance 와 hand_admittance(학습 값)는 같이 있어야 한다")
    _validate_objects(c)
    _validate_block(c)
    _validate_arm_hold(c)


def make_block(c: RaContract) -> HB.HandBlockNp:
    """계약 hand_block + 쥔 규칙 → 한 손 막힘 상태(디코더 · 검사가 같은 길로 만든다)."""
    s = c.side()
    return HB.HandBlockNp.from_contract(c.hand_block, s.hand_joints, s.hand_open, s.hand_grip, c.grasp_rule,
                                        c.grasp_judge_axes, c.grasp_min_axes)


def _validate_block(c: RaContract) -> None:
    if not (np.isfinite(c.grasp_near_m) and c.grasp_near_m >= 0.0):
        raise RhAgltError(f"grasp_near_m ≥ 0(0 = 끔): {c.grasp_near_m}")
    if not c.hand_block:
        if c.grasp_rule or c.grasp_hold_rule or c.grasp_judge_axes or c.grasp_min_axes or c.grasp_near_m:
            raise RhAgltError(f"grasp_rule {c.grasp_rule!r} · grasp_hold_rule {c.grasp_hold_rule!r} · 판정 축 · near 는 "
                              "hand_block 과 같이 있어야 한다")
        return
    if c.grasp_rule not in HB.GRASP_RULES:
        raise RhAgltError(f"grasp_rule {c.grasp_rule!r} — {HB.GRASP_RULES}")
    try:
        block = make_block(c)
    except HB.HandBlockError as e:
        raise RhAgltError(str(e)) from e
    if c.grasp_hold_rule:
        if c.grasp_hold_rule not in HB.HOLD_RULES:
            raise RhAgltError(f"grasp_hold_rule {c.grasp_hold_rule!r} — {HB.HOLD_RULES}")
        try:
            HB.GraspHoldNp(block, c.grasp_release_open_rad, c.grasp_release_steps)
        except HB.HandBlockError as e:
            raise RhAgltError(str(e)) from e
    from policy_control.pd_backends import HAND_MAX_LEAD
    if abs(float(c.hand_lead_cap_rad) - HAND_MAX_LEAD) > 1e-9:
        raise RhAgltError(f"hand_lead_cap_rad {c.hand_lead_cap_rad} ≠ 실기 pd HAND_MAX_LEAD {HAND_MAX_LEAD} — "
                          "막힘 신호를 학습한 sim 의 앞섬 상한이 실기와 달라진다")


def _validate_arm_hold(c: RaContract) -> None:
    rise, rel, n = float(c.arm_hold_rise_m), float(c.arm_hold_release_m), int(c.arm_hold_release_steps)
    if not (np.isfinite(rise) and np.isfinite(rel) and rise >= 0.0 and rel >= 0.0 and n >= 1):
        raise RhAgltError(f"arm_hold_rise_m {rise} · arm_hold_release_m {rel} ≥ 0 · arm_hold_release_steps {n} ≥ 1")
    if rel > 0.0 and not rel < rise:
        raise RhAgltError(f"arm_hold_release_m {rel} 는 arm_hold_rise_m {rise} 보다 작아야 한다(0 = 풀지 않음)")


def _validate_objects(c: RaContract) -> None:
    names = list(c.object_names)
    if len(set(names)) != len(names):
        raise RhAgltError(f"object_names {names} 에 중복 — one-hot id 는 서로 다른 키")
    if names and sorted(c.object_geometry) != sorted(names):
        raise RhAgltError(f"object_geometry 키 {sorted(c.object_geometry)} ≠ object_names {names}")
    for k, g in c.object_geometry.items():
        if set(g) != set(GEOM_FIELDS) or float(g["half_height"]) <= 0:
            raise RhAgltError(f"object_geometry[{k}] = {g} — {GEOM_FIELDS}, 반높이 > 0")
    if c.active_object and c.active_object not in names:
        raise RhAgltError(f"active_object {c.active_object!r} 가 object_names {names} 에 없다")


def load_contract(path: str | Path) -> RaContract:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise RhAgltError(f"{path}: schema {raw.get('schema')!r} ≠ {SCHEMA}")
    raw["sides"] = {r: RaSide(**s) for r, s in raw["sides"].items()}
    c = RaContract(**raw)
    validate(c)
    return c


# ---------------------------------------------------------------- 빌드
#: hdgp rh_aglt real_response(09.30 9a46c174): 명령 지연 · 편 손 하한 · 리셋 엄지 외전 — PD 에 들어가는 목표만 바꾼다
SIM_ONLY_RESPONSE_KEYS = ("arm_cmd_delay_steps", "arm_cmd_delay_joint_steps", "arm_cmd_delay_joint_jitter", "hand_cmd_delay_steps",
                          "hand_open_floor_deg_lo", "hand_open_floor_deg_hi", "thumb1_reset_range")


#: hdgp tasks/rh_aglt_r/rh_aglt_env_cfg.py OBJECTS · CUP_UNIT(hdgp 2721a946 :44-58) — 컵 USD 원점 기준 치수(m) · cup_scale 을 받는지
CUP_OBJECTS: dict[str, tuple[dict[str, float], bool]] = {
    "shaker": ({"bottom_z": -0.0921, "rim_z": 0.0829}, True),
    "cyl60": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "cyl65": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    # 10.09 hdgp OBJECTS(rh_aglt_env_cfg.py, e2bc225f) — 병은 unit 을 cyl60 값 그대로(목표 키포인트 · 보상 문맥 공유), 쉐이커는 높이 130
    "cyl60_box32w25": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "source240": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "source200": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "source240_pla": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "source200_pla": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "shaker_c": ({"bottom_z": -0.065, "rim_z": 0.065}, False),
    "shaker_b85": ({"bottom_z": -0.085, "rim_z": 0.045}, False),
    # 10.11 hdgp OBJECTS *_real(실물 무게 · 실물 치수) — 10.12 다물체 한 정책(rg3)
    "source240_real": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "source200_real": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "shaker_c_real": ({"bottom_z": -0.065, "rim_z": 0.065}, False),
}
#: hdgp OBJECTS unit outer_r(m) — 다물체 기하 표(object_geometry radius)용. 없는 물체는 다물체 계약에 못 쓴다
CUP_OUTER_R = {"source240_real": 0.030, "source200_real": 0.030, "shaker_c_real": 0.0354}
#: hdgp tasks/rh_aglt_r/obj_geometry.GEOM_FIELDS
GEOM_FIELDS = ("origin_offset_z", "mouth_z", "bottom_z", "radius", "half_height")


#: ★10.17 hdgp OBJECTS 의 충돌체만 바꾼 사본(10.15 _solid 속 찬 회전체 · _cvx 볼록 조각) → 같은 실물 _real. unit · 질량 ·
#:   visual 이 _real 과 같고(hdgp rh_aglt_env_cfg.py :94-111, 808f39d7) 실기에는 충돌체가 없다 → 계약 물체 이름은 _real 로 적는다.
#:   one-hot 순서는 학습 object_names 순서 그대로(이름만 바뀐다) — 콘솔 · 에피소드 policy_object(source240_real …)가 그대로 맞는다.
SIM_COLLIDER_ALIAS = {f"{b}_{k}": f"{b}_real" for b in ("source240", "source200", "shaker_c") for k in ("cvx", "solid")}


def _names(env: Mapping) -> list[str]:
    names = [str(n) for n in (env.get("object_names") or ())] or [str(env.get("object_name", "shaker"))]
    names = [SIM_COLLIDER_ALIAS.get(n, n) for n in names]
    unknown = [n for n in names if n not in CUP_OBJECTS]
    if unknown:
        raise RhAgltError(f"object_name(s) {unknown} 을 모른다 — {sorted(CUP_OBJECTS)} (hdgp OBJECTS 를 옮겨 와라)")
    return names


def object_geometry(name: str, cup_scale: float) -> dict[str, float]:
    """hdgp obj_geometry.object_geometry 와 같은 공식 — scaled 물체만 cup_scale 을 곱한다."""
    unit, scaled = CUP_OBJECTS[name]
    if name not in CUP_OUTER_R:
        raise RhAgltError(f"{name}: 반지름(hdgp OBJECTS outer_r)을 모른다 — CUP_OUTER_R 에 옮겨 와라")
    s = float(cup_scale) if scaled else 1.0
    return {"origin_offset_z": -unit["bottom_z"] * s, "mouth_z": unit["rim_z"] * s, "bottom_z": unit["bottom_z"] * s,
            "radius": CUP_OUTER_R[name] * s, "half_height": 0.5 * (unit["rim_z"] - unit["bottom_z"]) * s}


def object_ids(env: Mapping) -> list[str]:
    """hdgp object_id_obs 의 one-hot 순서 = object_names 의 서로 다른 키, 처음 나온 순서(obj_geometry.distinct_keys)."""
    return list(dict.fromkeys(_names(env))) if bool(env.get("object_id_obs", False)) else []


def cup_geometry(env: Mapping) -> tuple[float, float]:
    """(cup_half_height, cup_origin_offset_z) — hdgp resolve_cfg(:410-422)와 같은 규칙으로 object_name 에서 다시 푼다.

    env.yaml 의 파생 값은 쓰지 않는다: train.py 는 hydra 가 object_name 을 덮은 뒤 · env 가 resolve_cfg 를 다시 부르기 전에
    덤프하므로 cyl60 런의 덤프에는 기본 shaker × cup_scale 값이 남는다(10.04 확인 — 학습 env 는 cyl60 으로 돌았다)."""
    names = _names(env)
    # 10.09 다물체 런(object_names) — hdgp resolve_cfg 처럼 unit · scaled 가 모두 같아야 기하를 공유한다.
    #   10.11 per_env_geometry 면 unit 은 달라도 된다(스칼라 = 첫 물체, 물체별 값은 계약 object_geometry · select_object)
    per_env = bool(env.get("per_env_geometry", False))
    if any(CUP_OBJECTS[n][1] != CUP_OBJECTS[names[0]][1] or (not per_env and CUP_OBJECTS[n] != CUP_OBJECTS[names[0]])
           for n in names):
        raise RhAgltError(f"object_names {names} 의 unit · scaled 가 다르다 — 학습 env 도 거부한다(unit 만 다르면 per_env_geometry)")
    unit, scaled = CUP_OBJECTS[names[0]]
    s = float(env["cup_scale"]) if scaled else 1.0
    return 0.5 * (unit["rim_z"] - unit["bottom_z"]) * s, -unit["bottom_z"] * s


def is_rh_aglt_run(env: Mapping) -> bool:
    """rh_place(hdgp open-rh_*_place)도 프로필 · 관측 96 · 행동 13 이 같다 — 홀더 키(target_holders)로 가른다(10.04)."""
    return (str(env.get("profile_name", "")).startswith("rh56f1_") and int(env.get("action_space", 0)) == 13
            and _dumped_obs_ok(env) and "target_holders" not in env)


def _dumped_obs_ok(env: Mapping, extra: int = 0) -> bool:
    """덤프된 observation_space 가 이 런의 관측 폭으로 읽히는가. hdgp 는 env 초기화 전 cfg 기본값(96, 촉각 있음)을
    덤프한다 — 10.13 rg3s(actor_tactile_obs False, 실제 94)도 96 으로 남았다. 실제 폭은 촉각 · 물체에서 다시 계산하고,
    체크포인트 running_mean 폭이 다르면 정책 로드에서 막힌다."""
    base, k = _base_dim(env), _k_obs(env)
    dflt = obs_dim_of(tactile=True)                 # cfg 기본 96 — rh_place 는 계열 칸(latch)을 더해 97 로 덤프한다(10.13 rg3bfo)
    return int(env.get("observation_space", 0)) in (base, base + k, base + k + extra, dflt, dflt + extra)


def _actor_tactile(env: Mapping) -> bool:
    """hdgp cfg actor_tactile_obs(10.12, 기본 True — 이 키 전 런 전부)."""
    return bool(env.get("actor_tactile_obs", True))


def _base_dim(env: Mapping) -> int:
    return obs_dim_of(tactile=_actor_tactile(env))


def _k_obs(env: Mapping) -> int:
    """물체 one-hot 폭 — 덤프된 observation_space 는 96(해석 전) 또는 96+K 일 수 있다."""
    try:
        return len(object_ids(env))
    except RhAgltError:
        return 0


def _left(name: str) -> str:
    return "l_" + name[2:] if name.startswith("r_") else name


def build(run_dir: Path, checkpoint: Path, right_profile, urdf: Path, *, asset: str) -> RaContract:
    """런 덤프(params/env.yaml · agent.yaml) + hdgp RH56F1_RIGHT 프로필 → 계약. 왼팔은 hdgp tasks/rh_aglt_l/profile.py 와
    같은 규칙으로 이름만 바꾼다(손 거울 부호 +1 — 좌 URDF 가 엄지 축을 이미 뒤집었다). 팔 시작 자세는 env.yaml arm_start_q
    (좌 런은 이미 거울상)."""
    env_p = Path(run_dir) / "params" / "env.yaml"
    env = read_env(env_p)
    if not is_rh_aglt_run(env):
        raise RhAgltError(f"{env_p}: rh_aglt 런이 아니다(profile rh56f1_* · obs {_base_dim(env)}(+물체 one-hot) · action 13 · "
                          "홀더 키 없음 — rh_place 는 따로)")
    return build_from_env(Path(run_dir), checkpoint, right_profile, urdf, asset=asset, env=env)


def build_from_env(run_dir: Path, checkpoint: Path, right_profile, urdf: Path, *, asset: str, env: Mapping,
                   extra_obs: int = 0) -> RaContract:
    """build 의 본체 — 런 종류 검사 없이(rh_place 가 같은 관측 · 디코더 칸을 여기서 받는다, rh_place.build)."""
    env_p, agent_p = run_dir / "params" / "env.yaml", run_dir / "params" / "agent.yaml"
    agent = yaml.safe_load(agent_p.read_text())
    net, cfg_a = agent["params"]["network"], agent["params"]["config"]
    # ★10.20 2판 손 USD 프로필(rh56f1_v2_left · ap_05 · rcg_02 · rcg_03)도 왼팔 — hdgp 프로필은 이름 · USD 경로만 다르다
    side = "left" if str(env["profile_name"]).endswith("_left") else "right"
    rename = _left if side == "left" else (lambda n: n)
    p = side[0]
    arm = [f"{p}_aj_{i}" for i in range(1, 8)]
    hand = [rename(n) for n in right_profile.hand_joint_names]
    lim = _urdf_limits(urdf, arm + hand)
    home = [float(v) for v in env["arm_start_q"]]
    s = RaSide(role="arm", side=side, arm_joints=arm, arm_home=home, arm_lo=[lim[j][0] for j in arm],
               arm_hi=[lim[j][1] for j in arm], hand_joints=hand,
               hand_open=[float(v) for v in right_profile.hand_open_pose],
               hand_grip=[float(v) for v in right_profile.hand_grip_pose],
               hand_lim_lo=[lim[j][0] for j in hand], hand_lim_hi=[lim[j][1] for j in hand],
               palm_body=rename(str(right_profile.palm_body)), tip_bodies=[rename(b) for b in right_profile.fingertip_bodies])
    z = [float(v) for v in env["goal_first_z_range"]]
    ctr, hw, m = [float(v) for v in env["spawn_center"]], [float(v) for v in env["spawn_half"]], float(env["goal_box_xy_margin"])
    half_h, origin_z = cup_geometry(env)
    z0 = float(env["table_surface_z"]) + origin_z
    zb = [float(v) for v in env["goal_box_z_range"]]
    for k in ("goal_first_tilt_deg", "goal_delta_rotation_deg"):
        if float(env.get(k, 0.0)) != 0.0:
            raise RhAgltError(f"{k} ≠ 0 — 기울인 목표는 배포 목표 입력이 모른다")
    notes = [f"첫 목표 기본 = 리셋 때 컵 + (0, 0, {0.5 * (z[0] + z[1]):.2f}) — 학습 분포 xy ±{float(env['goal_first_xy_range'])} m · z {z}",
             f"목표 입력: 박스 밖 거부 · 먼 목표는 축마다 ±{float(env['goal_delta_distance'])} m 이내 중간 목표로 나눔",
             f"목표 달성 = 키포인트 최대거리 ≤ tol_floor {float(env['tol_floor'])} m 누적 {int(env['goal_success_steps'])} 스텝 + 쥠"
             f"(엄지 AND 다른 손가락 촉각 > {float(env['contact_force_threshold'])} N — 학습은 첫마디+손끝 컵 접촉, 실기는 손끝 촉각)"]
    sim_only = {k: env[k] for k in SIM_ONLY_RESPONSE_KEYS if k in env}
    if sim_only:                                  # hdgp real_response.py — PD 앞단에서 sim 이 실기 반응을 흉내 낸 것(관측은 명령 목표)
        notes.append("학습 실기 반응(sim 전용, 배포 디코더는 따라 하지 않는다 — 실기가 스스로 낸다): "
                     + " · ".join(f"{k} {list(v) if isinstance(v, (list, tuple)) else v}" for k, v in sim_only.items()))
    dumped = (float(env["cup_half_height"]), float(env["cup_origin_offset_z"]))
    if max(abs(a - b) for a, b in zip(dumped, (half_h, origin_z))) > 1e-6:
        what = (f"스칼라(첫 물체 {_names(env)[0]} — 물체별 값은 object_geometry)" if env.get("per_env_geometry")
                else '+'.join(env.get('object_names') or ()) or env.get('object_name', 'shaker'))
        notes.append(f"컵 {what}: 반높이 {half_h:.4f} · 원점 높이 {origin_z:.4f} m — env.yaml 덤프"
                     f"({dumped[0]:.4f} · {dumped[1]:.4f})는 hydra 오버라이드 전 기본 shaker 값이라 쓰지 않았다")
    if not bool(cfg_a.get("normalize_input", False)):
        notes.append("normalize_input false")
    obj_names = object_ids(env)
    tactile = _actor_tactile(env)
    base = obs_dim_of(tactile=tactile)
    if not tactile:
        notes.append(f"★actor 촉각 없음(actor_tactile_obs False): 관측 {base}(+물체 one-hot) — 손끝 촉각은 critic 에만. "
                     "쥠 멈춤 · 쥔 판정은 여전히 촉각 · 관절 힘을 쓴다(디코더 · cup_attach)")
    if obj_names and not _dumped_obs_ok(env, extra_obs):
        raise RhAgltError(f"object_id_obs: observation_space {env['observation_space']} ≠ {base}(+{len(obj_names)})")
    geo = {k: object_geometry(k, float(env.get("cup_scale", 1.0))) for k in obj_names}
    if obj_names:
        notes.append(f"★다물체 한 정책: 관측 {base} + 물체 one-hot {len(obj_names)} (id 순 {obj_names}) — 에피소드 reset 때 "
                     "노드 파라미터 object 로 물체를 고른다(모르면 reset 거부). 컵 반높이 · 목표 박스 높이는 물체를 따른다: "
                     + " · ".join(f"{k} 반높이 {g['half_height']:.4f} 원점 {g['origin_offset_z']:.4f}" for k, g in geo.items()))
    robust = {k: env[k] for k in ("cmd_delay_adr", "cmd_delay_adr_mix", "cmd_delay_adr_scale_end", "fpp_enable",
                                  "fpp_attach_enable", "critic_delay_obs") if env.get(k)}
    if obj_names and robust:
        notes.append("학습 강건화(sim 전용, 배포 디코더 · 관측은 따라 하지 않는다): "
                     + " · ".join(f"{k} {v}" for k, v in robust.items()))
    hand_cmd, adm = hand_command_of(env)
    block, grasp_rule, lead_cap = hand_block_of(env)
    hold = grasp_hold_of(env, block)
    judge = grasp_judge_of(env, block)
    if judge:
        grasp_rule = judge.pop("grasp_rule")
        notes.append(f"★쥔 규칙(10.14 {env.get('hand_grasp_judge')}): 판정 축 {judge['grasp_judge_axes']} 중 래치 "
                     f"{judge['grasp_min_axes']} 개 이상" + (f" AND 물체 원점-손바닥 < {judge['grasp_near_m']} m"
                                                         if judge['grasp_near_m'] > 0 else ""))
    if hold:
        notes.append(f"★쥔 판정 유지(10.13 {hold['grasp_hold_rule']}): 막힘 쥔 판정이 한 번 참이면 유지 — 실측 손가락이 쥔 자세보다 "
                     f"{hold['grasp_release_open_rad']} rad 이상 열리고 래치 없음 · 목표 따라감이 {hold['grasp_release_steps']} 틱 연속이면 풀림")
    if block:
        notes.append(f"★쥠 멈춤 · 쥔 판정 = 막힘 신호(hand_block {block} · {grasp_rule}) — 손끝 촉각 · 접촉력 문턱 대신. "
                     f"sim DR(배포는 따라 하지 않는다): G {env.get('hand_block_g_rad')} · ε {env.get('hand_block_eps_rad')} · "
                     f"N {env.get('hand_block_n_ticks')} · 누락 {env.get('hand_block_miss_p')} · 쥔 판정 지연 "
                     f"{env.get('hand_grasp_delay_steps')} 틱 · firmware_stop {env.get('hand_firmware_stop_enable', True)}")
    arm_hold = {"arm_hold_rise_m": float(env.get("arm_hold_rise_m", 0.0) or 0.0),
                "arm_hold_release_m": float(env.get("arm_hold_release_m", 0.0) or 0.0),
                "arm_hold_release_steps": int(env.get("arm_hold_release_steps", 30) or 30)}
    if arm_hold["arm_hold_rise_m"] > 0.0:
        notes.append(f"★팔 들림 래치(10.18 arm_hold_rise_m): 관측 컵 z 가 처음 높이(정책 첫 스텝)보다 +{arm_hold['arm_hold_rise_m']} m "
                     f"넘으면 그 스텝의 팔 목표를 래치 → 팔 목표 = EMA(직전, 래치 + k_arm·a) · 해제 들림 < "
                     f"{arm_hold['arm_hold_release_m']} m {arm_hold['arm_hold_release_steps']} 스텝 연속(0 = 풀지 않음)")
    arm_start = str(env.get("aglt_start") or "home")
    # 사전 파지 범위 — ready 시작 정책(rcg)은 시작 검사에, 같은 env 의 접근 정책(ap05, home 시작)은 넘김 끝 판정에 쓴다
    has_ready = bool(env.get("ready_pre_r_mult")) and bool(env.get("ready_pre_h_m"))
    ready_range: dict = ({"ready_r_mult": [float(v) for v in env["ready_pre_r_mult"]], "ready_h_m": [float(v) for v in env["ready_pre_h_m"]]}
                         if has_ready else {})
    if arm_start == "ready":
        notes.append("★시작 자세 = ReadyApproach 사전 파지(10.19 aglt_start=ready): 학습은 홈 → 스크립트 접근 · 정착 뒤 정책이 넘겨받는다"
                     "(그동안 기록 행동 = 팔 0 · 손 폄). 배포는 사전 파지 고리 근처에서만 reset 을 받고 대기 동안 팔을 실측 자리에 둔다"
                     " — 앞 단계(ap05 접근 정책 · 1-B reach)가 그 자리까지 데려와야 한다")
    if hand_cmd == "admittance":
        notes.append("손 = 어드민턴스 전제(hand_adm_enable): pd 가 손 목표를 /hand_<s>/angle_target 으로 보낸다 — 학습 "
                     + " · ".join(f"{k} {v:g}" for k, v in adm.items()) + "(엄지 회전은 위치만 — 드라이버 joints 와 같다)")
    c = RaContract(
        schema=SCHEMA, task=TASK_PREFIX.format(p), run_dir=str(run_dir), checkpoint=str(checkpoint),
        checkpoint_md5=hashlib.md5(Path(checkpoint).read_bytes()).hexdigest(),
        env_yaml_sha1=hashlib.sha1(env_p.read_bytes()).hexdigest(),
        agent_yaml_sha1=hashlib.sha1(agent_p.read_bytes()).hexdigest(), asset=asset, profile=str(env["profile_name"]),
        policy_hz=1.0 / (float(env["sim"]["dt"]) * int(env["decimation"])), episode_s=float(env["episode_length_s"]),
        obs_dim=base + len(obj_names) if obj_names else int(env["observation_space"]), action_dim=int(env["action_space"]),
        obs_clip=float(agent["params"]["env"].get("clip_observations", 5.0)),
        action_clip=float(agent["params"]["env"].get("clip_actions", 1.0)), recurrent=bool(net.get("rnn")),
        hold_steps=int(env["hold_steps"]), k_arm=float(env["k_arm"]), arm_ema=float(env["arm_ema"]),
        hand_ema=float(env["hand_ema"]), hand_full_range_s=float(env["hand_full_range_s"]),
        freeze_threshold_n=float(env["contact_freeze_threshold"]), joint_err_norm=float(env["joint_err_norm"]),
        tactile_clip_n=float(env["tactile_obs_clip_n"]), tactile_tanh_n=float(env["tactile_obs_tanh_n"]),
        cup_half_height=half_h, goal_offset=[0.0, 0.0, 0.5 * (z[0] + z[1])],
        hand_obs_order_source="profile(hdgp rh_aglt_env hand_ids = 이름 순)", sides={"arm": s}, notes=notes,
        goal_box_min=[ctr[0] - hw[0] - m, ctr[1] - hw[1] - m, z0 + zb[0]],
        goal_box_max=[ctr[0] + hw[0] + m, ctr[1] + hw[1] + m, z0 + zb[1]],
        goal_first_xy_range=float(env["goal_first_xy_range"]), goal_first_z_range=z,
        goal_delta_distance=float(env["goal_delta_distance"]), goal_success_steps=int(env["goal_success_steps"]),
        goal_tol=float(env["tol_floor"]), grasp_threshold_n=float(env["contact_force_threshold"]),
        hand_command=hand_cmd, hand_admittance=adm, object_names=obj_names, object_geometry=geo,
        goal_force_consecutive=bool(env.get("goal_force_consecutive", False)), actor_tactile_obs=tactile,
        hand_block=block, grasp_rule=grasp_rule, hand_lead_cap_rad=lead_cap, **hold, **judge, **arm_hold,
        arm_start=arm_start, **ready_range)
    validate(c)
    return c


#: 10.12 grasping 확정 명목값(배포가 쓰는 값) — sim 은 env 마다 이 둘레 DR(hand_block_*_rad · n_ticks 범위)
HAND_BLOCK_NOMINAL = {"g_rad": 0.0218, "eps_rad": 0.0109, "n_ticks": 3}
HAND_BLOCK_AXES = ["thumb_flex", "index", "middle", "ring", "pinky"]
#: GripAssist C 상한 — 엄지 굽힘 편 끝(0.046~0.049 rad) 근처 값만 뜻이 있다. 크게 주면 엄지가 굽은 채여도 풀린다
THUMB_OPEN_FLOOR_MAX = 0.15


def hand_block_of(env: Mapping) -> tuple[dict, str, float]:
    """hdgp hand_block_enable → (hand_block, grasp_rule, hand_lead_cap_rad). 꺼져 있으면 ({}, "", 0.0)."""
    if not bool(env.get("hand_block_enable", False)):
        return {}, "", 0.0
    # ★10.14 위치제어 런(hdgp hand_block_pos): 축별 G(hand_block_g_axis_rad, 엄지굽힘·검지·중지·약지·새끼 순, 0 = env 의 g_rad)
    #   · 여는 쪽으로 안 풀림. 이런 런은 g_rad DR 을 4~10 칸으로 낮춰 "실기 G 는 측정 뒤 이 범위에서 고른다"(hand_block_pos.py)
    #   → 명목 12 칸이 범위 밖이면 범위 끝으로 붙인다(10.12 실기 검증값에 가장 가까운 학습 값). 축별 G 가 없는 옛 런은 그대로 거부.
    #   축별 G 커리큘럼(hand_block_g_axis_steps)은 끝값을 쓴다 — 체크포인트가 끝까지 갔는지는 학습 summaries diag/g_axis_frac 로 본다.
    gax = [float(v) for v in (env.get("hand_block_g_axis_rad") or ())]
    if gax and (len(gax) != len(HAND_BLOCK_AXES) or any(not np.isfinite(v) or v < 0.0 for v in gax)):
        raise RhAgltError(f"hand_block_g_axis_rad {gax} — {HAND_BLOCK_AXES} 순 {len(HAND_BLOCK_AXES)} 개, 0 이상")
    if any(float(v) != 0.0 for v in (env.get("hand_block_hold_lead_rad") or (0.0, 0.0))):
        raise RhAgltError(f"hand_block_hold_lead_rad {list(env['hand_block_hold_lead_rad'])} — 배포 막힘에 유지 앞섬이 없다")
    nominal = dict(HAND_BLOCK_NOMINAL)
    judge = str(env.get("hand_grasp_judge", "block"))
    # 10.14 위치제어 판정 런(G 4~10 칸). ★10.18 contact 판정 위치제어 런(rg3wg)도 같은 G DR 로 학습했다 — 막힘은 쥠 멈춤에만 쓴다
    pos_judge = judge in POS_JUDGES or (judge == "contact" and not bool(env.get("hand_adm_enable", False)))
    for k, lo_hi in (("hand_block_g_rad", "g_rad"), ("hand_block_eps_rad", "eps_rad"), ("hand_block_n_ticks", "n_ticks")):
        lo, hi = (float(v) for v in env[k])
        end = env.get(f"{k}_end") or ()
        if end and bool(env.get("adr_env_mix", False)) and bool(env.get("enable_adr", False)):
            # ★10.17 rg3wo: hdgp adr_env_mix 는 env 마다 세기 s~U(0, 진행도) 로 시작 범위 → 종점 범위(*_end)를 선형 보간한다
            #   (rh_aglt_env.py :542-548, HandBlock p_end). 끝까지 간 런은 두 범위를 잇는 구간 전체를 봤다 → 합집합으로 덮는지 본다.
            #   진행도가 1 에 닿았는지는 env.yaml 로 알 수 없다 — 카드(policy.yaml)의 학습 끝 epoch 로 사람이 확인한다.
            if len(end) != 2:
                raise RhAgltError(f"{k}_end {list(end)} — [하한, 상한] 두 값이어야 한다")
            lo, hi = min(lo, float(end[0])), max(hi, float(end[1]))
        if lo - 1e-9 <= nominal[lo_hi] <= hi + 1e-9:
            continue
        if (gax or pos_judge) and lo_hi == "g_rad":
            nominal["g_rad"] = round(min(max(nominal["g_rad"], lo), hi), 4)
            continue
        raise RhAgltError(f"{k} {list(env[k])} 가 배포 명목값 {lo_hi} {HAND_BLOCK_NOMINAL[lo_hi]} 를 안 덮는다")
    if bool(env.get("hand_firmware_stop_enable", True)):
        raise RhAgltError("hand_block_enable 인데 hand_firmware_stop_enable 이 켜져 있다 — 실기 펌웨어 힘 멈춤은 증거가 없다(10.12)")
    cap = float(env.get("hand_lead_cap_rad", 0.0))
    block = dict(nominal, lag_ticks=int(env["hand_block_lag_ticks"]), axes=list(HAND_BLOCK_AXES))
    if gax:
        block["g_axis_rad"] = dict(zip(HAND_BLOCK_AXES, gax))
    if not bool(env.get("hand_block_release_on_open", True)):
        block["release_on_open"] = False
    return block, HB.GRASP_RULES[0], cap


#: hdgp hand_grasp_judge 중 래치 축 수 규칙(hand_block_pos) — "block" · "contact" 는 옛 규칙(엄지굽힘 AND 손가락 하나)
POS_JUDGES = ("pos", "opp")
#: hdgp hand_block_judge_axes 이름 → 계약 axes 이름
JUDGE_AXIS_OF = {"thumb_2": "thumb_flex", "index": "index", "middle": "middle", "ring": "ring", "pinky": "pinky"}


def grasp_judge_of(env: Mapping, block: dict) -> dict:
    """hdgp hand_grasp_judge pos · opp → 계약 키 dict(grasp_rule · grasp_judge_axes · grasp_min_axes · grasp_near_m).
    block · contact · 키 없음이면 {}(옛 규칙). hand_grasp_near_m 은 hdgp 에서 막힘 판정 전체에 곱하므로 규칙과 같이 옮긴다."""
    judge = str(env.get("hand_grasp_judge", "block"))
    near = float(env.get("hand_grasp_near_m", 0.0) or 0.0)
    if judge not in POS_JUDGES:
        if near > 0.0 and block:
            raise RhAgltError(f"hand_grasp_near_m {near} 인데 hand_grasp_judge {judge!r} — 배포 옛 규칙에는 거리 조건이 없다")
        return {}
    if not block:
        raise RhAgltError(f"hand_grasp_judge {judge!r} 인데 hand_block_enable 이 꺼져 있다")
    if judge == "pos":
        raw = env.get("hand_block_judge_axes") or tuple(JUDGE_AXIS_OF)
        bad = [k for k in raw if str(k) not in JUDGE_AXIS_OF]
        if bad:
            raise RhAgltError(f"hand_block_judge_axes {list(raw)} — {list(JUDGE_AXIS_OF)} 중")
        axes = [JUDGE_AXIS_OF[str(k)] for k in raw]
        k = int(env.get("hand_block_min_axes", 2))
    else:                                                # opp = 네 손가락(엄지굽힘 제외, hdgp HandBlockOpp)
        axes = [a for a in HAND_BLOCK_AXES if a != "thumb_flex"]
        k = int(env.get("hand_block_min_axes", 1))
    if not (np.isfinite(near) and near >= 0.0):
        raise RhAgltError(f"hand_grasp_near_m ≥ 0: {near}")
    return {"grasp_rule": HB.GRASP_RULES[1], "grasp_judge_axes": axes, "grasp_min_axes": k, "grasp_near_m": near}


def grasp_hold_of(env: Mapping, block: dict) -> dict:
    """hdgp grasp_hold_rule → 계약 키 dict. "none"(기본) · 키 없음이면 {}(10.12 계약 그대로)."""
    rule = str(env.get("grasp_hold_rule", "none"))
    if rule == "none":
        return {}
    if not block:
        raise RhAgltError(f"grasp_hold_rule {rule!r} 인데 hand_block_enable 이 꺼져 있다")
    return {"grasp_hold_rule": rule, "grasp_release_open_rad": [float(v) for v in env["grasp_release_open_rad"]],
            "grasp_release_steps": int(env["grasp_release_steps"])}


def select_object(c: RaContract, name: str | None) -> RaContract:
    """에피소드에서 지금 잡는 물체 → 그 물체의 계약 보기. 다물체 계약이 아니면 그대로(name 무시하지 않고 비어 있어야 한다).

    hdgp rh_aglt_env(10.11 per_env_geometry): 키포인트 반높이 = 물체 반높이, 목표 박스 z = 첫 물체 원점 기준 박스 +
    (물체 원점 − 첫 물체 원점)(GoalState box_z_shift). 첫 목표 오프셋 · xy 박스는 그대로."""
    if not c.object_names:
        if name:
            raise RhAgltError(f"물체 {name!r}: 이 계약은 물체 관측이 없다(단일 물체 정책) — object 를 비워라")
        return c
    if not name or name not in c.object_names:
        raise RhAgltError(f"물체 {name!r} 를 모른다 — 이 정책은 {list(c.object_names)} 중 하나(노드 파라미터 object · "
                          "에피소드 단계 policy_object)가 있어야 돈다")
    g, g0 = c.object_geometry[name], c.object_geometry[c.object_names[0]]
    dz = float(g["origin_offset_z"]) - float(g0["origin_offset_z"])
    shift = lambda b: [*b[:2], b[2] + dz] if b else b  # noqa: E731
    return with_run(c, active_object=name, cup_half_height=float(g["half_height"]),
                    goal_box_min=shift(c.goal_box_min), goal_box_max=shift(c.goal_box_max))


# ---------------------------------------------------------------- 디코더
@dataclass
class _State:
    arm_target: np.ndarray
    hand_target: np.ndarray


@dataclass(frozen=True)
class GripAssist:
    """★10.13 배포 전용 손 보조(사용자 결정 "a, b 둘 다 — 제어가 정책을 충실히 따라가게", s2r_1st). 기본 = 꺼짐(sim 과 같다).

    실기 bag(20261013 164845 오른손 · 165303 왼손):
      A unfreeze_on_grasp — 막힘 쥠 멈춤이 첫 접촉 자리에서 손가락 목표를 묶었다(오른손 검지: 정책 0.60~0.76 rad,
        보낸 목표 0.51 에서 고정 → 쥐다가 들 때 빠짐). 쥔 판정이 켜진 뒤에는 고정을 풀어 정책 목표까지 닫는다.
        힘 상한은 pd 앞섬 상한(0.2 rad)과 C 마스터 어드민턴스(f_max 800 g)가 그대로 맡는다.
      B thumb_support_g — 엄지는 받치는 링크라 굽힘 명령이 거의 없고(0.00~0.04 rad), 실기 엄지 굽힘은 편 끝이
        0.046(왼) · 0.049(오) rad 라 막힘 래치가 안 걸린다(왼손 쉐이커를 들고도 쥔 판정 0/899 틱 → lift_hold 멈춤 없음).
        엄지 굽힘 관절 힘(g)이 이 값을 넘으면 엄지 래치 대신 받침으로 본다(손가락 막힘 래치는 그대로 필요).
        bag: 빈손 −33~+28 g, 쥠 95~474 g. 0 = 끔.
      C thumb_open_floor_rad — (bag 20261013_223839 오른손) 컵이 엄지 굽힘을 편 끝(0.049)보다 더 펴지게 밀어 쥔 자세
        q_g 가 0.038 로 남았다. 쥔 판정 유지(GraspHold)는 '엄지가 q_g 보다 더 굽지 않았을 것'을 요구해 손을 다 편 뒤에도
        (엄지 0.049) 3.5 s 동안 안 풀렸다 → 컵 감시가 뗀 뒤 다시 붙지 않음 → 두 번째 쥠으로 9.6 cm 들고도 실행기가 쥠 실패로
        보고 손을 펴 컵을 떨굼. q_g 의 엄지 굽힘을 이 값보다 더 편 쪽으로 기록하지 않는다(편 끝 + 여유). 0 = 끔.
    """

    unfreeze_on_grasp: bool = False
    thumb_support_g: float = 0.0
    thumb_open_floor_rad: float = 0.0
    # ★10.19 D squeeze — 사용자: "sim 은 세게 쥐면 물리가 터져 적당히 쥐게 학습했다. 실기는 그럴 이유가 없으니 제어로 더 세게 쥐게".
    #   bag 20261019_133007(왼손 neo03 · source240): 접근은 맞았는데 손가락을 0.55~0.85 rad(grip 1.08 · 0.85)까지만 닫고
    #   목표에 그대로 도달(막힘 0) → 첫 시도는 병을 못 감싸고 팔만 들림, 둘째 시도는 10 cm 들었다가 정책이 검지 · 중지를 풀어 떨굼.
    #   ★10.19 저녁 다시 봄(같은 bag): 정책 목표(EMA)는 쥔 자세의 0.7~0.85 까지 가고 손가락이 그 목표에 그대로 닿아
    #     (hand_err ≈ 0) 힘이 2~50 g 뿐 — 5.9 s 에 잠깐 1.3 kg 눌렀다가 정책이 0.1 rad 풀어 힘이 0 이 됐다. = 덜(약하게) 쥠.
    #   사용자 10.19: "정책은 원래 상태를 보고, 제어가 움직임을 보조해 더 닫게" → 보조는 보내는 명령에만 얹는다. 정책 자기 목표
    #     (law 상태 · 관측 hand_err · 막힘 래치)는 그대로 둔다 — 덮어쓰면 다음 스텝 EMA 가 보조 값에서 시작해 정책 뜻과 갈린다.
    #   켜기: 정책 자기 목표의 네 손가락 평균 닫힘(open 0 · grip 1)이 squeeze_trigger_frac 이상이고 컵이 손바닥 squeeze_near_m 안.
    #   보조: 보내는 네 손가락 = 정책 목표 + 닫는 쪽 boost. boost 는 squeeze_rate_rad_s 로 squeeze_boost_rad 까지 오르고, 들리기
    #     전에는 정책이 손을 펴는 만큼 줄어든다(닫힘 trigger → 1 배, release → 0 배). 팔 들림 래치 동안에는 보낸 명령보다 펴지 않는다
    #     (bag: 10 cm 든 뒤 정책이 검지 · 중지를 풀어 떨굼). 앞섬 ≤ 실측 + hand_lead_cap(= pd HAND_MAX_LEAD) · 관절 끝 앞.
    #     힘 상한은 손 펌웨어 FORCESET(pd hw_force)가 맡는다. 엄지는 정책 그대로.
    #   풀기: (헛잡음) 정책이 닫는 중(trigger 이상)이고 boost 가 다 오른 뒤 네 손가락 모두 보낸 명령을 squeeze_miss_rad 안으로
    #     따라오면(막는 것이 없음) squeeze_miss_s · (정책이 다시 시도) 들리지 않았고 정책 닫힘이 squeeze_release_frac 아래로 squeeze_release_s.
    #     푼 뒤에는 정책이 손을 반 넘게 편 뒤에야 다시 켠다. 켜진 동안 둘 이상의 손가락이 명령보다 squeeze_miss_rad 넘게 뒤처진 채
    #     멈춰 있으면(밀고 있음) 쥔 것으로 본다(엄지 래치 · 받침 없이도).
    squeeze: bool = False
    squeeze_trigger_frac: float = 0.5
    squeeze_near_m: float = 0.10
    squeeze_rate_rad_s: float = 1.5
    squeeze_boost_rad: float = 0.2
    squeeze_release_frac: float = 0.15
    squeeze_release_s: float = 1.0
    squeeze_miss_rad: float = 0.04
    squeeze_miss_s: float = 0.25

    def __post_init__(self) -> None:
        if not (np.isfinite(self.thumb_support_g) and self.thumb_support_g >= 0.0):
            raise RhAgltError(f"thumb_support_g ≥ 0(0 = 끔): {self.thumb_support_g}")
        if not (np.isfinite(self.thumb_open_floor_rad) and 0.0 <= self.thumb_open_floor_rad <= THUMB_OPEN_FLOOR_MAX):
            raise RhAgltError(f"thumb_open_floor_rad 0~{THUMB_OPEN_FLOOR_MAX} rad(0 = 끔): {self.thumb_open_floor_rad}")
        checks = (("squeeze_trigger_frac", 0.05, 1.0), ("squeeze_near_m", 0.0, 0.3), ("squeeze_rate_rad_s", 0.05, 5.0),
                  ("squeeze_boost_rad", 0.02, 0.5),
                  ("squeeze_release_frac", 0.0, 0.9), ("squeeze_release_s", 0.0, 10.0), ("squeeze_miss_rad", 0.005, 0.2),
                  ("squeeze_miss_s", 0.0, 5.0))
        for name, lo, hi in checks:
            v = float(getattr(self, name))
            if not (np.isfinite(v) and lo <= v <= hi):
                raise RhAgltError(f"{name} {lo}~{hi}: {v}")
        if self.squeeze_release_frac >= self.squeeze_trigger_frac:
            raise RhAgltError(f"squeeze_release_frac {self.squeeze_release_frac} < squeeze_trigger_frac {self.squeeze_trigger_frac} 이어야")


class RaDecoder:
    """행동 13 → (팔 q* 7, 손 q* 6). 직전 목표를 든다 — 에피소드 시작에 reset()."""

    def __init__(self, c: RaContract, assist: GripAssist | None = None) -> None:
        self.c = c
        self.law = hand_law(c)
        s = c.side()
        self._lo, self._hi, self._home = (np.asarray(v, float) for v in (s.arm_lo, s.arm_hi, s.arm_home))
        self.state: dict[str, _State] = {}
        self.block = make_block(c) if c.hand_block else None
        self.hold = (HB.GraspHoldNp(self.block, c.grasp_release_open_rad, c.grasp_release_steps)
                     if self.block is not None and c.grasp_hold_rule else None)
        self.assist = assist or GripAssist()
        self._support = False                         # 직전 step 의 엄지 받침(B) — hold 는 직전 틱 판정을 받는다
        self._near = True                             # 직전 step 의 물체-손바닥 거리 조건(grasp_near_m 이 0 이면 늘 참)
        self.arm_latched = False                      # ★10.19 팔 들림 래치(계약 arm_hold_rise_m > 0 일 때만)
        self.arm_rise = float("nan")                  # 관측 컵 z − 처음 높이(직전 step)
        self._q_latch = self._home.copy()
        self._below = 0
        self._z0: float | None = None
        names = list(s.hand_joints)                   # ★10.19 D squeeze — 네 손가락(검지 · 중지 · 약지 · 새끼 첫 마디)
        self._sq_f = np.array([i for i, n in enumerate(names) if any(n.endswith(f"_{f}_1") for f in ("index", "middle", "ring", "pinky"))],
                              int)
        o, g = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
        sgn = np.sign(g - o)
        self._sq_sgn = np.where(sgn == 0, 1.0, sgn)
        self._sq_open, self._sq_span = o, np.where(np.abs(g - o) > 1e-9, g - o, 1.0)
        margin = 0.02                                 # pd hand_safe_target 처럼 관절 끝 바로 앞
        self._sq_close = np.where(self._sq_sgn > 0, np.asarray(s.hand_lim_hi, float) - margin, np.asarray(s.hand_lim_lo, float) + margin)
        self._sq_lead = float(c.hand_lead_cap_rad) if c.hand_lead_cap_rad > 0.0 else 0.2
        self.squeeze_counts = {"on": 0, "miss": 0, "open": 0}
        self.reset()

    def reset(self, arm_start: Sequence[float] | None = None, hand_start: Sequence[float] | None = None) -> None:
        """기본 = 홈 · 편 손(rh_aglt 학습 리셋). rh_place 는 인계 순간의 팔 · 손 목표에서 시작한다(sim 뱅크 arm_q_target · hand_target)."""
        arm = self._home if arm_start is None else np.clip(np.asarray(arm_start, float), self._lo, self._hi)
        hand = self.c.side().hand_open if hand_start is None else hand_start
        self.state = {"arm": _State(np.array(arm, float), np.array(hand, float))}
        # ★10.20 대기(active=False) 동안 팔 목표 = reset 자세 — 기본은 홈(sim q_idle = q_home 그대로), arm_start ready 는 사전 파지
        #   자리 실측(sim 정착 뒤 Mover 명령). 예전엔 늘 홈이라 ready 정책을 대기 동안 홈으로 끌고 갔다.
        self._idle = np.array(arm, float)
        if self.block is not None:
            self.block.reset()
        if self.hold is not None:
            self.hold.reset()
        self._support = False
        self._near = True
        self.arm_latched, self.arm_rise, self._below, self._z0 = False, float("nan"), 0, None
        self._q_latch = self._home.copy()
        self.squeeze_on = False                       # ★10.19 D squeeze — 에피소드마다 처음부터
        self._sq_cmd: np.ndarray | None = None        # 직전에 보낸 네 손가락 명령(보조가 켜진 동안)
        self._sq_meas: np.ndarray | None = None       # 직전 실측(멈춤 판정)
        self._sq_d, self._sq_boost, self._sq_blocked = 0.0, 0.0, 0
        self._sq_armed, self._sq_open_n, self._sq_miss_n = True, 0, 0
        self._sq_frac = float("nan")
        self._sq_last = ""
        self.squeeze_counts = {"on": 0, "miss": 0, "open": 0}

    def _squeeze(self, target: np.ndarray, hand_q: Sequence[float] | None, obj_palm_m: float | None) -> np.ndarray:
        """D squeeze 한 스텝(active 스텝만) — 정책 자기 목표(law 상태, 바꾸지 않는다) → 손에 보낼 명령. 끄면 그대로."""
        A, F = self.assist, self._sq_f
        if not A.squeeze or F.size != 4:
            return target
        s = self._sq_sgn[F]
        frac = float(np.mean((target[F] - self._sq_open[F]) / self._sq_span[F]))   # 정책 자기 목표의 닫힘(open 0 · grip 1)
        self._sq_frac = frac
        hz = float(self.c.policy_hz)
        meas = None if hand_q is None else np.asarray(hand_q, float).reshape(-1)
        mf = None if meas is None else meas[F]
        if self._sq_cmd is not None and mf is not None:     # 직전 명령보다 뒤처진 채 멈춘 손가락 = 무언가를 밀고 있다
            lag = (self._sq_cmd - mf) * s
            still = np.ones(4, bool) if self._sq_meas is None else np.abs(mf - self._sq_meas) < 0.006
            self._sq_blocked = int(np.sum((lag > A.squeeze_miss_rad) & still))
            free = bool(np.all(lag <= A.squeeze_miss_rad))
        else:
            self._sq_blocked, free = 0, False
        self._sq_meas = None if mf is None else mf.copy()
        if not self.squeeze_on:
            if not self._sq_armed and frac < 0.5 * A.squeeze_trigger_frac:
                self._sq_armed = True
            near = obj_palm_m is not None and bool(np.isfinite(obj_palm_m)) and float(obj_palm_m) <= A.squeeze_near_m
            if not (self._sq_armed and frac >= A.squeeze_trigger_frac and near):
                return target
            self.squeeze_on, self._sq_open_n, self._sq_miss_n, self._sq_d = True, 0, 0, 0.0
            self._sq_cmd, self._sq_blocked = None, 0
            self.squeeze_counts["on"] += 1
            self._sq_last = f"on frac {frac:.2f} · 손바닥 {float(obj_palm_m):.3f} m"
        self._sq_open_n = self._sq_open_n + 1 if (frac < A.squeeze_release_frac and not self.arm_latched) else 0
        full = self._sq_d >= A.squeeze_boost_rad - 1e-9
        self._sq_miss_n = self._sq_miss_n + 1 if (full and free and frac >= A.squeeze_trigger_frac) else 0   # 정책이 닫는 중일 때만
        why = ("open" if self._sq_open_n >= max(1, round(A.squeeze_release_s * hz))
               else "miss" if self._sq_miss_n >= max(1, round(A.squeeze_miss_s * hz)) else "")
        if why:
            self.squeeze_on, self._sq_cmd, self._sq_armed = False, None, False
            self._sq_d, self._sq_boost, self._sq_blocked = 0.0, 0.0, 0
            self.squeeze_counts[why] += 1
            self._sq_last = "off " + ("헛잡음(막는 것 없이 명령까지 닫힘)" if why == "miss" else "정책이 손을 폄(들리지 않음)")
            return target
        self._sq_d = min(A.squeeze_boost_rad, self._sq_d + A.squeeze_rate_rad_s / hz)
        span = A.squeeze_trigger_frac - A.squeeze_release_frac
        fade = 1.0 if self.arm_latched else float(np.clip((frac - A.squeeze_release_frac) / span, 0.0, 1.0))
        self._sq_boost = self._sq_d * fade
        q = target[F] + s * self._sq_boost
        if self.arm_latched and self._sq_cmd is not None:                       # 들린 동안 보낸 명령보다 펴지 않는다
            q = np.where((self._sq_cmd - q) * s > 0, self._sq_cmd, q)
        q = np.where((q - self._sq_close[F]) * s > 0, self._sq_close[F], q)        # 관절 끝 앞까지
        if mf is not None:                                                       # 앞섬 ≤ 실측 + lead(pd 와 같다)
            cap = mf + s * self._sq_lead
            q = np.where((q - cap) * s > 0, cap, q)
        q = np.where((target[F] - q) * s > 0, target[F], q)                       # 정책이 더 닫으면 정책 값
        self._sq_cmd = q.copy()
        out = target.copy()
        out[F] = q
        return out

    def squeeze_view(self) -> dict:
        """상태용 — D 를 끄면 {}."""
        if not self.assist.squeeze:
            return {}
        frac = None if not np.isfinite(self._sq_frac) else round(self._sq_frac, 2)
        return {"squeeze": {"on": bool(self.squeeze_on), "frac": frac, "armed": bool(self._sq_armed), "n": dict(self.squeeze_counts),
                            "boost_rad": round(float(self._sq_boost), 3), "pushing": int(self._sq_blocked), "last": self._sq_last}}

    def _arm_hold(self, q_prev: np.ndarray, a_arm: np.ndarray, q_inc: np.ndarray, cup_z: float | None) -> np.ndarray:
        """hdgp update_arm_latch + step_arm_hold 한 스텝(active 스텝만). q_prev = 직전 팔 목표 · q_inc = 증분 결과.
        처음 높이 z0 = 첫 active 스텝의 관측 컵 z(hdgp _fz_z0 — 정책이 처음 움직이는 스텝의 관측). 래치 스텝의 래치 자세는
        q_prev 라 그 스텝 값은 증분과 같다(튀지 않음). 유한하지 않은 컵 z 는 래치 · 해제 판정에 쓰지 않는다(해제 연속은 끊긴다)."""
        if cup_z is None:
            raise RhAgltError("arm_hold_rise_m 계약: 디코더에 관측 컵 z(cup_z)가 필요하다")
        z = float(cup_z)
        if self._z0 is None and np.isfinite(z):
            self._z0 = z
        rise = z - self._z0 if self._z0 is not None and np.isfinite(z) else float("nan")
        self.arm_rise = rise
        if rise > self.c.arm_hold_rise_m:             # nan 이면 거짓
            if not self.arm_latched:
                self._q_latch = np.array(q_prev, float)
            self.arm_latched = True
        if self.c.arm_hold_release_m > 0.0:
            low = self.arm_latched and rise < self.c.arm_hold_release_m
            self._below = self._below + 1 if low else 0
            if self._below >= max(1, int(self.c.arm_hold_release_steps)):
                self.arm_latched, self._below = False, 0
        if not self.arm_latched:
            return q_inc
        q_raw = np.clip(self._q_latch + self.c.k_arm * a_arm, self._lo, self._hi)
        return np.clip(self.c.arm_ema * q_raw + (1.0 - self.c.arm_ema) * q_prev, self._lo, self._hi)

    def arm_hold_view(self) -> dict:
        """상태용 — 계약에 래치가 없으면 {}."""
        if self.c.arm_hold_rise_m <= 0.0:
            return {}
        rise = None if not np.isfinite(self.arm_rise) else round(float(self.arm_rise), 4)
        z0 = None if self._z0 is None else round(float(self._z0), 4)    # 처음 높이 — 실기에서 첫 컵 z 가 튀었는지 본다
        return {"arm_latched": bool(self.arm_latched), "arm_rise": rise, "arm_z0": z0}

    def _thumb_support(self, joint_force: Sequence[float] | None) -> bool:
        """B — 엄지 굽힘 관절 힘(RaMeas.joint_force[0], g) > thumb_support_g. 힘이 없으면(결손) 받침 아님."""
        if self.assist.thumb_support_g <= 0.0 or joint_force is None:
            return False
        f = np.asarray(joint_force, float).reshape(-1)
        return bool(f.size >= 1 and np.isfinite(f[0]) and f[0] > self.assist.thumb_support_g)

    def _hold_floor(self) -> np.ndarray | None:
        """C — 쥔 판정 유지의 q_g 하한(엄지 굽힘만, 나머지 nan). 끄면 None(sim 과 같다)."""
        if self.assist.thumb_open_floor_rad <= 0.0 or self.block is None:
            return None
        f = np.full(self.block.h, np.nan)
        f[self.block.thumb_flex] = self.assist.thumb_open_floor_rad
        return f

    def _raw_grasped(self) -> bool:
        """막힘 쥔 판정(계약 쥔 규칙) — B 가 켜져 있으면 엄지 래치 대신 엄지 받침도 인정한다. grasp_near_m 이면 거리 조건을 곱한다
        (hdgp: 막힘 판정 AND 물체-손바닥 < near 를 판정 기록에 넣고, 유지 규칙은 그 기록을 받는다)."""
        b = self.block
        if b is None:
            return False
        squeezed = self.squeeze_on and self._sq_blocked >= 2                  # ★10.19 D — 보조 명령을 둘 이상이 밀고 있음 = 쥠
        return bool((b.grasped() or (self._support and b.latch[b.fingers].any()) or squeezed) and self._near)

    def _near_of(self, obj_palm_m: float | None) -> bool:
        """grasp_near_m 거리 조건 — 0 이면 참. 켜져 있는데 거리가 없거나 유한하지 않으면 거짓(쥔 것으로 치지 않는다)."""
        if self.c.grasp_near_m <= 0.0:
            return True
        return obj_palm_m is not None and bool(np.isfinite(obj_palm_m)) and float(obj_palm_m) < self.c.grasp_near_m

    def step(self, action: Sequence[float], *, active: bool, tactile_n: Sequence[float] | None = None,
             hand_q: Sequence[float] | None = None,
             joint_force: Sequence[float] | None = None,
             obj_palm_m: float | None = None,
             cup_z: float | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """hand_q = 실측 손 각도(계약 hand_joints 순, rad) — hand_block 계약이면 반드시(막힘 신호 쥠 멈춤).
        joint_force = RaMeas.joint_force(엄지 굽힘 먼저, g) — GripAssist B 만 쓴다.
        obj_palm_m = 지금 |물체 원점 − 손바닥| [m](RaMeas cup_pos · palm_pos) — 계약 grasp_near_m > 0 이면 반드시.
        cup_z = 이 관측의 컵 z [m](RaMeas cup_pos[2], base) — 계약 arm_hold_rise_m > 0 이면 반드시(팔 들림 래치)."""
        a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
        if a.size != 13 or not np.all(np.isfinite(a)):
            raise RhAgltError(f"행동은 유한한 13 개: {a.size}")
        st = self.state["arm"]
        if active:
            q_raw = np.clip(st.arm_target + self.c.k_arm * a[:7], self._lo, self._hi)
            q_inc = np.clip(self.c.arm_ema * q_raw + (1.0 - self.c.arm_ema) * st.arm_target, self._lo, self._hi)
            if self.c.arm_hold_rise_m > 0.0:          # ★10.19 팔 들림 래치(hdgp _pre_physics_step 957-966 순서)
                q_inc = self._arm_hold(st.arm_target, a[:7], q_inc, cup_z)
            st.arm_target = q_inc
        else:
            st.arm_target = self._idle.copy()
        if self.block is not None:                    # 10.12 막힘 쥠 멈춤 — 직전 q* 와 지금 실측(hdgp _pre_physics_step 순서)
            if hand_q is None:
                raise RhAgltError("hand_block 계약: 디코더에 실측 손 각도(hand_q)가 필요하다")
            if self.hold is not None:                 # 10.13 쥔 판정 유지 — 이번 block.step 전(sim tick 끝 색인)
                self.hold.step(self._raw_grasped(), st.hand_target, hand_q, floor=self._hold_floor())
            if self.c.grasp_near_m > 0.0 and obj_palm_m is None:
                raise RhAgltError("grasp_near_m 계약: 디코더에 물체-손바닥 거리(obj_palm_m)가 필요하다")
            freeze = self.block.step(st.hand_target, hand_q)
            self._support = self._thumb_support(joint_force)
            self._near = self._near_of(obj_palm_m)
            if self.assist.unfreeze_on_grasp and self.grasped():
                freeze = None                         # A — 쥔 뒤에는 정책 목표까지 닫는다(배포 전용)
        else:
            freeze = self.law.touch_to_freeze(tactile_n)
        st.hand_target = self.law.step(st.hand_target, a[7:], active=active, freeze=freeze)
        hand_cmd = st.hand_target                     # ★10.19 D squeeze(배포 전용, 끄면 그대로) — 보내는 명령에만 얹는다.
        if active:                                    #   정책 자기 목표(st.hand_target)는 law · 관측 hand_err · 막힘 래치가 그대로 쓴다
            hand_cmd = self._squeeze(st.hand_target, hand_q, obj_palm_m)
        return {"arm": (st.arm_target.copy(), hand_cmd.copy())}

    def grasped(self) -> bool | None:
        """막힘 신호 쥔 판정(직전 step 기준) — hand_block 계약이 아니면 None(호출자가 촉각 · 관절 힘 규칙을 쓴다)."""
        if self.block is None:
            return None
        return self.hold.hold if self.hold is not None else self._raw_grasped()


# ---------------------------------------------------------------- 관측
@dataclass(frozen=True)
class RaMeas:
    arm_q: np.ndarray           # 7
    arm_qd: np.ndarray          # 7
    hand_q: Mapping[str, float]  # 이름 → rad (구동 6)
    palm_pos: np.ndarray        # 3 (base)
    palm_R: np.ndarray          # 3×3
    tips: np.ndarray            # 5×3 (tip_bodies 순)
    cup_pos: np.ndarray         # 3 (base)
    cup_quat: np.ndarray        # 4 wxyz
    tactile_n: np.ndarray       # 5 (엄지 → 새끼, N)
    joint_force: np.ndarray | None = None   # 6 손 관절 힘(엄지 굽힘 · 엄지 회전 · 검지 · 중지 · 약지 · 새끼, g) — 10.04, 없으면 None


def quat_apply(q: Sequence[float], v: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(t) for t in q)
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def keypoints(c: RaContract, pos: Sequence[float], quat: Sequence[float]) -> np.ndarray:
    off = np.asarray(KEYPOINT_AXIAL_UNIT, float) * c.cup_half_height
    return np.asarray(pos, float)[None, :] + np.stack([quat_apply(quat, o) for o in off])


@dataclass(frozen=True)
class Goal:
    pos: np.ndarray
    quat: np.ndarray


def first_goal(c: RaContract, cup_pos: Sequence[float], cup_quat: Sequence[float]) -> Goal:
    return Goal(np.asarray(cup_pos, float) + np.asarray(c.goal_offset, float), np.asarray(cup_quat, float))


def build_obs(c: RaContract, m: RaMeas, dec: RaDecoder, goal: Goal, prev_action: Sequence[float],
              extra: Mapping[str, float] | None = None) -> np.ndarray:
    """관측 96(+물체 one-hot K) — 학습 `_actor_parts` 순서 그대로. 목표는 디코더의 지금 q*(직전 스텝 결과)."""
    s, st = c.side(), dec.state["arm"]
    hand_q = np.array([m.hand_q[j] for j in s.hand_joints], float)
    tips = np.asarray(m.tips, float).reshape(5, 3)
    parts = [m.arm_q, m.arm_qd, st.arm_target, hand_q,
             np.clip((st.hand_target - hand_q) / c.joint_err_norm, -1.0, 1.0),
             m.palm_pos, np.concatenate([m.palm_R[:, 0], m.palm_R[:, 1]]),
             m.cup_pos - m.palm_pos, quat_apply(m.cup_quat, np.array([0.0, 0.0, 1.0])), (tips - m.cup_pos).reshape(-1),
             (keypoints(c, m.cup_pos, m.cup_quat) - keypoints(c, goal.pos, goal.quat)).reshape(-1),
             goal.pos - m.palm_pos]
    if c.actor_tactile_obs:                         # 10.12 actor_tactile_obs False 면 이 칸이 없다(hdgp layout)
        parts.append(RH.tactile_obs(m.tactile_n, c.tactile_clip_n, c.tactile_tanh_n))
    parts.append(np.asarray(prev_action, float))
    if c.object_names:                              # ★10.12 물체 one-hot — 모르면 거부(조용한 0 벡터 금지)
        if c.active_object not in c.object_names:
            raise RhAgltError(f"물체 one-hot: active_object {c.active_object!r} ∉ {list(c.object_names)} — select_object 먼저")
        parts.append(np.eye(len(c.object_names))[c.object_names.index(c.active_object)])
    for name in c.extra_actor_obs:                  # 계열 고유 칸(rh_place latch) — 체인이 값을 준다(없으면 거부)
        if extra is None or name not in extra:
            raise RhAgltError(f"관측 '{name}' 값이 없다 — 체인이 extra 로 줘야 한다")
        parts.append(np.array([float(extra[name])]))
    layout = actor_layout(c)
    if len(parts) != len(layout):
        raise RhAgltError(f"관측 조각 {len(parts)} ≠ 레이아웃 {len(layout)}")
    for p, (name, w) in zip(parts, layout):
        if np.asarray(p).size != w:
            raise RhAgltError(f"관측 '{name}' 폭 {np.asarray(p).size} ≠ {w}")
    obs = np.concatenate([np.asarray(p, float).reshape(-1) for p in parts])
    return np.clip(np.nan_to_num(obs), -c.obs_clip, c.obs_clip)


def with_run(c: RaContract, **kw) -> RaContract:
    out = replace(c, **kw)
    validate(out)
    return out

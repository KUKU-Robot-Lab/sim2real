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
    # ★10.12 계열 고유 관측 칸(폭 1씩, 물체 one-hot 뒤) — rh_place 관측 래치("latch", hdgp extra_actor_obs). 값은 체인이 준다.
    extra_actor_obs: list = field(default_factory=list)

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
    hand_law(c)                                    # 손 자세 · 한계 검사
    if c.hand_command not in HAND_COMMANDS:
        raise RhAgltError(f"hand_command {c.hand_command!r} — {HAND_COMMANDS}")
    if (c.hand_command == "admittance") != bool(c.hand_admittance):
        raise RhAgltError("hand_command admittance 와 hand_admittance(학습 값)는 같이 있어야 한다")
    _validate_objects(c)
    _validate_block(c)


def _validate_block(c: RaContract) -> None:
    if not c.hand_block:
        if c.grasp_rule or c.grasp_hold_rule:
            raise RhAgltError(f"grasp_rule {c.grasp_rule!r} · grasp_hold_rule {c.grasp_hold_rule!r} 는 hand_block 과 같이 있어야 한다")
        return
    try:
        HB.HandBlockNp.from_contract(c.hand_block, c.side().hand_joints, c.side().hand_open, c.side().hand_grip)
    except HB.HandBlockError as e:
        raise RhAgltError(str(e)) from e
    if c.grasp_rule not in HB.GRASP_RULES:
        raise RhAgltError(f"grasp_rule {c.grasp_rule!r} — {HB.GRASP_RULES}")
    if c.grasp_hold_rule:
        if c.grasp_hold_rule not in HB.HOLD_RULES:
            raise RhAgltError(f"grasp_hold_rule {c.grasp_hold_rule!r} — {HB.HOLD_RULES}")
        try:
            HB.GraspHoldNp(HB.HandBlockNp.from_contract(c.hand_block, c.side().hand_joints, c.side().hand_open,
                                                        c.side().hand_grip), c.grasp_release_open_rad, c.grasp_release_steps)
        except HB.HandBlockError as e:
            raise RhAgltError(str(e)) from e
    from policy_control.pd_backends import HAND_MAX_LEAD
    if abs(float(c.hand_lead_cap_rad) - HAND_MAX_LEAD) > 1e-9:
        raise RhAgltError(f"hand_lead_cap_rad {c.hand_lead_cap_rad} ≠ 실기 pd HAND_MAX_LEAD {HAND_MAX_LEAD} — "
                          "막힘 신호를 학습한 sim 의 앞섬 상한이 실기와 달라진다")


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


def _names(env: Mapping) -> list[str]:
    names = [str(n) for n in (env.get("object_names") or ())] or [str(env.get("object_name", "shaker"))]
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
    return int(env.get("observation_space", 0)) in (base, base + k, base + k + extra, obs_dim_of(tactile=True))


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
    side = "left" if env["profile_name"] == "rh56f1_left" else "right"
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
    if hold:
        notes.append(f"★쥔 판정 유지(10.13 {hold['grasp_hold_rule']}): 막힘 쥔 판정이 한 번 참이면 유지 — 실측 손가락이 쥔 자세보다 "
                     f"{hold['grasp_release_open_rad']} rad 이상 열리고 래치 없음 · 목표 따라감이 {hold['grasp_release_steps']} 틱 연속이면 풀림")
    if block:
        notes.append(f"★쥠 멈춤 · 쥔 판정 = 막힘 신호(hand_block {block} · {grasp_rule}) — 손끝 촉각 · 접촉력 문턱 대신. "
                     f"sim DR(배포는 따라 하지 않는다): G {env.get('hand_block_g_rad')} · ε {env.get('hand_block_eps_rad')} · "
                     f"N {env.get('hand_block_n_ticks')} · 누락 {env.get('hand_block_miss_p')} · 쥔 판정 지연 "
                     f"{env.get('hand_grasp_delay_steps')} 틱 · firmware_stop {env.get('hand_firmware_stop_enable', True)}")
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
        hand_block=block, grasp_rule=grasp_rule, hand_lead_cap_rad=lead_cap, **hold)
    validate(c)
    return c


#: 10.12 grasping 확정 명목값(배포가 쓰는 값) — sim 은 env 마다 이 둘레 DR(hand_block_*_rad · n_ticks 범위)
HAND_BLOCK_NOMINAL = {"g_rad": 0.0218, "eps_rad": 0.0109, "n_ticks": 3}
HAND_BLOCK_AXES = ["thumb_flex", "index", "middle", "ring", "pinky"]


def hand_block_of(env: Mapping) -> tuple[dict, str, float]:
    """hdgp hand_block_enable → (hand_block, grasp_rule, hand_lead_cap_rad). 꺼져 있으면 ({}, "", 0.0)."""
    if not bool(env.get("hand_block_enable", False)):
        return {}, "", 0.0
    for k, lo_hi in (("hand_block_g_rad", "g_rad"), ("hand_block_eps_rad", "eps_rad"), ("hand_block_n_ticks", "n_ticks")):
        lo, hi = (float(v) for v in env[k])
        if not lo - 1e-9 <= HAND_BLOCK_NOMINAL[lo_hi] <= hi + 1e-9:
            raise RhAgltError(f"{k} {list(env[k])} 가 배포 명목값 {lo_hi} {HAND_BLOCK_NOMINAL[lo_hi]} 를 안 덮는다")
    if bool(env.get("hand_firmware_stop_enable", True)):
        raise RhAgltError("hand_block_enable 인데 hand_firmware_stop_enable 이 켜져 있다 — 실기 펌웨어 힘 멈춤은 증거가 없다(10.12)")
    cap = float(env.get("hand_lead_cap_rad", 0.0))
    block = dict(HAND_BLOCK_NOMINAL, lag_ticks=int(env["hand_block_lag_ticks"]), axes=list(HAND_BLOCK_AXES))
    return block, HB.GRASP_RULES[0], cap


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


class RaDecoder:
    """행동 13 → (팔 q* 7, 손 q* 6). 직전 목표를 든다 — 에피소드 시작에 reset()."""

    def __init__(self, c: RaContract) -> None:
        self.c = c
        self.law = hand_law(c)
        s = c.side()
        self._lo, self._hi, self._home = (np.asarray(v, float) for v in (s.arm_lo, s.arm_hi, s.arm_home))
        self.state: dict[str, _State] = {}
        self.block = (HB.HandBlockNp.from_contract(c.hand_block, s.hand_joints, s.hand_open, s.hand_grip)
                      if c.hand_block else None)
        self.hold = (HB.GraspHoldNp(self.block, c.grasp_release_open_rad, c.grasp_release_steps)
                     if self.block is not None and c.grasp_hold_rule else None)
        self.reset()

    def reset(self, arm_start: Sequence[float] | None = None, hand_start: Sequence[float] | None = None) -> None:
        """기본 = 홈 · 편 손(rh_aglt 학습 리셋). rh_place 는 인계 순간의 팔 · 손 목표에서 시작한다(sim 뱅크 arm_q_target · hand_target)."""
        arm = self._home if arm_start is None else np.clip(np.asarray(arm_start, float), self._lo, self._hi)
        hand = self.c.side().hand_open if hand_start is None else hand_start
        self.state = {"arm": _State(np.array(arm, float), np.array(hand, float))}
        if self.block is not None:
            self.block.reset()
        if self.hold is not None:
            self.hold.reset()

    def step(self, action: Sequence[float], *, active: bool, tactile_n: Sequence[float] | None = None,
             hand_q: Sequence[float] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """hand_q = 실측 손 각도(계약 hand_joints 순, rad) — hand_block 계약이면 반드시(막힘 신호 쥠 멈춤)."""
        a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
        if a.size != 13 or not np.all(np.isfinite(a)):
            raise RhAgltError(f"행동은 유한한 13 개: {a.size}")
        st = self.state["arm"]
        if active:
            q_raw = np.clip(st.arm_target + self.c.k_arm * a[:7], self._lo, self._hi)
            st.arm_target = np.clip(self.c.arm_ema * q_raw + (1.0 - self.c.arm_ema) * st.arm_target, self._lo, self._hi)
        else:
            st.arm_target = self._home.copy()
        if self.block is not None:                    # 10.12 막힘 쥠 멈춤 — 직전 q* 와 지금 실측(hdgp _pre_physics_step 순서)
            if hand_q is None:
                raise RhAgltError("hand_block 계약: 디코더에 실측 손 각도(hand_q)가 필요하다")
            if self.hold is not None:                 # 10.13 쥔 판정 유지 — 이번 block.step 전(sim tick 끝 색인)
                self.hold.step(self.block.grasped(), st.hand_target, hand_q)
            freeze = self.block.step(st.hand_target, hand_q)
        else:
            freeze = self.law.touch_to_freeze(tactile_n)
        st.hand_target = self.law.step(st.hand_target, a[7:], active=active, freeze=freeze)
        return {"arm": (st.arm_target.copy(), st.hand_target.copy())}

    def grasped(self) -> bool | None:
        """막힘 신호 쥔 판정(직전 step 기준) — hand_block 계약이 아니면 None(호출자가 촉각 · 관절 힘 규칙을 쓴다)."""
        if self.block is None:
            return None
        return self.hold.hold if self.hold is not None else self.block.grasped()


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

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

    def side(self, role: str = "arm") -> RaSide:
        return self.sides[role]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def hand_law(c: RaContract) -> RH.HandLaw:
    s = c.side()
    return RH.HandLaw(q_open=tuple(s.hand_open), q_grip=tuple(s.hand_grip), lim_lo=tuple(s.hand_lim_lo),
                      lim_hi=tuple(s.hand_lim_hi), range_mode="grip", ema=c.hand_ema, full_range_s=c.hand_full_range_s,
                      policy_hz=c.policy_hz, freeze=True, freeze_joints=(True,) * 6,
                      freeze_threshold_n=c.freeze_threshold_n, hold="open")


def obs_dim_of(_c: RaContract | None = None) -> int:
    return sum(w for _, w in ACTOR_LAYOUT)


def validate(c: RaContract) -> None:
    if c.action_dim != 13 or c.obs_dim != obs_dim_of():
        raise RhAgltError(f"obs/action {c.obs_dim}/{c.action_dim} ≠ 레이아웃 {obs_dim_of()}/13")
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
SIM_ONLY_RESPONSE_KEYS = ("arm_cmd_delay_steps", "hand_cmd_delay_steps", "hand_open_floor_deg_lo", "hand_open_floor_deg_hi",
                          "thumb1_reset_range")


#: hdgp tasks/rh_aglt_r/rh_aglt_env_cfg.py OBJECTS · CUP_UNIT(hdgp 2721a946 :44-58) — 컵 USD 원점 기준 치수(m) · cup_scale 을 받는지
CUP_OBJECTS: dict[str, tuple[dict[str, float], bool]] = {
    "shaker": ({"bottom_z": -0.0921, "rim_z": 0.0829}, True),
    "cyl60": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
    "cyl65": ({"bottom_z": -0.085, "rim_z": 0.085}, False),
}


def cup_geometry(env: Mapping) -> tuple[float, float]:
    """(cup_half_height, cup_origin_offset_z) — hdgp resolve_cfg(:410-422)와 같은 규칙으로 object_name 에서 다시 푼다.

    env.yaml 의 파생 값은 쓰지 않는다: train.py 는 hydra 가 object_name 을 덮은 뒤 · env 가 resolve_cfg 를 다시 부르기 전에
    덤프하므로 cyl60 런의 덤프에는 기본 shaker × cup_scale 값이 남는다(10.04 확인 — 학습 env 는 cyl60 으로 돌았다)."""
    name = str(env.get("object_name", "shaker"))
    if name not in CUP_OBJECTS:
        raise RhAgltError(f"object_name {name!r} 을 모른다 — {sorted(CUP_OBJECTS)} (hdgp OBJECTS 를 옮겨 와라)")
    unit, scaled = CUP_OBJECTS[name]
    s = float(env["cup_scale"]) if scaled else 1.0
    return 0.5 * (unit["rim_z"] - unit["bottom_z"]) * s, -unit["bottom_z"] * s


def is_rh_aglt_run(env: Mapping) -> bool:
    """rh_place(hdgp open-rh_*_place)도 프로필 · 관측 96 · 행동 13 이 같다 — 홀더 키(target_holders)로 가른다(10.04)."""
    return (str(env.get("profile_name", "")).startswith("rh56f1_") and int(env.get("action_space", 0)) == 13
            and int(env.get("observation_space", 0)) == obs_dim_of() and "target_holders" not in env)


def _left(name: str) -> str:
    return "l_" + name[2:] if name.startswith("r_") else name


def build(run_dir: Path, checkpoint: Path, right_profile, urdf: Path, *, asset: str) -> RaContract:
    """런 덤프(params/env.yaml · agent.yaml) + hdgp RH56F1_RIGHT 프로필 → 계약. 왼팔은 hdgp tasks/rh_aglt_l/profile.py 와
    같은 규칙으로 이름만 바꾼다(손 거울 부호 +1 — 좌 URDF 가 엄지 축을 이미 뒤집었다). 팔 시작 자세는 env.yaml arm_start_q
    (좌 런은 이미 거울상)."""
    run_dir = Path(run_dir)
    env_p, agent_p = run_dir / "params" / "env.yaml", run_dir / "params" / "agent.yaml"
    env = read_env(env_p)
    if not is_rh_aglt_run(env):
        raise RhAgltError(f"{env_p}: rh_aglt 런이 아니다(profile rh56f1_* · obs {obs_dim_of()} · action 13 · 홀더 키 없음 — rh_place 는 따로)")
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
        notes.append(f"컵 {env.get('object_name', 'shaker')}: 반높이 {half_h:.4f} · 원점 높이 {origin_z:.4f} m — env.yaml 덤프"
                     f"({dumped[0]:.4f} · {dumped[1]:.4f})는 hydra 오버라이드 전 기본 shaker 값이라 쓰지 않았다")
    if not bool(cfg_a.get("normalize_input", False)):
        notes.append("normalize_input false")
    c = RaContract(
        schema=SCHEMA, task=TASK_PREFIX.format(p), run_dir=str(run_dir), checkpoint=str(checkpoint),
        checkpoint_md5=hashlib.md5(Path(checkpoint).read_bytes()).hexdigest(),
        env_yaml_sha1=hashlib.sha1(env_p.read_bytes()).hexdigest(),
        agent_yaml_sha1=hashlib.sha1(agent_p.read_bytes()).hexdigest(), asset=asset, profile=str(env["profile_name"]),
        policy_hz=1.0 / (float(env["sim"]["dt"]) * int(env["decimation"])), episode_s=float(env["episode_length_s"]),
        obs_dim=int(env["observation_space"]), action_dim=int(env["action_space"]),
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
        goal_tol=float(env["tol_floor"]), grasp_threshold_n=float(env["contact_force_threshold"]))
    validate(c)
    return c


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
        self.reset()

    def reset(self) -> None:
        self.state = {"arm": _State(self._home.copy(), np.asarray(self.c.side().hand_open, float))}

    def step(self, action: Sequence[float], *, active: bool,
             tactile_n: Sequence[float] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
        if a.size != 13 or not np.all(np.isfinite(a)):
            raise RhAgltError(f"행동은 유한한 13 개: {a.size}")
        st = self.state["arm"]
        if active:
            q_raw = np.clip(st.arm_target + self.c.k_arm * a[:7], self._lo, self._hi)
            st.arm_target = np.clip(self.c.arm_ema * q_raw + (1.0 - self.c.arm_ema) * st.arm_target, self._lo, self._hi)
        else:
            st.arm_target = self._home.copy()
        st.hand_target = self.law.step(st.hand_target, a[7:], active=active, freeze=self.law.touch_to_freeze(tactile_n))
        return {"arm": (st.arm_target.copy(), st.hand_target.copy())}


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


def build_obs(c: RaContract, m: RaMeas, dec: RaDecoder, goal: Goal, prev_action: Sequence[float]) -> np.ndarray:
    """관측 96 — 학습 `_actor_parts` 순서 그대로. 목표는 디코더의 지금 q*(직전 스텝 결과)."""
    s, st = c.side(), dec.state["arm"]
    hand_q = np.array([m.hand_q[j] for j in s.hand_joints], float)
    tips = np.asarray(m.tips, float).reshape(5, 3)
    parts = [m.arm_q, m.arm_qd, st.arm_target, hand_q,
             np.clip((st.hand_target - hand_q) / c.joint_err_norm, -1.0, 1.0),
             m.palm_pos, np.concatenate([m.palm_R[:, 0], m.palm_R[:, 1]]),
             m.cup_pos - m.palm_pos, quat_apply(m.cup_quat, np.array([0.0, 0.0, 1.0])), (tips - m.cup_pos).reshape(-1),
             (keypoints(c, m.cup_pos, m.cup_quat) - keypoints(c, goal.pos, goal.quat)).reshape(-1),
             goal.pos - m.palm_pos, RH.tactile_obs(m.tactile_n, c.tactile_clip_n, c.tactile_tanh_n),
             np.asarray(prev_action, float)]
    for p, (name, w) in zip(parts, ACTOR_LAYOUT):
        if np.asarray(p).size != w:
            raise RhAgltError(f"관측 '{name}' 폭 {np.asarray(p).size} ≠ {w}")
    obs = np.concatenate([np.asarray(p, float).reshape(-1) for p in parts])
    return np.clip(np.nan_to_num(obs), -c.obs_clip, c.obs_clip)


def with_run(c: RaContract, **kw) -> RaContract:
    out = replace(c, **kw)
    validate(out)
    return out

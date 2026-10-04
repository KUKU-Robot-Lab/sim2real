"""pour_fj — RH56F1 양팔 붓기(hdgp `open-rh_b_pour_fj`) 배포 계약 · 행동 디코더 · 관측 조립. 순수(ROS 없음).

09.29 사용자: "rh56f1 정책: pour_fj 양손 다음에 rh_aglt — 곧 정책 나올 예정, 미리 준비".
hdgp 기준 코드(09.29): tasks/pour_fabric_mimic/{pour_fabric_env.py _pre_physics_step · _side_obs · _get_observations,
side_rig.py step_joint_arm · direct_hand_targets · joint_err, pour_fabric_env_cfg.py PourFJEnvCfg}.

행동 26 = [src 오른팔 7 · 오른손 6][rcv 왼팔 7 · 왼손 6], clamp ±1.
  팔  increment: q_raw = clip(q* + k·a) → q* = clip(α·q_raw + (1−α)·q*)                          (f00~f03)
      absolute : q_raw = home + a·(hi−home) (a≥0) | home + a·(home−lo) → q* += clip(α·(q_raw−q*), ±vmax·dt)   (f04~)
      대기(hold_steps) 동안 q* = home
  손  policy_control/rh56f1_hand.py HandLaw(모든 RH56F1 정책 공통) — range grip | limits · EMA · 전 범위 T 초 · 접촉 동결.
      pour_fj 는 대기 중에도 손이 행동을 따른다(hold "follow").
관측(팔마다 63): arm_q 7 · arm_qd 7 · hand_q 6(★PhysX 순) · palm_pos 3 · palm R 열0+열1 6 · [tips−palm 15] · cup−palm 3 ·
  tips−cup 15 · joint_err 6((q*−q)/joint_pos_err_max, ±1, 프로필 순) · cup_up 3 · arm q* 7
  그 뒤 rcv 컵 − src 컵 3 · [입구 차 3] · 촉각 src 5 · rcv 5 (tanh(clip(F,0,10)/3)) · 직전 행동 26  → 165 (f01 · f04)

★손 관측 순서(hand_q)는 PhysX 가 정한다(find_joints(regex)). 기록된 실측이 없어 `hand_obs_order_source` 가 assumed 인 계약은
  배포 등록부가 verified 로 올리지 못하게 한다(joint family 와 같은 규칙). Isaac trace 의 joint_names 로 다시 만든다.
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

SCHEMA = "policy_control/pour_fj_contract/v1"
ROLES = ("src", "rcv")
SIDE_OF = {"src": "right", "rcv": "left"}           # hdgp 쌍 규약: source = 우(붓기), receiver = 좌(받기)
TASK_PREFIX = "open-rh_b_pour_fj"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
#: PhysX 관절 순서 추정(실측 전) — URDF 트리 너비 우선 · 같은 깊이 알파벳(DG-5F 선례 modules/robots.py:119)
ASSUMED_OBS_ORDER = ("index_1", "middle_1", "pinky_1", "ring_1", "thumb_1", "thumb_2")


class PourFjError(ValueError):
    pass


#: 왼팔 거울 부호 — hdgp pour_fabric_mimic/side_rig._ARM_SIGN_L(= robot_profiles._ARM_SIGN_L)과 같다
ARM_SIGN_L = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)


def arm_abs_home_for(side: str, raw) -> list:
    """env arm_abs_home_rad(오른팔 기준 7) → 그 팔의 a=0 자세. 왼팔은 거울 부호(hdgp side_rig.init_joint_arm). 없으면 []. 순수."""
    home = tuple(raw or ())
    if not home:
        return []
    if len(home) != 7:
        raise PourFjError(f"arm_abs_home_rad 길이 {len(home)} ≠ 팔 관절 7")
    sign = ARM_SIGN_L if side == "left" else (1.0,) * 7
    return [s_ * float(v) for s_, v in zip(sign, home)]


# ---------------------------------------------------------------- 계약
@dataclass(frozen=True)
class FjSide:
    role: str
    side: str
    arm_joints: list
    arm_home: list
    arm_lo: list
    arm_hi: list
    hand_joints: list           # 프로필 순 = 행동 슬롯 순(slot 이 항등) = joint_err 순
    hand_obs_order: list        # 관측 hand_q 순(PhysX)
    hand_open: list
    hand_grip: list
    hand_lim_lo: list
    hand_lim_hi: list
    hand_finger: list           # 관절 → 손가락 인덱스(FINGERS)
    hand_freeze: list           # 접촉 동결 대상 관절(프로필 hand_freeze_suffixes)
    palm_body: str
    tip_bodies: list
    # ★10.04 absolute 모드 a=0 자세(hdgp side_rig.abs_home) — env arm_abs_home_rad(왼팔은 거울 부호). 비면 arm_home(시작 자세).
    arm_abs_home: list = field(default_factory=list)


@dataclass(frozen=True)
class FjContract:
    schema: str
    task: str
    run_dir: str
    checkpoint: str
    checkpoint_md5: str
    env_yaml_sha1: str
    agent_yaml_sha1: str
    asset: str
    policy_hz: float
    episode_s: float
    obs_dim: int
    action_dim: int
    obs_clip: float
    action_clip: float
    recurrent: bool
    hold_steps: int
    arm_mode: str               # increment | absolute
    k_arm: float
    arm_ema: float
    arm_abs_vmax: float
    hand_ema: float
    hand_full_range_s: float
    hand_range: str             # grip | limits
    hand_freeze: bool
    freeze_threshold_n: float
    joint_pos_err_max: float
    tactile_clip_n: float
    tactile_tanh_n: float
    obs_tips_rel_palm: bool
    obs_mouth_diff: bool
    cup_mouth_z: float
    hand_obs_order_source: str
    sides: dict                 # role -> FjSide
    notes: list = field(default_factory=list)
    # ★09.30 hdgp 실기 맞춤(side_rig direct_hand_targets) — 0 = 그 기능 이전 런
    hand_finger_open_floor_rad: float = 0.0
    hand_vel_cap_rad_s: float = 0.0
    hand_thumb_flex_vel_cap_rad_s: float = 0.0
    # ★10.04 hdgp t2r_rh5_b16(53050c2e) 손 닫힘 상한 — 에피소드 시작(인계)의 손 목표 q*_0 에서 닫는 쪽으로 이만큼까지만.
    #   0 = 그 기능 이전 런. >0 이면 디코더를 인계 손 목표(직전 정책의 마지막 joint_target)로 시작해야 한다.
    hand_close_margin_rad: float = 0.0
    # ★10.04 hdgp t2r_rh5_b17(92237130) absolute 팔 목표 가속 한계 [rad/s²] — 0 = 끔(그 이전 런)
    arm_abs_amax: float = 0.0
    # ★10.04 인계 뱅크에서만 시작하는 런(env start_bank_frac ≥ 1, b-계열) — sim 은 hold 를 건너뛰고(start_bank.restore 가
    #   episode_length_buf = hold_steps) 팔 · 손 q* 를 수집 순간 실측으로 둔다. 배포도 인계 순간 실측에서, hold 없이 시작한다.
    bank_start: bool = False

    def side(self, role: str) -> FjSide:
        return self.sides[role]

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)


def load_contract(path: str | Path) -> FjContract:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise PourFjError(f"{path}: schema {raw.get('schema')!r} ≠ {SCHEMA}")
    raw["sides"] = {r: FjSide(**s) for r, s in raw["sides"].items()}
    c = FjContract(**raw)
    validate(c)
    return c


def obs_dim_of(c: FjContract) -> int:
    per = 7 + 7 + 6 + 3 + 6 + (15 if c.obs_tips_rel_palm else 0) + 3 + 15 + 6 + 3 + 7
    return 2 * per + 3 + (3 if c.obs_mouth_diff else 0) + 5 + 5 + c.action_dim


def validate(c: FjContract) -> None:
    if c.arm_mode not in ("increment", "absolute") or c.hand_range not in ("grip", "limits"):
        raise PourFjError(f"arm_mode {c.arm_mode!r} · hand_range {c.hand_range!r}")
    if c.action_dim != 26 or obs_dim_of(c) != c.obs_dim:
        raise PourFjError(f"obs/action {c.obs_dim}/{c.action_dim} ≠ 레이아웃 {obs_dim_of(c)}/26")
    for r in ROLES:
        s = c.sides[r]
        if len(s.arm_joints) != 7 or len(s.hand_joints) != 6 or sorted(s.hand_obs_order) != sorted(s.hand_joints):
            raise PourFjError(f"{r}: 팔 7 · 손 6 · 관측 손 순서는 같은 6 관절")
        if not all(lo <= h <= hi for lo, h, hi in zip(s.arm_lo, s.arm_home, s.arm_hi)):
            raise PourFjError(f"{r}: arm_home 이 관절 한계 밖")
        if s.arm_abs_home and (len(s.arm_abs_home) != 7
                               or not all(lo <= h <= hi for lo, h, hi in zip(s.arm_lo, s.arm_abs_home, s.arm_hi))):
            raise PourFjError(f"{r}: arm_abs_home_rad 가 관절 한계 밖이거나 7 개가 아니다 {s.arm_abs_home}")


# ---------------------------------------------------------------- 빌드 (런 덤프 + hdgp 프로필)
class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, n: ld.construct_sequence(n))
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda ld, suffix, n: None)
_Loader.add_multi_constructor("!!python/", lambda ld, suffix, n: None)


def read_env(path: Path) -> dict:
    return yaml.load(Path(path).read_text(), Loader=_Loader) or {}


def _sha1(p: Path) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def _md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


#: hdgp tasks/pour_fabric_mimic/pour_fabric_env_cfg.py CUP_OBJECTS 의 geom.cup_mouth_z(실물 크기, cup_scale 안 받음, hdgp a0c0f2a3).
#: shaker 는 덤프 값 그대로 — resolve_cfg 가 같은 값을 다시 쓴다.
CUP_MOUTH_Z = {"cyl60": 0.085, "cyl60_box32": 0.085, "cyl60_box32w25": 0.085, "cyl60_sdf256": 0.085}


def cup_mouth_z_of(env: Mapping) -> float:
    """컵 원점 → 입구 높이. env.yaml 의 cup_mouth_z 는 hydra 가 cup_object 를 덮기 전 기본 shaker 값으로 덤프된다
    (train.py 가 resolve_cfg 재호출 전에 덤프 — 10.04 b16~b18 확인) — cup_object 에서 다시 푼다."""
    name = str(env.get("cup_object", "shaker"))
    if name == "shaker":
        return float(env["cup_mouth_z"])
    if name not in CUP_MOUTH_Z:
        raise PourFjError(f"cup_object {name!r} 의 입구 높이를 모른다 — hdgp CUP_OBJECTS 에서 CUP_MOUTH_Z 로 옮겨 와라")
    return CUP_MOUTH_Z[name]


def is_pour_fj_run(env: Mapping) -> bool:
    return (env.get("pair_name") == "rh" and env.get("hand_control") == "direct"
            and int(env.get("action_space", 0)) == 26)


#: hdgp 에 생겼지만 배포 디코더로 아직 옮기지 않은 env 키와 그 '끔' 값(hdgp pour_fabric_env_cfg 기본값).
#: 켜진 런은 계약을 만들지 않는다 — 조용히 무시하면 실기 목표가 학습과 어긋난다(10.01 손 q* 어긋남과 같은 부류).
#: 옮길 때는 여기서 빼고 디코더와 trace 한 스텝 재생 대조를 더한다.
UNPORTED_OFF: Mapping[str, object] = {
    "hand_force_stop_nm": 0.0,      # sim 힘 멈춤(b16 은 0)
    "oppose_grip_delta_rad": 0.0,   # 대향 grip 보정
    "arm_abs_range_rad": (),        # hdgp 23c2ab6f absolute ±1 범위 · 733b3365 증분 목표 상자(arm_abs_home_rad ± 범위)
    "arm_action_lpf": 1.0,          # hdgp 733b3365 팔 행동 저역 필터 a_f = β·a + (1−β)·a_f
    "hand_hold": False,             # hdgp 733b3365 두 손 목표를 인계 쥔 목표로 고정, 손 행동 무시
    "rcv_arm_hold": False,          # hdgp 733b3365 리시버 팔 목표를 인계 목표로 고정, 그 행동 무시
}


def _is_on(value, off) -> bool:
    if isinstance(off, tuple):
        return bool(tuple(value or ()))
    if isinstance(off, bool):
        return bool(value) != off
    return float(value) != float(off)


def unported_keys(env: Mapping) -> list[str]:
    """env 에서 켜진 미이식 기능 'key=값'. 키가 없으면 그 기능 이전 런(끔). 순수."""
    return [f"{k}={env[k]}" for k, off in UNPORTED_OFF.items() if env.get(k) is not None and _is_on(env[k], off)]


def _urdf_limits(urdf: Path, joints: Sequence[str]) -> dict:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(Path(urdf).read_text())
    out = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if j.get("name") in joints and lim is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    missing = [j for j in joints if j not in out]
    if missing:
        raise PourFjError(f"URDF 에 한계가 없다: {missing}")
    return out


def build(run_dir: Path, checkpoint: Path, pair, urdf: Path, *, asset: str,
          hand_obs_order: Mapping[str, Sequence[str]] | None = None, obs_order_source: str = "") -> FjContract:
    """런 덤프(params/env.yaml · agent.yaml) + hdgp RH 쌍 프로필(pair: source/receiver RobotProfile) → 계약.

    env.yaml 에 없는 키는 **그 기능이 생기기 전 런**이다 — 그때의 동작을 쓴다(없으면 increment · limits · 동결 없음).
    """
    run_dir = Path(run_dir)
    env_p, agent_p = run_dir / "params" / "env.yaml", run_dir / "params" / "agent.yaml"
    env = read_env(env_p)
    if not is_pour_fj_run(env):
        raise PourFjError(f"{env_p}: pour_fj 런이 아니다(pair_name rh · hand_control direct · action 26)")
    unported = unported_keys(env)
    if unported:
        raise PourFjError(f"{env_p}: 배포 디코더에 아직 없는 기능이 켜져 있다 {unported} — 옮기고 trace 대조한 뒤 계약을 만든다")
    agent = yaml.safe_load(agent_p.read_text())
    net, cfg_a = agent["params"]["network"], agent["params"]["config"]
    notes = []
    absent = [k for k in ("arm_joint_mode", "hand_direct_range", "hand_direct_contact_freeze") if k not in env]
    if absent:
        notes.append(f"env.yaml 에 {absent} 없음 — 그 기능 이전 런(increment · limits · 동결 없음)")
    sim_dt = float(env["sim"]["dt"])
    decim = int(env["decimation"])
    sides = {}
    for role, prof in (("src", pair.source), ("rcv", pair.receiver)):
        side = SIDE_OF[role]
        p = side[0]
        arm = [f"{p}_aj_{i}" for i in range(1, 8)]
        hand = list(prof.hand_joint_names)
        if [n.split("_hj_")[1] for n in hand] != ["thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1"]:
            raise PourFjError(f"{prof.name}: 손 관절 {hand} — RH56F1 프로필 순서가 아니다")
        slots = [int(prof.hand_finger_channels[_finger(n)][n.rsplit("_", 1)[1]]) for n in hand]
        if slots != list(range(6)):
            raise PourFjError(f"{prof.name}: 손가락 → 행동 슬롯이 항등이 아니다 {slots} — 디코더가 가정한다")
        lim = _urdf_limits(urdf, arm + hand)
        init = prof.init_joint_pos if hasattr(prof, "init_joint_pos") else prof.init_state_joint_pos
        order = list((hand_obs_order or {}).get(role) or [f"{p}_hj_{n}" for n in ASSUMED_OBS_ORDER])
        abs_home = arm_abs_home_for(side, env.get("arm_abs_home_rad") if str(env.get("arm_joint_mode", "increment")) == "absolute" else None)
        sides[role] = FjSide(
            role=role, side=side, arm_joints=arm, arm_home=[float(init[j]) for j in arm], arm_abs_home=abs_home,
            arm_lo=[lim[j][0] for j in arm], arm_hi=[lim[j][1] for j in arm],
            hand_joints=hand, hand_obs_order=order,
            hand_open=[float(v) for v in prof.hand_open_pose], hand_grip=[float(v) for v in prof.hand_grip_pose],
            hand_lim_lo=[lim[j][0] for j in hand], hand_lim_hi=[lim[j][1] for j in hand],
            hand_finger=[FINGERS.index(_finger(n)) for n in hand],
            hand_freeze=[n.rsplit("_", 1)[1] in tuple(prof.hand_freeze_suffixes) for n in hand],
            palm_body=str(prof.palm_body), tip_bodies=list(prof.fingertip_bodies))
    c = FjContract(
        schema=SCHEMA, task=TASK_PREFIX, run_dir=str(run_dir), checkpoint=str(checkpoint),
        checkpoint_md5=_md5(checkpoint), env_yaml_sha1=_sha1(env_p), agent_yaml_sha1=_sha1(agent_p), asset=asset,
        policy_hz=1.0 / (sim_dt * decim), episode_s=float(env["episode_length_s"]),
        obs_dim=int(env["observation_space"]), action_dim=int(env["action_space"]),
        obs_clip=float(agent["params"]["env"].get("clip_observations", 5.0)),
        action_clip=float(agent["params"]["env"].get("clip_actions", 1.0)),
        recurrent=bool(net.get("rnn")), hold_steps=int(env.get("hold_steps", 0)),
        arm_mode=str(env.get("arm_joint_mode", "increment")), k_arm=float(env["k_arm"]), arm_ema=float(env["arm_ema"]),
        arm_abs_vmax=float(env.get("arm_abs_vmax", 0.3)), hand_ema=float(env["hand_ema"]),
        hand_full_range_s=float(env["hand_full_range_s"]), hand_range=str(env.get("hand_direct_range", "limits")),
        hand_freeze=bool(env.get("hand_direct_contact_freeze", False)),
        freeze_threshold_n=float(env["contact_freeze_threshold"]), joint_pos_err_max=float(env["joint_pos_err_max"]),
        tactile_clip_n=float(env["tactile_obs_clip_n"]), tactile_tanh_n=float(env["tactile_obs_tanh_n"]),
        obs_tips_rel_palm=bool(env.get("obs_tips_rel_palm", True)), obs_mouth_diff=bool(env.get("obs_mouth_diff", True)),
        cup_mouth_z=cup_mouth_z_of(env),
        hand_obs_order_source=obs_order_source or ("assumed:" + ",".join(ASSUMED_OBS_ORDER)),
        sides=sides, notes=notes,
        hand_finger_open_floor_rad=float(env.get("hand_finger_open_floor_rad", 0.0)),
        hand_vel_cap_rad_s=float(env.get("hand_vel_cap_rad_s", 0.0)),
        hand_thumb_flex_vel_cap_rad_s=float(env.get("hand_thumb_flex_vel_cap_rad_s", 0.0)),
        hand_close_margin_rad=float(env.get("hand_close_margin_rad", 0.0)),
        arm_abs_amax=float(env.get("arm_abs_amax", 0.0)),
        bank_start=float(env.get("start_bank_frac", 0.0) or 0.0) >= 1.0 and bool(env.get("start_bank_path")))
    if not bool(cfg_a.get("normalize_input", False)):
        notes.append("normalize_input false")
    validate(c)
    return c


def _finger(joint: str) -> str:
    hit = [f for f in FINGERS if f"_{f}_" in joint]
    if len(hit) != 1:
        raise PourFjError(f"관절 {joint} 의 손가락을 모른다")
    return hit[0]


# ---------------------------------------------------------------- 디코더
def hand_vel_cap(c: FjContract) -> tuple:
    """hdgp side_rig init_real_control: 공통 상한, thumb_2(엄지 굽힘)만 따로(0 이면 공통). 공통이 0 이면 ()(전 범위/1 s)."""
    if c.hand_vel_cap_rad_s <= 0.0:
        return ()
    th = c.hand_thumb_flex_vel_cap_rad_s
    return tuple(th if (th > 0.0 and j == "thumb_2") else c.hand_vel_cap_rad_s for j in RH.JOINTS)


@dataclass
class _SideState:
    arm_target: np.ndarray
    hand_target: np.ndarray
    arm_prev_q: np.ndarray | None = None    # 직전 스텝 갱신 앞의 q*(가속 한계의 prev_step 기준, sim _arm_prev_q)


class FjDecoder:
    """행동 26 → 팔마다 (팔 q* 7, 손 q* 6). 상태(직전 목표)를 든다 — 에피소드 시작에 reset()."""

    def __init__(self, c: FjContract) -> None:
        self.c = c
        self.dt = 1.0 / c.policy_hz
        self.state: dict[str, _SideState] = {}
        self.reset()

    def reset(self, hand_start: Mapping[str, Sequence[float]] | None = None,
              arm_start: Mapping[str, Sequence[float]] | None = None) -> None:
        """hand_start[role] = 인계 순간의 손 q*_0(hand_joints 순 — 배포는 손 실측 관절각, sim 뱅크와 같다). 없으면 편 손에서 시작.

        ★hand_close_margin_rad > 0(b16~) 계약은 hand_start 가 있어야 한다 — 편 손을 q*_0 로 두면 손을 못 쥔다. 없으면 step 이 거부.
        """
        start, arm0 = hand_start or {}, arm_start or {}
        self.state = {}
        for r in ROLES:
            q_arm = np.array(arm0[r] if r in arm0 else self.c.sides[r].arm_home, float)
            # 리셋 = 목표 속도 0(sim reset_arm_rate: _arm_prev_q = q*)
            self.state[r] = _SideState(q_arm, np.array(start[r] if r in start else self.c.sides[r].hand_open, float),
                                       q_arm.copy())
        self.hand_q0 = {r: np.array(start[r], float) for r in ROLES if r in start}
        self.close_dir = {r: np.sign(np.asarray(self.c.sides[r].hand_grip, float) - np.asarray(self.c.sides[r].hand_open, float))
                          for r in ROLES}

    def law(self, s: FjSide) -> RH.HandLaw:
        """손 행동 법칙(policy_control/rh56f1_hand.py) — pour_fj 는 대기 중에도 손이 행동을 따른다(hdgp side_rig 게이트 없음)."""
        c = self.c
        return RH.HandLaw(q_open=tuple(s.hand_open), q_grip=tuple(s.hand_grip), lim_lo=tuple(s.hand_lim_lo),
                          lim_hi=tuple(s.hand_lim_hi), range_mode=c.hand_range, ema=c.hand_ema,
                          full_range_s=c.hand_full_range_s, policy_hz=60.0, freeze=c.hand_freeze,
                          freeze_joints=tuple(bool(x) for x in s.hand_freeze), freeze_threshold_n=c.freeze_threshold_n,
                          hold="follow", finger_open_floor=c.hand_finger_open_floor_rad, vel_cap_rad_s=hand_vel_cap(c))

    def hand_range(self, s: FjSide) -> tuple[np.ndarray, np.ndarray]:
        return self.law(s).bounds()

    def step(self, action: Sequence[float], *, active: bool,
             touch: Mapping[str, Sequence[bool]] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """active=False = 대기(hold). touch[role] = 손가락 5개가 컵에 닿았나(동결용, 없으면 동결 안 함)."""
        a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
        if a.size != 26 or not np.all(np.isfinite(a)):
            raise PourFjError(f"행동은 유한한 26 개: {a.size}")
        out = {}
        for i, r in enumerate(ROLES):
            s, st = self.c.sides[r], self.state[r]
            a_arm, a_hand = a[i * 13:i * 13 + 7], a[i * 13 + 7:(i + 1) * 13]
            q_new = self._arm(s, st, a_arm)
            st.arm_prev_q = st.arm_target.copy()                         # sim: 갱신 앞 q* 를 기록(모드 · hold 무관)
            st.arm_target = q_new if active else np.array(s.arm_home, float)
            st.hand_target = self._hand(s, st.hand_target, a_hand, None if touch is None else touch.get(r), r)
            out[r] = (st.arm_target.copy(), st.hand_target.copy())
        return out

    def _arm(self, s: FjSide, st: "_SideState", a: np.ndarray) -> np.ndarray:
        c, lo, hi, q = self.c, np.array(s.arm_lo), np.array(s.arm_hi), st.arm_target
        if c.arm_mode == "absolute":
            home = np.array(s.arm_abs_home if s.arm_abs_home else s.arm_home, float)   # a=0 자세(sim abs_home)
            q_raw = np.where(a >= 0.0, home + a * (hi - home), home + a * (home - lo))
            cap = c.arm_abs_vmax * self.dt
            err = q_raw - q
            step = np.clip(c.arm_ema * err, -cap, cap)
            if c.arm_abs_amax > 0.0:
                # ★10.04 b17 가속 한계(sim _accel_limit): 멈출 수 있는 속도 √(2·amax·|err|)·dt, |step − prev_step| ≤ amax·dt²
                prev_step = np.zeros_like(step) if st.arm_prev_q is None else q - st.arm_prev_q
                v_stop = np.sqrt(2.0 * c.arm_abs_amax * np.abs(err)) * self.dt
                step = np.clip(step, -v_stop, v_stop)
                dv = c.arm_abs_amax * self.dt * self.dt
                step = np.clip(step, prev_step - dv, prev_step + dv)
            return np.clip(q + step, lo, hi)
        q_raw = np.clip(q + c.k_arm * a, lo, hi)
        return np.clip(c.arm_ema * q_raw + (1.0 - c.arm_ema) * q, lo, hi)

    def _hand(self, s: FjSide, prev: np.ndarray, a: np.ndarray, touch, role: str | None = None) -> np.ndarray:
        """touch = 손가락 5개 닿음(bool) — 계약의 동결 임계로 이미 판정한 값."""
        law = self.law(s)
        freeze = None
        if law.freeze and touch is not None:
            freeze = np.asarray(touch, bool)[list(RH.FINGER_OF)] & np.asarray(law.freeze_joints, bool)
        q = law.step(prev, a, active=True, freeze=freeze)          # 학습의 hard-coded 60 — policy_hz 와 같다
        m = self.c.hand_close_margin_rad
        if m > 0.0:
            # ★10.04 b16: EMA · 속도 상한 · grip 범위 clamp 다음에(side_rig.direct_hand_targets 순서) 닫는 쪽만 q*_0 + margin 까지.
            #   close_dir = sign(grip − open), 엄지 회전(thumb_1)도 같은 규칙. 펴는 쪽은 자르지 않는다.
            q0 = self.hand_q0.get(role)
            if q0 is None:
                raise PourFjError("hand_close_margin_rad 계약은 인계 손 목표(reset(hand_start=…)) 없이 돌 수 없다")
            d = self.close_dir[role]
            lim = q0 + d * m
            q = np.where(d > 0, np.minimum(q, lim), np.where(d < 0, np.maximum(q, lim), q))
        return q


# ---------------------------------------------------------------- 관측
@dataclass(frozen=True)
class FjSideMeas:
    arm_q: np.ndarray           # 7 (arm_joints 순)
    arm_qd: np.ndarray          # 7
    hand_q: Mapping[str, float]  # 이름 → rad (구동 6)
    palm_pos: np.ndarray        # 3 (base)
    palm_R: np.ndarray          # 3×3
    tips: np.ndarray            # 5×3 (tip_bodies 순)
    cup_pos: np.ndarray         # 3 (그 팔의 컵 원점, base)
    cup_quat: np.ndarray        # 4 wxyz
    tactile_n: np.ndarray       # 5 (엄지 → 새끼, N)
    joint_force: np.ndarray | None = None   # 6 손 관절 힘(g) — 10.04, 없으면 None


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _side_obs(c: FjContract, s: FjSide, m: FjSideMeas, hand_target: np.ndarray, arm_target: np.ndarray) -> list:
    hand_obs = np.array([m.hand_q[j] for j in s.hand_obs_order])
    hand_prof = np.array([m.hand_q[j] for j in s.hand_joints])
    parts = [m.arm_q, m.arm_qd, hand_obs, m.palm_pos, np.concatenate([m.palm_R[:, 0], m.palm_R[:, 1]])]
    if c.obs_tips_rel_palm:
        parts.append((m.tips - m.palm_pos).reshape(-1))
    parts += [m.cup_pos - m.palm_pos, (m.tips - m.cup_pos).reshape(-1),
              np.clip((hand_target - hand_prof) / c.joint_pos_err_max, -1.0, 1.0),
              _quat_apply(np.asarray(m.cup_quat, float), np.array([0.0, 0.0, 1.0])), arm_target]
    return parts


def tactile_obs(c: FjContract, f: Sequence[float]) -> np.ndarray:
    return RH.tactile_obs(f, c.tactile_clip_n, c.tactile_tanh_n)


def build_obs(c: FjContract, meas: Mapping[str, FjSideMeas], dec: FjDecoder, prev_action: Sequence[float]) -> np.ndarray:
    """관측 165 — 학습 `_get_observations` 의 actor 순서 그대로. 목표는 디코더의 지금 q*(직전 스텝 결과)."""
    parts = []
    for r in ROLES:
        st = dec.state[r]
        parts += _side_obs(c, c.sides[r], meas[r], st.hand_target, st.arm_target)
    src, rcv = meas["src"], meas["rcv"]
    parts.append(rcv.cup_pos - src.cup_pos)
    if c.obs_mouth_diff:
        z = np.array([0.0, 0.0, c.cup_mouth_z])
        parts.append((rcv.cup_pos + _quat_apply(rcv.cup_quat, z)) - (src.cup_pos + _quat_apply(src.cup_quat, z)))
    parts += [tactile_obs(c, src.tactile_n), tactile_obs(c, rcv.tactile_n), np.asarray(prev_action, float)]
    obs = np.concatenate([np.asarray(p, float).reshape(-1) for p in parts])
    if obs.size != c.obs_dim:
        raise PourFjError(f"관측 {obs.size} ≠ 계약 {c.obs_dim}")
    return np.clip(np.nan_to_num(obs), -c.obs_clip, c.obs_clip)


def with_obs_order(c: FjContract, order: Mapping[str, Sequence[str]], source: str) -> FjContract:
    """Isaac trace 로 잰 PhysX 손 순서를 넣은 새 계약."""
    sides = {r: replace(s, hand_obs_order=list(order[r])) for r, s in c.sides.items()}
    out = replace(c, sides=sides, hand_obs_order_source=source)
    validate(out)
    return out

"""rh_place 계열 — RH56F1 한 팔 컵 홀더 놓기(hdgp open-rh_{r,l}_place, 51013e96). 10.04 사용자: 성공한 정책부터 실기에.

관측 96 · 행동 13 · 디코더는 rh_aglt 와 같다(rh_aglt.build_obs · RaDecoder 를 그대로 쓴다). 다른 것:
  · 목표 = 홀더 자리: 홀더 원점(컵 축 위 받침 윗면, /objects/cup_holder_<k>/pose) + (0, 0, HOLDER_FLOOR_Z − cup_bottom_z), 세운 컵.
  · 시작 = aglt 가 컵을 쥐고 멈춘 상태의 인계. 팔 · 손 목표는 그 순간의 목표(sim 뱅크 arm_q_target · hand_target = aglt 마지막 q*,
    배포 = pd 가 붙잡고 있는 마지막 joint_target), hold 0, LSTM 0.
  · 컵 관측 = 인계 순간 손바닥 기준 상대 자세로 붙인 것(sim cup_obs_mode "attached") — 에피소드 내내 손바닥 FK 로만.
  · 놓음 = 손가락 · 손바닥 접촉 < 1 N 이 release_steps 연속 → settle_steps 동안 스크립트(손 = 편 손 행동, 팔 = 시작 관절 쪽 최대 증분)
    → 끝. 실기 판정은 손끝 촉각 전부 < 1 N 이고 관절 힘(있으면) 전부 < 문턱(쥠 판정과 같은 300 g, 임시)이며, 네 손가락 손
    목표가 인계 때보다 open 쪽으로 OPEN_MIN_RAD 넘게 열린 뒤(실기는 손바닥 · 첫마디 접촉을 못 본다 — 10.04 PLACE 검토).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from policy_control import rh_aglt as A
from policy_control.cup_attach import AttachCfg, grasp_signal

SCHEMA = "policy_control/rh_place_contract/v1"
TASK_PREFIX = "open-rh_{}_place"
#: hdgp tasks/rh_place_r/place_geom.py:12 — 홀더 원점 기준 컵이 앉는 바닥 높이
HOLDER_FLOOR_Z = -0.025
JOINT_FREE_G = AttachCfg.joint_force_g
#: 놓음 안전 게이트(10.04 PLACE 검토) — 네 손가락 손 목표가 인계 때보다 open 쪽으로 평균 이만큼 열린 뒤에만 '빈 손'을 센다.
#  sim 놓음은 정책이 손을 연 뒤에만 났다. 실기는 손바닥 · 첫마디 접촉을 못 봐서 손끝이 비어도 마디로 쥐고 있을 수 있다.
OPEN_MIN_RAD = 0.15


class RhPlaceError(A.RhAgltError):
    pass


@dataclass(frozen=True)
class PlaceContract(A.RaContract):
    seat_dz: float = 0.0                 # 홀더 원점 → 자리(컵 원점) 높이
    target_holders: list = field(default_factory=list)
    release_steps: int = 0
    settle_steps: int = 0
    release_force_n: float = 0.0         # 놓음 = 접촉이 모두 이 힘 아래(env contact_force_threshold)


def is_rh_place_run(env: Mapping) -> bool:
    return ("target_holders" in env and str(env.get("cup_obs_mode", "")) == "attached"
            and int(env.get("hold_steps", -1)) == 0 and str(env.get("profile_name", "")).startswith("rh56f1_"))


def build(run_dir: Path, checkpoint: Path, right_profile, urdf: Path, *, asset: str) -> PlaceContract:
    env_p = Path(run_dir) / "params" / "env.yaml"
    env = A.read_env(env_p)
    if not is_rh_place_run(env):
        raise RhPlaceError(f"{env_p}: rh_place 런이 아니다(target_holders · cup_obs_mode attached · hold 0)")
    base = A.build_from_env(Path(run_dir), checkpoint, right_profile, urdf, asset=asset, env=env)
    _, origin_z = A.cup_geometry(env)                       # 컵 원점 → 바닥 = −origin_z
    kw = {k: v for k, v in asdict(base).items() if not k.startswith("goal_")}
    kw["sides"] = base.sides
    c = PlaceContract(**{**kw, "schema": SCHEMA, "task": TASK_PREFIX.format(base.side().side[0]),
                         "goal_offset": [0.0, 0.0, 0.0],
                         "notes": [n for n in base.notes if not n.startswith(("첫 목표", "목표 입력", "목표 달성"))] + [
                             f"목표 = 홀더 원점 + (0, 0, {HOLDER_FLOOR_Z + origin_z:.3f}) · 세운 컵 — 학습 목표 홀더 "
                             f"{list(env['target_holders'])} (/objects/cup_holder_<k>/pose)",
                             "시작 = aglt 가 쥐고 멈춘 상태: 팔 · 손 목표는 pd 가 붙잡은 마지막 joint_target, hold 0, LSTM 0",
                             f"놓음 = 접촉 < {float(env['contact_force_threshold'])} N {int(env['release_steps'])} 스텝 → "
                             f"{int(env['settle_steps'])} 스텝 스크립트(손 펴기 · 팔 시작 관절로) → 끝 — 배포는 손끝 촉각 · 관절 힘으로 "
                             f"보고 네 손가락 손 목표가 인계보다 {OPEN_MIN_RAD} rad 넘게 열린 뒤에만 센다(손바닥 · 첫마디 접촉 없음)"]},
                       seat_dz=HOLDER_FLOOR_Z + origin_z, target_holders=[int(h) for h in env["target_holders"]],
                       release_steps=int(env["release_steps"]), settle_steps=int(env["settle_steps"]),
                       release_force_n=float(env["contact_force_threshold"]))
    validate(c)
    return c


def validate(c: PlaceContract) -> None:
    A.validate(c)
    if c.hold_steps != 0 or not c.target_holders or c.release_steps < 1 or c.settle_steps < 1 or c.release_force_n <= 0:
        raise RhPlaceError("hold 0 · 목표 홀더 · release/settle 스텝 · 놓음 힘")


def load_contract(path: str | Path) -> PlaceContract:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise RhPlaceError(f"{path}: schema {raw.get('schema')!r} ≠ {SCHEMA}")
    raw["sides"] = {r: A.RaSide(**s) for r, s in raw["sides"].items()}
    c = PlaceContract(**raw)
    validate(c)
    return c


def seat_goal(c: PlaceContract, holder_pos: Sequence[float]) -> A.Goal:
    return A.Goal(np.asarray(holder_pos, float) + np.array([0.0, 0.0, c.seat_dz]), np.array([1.0, 0.0, 0.0, 0.0]))


def open_hand_action(c: PlaceContract) -> np.ndarray:
    """편 손의 행동값 — sim _a_open(linear_grip_raw 역사상): (2·(open − lo)/(hi − lo) − 1), lo/hi = min/max(open, grip)."""
    s = c.side()
    o, g = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
    lo, hi = np.minimum(o, g), np.maximum(o, g)
    return np.clip(2.0 * (o - lo) / np.maximum(hi - lo, 1e-6) - 1.0, -1.0, 1.0)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """회전 행렬 → (w, x, y, z), 가장 큰 대각 성분으로 나눈다(Shepperd)."""
    m = np.asarray(R, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def is_free(c: PlaceContract, tactile_n: Sequence[float], joint_force: Sequence[float] | None,
            joint_free_g: float = JOINT_FREE_G) -> bool:
    """손이 컵에서 떨어졌다 — 손끝 촉각 전부 < 놓음 힘이고 관절 힘(있으면) 전부 < 문턱."""
    if float(np.max(np.asarray(tactile_n, float))) >= c.release_force_n:
        return False
    return joint_force is None or float(np.max(np.asarray(joint_force, float))) < joint_free_g


def start_refusals(c: PlaceContract, meas: Mapping) -> list:
    m = meas["arm"]
    if not grasp_signal(m.tactile_n, m.joint_force, AttachCfg(force_n=c.release_force_n)):
        return ["cup is not grasped (thumb AND another finger) — place starts from the aglt hand-off"]
    return []


class PlaceChain:
    """rh_place 한 스텝 — 붙은 컵 · 홀더 자리 목표로 관측 → 정책(놓은 뒤에는 스크립트) → rh_aglt 디코더."""

    def __init__(self, c: PlaceContract, policy, *, joint_free_g: float = JOINT_FREE_G, open_min_rad: float = OPEN_MIN_RAD) -> None:
        self.c, self.policy, self.joint_free_g, self.open_min_rad = c, policy, joint_free_g, open_min_rad
        self.dec = A.RaDecoder(c)
        self.prev = np.zeros(c.action_dim)
        self.step_i, self.free_streak, self.settle_t = 0, 0, -1
        self.goal: A.Goal | None = None
        self.rel: tuple[np.ndarray, np.ndarray] | None = None   # 손바닥 좌표계의 컵 (위치, 쿼터니언)
        self.start_q: np.ndarray | None = None
        self._a_open = open_hand_action(c)
        s = c.side()
        self._close_dir = np.sign(np.asarray(s.hand_grip, float) - np.asarray(s.hand_open, float))
        self._fingers = [i for i, j in enumerate(s.hand_joints) if "thumb" not in j]
        self.hand0: np.ndarray | None = None
        self.opened, self.tact_max, self.jf_max = 0.0, 0.0, None

    @property
    def settling(self) -> bool:
        return self.settle_t >= 0

    @property
    def done(self) -> bool:
        return self.settle_t >= self.c.settle_steps

    def reset(self, meas: Mapping, *, holder: Sequence[float] | None, held: tuple | None,
              allow_measured: bool = False) -> None:
        m = meas["arm"]
        if holder is None:
            raise RhPlaceError("holder pose missing — /objects/cup_holder_<k>/pose 가 와야 목표(자리)를 안다")
        s = self.c.side()
        hand_meas = np.array([m.hand_q[j] for j in s.hand_joints], float)
        if held is None and not allow_measured:
            raise RhPlaceError("held target missing — pd 가 붙잡은 마지막 joint_target 이 없다(aglt 뒤에 띄울 것, "
                               "실측에서 시작하려면 allow_measured)")
        arm0, hand0 = (m.arm_q, hand_meas) if held is None else held
        if len(arm0) != len(s.arm_joints) or len(hand0) != len(s.hand_joints):
            raise RhPlaceError("held target 은 팔 7 · 손 6")
        self.dec.reset(arm_start=arm0, hand_start=hand0)
        self.hand0 = np.asarray(hand0, float).copy()
        self.opened, self.tact_max, self.jf_max = 0.0, 0.0, None
        self.goal = seat_goal(self.c, holder)
        R = np.asarray(m.palm_R, float)
        self.rel = (R.T @ (np.asarray(m.cup_pos, float) - np.asarray(m.palm_pos, float)),
                    _quat_mul(_quat_conj(_matrix_to_quat(R)), np.asarray(m.cup_quat, float)))
        self.start_q = np.asarray(m.arm_q, float).copy()          # sim _start_q = 리셋 때 팔 실측(놓은 뒤 복귀 목표)
        self.prev = np.zeros(self.c.action_dim)
        self.step_i, self.free_streak, self.settle_t = 0, 0, -1
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def cup_from_palm(self, palm_pos: np.ndarray, palm_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rel is None:
            raise RhPlaceError("reset first — 컵이 아직 손바닥에 붙지 않았다")
        R = np.asarray(palm_R, float)
        return np.asarray(palm_pos, float) + R @ self.rel[0], _quat_mul(_matrix_to_quat(R), self.rel[1])

    def step(self, meas: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
        m = meas["arm"]
        pos, quat = self.cup_from_palm(m.palm_pos, m.palm_R)
        m = replace(m, cup_pos=pos, cup_quat=quat)
        obs = A.build_obs(self.c, m, self.dec, self.goal, self.prev)
        if self.settling:
            st = self.dec.state["arm"]
            a = np.concatenate([np.clip((self.start_q - st.arm_target) / self.c.k_arm, -1.0, 1.0), self._a_open])
        else:
            a = np.clip(np.asarray(self.policy.forward(obs), float), -self.c.action_clip, self.c.action_clip)
        targets = self.dec.step(a, active=True, tactile_n=m.tactile_n)
        f = self._fingers
        self.opened = float(np.mean(self._close_dir[f] * (self.hand0[f] - targets["arm"][1][f])))
        self.tact_max = float(np.max(np.asarray(m.tactile_n, float)))
        self.jf_max = None if m.joint_force is None else float(np.max(np.asarray(m.joint_force, float)))
        if self.settling:
            self.settle_t += 1
        else:
            free = self.opened >= self.open_min_rad and is_free(self.c, m.tactile_n, m.joint_force, self.joint_free_g)
            self.free_streak = self.free_streak + 1 if free else 0
            if self.free_streak >= self.c.release_steps:
                self.settle_t = 0
        self.prev = np.clip(a, -1.0, 1.0)
        self.step_i += 1
        return obs, a, targets

    def as_dict(self) -> dict:
        return {"phase": "done" if self.done else "settle" if self.settling else "place", "free_streak": self.free_streak,
                "settle_t": self.settle_t, "seat": None if self.goal is None else [round(float(v), 4) for v in self.goal.pos],
                # 첫 실기 문턱 정하기용(PLACE 검토) — 손 목표가 인계보다 연 양 · 손끝 최대 · 관절 힘 최대
                "opened": round(self.opened, 4), "tact_max": round(self.tact_max, 3),
                "jf_max": None if self.jf_max is None else round(self.jf_max, 1)}


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])

"""rh_cap 계열 — RH56F1 오른팔 캡 끼우기, 단계 분할 두 정책(hdgp open-rh_r_cap, cap_phase grasp · seat). 10.12 사용자: 간단 배포.

관측 91(촉각 없음) · 행동 13 · 디코더는 rh_aglt 와 같다(rh_aglt.build_obs · RaDecoder). 손 자세는 캡 프로필(rh56f1_right_captop:
편 손 = 엄지 옆으로 벌림 0.60, 쥔 손 = 엄지 1.57 · 네 손가락 1.40). 물체(cup) = 캡, 키포인트 반높이 = 캡 반높이.
목표 = 끼움 자리 = 리시버 원점 + 리시버 축 × seat_offset, 자세 = 리시버 자세(hdgp rh_cap_env._set_goal).

  grasp  시작 = 사전 파지 자세(배포 S0 = 사람이 맞춘 자세 · 미리 계산한 관절 경로). 팔 목표는 시작 자세 ± phase_arm_box_rad
         (관절마다) 안으로 자른다. 쥔 판정이 grasp_ik_lift_after 스텝 이어지면 팔은 IK 가 맡는다 — 그 순간 손바닥 자세(방향 고정)
         에서 grasp_ik_lift_speed 로 grasp_ik_lift_m 까지 수직으로(DLS, 증분 행동으로 되돌림). 손은 계속 정책.
         끝 = 들기 시작 뒤 캡이 시작 높이보다 grasp_success_lift_m 위 · 기울기 ≤ grasp_success_tilt_deg · 캡–손바닥 ≤
         grasp_carry_max_m 을 phase_success_steps 연속.
  seat   시작 = 끼우기 직전 자세(배포 S2). 캡은 시작 순간 손바닥 기준 상대 자세로 붙인다(rh_place 와 같다 — 쥔 채 FP++ 를 못 믿는다).
         끝 = |seat_dz| ≤ seat_tol · 리시버 축과 기울기 ≤ min(grade_perfect_tilt_deg, 기하 한계 + 1°) · 캡 속도 <
         seat_still_speed 를 phase_success_steps 연속 → 정지(놓지 않는다).
hdgp 기준 코드: tasks/rh_cap_r/rh_cap_env.py(_box_arm_actions · _ik_lift_actions · _phase_ok · _set_goal), cap_geom.py(seat_state).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from policy_control import rh_aglt as A
from policy_control.cup_attach import AttachCfg, grasp_signal
from policy_control.rh_place import _matrix_to_quat, _quat_conj, _quat_mul

SCHEMA = "policy_control/rh_cap_contract/v1"
TASK_PREFIX = "open-rh_{}_cap_{}"
PHASES = ("grasp", "seat")
PROFILE = "rh56f1_right_captop"
#: hdgp rh_cap_env_cfg.CAPTOP_PROFILE — RH56F1_RIGHT 에서 손 자세만 바꾼다
CAPTOP_OPEN = (0.60, 0.00, 0.00, 0.00, 0.00, 0.00)
CAPTOP_GRIP = (1.57, 0.24, 1.40, 1.40, 1.40, 1.40)
#: hdgp cap_geom.RCV_HALF_H — 리시버(cyl60) 원점 → 입구
RCV_HALF_H = 0.085
#: 계약 키(사용자 10.12 고정, s2r_1st 에 알림) — env.yaml 에서 그대로 옮긴다
PHASE_KEYS = ("cap_phase", "phase_arm_box_rad", "grasp_ik_lift_m", "grasp_ik_lift_after", "grasp_ik_lift_speed",
              "grasp_ik_damping", "grasp_carry_max_m", "grasp_success_lift_m", "grasp_success_tilt_deg",
              "phase_success_steps", "seat_tol", "grade_perfect_tilt_deg", "seat_still_speed", "grasp_min_others")
START_TILT_MAX_DEG = 15.0
FD_EPS = 1e-4                                       # 수치 야코비안 관절 섭동 [rad]


class RhCapError(A.RhAgltError):
    pass


@dataclass(frozen=True)
class CapContract(A.RaContract):
    cap_phase: str = ""
    phase_arm_box_rad: float = 0.0
    grasp_ik_lift_m: float = 0.0
    grasp_ik_lift_after: int = 0
    grasp_ik_lift_speed: float = 0.0
    grasp_ik_damping: float = 0.0
    grasp_carry_max_m: float = 0.0
    grasp_success_lift_m: float = 0.0
    grasp_success_tilt_deg: float = 0.0
    phase_success_steps: int = 0
    seat_tol: float = 0.0
    grade_perfect_tilt_deg: float = 0.0
    seat_still_speed: float = 0.0
    grasp_min_others: int = 1
    # 캡 기하(hdgp assets/cyl_cap/<dir>/cap_c<N>.json, 캡 원점 기준 m) — 학습 cap_asset_dir · cap_clearance_mm
    cap_json: str = ""
    seat_offset: float = 0.0            # 끼움 완료: 리시버 원점 → 캡 원점(리시버 축 방향)
    cap_z_bottom: float = 0.0           # 입(음수)
    cap_z_under: float = 0.0            # 윗판 아랫면
    cap_z_top: float = 0.0
    cap_tilt_max_deg: float = 0.0       # 끝까지 끼운 채 기울 수 있는 기하 한계
    rcv_r_rim: float = 0.0              # 리시버 입구 바깥 반지름(축 위 판정)


def is_rh_cap_run(env: Mapping) -> bool:
    return str(env.get("cap_phase", "")) in PHASES and str(env.get("profile_name", "")) == PROFILE


def cap_dims(hdgp_root: Path, env: Mapping) -> tuple[Path, dict]:
    """학습 런의 캡 json(hdgp cap_geom.load_cap_dims 와 같은 경로 규칙: 빈 값 = assets/cyl_cap, 상대 = hdgp 루트 기준)."""
    d = str(env.get("cap_asset_dir", "") or "")
    base = Path(hdgp_root) / "assets" / "cyl_cap" if not d else (Path(d).expanduser() if Path(d).expanduser().is_absolute()
                                                               else Path(hdgp_root) / d)
    p = base / f"cap_c{float(env['cap_clearance_mm']):g}.json"
    if not p.is_file():
        raise RhCapError(f"{p} 없음 — 학습 캡 기하(cap_asset_dir · cap_clearance_mm)를 못 찾았다")
    return p, json.loads(p.read_text())


def captop_profile(right_profile):
    """hdgp RH56F1_RIGHT → rh56f1_right_captop(손 자세만 바꾼다)."""
    return replace(right_profile, name=PROFILE, hand_open_pose=CAPTOP_OPEN, hand_grip_pose=CAPTOP_GRIP)


def build(run_dir: Path, checkpoint: Path, right_profile, urdf: Path, *, asset: str, hdgp_root: Path,
          obs_dim: int | None = None) -> CapContract:
    """rh_cap 단계 런 → 계약. obs_dim = 체크포인트가 실제로 받는 폭(덤프 observation_space 는 해석 전 96 일 수 있다)."""
    env_p = Path(run_dir) / "params" / "env.yaml"
    env = A.read_env(env_p)
    if not is_rh_cap_run(env):
        raise RhCapError(f"{env_p}: rh_cap 단계 런이 아니다(cap_phase grasp|seat · profile {PROFILE})")
    if int(env.get("hold_steps", -1)) != 0:
        raise RhCapError("rh_cap 단계 런은 hold_steps 0 이다")
    missing = [k for k in PHASE_KEYS if k not in env]
    if missing:
        raise RhCapError(f"{env_p}: 계약 키 {missing} 가 없다")
    jp, cj = cap_dims(hdgp_root, env)
    half_h = 0.5 * (float(cj["z_top"]) - float(cj["z_bottom"]))
    # 부모 빌더는 컵 표(CUP_OBJECTS)에 있는 물체만 안다 — 캡은 여기서 기하를 넣는다(관측 · 디코더 칸은 그대로 받는다)
    env_b = {**env, "object_name": "shaker", "object_names": None, "object_id_obs": False, "per_env_geometry": False}
    base = A.build_from_env(Path(run_dir), checkpoint, captop_profile(right_profile), urdf, asset=asset, env=env_b)
    tac = bool(env.get("actor_tactile_obs", True))
    want = A.obs_dim_of(tactile=tac)
    if obs_dim is not None and int(obs_dim) != want:
        raise RhCapError(f"체크포인트 관측 {obs_dim} ≠ 레이아웃 {want}(actor_tactile_obs {tac})")
    kw = {k: v for k, v in asdict(base).items() if not k.startswith("goal_")}
    kw["sides"] = base.sides
    phase = str(env["cap_phase"])
    notes = [n for n in base.notes if not n.startswith(("첫 목표", "목표 입력", "목표 달성", "컵 "))] + [
        f"물체 = 캡({jp.parent.name}/{jp.name}): 반높이 {half_h:.4f} m · 끼움 자리 리시버 축 +{float(cj['seat_offset']):.4f} m",
        "목표 = 리시버 원점 + 리시버 축 × seat_offset, 자세 = 리시버 자세(리시버 자세 토픽 · 왼손이 쥔 cyl60)",
        f"손 자세 = {PROFILE}(편 {list(CAPTOP_OPEN)} · 쥔 {list(CAPTOP_GRIP)})",
    ]
    if phase == "grasp":
        notes.append(f"grasp: 시작 = 사전 파지 자세(S0), 팔 목표 시작 ± {float(env['phase_arm_box_rad'])} rad · 쥔 판정 "
                     f"{int(env['grasp_ik_lift_after'])} 스텝 → IK 수직 {float(env['grasp_ik_lift_m'])} m @ "
                     f"{float(env['grasp_ik_lift_speed'])} m/s(DLS {float(env['grasp_ik_damping'])}) · 끝 = 들기 "
                     f"{float(env['grasp_success_lift_m'])} m · 기울기 ≤ {float(env['grasp_success_tilt_deg'])}° · "
                     f"{int(env['phase_success_steps'])} 스텝")
    else:
        notes.append(f"seat: 시작 = 끼우기 직전 자세(S2), 캡은 시작 순간 손바닥에 붙인다 · 팔 목표 시작 ± "
                     f"{float(env['phase_arm_box_rad'])} rad · 끝 = |seat_dz| ≤ {float(env['seat_tol'])} m · 기울기 ≤ "
                     f"{float(env['grade_perfect_tilt_deg'])}° · 캡 속도 < {float(env['seat_still_speed'])} m/s · "
                     f"{int(env['phase_success_steps'])} 스텝 → 정지(놓지 않음)")
    c = CapContract(**{**kw, "schema": SCHEMA, "task": TASK_PREFIX.format(base.side().side[0], phase),
                       "obs_dim": want, "cup_half_height": half_h, "goal_offset": [0.0, 0.0, 0.0], "notes": notes,
                       "actor_tactile_obs": tac},
                    cap_phase=phase, phase_arm_box_rad=float(env["phase_arm_box_rad"]),
                    grasp_ik_lift_m=float(env["grasp_ik_lift_m"]), grasp_ik_lift_after=int(env["grasp_ik_lift_after"]),
                    grasp_ik_lift_speed=float(env["grasp_ik_lift_speed"]), grasp_ik_damping=float(env["grasp_ik_damping"]),
                    grasp_carry_max_m=float(env["grasp_carry_max_m"]), grasp_success_lift_m=float(env["grasp_success_lift_m"]),
                    grasp_success_tilt_deg=float(env["grasp_success_tilt_deg"]),
                    phase_success_steps=int(env["phase_success_steps"]), seat_tol=float(env["seat_tol"]),
                    grade_perfect_tilt_deg=float(env["grade_perfect_tilt_deg"]),
                    seat_still_speed=float(env["seat_still_speed"]), grasp_min_others=int(env["grasp_min_others"]),
                    cap_json=str(jp), seat_offset=float(cj["seat_offset"]), cap_z_bottom=float(cj["z_bottom"]),
                    cap_z_under=float(cj["z_under"]), cap_z_top=float(cj["z_top"]),
                    cap_tilt_max_deg=float(cj["tilt_max_deg"]), rcv_r_rim=float(cj.get("rcv_r_rim", 0.030)))
    validate(c)
    return c


def validate(c: CapContract) -> None:
    A.validate(c)
    if c.cap_phase not in PHASES or c.hold_steps != 0 or c.phase_success_steps < 1 or c.seat_offset <= 0:
        raise RhCapError("cap_phase grasp|seat · hold 0 · 성공 스텝 ≥ 1 · seat_offset > 0")
    if c.cap_phase == "grasp" and (c.grasp_ik_lift_m <= 0 or c.grasp_ik_lift_speed <= 0 or c.grasp_ik_lift_after < 1):
        raise RhCapError("grasp 계약은 IK 들기(grasp_ik_lift_m · speed > 0 · after ≥ 1)가 있어야 한다")
    if list(c.side().hand_open) != list(CAPTOP_OPEN) or list(c.side().hand_grip) != list(CAPTOP_GRIP):
        raise RhCapError(f"손 자세가 {PROFILE} 가 아니다")


def load_contract(path: str | Path) -> CapContract:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise RhCapError(f"{path}: schema {raw.get('schema')!r} ≠ {SCHEMA}")
    raw["sides"] = {r: A.RaSide(**s) for r, s in raw["sides"].items()}
    c = CapContract(**raw)
    validate(c)
    return c


# ---------------------------------------------------------------- 기하(hdgp cap_geom 과 같은 식, numpy)
def axis_up(quat: Sequence[float]) -> np.ndarray:
    return A.quat_apply(quat, np.array([0.0, 0.0, 1.0]))


def seat_goal(c: CapContract, rcv_pos: Sequence[float], rcv_quat: Sequence[float]) -> A.Goal:
    return A.Goal(np.asarray(rcv_pos, float) + axis_up(rcv_quat) * c.seat_offset, np.asarray(rcv_quat, float))


def seat_state(c: CapContract, cap_pos, cap_up, rcv_pos, rcv_up) -> dict:
    """hdgp cap_geom.seat_state — seat_dz(+ = 덜 들어감) · lat(입구 높이 축 간 수평 거리) · rel_tilt(rad) · engaged."""
    d = np.asarray(cap_pos, float) - np.asarray(rcv_pos, float)
    cu, ru = np.asarray(cap_up, float), np.asarray(rcv_up, float)
    ax = float(d @ ru)
    cos_t = float(np.clip(cu @ ru, -1.0, 1.0))
    s = (RCV_HALF_H - ax) / max(cos_t, 0.2)
    p = d + cu * s
    lat = float(np.linalg.norm(p - ru * (p @ ru)))
    mouth_ax = ax + c.cap_z_bottom * cos_t
    engaged = float(np.clip(RCV_HALF_H - mouth_ax, 0.0, c.cap_z_under - c.cap_z_bottom)) * float(lat < c.rcv_r_rim)
    return {"seat_dz": ax - c.seat_offset, "lat": lat, "rel_tilt": float(np.arccos(cos_t)), "engaged": engaged}


def box_arm_action(c: CapContract, a_arm: np.ndarray, arm_target: np.ndarray, center: np.ndarray) -> np.ndarray:
    """hdgp _box_arm_actions — q_raw = q* + k·a 가 center ± box 안에 있게 팔 행동을 자른다(box 0 이면 그대로)."""
    box = c.phase_arm_box_rad
    a = np.clip(np.asarray(a_arm, float), -1.0, 1.0)
    if box <= 0.0:
        return a
    lo = np.minimum((center - box - arm_target) / c.k_arm, 1.0)
    hi = np.maximum((center + box - arm_target) / c.k_arm, -1.0)
    return np.maximum(np.minimum(a, hi), lo)


def _rotvec(R: np.ndarray) -> np.ndarray:
    """회전 행렬 → 회전 벡터(axis · angle)."""
    cos = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    ang = float(np.arccos(cos))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    if np.pi - ang < 1e-4:                         # 180° 근처 — 대각에서 축
        k = int(np.argmax(np.diag(R)))
        axis = R[:, k] + np.eye(3)[k]
        return ang * axis / np.linalg.norm(axis)
    return ang * v / (2.0 * np.sin(ang))


def palm_jacobian(fk, arm_q: np.ndarray, hand_q: np.ndarray) -> np.ndarray:
    """손바닥 (6, 7) 야코비안 [선속도 · 각속도(world)] — 수치 미분(FK 14 번, 60 Hz 에서 충분)."""
    q = np.asarray(arm_q, float)
    p0 = fk.palm_pose(q, hand_q)
    R0 = p0.extra["palm_rot"]
    J = np.zeros((6, q.size))
    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = FD_EPS
        p = fk.palm_pose(q + dq, hand_q)
        J[:3, i] = (p.palm_pos - p0.palm_pos) / FD_EPS
        J[3:, i] = _rotvec(p.extra["palm_rot"] @ R0.T) / FD_EPS
    return J


def ik_lift_action(c: CapContract, fk, arm_q: np.ndarray, hand_q: np.ndarray, arm_target: np.ndarray,
                   p0: np.ndarray, R0: np.ndarray, t: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """hdgp _ik_lift_actions — 손바닥을 p0 에서 방향 R0 고정으로 dz = min(t·v·dt, lift_m) 수직, DLS → 증분 행동."""
    dz = min(t * c.grasp_ik_lift_speed / c.policy_hz, c.grasp_ik_lift_m)
    pose = fk.palm_pose(arm_q, hand_q)
    err_p = np.asarray(p0, float) + np.array([0.0, 0.0, dz]) - pose.palm_pos
    err_r = _rotvec(np.asarray(R0, float) @ pose.extra["palm_rot"].T)
    J = palm_jacobian(fk, arm_q, hand_q)
    lam = c.grasp_ik_damping
    dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), np.concatenate([err_p, err_r]))
    q_des = np.clip(np.asarray(arm_q, float) + dq, lo, hi)
    return np.clip((q_des - np.asarray(arm_target, float)) / c.k_arm, -1.0, 1.0)


def _tilt_deg(quat) -> float:
    return float(np.degrees(np.arccos(np.clip(axis_up(quat)[2], -1.0, 1.0))))


def start_refusals(c: CapContract, meas: Mapping, grasp_cfg: AttachCfg | None = None) -> list:
    m = meas["arm"]
    if c.cap_phase == "grasp":
        tilt = _tilt_deg(m.cup_quat)
        return [] if tilt <= START_TILT_MAX_DEG else [f"cap tilt {tilt:.1f} deg (> {START_TILT_MAX_DEG:.0f}) — 서 있는 캡이어야 한다"]
    cfg = grasp_cfg or AttachCfg(force_n=c.grasp_threshold_n)
    if not grasp_signal(m.tactile_n, m.joint_force, cfg):
        return ["cap is not grasped — seat 는 쥔 캡을 끼우기 직전 자세(S2)에서 시작한다"]
    return []


# ---------------------------------------------------------------- 체인
class CapChain:
    """rh_cap 한 스텝 — 관측(리시버 끼움 자리 목표) → 정책 → 팔 상자 · IK 들기 → rh_aglt 디코더 → 단계 성공 판정.
    fk(UrdfChainFK) · grasp_cfg 는 노드가 넣는다(테스트는 직접)."""

    def __init__(self, c: CapContract, policy) -> None:
        self.c, self.policy = c, policy
        self.dec = A.RaDecoder(c)
        s = c.side()
        self._lo, self._hi = np.asarray(s.arm_lo, float), np.asarray(s.arm_hi, float)
        self.fk = None
        self.grasp_cfg: AttachCfg | None = None
        self.prev = np.zeros(c.action_dim)
        self.goal: A.Goal | None = None
        self.rcv: tuple[np.ndarray, np.ndarray] | None = None
        self.center: np.ndarray | None = None
        self.rel: tuple[np.ndarray, np.ndarray] | None = None
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.step_i, self.g_streak, self.succ_streak = 0, 0, 0
        self.lift_on, self.lift_t = False, 0
        self.lift_p0 = self.lift_R0 = None
        self.cap_z0 = 0.0
        self._prev_cap: np.ndarray | None = None
        self.last: dict = {}

    @property
    def done(self) -> bool:
        return self.succ_streak >= self.c.phase_success_steps

    def reset(self, meas: Mapping, *, receiver: tuple | None, held: tuple | None = None) -> None:
        """receiver = (pos, quat) base. held = (팔 7, 손 6) pd 가 붙잡은 마지막 목표 — 없으면 팔 실측 ·
        grasp 는 편 손 · seat 는 손 실측에서 시작한다(sim 뱅크 arm_q_target · hand_target)."""
        if receiver is None:
            raise RhCapError("receiver pose missing — 리시버 자세가 와야 목표(끼움 자리)를 안다")
        m = meas["arm"]
        s = self.c.side()
        hand_meas = np.array([m.hand_q[j] for j in s.hand_joints], float)
        arm0 = np.asarray(m.arm_q, float) if held is None else np.asarray(held[0], float)
        if self.c.cap_phase == "grasp":
            hand0 = np.asarray(s.hand_open, float)
        else:
            hand0 = hand_meas if held is None else np.asarray(held[1], float)
        if arm0.size != 7 or hand0.size != 6:
            raise RhCapError("시작 목표는 팔 7 · 손 6")
        self.dec.reset(arm_start=arm0, hand_start=hand0)
        self.center = np.clip(arm0, self._lo, self._hi)
        self.rcv = (np.asarray(receiver[0], float), np.asarray(receiver[1], float))
        self.goal = seat_goal(self.c, *self.rcv)
        self.rel = None
        if self.c.cap_phase == "seat":             # 쥔 캡 — 시작 순간 손바닥 기준으로 붙인다(rh_place 와 같다)
            R = np.asarray(m.palm_R, float)
            self.rel = (R.T @ (np.asarray(m.cup_pos, float) - np.asarray(m.palm_pos, float)),
                        _quat_mul(_quat_conj(_matrix_to_quat(R)), np.asarray(m.cup_quat, float)))
        self.prev = np.zeros(self.c.action_dim)
        self._reset_counters()
        self.cap_z0 = float(m.cup_pos[2])
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def cup_from_palm(self, palm_pos, palm_R) -> tuple[np.ndarray, np.ndarray]:
        if self.rel is None:
            raise RhCapError("seat: 캡이 손바닥에 붙지 않았다(reset 먼저)")
        R = np.asarray(palm_R, float)
        return np.asarray(palm_pos, float) + R @ self.rel[0], _quat_mul(_matrix_to_quat(R), self.rel[1])

    def _grasped(self, m) -> bool:
        cfg = self.grasp_cfg or AttachCfg(force_n=self.c.grasp_threshold_n)
        return bool(grasp_signal(m.tactile_n, m.joint_force, replace(cfg, force_n=self.c.grasp_threshold_n)))

    def step(self, meas: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.goal is None or self.center is None:
            raise RhCapError("reset first")
        m = meas["arm"]
        if self.c.cap_phase == "seat":
            pos, quat = self.cup_from_palm(m.palm_pos, m.palm_R)
            m = replace(m, cup_pos=pos, cup_quat=quat)
        s = self.c.side()
        obs = A.build_obs(self.c, m, self.dec, self.goal, self.prev)
        a = np.clip(np.asarray(self.policy.forward(obs), float), -self.c.action_clip, self.c.action_clip)
        st = self.dec.state["arm"]
        a_arm = box_arm_action(self.c, a[:7], st.arm_target, self.center)
        if self.c.cap_phase == "grasp":
            if not self.lift_on and self.g_streak >= self.c.grasp_ik_lift_after:
                if self.fk is None:
                    raise RhCapError("IK 들기에 FK 가 필요하다(노드가 chain.fk 를 넣는다)")
                pose = self.fk.palm_pose(m.arm_q, [m.hand_q[j] for j in s.hand_joints])
                self.lift_on, self.lift_t = True, 0
                self.lift_p0, self.lift_R0 = np.asarray(pose.palm_pos, float), np.asarray(pose.extra["palm_rot"], float)
            if self.lift_on:
                self.lift_t += 1
                a_arm = ik_lift_action(self.c, self.fk, m.arm_q, np.array([m.hand_q[j] for j in s.hand_joints]),
                                       st.arm_target, self.lift_p0, self.lift_R0, self.lift_t, self._lo, self._hi)
        a_out = np.concatenate([a_arm, np.clip(a[7:], -1.0, 1.0)])
        targets = self.dec.step(a_out, active=True, tactile_n=m.tactile_n)
        g = self._grasped(m)
        self.g_streak = self.g_streak + 1 if g else 0
        ok = self._phase_ok(m)
        self.succ_streak = self.succ_streak + 1 if ok else 0
        self.prev = np.clip(a_out, -1.0, 1.0)        # 학습 prev_actions = 상자 · IK 로 바꾼 뒤 행동(hdgp RhCapEnv 가 부모에 넘긴 값)
        self.step_i += 1
        return obs, a_out, targets

    def _phase_ok(self, m) -> bool:
        c = self.c
        cap = np.asarray(m.cup_pos, float)
        if c.cap_phase == "grasp":
            lift = float(cap[2] - self.cap_z0)
            tilt = _tilt_deg(m.cup_quat)
            carry = float(np.linalg.norm(cap - np.asarray(m.palm_pos, float)))
            self.last = {"lift_m": round(lift, 4), "tilt_deg": round(tilt, 1), "carry_m": round(carry, 4)}
            return (self.lift_on and carry <= c.grasp_carry_max_m and lift >= c.grasp_success_lift_m
                    and tilt <= c.grasp_success_tilt_deg)
        st = seat_state(c, cap, axis_up(m.cup_quat), self.rcv[0], axis_up(self.rcv[1]))
        speed = 0.0 if self._prev_cap is None else float(np.linalg.norm(cap - self._prev_cap)) * c.policy_hz
        self._prev_cap = cap.copy()
        tilt_ok = np.degrees(st["rel_tilt"]) <= min(c.grade_perfect_tilt_deg, c.cap_tilt_max_deg + 1.0)
        self.last = {k: round(float(v), 4) for k, v in st.items()} | {"speed": round(speed, 4)}
        return abs(st["seat_dz"]) <= c.seat_tol and bool(tilt_ok) and speed < c.seat_still_speed

    def as_dict(self) -> dict:
        return {"phase": self.c.cap_phase, "step": self.step_i, "g_streak": self.g_streak, "lift_on": self.lift_on,
                "lift_t": self.lift_t, "succ_streak": self.succ_streak, "done": self.done, **self.last,
                "goal": None if self.goal is None else [round(float(v), 4) for v in self.goal.pos]}

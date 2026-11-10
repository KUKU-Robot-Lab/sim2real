"""rh_place 10.25 준비 자세 이송(ready_carry) — hdgp rh_place_env._carry_actions_ready(carry_mode ready)와 같은 식(배포 잡음 0).

newton_07_r · newton_06_l 은 이송 끝 상태(1023_grasp_bank/carry_end_rcg04_{r,l}.npz, carry_mode ready)에서 학습을 시작했다.
그래서 배포도 같은 이송을 정책보다 먼저 한다(e2e 계약 hdgp scripts/tools/e2e_chain/e2e_contract.yaml 3_carry). 순서:
  · 리셋(넘김 묶음으로 컵을 붙인 뒤): 앵커 = (홀더 xy, 상판 + 물체 원점 높이 + dz). 상판 = 홀더 원점 − HOLDER_BASE_DZ(place_carry.carry_target).
    T_to = Mover 사슬 손바닥 FK(실측 q)⁻¹ ⊗ 지각 컵(손바닥 FK ⊗ 붙인 자세). 작업 공간 = line_region(앵커, +z, plan_down, 4, plan_xy, plan_tilt).
    끝 자세는 Mover.start 로 푼다(seeds 시드).
  · 이송 틱: dq = Mover.step(실측 q) → 팔 행동 = clamp(dq / (k_arm · arm_ema), ±1), 손 행동 = 리셋 때 손 목표(행동 단위)를 유지.
    정책 · 래치 추정기 · 래치는 돌리지 않는다.
  · 끝 = Mover done(도착, 또는 계획 시간 + settle_steps). 도착 = 계획 시간 안에 끝남(sim carry_arrive = carry_t < carry_max_steps).
    carry_ok(sim e2e carry_arrive) = 끝난 틱 수 < max_steps. 다음 틱부터 정책을 시작한다. 정책 RNN · 직전 행동 · 래치 추정기 상태는 0으로 둔다(place_grid_eval rand 와 같은 첫 관측 조건).
sim 과 다른 점: 앵커에 carry_noise ±(2, 2, 1) cm 를 더하지 않는다. 계획 ok(작업 공간 표본 전체 도달)는 sim 처럼 기록만 한다
(10.25 sim 잡기 뱅크 60 행 오프라인: ok 50/60 · 36/60 이어도 도착 58/60 · 60/60).
Mover 코드는 배포가 고정한 hdgp 커밋(config/hdgp_snapshot.yaml) 사본에서 읽는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: 계약 ready_carry 키 → env.yaml 키. env 에 모두 있어야 한다(sim 상수를 배포가 추측하지 않는다). max_steps 는 따로 받는다
#: (학습 carry_max_steps 240 · e2e 평가 상한 600 — 계약 3_carry stop).
ENV_KEYS = {"dz": "carry_dz", "plan_down": "carry_plan_down", "plan_xy": "carry_plan_xy", "plan_tilt_deg": "carry_plan_tilt_deg",
            "seeds": "carry_plan_seeds", "v_max": "carry_speed", "qd_max": "carry_qd_max", "min_steps": "carry_min_steps",
            "tol": "carry_tol", "q_tol": "carry_q_tol"}
KEYS = tuple(ENV_KEYS) + ("max_steps",)
N_DOWN = 4                      # sim line_region(…, carry_plan_down, 4, …)
# ★11.10 이송 끝 → 정책 시작 문턱 = sim 학습 뱅크(정책이 본 시작 상태) 이송 끝 오차 p95
#   (deploy/policies/rh56f1/place/right_n10_e4000/bank/bank_right_train.npz, 350 행: carry_err_mm p95 29.04 · carry_axis_deg p95 4.69,
#   컵 z p5~p95 0.352~0.394 m). 넘으면 정책을 시작하지 않는다(실기 161342 끝 err 48 mm · 컵 z 0.421 m 에서 손이 펴졌다).
#   Mover arrived(q_tol 0.05 rad 포함)는 쓰지 않는다 — 계획 ok False 면 컵이 목표 0.8 mm 에 닿아도 q 가 q_goal 과 달라 False 다(오프라인 dry-run).
START_ERR_MAX_M = 0.029
START_AXIS_MAX_DEG = 4.7
# ★11.11 이송 끝(도착 또는 시간 끝) → 정책 넘기기 전 "팔이 멈췄다" 확인. 모든 place 계약 policy_hz 60 기준 틱.
#   근거 bag 261110/184024_episode_cocktail_1c_e4000 arm_0.db3(오른팔, js 740 Hz · pd diagnostics):
#   · 정지 구간(홈 735.2~738.7 s · 놓은 뒤 pd 유지 712.5~721.3 s) 실측 max|q̇| < 0.05 rad/s 인 60 Hz 틱 91~100 %,
#     6 틱 연속이 1 s 안에 나올 확률 1.00(중앙 대기 6 틱). 0.02 rad/s 는 속도 양자화 · 잡음 때문에 0.03~0.24 라 쓸 수 없다.
#   · (같은 bag 의 이송 끝 50 ms 실측 |q̇| 최대 0.12~0.16 rad/s — 떨리는 채로 정책을 넘겼다)
#   · 옛 명령 − 실측 문턱 STILL_DQ_RAD 0.03 은 sim 처짐 0.061 에서 늘 timeout 이라 SAG_DONE_RAD(|q_ref − q|)로 바꿨다(11.11).
STILL_QD_RAD_S = 0.05
STILL_TICKS = 6                 # 0.1 s
# ★11.11 처짐 보정 1.5 s(아래 SAG_*: 처짐 0.061 → 0.01 rad 이 K_SAG 0.05/틱 · 지연 0.14 s 에서 약 0.6 s + 지연 + 멈춤 6 틱,
#   움직이는 틱은 쉬므로 여유 2 배)를 더해 1.0 → 2.5 s. 넘으면 그대로 정책으로 넘기고 노드가 경고(as_dict still "timeout").
STILL_MAX_TICKS = 150
# ★11.11 처짐 보정(안 A): 개루프 이송(fa3a7b5)은 컵 든 팔의 정상상태 중력 처짐(sim j7 0.061 · j4 0.021 · j6 0.014 rad, 실기 j7 약 0.019)을
#   못 고쳐 끝 컵 축 오차가 4.7° 이상 남는다(FK 처짐만 p50 4.65°). sim 학습은 실측 q 닫힌 고리라 뱅크 끝 명령이 실측보다 처짐만큼 앞선다.
#   Mover 가 끝난 뒤 끝 명령 q_ref 를 고정하고, 실측이 멈춘 틱(max|q̇| < STILL_QD_RAD_S)에만 q_cmd += clip(K_SAG·(q_ref − q), ±SAG_STEP_MAX)
#   (|q_cmd − q_ref| ≤ SAG_MAX_RAD). 실측 q 빠른 고리(30/s, 지연 약 0.14 s)는 1.8 Hz 로 떨렸다(root_cause/cocktail_1111/q4_place_vibration.md).
#   K_SAG 0.05/틱 = 3/s → K·τ ≈ 0.42 rad, 위상 여유 약 66°, 옛 고리 이득의 1/10.
K_SAG = 0.05
SAG_STEP_MAX = 0.005            # rad/틱 = 행동 ±1 증분(k_arm·arm_ema)
SAG_MAX_RAD = 0.08              # 관절별 누적 상한 — sim 처짐 최대 0.061 + 여유
SAG_DONE_RAD = 0.01             # max|q_ref − q| 이 아래로 STILL_TICKS 연속 + 멈춤 → still "ok"
# ★11.11 개루프 이송의 명령 − 실측 이탈 가드: max|q_cmd − q| > TRACK_ABORT_RAD 가 TRACK_ABORT_TICKS 연속이면 이송 실패
#   (still "track", start_ok False → 노드는 정책을 시작하지 않고 멈춘다). 근거 184024 이송 536 틱(697.98~706.91 s):
#   max|q* − q| p50 0.023 · p99 0.041 · 최대 0.041 rad(관절 j1 · j5 · j7), 0.05 넘는 틱 0 · 명령 한 틱 변화 ≤ 0.005 rad.
#   0.1 rad 은 정상 최대의 2.4 배. pd abort_tracking 0.5 rad(세트포인트 − 실측 → HOLD)보다 먼저 이송을 멈춘다.
TRACK_ABORT_RAD = 0.1
TRACK_ABORT_TICKS = 6           # 0.1 s — 한두 틱 잡음 · 지연으로 끊지 않는다


class ReadyCarryError(ValueError):
    pass


@dataclass(frozen=True)
class ReadyCarryCfg:
    dz: float
    plan_down: float
    plan_xy: float
    plan_tilt_deg: float
    seeds: int
    v_max: float
    qd_max: float
    min_steps: int
    tol: float
    q_tol: float
    max_steps: int

    @classmethod
    def from_env(cls, env: Mapping, max_steps: int) -> "ReadyCarryCfg":
        """학습 env.yaml 에서. 이송 끝 뱅크로 학습한 런(start_target_from_bank · carry_mode ready)만 받는다."""
        if str(env.get("carry_mode", "")) != "ready":
            raise ReadyCarryError(f"carry_mode {env.get('carry_mode')!r} — ready 런만(Mover 이송)")
        if bool(env.get("carry_enable", False)):
            raise ReadyCarryError("carry_enable 런은 학습 안에서 이송했다 — 이 모듈은 이송 끝 뱅크로 학습한 런용")
        if str(env.get("carry_ref_bank", "") or ""):
            raise ReadyCarryError("carry_ref_bank(끝 자세 기준 뱅크) 런은 아직 배포에 없다")
        missing = [v for v in ENV_KEYS.values() if v not in env]
        if missing:
            raise ReadyCarryError(f"env.yaml 에 {missing} 가 없다")
        return cls.from_dict({**{k: env[v] for k, v in ENV_KEYS.items()}, "max_steps": max_steps})

    @classmethod
    def from_dict(cls, d: Mapping) -> "ReadyCarryCfg":
        if set(d) != set(KEYS):
            raise ReadyCarryError(f"ready_carry 키 {sorted(d)} ≠ {sorted(KEYS)}")
        c = cls(dz=float(d["dz"]), plan_down=float(d["plan_down"]), plan_xy=float(d["plan_xy"]),
                plan_tilt_deg=float(d["plan_tilt_deg"]), seeds=int(d["seeds"]), v_max=float(d["v_max"]),
                qd_max=float(d["qd_max"]), min_steps=int(d["min_steps"]), tol=float(d["tol"]), q_tol=float(d["q_tol"]),
                max_steps=int(d["max_steps"]))
        if not (c.dz > 0 and c.plan_down >= 0 and c.plan_xy >= 0 and c.plan_tilt_deg >= 0 and c.seeds >= 1
                and 0 < c.v_max <= 0.005 and 0 < c.qd_max <= 0.03 and 0 <= c.min_steps <= c.max_steps
                and c.tol > 0 and c.q_tol > 0):
            raise ReadyCarryError(f"ready_carry 값이 범위 밖이다: {c}")
        return c

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in KEYS}


def quat_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, float) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_chain(palm_body: str, arm_joints: Sequence[str], hdgp=None):
    """(ready_carry 패키지, Mover 사슬) — 배포 고정 hdgp 사본의 URDF · 코드(sim cfg_profile_palm 과 같은 손바닥 몸체)."""
    from policy_control import hdgp_pin as H
    root = H.hdgp_dir() if hdgp is None else hdgp
    RC = H.load_ready_carry(root)
    return RC, RC.chain.Chain.from_urdf(RC.chain.default_urdf(), str(palm_body), list(arm_joints))


class ReadyCarry:
    """한 에피소드 이송 상태(B = 1, CPU). start → step… → on False. 사슬은 노드가 한 번 만들어 넘긴다(URDF 읽기)."""

    def __init__(self, cfg: ReadyCarryCfg, RC, chain, *, k_arm: float, arm_ema: float, hand_action: Sequence[float],
                 track_abort_rad: float = TRACK_ABORT_RAD, sag_max_rad: float = SAG_MAX_RAD) -> None:
        import torch
        self.torch, self.cfg, self.RC, self.ch = torch, cfg, RC, chain
        self.scale = float(k_arm) * float(arm_ema)
        self.hand_action = np.asarray(hand_action, float).copy()
        self.track_abort_rad = float(track_abort_rad)   # sim 런타임은 0.2 를 줄 수 있다(실기 기본 0.1)
        self.sag_max_rad = float(sag_max_rad)           # 기본 0.08(sim 처짐 0.061) · 실기 첫 시험은 0.04 를 줄 수 있다
        # ★11.10 Mover 관절 상한 = 행동 ±1 이 낼 수 있는 틱당 증분(k_arm·arm_ema = 0.05·0.1 = 0.005 rad/틱, 계약 rh_place_contract.json
        #   k_arm · arm_ema). 계약 qd_max 0.012(hdgp MoverCfg 기본)는 step 의 ±1 자름에 잘려 계획 T 가 짧고 관절 4 가 못 따라갔다
        #   (실험기 logs/runs/261110/root_cause/place_offline/carry_clip_out.txt: 잘린 틱 0.87~0.95 · 관절 4 누적 −0.74~−0.89 rad ·
        #   끝 err 37~48 mm). 상한을 행동 한계로 낮추면 Mover 가 비율 축소로 방향을 지키고 T 를 그만큼 늘린다. 실제 관절 속도는 전과 같거나 느리다.
        self.qd_max = min(float(cfg.qd_max), self.scale)
        self.mv = RC.mover.Mover(chain, RC.mover.MoverCfg(v_max=cfg.v_max, qd_max=self.qd_max, min_steps=cfg.min_steps,
                                                          max_steps=cfg.max_steps, tol=cfg.tol, q_tol=cfg.q_tol, seeds=cfg.seeds),
                                 1, "cpu")
        self.on, self.arrived, self.t = False, False, 0
        self.moving = False                 # Mover 가 도는 중(on 이면서 moving 아니면 멈춤 확인 중)
        self.still_n = self.still_t = 0
        self.still = ""                     # "" · "ok" · "timeout" · "track"(명령 − 실측 이탈로 실패)
        self.track_n = 0
        self.track_dev_max = float("nan")   # 이송 중 max|q_cmd − q| 최대(NaN 이 한 번이라도 오면 NaN) — 실기 로그로 문턱 재검토
        self.plan: dict = {}
        self.target: np.ndarray | None = None
        self.err = float("nan")             # 실측 q 의 컵 오차(start_ok 판정)
        self.axis_deg = float("nan")
        self.err_cmd = float("nan")         # Mover(명령 q) 쪽 오차 — 기록용
        self.q_ref: np.ndarray | None = None  # 처짐 보정 기준 = Mover 끝 명령(멈춤 확인 첫 틱의 q_cmd)
        self.sag_ticks = 0
        self.sag_max: np.ndarray | None = None

    def start(self, arm_q: Sequence[float], cup_pos: Sequence[float], cup_quat: Sequence[float], target: Sequence[float]) -> dict:
        """끝 자세를 푼다(움직이지 않는다) → 계획 요약. 실측 팔 q · 지각 컵(세계 = 로봇 루트) · 앵커."""
        torch, c, P = self.torch, self.cfg, self.RC.posture
        q = torch.tensor([list(arm_q)], dtype=torch.float32)
        Tp = self.ch.fk(q)
        Tc = torch.eye(4).repeat(1, 1, 1)
        Tc[0, :3, :3] = torch.tensor(quat_matrix(cup_quat), dtype=torch.float32)
        Tc[0, :3, 3] = torch.tensor(np.asarray(cup_pos, float), dtype=torch.float32)
        T_to = torch.linalg.inv(Tp) @ Tc
        tg = torch.tensor(np.asarray(target, float)[None], dtype=torch.float32)
        za = torch.tensor([[0.0, 0.0, 1.0]])
        reg = P.line_region(tg, za, c.plan_down, N_DOWN, c.plan_xy, c.plan_tilt_deg)
        plan = self.mv.start(torch.arange(1), q, T_to, None, tg, za, reg)
        self.T_to, self.target = T_to, np.asarray(target, float).copy()
        self.on, self.moving, self.arrived, self.t = True, True, False, 0
        self.still_n = self.still_t = self.track_n = 0
        self.track_dev_max = 0.0
        self.still = ""
        self.q_ref, self.sag_ticks, self.sag_max = None, 0, np.zeros(len(arm_q))
        self.plan = {"ok": bool(plan.ok[0]), "margin_deg": round(float(np.degrees(float(plan.margin[0]))), 2),
                     "anchor_err_mm": round(float(plan.err_pos[0]) * 1000, 2), "anchor_err_deg": round(float(plan.err_axis[0]), 2),
                     "T_steps": int(self.mv.T[0]), "q_goal": [round(float(v), 4) for v in plan.q[0]],
                     "target": [round(float(v), 4) for v in self.target]}
        return self.plan

    def step(self, arm_q: Sequence[float], q_cmd: Sequence[float] | None = None,
             arm_qd: Sequence[float] | None = None) -> np.ndarray:
        """이송 한 틱 → 행동(팔 7 + 손). 끝나면 on False(다음 틱부터 정책).

        ★11.11 q_cmd(디코더 팔 목표 q*)를 주면 Mover 를 명령으로 굴린다(개루프, 사용자 "mover 말고 ready-approach 처럼 명령 목표로").
        실측 q 로 굴리면 PD 추종 오차가 되먹여 1.5~2 Hz 로 떨렸다(q4_place_vibration). 디코더 목표 = 목표 + k_arm·arm_ema·a 라
        q* 는 Mover dq 를 그대로 적분한다 — plan_reach.roll_mover(배포 ReadyApproach)와 같은 식. Mover 가 끝나면 arm_qd 까지 주어진
        경우 처짐 보정(SAG_*, 멈춘 틱에만 명령을 q_ref 쪽으로 조금씩)과 멈춤 확인 뒤 끝낸다. start_ok 오차는 늘 실측 q 의 컵으로 본다."""
        torch = self.torch
        q = torch.tensor([list(arm_q)], dtype=torch.float32)
        a_arm = np.zeros(len(arm_q))
        if q_cmd is not None:                          # ★11.11 명령 − 실측 이탈 가드(이동 · 멈춤 확인 둘 다). NaN 은 위반으로 센다
            qc = np.asarray(q_cmd, float)
            if not self.moving and self.q_ref is None:
                self.q_ref = qc.copy()
            ref = qc if self.moving else self.q_ref        # 보정 단계는 의도한 앞섬(≤ SAG_MAX_RAD)을 이탈로 세지 않는다
            dev = float(np.max(np.abs(ref - np.asarray(arm_q, float))))
            self.track_dev_max = (float("nan") if np.isnan(dev) or np.isnan(self.track_dev_max)
                                  else max(self.track_dev_max, dev))
            self.track_n = self.track_n + 1 if not (dev <= self.track_abort_rad) else 0
            if self.track_n >= TRACK_ABORT_TICKS:      # 팔이 명령을 못 따라간다 — 이송 실패(호출자가 팔 목표를 실측으로 덮는다)
                self.on = self.moving = False
                self.still = "track"
                self.err, self.axis_deg = self._meas_err(q)
                return np.concatenate([a_arm, self.hand_action])
        if self.moving:
            qm = q if q_cmd is None else torch.tensor([list(q_cmd)], dtype=torch.float32)
            dq, _, done, arrive = self.mv.step(qm)
            self.t += 1
            self.err_cmd = float(self.mv.last_err[0])
            if bool(done[0]):
                self.moving, self.arrived = False, bool(arrive[0])
                if q_cmd is None or arm_qd is None:     # 멈춤 확인 재료가 없으면 옛 동작(이 틱에 끝)
                    self.on = False
            a_arm = np.clip(dq[0].double().numpy() / self.scale, -1.0, 1.0)
        else:
            self.still_t += 1
            qd_max = float(np.max(np.abs(np.asarray(arm_qd, float))))
            e = self.q_ref - np.asarray(arm_q, float)
            dev = float(np.max(np.abs(e)))
            if qd_max < STILL_QD_RAD_S and np.isfinite(dev):   # 움직이는 틱 · NaN 은 적분하지 않는다(명령 유지)
                off = np.asarray(q_cmd, float) - self.q_ref
                dq_s = np.clip(off + np.clip(K_SAG * e, -SAG_STEP_MAX, SAG_STEP_MAX), -self.sag_max_rad, self.sag_max_rad) - off
                a_arm = np.clip(dq_s / self.scale, -1.0, 1.0)   # 디코더: q* += k_arm·arm_ema·a
                self.sag_ticks += 1
                self.sag_max = np.maximum(self.sag_max, np.abs(off + dq_s))
            still = qd_max < STILL_QD_RAD_S and dev < SAG_DONE_RAD   # NaN 이면 False(멈춤 아님)
            self.still_n = self.still_n + 1 if still else 0
            if self.still_n >= STILL_TICKS:
                self.on, self.still = False, "ok"
            elif self.still_t >= STILL_MAX_TICKS:
                self.on, self.still = False, "timeout"
        if not self.on:
            self.err, self.axis_deg = self._meas_err(q)
        return np.concatenate([a_arm, self.hand_action])

    def _meas_err(self, q) -> tuple[float, float]:
        """실측 q 의 컵 원점 · 축 오차(앵커 기준) — Mover last_err 와 같은 식."""
        _, p, z, _, _ = self.RC.posture.object_state(self.ch, q, self.T_to, None)
        err = float((self.mv.pa - p).norm(dim=-1)[0])
        ax = float(np.degrees(np.arccos(np.clip(float((z * self.mv.za).sum(-1)[0]), -1.0, 1.0))))
        return err, ax

    @property
    def start_ok(self) -> bool:
        """★11.10 이송이 끝났고 컵이 sim 뱅크 시작 분포 안(오차 · 축)이다 — False 면 정책을 시작하지 않는다(NaN 은 False)."""
        return (not self.on and self.t > 0 and self.still != "track" and self.err <= START_ERR_MAX_M and self.axis_deg <= START_AXIS_MAX_DEG)

    def as_dict(self) -> dict:
        return {"on": self.on, "t": self.t, "arrived": self.arrived, "start_ok": self.start_ok, "qd_max": self.qd_max,
                "carry_ok": (not self.on) and self.t < self.cfg.max_steps and self.still != "track",
                "err_mm": None if np.isnan(self.err) else round(self.err * 1000, 2),
                "axis_deg": None if np.isnan(self.axis_deg) else round(self.axis_deg, 2),
                "err_cmd_mm": None if np.isnan(self.err_cmd) else round(self.err_cmd * 1000, 2),
                "still": self.still, "still_ticks": self.still_t, "plan": self.plan,
                "track_dev_max": None if np.isnan(self.track_dev_max) else round(self.track_dev_max, 4),
                "sag_ticks": self.sag_ticks,
                "sag_max": None if self.sag_max is None else [round(float(v), 4) for v in self.sag_max]}


def roll(rc: ReadyCarry, arm_q: Sequence[float], *, k_arm: float, arm_ema: float) -> tuple[np.ndarray, dict]:
    """오프라인 점검(dry-run): start 뒤 이송 행동(±1 자름)을 배포 디코더 식(목표 = EMA(목표, 목표 + k_arm·a))으로 굴리고 실측 = 목표
    (완벽 추종)로 둔다 — sim _carry_actions_ready 와 같은 행동 경로. → (경로 (N,7), 요약). 움직이지 않는다."""
    q = np.asarray(arm_q, float).copy()
    rows = [q.copy()]
    for _ in range(rc.cfg.max_steps + rc.mv.cfg.settle_steps + 5):
        a = rc.step(q)
        q = arm_ema * (q + k_arm * a[:len(q)]) + (1.0 - arm_ema) * q
        rows.append(q.copy())
        if not rc.on:
            break
    return np.asarray(rows), rc.as_dict()


def cup_track(rc: ReadyCarry, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """경로 → (컵 원점 (N,3), 컵 축 (N,3)) — Mover 와 같은 T_to."""
    torch = rc.torch
    q = torch.tensor(path, dtype=torch.float32)
    _, p, z, _, _ = rc.RC.posture.object_state(rc.ch, q, rc.T_to.expand(len(path), 4, 4), None)
    return p.double().numpy(), z.double().numpy()

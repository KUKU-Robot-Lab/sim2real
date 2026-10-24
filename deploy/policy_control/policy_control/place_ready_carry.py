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

    def __init__(self, cfg: ReadyCarryCfg, RC, chain, *, k_arm: float, arm_ema: float, hand_action: Sequence[float]) -> None:
        import torch
        self.torch, self.cfg, self.RC, self.ch = torch, cfg, RC, chain
        self.scale = float(k_arm) * float(arm_ema)
        self.hand_action = np.asarray(hand_action, float).copy()
        self.mv = RC.mover.Mover(chain, RC.mover.MoverCfg(v_max=cfg.v_max, qd_max=cfg.qd_max, min_steps=cfg.min_steps,
                                                          max_steps=cfg.max_steps, tol=cfg.tol, q_tol=cfg.q_tol, seeds=cfg.seeds),
                                 1, "cpu")
        self.on, self.arrived, self.t = False, False, 0
        self.plan: dict = {}
        self.target: np.ndarray | None = None
        self.err = float("nan")
        self.axis_deg = float("nan")

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
        self.on, self.arrived, self.t = True, False, 0
        self.plan = {"ok": bool(plan.ok[0]), "margin_deg": round(float(np.degrees(float(plan.margin[0]))), 2),
                     "anchor_err_mm": round(float(plan.err_pos[0]) * 1000, 2), "anchor_err_deg": round(float(plan.err_axis[0]), 2),
                     "T_steps": int(self.mv.T[0]), "q_goal": [round(float(v), 4) for v in plan.q[0]],
                     "target": [round(float(v), 4) for v in self.target]}
        return self.plan

    def step(self, arm_q: Sequence[float]) -> np.ndarray:
        """이송 한 틱 → 행동(팔 7 + 손). 이 틱으로 Mover 가 끝나면 on False(다음 틱부터 정책)."""
        torch = self.torch
        q = torch.tensor([list(arm_q)], dtype=torch.float32)
        dq, _, done, arrive = self.mv.step(q)
        self.t += 1
        self.err = float(self.mv.last_err[0])
        self.axis_deg = float(self.mv.last_axis_deg[0])
        if bool(done[0]):
            self.on, self.arrived = False, bool(arrive[0])
        a_arm = np.clip(dq[0].double().numpy() / self.scale, -1.0, 1.0)
        return np.concatenate([a_arm, self.hand_action])

    def as_dict(self) -> dict:
        return {"on": self.on, "t": self.t, "arrived": self.arrived, "carry_ok": (not self.on) and self.t < self.cfg.max_steps,
                "err_mm": None if np.isnan(self.err) else round(self.err * 1000, 2),
                "axis_deg": None if np.isnan(self.axis_deg) else round(self.axis_deg, 2), "plan": self.plan}


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

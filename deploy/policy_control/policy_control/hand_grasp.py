"""경로 B 손 쥐기 — 팔은 제자리, 손을 닿을 때까지 닫고 막힘 래치로 쥔 판정(10.17 사용자, docs/PLAN_GRASP_TILT_EVAL_2026-10-17.md).

  "팔을 tilt 시작 자세로 → 손 펴기 → 사용자가 컵을 손에 댐 → 닿을 때까지 닫고 래치. 확인 두 번(컵 대기 전, 닫기 전)."
확인 두 번은 에피소드 confirm 노드가 받는다. 이 모듈은 닫기 · 판정 · 유지만 한다(순수, ROS 없음 — tools/hand_grasp.py 가 돌린다).

경로 C(approach_grasp)의 close · judge · settle 과 같은 규칙이다:
  close   엄지 회전(thumb_1)은 시작 값 그대로. 엄지 굽힘(thumb_delay_s 뒤)과 네 손가락을 close_rad_s 로 쥔 자세(q_grip) 쪽으로.
          막힘이 래치된 축은 그 자리에서 더 닫지 않는다(그 목표 ↔ 실측 간극이 쥐는 힘, pd 앞섬 상한 0.2 rad).
  judge   네 손가락 중 judge_min_axes 축 이상 래치가 judge_ticks 이어지면 쥠. close_max_s 안에 못 쥐면 실패.
          거리 조건(컵 ↔ 손바닥)은 보지 않는다 — 사람 손이 컵을 가려 FP++ 가 못 볼 수 있고, 컵은 사람이 손에 댄 것이다.
  hold    hold_s 동안 래치 수가 judge_min_axes 아래로 loss_ticks 넘게 떨어지면 실패(결정 1 의 3 s 유지와 같은 판정).
  done    성공 — 도구가 episode stop 을 내고 pd 가 마지막 팔 · 손 목표로 붙든다(다음 단계 tilt 가 이어받는다).
막힘 명목값 · 판정 축은 경로 C 와 같다(approach_grasp.close_contract — 배포 rh_aglt.HAND_BLOCK_NOMINAL, opp k=2).
★10.18 fake 리허설: pd 가 붙들던 손(내부 목표)에서 이 도구의 목표로 넘어갈 때 손 실측이 약 0.6 s 동안 안 움직였다 — 명령은 앞서는데
  손가락이 멈춰 있으니 막힘 판정이 컵 접촉으로 오인해 네 손가락을 래치했고, 손이 움직이기 시작하자 래치가 풀려 '유지 중 풀림'으로
  끝났다. 그래서 손가락마다 시작 실측에서 RESPOND_RAD 넘게 움직인 뒤에만 래치를 인정한다(손이 명령을 따르기 시작한 뒤의 멈춤만
  막힘이다). 사람이 댄 컵은 편 손가락과 떨어져 있으므로 닿기 전에 반드시 움직인다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Mapping, Sequence

import numpy as np

from policy_control import rh_aglt as A
from policy_control.approach_grasp import SAFE_RANGE as _AG_RANGE
from policy_control.approach_grasp import close_contract

PHASES = ("close", "hold", "done", "failed")
RESPOND_RAD = 0.01               # 손가락이 명령을 따르기 시작했다고 보는 이동(레지스터 칸 0.0018 rad 의 5 배)
_KEYS = ("close_rad_s", "thumb_delay_s", "close_max_s", "judge_min_axes", "judge_ticks", "hold_s", "loss_ticks")
SAFE_RANGE: Mapping[str, tuple] = {k: _AG_RANGE[k] for k in _KEYS}


class HandGraspError(ValueError):
    pass


@dataclass(frozen=True)
class GraspCfg:
    close_rad_s: float = 0.6        # 경로 C 와 같다(컵 밀림을 줄이려 낮게)
    thumb_delay_s: float = 0.3
    close_max_s: float = 4.0
    judge_min_axes: int = 2         # 네 손가락 중 래치 축 수(opp k=2)
    judge_ticks: int = 15           # 0.25 s(60 Hz)
    hold_s: float = 3.0             # 결정 1: 3 s 유지
    loss_ticks: int = 30

    def __post_init__(self) -> None:
        bad = [f"{f.name}={getattr(self, f.name)} (허용 {SAFE_RANGE[f.name][0]:g}~{SAFE_RANGE[f.name][1]:g})"
               for f in fields(self)
               if not (isinstance(getattr(self, f.name), (int, float)) and math.isfinite(getattr(self, f.name))
                       and SAFE_RANGE[f.name][0] <= getattr(self, f.name) <= SAFE_RANGE[f.name][1])]
        if bad:
            raise HandGraspError(f"hand grasp: 안전 범위 밖 — {bad}")

    @staticmethod
    def from_mapping(over: Mapping | None) -> "GraspCfg":
        """에피소드 YAML cfg {키: 값} → GraspCfg(모르는 키 · 숫자 아닌 값 거부)."""
        over = dict(over or {})
        unknown = sorted(set(over) - set(_KEYS))
        if unknown:
            raise HandGraspError(f"hand grasp cfg: 모르는 키 {unknown} — {list(_KEYS)}")
        kw = {}
        for k, v in over.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise HandGraspError(f"hand grasp cfg {k}={v!r} — 숫자가 아니다")
            kw[k] = int(v) if k in ("judge_min_axes", "judge_ticks", "loss_ticks") else float(v)
        return GraspCfg(**kw)

    @property
    def worst_s(self) -> float:
        return self.close_max_s + self.judge_ticks / 60.0 + self.hold_s


def block_contract(c: A.RaContract, cfg: GraspCfg) -> A.RaContract:
    """경로 C 와 같은 막힘 계약(명목값 · opp 판정) — 거리 조건은 끈다(grasp_near_m 0)."""
    cc = replace(close_contract(c), grasp_min_axes=int(cfg.judge_min_axes), grasp_near_m=0.0)
    A.validate(cc)
    return cc


class HandGrasp:
    """한 손 닫기 · 판정 · 유지. step(q_meas) 를 hz 로 부르면 그 틱의 손 목표를 돌려준다(손 관절 순 = 계약 hand_joints)."""

    def __init__(self, c: A.RaContract, cfg: GraspCfg, hz: float) -> None:
        self.cfg, self.hz = cfg, float(hz)
        self.c = block_contract(c, cfg)
        s = self.c.side()
        self.names = list(s.hand_joints)
        self.q_open, self.q_grip = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
        sgn = np.sign(self.q_grip - self.q_open)
        self.sgn = np.where(sgn == 0, 1.0, sgn)
        self.thumb_rot = [i for i, n in enumerate(self.names) if n.endswith("thumb_1")]
        self.block = A.make_block(self.c)
        self.q_cmd: np.ndarray | None = None
        self.phase, self.t, self.streak, self.lost, self.why = "close", 0, 0, 0, ""
        self.last: dict = {}

    def start(self, q_meas: Sequence[float]) -> None:
        """닫기 시작 — 손 목표를 지금 실측에서 출발한다(엄지 회전은 이 값 그대로)."""
        q = np.asarray(q_meas, float).reshape(-1)
        if q.size != len(self.names) or not np.all(np.isfinite(q)):
            raise HandGraspError(f"손 실측은 유한한 {len(self.names)} 개: {q.size}")
        self.q_cmd = q.copy()
        self.q0 = q.copy()
        self.responded = np.zeros(len(self.names), bool)
        self.block.reset()
        self.phase, self.t, self.streak, self.lost, self.why = "close", 0, 0, 0, ""

    @property
    def finished(self) -> bool:
        return self.phase in ("done", "failed")

    def latched_fingers(self) -> int:
        return int(self.block.latch[self.block.fingers].sum())

    def step(self, q_meas: Sequence[float]) -> np.ndarray:
        if self.q_cmd is None:
            raise HandGraspError("start 전에 step")
        q = np.asarray(q_meas, float).reshape(-1)
        self.block.step(self.q_cmd, q)                   # 직전 목표 ↔ 지금 실측(rh_aglt 디코더와 같은 순서)
        self.responded |= np.abs(q - self.q0) >= RESPOND_RAD
        self.block.latch &= self.responded               # 아직 움직이지 않은 손가락의 멈춤은 응답 지연이지 막힘이 아니다
        self.block.blocked &= self.responded
        if self.phase in ("close", "hold"):
            self._ramp()
        self.t += 1
        self._advance()
        return self.q_cmd.copy()

    def _ramp(self) -> None:
        assert self.q_cmd is not None
        step = np.full(len(self.names), self.cfg.close_rad_s / self.hz)
        step[self.thumb_rot] = 0.0
        if self.phase == "close" and self.t < round(self.cfg.thumb_delay_s * self.hz):
            step[self.block.thumb_flex] = 0.0
        step[self.block.latch] = 0.0
        g, cur = self.q_grip, self.q_cmd
        nxt = cur + self.sgn * step
        # 쥔 자세를 넘지 않는다(이미 넘었으면 그 자리)
        self.q_cmd = np.where(self.sgn > 0, np.minimum(nxt, np.maximum(g, cur)), np.maximum(nxt, np.minimum(g, cur)))

    def _advance(self) -> None:
        cfg, n = self.cfg, self.latched_fingers()
        self.last = {"phase": self.phase, "t_s": round(self.t / self.hz, 2), "latched_fingers": n,
                     "latch": self.block.latch.astype(int).tolist(), "lead": [round(float(v), 4) for v in self.block.lead]}
        if self.phase == "close":
            self.streak = self.streak + 1 if self.block.grasped() else 0
            if self.streak >= cfg.judge_ticks:
                self.phase, self.t, self.why = "hold", 0, f"쥠 {n} 손가락"
            elif self.t >= round(cfg.close_max_s * self.hz):
                still = "" if self.responded[self.block.fingers].any() else " · 손가락이 하나도 움직이지 않았다(손 명령 경로 확인)"
                self.phase, self.why = "failed", (f"not grasped — {cfg.close_max_s:.1f} s 안에 래치 손가락 {n} < "
                                                  f"{cfg.judge_min_axes}(컵이 손에 없거나 닿지 않음){still}")
            return
        if self.phase == "hold":
            self.lost = self.lost + 1 if n < cfg.judge_min_axes else 0
            if self.lost > cfg.loss_ticks:
                self.phase, self.why = "failed", (f"not grasped — 유지 중 풀림(래치 손가락 {n} 이 {cfg.loss_ticks} 틱 넘게 "
                                                  f"{cfg.judge_min_axes} 미만)")
            elif self.t >= round(cfg.hold_s * self.hz):
                self.phase, self.why = "done", f"{self.why} · {cfg.hold_s:.1f} s 유지"

    def as_dict(self) -> dict:
        return {**self.last, "phase": self.phase, "why": self.why}

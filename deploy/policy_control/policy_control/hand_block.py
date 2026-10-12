"""10.12 손가락 막힘 신호 — 쥠 멈춤 · 쥔 판정(grasping 확정 정의, hdgp tasks/rh_aglt_r/hand_block.py 의 numpy 판, 한 손).

배경(10.12 bag 분석): sim 은 링크 컵 접촉력 > 1 N, 실기는 손끝 촉각 > 1 N 으로 판정해 감지 면이 달랐다. 막힘 신호는
sim 과 실기에서 같은 식으로 계산되고 g ↔ N 보정이 필요 없다(14:50 실기 검출 0.93~1.00 · 비접촉 오탐 ≤ 0.026).

정책 틱마다(손 목표를 정하기 전, 직전 q* 기준) — hdgp HandBlock.step 과 같은 순서:
  q      = 그 시각 실측 손 각도(관측 지연 없음), 칸 단위로 반올림(엄지굽힘 1/527 rad, 나머지 1/550 rad)
  ref    = 닫힘 방향 기준 min(q*(t−D), q*(t))  — D = lag_ticks(명령 → 움직임 지연보다 길게)
  lead   = (ref − q) · 닫힘방향,  moved = |q(t) − q(t−N)|
  blocked = 대상 축 AND lead > G AND moved < ε
  latch  : blocked 면 걸리고, q* 가 여는 쪽으로 움직이거나 lead < G/2 면 풀린다
  쥔 판정 = latch[엄지굽힘] AND latch[네 손가락 중 하나]
DR(G · ε · N 범위, 손가락 누락, 쥔 판정 지연)은 sim 전용 — 배포는 계약의 명목값 하나를 쓴다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: 계약 axes 이름 → RH56F1 손 관절 이름 끝(hdgp 프로필 hand_joint_names)
AXIS_SUFFIX = {"thumb_flex": "thumb_2", "index": "index_1", "middle": "middle_1", "ring": "ring_1", "pinky": "pinky_1"}
FINGERS = ("index", "middle", "ring", "pinky")
#: 레지스터 1 칸 [rad] — 손가락 약 550 칸/rad, 엄지굽힘 약 527 칸/rad(rh56f1_hand_map)
CELL_THUMB_FLEX, CELL_FINGER = 1.0 / 527.0, 1.0 / 550.0
KEYS = ("g_rad", "eps_rad", "n_ticks", "lag_ticks", "axes")
GRASP_RULES = ("thumb_flex_and_any_finger",)
#: 10.13 쥔 판정 유지 규칙(hdgp grasp_hold_rule) — "none" 은 계약에 빈 값으로 둔다
HOLD_RULES = ("block_follow_open",)
HOLD_AXES = ("thumb_flex", "index", "middle", "ring", "pinky")   # grasp_release_open_rad 순서


class HandBlockError(ValueError):
    pass


@dataclass(frozen=True)
class BlockCfg:
    g_rad: float
    eps_rad: float
    n_ticks: int
    lag_ticks: int
    axes: tuple

    @staticmethod
    def from_contract(raw: Mapping) -> "BlockCfg":
        if not isinstance(raw, Mapping) or set(raw) != set(KEYS):
            raise HandBlockError(f"hand_block 키는 {KEYS}: {sorted(raw) if isinstance(raw, Mapping) else raw}")
        cfg = BlockCfg(float(raw["g_rad"]), float(raw["eps_rad"]), int(raw["n_ticks"]), int(raw["lag_ticks"]),
                       tuple(str(a) for a in raw["axes"]))
        if not (cfg.g_rad > 0 and cfg.eps_rad > 0 and cfg.n_ticks >= 1 and cfg.lag_ticks >= 0):
            raise HandBlockError(f"hand_block: g_rad > 0 · eps_rad > 0 · n_ticks ≥ 1 · lag_ticks ≥ 0 — {raw}")
        bad = [a for a in cfg.axes if a not in AXIS_SUFFIX]
        if bad or len(set(cfg.axes)) != len(cfg.axes) or "thumb_flex" not in cfg.axes \
                or not any(f in cfg.axes for f in FINGERS):
            raise HandBlockError(f"hand_block.axes {list(cfg.axes)} — {list(AXIS_SUFFIX)} 중, 엄지굽힘 + 손가락 하나 이상")
        return cfg


class HandBlockNp:
    """한 손 막힘 상태. `step(q_star_prev, q)` 를 매 정책 틱 한 번(대기 hold 중에도) — 에피소드 시작에 reset()."""

    def __init__(self, cfg: BlockCfg, joint_names: Sequence[str], q_open: Sequence[float], q_grip: Sequence[float]) -> None:
        names = list(joint_names)
        idx = {}
        for a in cfg.axes:
            hit = [i for i, n in enumerate(names) if n.endswith(AXIS_SUFFIX[a])]
            if len(hit) != 1:
                raise HandBlockError(f"축 {a}: 손 관절 {names} 에서 *{AXIS_SUFFIX[a]} 가 하나가 아니다")
            idx[a] = hit[0]
        self.cfg = cfg
        self.h = len(names)
        self.axes = np.zeros(self.h, bool)
        self.axes[list(idx.values())] = True
        self.idx = idx
        self.thumb_flex = idx["thumb_flex"]
        self.fingers = sorted(idx[f] for f in FINGERS if f in idx)
        sgn = np.sign(np.asarray(q_grip, float) - np.asarray(q_open, float))
        self.sgn = np.where(sgn == 0, 1.0, sgn)
        self.cell = np.array([CELL_THUMB_FLEX if n.endswith(AXIS_SUFFIX["thumb_flex"]) else CELL_FINGER for n in names])
        self.reset()

    @staticmethod
    def from_contract(raw: Mapping, joint_names: Sequence[str], q_open, q_grip) -> "HandBlockNp":
        return HandBlockNp(BlockCfg.from_contract(raw), joint_names, q_open, q_grip)

    def reset(self) -> None:
        self._q: deque = deque(maxlen=self.cfg.n_ticks + 1)          # [-1] = 지금
        self._star: deque = deque(maxlen=self.cfg.lag_ticks + 1)
        self._prev_star: np.ndarray | None = None
        self.latch = np.zeros(self.h, bool)
        self.blocked = np.zeros(self.h, bool)
        self.lead = np.zeros(self.h)

    def quantize(self, q: np.ndarray) -> np.ndarray:
        return np.round(q / self.cell) * self.cell

    def step(self, q_star: Sequence[float], q: Sequence[float]) -> np.ndarray:
        """q_star (H) 직전 정책 손 목표, q (H) 실측 → 쥠 멈춤 (H) bool(대상 축만)."""
        s, qq = np.asarray(q_star, float).reshape(-1), self.quantize(np.asarray(q, float).reshape(-1))
        if s.size != self.h or qq.size != self.h or not (np.all(np.isfinite(s)) and np.all(np.isfinite(qq))):
            raise HandBlockError(f"손 q* · q 는 유한한 {self.h} 개: {s.size} · {qq.size}")
        if not self._q:                               # 첫 틱(리셋 뒤) — 기록을 지금 값으로 채운다(hdgp flush)
            self._q.extend([qq.copy()] * (self.cfg.n_ticks + 1))
            self._star.extend([s.copy()] * (self.cfg.lag_ticks + 1))
            self._prev_star = s.copy()
        self._q.append(qq.copy())
        self._star.append(s.copy())
        old = self._q[0]                              # N 틱 전
        ref = np.minimum(self._star[0] * self.sgn, s * self.sgn) * self.sgn   # 지금 더 열었으면 지금 값
        lead = (ref - qq) * self.sgn
        moved = np.abs(qq - old)
        self.blocked = self.axes & (lead > self.cfg.g_rad) & (moved < self.cfg.eps_rad)
        opening = ((s - self._prev_star) * self.sgn) < -1e-6
        release = opening | (lead < 0.5 * self.cfg.g_rad)
        self.latch = (self.latch | self.blocked) & ~release & self.axes
        self.lead = lead
        self._prev_star = s.copy()
        return self.latch.copy()

    def grasped(self) -> bool:
        """엄지굽힘 멈춤 AND 네 손가락 중 하나 멈춤."""
        return bool(self.latch[self.thumb_flex] and self.latch[self.fingers].any())

    def as_dict(self) -> dict:
        return {"latch": self.latch.astype(int).tolist(), "lead": [round(float(v), 4) for v in self.lead],
                "grasped": self.grasped()}


class GraspHoldNp:
    """10.13 쥔 판정 유지(grasp_hold_rule="block_follow_open") — hdgp hand_block.GraspHold 의 numpy 판(한 손).

    막힘 쥔 판정(raw)은 정책 q* 가 1 틱만 여는 쪽으로 흔들려도 풀려 쥔 채 들고 있는 동안 22~35 % 만 참이었다(grasping 10.13).
    hold 는 raw 가 한 번 참이면 켜지고, 실측 손가락이 마지막으로 쥔 자세 q_g 보다 축별 X 이상 열리고 · 막힘 래치 축이 없고 ·
    대상 축이 모두 목표를 따라갈 때(free)가 release_steps 틱 연속이면 꺼진다. q_g 는 raw 가 참인 틱에만 갱신한다.

    틱 순서(sim 과 같은 색인): 이번 틱 block.step 전에 step(raw = 직전 래치의 grasped(), q* = 직전 정책 손 목표, q = 지금 실측)
    — sim 은 tick t 끝에서 (tick t 래치, q*_t, 물리 뒤 q_{t+1}) 로 계산한다.
    """

    def __init__(self, block: HandBlockNp, open_rad: Sequence[float], release_steps: int) -> None:
        x = [float(v) for v in open_rad]
        if len(x) != len(HOLD_AXES) or min(x) < 0 or not all(np.isfinite(x)):
            raise HandBlockError(f"grasp_release_open_rad 는 {HOLD_AXES} 순 {len(HOLD_AXES)} 개(≥ 0): {list(open_rad)}")
        if int(release_steps) < 1:
            raise HandBlockError(f"grasp_release_steps ≥ 1: {release_steps}")
        self.b = block
        self.x = np.zeros(block.h)
        for a, v in zip(HOLD_AXES, x):
            if a in block.idx:
                self.x[block.idx[a]] = v
        self.steps = int(release_steps)
        self.reset()

    def reset(self) -> None:
        self.hold = False
        self.cnt = 0
        self.q_g = np.zeros(self.b.h)

    def step(self, raw: bool, q_star: Sequence[float], q: Sequence[float]) -> bool:
        b = self.b
        qq = b.quantize(np.asarray(q, float).reshape(-1))
        s = np.asarray(q_star, float).reshape(-1)
        if raw:
            self.q_g = qq.copy()
        self.hold = self.hold or bool(raw)
        open_amt = (self.q_g - qq) * b.sgn
        follow = ((s - qq) * b.sgn) < b.cfg.g_rad
        no_latch = not (b.latch & b.axes).any()
        ok_f = bool((follow | ~b.axes).all())
        ok_o = bool(((open_amt >= self.x) | ~b.axes).all())
        free = self.hold and not raw and no_latch and ok_f and ok_o
        self.cnt = self.cnt + 1 if free else 0
        if self.cnt >= self.steps:
            self.hold, self.cnt = False, 0
        return self.hold

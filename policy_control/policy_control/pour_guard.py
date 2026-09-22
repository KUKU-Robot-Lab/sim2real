"""pour 실기 안전망 — 정책과 무관하게 항상 켜 두는 두 가지 감시.

정책이 어떤 체크포인트든 sim 에서 보이지 않던 두 위험이 실기에 남는다.

1. **과기울임.** 소스 컵이 수평(90도)을 지나면 손바닥 법선의 중력 성분 부호가 뒤집힌다. sim 은
   접촉 동결 시너지로 버티지만 실기 DG-5F 는 백래시가 있어 컵을 놓칠 수 있고, 그 자세는 `*_aj_6/7`
   을 한계 부근에 오래 두어 j7 발열(임계 70 °C)로 간다. 정책은 기울기를 직접 지령하지 않으므로
   계약값만으로는 못 막는다.
2. **중앙선 교차.** 배포 fabric 세계의 장애물은 테이블 박스 하나뿐이고(`pour_fabric.py`), 실기에서는
   팔마다 fabric 이 따로 돈다 — **서로의 존재를 모른다.** sim 은 두 팔이 한 articulation 이라
   자가충돌이 처리되지만 실기는 아니다. i18 은 64/64 env 에서 소스 컵이 수신 쪽 반면으로 넘어갔다.

코어는 ROS 를 모른다. 판정은 전부 여기서 하고, 노드 껍질은 구독과 `episode/abort` 호출만 한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

TILT_ABORT_DEG = 125.0        # ckpt_gate 의 게이트(120)보다 살짝 위 — 게이트는 평가, 이것은 중단선이다
CROSS_MARGIN_M = 0.02         # 중앙선을 이만큼 넘어야 중단. 포즈 추정 떨림으로 끊기지 않게
PALM_MIN_DIST_M = 0.12        # 두 손바닥이 이보다 가까우면 중단 (fabric 이 서로를 모른다)
STALE_S = 0.5                 # 컵 포즈가 이만큼 끊기면 판정할 수 없다


@dataclass(frozen=True)
class GuardLimits:
    tilt_abort_deg: float = TILT_ABORT_DEG
    cross_margin_m: float = CROSS_MARGIN_M
    palm_min_dist_m: float = PALM_MIN_DIST_M
    stale_s: float = STALE_S


@dataclass(frozen=True)
class GuardInput:
    """모두 base_link 기준. 없는 값은 None 으로 두고, 그 항목은 판정하지 않는다."""
    t: float
    src_cup_pos: Sequence[float] | None = None
    src_cup_up: Sequence[float] | None = None
    rcv_cup_pos: Sequence[float] | None = None
    src_palm: Sequence[float] | None = None
    rcv_palm: Sequence[float] | None = None
    src_cup_t: float | None = None
    rcv_cup_t: float | None = None


def tilt_deg(up: Sequence[float]) -> float:
    """컵 위쪽 축과 월드 z 의 각도. 90 을 넘으면 수평을 지난 것이다."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(tuple(float(x) for x in a), tuple(float(x) for x in b))


class GuardCore:
    """중단 사유를 만든다. 스스로 아무것도 발행하지 않고, 한 번 걸리면 리셋 전까지 걸린 채로 있다."""

    def __init__(self, limits: GuardLimits | None = None) -> None:
        self.limits = limits or GuardLimits()
        self._latched: list[str] = []

    @property
    def latched(self) -> list[str]:
        return list(self._latched)

    def reset(self) -> None:
        self._latched = []

    def check(self, g: GuardInput) -> list[str]:
        """이번 tick 의 사유. 한 번이라도 걸리면 `latched` 에 남는다."""
        lim, out = self.limits, []

        if g.src_cup_up is not None:
            t = tilt_deg(g.src_cup_up)
            if t > lim.tilt_abort_deg:
                out.append(f"source cup tilt {t:.1f} deg > {lim.tilt_abort_deg:.0f}")

        if g.src_cup_pos is not None and g.rcv_cup_pos is not None:
            # 중앙선은 수신 컵이 있는 쪽 부호로 정한다. 좌우 배치가 바뀌어도 같은 규칙이 선다.
            sign = 1.0 if float(g.rcv_cup_pos[1]) >= 0 else -1.0
            over = float(g.src_cup_pos[1]) * sign
            if over > lim.cross_margin_m:
                out.append(f"source cup crossed the centre line by {over * 100:.1f} cm")

        if g.src_palm is not None and g.rcv_palm is not None:
            d = _dist(g.src_palm, g.rcv_palm)
            if d < lim.palm_min_dist_m:
                out.append(f"palms {d * 100:.1f} cm apart < {lim.palm_min_dist_m * 100:.0f}")

        for name, ts in (("src", g.src_cup_t), ("rcv", g.rcv_cup_t)):
            if ts is not None and (g.t - ts) > lim.stale_s:
                out.append(f"{name} cup pose stale {(g.t - ts):.2f} s")

        for r in out:
            if r not in self._latched:
                self._latched.append(r)
        return out


def limits_from_contract(contract) -> GuardLimits:
    """계약이 값을 가지고 있으면 그것을 쓰고, 없으면 모듈 기본값. 계약은 진실원천이지 필수는 아니다."""
    def get(name, default):
        v = getattr(contract, name, None)
        return default if v is None else float(v)

    return GuardLimits(tilt_abort_deg=get("src_tilt_abort_deg", TILT_ABORT_DEG),
                       cross_margin_m=get("cross_margin_m", CROSS_MARGIN_M),
                       palm_min_dist_m=get("palm_min_dist_m", PALM_MIN_DIST_M),
                       stale_s=get("guard_stale_s", STALE_S))

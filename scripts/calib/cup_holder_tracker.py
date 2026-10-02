"""컵홀더 자세 연속 추정의 순수부(ROS 무관) — scripts/nodes/cup_holder_pose_node.py 가 감싼다.

한 장마다 cup_holder_pose.estimate 를 부르되
  · 직전 장의 무늬 맞춤 결과로 추적한다(0.2 s). 놓치면 그 id 만 전체 탐색(~6 s)으로 되돌아간다.
  · 홀더별 최근 window 장의 중앙값을 내고, 그 폭이 stable 기준 안이면 stable 로 본다.
  · 한 장이 실패(공유 제약 rms 초과 · 마커 없음)하면 그 장은 버린다. 연속 lost_after 장 못 보면 그 홀더를 지운다.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cup_holder_pose import SHARED_FAIL_PX, HolderCfg, estimate, wrap

STABLE_XY_M = 0.002
STABLE_YAW_RAD = math.radians(1.5)


@dataclass
class HolderState:
    name: str
    marker_id: int
    hist: deque = field(default_factory=lambda: deque(maxlen=5))
    misses: int = 0
    src: str = ""
    rms: float = float("nan")
    ncc: float = float("nan")

    def pose(self) -> np.ndarray | None:
        if not self.hist:
            return None
        P = np.array(self.hist)
        yaw = math.atan2(np.sin(P[:, 3]).mean(), np.cos(P[:, 3]).mean())
        out = np.median(P, axis=0)
        out[3] = yaw
        return out

    def spread(self) -> tuple[float, float]:
        if len(self.hist) < 2:
            return float("inf"), float("inf")
        P = np.array(self.hist)
        ref = np.median(P, axis=0)
        ref[3] = math.atan2(np.sin(P[:, 3]).mean(), np.cos(P[:, 3]).mean())
        xy = float(np.max(np.linalg.norm(P[:, :2] - ref[:2], axis=1)))
        yaw = float(np.max(np.abs([wrap(a - ref[3]) for a in P[:, 3]])))
        return xy, yaw

    def stable(self) -> bool:
        if len(self.hist) < (self.hist.maxlen or 1):
            return False
        xy, yaw = self.spread()
        return xy <= STABLE_XY_M and yaw <= STABLE_YAW_RAD


class HolderTracker:
    def __init__(self, cfg: HolderCfg, T_base_cam: np.ndarray, *, table_z: float, window: int = 5,
                 lost_after: int = 5, use_template: bool = True):
        self.cfg = cfg
        self.T_bc = T_base_cam
        self.table_z = float(table_z)
        self.use_template = use_template
        self.lost_after = int(lost_after)
        self.states = {i: HolderState(n, i, deque(maxlen=window)) for n, i in zip(cfg.names, cfg.ids)}
        self._hits: dict = {}
        self.frames = 0
        self.last_error = ""
        self.last_ms = 0.0

    def update(self, gray: np.ndarray, K: np.ndarray) -> None:
        t0 = time.monotonic()
        self.frames += 1
        est = estimate(gray, K, self.T_bc, self.cfg, table_z=self.table_z, use_template=self.use_template,
                       prev_hits=self._hits)
        self.last_ms = (time.monotonic() - t0) * 1e3
        # 다음 장 추적용: 이번에 무늬로 찾은 것 + (ArUco 로 찾았으면 무늬 결과가 없으니 직전 것을 버린다)
        self._hits = dict(est.hits)
        bad = est.worst_rms > SHARED_FAIL_PX
        self.last_error = (f"공유 제약 rms {est.worst_rms:.1f} px > {SHARED_FAIL_PX} — 이 장 버림" if bad
                           else "" if est.poses else "마커 없음")
        for i, st in self.states.items():
            if bad or i not in est.poses:
                st.misses += 1
                if st.misses >= self.lost_after:
                    st.hist.clear()
                    st.src = ""
                continue
            st.misses = 0
            st.hist.append(np.asarray(est.poses[i], float).copy())
            st.src = est.src[i]
            st.rms = est.rms[i]
            st.ncc = float(est.hits[i].ncc) if i in est.hits else float("nan")

    def poses(self) -> dict[str, np.ndarray]:
        return {st.name: p for st in self.states.values() if (p := st.pose()) is not None}

    def all_stable(self) -> bool:
        return all(st.stable() for st in self.states.values())

    def status(self) -> dict:
        hs = {}
        for st in self.states.values():
            p = st.pose()
            xy, yaw = st.spread()
            hs[st.name] = {
                "marker_id": st.marker_id, "seen": p is not None, "stable": st.stable(), "src": st.src,
                "misses": st.misses, "n": len(st.hist),
                "pose": None if p is None else [round(float(v), 4) for v in p[:3]] + [round(math.degrees(p[3]), 2)],
                "spread_mm": None if not math.isfinite(xy) else round(xy * 1e3, 2),
                "spread_deg": None if not math.isfinite(yaw) else round(math.degrees(yaw), 2),
                "rms_px": None if not math.isfinite(st.rms) else round(st.rms, 3),
                "ncc": None if not math.isfinite(st.ncc) else round(st.ncc, 3),
            }
        return {"ok": self.all_stable(), "frames": self.frames, "ms": round(self.last_ms, 1),
                "error": self.last_error, "shared": list(self.cfg.shared), "holders": hs}

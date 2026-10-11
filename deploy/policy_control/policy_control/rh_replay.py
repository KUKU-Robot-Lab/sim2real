"""방안 2 재생 계획 — sim 성공 궤적(s2r_replay_v1, 팔 7 + 손 6)을 pd 입력으로 바꾸는 순수 코어(numpy, ROS 무의존).

hdgp `scripts/tools/replay_chain/stitch.py` 가 만든 npz 를 읽는다. 무엇을 언제 보낼지는 여기서만 정하고, 노드
(tools/replay_traj_rh.py)는 발행만 한다 — 그래야 로봇 없이 시험으로 잠근다.

여기서 지키는 것:
  · **열 선택** — 팔은 sim 이 실제로 간 자리(arm_q, 측정), 손은 정책 지령(hand_q_cmd). 손을 측정값으로 보내면
    지령과 실측의 차이(= 쥐는 힘)가 0 이 되어 컵이 미끄러진다(sim 쥔 상태: 검지 지령 0.86 · 측정 0.60).
  · **고정 주기 보간** — rate_scale 이 작아도 pd 워치독(0.25 s)보다 촘촘히 낸다. 프레임 사이는 선형.
  · **손 입력 모드** — 구간 경계에서 hand_command(admittance | position) 사건을 낸다(pd on_episode 가 받는다).
  · 거부: 이름·측·모양·비유한 값, 계약 한계 밖, rate_scale 적용 뒤 속도가 단계 한계 넘음.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

SCHEMA = "s2r_replay_v1"
ARM_SOURCES = ("arm_q", "arm_q_cmd", "arm_q_pd")
HAND_SOURCES = ("hand_q_cmd", "hand_q", "hand_q_pd")
HAND_AXES = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
HAND_COMMANDS = ("admittance", "position")
#: robotctl·replay_to_pd 와 같은 진입 램프 속도(사람이 반응할 수 있는 속도).
PARK_SPEED_RAD_PER_SEC = 0.1


class ReplayError(ValueError):
    pass


@dataclass(frozen=True)
class ReplayTraj:
    side: str
    names: tuple[str, ...]               # 팔 7 + 손 6 (canonical)
    q: np.ndarray                        # (T, 13)
    step_dt: float
    segments: tuple[dict, ...]
    arm_src: str
    hand_src: str
    cup: np.ndarray | None = None
    source: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.q.shape[0])


def expected_names(side: str) -> tuple[str, ...]:
    p = side[0]
    return tuple(f"{p}_aj_{i}" for i in range(1, 8)) + tuple(f"{p}_hj_{a}" for a in HAND_AXES)


def load_replay(path: str | Path, *, arm_src: str = "arm_q", hand_src: str = "hand_q_cmd") -> ReplayTraj:
    if arm_src not in ARM_SOURCES or hand_src not in HAND_SOURCES:
        raise ReplayError(f"열 선택 {arm_src}/{hand_src} — 팔 {ARM_SOURCES} · 손 {HAND_SOURCES}")
    z = np.load(path, allow_pickle=False)
    schema = str(z["meta_schema"]) if "meta_schema" in z.files else ""
    if schema != SCHEMA:
        raise ReplayError(f"{path}: schema {schema!r} ≠ {SCHEMA}")
    side = str(z["meta_side"])
    if side not in ("left", "right"):
        raise ReplayError(f"side {side!r}")
    names = tuple(str(n) for n in z["meta_arm_names"]) + tuple(str(n) for n in z["meta_hand_names"])
    want = expected_names(side)
    # 손 이름은 sim profile 순서 그대로여야 열이 맞는다 — 순서까지 같아야 한다
    if names != want:
        raise ReplayError(f"관절 이름이 {side} canonical 순서와 다르다\n 파일 {names}\n 기대 {want}")
    arm, hand = np.asarray(z[arm_src], float), np.asarray(z[hand_src], float)
    if arm.ndim != 2 or arm.shape[1] != 7 or hand.shape != (arm.shape[0], 6) or arm.shape[0] < 2:
        raise ReplayError(f"모양: 팔 {arm.shape} · 손 {hand.shape}")
    q = np.hstack([arm, hand])
    if not np.isfinite(q).all():
        bad = int(np.argwhere(~np.isfinite(q))[0][0])
        raise ReplayError(f"프레임 {bad} 비유한 값 — 보간하지 않고 거부")
    segs = tuple(json.loads(str(z["meta_segments"])))
    for s in segs:
        if s.get("hand_command") not in HAND_COMMANDS:
            raise ReplayError(f"구간 {s.get('label')}: hand_command {s.get('hand_command')!r}")
    if not segs or segs[0]["start"] != 0 or segs[-1]["end"] != q.shape[0] - 1:
        raise ReplayError("구간 목록이 프레임 전체를 덮지 않는다")
    for a, b in zip(segs, segs[1:]):
        if b["start"] != a["end"] + 1:
            raise ReplayError(f"구간 {a['label']}→{b['label']} 사이가 비거나 겹친다")
    step_dt = float(z["meta_step_dt"])
    if not 0.0 < step_dt < 1.0:
        raise ReplayError(f"step_dt {step_dt}")
    cup = np.asarray(z["cup"], float) if "cup" in z.files else None
    return ReplayTraj(side=side, names=names, q=q, step_dt=step_dt, segments=segs, arm_src=arm_src,
                      hand_src=hand_src, cup=cup, source={"path": str(path)})


def limit_violations(q: np.ndarray, lo: np.ndarray, hi: np.ndarray, names: tuple[str, ...],
                     margin: float = 0.0) -> list[str]:
    """(T, J) 가 [lo + margin, hi − margin] 밖인 관절 — 처음 프레임과 초과량."""
    out = []
    for j, n in enumerate(names):
        under = np.flatnonzero(q[:, j] < lo[j] + margin)
        over = np.flatnonzero(q[:, j] > hi[j] - margin)
        for idx, side, amt in ((under, "아래", lo[j] + margin - q[:, j].min()), (over, "위", q[:, j].max() - hi[j] + margin)):
            if len(idx):
                out.append(f"{n}: 프레임 {int(idx[0])} 부터 한계 {side} {amt:.4f} rad")
    return out


@dataclass(frozen=True)
class Plan:
    q: np.ndarray            # (K, 13) 발행 목표(진입 램프 포함)
    qd: np.ndarray           # (K, 13) 속도 전향(램프 구간 0)
    pub_dt: float
    n_ramp: int
    mode_at: tuple[tuple[int, str], ...]   # (발행 인덱스, hand_command) — 그 행을 보내기 **전에** 사건
    frame_of: np.ndarray     # (K,) 원본 프레임 위치(실수, 램프 −1)

    @property
    def seconds(self) -> float:
        return self.q.shape[0] * self.pub_dt


def approach_ramp(start: np.ndarray, first: np.ndarray, *, speed, dt: float) -> np.ndarray:
    """실측 → 첫 프레임 직선. speed 는 스칼라 또는 관절별(팔 0.1 · 손은 더 빨라도 된다 — 차렷 손 = 주먹 1.45 rad)."""
    start, first = np.asarray(start, float).reshape(-1), np.asarray(first, float).reshape(-1)
    if start.shape != first.shape:
        raise ReplayError(f"램프 길이 {start.shape} / {first.shape}")
    sp = np.broadcast_to(np.asarray(speed, float), start.shape)
    if np.any(sp <= 0.0) or dt <= 0.0:
        raise ReplayError("램프 속도·dt 는 양수")
    steps = int(np.ceil(float(np.max(np.abs(first - start) / sp)) / dt))
    if steps <= 0:
        return start.reshape(1, -1).copy()
    w = np.linspace(0.0, 1.0, steps + 1).reshape(-1, 1)
    return start + w * (first - start)


def resample(q: np.ndarray, step_dt: float, rate_scale: float, pub_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """경로는 그대로, 시간만 1/rate_scale 배 — pub_dt 격자에서 선형 보간. (값, 원본 프레임 위치)."""
    if not 0.0 < rate_scale <= 1.0:
        raise ReplayError(f"rate_scale 은 (0, 1] — {rate_scale}")
    if pub_dt <= 0.0:
        raise ReplayError("pub_dt 양수")
    n = q.shape[0]
    dur = (n - 1) * step_dt / rate_scale
    k = int(np.floor(dur / pub_dt + 1e-9)) + 1
    pos = np.minimum(np.arange(k) * pub_dt * rate_scale / step_dt, n - 1)
    if pos[-1] < n - 1:                                   # 끝 프레임은 반드시 정확히 보낸다
        pos = np.append(pos, n - 1.0)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (pos - i0).reshape(-1, 1)
    return q[i0] * (1.0 - w) + q[i1] * w, pos


def build_plan(traj: ReplayTraj, start: np.ndarray, *, rate_scale: float, pub_hz: float = 60.0,
               hand_ramp_speed: float = PARK_SPEED_RAD_PER_SEC) -> Plan:
    pub_dt = 1.0 / float(pub_hz)
    body, pos = resample(traj.q, traj.step_dt, rate_scale, pub_dt)
    speed = np.array([PARK_SPEED_RAD_PER_SEC if "_aj_" in n else hand_ramp_speed for n in traj.names])
    ramp = approach_ramp(start, body[0], speed=speed, dt=pub_dt)
    q = np.vstack([ramp, body])
    qd = np.vstack([np.zeros((1, q.shape[1])), np.diff(q, axis=0) / pub_dt])
    qd[: len(ramp)] = 0.0
    frame_of = np.concatenate([np.full(len(ramp), -1.0), pos])
    modes: list[tuple[int, str]] = [(0, traj.segments[0]["hand_command"])]   # 램프 전에 첫 구간 모드
    for s in traj.segments[1:]:
        if s["hand_command"] == modes[-1][1]:
            continue
        k = len(ramp) + int(np.searchsorted(pos, s["start"] - 1e-9))
        modes.append((k, s["hand_command"]))
    return Plan(q=q, qd=qd, pub_dt=pub_dt, n_ramp=len(ramp), mode_at=tuple(modes), frame_of=frame_of)


def speed_violations(plan: Plan, names: tuple[str, ...], *, arm_vmax: float, hand_vmax: float) -> list[str]:
    """램프 뒤 본 구간의 관절 속도가 단계 한계를 넘는 것."""
    body = plan.q[plan.n_ramp:]
    if body.shape[0] < 2:
        return []
    v = np.abs(np.diff(body, axis=0)).max(axis=0) / plan.pub_dt
    out = []
    for j, n in enumerate(names):
        vmax = arm_vmax if "_aj_" in n else hand_vmax
        if v[j] > vmax:
            out.append(f"{n}: {v[j]:.3f} rad/s > {vmax}")
    return out


def max_rate_scale(traj: ReplayTraj, *, arm_vmax: float, hand_vmax: float) -> float:
    """속도 한계를 지키는 가장 큰 rate_scale(≤ 1)."""
    v = np.abs(np.diff(traj.q, axis=0)).max(axis=0) / traj.step_dt
    lim = np.array([arm_vmax if "_aj_" in n else hand_vmax for n in traj.names])
    with np.errstate(divide="ignore"):
        r = np.where(v > 0, lim / v, np.inf)
    return float(min(1.0, r.min()))

"""place 직접 교시 — pd 를 engage 한 채, 사람이 팔을 손으로 끌어 홀더까지 옮기는 궤적과 손 쥐기·펴기를 녹화한다(순수 코어, ROS 무의존).

11.04 사용자: "틸트 시작자세 → 오른손으로 컵을 쥠 → 팔을 내가 움직여 홀더에 맞춤 → 손가락을 펴 홀더에 떨어뜨림.
홀더 1·2 는 오른손, 홀더 0 은 왼손, 각각 하나씩 세 개." 콘솔·pd 를 오가지 않게 pd 를 붙든 그대로 한다.

방법: 매 틱 /policy_control/joint_target 으로 팔 7 + 손 6 목표를 낸다(노드 tools/teach_place.py). 팔 목표는
robot_control.teaching 상태기계 — 멈춰 있으면 목표 고정, 밀면(처짐 기준선 넘는 편차) 목표 = 실측 + 기준선으로 따라가고,
멈추면 다시 고정한다. pd 는 그 목표를 kp·kd + 중력 ff(model_tau_ff)로 따른다. 손 목표는 키로만 바뀐다(g 쥠 · o 폄).

여기서 지키는 것:
  · 시작 목표 = pd 세트포인트(status q_set) — 실측으로 다시 잡으면 처짐만큼 팔이 움직인다(hand_grasp.py 와 같다).
  · 팔 목표는 관절 한계 안(LIMIT_MARGIN_RAD)으로 자르고, 틱당 max_vel·dt 로 제한한다. 기본 max_vel = 0.2 rad/s =
    pd 'reduced' 0.25 의 0.8 — replay_traj_rh.py 가 rate_scale 1.0 으로 받는 상한이라 녹화가 그대로 재생된다.
  · 녹화 npz = s2r_replay_v1(replay_traj_rh.py 가 읽는 팔 arm_q + 손 hand_q_cmd + 구간) + arm_target · meta_step_dt
    (replay_to_pd.py) + 손 사건(hand_event_*). 앞뒤 정지 구간은 잘라 내고, 쥔 손으로 PRE_ROLL_S, 편 뒤 POST_ROLL_S 를 남긴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import _paths  # noqa: F401  (robot_control on sys.path)
from robot_control.teaching import Mode, TeachLimits, acknowledge, start, step, toggle_lock  # noqa: E402

SCHEMA = "s2r_replay_v1"
HAND_AXES = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
#: rh_aglt_contract.json hand_open · hand_grip(thumb_1, thumb_2, index, middle, ring, pinky).
OPEN_RAD = (1.57, 0.0, 0.0, 0.0, 0.0, 0.0)
GRIP_RAD = (1.2, 0.24, 1.08, 1.08, 0.85, 0.85)
GRIP, OPEN = "grip", "open"
POLICY_HZ = 60.0
MAX_VEL_RAD_S = 0.2
LIMIT_MARGIN_RAD = 0.02
TRIM_RAD = 0.005
PRE_ROLL_S = 0.5
POST_ROLL_S = 1.0
#: 녹화의 손 목표는 키 순간 계단이 아니라 이 속도로 옮긴다 — replay_traj_rh 가 손 속도 0.8 × 2.1 rad/s 를 넘으면 거부하고,
#: pd 가 스스로 손을 옮길 때(hand_home · hand_release)도 1.0 rad/s 다(pd_rh56f1.yaml hand.max_vel).
HAND_VEL_RAD_S = 1.0
#: robot_control teach 기본값(어깨 kp 70 에서 약 1.4 N·m) — pd 도 같은 OpenArm 게인이다.
PUSH_RAD = 0.02
STILL_RAD_S = 0.03
STILL_SEC = 0.5
SETTLE_SEC = 1.0
MAX_FOLLOW_SEC = 30.0
#: 손목 회전(j7) — kp 가 가장 낮아 팔을 끌면 손 무게·관성만으로 따라 돌았다(11.04 holder1, 놓은 뒤 1.0→1.51→−0.31 rad).
#: 그래서 j7 은 팔 따라가기에 끌려가지 않고, 손을 직접 잡고 이만큼 넘게 비틀 때만 따라간다.
WRIST = 6
WRIST_PUSH_RAD = 0.1


class TeachError(ValueError):
    pass


def arm_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side[0]}_aj_{i}" for i in range(1, 8))


def hand_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side[0]}_hj_{a}" for a in HAND_AXES)


def parse_pose(text: str | None, default: Sequence[float], flag: str) -> np.ndarray:
    if text is None:
        return np.asarray(default, float)
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        raise TeachError(f"{flag}: {text!r} 는 쉼표로 나눈 숫자 6 개가 아니다") from None
    if len(values) != 6 or not np.all(np.isfinite(values)):
        raise TeachError(f"{flag} 는 {', '.join(HAND_AXES)} 순서 6 개 — {len(values)} 개")
    return np.asarray(values)


def default_limits() -> TeachLimits:
    push = np.full(7, PUSH_RAD)
    push[WRIST] = WRIST_PUSH_RAD
    return TeachLimits(push_rad=push, still_rad_s=np.full(7, STILL_RAD_S), still_sec=STILL_SEC,
                       settle_sec=SETTLE_SEC, max_follow_sec=MAX_FOLLOW_SEC)


@dataclass(frozen=True)
class HandEvent:
    name: str
    t: float              # 틱 시각(초, 교시 시작 기준)
    q: tuple[float, ...]  # 손 6 목표(rad)


@dataclass
class Teacher:
    """한 팔의 교시. start → (매 틱) tick · (키) key → recording."""

    side: str
    lower: np.ndarray
    upper: np.ndarray
    grip: np.ndarray
    open: np.ndarray
    dt: float = 1.0 / POLICY_HZ
    max_vel: float = MAX_VEL_RAD_S
    limits: TeachLimits = field(default_factory=default_limits)
    state: Any = None
    command: np.ndarray | None = None
    hand: np.ndarray | None = None
    events: list[HandEvent] = field(default_factory=list)
    rows: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, str]] = field(default_factory=list)  # t, 실측, 팔 목표, 손 목표, 모드
    #: g · o 가 건 잠금(기준선을 다시 잡으면 푼다). 사용자가 space 로 건 잠금은 건드리지 않는다.
    hand_lock: bool = False
    #: 따라가기 동안 손목(j7)이 붙드는 자리 · 이번 따라가기에서 손목을 직접 비틀었는지.
    wrist: float | None = None
    wrist_follow: bool = False

    def __post_init__(self) -> None:
        if self.side not in ("right", "left"):
            raise TeachError(f"side {self.side!r}")
        self.lower, self.upper = np.asarray(self.lower, float), np.asarray(self.upper, float)
        if self.lower.shape != (7,) or self.upper.shape != (7,) or np.any(self.upper - self.lower <= 2 * LIMIT_MARGIN_RAD):
            raise TeachError("팔 관절 한계는 7 개, 아래 < 위")
        if not 0 < self.max_vel <= 2.0 or not 0 < self.dt < 1:
            raise TeachError(f"max_vel {self.max_vel} · dt {self.dt}")

    def start(self, q_set: Sequence[float], q_meas: Sequence[float], q_hand: Sequence[float], t: float) -> None:
        """시작 목표 = pd 세트포인트, 손은 지금 실측 자리(pd 가 옛 손 목표를 다시 내지 않게 늘 6 개를 싣는다)."""
        sp, meas = np.asarray(q_set, float), np.asarray(q_meas, float)
        self.state = replace(start(meas, t, self.limits.settle_sec), target=sp.copy())
        self.command = self._clip(sp)
        self.hand = np.asarray(q_hand, float).copy()

    def _clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower + LIMIT_MARGIN_RAD, self.upper - LIMIT_MARGIN_RAD)

    def tick(self, q_meas: Sequence[float], t: float) -> tuple[np.ndarray, np.ndarray, str | None]:
        """이번 틱의 (팔 목표 7, 손 목표 6, 사건). 녹화 행을 남긴다."""
        if self.state is None or self.command is None or self.hand is None:
            raise TeachError("start 먼저")
        meas = np.asarray(q_meas, float)
        self.state, desired, event = step(self.state, self.limits, meas, t)
        prev = self.command
        desired = np.asarray(desired, float).copy()
        if self.state.mode is Mode.FOLLOW:
            # 팔을 끄는 동안 손목은 직전 명령 자리를 붙든다 — 손목을 직접 비틀어 기준선 너머 WRIST_PUSH_RAD 를 넘기면 그때부터 따라간다.
            if self.wrist is None:
                self.wrist = float(prev[WRIST])
            b = float(self.state.baseline[WRIST])
            if self.wrist_follow or abs((self.wrist - meas[WRIST]) - b) > self.limits.push_rad[WRIST]:
                self.wrist_follow, self.wrist = True, float(meas[WRIST]) + b
            desired[WRIST] = self.wrist
        else:
            self.wrist, self.wrist_follow = None, False
        cmd = self._clip(prev + np.clip(np.asarray(desired, float) - prev, -self.max_vel * self.dt, self.max_vel * self.dt))
        self.state = acknowledge(self.state, cmd)
        self.command = cmd
        if self.hand_lock and self.state.rebaseline_at is None:
            self.state, self.hand_lock = replace(self.state, locked=False), False
            event = "기준선 다시 잡음(손 · 컵 무게 포함) — 잠금 풀림, 이제 밀면 따라간다"
        self.rows.append((t, meas.copy(), cmd.copy(), self.hand.copy(), self.state.mode.value))
        return cmd, self.hand.copy(), event

    def key(self, k: str, t: float) -> str | None:
        if self.state is None:
            return None
        if k == " ":
            self.hand_lock = False
            self.state = toggle_lock(self.state)
            return "잠금: 밀어도 따라가지 않는다" if self.state.locked else "잠금 풀림"
        if k in ("g", "o"):
            name, q = (GRIP, self.grip) if k == "g" else (OPEN, self.open)
            q = np.asarray(q, float)
            # 손이 컵을 쥐거나 놓으면 팔 처짐이 바뀐다(손목 kp 가 낮아 컵 무게만으로 push_rad 를 넘는다). 기준선을 그대로 두면
            # 그 처짐을 미는 손으로 읽고, 따라가기 목표(실측 + 옛 기준선)가 다시 처지며 팔이 흘러내린다(11.04 실기 j7 −0.73 rad).
            # 그래서 지금 목표에서 팔을 잠그고, 손이 다 움직이고 settle_sec 뒤 기준선을 다시 잡은 다음 푼다(tick).
            settle = float(np.abs(q - self.hand).max()) / HAND_VEL_RAD_S + self.limits.settle_sec
            user_lock = bool(self.state.locked) and not self.hand_lock
            self.state = replace(self.state, mode=Mode.HOLD, target=np.asarray(self.command, float).copy(), locked=True,
                                 rebaseline_at=float(t) + settle, follow_since=None, still_since=None, moving_since=None)
            self.hand_lock = not user_lock
            self.hand = q.copy()
            self.events.append(HandEvent(name, float(t), tuple(float(v) for v in q)))
            hold = f"{settle:.1f} s 잠그고 기준선을 다시 잡는다" if self.hand_lock else "space 잠금 유지 · 기준선만 다시 잡는다"
            return f"손 {name}: {' '.join(f'{v:.2f}' for v in q)} — 팔 {hold}"
        return None


def _motion_span(t: np.ndarray, q: np.ndarray, trim: float) -> tuple[float, float] | None:
    """팔이 처음 자리에서 trim 넘게 벗어난 첫 시각 · 마지막 자리에서 trim 넘게 떨어져 있던 마지막 시각."""
    away0 = np.flatnonzero(np.abs(q - q[0]).max(axis=1) > trim)
    away1 = np.flatnonzero(np.abs(q - q[-1]).max(axis=1) > trim)
    if not len(away0) or not len(away1):
        return None
    return float(t[max(away0[0] - 1, 0)]), float(t[min(away1[-1] + 1, len(t) - 1)])


def _rate_limited(q: np.ndarray, step: float) -> np.ndarray:
    out = q.copy()
    for k in range(1, len(out)):
        out[k] = out[k - 1] + np.clip(q[k] - out[k - 1], -step, step)
    return out


def build_recording(teacher: Teacher, step_dt: float = 1.0 / POLICY_HZ, trim_rad: float = TRIM_RAD,
                    pre_roll: float = PRE_ROLL_S, post_roll: float = POST_ROLL_S) -> dict[str, np.ndarray]:
    """npz 필드. 팔 = 실측(사람이 실제로 옮긴 자리), 손 = 목표(쥔 힘 = 목표 − 실측, rh_replay 와 같은 선택)."""
    if len(teacher.rows) < 2:
        raise TeachError("녹화가 비었다")
    t = np.array([r[0] for r in teacher.rows])
    q = np.vstack([r[1] for r in teacher.rows])
    hand = np.vstack([r[3] for r in teacher.rows])
    # 녹화는 손이 컵을 다 쥔 뒤부터 — 쥐기 전 프레임이 들어가면 재생이 첫 프레임으로 가며 쥔 손을 편다.
    # 쥘 때 컵 무게로 손목이 처지는 것도 '움직임'이라 시작점을 g 앞으로 끌어당길 수 있어서, 움직임도 그 뒤에서만 찾는다.
    grips = [e for e in teacher.events if e.name == GRIP]
    if not grips:
        raise TeachError("g(쥠)가 없다 — 컵을 쥔 채 옮긴 궤적만 녹화한다")
    g0 = grips[0]
    g_before = hand[max(int(np.searchsorted(t, g0.t, side="left")) - 1, 0)]
    gripped = g0.t + float(np.abs(np.asarray(g0.q) - g_before).max()) / HAND_VEL_RAD_S
    after = t >= gripped
    if after.sum() < 2:
        raise TeachError("손을 다 쥔 뒤 녹화가 없다")
    span = _motion_span(t[after], q[after], trim_rad)
    if span is None:
        raise TeachError(f"쥔 뒤 팔이 {trim_rad} rad 넘게 움직이지 않았다 — 녹화할 궤적이 없다")
    opens = [e.t for e in teacher.events if e.name == OPEN and e.t >= span[0]]
    # 편 뒤: 녹화 손이 HAND_VEL 로 다 펴질 때까지 + post_roll(컵이 떨어질 시간)
    ends = [span[1] + post_roll]
    for e in teacher.events:
        if e.name == OPEN and e.t >= span[0]:
            before = hand[max(int(np.searchsorted(t, e.t, side="left")) - 1, 0)]
            ends.append(e.t + float(np.abs(np.asarray(e.q) - before).max()) / HAND_VEL_RAD_S + post_roll)
    t0 = max(gripped, span[0] - pre_roll)
    t1 = min(t[-1], max(ends))
    grid = np.arange(int(np.floor((t1 - t0) / step_dt + 1e-9)) + 1) * step_dt + t0
    arm = np.column_stack([np.interp(grid, t, q[:, j]) for j in range(7)])
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)   # 손은 계단: 그 시각까지 낸 목표
    hand_q = _rate_limited(hand[idx], HAND_VEL_RAD_S * step_dt)
    open_frames = [int(np.searchsorted(grid, e, side="left")) for e in opens]
    cut = min(open_frames) if open_frames else None
    n = len(grid)
    if cut is None or cut <= 0 or cut >= n:
        segments = [{"label": "place", "start": 0, "end": n - 1, "hand_command": "position"}]
    else:
        segments = [{"label": "carry", "start": 0, "end": cut - 1, "hand_command": "position"},
                    {"label": "release", "start": cut, "end": n - 1, "hand_command": "position"}]
    side = teacher.side
    return {
        "meta_schema": np.array(SCHEMA),
        "meta_side": np.array(side),
        "meta_arm_names": np.array(arm_names(side)),
        "meta_hand_names": np.array(hand_names(side)),
        "meta_step_dt": np.array(step_dt),
        "meta_segments": np.array(json.dumps(segments)),
        "meta_source": np.array("teach_place"),
        "arm_q": arm,
        "hand_q_cmd": hand_q,
        "arm_target": arm,                                   # replay_to_pd.py (팔만)
        "joint_names": np.array(arm_names(side)),
        "hand_event_names": np.array([e.name for e in teacher.events], dtype=str),
        "hand_event_t": np.array([e.t - t0 for e in teacher.events], float),     # 녹화 첫 프레임 기준(음수 = 그 전)
        "hand_event_rad": np.array([e.q for e in teacher.events], float).reshape(-1, 6),
    }


def raw_rows(teacher: Teacher) -> dict[str, np.ndarray]:
    """진단용 원본(모든 틱): 실측 · 팔 명령 · 손 명령 · 모드 · 손 사건. 재생 파일과 따로 저장한다."""
    if not teacher.rows:
        raise TeachError("녹화 행이 없다")
    return {
        "t": np.array([r[0] for r in teacher.rows], float),
        "q_meas": np.vstack([r[1] for r in teacher.rows]),
        "q_cmd": np.vstack([r[2] for r in teacher.rows]),
        "hand_cmd": np.vstack([r[3] for r in teacher.rows]),
        "mode": np.array([r[4] for r in teacher.rows], dtype=str),
        "event_names": np.array([e.name for e in teacher.events], dtype=str),
        "event_t": np.array([e.t for e in teacher.events], float),
    }


def save_recording(path: str | Path, fields: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        raise TeachError(f"녹화 파일은 .npz: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fields)
    return path

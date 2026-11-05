"""목 이동 요청·상태 규약과 정착 판정 — 순수부(ROS·하드웨어 무관).

목은 `head_joint_publisher.py` 한 프로세스가 시리얼 포트를 단독으로 맡는다(2026-10-15 사용자 결정). 그 프로세스가
`/head/joint_states` 를 계속 발행하면서 이동 요청도 받는다. 다른 도구(`head_move.py`, locate, survey)는 포트를 열지
않고 `/head/move_request` 로 요청해 `/head/move_status` 로 결과를 받는다.

규약(둘 다 std_msgs/String 안의 JSON):
  요청  {"id", "pan_deg", "tilt_deg"}   각도는 인코더 deg. 둘 다 null 이면 기준 자세(config/head_home.yaml)로.
  상태  {"id", "state", ...}            state: accepted → done | failed, 또는 rejected(busy·범위 밖).
        done/failed 는 실측 각도 pan_deg·tilt_deg(정착 창 평균)·pan_sigma_deg·tilt_sigma_deg·ok·reason 을 싣는다.

정착 판정은 이 프로세스가 읽은 현재 각도로 한다: 정착 창(최근 N 샘플)의 폭이 `still_deg` 이하이고 최소 `min_s` 가
지났으면 정지로 본다. 정지했을 때 평균이 목표 `tol_deg` 안이면 ok.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable

REQUEST_TOPIC = "/head/move_request"
STATUS_TOPIC = "/head/move_status"

#: 사용자 지정(2026-10-13): tilt 는 인코더 기준 -10° 이하만(위쪽은 제한 없음). pan 은 한 바퀴 안 어디든.
MAX_TILT_DEG = -10.0
PAN_RANGE_DEG = (-180.0, 180.0)
TILT_MIN_DEG = -180.0

#: 정지했는데 목표와 어긋나면 목표를 오차만큼 반대로 옮겨 다시 맞춘다(피드백 보정). 횟수·폭 상한.
MAX_TRIMS = 3
MAX_TRIM_DEG = 1.0
#: 보정 이득. 1 이면 불감대·잔여 이동이 섞인 오차를 통째로 되돌려 오버슈트한다(2026-10-15 실기: +0.53 → -0.45). 절반만 되돌린다.
TRIM_GAIN = 0.5

STATE_ACCEPTED, STATE_DONE, STATE_FAILED, STATE_REJECTED = "accepted", "done", "failed", "rejected"


class BadRequest(ValueError):
    """요청이 규약에 안 맞거나 범위 밖이다."""


#: 밖에서 온 JSON 을 읽다 날 수 있는 것 — 400자리 정수의 float() 는 OverflowError, '[' 10만 개는 RecursionError
_PARSE_ERRORS = (ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError)


@dataclass(frozen=True)
class MoveRequest:
    request_id: str
    pan_deg: float | None  # None 이면 기준 자세
    tilt_deg: float | None

    @property
    def is_home(self) -> bool:
        return self.pan_deg is None and self.tilt_deg is None


def encode_request(req: MoveRequest) -> str:
    return json.dumps({"id": req.request_id, "pan_deg": req.pan_deg, "tilt_deg": req.tilt_deg})


def parse_request(text: str, max_tilt_deg: float = MAX_TILT_DEG) -> MoveRequest:
    """JSON 을 요청으로 읽고 범위를 검사한다. 모터까지 틀린 값이 가면 안 된다."""
    try:
        raw = json.loads(text)
        request_id = str(raw["id"])
        pan, tilt = raw.get("pan_deg"), raw.get("tilt_deg")
    except _PARSE_ERRORS as exc:
        raise BadRequest(f"요청을 읽을 수 없다: {exc}") from exc
    if (pan is None) != (tilt is None):
        raise BadRequest("pan_deg 와 tilt_deg 는 둘 다 주거나 둘 다 비워야 한다")
    if pan is None:
        return MoveRequest(request_id, None, None)
    try:
        pan, tilt = float(pan), float(tilt)
    except _PARSE_ERRORS as exc:
        raise BadRequest(f"각도가 숫자가 아니다: {pan!r}, {tilt!r}") from exc
    if not (math.isfinite(pan) and math.isfinite(tilt)):
        raise BadRequest("각도가 유한한 숫자가 아니다")
    if not PAN_RANGE_DEG[0] <= pan <= PAN_RANGE_DEG[1]:
        raise BadRequest(f"pan {pan:+.1f}° 가 범위 {PAN_RANGE_DEG} 밖이다")
    if not TILT_MIN_DEG <= tilt <= max_tilt_deg:
        raise BadRequest(f"tilt {tilt:+.1f}° 가 허용 범위 [{TILT_MIN_DEG:g}, {max_tilt_deg:g}]° 밖이다(사용자 지정 한계)")
    return MoveRequest(request_id, pan, tilt)


def status_message(request_id: str, state: str, **fields) -> str:
    return json.dumps({"id": request_id, "state": state, **fields}, ensure_ascii=False)


def parse_status(text: str) -> dict:
    try:
        raw = json.loads(text)
        if "id" not in raw or "state" not in raw:
            raise KeyError("id/state")
    except (ValueError, KeyError, TypeError) as exc:
        raise BadRequest(f"상태를 읽을 수 없다: {exc}") from exc
    return raw


@dataclass(frozen=True)
class Settled:
    pan_deg: float
    tilt_deg: float
    pan_sigma_deg: float
    tilt_sigma_deg: float
    ok: bool
    reason: str = ""
    still: bool = False  # 정지해 있었는가(정지했지만 어긋난 경우와 미정착을 구분)


@dataclass
class SettleTracker:
    """이동 명령 뒤 읽은 현재 각도로 정지·목표 도달을 판정한다. 시계는 주입한다(시험용)."""

    target_pan_deg: float
    target_tilt_deg: float
    clock: Callable[[], float]
    window: int = 20
    still_deg: float = 0.2
    tol_deg: float = 0.3
    min_s: float = 1.0
    timeout_s: float = 20.0
    max_read_failures: int = 20
    _t0: float = field(init=False)
    _pan: deque = field(init=False)
    _tilt: deque = field(init=False)
    _failures: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._t0 = self.clock()
        self._pan, self._tilt = deque(maxlen=self.window), deque(maxlen=self.window)

    def add(self, pan_deg: float, tilt_deg: float) -> None:
        self._failures = 0
        self._pan.append(pan_deg)
        self._tilt.append(tilt_deg)

    def add_failure(self) -> None:
        self._failures += 1

    def _summary(self, ok: bool, reason: str) -> Settled:
        pan, tilt = list(self._pan), list(self._tilt)
        if not pan:
            return Settled(float("nan"), float("nan"), float("nan"), float("nan"), False, reason or "읽은 각도가 없다")
        return Settled(statistics.fmean(pan), statistics.fmean(tilt), statistics.pstdev(pan), statistics.pstdev(tilt), ok, reason)

    def poll(self) -> Settled | None:
        """아직 이동 중이면 None, 끝났으면 결과(ok 여부 포함)."""
        elapsed = self.clock() - self._t0
        if self._failures > self.max_read_failures:
            return self._summary(False, f"각도 읽기가 연속 {self._failures}회 실패")
        full = len(self._pan) == self.window
        still = full and max(max(self._pan) - min(self._pan), max(self._tilt) - min(self._tilt)) <= self.still_deg
        if still and elapsed >= self.min_s:
            err = max(abs(statistics.fmean(self._pan) - self.target_pan_deg), abs(statistics.fmean(self._tilt) - self.target_tilt_deg))
            if err <= self.tol_deg:
                return replace(self._summary(True, ""), still=True)
            return replace(self._summary(False, f"정지했지만 목표에서 {err:.2f}° 떨어져 있다(허용 {self.tol_deg}°)"), still=True)
        if elapsed >= self.timeout_s:
            return self._summary(False, f"{self.timeout_s:g}s 안에 정착하지 못했다")
        return None


def format_result_lines(status: dict) -> list[str]:
    """done/failed 상태를 `head_home.py` 검증 출력과 같은 줄 형식으로 만든다(survey.parse_head_home 이 그대로 읽는다).

    움직이기 전에 실패한 상태(검사 · 쓰기 실패)에는 목표 · 실측 칸이 없다 — 그 축은 건너뛴다(사유는 호출자가 보인다).
    """
    ok = bool(status.get("ok"))
    lines = []
    for axis in ("pan", "tilt"):
        if any(k not in status for k in (f"target_{axis}_deg", f"{axis}_deg", f"{axis}_sigma_deg")):
            continue
        target, actual = float(status[f"target_{axis}_deg"]), float(status[f"{axis}_deg"])
        sigma = float(status[f"{axis}_sigma_deg"])
        good = ok and abs(actual - target) <= 0.3
        lines.append(f"  {axis}: 목표 {target:+.2f}° · 실제 {actual:+.3f}° · 오차 {actual - target:+.3f}° · σ {sigma:.3f}° · "
                     f"{'✓' if good else '❌'}")
    return lines


# ---------- 연속 목표(stream) 규약 — 2026-11-05 사용자 요청 ----------
# /head/target_stream  {"seq", "pan_deg", "tilt_deg"}  절대 목표, 인코더 deg(드라이버 규약 tick·360/4095 − 180).
# 한 번 이동이 없을 때 목표가 오면 연속 모드로 들어가 tick 마다 클램프한 goal 만 쓴다. STREAM_STALE_S 동안 안 오면 마지막
# goal 을 두고 한 번 이동 모드로 돌아간다. /head/stream_log 는 연속 모드 tick 마다 시각·읽기·쓰기를 남긴다.
STREAM_TOPIC = "/head/target_stream"
STREAM_LOG_TOPIC = "/head/stream_log"
STREAM_ID = "stream"  # move_status 에 연속 모드 진입 · 이탈 · 거절을 남길 때 쓰는 id
STREAM_STALE_S = 0.5
STREAM_MAX_VEL_DPS = 60.0
STREAM_MAX_ACC_DPS2 = 300.0
#: 학생 정책의 관절 한계(sim_e2e 11.05). parse_request 범위 · droop 경계와 겹치는 부분만 쓴다.
STREAM_PAN_RANGE_DEG = (-45.0, 30.0)
STREAM_TILT_RANGE_DEG = (-2.0, 82.0)
#: 처짐 하한(droop_limit_tick)에서 이만큼 안쪽까지만 연속 목표를 보낸다.
STREAM_DROOP_MARGIN_DEG = 10.0
#: 속도 · 가속 계산에 쓰는 tick 간격 범위(s). 위는 긴 정지 뒤 한 tick 에 크게 뛰지 않게 묶는다. 아래는 0 나눗셈만 막는다 —
#  아래를 주기(0.05)로 두면 늦은 tick 뒤 0.02 s 만에 온 tick 이 0.05 s 몫(3°)을 써서 실제 속도가 150 °/s 가 된다.
STREAM_DT_RANGE_S = (0.005, 0.1)
STREAM_FAIL_COOLDOWN_S = 2.0   # 실패로 나온 뒤 이만큼 다시 들어가지 않는다
STREAM_MAX_READ_FAILS = 10     # 연속 읽기 실패 tick 수(20 Hz 에서 0.5 s) → 실패 이탈
STREAM_TRACK_ERR_DEG = 10.0    # |읽은 각 − 직전 goal| 이 이보다 크게
STREAM_TRACK_TICKS = 20        # 이만큼 이어지면(20 Hz 에서 1 s) 실패 이탈 — 막혔거나 토크가 빠졌다
EXIT_STALE, EXIT_FAIL = "stale", "fail"


@dataclass(frozen=True)
class StreamTarget:
    seq: int
    pan_deg: float
    tilt_deg: float


def parse_stream(text: str) -> StreamTarget:
    """연속 목표 JSON 을 읽는다. 범위는 여기서 거르지 않고 드라이버가 클램프한다(학생 출력이 한계에 붙어도 멈추지 않게)."""
    try:
        raw = json.loads(text)
        seq = raw["seq"]
        target = StreamTarget(seq, float(raw["pan_deg"]), float(raw["tilt_deg"]))
    except _PARSE_ERRORS as exc:
        raise BadRequest(f"연속 목표를 읽을 수 없다: {exc}") from exc
    if not isinstance(seq, int) or isinstance(seq, bool):  # int() 는 1.9 · Infinity · "7" 을 바꾸거나 터진다
        raise BadRequest(f"seq 가 정수가 아니다: {seq!r}")
    if not (math.isfinite(target.pan_deg) and math.isfinite(target.tilt_deg)):
        raise BadRequest("연속 목표 각도가 유한한 숫자가 아니다")
    return target


@dataclass(frozen=True)
class StreamLimits:
    pan: tuple[float, float]
    tilt: tuple[float, float]
    max_vel_dps: float = STREAM_MAX_VEL_DPS
    max_acc_dps2: float = STREAM_MAX_ACC_DPS2

    def problem(self) -> str:
        """쓸 수 없는 한계면 그 사유, 아니면 빈 문자열."""
        for axis, (lo, hi) in (("pan", self.pan), ("tilt", self.tilt)):
            if not lo <= hi:
                return f"연속 모드 {axis} 범위가 비었다 [{lo:g}, {hi:g}]° — --max-tilt · --stream-{axis}-min/max 를 확인"
        if not (self.max_vel_dps > 0 and self.max_acc_dps2 > 0):
            return f"속도 · 가속 상한은 양수여야 한다: {self.max_vel_dps}, {self.max_acc_dps2}"
        return ""


def stream_limits(max_tilt_deg: float, droop_tilt_deg: float | None, *,
                  pan_range: tuple[float, float] = STREAM_PAN_RANGE_DEG,
                  tilt_range: tuple[float, float] = STREAM_TILT_RANGE_DEG,
                  max_vel_dps: float = STREAM_MAX_VEL_DPS, max_acc_dps2: float = STREAM_MAX_ACC_DPS2,
                  pan_bias_deg: float = 0.0, tilt_bias_deg: float = 0.0) -> StreamLimits:
    """학생 한계 ∩ parse_request 범위(PAN_RANGE_DEG, [TILT_MIN_DEG, max_tilt]) ∩ 처짐 하한 − 여유.

    goal_bias 가 양수면 레지스터에 쓰는 값이 목표보다 크다 — 상한에서 그만큼 뺀다.
    """
    tilt_hi = min(tilt_range[1], max_tilt_deg)
    if droop_tilt_deg is not None:
        tilt_hi = min(tilt_hi, droop_tilt_deg - STREAM_DROOP_MARGIN_DEG)
    pan_hi = min(pan_range[1], PAN_RANGE_DEG[1]) - max(pan_bias_deg, 0.0)
    return StreamLimits((max(pan_range[0], PAN_RANGE_DEG[0]), pan_hi),
                        (max(tilt_range[0], TILT_MIN_DEG), tilt_hi - max(tilt_bias_deg, 0.0)),
                        float(max_vel_dps), float(max_acc_dps2))


def stream_step(cmd: float, vel: float, target: float, lo: float, hi: float,
                max_vel: float, max_acc: float, dt: float) -> tuple[float, float, str]:
    """한 축 한 tick: 범위 → 속도 → 가속 순으로 묶는다. (다음 goal, 다음 속도, 클램프 "range"·"rate"·"")."""
    cmd = min(hi, max(lo, cmd))
    goal = min(hi, max(lo, target))
    err = goal - cmd
    # tick 마다 max_acc·dt 씩 줄여 goal 에서 멈출 수 있는 가장 큰 속도 u·a·dt: Σ_j max(u − j, 0) ≤ |err|/(a·dt²)
    e = abs(err) / (max_acc * dt * dt)
    m = math.floor((math.sqrt(1.0 + 8.0 * e) - 1.0) / 2.0)
    want = math.copysign(min(abs(err) / dt, max_vel, max_acc * dt * (e + m * (m + 1) / 2.0) / (m + 1)), err)
    v = min(vel + max_acc * dt, max(vel - max_acc * dt, want))
    new = min(hi, max(lo, cmd + v * dt))
    v = (new - cmd) / dt
    if goal != target:
        return new, v, "range"
    return new, v, "rate" if abs(new - goal) > 1e-9 else ""

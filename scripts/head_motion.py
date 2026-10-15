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
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise BadRequest(f"요청을 읽을 수 없다: {exc}") from exc
    if (pan is None) != (tilt is None):
        raise BadRequest("pan_deg 와 tilt_deg 는 둘 다 주거나 둘 다 비워야 한다")
    if pan is None:
        return MoveRequest(request_id, None, None)
    try:
        pan, tilt = float(pan), float(tilt)
    except (TypeError, ValueError) as exc:
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
    """done/failed 상태를 `head_home.py` 검증 출력과 같은 줄 형식으로 만든다(survey.parse_head_home 이 그대로 읽는다)."""
    ok = bool(status.get("ok"))
    lines = []
    for axis in ("pan", "tilt"):
        target, actual = float(status[f"target_{axis}_deg"]), float(status[f"{axis}_deg"])
        sigma = float(status[f"{axis}_sigma_deg"])
        good = ok and abs(actual - target) <= 0.3
        lines.append(f"  {axis}: 목표 {target:+.2f}° · 실제 {actual:+.3f}° · 오차 {actual - target:+.3f}° · σ {sigma:.3f}° · "
                     f"{'✓' if good else '❌'}")
    return lines

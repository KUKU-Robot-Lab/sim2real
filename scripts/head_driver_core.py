"""목 드라이버의 상태기계 — 시리얼 포트를 단독으로 쓰며 읽기(발행)와 이동 요청을 한 스레드에서 번갈아 처리한다.

ROS 와 분리돼 있어 가짜 컨트롤러로 시험한다. `tick()` 이 매 주기 각도를 읽어 `publish_angles` 로 내보내고, 이동 중이면
같은 읽기를 정착 판정에 쓴다 — 이동 중에도 각도 발행이 끊기지 않는다. 요청은 한 번에 하나만 받는다(이동 중 새 요청은
busy 로 거절). 이것이 "두 명령자가 동시에 목을 움직이는 경우"를 막는 구조다(포트 잠금은 별개로 다른 프로세스를 막는다).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Protocol

import head_motion as hm
from head_compliant_hold import tick_to_deg
from head_home import ADDR_GOAL_POSITION, HeadHome, apply_one, check_reach, deg_to_tick
from head_turns import HeadReachError


class Controller(Protocol):
    def read_present_tick(self, dxl_id: int) -> int: ...


class HeadDriver:
    """`tick` 과 `on_request` 는 같은 스레드에서만 부른다(포트가 한 개라 직렬이어야 한다)."""

    def __init__(self, controller, config: HeadHome, *, pan_id: int, tilt_id: int,
                 publish_angles: Callable[[float, float], None], publish_status: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic, max_tilt_deg: float = hm.MAX_TILT_DEG,
                 apply: Callable = apply_one, log: Callable[[str], None] = lambda _m: None):
        self.controller, self.config = controller, config
        self.pan_id, self.tilt_id = pan_id, tilt_id
        self._publish_angles, self._publish_status = publish_angles, publish_status
        self._clock, self._max_tilt, self._apply, self._log = clock, max_tilt_deg, apply, log
        self._active: tuple[hm.MoveRequest, hm.SettleTracker] | None = None
        self._trims = 0  # 이 이동에서 이미 쓴 보정 횟수
        self._goal: dict[int, float] = {}  # 지금 모터에 쓴 목표(보정 포함)

    @property
    def moving(self) -> bool:
        return self._active is not None

    def _read(self) -> tuple[float, float]:
        return (tick_to_deg(self.controller.read_present_tick(self.pan_id)),
                tick_to_deg(self.controller.read_present_tick(self.tilt_id)))

    def on_request(self, text: str) -> None:
        try:
            req = hm.parse_request(text, self._max_tilt)
        except hm.BadRequest as exc:
            self._publish_status(hm.status_message(_id_of(text), hm.STATE_REJECTED, reason=str(exc)))
            return
        if self._active is not None:
            busy_id = self._active[0].request_id
            self._publish_status(hm.status_message(req.request_id, hm.STATE_REJECTED, reason=f"busy: 요청 {busy_id} 이동 중"))
            return
        pan = self.config.targets_deg[self.pan_id] if req.is_home else req.pan_deg
        tilt = self.config.targets_deg[self.tilt_id] if req.is_home else req.tilt_deg
        goal = replace(self.config, targets_deg={**self.config.targets_deg, self.pan_id: pan, self.tilt_id: tilt})
        ids = (self.pan_id, self.tilt_id)
        try:
            check_reach(self.controller, goal, ids)  # 둘 다 확인한 뒤에만 쓴다 — 한쪽만 움직이고 멈추지 않게
            for dxl_id in ids:
                self._apply(self.controller, goal, dxl_id)
        except HeadReachError as exc:
            self._publish_status(hm.status_message(req.request_id, hm.STATE_FAILED, ok=False, reason=str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 - 쓰기가 중간에 실패해도 상태를 알려야 한다
            self._publish_status(hm.status_message(req.request_id, hm.STATE_FAILED, ok=False, reason=f"모터 쓰기 실패: {exc}"))
            return
        self._trims = 0
        self._goal = {self.pan_id: pan, self.tilt_id: tilt}
        self._active = (req, hm.SettleTracker(pan, tilt, self._clock))
        self._publish_status(hm.status_message(req.request_id, hm.STATE_ACCEPTED, target_pan_deg=pan, target_tilt_deg=tilt))
        self._log(f"이동 요청 {req.request_id}: pan {pan:+.2f} tilt {tilt:+.2f}")

    def tick(self) -> None:
        try:
            pan, tilt = self._read()
        except Exception as exc:  # noqa: BLE001 - 한 번 실패해도 노드를 죽이지 않는다
            self._log(f"목 각도 읽기 실패: {exc}")
            if self._active is not None:
                self._active[1].add_failure()
                self._finish_if_settled()
            return
        self._publish_angles(pan, tilt)
        if self._active is not None:
            self._active[1].add(pan, tilt)
            self._finish_if_settled()

    def _finish_if_settled(self) -> None:
        assert self._active is not None
        req, tracker = self._active
        result = tracker.poll()
        if result is None:
            return
        if self._trim(req, tracker, result):
            return
        self._active = None
        self._publish_status(hm.status_message(
            req.request_id, hm.STATE_DONE if result.ok else hm.STATE_FAILED, ok=result.ok, reason=result.reason,
            pan_deg=result.pan_deg, tilt_deg=result.tilt_deg,
            pan_sigma_deg=result.pan_sigma_deg, tilt_sigma_deg=result.tilt_sigma_deg,
            target_pan_deg=tracker.target_pan_deg, target_tilt_deg=tracker.target_tilt_deg))


    def _trim(self, req: hm.MoveRequest, tracker: hm.SettleTracker, result: hm.Settled) -> bool:
        """정지했는데 목표와 어긋나면(마찰·불감대) 목표를 오차만큼 반대로 옮겨 다시 맞춘다(피드백 보정).

        최대 `MAX_TRIMS` 회, 보정 폭은 `MAX_TRIM_DEG` 이하로 제한한다. 보정 목표를 쓴 뒤에도 판정은 원래 목표 기준이다.
        """
        if result.ok or not result.still or self._trims >= hm.MAX_TRIMS:
            return False
        err_pan, err_tilt = result.pan_deg - tracker.target_pan_deg, result.tilt_deg - tracker.target_tilt_deg
        if max(abs(err_pan), abs(err_tilt)) > hm.MAX_TRIM_DEG:
            return False
        self._trims += 1
        goal_pan, goal_tilt = self._goal[self.pan_id] - hm.TRIM_GAIN * err_pan, self._goal[self.tilt_id] - hm.TRIM_GAIN * err_tilt
        self._goal = {self.pan_id: goal_pan, self.tilt_id: goal_tilt}
        try:  # 토크·모드·게인은 건드리지 않고 목표 위치만 쓴다(토크를 끄면 처진다)
            for dxl_id, deg in self._goal.items():
                self.controller.write4(dxl_id, ADDR_GOAL_POSITION, deg_to_tick(deg), "goal trim")
        except Exception:  # noqa: BLE001 - 보정 실패는 원래 결과로 끝낸다
            return False
        self._active = (req, hm.SettleTracker(tracker.target_pan_deg, tracker.target_tilt_deg, self._clock))
        self._log(f"보정 {self._trims}: 오차 pan {err_pan:+.2f} tilt {err_tilt:+.2f} → 목표 {goal_pan:+.2f}/{goal_tilt:+.2f}")
        return True


def _id_of(text: str) -> str:
    """파싱이 실패한 요청에도 가능하면 id 를 되돌려 준다(클라이언트가 자기 요청의 거절을 알아보게)."""
    import json

    try:
        return str(json.loads(text)["id"])
    except (ValueError, KeyError, TypeError):
        return ""

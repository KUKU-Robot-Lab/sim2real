"""목 드라이버의 상태기계 — 시리얼 포트를 단독으로 쓰며 읽기(발행)와 이동 요청을 한 스레드에서 번갈아 처리한다.

ROS 와 분리돼 있어 가짜 컨트롤러로 시험한다. `tick()` 이 매 주기 각도를 읽어 `publish_angles` 로 내보내고, 이동 중이면
같은 읽기를 정착 판정에 쓴다 — 이동 중에도 각도 발행이 끊기지 않는다. 요청은 한 번에 하나만 받는다(이동 중 새 요청은
busy 로 거절). 이것이 "두 명령자가 동시에 목을 움직이는 경우"를 막는 구조다(포트 잠금은 별개로 다른 프로세스를 막는다).

연속 모드(2026-11-05, head_motion 의 stream 규약): `on_stream` 은 최신 목표만 저장한다. 한 번 이동이 없을 때 목표가 오면
`tick` 이 토크 · 모드 · 지금 자리를 확인하고 들어간다. 그 뒤 tick 마다 읽기 → 발행 → 클램프한 goal 쓰기(goal 레지스터만)를
한다. 목표가 stale_s 동안 안 오면 마지막 goal 을 두고 나온다. 연속 모드 중 이동 요청은 busy(stream) 로 거절한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Callable, Protocol

import head_motion as hm
from head_compliant_hold import tick_to_deg
from head_home import (ADDR_GOAL_POSITION, ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE, TORQUE_ON, HeadHome, apply_one,
                       bring_to, goal_tick)
from head_turns import HeadReachError, physical_tick


READ_STATS_S = 10.0  # tick 읽기 소요 시간 p50/p95/max 를 info 로 내는 주기


def stream_limits_for(config: HeadHome, pan_id: int, tilt_id: int, max_tilt_deg: float, **kwargs) -> hm.StreamLimits:
    """설정의 처짐 하한 · goal_bias_tick 을 넣은 연속 모드 한계(kwargs 는 hm.stream_limits 의 범위 · 속도 · 가속)."""
    droop = config.droop_limit_tick.get(tilt_id)

    def bias_deg(dxl_id: int) -> float:
        return tick_to_deg(int(config.goal_bias_tick.get(dxl_id, 0))) - tick_to_deg(0)

    return hm.stream_limits(max_tilt_deg, None if droop is None else tick_to_deg(droop),
                            pan_bias_deg=bias_deg(pan_id), tilt_bias_deg=bias_deg(tilt_id), **kwargs)


class Controller(Protocol):
    def read_present_tick(self, dxl_id: int) -> int: ...


class HeadDriver:
    """`tick` 과 `on_request` 는 같은 스레드에서만 부른다(포트가 한 개라 직렬이어야 한다)."""

    def __init__(self, controller, config: HeadHome, *, pan_id: int, tilt_id: int,
                 publish_angles: Callable[[float, float, float], None], publish_status: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic, max_tilt_deg: float = hm.MAX_TILT_DEG,
                 apply: Callable = apply_one, log: Callable[[str], None] = lambda _m: None,
                 period_s: float = 0.05, stream_limits: hm.StreamLimits | None = None,
                 stream_stale_s: float = hm.STREAM_STALE_S, publish_stream_log: Callable[[str], None] = lambda _t: None,
                 wall: Callable[[float], float] = lambda t: t):
        """publish_angles(pan, tilt, 읽기 가운데 시각) — 시각은 `wall` 로 바꾼 값(노드는 ROS 시계 초)."""
        self.controller, self.config = controller, config
        self.pan_id, self.tilt_id = pan_id, tilt_id
        self._publish_angles, self._publish_status = publish_angles, publish_status
        self._clock, self._max_tilt, self._apply, self._log = clock, max_tilt_deg, apply, log
        self._active: tuple[hm.MoveRequest, hm.SettleTracker] | None = None
        self._trims = 0  # 이 이동에서 이미 쓴 보정 횟수
        self._goal: dict[int, float] = {}  # 지금 모터에 쓴 목표(보정 포함)
        self._period, self._stale_s, self._wall = float(period_s), float(stream_stale_s), wall
        self._publish_stream_log = publish_stream_log
        self._limits = stream_limits or stream_limits_for(config, pan_id, tilt_id, max_tilt_deg)
        self._target: tuple[hm.StreamTarget, float] | None = None  # (최신 목표, 받은 시각) — 아직 반영 안 했으면 남아 있다
        self._stream: dict | None = None  # 연속 모드 상태: cmd · vel · 마지막 목표
        self._stream_note = ""  # 같은 거절 사유를 30 Hz 로 반복해 내지 않게
        self._fail_note = ""    # 같은 실패 사유도 한 번만
        self._cooldown_until = float("-inf")  # 실패 이탈 뒤 재진입 금지 끝 시각
        self._read_ms: list[float] = []
        self._stats_t0 = clock()

    @property
    def moving(self) -> bool:
        return self._active is not None

    @property
    def streaming(self) -> bool:
        return self._stream is not None

    def _read(self) -> tuple[float, float]:
        return self._deg(self.pan_id), self._deg(self.tilt_id)

    def _deg(self, dxl_id: int) -> float:
        """처짐 하한이 있는 모터는 실제 자리 하나로 펼쳐 읽는다 — 처짐 복구가 실패해 모드 4 로 남으면 Present 가 −1284 처럼 센다."""
        tick = self.controller.read_present_tick(dxl_id)
        limit = self.config.droop_limit_tick.get(dxl_id)
        return tick_to_deg(tick if limit is None else physical_tick(tick, limit))

    def on_request(self, text: str) -> None:
        try:
            req = hm.parse_request(text, self._max_tilt)
        except hm.BadRequest as exc:
            self._publish_status(hm.status_message(_id_of(text), hm.STATE_REJECTED, reason=str(exc)))
            return
        if self._stream is not None:
            self._publish_status(hm.status_message(req.request_id, hm.STATE_REJECTED, reason="busy: stream 연속 모드 중"))
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
            # 둘 다 확인한 뒤에만 쓴다. 처진 tilt 는 여기서 먼저 올린다.
            # shortcut: 처짐 복구(약 1 s)는 이 스레드를 막아 그동안 각도 발행이 멈춘다, 복구가 잦아지면 tick 상태기계로 옮긴다
            bring_to(self.controller, goal, ids, apply=self._apply)
        except HeadReachError as exc:
            self._publish_status(hm.status_message(req.request_id, hm.STATE_FAILED, ok=False, reason=str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 - 쓰기가 중간에 실패해도 상태를 알려야 한다
            self._publish_status(hm.status_message(req.request_id, hm.STATE_FAILED, ok=False, reason=f"모터 쓰기 실패: {exc}"))
            return
        self._trims = 0
        self._goal = {self.pan_id: pan, self.tilt_id: tilt}
        self._active = (req, hm.SettleTracker(pan, tilt, self._clock))
        self._target = None  # 이동 전에 받아 둔 연속 목표로 이동 뒤 들어가지 않게
        self._publish_status(hm.status_message(req.request_id, hm.STATE_ACCEPTED, target_pan_deg=pan, target_tilt_deg=tilt))
        self._log(f"이동 요청 {req.request_id}: pan {pan:+.2f} tilt {tilt:+.2f}")

    def on_stream(self, text: str) -> None:
        """최신 연속 목표와 받은 시각만 둔다. 모터는 tick 에서만 건드린다."""
        try:
            target = hm.parse_stream(text)
        except Exception as exc:  # noqa: BLE001 - 콜백 밖으로 나가면 노드가 죽는다
            self._note(f"연속 목표 거절: {exc}")
            return
        if self._active is not None:
            self._note(f"busy: 요청 {self._active[0].request_id} 이동 중 — 연속 목표 무시")
            return
        self._target = (target, self._clock())

    def tick(self) -> None:
        t0 = self._clock()
        try:
            pan, tilt = self._read()
        except Exception as exc:  # noqa: BLE001 - 한 번 실패해도 노드를 죽이지 않는다
            self._log(f"목 각도 읽기 실패: {exc}")
            if self._active is not None:
                self._active[1].add_failure()
                self._finish_if_settled()
            elif self._stream is not None:
                self._stream["read_fails"] += 1
                if self._stream["read_fails"] >= hm.STREAM_MAX_READ_FAILS:
                    self._leave(hm.EXIT_FAIL, f"읽기 {hm.STREAM_MAX_READ_FAILS} tick 연속 실패: {exc}")
                else:
                    self._leave_if_stale()
            return
        t1 = self._clock()
        self._read_stats(t1 - t0, t1)
        self._publish_angles(pan, tilt, self._wall((t0 + t1) / 2))
        if self._active is not None:
            self._active[1].add(pan, tilt)
            self._finish_if_settled()
        elif self._stream is not None:
            if not self._leave_if_stale():
                self._stream_tick(pan, tilt, t0, t1)
        elif self._target is not None and self._enter_stream(pan, tilt):
            self._stream_tick(pan, tilt, t0, t1)

    def _note(self, reason: str) -> None:
        if reason != self._stream_note:
            self._stream_note = reason
            self._publish_status(hm.status_message(hm.STREAM_ID, hm.STATE_REJECTED, reason=reason))

    def _read_stats(self, seconds: float, now: float) -> None:
        self._read_ms.append(seconds * 1000.0)
        if now - self._stats_t0 < READ_STATS_S:
            return
        ms = sorted(self._read_ms)
        n = len(ms)
        self._log(f"읽기 두 모터 {n}회 p50 {ms[n // 2]:.2f} · p95 {ms[min(n - 1, int(0.95 * n))]:.2f} · max {ms[-1]:.2f} ms")
        self._read_ms, self._stats_t0 = [], now

    def _enter_stream(self, pan: float, tilt: float) -> bool:
        """토크가 켜져 있고 모드가 설정과 같고 지금 자리가 연속 범위 안일 때만 들어간다 — 아니면 목표를 버리고 거절."""
        assert self._target is not None
        if self._clock() - self._target[1] > self._stale_s:  # 이동 중 직전에 받아 둔 낡은 목표로는 들어가지 않는다
            self._target = None
            return False
        problem = self._limits.problem()
        if self._clock() < self._cooldown_until:
            problem = f"실패 이탈 뒤 {hm.STREAM_FAIL_COOLDOWN_S:g} s 동안 연속 모드에 들어가지 않는다"
        if not problem:
            try:
                for dxl_id in (self.pan_id, self.tilt_id):
                    name = self.config.names.get(dxl_id, f"id {dxl_id}")
                    if self.controller.read1(dxl_id, ADDR_TORQUE_ENABLE, "torque") != TORQUE_ON:
                        problem = f"{name} 토크가 꺼져 있다 — 먼저 기준 자세 이동(head_home)"
                        break
                    if self.controller.read1(dxl_id, ADDR_OPERATING_MODE, "mode") != int(self.config.operating_mode):
                        problem = f"{name} 모드가 설정({self.config.operating_mode})과 다르다 — 먼저 기준 자세 이동"
                        break
            except Exception as exc:  # noqa: BLE001 - 확인 못 하면 들어가지 않는다
                problem = f"토크 · 모드 읽기 실패: {exc}"
        if not problem:
            for axis, deg, (lo, hi) in (("pan", pan, self._limits.pan), ("tilt", tilt, self._limits.tilt)):
                if not lo <= deg <= hi:
                    problem = f"지금 {axis} {deg:+.2f}° 가 연속 범위 [{lo:g}, {hi:g}]° 밖이다(처짐 · 다른 자세)"
                    break
        if problem:
            self._target = None
            self._note(problem)
            return False
        self._stream = {"cmd": [pan, tilt], "vel": [0.0, 0.0], "t": None, "read_fails": 0, "track": 0}
        self._stream_note = ""
        self._publish_status(hm.status_message(hm.STREAM_ID, hm.STATE_ACCEPTED, pan_deg=pan, tilt_deg=tilt))
        self._log(f"연속 모드 시작: pan {pan:+.2f} tilt {tilt:+.2f}")
        return True

    def _leave_if_stale(self) -> bool:
        assert self._stream is not None and self._target is not None
        age = self._clock() - self._target[1]
        if age <= self._stale_s:
            return False
        self._leave(hm.EXIT_STALE, f"stale {age:.2f} s")
        return True

    def _leave(self, exit_kind: str, reason: str) -> None:
        """연속 모드를 끝낸다. goal 레지스터는 마지막으로 쓴 값 그대로다. 실패면 쿨다운을 걸고 같은 사유는 한 번만 낸다."""
        self._stream, self._target = None, None
        pan, tilt = self._goal.get(self.pan_id), self._goal.get(self.tilt_id)
        ok = exit_kind == hm.EXIT_STALE
        self._log(f"연속 모드 끝({exit_kind}): {reason} · goal pan {pan} tilt {tilt}")
        if not ok:
            self._cooldown_until = self._clock() + hm.STREAM_FAIL_COOLDOWN_S
            if reason == self._fail_note:
                return
            self._fail_note = reason
        self._publish_status(hm.status_message(hm.STREAM_ID, hm.STATE_DONE if ok else hm.STATE_FAILED, ok=ok,
                                               exit=exit_kind, reason=reason, goal_pan_deg=pan, goal_tilt_deg=tilt))

    def _write_goals(self, goal: dict[int, float]) -> str:
        """goal 레지스터에만 쓴다(_trim 과 같은 경로). 실패하면 사유를 돌려주고 self._goal 은 실제로 쓴 모터만 바꾼다."""
        written = []
        try:
            for dxl_id, deg in goal.items():
                self.controller.write4(dxl_id, ADDR_GOAL_POSITION, goal_tick(self.config, dxl_id, deg), "stream goal")
                self._goal[dxl_id] = deg
                written.append(self.config.names.get(dxl_id, f"id {dxl_id}"))
        except Exception as exc:  # noqa: BLE001 - 쓰기 실패면 연속 모드를 끝내고 알린다
            part = f"({' · '.join(written)} 만 썼다)" if written else ""
            return f"모터 쓰기 실패{part}: {exc}"
        return ""

    def _stream_tick(self, pan: float, tilt: float, t0: float, t1: float) -> None:
        """추종 감시 → 클램프 → goal 레지스터 쓰기 → stream_log. 토크 · 모드 · 게인은 건드리지 않는다."""
        assert self._stream is not None and self._target is not None
        target, received = self._target
        st, lim = self._stream, self._limits
        st["read_fails"] = 0
        lag = max(abs(pan - st["cmd"][0]), abs(tilt - st["cmd"][1]))
        st["track"] = st["track"] + 1 if lag > hm.STREAM_TRACK_ERR_DEG else 0
        if st["track"] >= hm.STREAM_TRACK_TICKS:  # 막혔거나 토크가 빠졌다 — 읽은 자리를 목표로 두고 나온다
            problem = self._write_goals({self.pan_id: pan, self.tilt_id: tilt})
            self._leave(hm.EXIT_FAIL, f"추종 오차 {lag:.1f}° 가 {hm.STREAM_TRACK_TICKS} tick 이어졌다"
                        + (f" · {problem}" if problem else ""))
            return
        lo_dt, hi_dt = hm.STREAM_DT_RANGE_S
        dt = self._period if st["t"] is None else min(hi_dt, max(lo_dt, t1 - st["t"]))
        st["t"] = t1
        clamp = {}
        for i, (axis, want, (lo, hi)) in enumerate((("pan", target.pan_deg, lim.pan), ("tilt", target.tilt_deg, lim.tilt))):
            st["cmd"][i], st["vel"][i], clamp[axis] = hm.stream_step(
                st["cmd"][i], st["vel"][i], want, lo, hi, lim.max_vel_dps, lim.max_acc_dps2, dt)
        goal = {self.pan_id: st["cmd"][0], self.tilt_id: st["cmd"][1]}
        problem = self._write_goals(goal)
        if problem:
            self._leave(hm.EXIT_FAIL, problem)
            return
        self._fail_note = ""
        written = self._clock()
        self._publish_stream_log(json.dumps({
            "seq": target.seq, "recv_t": self._wall(received), "read_t0": self._wall(t0), "read_t1": self._wall(t1),
            "pan_deg": pan, "tilt_deg": tilt, "target_pan_deg": target.pan_deg, "target_tilt_deg": target.tilt_deg,
            "goal_pan_deg": goal[self.pan_id], "goal_tilt_deg": goal[self.tilt_id], "write_t": self._wall(written),
            "dt": dt, "clamp": clamp}))

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
            for dxl_id, deg in self._goal.items():   # goal_bias_tick 도 같이(apply_one 과 같은 목표 규약)
                self.controller.write4(dxl_id, ADDR_GOAL_POSITION, goal_tick(self.config, dxl_id, deg), "goal trim")
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
    except Exception:  # noqa: BLE001 - 거절 상태를 내는 길에서 다시 터지면 안 된다('[' 10만 개 → RecursionError)
        return ""

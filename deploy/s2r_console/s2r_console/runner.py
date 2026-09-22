"""단계 하나를 실행한다 — `mission_run.run_stage` 의 콘솔판.

CLI 와 **다른 점 하나**: CLI 는 단계가 끝나면 그 단계의 배경 프로세스를 정지한다
(`finally: _stop(background)`). 그래서 "pd 노드를 띄운다" 같은 단계는 CLI 에서 뜨자마자 죽는다.
콘솔에서는 배경 프로세스가 **run 이 끝날 때까지** 산다 — 감독자가 들고 있다가 run 종료 때 정지한다.

나머지는 같다: 명령은 순서대로, manual 은 절대 실행하지 않고 사람의 확인을 기다리고,
전경 명령이 0 이 아닌 값으로 끝나면 그 자리에서 FAILED.

덧붙인 규칙 둘:
  · 배경 명령은 띄운 뒤 `settle_s` 동안 지켜본다 — 그 사이 죽으면 FAILED (launch 가 즉사한 것을 다음 단계에서야 아는 일을 막는다).
  · 같은 키의 배경 프로세스가 이미 살아 있으면 다시 띄우지 않는다 — 반복(loop_to) 때 체인을 두 번 띄우지 않는다.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .supervisor import Supervisor, SupervisorError

DONE, FAILED, ABORTED = "DONE", "FAILED", "ABORTED"   # mission_core.STATUS_* 와 같은 글자 (테스트가 잠근다)

#: 배경 프로세스를 띄운 뒤 살아 있는지 지켜보는 시간 [s].
SETTLE_S = 2.0
#: `ros2 launch` 는 자식 노드가 import 하다 죽는 데 몇 초 걸린다 — 그만큼 더 지켜본다(09.22 fake 팔 브리지 사례).
LAUNCH_SETTLE_S = 5.0
_POLL_S = 0.1


@dataclass
class Step:
    index: int
    note: str
    argv: tuple[str, ...]
    kind: str                       # background | foreground | manual | stop
    status: str = "pending"         # pending | running | waiting | up | done | failed | aborted | kept
    rc: int | None = None
    key: str = ""
    detail: str = ""
    targets: tuple[str, ...] = ()   # stop 이 내릴 배경 명령 키

    def as_dict(self) -> dict:
        return {"index": self.index, "note": self.note, "argv": list(self.argv), "kind": self.kind,
                "status": self.status, "rc": self.rc, "key": self.key, "detail": self.detail,
                "targets": list(self.targets)}


def step_kind(c) -> str:
    """명령 하나의 종류. `stop` 은 실행이 아니라 앞 단계가 띄운 배경 명령을 내린다."""
    if getattr(c, "stop", ()):
        return "stop"
    return "manual" if c.manual else ("background" if c.background else "foreground")


@dataclass
class _Flags:
    abort: bool = False
    acks: dict[int, bool] = field(default_factory=dict)   # index → 정상이었는가


class StageRunner:
    def __init__(self, stage_id: str, commands: Sequence, supervisor: Supervisor, *,
                 on_done: Callable[[str, str, str], None], settle_s: float = SETTLE_S) -> None:
        self.stage_id = stage_id
        self._sup, self._on_done, self._settle = supervisor, on_done, settle_s
        self.steps = [Step(index=i, note=c.note or " ".join(c.argv[:3]) or "정지 " + ", ".join(c.stop), argv=tuple(c.argv),
                           kind=step_kind(c), key=f"{stage_id}#{i}", targets=tuple(getattr(c, "stop", ()) or ()))
                      for i, c in enumerate(commands)]
        self.outcome: str | None = None
        self.note = ""
        self._flags = _Flags()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"stage-{stage_id}", daemon=True)

    # ── 바깥에서 부르는 것 ──────────────────────────────────────────────
    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def active(self) -> bool:
        return self.outcome is None

    def ack(self, index: int, ok: bool) -> None:
        with self._lock:
            step = self.steps[index] if 0 <= index < len(self.steps) else None
            if step is None or step.kind != "manual" or step.status != "waiting":
                raise ValueError("지금 확인을 기다리는 수동 명령이 아니다")
            self._flags.acks[index] = ok

    def abort(self) -> None:
        with self._lock:
            self._flags.abort = True

    def view(self) -> dict:
        with self._lock:
            return {"stage": self.stage_id, "active": self.active, "outcome": self.outcome,
                    "note": self.note, "steps": [s.as_dict() for s in self.steps]}

    # ── 안 ──────────────────────────────────────────────────────────────
    def _set(self, step: Step, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(step, k, v)

    def _aborting(self) -> bool:
        with self._lock:
            return self._flags.abort

    def _run(self) -> None:
        outcome, note = DONE, ""
        try:
            for step in self.steps:
                if self._aborting():
                    outcome, note = ABORTED, "운영자가 중단했다"
                    break
                result = getattr(self, f"_do_{step.kind}")(step)
                if result is not None:
                    outcome, note = result
                    break
            else:
                outcome, note = self._late_deaths() or (DONE, "")
        except Exception as exc:  # noqa: BLE001 — 실행 스레드가 조용히 죽으면 화면이 영원히 "실행 중" 이다
            outcome, note = FAILED, f"러너 오류: {exc}"
        with self._lock:
            for s in self.steps:
                if s.status in ("pending", "running", "waiting") and outcome != DONE:
                    s.status = "aborted" if s.status != "pending" else "pending"
            self.outcome, self.note = outcome, note
        self._on_done(self.stage_id, outcome, note)

    def _late_deaths(self):
        """단계가 끝나는 시점에 이 단계가 띄운 launch 의 자식이 죽어 있으면 실패 — 뜬 뒤에 죽은 것도 잡는다."""
        for step in self.steps:
            if step.kind == "background" and step.status == "up":
                died = self._sup.child_deaths(step.key)
                if died:
                    self._set(step, status="failed", detail=died[0][-240:])
                    return FAILED, f"{step.note}: launch 안의 노드가 죽었다 — {died[0][-200:]}"
        return None

    def _do_manual(self, step: Step):
        self._set(step, status="waiting")
        while True:
            with self._lock:
                ok = self._flags.acks.get(step.index)
            if ok is not None:
                break
            if self._aborting():
                return ABORTED, "운영자가 중단했다"
            time.sleep(_POLL_S)
        if not ok:
            self._set(step, status="failed", detail="운영자가 '정상이 아니다' 로 답했다")
            return ABORTED, f"수동 명령이 정상이 아니었다: {step.note}"
        self._set(step, status="done", detail="운영자 확인")
        return None

    def _do_background(self, step: Step):
        if self._sup.is_alive(step.key):
            self._set(step, status="kept", detail="이미 떠 있다 — 다시 띄우지 않는다")
            return None
        foreign = self._sup.foreign_launches(step.argv)
        if foreign:                          # 콘솔 밖에서 같은 launch 가 떠 있다 — 둘을 같은 하드웨어에 붙이지 않는다
            pids = ", ".join(str(p) for p, _ in foreign)
            self._set(step, status="failed", detail=f"콘솔 밖에서 이미 떠 있다: pid {pids}")
            return FAILED, (f"{step.note}: 같은 launch 가 콘솔 밖에서 이미 떠 있다(pid {pids}) — 두 번 띄우지 않는다. "
                            "하드웨어를 받친 뒤 그 PID 를 정리하고 다시 실행할 것")
        try:
            self._sup.spawn(step.key, stage=self.stage_id, note=step.note, argv=step.argv, background=True, manual=False)
        except SupervisorError as exc:
            self._set(step, status="failed", detail=str(exc))
            return FAILED, f"{step.note}: {exc}"
        self._set(step, status="running")
        settle = max(self._settle, LAUNCH_SETTLE_S) if tuple(step.argv[:2]) == ("ros2", "launch") and self._settle > 0.5 \
            else self._settle
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            if not self._sup.is_alive(step.key):
                rc = self._sup.rc(step.key)
                self._set(step, status="failed", rc=rc, detail=f"띄운 지 {settle:.0f} s 안에 끝났다 (rc={rc})")
                return FAILED, f"{step.note}: 뜨자마자 끝났다 (rc={rc}) — 로그를 볼 것"
            if self._aborting():
                return ABORTED, "운영자가 중단했다"
            time.sleep(_POLL_S)
        died = self._sup.child_deaths(step.key)
        if died:
            self._set(step, status="failed", detail=died[0][-240:])
            return FAILED, f"{step.note}: launch 안의 노드가 죽었다 — {died[0][-200:]}"
        self._set(step, status="up")
        return None

    def _do_stop(self, step: Step):
        """앞 단계가 띄운 배경 명령을 **적힌 순서대로** 하나씩 내린다(pd → 손 → 팔). 이미 없는 것은 넘어간다."""
        self._set(step, status="running")
        gone = []
        for key in step.targets:
            if self._aborting():
                return ABORTED, "운영자가 중단했다"
            if self._sup.is_alive(key):
                self._sup.stop([key])
                if self._sup.is_alive(key):
                    self._set(step, status="failed", detail=f"{key} 가 정지되지 않았다")
                    return FAILED, f"{step.note}: {key} 가 정지되지 않았다 — PID 를 확인할 것"
                gone.append(key)
        self._set(step, status="done", detail=f"정지: {', '.join(gone) or '이미 모두 내려가 있었다'}")
        return None

    def _do_foreground(self, step: Step):
        try:
            self._sup.spawn(step.key, stage=self.stage_id, note=step.note, argv=step.argv, background=False, manual=False)
        except SupervisorError as exc:
            self._set(step, status="failed", detail=str(exc))
            return FAILED, f"{step.note}: {exc}"
        self._set(step, status="running")
        while self._sup.is_alive(step.key):
            if self._aborting():
                self._sup.stop([step.key])
                self._set(step, status="aborted", rc=self._sup.rc(step.key))
                return ABORTED, "운영자가 중단했다"
            time.sleep(_POLL_S)
        rc = self._sup.rc(step.key)
        if rc != 0:
            self._set(step, status="failed", rc=rc, detail=f"rc={rc}")
            return FAILED, f"{step.note}: rc={rc}"
        self._set(step, status="done", rc=0)
        return None

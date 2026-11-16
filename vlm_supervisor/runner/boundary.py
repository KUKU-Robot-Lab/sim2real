"""경계 러너 — EpisodeManager 를 고치지 않고 step() 과 approve 콜백으로 구동하며 노드 경계마다 VLM 판정을 받는다.

순수 논리(ROS · sim 없음). 순서는 yaml 그대로이고 em.index 를 옮기지 않는다. 결정은 advance · look_again · handoff · stop 뿐.
FAILURE · 정지는 판정기를 거치지 않는다. VLM 답 · 사진 · 기록 · 승인 가운데 하나라도 불확실하면 handoff 다.
정지(em.stop)는 grant 종료 · 기록보다 먼저 로봇에 간다.

auto 모드: 첫 노드와 handoff 뒤 재개의 첫 노드는 앞 경계가 없어 판정 없이 에피소드 grant 한 줄로 돈다(confirm 노드는
그 노드의 사람 줄이 따로 있어야 한다). 재개에 쓰는 줄은 handoff 뒤에 새로 쓴 줄이어야 한다(Grants.end 의 시각 뒤).
"""
from __future__ import annotations

import copy
import logging
import math
import queue
import threading
import time
from dataclasses import replace
from typing import Callable, Mapping, Sequence

from policy_control.episode_runner import FAILURE, READY, RUNNING, STOPPED, SUCCESS, EpisodeManager
from policy_control.episode_spec import Episode, Node

from .grants import Grants
from .protocols import (ACTIONS, ADVANCE, AUTO, HANDOFF, LOOK_AGAIN, MODES, RECORD, STOP, ArmStill, Decision,
                        FrameSource, Judge, Question)

HANDED = "HANDOFF"          # step() · run() 이 돌려주는 값 — 사람에게 넘김(에피소드 상태는 READY 그대로, 실패가 아니다)
MIN_MARGIN = 0.2            # shortcut: 1등 · 2등 점수 격차 하한, 근거 없는 초안값 — 벤치 점수 분포를 보고 다시 정한다
BEFORE_MAX_AGE_S = 2.0      # 앞 사진은 노드 실행 직전 이 시간 안에 찍힌 것만
_DONE = (SUCCESS, FAILURE, STOPPED)
_log = logging.getLogger(__name__)


def picks(answers: Sequence[Mapping], n: int) -> list[str]:
    """1단 답 → 질문별 고른 선택지. 개수 · 형식이 틀리거나 low_mass · 점수 없음 · NaN · 동점 · 마진 미달이면 ValueError."""
    if len(answers) != n:
        raise ValueError(f"답 {len(answers)}개 != 질문 {n}개")
    out = []
    for a in answers:
        scores = a.get("scores")
        if a.get("low_mass") or not isinstance(scores, Mapping) or not scores:
            raise ValueError("low_mass 또는 점수 없음")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in scores.values()):
            raise ValueError("점수가 유한한 숫자가 아니다")
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if len(top) > 1 and top[0][1] - top[1][1] < MIN_MARGIN:
            raise ValueError(f"1등 · 2등 격차 {top[0][1] - top[1][1]:.3f} < {MIN_MARGIN}")
        out.append(top[0][0])
    return out


def _err(errors: list, what: str, exc: BaseException) -> None:
    msg = f"{what}: {type(exc).__name__}: {exc}"
    if msg not in errors:       # 폴링 중 같은 오류를 쌓지 않는다
        errors.append(msg)


class BoundaryRunner:
    def __init__(self, ep: Episode, executor, *, mode: str, judge: Judge, frames: FrameSource, arm: ArmStill,
                 questions: Callable[[Node], Sequence[Question]], sink: Callable[[dict], None],
                 grants_path=None, approve: Callable[[str, str], bool] | None = None,
                 boundary_budget_s: float = 20.0, hold_limit_s: float = 30.0, poll_s: float = 0.05,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 wall: Callable[[], float] = time.time, manager=EpisodeManager, **em_kw) -> None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} — {MODES} 중 하나")
        if not 0 < boundary_budget_s < hold_limit_s:
            raise ValueError(f"boundary_budget_s {boundary_budget_s} 는 0 보다 크고 hold_limit_s {hold_limit_s} 보다 작아야 한다")
        if not poll_s > 0:
            raise ValueError(f"poll_s {poll_s} 는 0 보다 커야 한다")
        if (mode == RECORD) != (approve is not None) or (mode != RECORD and grants_path is None):
            raise ValueError("record 는 바깥 approve 를, propose · auto 는 grants_path 를 준다(섞지 않는다)")
        self.mode, self.judge, self.frames, self.arm, self.questions, self.sink = mode, judge, frames, arm, questions, sink
        self.budget, self.hold, self.poll = boundary_budget_s, hold_limit_s, poll_s
        self.clock, self.sleep, self.wall = clock, sleep, wall
        if approve is not None: # record(위 검사) — 기존 흐름 그대로 — auto 를 끄지 않고 승인도 바깥 것
            self.em = manager(ep, executor, approve=approve, **em_kw)
        else:                   # 모든 노드가 경계가 되게 auto 를 끈 사본. approve 는 step() 안에서 막지 않고 바로 답한다
            every = replace(ep, nodes=tuple(replace(n, auto=False) for n in ep.nodes))
            self.em = manager(every, executor, approve=lambda what, why: what == self._go, **em_kw)
            self.grants = Grants(grants_path, self.em.episode_id, wall)
        self.seq = 0
        self.workers: list[threading.Thread] = []    # record 모드 판정 스레드 가운데 아직 도는 것(시험 · 종료 때 join)
        self._judging: threading.Thread | None = None    # propose · auto 의 마지막 판정 스레드
        self._go: str | None = None                  # 지금 step() 이 돌려도 되는 노드 id
        self._next: Decision | None = None           # 앞 경계의 advance — 다음 노드 승인을 기다린다
        self._since = wall()                         # node 승인 줄은 이 뒤(그 제안 뒤)에 쓰인 것만
        self._t_b = clock()                          # 지금 경계가 시작한 시각
        self._q: queue.Queue = queue.Queue()
        self._dropped = 0

    # ---------------------------------------------------------------- 공개
    def run(self) -> str:
        """끝 · handoff · 정지까지. 돌려주는 값 = SUCCESS · FAILURE · STOPPED · HANDED(record 모드는 승인 없으면 READY)."""
        idx = self.em.index
        while (out := self.step()) == READY:
            if self.mode == RECORD and self.em.index == idx:    # 바깥 승인이 없다 — 돌지 않고 돌려준다
                break
            idx = self.em.index
        return out

    def stop(self, reason: str = "operator stop") -> str:
        """다른 스레드에서 불러도 된다 — 판정 대기 중에도 바로 정지한다. 안전 정지가 먼저, grant 종료는 그 뒤."""
        try:
            return self.em.stop(reason)
        finally:
            self._end()

    def step(self) -> str:
        """노드 하나와 그 뒤 경계 하나. 예외가 나면 안전 정지(도는 중이면) → grant 종료 → 기록 뒤 다시 올린다."""
        try:
            return self._step()
        except BaseException as exc:
            try:
                if self.em.status == RUNNING:
                    self.em.stop("runner error")
            finally:
                self._go = None
                self._end()
                self._emit(self._rec(None, {}), Decision(HANDOFF, f"러너 오류 {type(exc).__name__}: {exc}", self.seq),
                           event="error")
            raise

    def _step(self) -> str:
        em = self.em
        node, idx = em.node, em.index
        if em.status in _DONE or node is None:       # 끝났다 — em.step() 은 움직이지 않고 상태만 돌려준다
            return em.step()
        if self.mode != RECORD:
            if self._next is None:                   # 시작 · handoff 뒤 재개 — 판정할 전후가 없다. 승인만 묻는다
                self._t_b, self._since = self.clock(), self.wall()
                if not self._emit(self._rec(None, {}), Decision("none", "시작", self.seq), event="start", next=node.id):
                    return self._act(self._rec(None, {}), Decision(HANDOFF, "기록 실패 — 시작하지 않는다", self.seq),
                                     event="no_log", next=node.id)
            if not self._authorised(node):
                return self._act(self._rec(None, {}), Decision(HANDOFF, f"{node.id} 승인 없음", self.seq),
                                 event="no_grant", next=node.id)
            self._go = node.id
        errs: list = []
        before = self._frame(self.clock() - BEFORE_MAX_AGE_S, errs)
        try:
            status = em.step()
        finally:
            self._go = None
        self.seq += 1
        self._t_b, self._next = self.clock(), None
        ran = em.index != idx or status == SUCCESS
        signal = {"node": node.id, "ok": ran and em.runs.get(node.id, {}).get("state") == "done",
                  "code": em.last.get("code"), "reason": em.last.get("reason") or "",
                  "results": copy.deepcopy(em.last.get("results", []))}
        rec = self._rec(node.id, signal)
        rec["errors"] += errs
        if status in (FAILURE, STOPPED):             # 안전 정지 · 실패 — VLM 을 거치지 않는다
            return self._act(rec, Decision(STOP, f"{status}: {signal['reason']}", self.seq), stopped=True)
        if self.mode == RECORD:
            return self._record(node, rec, before, signal, ran)
        if not ran:                                  # index 가 안 움직였다 — 복구 대기 · 확인 대기 · 승인 없음
            why = "복구 대기" if em.pending is not None else "확인 대기" if em.waiting is not None else "실행 안 됨"
            return self._act(rec, Decision(HANDOFF, f"{why}: {signal['reason']}", self.seq))
        return self._act(rec, self._boundary(node, rec, before, signal))

    # ---------------------------------------------------------------- 경계
    def _boundary(self, node: Node, rec: dict, before, signal: dict) -> Decision:
        seq, deadline, errs = self.seq, self._t_b + self.budget, rec["errors"]
        hand = lambda why: Decision(HANDOFF, why, seq)   # noqa: E731
        while not self._still(errs):
            if self.em.status == STOPPED:
                return Decision(STOP, "정지됨", seq)
            if self.clock() >= deadline:
                return hand("팔이 멈추지 않았다")
            self.sleep(self.poll)
        t_still = self.clock()                       # 뒤 사진은 팔이 멈춘 것을 확인한 뒤에 찍힌 것만
        qs = self._questions(node, rec)
        if not qs:
            return hand("질문 오류" if qs is None else "질문 없음 — VLM 없이 진행하지 않는다")
        # 러너 신호가 실패면 진행은 없다 — 놓기 실패 뒤 이어 가기 · confirm 으로 되돌아가기도 사람이 본다
        allowed = ACTIONS if signal["ok"] else (HANDOFF, STOP)
        d = hand("판정 없음")
        for look in (0, 1):
            after = self._frame(t_still if look == 0 else self.clock(), errs, deadline)
            if before is None or after is None:
                return hand("사진 없음")
            if self._judging is not None:
                self._judging.join(self.poll)        # 답을 막 넣은 스레드가 끝날 틈
                if self._judging.is_alive():
                    return hand("이전 판정이 아직 끝나지 않았다")
            rec["stamp"] = {"before": before[1], "after": after[1]}
            self._judging = threading.Thread(
                target=lambda lk=look, al=allowed, im=(before[0], after[0]):
                self._q.put((seq, lk, *self._judge(seq, list(im), qs, signal, al))), daemon=True)
            self._judging.start()
            rec["answers"], d = self._await(seq, look, deadline)
            rec["looks"] = look
            if d.action != LOOK_AGAIN:
                break
            allowed = tuple(a for a in allowed if a != LOOK_AGAIN)   # 경계당 1회 — 두 번째 요청은 _judge 가 handoff 로
        if not signal["ok"] and d.action in (ADVANCE, LOOK_AGAIN):  # _judge 의 allowed 검사와 겹치는 마지막 막이
            return hand(f"러너 신호 실패 — {d.action} 불가")
        return d

    def _questions(self, node: Node, rec: dict) -> tuple | None:
        """질문 목록(빈 것 포함). questions() 가 예외면 None — errors 에 남긴다."""
        try:
            qs = tuple(self.questions(node))
            rec["questions"] = [q.text for q in qs]
            return qs
        except Exception as exc:    # noqa: BLE001
            _log.exception("questions(%s)", node.id)
            _err(rec["errors"], "질문 오류", exc)
            return None

    def _judge(self, seq: int, images: list, qs: Sequence[Question], signal: Mapping,
               allowed: Sequence[str]) -> tuple[list, Decision]:
        """판정 스레드에서 돈다 — 1단(사진 답) → 어긋남 검사 → 2단(결정). 어떤 오류도 handoff."""
        answers: list = []
        try:
            answers = [dict(a) for a in self.judge.ask(images, [q.text for q in qs])]
            photo_ok = all(p == q.ok for p, q in zip(picks(answers, len(qs)), qs))
            if photo_ok != bool(signal["ok"]):
                return answers, Decision(HANDOFF, f"어긋남 — 러너 {'성공' if signal['ok'] else '실패'} · 사진 "
                                                  f"{'성공' if photo_ok else '실패'}", seq)
            action, reason = self.judge.decide(answers, signal, allowed)
            if action not in allowed:
                return answers, Decision(HANDOFF, f"허용 밖 결정 {action!r}", seq)
            return answers, Decision(action, str(reason), seq)
        except Exception as exc:    # noqa: BLE001 — 판정기 오류가 러너를 죽이지 않게
            return answers, Decision(HANDOFF, f"판정 오류 {type(exc).__name__}: {exc}", seq)

    def _await(self, seq: int, look: int, deadline: float) -> tuple[list, Decision]:
        """마감은 러너가 잰다. seq · look 이 다른 답과 마감 뒤에 온 답은 버린다."""
        while self.em.status != STOPPED:
            try:
                got = self._q.get(timeout=self.poll)
            except queue.Empty:
                got = None
            late = self.clock() > deadline
            if got is not None:
                s, lk, answers, d = got
                if (s, lk) == (seq, look) and d.seq == seq and not late:
                    return answers, d
                self._dropped += 1
            if late:
                return [], Decision(HANDOFF, "VLM 마감 초과", seq)
        return [], Decision(STOP, "정지됨", seq)

    def _record(self, node: Node, rec: dict, before, signal: dict, ran: bool) -> str:
        """record 모드 — 판정은 스레드에서 기록만 한다. 진행을 막거나 바꾸지 않는다(뒤 사진도 스레드가 기다린다)."""
        status, t_b, seq, errs = self.em.status, self._t_b, self.seq, rec["errors"]
        qs = (self._questions(node, rec) or ()) if ran else ()
        if before is None or not qs:
            self._emit(rec, Decision("unjudged", "판정 안 함(실행 안 됨 · 질문 없음 · 앞 사진 없음)", seq))
            return status
        rec["still"] = self._still(errs)

        def work() -> None:
            after = self._frame(t_b, errs, t_b + self.budget)
            if after is None:
                self._emit(rec, Decision("unjudged", "판정 안 함(뒤 사진 없음)", seq), t_b=t_b)
                return
            rec["stamp"] = {"before": before[1], "after": after[1]}
            rec["answers"], d = self._judge(seq, [before[0], after[0]], qs, signal, ACTIONS)
            rec["late"] = self.clock() - t_b > self.budget
            self._emit(rec, d, t_b=t_b)

        t = threading.Thread(target=work, daemon=True)
        self.workers = [w for w in self.workers if w.is_alive()] + [t]
        t.start()
        return status

    # ---------------------------------------------------------------- 권한 · 기록
    def _authorised(self, node: Node) -> bool:
        """hold_limit_s 안에 사람 승인을 기다린다. auto: 살아 있는 에피소드 승인(confirm 노드는 제외 — 사람이 그 노드를 승인)."""
        deadline = self._t_b + self.hold
        while self.em.status != STOPPED:
            if self.mode == AUTO and node.type != "confirm" and self.grants.episode() is not None:
                return True
            if self.grants.node(node.id, ADVANCE, self._since) is not None:
                return True
            if self.clock() >= deadline:
                return False
            self.sleep(self.poll)
        return False

    def _act(self, rec: dict, d: Decision, stopped: bool = False, **extra) -> str:
        em = self.em
        try:
            if d.action == STOP and not stopped and em.status not in _DONE:
                em.stop(f"vlm stop: {d.reason}")     # 정지가 먼저 — grant 종료 · 기록은 그 뒤
        finally:
            if d.action == ADVANCE:
                self._since = self.wall()            # 제안 기록보다 앞 — 기록을 보고 쓴 사람 줄이 이 뒤가 되게
            else:
                self._end()
        if self._emit(rec, d, **extra) and d.action == ADVANCE:
            self._next = d
            return em.status                         # READY 또는 마지막 경계의 SUCCESS
        self._end()                                  # 기록에 실패한 advance 는 진행하지 않는다 — handoff
        return em.status if d.action == STOP or em.status == STOPPED else HANDED

    def _end(self) -> None:
        self._next = None
        if self.mode != RECORD:
            self.grants.end()

    def _rec(self, node_id: str | None, signal: Mapping) -> dict:
        return {"seq": self.seq, "episode_id": self.em.episode_id, "mode": self.mode, "node": node_id, "signal": dict(signal),
                "questions": [], "answers": [], "stamp": {}, "looks": 0, "errors": []}

    def _emit(self, rec: dict, d: Decision, t_b: float | None = None, **extra) -> bool:
        """기록 한 줄 — 성공 여부를 돌려준다(propose · auto 는 실패하면 진행하지 않는다, record 는 로그만)."""
        try:
            row = {**rec, **extra, "errors": list(rec["errors"]), "decision": d.action, "reason": d.reason,
                   "status": self.em.status, "elapsed_s": round(self.clock() - (self._t_b if t_b is None else t_b), 3),
                   "discarded": self._dropped}
            self._dropped = 0
            self.sink(row)
            return True
        except Exception:    # noqa: BLE001
            _log.exception("sink")
            return False

    def _frame(self, after: float, errors: list, deadline: float = -math.inf):
        """after 이후에 찍힌 (jpg, stamp). 아직 없으면 deadline 까지 다시 묻는다. 없음 · 불량 · 예외는 None(errors 에 사유)."""
        try:
            while (f := self.frames.frame(after)) is None:
                if self.clock() >= deadline or self.em.status == STOPPED:
                    raise LookupError(f"{after:.3f} 이후 사진이 없다")
                self.sleep(self.poll)
            jpg, stamp = f[0], float(f[1])
            if not isinstance(jpg, (bytes, bytearray)) or not jpg or not math.isfinite(stamp) or stamp < after:
                raise ValueError(f"stamp {stamp} · 요청 {after:.3f} · jpg {type(jpg).__name__}")
            return bytes(jpg), stamp
        except Exception as exc:    # noqa: BLE001
            _log.warning("frame: %s", exc)
            _err(errors, "사진 없음", exc)
            return None

    def _still(self, errors: list) -> bool:
        try:
            return bool(self.arm.arm_still())
        except Exception as exc:    # noqa: BLE001
            _log.exception("arm_still")
            _err(errors, "팔 정지 확인 오류", exc)
            return False

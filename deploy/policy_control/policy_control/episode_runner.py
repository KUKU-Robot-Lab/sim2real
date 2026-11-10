"""EpisodeManager — 에피소드 YAML 순서대로 노드를 실행하고(구분 실행 step · 연속 실행 run), 실패면 복구한다. 순수(실행기는 주입).

10.04 사용자: "테스트할 때는 한 개씩 명령 내리면서 구분동작, 최종 목표는 한 번에(중간에 사용자 개입 없이)".
  · step()  노드 하나 — 승인 hook 이 그 노드마다 묻는다(상황판 [다음 노드]).
  · run()   끝까지 — 승인은 시작에 한 번(상황판 [연속 실행]). 실패면 정해진 복구(재시도 상한 · 홈 checkpoint 로
            되돌아가기)를 스스로 하고, 안전 규칙 · 복구 수단 없음이면 safe stop. 멈추면 승인이 끝난다.
실행기 계약(Executor):
  snapshot(node, world)            → (NodeResult, {물체: (pos, quat)})
  run_policies(node, plans, world) → [NodeResult, …]   plans = [PolicyPlan] (병렬이면 둘 이상)
  run_trajectory(node, traj, world)→ NodeResult
  run_hand(node, world)            → NodeResult           ★10.17 경로 B — 손 펴기 · 닿을 때까지 닫고 쥔 판정(팔 제자리)
  prepare(step, plans, world)      → NodeResult           refresh_perception · open_hand
★10.17 confirm 노드(경로 B "확인 두 번"): 로봇을 움직이지 않고, 연속 실행 중에도 그 노드 이름의 승인을 따로 받는다 —
  승인이 없으면 거기서 멈추고(waiting) 상황판 [다음] 으로 그 노드를 승인하면 지나간다.
  safe_stop(reason)                → None
  joints()                         → dict                 checkpoint 기록용(없으면 {})
로그(JSONL, 가이드 1-10): timestamp · episode_id · state_id · executor_type · policy_name · start/end/duration ·
  left/right_hand_object · pose_state · task_flags · termination_reason · event · result · next.
"""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Protocol

from policy_control.episode_failure import (F, FailureCode, FailureDetector, NodeResult, Recovery, RecoveryManager,
                                            ROLLBACK, SAFE_STOP, SAFETY)
from policy_control.episode_flow import flow_of
from policy_control.episode_spec import EMPTY, HAND_KEYS, PLACED_SNAPSHOT_TRIES, Episode, Node, region_miss
from policy_control.episode_world import World

SUCCESS, FAILURE, RUNNING, READY, STOPPED = "SUCCESS", "FAILURE", "RUNNING", "READY", "STOPPED"


@dataclass(frozen=True)
class PolicyPlan:
    """정책 노드 하나를 돌릴 때 실행기에 넘기는 것 — 역할 · 등록부 정책 · 팔 · 물체 · 홀더 · SETTING 점."""
    role: str
    kind: str
    side: str
    policy: str | None
    target_object: str | None
    source_object: str | None
    target_holder: str | None
    holder_id: int | None
    setting: tuple | None
    object_pose: tuple | None        # snapshot 이 기록한 물체 자세(pos, quat) — 집기 전 정지 컵
    policy_object: str | None = None  # ★10.12 다물체 aglt 정책의 물체 이름 → 노드 파라미터 object(reset 전)
    max_s: float = 15.0               # ★10.17 노드 max_episode_s(경로 C 는 최악 경로로 늘어난다) — 실행기 대기 한도의 바탕
    holder_tol_m: float | None = None  # ★10.29 카드가 이 놓기 job 에 넓힌 홀더 허용 → 노드 파라미터 holder_tol_m(reset 전)
    lift_guard: tuple | None = None    # ★11.08 (guard_mm, hstop_m, x, y) — 피할 물체 x · y 는 snapshot(base), 없으면 nan → 노드 reset 거부


def _lift_guard(j, world) -> tuple | None:
    if j.lift_guard is None:
        return None
    name, g, h = j.lift_guard
    loc = world.objects.get(name) or {}
    # 분홍 x · y 는 보정하지 않은 FP++ snapshot 원값(cup_obs_offset 빼지 않음): 실기 mesh_clear(logs/analysis/lift_sweep_1108)도
    # 원값으로 쟀고, 그 틈으로 본 밀림 거리(약 15~17 mm)가 bag 의 분홍 실제 이동과 맞는다
    xy = loc["pose"][0][:2] if loc.get("pose") is not None and loc.get("at", "table") == "table" else (float("nan"),) * 2
    return (g, h, float(xy[0]), float(xy[1]))


def _node_max_s(b) -> float:
    """정책 노드 한 번의 시간 한도 — 미션 생성기와 같은 식(approach_grasp.episode_s · ★11.05 학생 = sim 평가 에피소드 길이)."""
    if getattr(b, "student", None) and b.kind == "aglt":     # 놓기 학생(11.08)은 놓기 노드 한도 그대로
        from policy_control.student_chain import bundle_limits
        return bundle_limits(b.student)[1]
    if getattr(b, "approach_grasp", None) is None:
        return 15.0
    from policy_control.approach_grasp import episode_s     # 무거운 의존은 경로 C 에서만
    return episode_s(b.approach_grasp)


class Executor(Protocol):
    def snapshot(self, node: Node, world: World) -> tuple[NodeResult, dict]: ...
    def run_policies(self, node: Node, plans: list, world: World) -> list: ...
    def run_trajectory(self, node: Node, traj, world: World) -> NodeResult: ...
    def run_hand(self, node: Node, world: World) -> NodeResult: ...
    def prepare(self, step: str, plans: list, world: World) -> NodeResult: ...
    def safe_stop(self, reason: str) -> None: ...
    def joints(self) -> dict: ...


#: ★11.08 lanes 승인 · 설명 문구 — plan_carry 의 arms_clear 는 다른 팔을 실측 그대로 멈춘 팔로 본다
LANES_WARN = "두 팔 동시 진행: trajectory 는 한 팔씩, 정책 중인 팔과는 충돌 미검사"


def _cancelled(res: list) -> bool:
    """실행기가 정지(cancel)로 끊은 결과 — 실제 판정 실패와 구분한다."""
    return any(r.status == "aborted" and (r.reason.startswith("episode runner stop") or r.signals.get("guard_hold_cancelled"))
               for r in res)


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


@dataclass
class EpisodeManager:
    ep: Episode
    executor: Executor
    approve: Callable[[str, str], bool]                 # (무엇, 설명) → 허락. 연속 실행은 시작에 한 번만 부른다
    log: Callable[[dict], None] = lambda row: None
    episode_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    detector: FailureDetector = field(default_factory=FailureDetector)
    recovery: RecoveryManager = field(default_factory=RecoveryManager)
    world: World = field(default_factory=World)
    index: int = 0
    attempts: dict = field(default_factory=dict)        # 노드 id → 재시도 횟수
    checkpoints: dict = field(default_factory=dict)     # 이름 → {"world", "joints", "node", "at"}
    status: str = READY
    last: dict = field(default_factory=dict)            # 마지막 노드 결과 요약(상황판)
    history: list = field(default_factory=list)
    pending: tuple | None = None                        # 구분 실행: 승인을 기다리는 복구 (node, Recovery)
    runs: dict = field(default_factory=dict)            # 노드 id → {state, n(실행 횟수), s(마지막 걸린 시간), code} — 흐름 그림
    flow: dict = field(default_factory=dict)            # 흐름 그림의 고정 부분(episode_flow.flow_of)
    waiting: str | None = None                          # ★10.17 운영자 확인(confirm 노드)을 기다리는 노드 id
    place_failed_continued: list = field(default_factory=list)   # ★11.07 놓기 실패 뒤 다음 노드로 넘어간 노드 id(이 에피소드)
    #: ★10.17 지금 노드에 온 시각(wall) — 확인 노드 승인은 이 뒤에 쓰인 것만 받는다(리뷰: 서비스 실패로 남은 옛 승인 줄이
    #:   나중에 같은 노드에서 저절로 통과하지 않게, episode_runner_node 가 읽는다)
    node_since: float = field(default_factory=time.time)
    #: ★10.28 사람이 놓은 물체 다시 찍기(confirm 바로 뒤 snapshot objects) 실패 횟수 — 노드 id → 횟수
    placed_tries: dict = field(default_factory=dict)
    _auto: bool = False
    _chained: bool = False                              # ★11.09 auto 노드 이어 가기(승인만 생략)

    def __post_init__(self) -> None:
        if not self.flow:
            self.flow = flow_of(self.ep)
        for name, o in self.ep.objects.items():         # ★10.28 사람이 나중에 놓는 물체 — 처음 snapshot 이 찾지 않는다
            if o.get("at_start") == "away" and name not in self.world.objects:
                self.world = self.world.locate(name, "away")

    def _mark(self, node_id: str, state: str, **kw) -> None:
        prev = self.runs.get(node_id, {"n": 0})
        self.runs[node_id] = {**prev, "state": state, **kw}

    # ---------------------------------------------------------------- 공개
    @property
    def node(self) -> Node | None:
        return self.ep.nodes[self.index] if self.index < len(self.ep.nodes) else None

    def step(self) -> str:
        """노드 하나(구분 실행) — 그 노드를 승인 hook 에 묻는다. 돌아오는 값 = 에피소드 상태."""
        if self.status in (SUCCESS, FAILURE, STOPPED):
            return self.status
        self._auto = False
        if self.pending is not None:                    # 구분 실행 — 복구 동작도 한 번의 명령 · 승인
            node, rec = self.pending
            if not self.approve(f"recover:{node.id}:{rec.action}", rec.reason):
                return self._declined(node, f"recover:{node.id}:{rec.action}")
            self.pending = None
            self._do_recovery(node, rec)
        else:
            self._advance()
        return self._chain_auto()

    def _chain_auto(self) -> str:
        """★11.09 사용자 — 다음 노드가 auto 면 승인 없이 이어 간다(앞 노드가 READY 로 끝났을 때만 · 복구 대기 · 확인 · 정지면 멈춤)."""
        while (self.status == READY and self.pending is None and self.waiting is None
               and self.node is not None and self.node.type != "terminal" and self.node.auto):
            self._chained = True                        # 승인만 건너뛴다 — 실패 복구는 그대로 다음 명령(승인) 대기
            try:
                self._advance()
            finally:
                self._chained = False
        return self.status

    def run(self) -> str:
        """끝까지(연속 실행) — 승인은 여기서 한 번. 복구 동작도 이 승인 안이다(재시도 상한 · 안전 규칙은 그대로)."""
        if self.status in (SUCCESS, FAILURE, STOPPED):
            return self.status
        rest = " → ".join(n.id for n in self.ep.nodes[self.index:])
        lanes = [n for n in self.ep.nodes[self.index:] if n.type == "lanes"]
        warn = "".join(f" | {n.id}: {self._describe(n)}" for n in lanes)
        if not self.approve(f"episode:{self.ep.name}", f"연속 실행 — {rest}{warn}"):
            return self._declined(self.node, f"episode:{self.ep.name}")
        self._auto = True
        try:
            while self.status in (READY, RUNNING):
                self._advance()
                if self.waiting is not None:            # confirm 노드 — 연속 실행도 여기서 운영자 확인을 기다린다
                    break
        finally:
            self._auto = False
        return self.status

    def next_action(self) -> str | None:
        """다음 step() 이 승인을 물을 이름 — 상황판 확인 입력과 같다(없으면 None)."""
        if self.status in (SUCCESS, FAILURE, STOPPED):
            return None
        if self.pending is not None:
            return f"recover:{self.pending[0].id}:{self.pending[1].action}"
        node = self.node
        return None if node is None or node.type == "terminal" else node.id

    def stop(self, reason: str = "operator stop") -> str:
        self.status = STOPPED      # ★11.08 safe_stop 앞 — 그 사이 돌아온 aborted 를 줄 · 노드가 실패가 아니라 정지로 본다
        self.executor.safe_stop(reason)
        if self.node is not None and self.runs.get(self.node.id, {}).get("state") == "running":
            self._mark(self.node.id, "stopped")
        self._log("stop", self.node, termination_reason=reason)
        return self.status

    def view(self) -> dict:
        return {"episode": self.ep.name, "episode_id": self.episode_id, "status": self.status, "index": self.index,
                "node": None if self.node is None else self.node.id, "world": self.world.as_dict(),
                "nodes": [{"id": n.id, "type": n.type, "name": n.name or ",".join(j.role for j in n.jobs),
                           "state": ("done" if i < self.index else "current" if i == self.index else "pending")}
                          for i, n in enumerate(self.ep.nodes)],
                "attempts": dict(self.attempts), "checkpoints": sorted(self.checkpoints), "last": dict(self.last),
                "pending": None if self.pending is None else {"node": self.pending[0].id, "action": self.pending[1].action,
                                                              "reason": self.pending[1].reason},
                "next_action": self.next_action(), "flow": self.flow, "runs": {k: dict(v) for k, v in self.runs.items()},
                "confirm": (None if self.node is None or self.status not in (READY,)
                            else {"node": self.node.id, "text": self.node.text} if self.node.type == "confirm"
                            else None),
                "place_failed_continued": list(self.place_failed_continued), "history": self.history[-20:]}

    # ---------------------------------------------------------------- 내부
    def _advance(self) -> str:
        node = self.node
        if node is None:
            self.status = SUCCESS
            return self.status
        if node.type == "terminal":
            self.status = SUCCESS if (node.result or SUCCESS) == SUCCESS else FAILURE
            self._mark(node.id, "done" if self.status == SUCCESS else "failed", n=1)
            self._log("terminal", node, result=self.status, place_failed_continued=list(self.place_failed_continued))
            self._next_node()
            return self.status
        if node.type == "confirm":
            return self._confirm(node)
        if not (self._auto or self._chained) and not self.approve(node.id, self._describe(node)):
            return self._declined(node, node.id)
        self.status = RUNNING
        t0, start = time.monotonic(), _now()
        self._mark(node.id, "running", n=self.runs.get(node.id, {"n": 0})["n"] + 1)
        self._log("enter", node, start_time=start)
        before = self.world
        self.world = self.world.running() if node.type in ("policy", "parallel_policy", "lanes") else self.world
        if node.type == "lanes":
            results, code, who, why = self._run_lanes(node)
            located, before = {}, self.world     # 줄 항목이 끝낸 것(쥔 컵)은 실패 · 정지에도 되돌리지 않는다 — 실제 상태
        else:
            results, located = self._execute(node, before)
            why = ""
        if self.status == STOPPED:                       # 실행 중에 stop — 결과는 남기고 더 나가지 않는다
            self.world = before
            self._mark(node.id, "stopped", s=round(time.monotonic() - t0, 2))
            self._log("stopped", node, start_time=start, end_time=_now(),
                      termination_reason="; ".join(r.reason for r in results if r.reason))
            return self.status
        if node.type != "lanes":
            code, who = self.detector.evaluate(self.ep, node, results)
        dur = time.monotonic() - t0
        reason = why or "; ".join(r.reason for r in results if r.reason)
        self.last = {"node": node.id, "code": code.value, "who": who, "reason": reason, "duration_s": round(dur, 2),
                     "results": [{"role": r.role, "status": r.status, "reason": r.reason, "signals": dict(r.signals)}
                                 for r in results]}
        if code is F.NONE:
            self._mark(node.id, "done", s=round(dur, 2), code="COMPLETED")
            self.world = self._after_success(node, before, located)   # lanes: 줄이 쌓은 World 에 lanes expect
            if node.checkpoint:
                self.checkpoints[node.checkpoint] = {"world": self.world, "joints": self.executor.joints(), "node": node.id,
                                                     "at": _now()}
            self._next_node()
            self._log("exit", node, start_time=start, end_time=_now(), duration=round(dur, 3), result="COMPLETED",
                      termination_reason=reason, next=None if self.node is None else self.node.id,
                      signals=self._signal_rows(results))
            self.status = READY if self.node is not None else SUCCESS
            return self._close_terminal()
        self.world = before
        self._mark(node.id, "failed", s=round(dur, 2), code=code.value)
        self._log("failure", node, start_time=start, end_time=_now(), duration=round(dur, 3), result=code.value,
                  termination_reason=reason, failed_role=who, signals=self._signal_rows(results))
        if self._placed_retry(node, code):
            return self._back_to_confirm(node, code, reason)
        return self._recover(node, code)

    def _run_lanes(self, node: Node) -> tuple[list, FailureCode, str, str]:
        """★11.08 lanes — 줄마다 자기 항목을 차례로(스레드 하나씩), 두 줄은 동시에. 항목이 끝나면 그 expect · 집은 물체를
        World 에 바로 적용한다(락). 한 줄 항목이 실패하면 그 자리에서 실행기 safe_stop 으로 다른 줄도 멈춘다. 돌아오는 것 =
        (모든 항목 결과, 처음 실패 코드, 누구, 그 항목 사유). 항목은 enter · exit · failure · stopped 를 lane · group 과 같이 jsonl 에 남긴다."""
        lanes: dict = {}
        for it in self.ep.nodes:
            if it.group == node.id:
                lanes.setdefault(it.lane, []).append(it)
                self.runs.setdefault(it.id, {"n": 0})
                self.runs[it.id] = {**self.runs[it.id], "lane": it.lane}   # 상황판 view 가 도는 동안 키를 늘리지 않게 미리
        lock, traj_lock, results, fail = threading.Lock(), threading.Lock(), [], []
        busy: dict = {}                                   # 줄 → 지금 정책을 돌리는 항목 id

        def other_policy(side: str) -> str:
            return ", ".join(f"{s}:{i}" for s, i in busy.items() if s != side)

        def run_item(it: Node, w: World) -> tuple[list, dict, dict]:
            """궤적은 계획부터 재생 끝까지 traj_lock(★11.08 리뷰 1a) — 다른 팔 궤적은 기다리고, 정책과는 겹친다."""
            if it.type != "trajectory":
                if it.type in ("policy", "parallel_policy"):   # ★11.08 재검토 — 손 · 스냅샷은 "정책 진행 중" 이 아니다
                    with lock:
                        busy[it.lane] = it.id
                try:
                    res, located = self._execute(it, w)
                finally:
                    with lock:
                        busy.pop(it.lane, None)
                return res, located, {}
            q0 = time.monotonic()
            with traj_lock:
                extra = {"traj_wait_s": round(time.monotonic() - q0, 3)}
                with lock:
                    if fail or self.status == STOPPED:
                        return [NodeResult("aborted", "episode runner stop — 궤적 차례를 기다리는 중 정지")], {}, extra
                    w = self.world                    # 기다리는 동안 다른 줄이 바꾼 World
                    other = other_policy(it.lane)
                    if other:                         # ★11.08 리뷰 1b — plan_carry 는 이 팔을 멈춘 팔로 보고 검사한다
                        extra["other_arm_policy"] = f"다른 팔 정책 진행 중({other}) — 충돌 미검사"
                        self.runs[it.id] = {**self.runs[it.id], "note": extra["other_arm_policy"]}
                    self._log("plan", it, lane=it.lane, group=node.id, **extra)
                res, located = self._execute(it, w)
            if extra.get("other_arm_policy"):
                res = [dataclasses.replace(r, reason="; ".join(x for x in (r.reason, extra["other_arm_policy"]) if x))
                       for r in res]
            return res, located, extra

        def lane(items: list) -> None:
            for it in items:
                with lock:
                    if fail or self.status == STOPPED:
                        return
                    w = self.world
                    self._mark(it.id, "running", n=self.runs[it.id]["n"] + 1)
                    start, t0 = _now(), time.monotonic()
                    self._log("enter", it, start_time=start, lane=it.lane, group=node.id)
                res, located, extra = run_item(it, w)
                code, who = self.detector.evaluate(self.ep, it, res)
                dur = round(time.monotonic() - t0, 3)
                reason = "; ".join(r.reason for r in res if r.reason)
                row = dict(start_time=start, end_time=_now(), duration=dur, lane=it.lane, group=node.id,
                           signals=self._signal_rows(res), **extra)
                with lock:
                    results.extend(res)
                    if code is F.NONE:
                        self.world = self._after_success(it, self.world, located)
                        self._mark(it.id, "done", s=round(dur, 2), code="COMPLETED")
                        self._log("exit", it, result="COMPLETED", termination_reason=reason, **row)
                        continue
                    # ★11.08 리뷰 2 — 정지(다른 줄 실패 · 운영자 stop)가 끊은 항목은 실패가 아니라 정지, safe_stop 도 다시 안 부른다
                    if self.status == STOPPED or (fail and _cancelled(res)):
                        self._mark(it.id, "stopped", s=round(dur, 2))
                        self._log("stopped", it, termination_reason=reason or "다른 줄 정지", code=code.value, **row)
                        return
                    first = not fail                  # ★11.08 리뷰 3 — 실패는 모두 모은다(정지 뒤 실제 판정 실패 포함)
                    fail.append((code, f"{it.id}:{who or it.lane}", f"{it.id}({it.lane}) {code.value}: {reason or '-'}"))
                    self._mark(it.id, "failed", s=round(dur, 2), code=code.value)
                    self._log("failure", it, result=code.value, termination_reason=reason, failed_role=who, **row)
                if first:                             # 한 팔이 실패한 채 다른 팔만 움직이지 않게 — 다른 줄을 지금 멈춘다
                    self.executor.safe_stop(f"{it.id}({it.lane}) {code.value} — 다른 줄도 멈춘다")
                return

        def guarded(items: list) -> None:
            try:
                lane(items)
            except Exception as exc:                  # noqa: BLE001 — 스레드가 조용히 죽으면 그 줄이 성공처럼 보인다
                with lock:
                    first = not fail
                    fail.append((F.UNKNOWN, f"{items[0].lane}", f"{items[0].lane} 줄 오류 {type(exc).__name__}: {exc}"))
                if first:
                    self.executor.safe_stop(f"{items[0].lane} 줄 오류 {type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=guarded, args=(items,), daemon=True, name=f"lane_{side}")
                   for side, items in lanes.items()]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if not fail:
            return results, F.NONE, "", ""
        # 안전 코드(SAFETY: 가드 · 제어기 · 관측 …)가 있으면 그것이 노드 코드 — 나머지 사유는 뒤에 붙인다
        top = next((f for f in fail if f[0] in SAFETY), fail[0])
        return results, top[0], top[1], " | ".join([top[2], *(f[2] for f in fail if f is not top)])

    def _placed_retry(self, node: Node, code: FailureCode) -> bool:
        """★10.28 사람이 놓은 물체 다시 찍기(confirm 바로 뒤 snapshot objects)를 못 찾음 · 상자 밖으로 실패 — 상한 안이면 참."""
        if (node.type != "snapshot" or not node.objects or code is not F.TARGET_NOT_FOUND or self.index < 1
                or self.ep.nodes[self.index - 1].type != "confirm"):
            return False
        self.placed_tries[node.id] = self.placed_tries.get(node.id, 0) + 1
        return self.placed_tries[node.id] < PLACED_SNAPSHOT_TRIES

    def _back_to_confirm(self, node: Node, code: FailureCode, reason: str) -> str:
        """★10.28 로봇은 움직이지 않았다 — 바로 앞 confirm 으로 돌아가 운영자가 다시 놓고 확인하기를 기다린다(연속 실행도
        거기서 멈춘다). 그 노드에 돌아온 시각을 새로 찍어, 앞서 쓴 확인 승인이 저절로 다시 통과하지 않게 한다."""
        self.index -= 1
        self.node_since = time.time()
        back = self.ep.nodes[self.index]
        self.waiting = back.id
        self._mark(back.id, "waiting")
        self.last = {**self.last, "node": back.id, "code": "waiting_confirm",
                     "reason": f"{node.id} 실패 {self.placed_tries[node.id]}/{PLACED_SNAPSHOT_TRIES}({code.value}: {reason})"
                               f" — 다시 놓고 확인: {back.text}"}
        self._log("back_to_confirm", node, result=code.value, termination_reason=reason, next=back.id,
                  tries=self.placed_tries[node.id])
        self.status = READY
        return self.status

    def _execute(self, node: Node, world: World) -> tuple[list, dict]:
        try:
            if node.type == "snapshot":
                res, located = self._snapshot(node, world)
                return [res], located
            if node.type == "trajectory":
                return [self.executor.run_trajectory(node, self.ep.trajectories[node.name], world)], {}
            if node.type == "hand":
                return [self.executor.run_hand(node, world)], {}
            return list(self.executor.run_policies(node, self._plans(node, world), world)), {}
        except Exception as exc:                                       # 실행기 예외 = 관측 · 컨트롤러 이상 → safe stop
            return [NodeResult("error", f"{type(exc).__name__}: {exc}")], {}

    def _snapshot(self, node: Node, world: World) -> tuple[NodeResult, dict]:
        """실행기 snapshot + ★10.28 물체 region(검증 상자) — 밖이면 거부한다(로봇은 움직이지 않았다 · 기록하지 않는다)."""
        res, located = self.executor.snapshot(node, world)
        if res.status != "completed":
            return res, located
        miss = [f"{name} {why}" for name, pose in located.items()
                for why in (region_miss(self.ep.objects.get(name) or {}, pose[0]),) if why]
        if miss:
            return NodeResult("refused", "cup outside region — " + "; ".join(miss), res.duration_s), {}
        return res, located

    def _plans(self, node: Node, world: World) -> list:
        out = []
        for j in node.jobs:
            b = self.ep.policies[j.role]
            obj = j.target_object or j.source_object
            loc = world.objects.get(obj) if obj else None
            pose = tuple(loc["pose"]) if loc and loc.get("pose") is not None else None
            side = b.side if b.side != "both" else None
            out.append(PolicyPlan(role=j.role, kind=b.kind, side=b.side, policy=b.policy, target_object=j.target_object,
                                  source_object=j.source_object, target_holder=j.target_holder,
                                  holder_id=self.ep.holders.get(j.target_holder) if j.target_holder else None,
                                  setting=self.ep.setting.get(side) if side else None, object_pose=pose,
                                  policy_object=j.policy_object, max_s=_node_max_s(b), holder_tol_m=j.holder_tol_m,
                                  lift_guard=_lift_guard(j, world)))
        return out

    def _after_success(self, node: Node, before: World, located: Mapping) -> World:
        w = before.apply(node.expect)
        for name, pose in located.items():                          # snapshot — 물체는 테이블 위, 자세를 기록
            w = w.locate(name, "table", pose[0])
            w = _with_pose(w, name, pose)
        if node.type == "hand" and node.action == "grasp":          # 사람이 댄 컵을 쥐었다 — 물체는 그 손에
            for side in node.sides:
                if w.hand(side) != EMPTY:
                    w = w.locate(w.hand(side), f"{side}_hand")
        for j in node.jobs:
            b = self.ep.policies[j.role]
            if b.kind == "aglt" and j.target_object and w.hand(b.side) == j.target_object:
                w = w.locate(j.target_object, f"{b.side}_hand")
            if b.kind == "place" and j.source_object and j.target_holder and w.hand(b.side) == EMPTY:
                w = w.locate(j.source_object, f"holder:{j.target_holder}")
        return w

    def _recover(self, node: Node, code: FailureCode) -> str:
        if node.type == "lanes":                        # ★11.08 줄 노드 — 재시도 · 되돌아가기 · continue_on_fail 없이 정지
            return self._fail(node, code, f"{self.last.get('reason') or code.value} — 줄 노드는 복구 없이 정지(다른 줄도 멈춤)")
        n = self.attempts.get(node.id, 0)
        rec: Recovery = self.recovery.decide(self.ep, node, code, n, self.world, self.checkpoints)
        self._log("recovery", node, result=rec.action, termination_reason=rec.reason, checkpoint=rec.checkpoint)
        if rec.action == SAFE_STOP:
            # ★10.28 그 노드가 실패한 원인도 붙인다 — 콘솔 경고 줄은 last.reason 만 보여 "재시도 0/0 다 씀" 만으로는 무엇을
            #   고칠지 모른다(예: 순서 B 분홍이 자리 밖 → 다시 놓는다). last 는 바로 앞(step)에서 이 노드 결과로 채웠다.
            cause = str(self.last.get("reason") or "")
            reason = f"{rec.reason} · {cause}" if cause and cause not in rec.reason else rec.reason
            if self._continue_ok(node, code):
                return self._continue_after_fail(node, code, reason)
            return self._fail(node, code, reason)
        if not self._auto:                              # 구분 실행: 다음 명령(step)이 승인받고 복구한다
            self.pending = (node, rec)
            self._mark(node.id, "recovering", recovery=rec.action)
            self.status = READY
            self._log("recovery_pending", node, result=rec.action, termination_reason=rec.reason)
            return self.status
        return self._do_recovery(node, rec)

    def _do_recovery(self, node: Node, rec: Recovery) -> str:
        self.attempts[node.id] = self.attempts.get(node.id, 0) + 1
        plans = self._plans(node, self.world) if node.jobs else []
        for pre in rec.pre:
            res = self.executor.prepare(pre, plans, self.world)
            if res.status != "completed":
                code = F.CONTROLLER_ERROR if res.signals.get("pd_fault") else F.UNKNOWN
                return self._fail(node, code, f"복구 준비 {pre} 실패: {res.reason}")
        self._mark(node.id, "recovering", recovery=rec.action)
        if rec.action == ROLLBACK:
            cp = self.checkpoints[rec.checkpoint]
            back = self.ep.nodes[self.ep.node_index(cp["node"])]
            self._mark(back.id, "running", n=self.runs.get(back.id, {"n": 0})["n"] + 1)
            res = self.executor.run_trajectory(back, self.ep.trajectories[back.name], self.world)
            self._mark(back.id, "done" if res.status == "completed" else "failed", s=round(res.duration_s, 2))
            if res.status != "completed":
                return self._fail(node, F.POSE_MISMATCH, f"{rec.checkpoint} 로 못 돌아갔다: {res.reason}")
            self.world = World(pose=cp["world"].pose, hands=self.world.hands, flags=self.world.flags,
                               objects=self.world.objects)
            # 실패한 집기가 컵을 밀었을 수 있다 — 테이블 위 물체 자리를 다시 기록한다(10.04 사용자: 배치는 FP++ 로 기록)
            res, located = self._snapshot(Node(id=f"{node.id}:resnap", type="snapshot"), self.world)
            if res.status != "completed":
                return self._fail(node, F.TARGET_NOT_FOUND, f"되돌아간 뒤 배치 재기록 실패: {res.reason}")
            for name, pose in located.items():
                if (self.world.objects.get(name) or {}).get("at", "table") == "table":
                    self.world = _with_pose(self.world.locate(name, "table", pose[0]), name, pose)
            self._log("resnapshot", node, result="COMPLETED", objects=sorted(located))
        self.status = READY
        return self.status

    def _next_node(self) -> None:
        self.index += 1
        while self.node is not None and self.node.lane:   # ★11.08 줄 항목은 lanes 노드가 돌았다
            self.index += 1
        self.node_since = time.time()

    def _confirm(self, node: Node) -> str:
        """운영자 확인 — 로봇을 움직이지 않는다. 연속 실행이어도 이 노드 이름의 승인을 따로 받는다(사용자 "확인 두 번")."""
        if not self.approve(node.id, f"확인: {node.text}"):
            self.waiting = node.id
            self._mark(node.id, "waiting")
            self.last = {"node": node.id, "code": "waiting_confirm", "reason": f"운영자 확인을 기다린다 — {node.text}"}
            self._log("confirm_wait", node, termination_reason=node.text)
            self.status = READY
            return self.status
        self.waiting = None
        self._mark(node.id, "done", n=self.runs.get(node.id, {"n": 0})["n"] + 1, code="CONFIRMED")
        self.last = {"node": node.id, "code": "confirmed", "reason": node.text}
        self._next_node()
        self._log("confirm", node, result="CONFIRMED", termination_reason=node.text,
                  next=None if self.node is None else self.node.id)
        self.status = READY if self.node is not None else SUCCESS
        return self._close_terminal()

    def _continue_ok(self, node: Node, code: FailureCode) -> bool:
        """★11.07 continue_on_fail 에 든 kind 의 policy 노드 · 정책을 돌린 뒤의 POLICY_TIMEOUT · PLACE_FAILED 만(refused · 안전 코드 제외).
        손이 컵을 못 놓았거나(hand_empty False) 못 빠진(withdraw_timeout) 놓기는 손이 컵에 걸려 있어 그대로 복귀하면 위험하다 —
        기존대로 실패(episode_failure _one 의 place 절). last 는 바로 앞(step)에서 이 노드 결과로 채웠다."""
        sigs = [r["signals"] for r in self.last.get("results", [])]
        # ★11.11 래치 전 손 폄 종료는 hand_empty None(모름) — 컵이 홀더 밖이면 손에 걸렸을 수 있으니 걸림으로 본다
        unknown = any(s.get("outcome") == "opened_before_latch" and s.get("cup_in_holder") is False for s in sigs)
        stuck = unknown or any(s.get("hand_empty") is False or s.get("outcome") == "withdraw_timeout" for s in sigs)
        if unknown:
            self._log("place_hand_unknown", node, result=code.value,
                      termination_reason="손 상태 미확인(opened_before_latch) — 컵이 홀더 밖이라 걸림으로 보고 이어 가지 않는다")
        return (code in (F.POLICY_TIMEOUT, F.PLACE_FAILED) and not stuck and node.type == "policy" and bool(node.jobs)
                and all(self.ep.policies[j.role].kind in self.ep.continue_on_fail for j in node.jobs))

    def _continue_after_fail(self, node: Node, code: FailureCode, reason: str) -> str:
        """★11.07 사용자 "어차피 place 이후엔 실패하든 성공하든 그냥 home 자세로 가는게 맞는데" — 기다리지 않고 다음 노드로.
        손은 EMPTY(손이 빠졌다), 놓던 물체는 홀더로 단정하지 않고 자리 모름(unknown), *_placed 플래그는 세우지 않는다."""
        w = self.world.apply({k: v for k, v in node.expect.items() if k in HAND_KEYS})
        for j in node.jobs:
            if j.source_object:
                w = w.locate(j.source_object, "unknown")
        self.world = w
        self.place_failed_continued.append(node.id)
        self._mark(node.id, "failed", code=code.value)
        self.last = {**self.last, "node": node.id, "code": "place_failed_continued",
                     "reason": f"놓기 실패({code.value}: {reason}) — 다음 노드로 넘어간다"}
        self._next_node()
        self._log("place_failed_continued", node, result=code.value, termination_reason=reason,
                  next=None if self.node is None else self.node.id)
        self.status = READY if self.node is not None else SUCCESS
        return self._close_terminal()

    def _close_terminal(self) -> str:
        """다음 노드가 terminal 이면 바로 닫는다 — 로봇을 움직이지 않고 묻을 이름(next_action)도 없어서, 구분 실행에서는
        마지막 노드 뒤 READY 로 남아 에피소드가 끝나지 않았다(★10.20 가짜 1-B: lift 뒤 done 에서 15 분 대기)."""
        if self.status == READY and self.node is not None and self.node.type == "terminal":
            return self._advance()
        return self.status

    def _declined(self, node: Node | None, what: str) -> str:
        """승인이 없다 — 아무것도 움직이지 않았다. 에피소드는 그 자리에서 기다린다(실패 아님)."""
        self.last = {"node": None if node is None else node.id, "code": F.OPERATOR_DECLINED.value,
                     "reason": f"{what} 승인 없음 — 실행하지 않았다"}
        self._log("declined", node, termination_reason=f"{what} 승인 없음")
        return self.status

    def _fail(self, node: Node | None, code: FailureCode, reason: str) -> str:
        self.executor.safe_stop(reason)
        self.status = FAILURE
        self.last = {**self.last, "node": None if node is None else node.id, "code": code.value, "reason": reason}
        self._log("safe_stop", node, result=FAILURE, termination_reason=reason, failure=code.value)
        return self.status

    def _describe(self, node: Node) -> str:
        if node.type == "trajectory":
            return f"궤적 {node.name} (팔 {', '.join(self.ep.trajectories[node.name].sides)})"
        if node.type == "snapshot":
            if node.objects:                            # ★10.28 사람이 놓은 물체
                return f"FP++ 로 다시 찍어 기록 {', '.join(node.objects)}(사람이 놓은 물체)"
            on_table = [n for n in self.ep.objects if (self.world.objects.get(n) or {}).get("at", "table") == "table"]
            return f"물체 배치 기록 {', '.join(on_table)}"
        if node.type == "confirm":
            return f"확인: {node.text}"
        if node.type == "lanes":
            return " | ".join(f"{side} 팔: " + " → ".join(self._lane_item(n) for n in self.ep.nodes
                                                       if n.group == node.id and n.lane == side)
                              for side in ("right", "left")) + f" | {LANES_WARN}"
        if node.type == "hand":
            what = {"open": "손 펴기(팔 제자리)", "grasp": "닿을 때까지 닫고 쥔 판정(팔 제자리)"}[node.action]
            return f"{what} · {', '.join(node.sides)}"
        return " + ".join(f"{j.role}({self.ep.policies[j.role].policy or '정책 없음'}"
                          + (f", {j.target_object or j.source_object}" if (j.target_object or j.source_object) else "")
                          + (f" → {j.target_holder}" if j.target_holder else "") + ")" for j in node.jobs)

    def _lane_item(self, n: Node) -> str:
        if n.type == "trajectory":
            return f"{n.id}(궤적 {n.name})"
        return f"{n.id}(" + ", ".join(f"{j.role}={self.ep.policies[j.role].policy or '정책 없음'}"
                                      + (f" → {j.target_object or j.source_object}" if (j.target_object or j.source_object) else "")
                                      + (f" · 가드 {j.lift_guard[0]}({j.lift_guard[1]:.0f} mm · {j.lift_guard[2]} m)"
                                         if j.lift_guard else "") for j in n.jobs) + ")"

    @staticmethod
    def _signal_rows(results) -> dict:
        """★10.25 정책 노드 판정 신호(손 비움 · 홀더 안 · 이송 등)를 역할별로 jsonl 에 남긴다 — 실기 e2e CSV(tools/real_e2e_csv.py) 입력."""
        return {r.role: dict(r.signals) for r in results if getattr(r, "role", None) and r.signals}

    def _log(self, event: str, node: Node | None, **kw) -> None:
        w = self.world.as_dict()
        row = {"timestamp": _now(), "episode_id": self.episode_id, "episode": self.ep.name, "event": event,
               "state_id": None if node is None else node.id, "executor_type": None if node is None else node.type,
               "policy_name": None if node is None else (node.name if node.type == "trajectory"
                                                         else ",".join(j.role for j in node.jobs) or None),
               "left_hand_object": w["left_hand"], "right_hand_object": w["right_hand"], "pose_state": w["pose"],
               "task_flags": w["task_flags"], "auto": self._auto, **kw}
        self.history.append({k: row[k] for k in ("timestamp", "event", "state_id") if k in row} | {
            k: kw[k] for k in ("result", "termination_reason") if k in kw})
        self.log(row)


def _with_pose(w: World, name: str, pose) -> World:
    objects = dict(w.objects)
    objects[name] = {**objects[name], "pose": [list(map(float, pose[0])), list(map(float, pose[1]))]}
    return World(pose=w.pose, hands=w.hands, flags=w.flags, objects=objects)


def jsonl_logger(path) -> Callable[[dict], None]:
    def write(row: dict) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return write

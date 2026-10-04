"""EpisodeManager — 에피소드 YAML 순서대로 노드를 실행하고(구분 실행 step · 연속 실행 run), 실패면 복구한다. 순수(실행기는 주입).

10.04 사용자: "테스트할 때는 한 개씩 명령 내리면서 구분동작, 최종 목표는 한 번에(중간에 사용자 개입 없이)".
  · step()  노드 하나 — 승인 hook 이 그 노드마다 묻는다(상황판 [다음 노드]).
  · run()   끝까지 — 승인은 시작에 한 번(상황판 [연속 실행]). 실패면 정해진 복구(재시도 상한 · 홈 checkpoint 로
            되돌아가기)를 스스로 하고, 안전 규칙 · 복구 수단 없음이면 safe stop. 멈추면 승인이 끝난다.
실행기 계약(Executor):
  snapshot(node, world)            → (NodeResult, {물체: (pos, quat)})
  run_policies(node, plans, world) → [NodeResult, …]   plans = [PolicyPlan] (병렬이면 둘 이상)
  run_trajectory(node, traj, world)→ NodeResult
  prepare(step, plans, world)      → NodeResult           refresh_perception · open_hand
  safe_stop(reason)                → None
  joints()                         → dict                 checkpoint 기록용(없으면 {})
로그(JSONL, 가이드 1-10): timestamp · episode_id · state_id · executor_type · policy_name · start/end/duration ·
  left/right_hand_object · pose_state · task_flags · termination_reason · event · result · next.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Protocol

from policy_control.episode_failure import (F, FailureCode, FailureDetector, NodeResult, Recovery, RecoveryManager,
                                            ROLLBACK, SAFE_STOP)
from policy_control.episode_flow import flow_of
from policy_control.episode_spec import EMPTY, Episode, Node
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


class Executor(Protocol):
    def snapshot(self, node: Node, world: World) -> tuple[NodeResult, dict]: ...
    def run_policies(self, node: Node, plans: list, world: World) -> list: ...
    def run_trajectory(self, node: Node, traj, world: World) -> NodeResult: ...
    def prepare(self, step: str, plans: list, world: World) -> NodeResult: ...
    def safe_stop(self, reason: str) -> None: ...
    def joints(self) -> dict: ...


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
    _auto: bool = False

    def __post_init__(self) -> None:
        if not self.flow:
            self.flow = flow_of(self.ep)

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
            return self._do_recovery(node, rec)
        return self._advance()

    def run(self) -> str:
        """끝까지(연속 실행) — 승인은 여기서 한 번. 복구 동작도 이 승인 안이다(재시도 상한 · 안전 규칙은 그대로)."""
        if self.status in (SUCCESS, FAILURE, STOPPED):
            return self.status
        rest = " → ".join(n.id for n in self.ep.nodes[self.index:])
        if not self.approve(f"episode:{self.ep.name}", f"연속 실행 — {rest}"):
            return self._declined(self.node, f"episode:{self.ep.name}")
        self._auto = True
        try:
            while self.status in (READY, RUNNING):
                self._advance()
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
        self.executor.safe_stop(reason)
        if self.node is not None and self.runs.get(self.node.id, {}).get("state") == "running":
            self._mark(self.node.id, "stopped")
        self.status = STOPPED
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
                "history": self.history[-20:]}

    # ---------------------------------------------------------------- 내부
    def _advance(self) -> str:
        node = self.node
        if node is None:
            self.status = SUCCESS
            return self.status
        if node.type == "terminal":
            self.status = SUCCESS if (node.result or SUCCESS) == SUCCESS else FAILURE
            self._mark(node.id, "done" if self.status == SUCCESS else "failed", n=1)
            self._log("terminal", node, result=self.status)
            self.index += 1
            return self.status
        if not self._auto and not self.approve(node.id, self._describe(node)):
            return self._declined(node, node.id)
        self.status = RUNNING
        t0, start = time.monotonic(), _now()
        self._mark(node.id, "running", n=self.runs.get(node.id, {"n": 0})["n"] + 1)
        self._log("enter", node, start_time=start)
        before = self.world
        self.world = self.world.running() if node.type in ("policy", "parallel_policy") else self.world
        results, located = self._execute(node, before)
        if self.status == STOPPED:                       # 실행 중에 stop — 결과는 남기고 더 나가지 않는다
            self.world = before
            self._mark(node.id, "stopped", s=round(time.monotonic() - t0, 2))
            self._log("stopped", node, start_time=start, end_time=_now(),
                      termination_reason="; ".join(r.reason for r in results if r.reason))
            return self.status
        code, who = self.detector.evaluate(self.ep, node, results)
        dur = time.monotonic() - t0
        reason = "; ".join(r.reason for r in results if r.reason)
        self.last = {"node": node.id, "code": code.value, "who": who, "reason": reason, "duration_s": round(dur, 2),
                     "results": [{"role": r.role, "status": r.status, "reason": r.reason, "signals": dict(r.signals)}
                                 for r in results]}
        if code is F.NONE:
            self._mark(node.id, "done", s=round(dur, 2), code="COMPLETED")
            self.world = self._after_success(node, before, located)
            if node.checkpoint:
                self.checkpoints[node.checkpoint] = {"world": self.world, "joints": self.executor.joints(), "node": node.id,
                                                     "at": _now()}
            self.index += 1
            self._log("exit", node, start_time=start, end_time=_now(), duration=round(dur, 3), result="COMPLETED",
                      termination_reason=reason, next=None if self.node is None else self.node.id)
            self.status = READY if self.node is not None else SUCCESS
            return self.status
        self.world = before
        self._mark(node.id, "failed", s=round(dur, 2), code=code.value)
        self._log("failure", node, start_time=start, end_time=_now(), duration=round(dur, 3), result=code.value,
                  termination_reason=reason, failed_role=who)
        return self._recover(node, code)

    def _execute(self, node: Node, world: World) -> tuple[list, dict]:
        try:
            if node.type == "snapshot":
                res, located = self.executor.snapshot(node, world)
                return [res], located
            if node.type == "trajectory":
                return [self.executor.run_trajectory(node, self.ep.trajectories[node.name], world)], {}
            return list(self.executor.run_policies(node, self._plans(node, world), world)), {}
        except Exception as exc:                                       # 실행기 예외 = 관측 · 컨트롤러 이상 → safe stop
            return [NodeResult("error", f"{type(exc).__name__}: {exc}")], {}

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
                                  setting=self.ep.setting.get(side) if side else None, object_pose=pose))
        return out

    def _after_success(self, node: Node, before: World, located: Mapping) -> World:
        w = before.apply(node.expect)
        for name, pose in located.items():                          # snapshot — 물체는 테이블 위, 자세를 기록
            w = w.locate(name, "table", pose[0])
            w = _with_pose(w, name, pose)
        for j in node.jobs:
            b = self.ep.policies[j.role]
            if b.kind == "aglt" and j.target_object and w.hand(b.side) == j.target_object:
                w = w.locate(j.target_object, f"{b.side}_hand")
            if b.kind == "place" and j.source_object and j.target_holder and w.hand(b.side) == EMPTY:
                w = w.locate(j.source_object, f"holder:{j.target_holder}")
        return w

    def _recover(self, node: Node, code: FailureCode) -> str:
        n = self.attempts.get(node.id, 0)
        rec: Recovery = self.recovery.decide(self.ep, node, code, n, self.world, self.checkpoints)
        self._log("recovery", node, result=rec.action, termination_reason=rec.reason, checkpoint=rec.checkpoint)
        if rec.action == SAFE_STOP:
            return self._fail(node, code, rec.reason)
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
            res, located = self.executor.snapshot(Node(id=f"{node.id}:resnap", type="snapshot"), self.world)
            if res.status != "completed":
                return self._fail(node, F.TARGET_NOT_FOUND, f"되돌아간 뒤 배치 재기록 실패: {res.reason}")
            for name, pose in located.items():
                if (self.world.objects.get(name) or {}).get("at", "table") == "table":
                    self.world = _with_pose(self.world.locate(name, "table", pose[0]), name, pose)
            self._log("resnapshot", node, result="COMPLETED", objects=sorted(located))
        self.status = READY
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
            return f"물체 배치 기록 {', '.join(self.ep.objects)}"
        return " + ".join(f"{j.role}({self.ep.policies[j.role].policy or '정책 없음'}"
                          + (f", {j.target_object or j.source_object}" if (j.target_object or j.source_object) else "")
                          + (f" → {j.target_holder}" if j.target_holder else "") + ")" for j in node.jobs)

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

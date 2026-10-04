"""모의 실행기 — 로봇 없이 에피소드 순서 · 상태 전이 · 실패 → 복구를 돌린다(tools/episode_run.py --dry-run · 테스트).

실패 주입: {노드 id: [FailureCode, …]} — 그 노드를 부를 때마다 앞에서 하나씩 꺼내 그 실패가 나는 결과를 만든다
(실제 실행기가 싣는 신호와 같은 모양 — 판정은 episode_failure 가 한다). 비면 정상 결과.
"""
from __future__ import annotations

from typing import Mapping

from policy_control.episode_failure import F, FailureCode, NodeResult
from policy_control.episode_spec import Episode, Node

_OK = {"aglt": {"attached": True, "setting_err_m": 0.01}, "place": {"cup_in_holder": True, "hand_empty": True},
       "pour": {"pour_ok": True}, "lock": {"locked": True}, "shake": {}}

_FAIL = {
    F.GRASP_FAILED_LEFT: ("completed", "", {"attached": False}),
    F.GRASP_FAILED_RIGHT: ("completed", "", {"attached": False}),
    F.OBJECT_DROPPED_LEFT: ("completed", "", {"attached": True, "dropped": True}),
    F.OBJECT_DROPPED_RIGHT: ("completed", "", {"attached": True, "dropped": True}),
    F.POLICY_TIMEOUT: ("timeout", "episode time >= 15.0 s", {}),
    F.PLACE_FAILED: ("completed", "", {"cup_in_holder": False, "hand_empty": True}),
    F.HOLDER_NOT_FOUND: ("refused", "holder pose missing", {}),
    F.TARGET_NOT_FOUND: ("refused", "arm 컵 자세가 없다(또는 0.5 s 넘게 끊겼다)", {}),
    F.POSE_MISMATCH: ("completed", "", {"attached": True, "setting_err_m": 0.12}),
    F.CONTROLLER_ERROR: ("completed", "", {"pd_fault": "HOLD"}),
    F.OBSERVATION_ERROR: ("error", "source 'arm' is stale", {}),
    F.POUR_FAILED: ("completed", "", {"pour_ok": False}),
    F.POUR_PARTIAL: ("completed", "", {"pour_ok": False, "partial": True}),
    F.LOCK_FAILED: ("completed", "", {"locked": False}),
}


class FakeExecutor:
    def __init__(self, ep: Episode, inject: Mapping[str, list] | None = None) -> None:
        self.ep = ep
        self.inject = {k: [FailureCode(c) for c in v] for k, v in (inject or {}).items()}
        self.calls: list[tuple] = []
        self.stopped: list[str] = []

    def _take(self, key: str) -> FailureCode | None:
        q = self.inject.get(key) or []
        return q.pop(0) if q else None

    def snapshot(self, node: Node, world) -> tuple[NodeResult, dict]:
        self.calls.append(("snapshot", node.id))
        code = self._take(node.id)
        if code is not None:
            status, reason, signals = _FAIL[code]
            return NodeResult(status, reason, signals=signals), {}
        poses = {name: ((0.25, (-0.20 if i % 2 == 0 else 0.20), 0.29), (1.0, 0.0, 0.0, 0.0))
                 for i, name in enumerate(self.ep.objects)}
        return NodeResult("completed", "", signals={"objects": sorted(poses)}), poses

    def run_policies(self, node: Node, plans: list, world) -> list:
        self.calls.append(("policy", node.id, tuple(p.role for p in plans)))
        code = self._take(node.id)
        out = []
        for i, p in enumerate(plans):
            if code is not None and i == 0:
                status, reason, signals = _FAIL[code]
                out.append(NodeResult(status, reason, 1.0, p.role, signals))
            else:
                out.append(NodeResult("completed", "", 1.0, p.role, dict(_OK[p.kind])))
        return out

    def run_trajectory(self, node: Node, traj, world) -> NodeResult:
        self.calls.append(("trajectory", node.id, traj.name))
        code = self._take(node.id)
        if code is not None:
            status, reason, signals = _FAIL[code]
            return NodeResult(status, reason, signals=signals)
        return NodeResult("completed", "", 2.0)

    def prepare(self, step: str, plans: list, world) -> NodeResult:
        self.calls.append(("prepare", step, tuple(p.role for p in plans)))
        return NodeResult("completed", "")

    def safe_stop(self, reason: str) -> None:
        self.stopped.append(reason)

    def joints(self) -> dict:
        return {}

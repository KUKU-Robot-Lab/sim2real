"""모의 실행기 — 로봇 없이 에피소드 순서 · 상태 전이 · 실패 → 복구를 돌린다(tools/episode_run.py --dry-run · 테스트).

실패 주입: {노드 id: [FailureCode, …]} — 그 노드를 부를 때마다 앞에서 하나씩 꺼내 그 실패가 나는 결과를 만든다
(실제 실행기가 싣는 신호와 같은 모양 — 판정은 episode_failure 가 한다). 비면 정상 결과.
"""
from __future__ import annotations

from typing import Mapping

from policy_control.episode_failure import F, FailureCode, NodeResult
from policy_control.episode_spec import Episode, Node, region_box

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


def _fake_pos(obj: Mapping, i: int) -> tuple:
    """모의 물체 자리 — region(★10.28 검증 상자)이 있으면 그 가운데, 아니면 앞 물체부터 번갈아 오른쪽 · 왼쪽."""
    box = region_box(obj["region"]) if obj.get("region") is not None else {}
    x = sum(box["x"]) / 2 if "x" in box else 0.25
    y = sum(box["y"]) / 2 if "y" in box else (-0.20 if i % 2 == 0 else 0.20)
    return (x, y, 0.29)


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
        # 실제 실행기와 같은 물체만 — ★10.28 이름을 준 snapshot 은 그 물체, 아니면 탁자 위(at_start away 는 아직 없다)
        names = set(node.objects) or {n for n in self.ep.objects if (world.objects.get(n) or {}).get("at", "table") == "table"}
        poses = {name: (_fake_pos(self.ep.objects[name], i), (1.0, 0.0, 0.0, 0.0))
                 for i, name in enumerate(self.ep.objects) if name in names}
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

    def run_hand(self, node: Node, world) -> NodeResult:
        """★10.17 경로 B 손 노드 — 쥠 실패 주입은 실기 도구(hand_grasp.py rc 2)와 같은 문구로."""
        self.calls.append(("hand", node.id, node.action, tuple(node.sides)))
        code = self._take(node.id)
        if code in (F.GRASP_FAILED_LEFT, F.GRASP_FAILED_RIGHT):
            return NodeResult("aborted", "hand_grasp rc 2: not grasped — 4.0 s 안에 래치 손가락 0 < 2", 4.0)
        if code is not None:
            status, reason, signals = _FAIL[code]
            return NodeResult(status, reason, signals=signals)
        return NodeResult("completed", "", 3.5)

    def prepare(self, step: str, plans: list, world) -> NodeResult:
        self.calls.append(("prepare", step, tuple(p.role for p in plans)))
        return NodeResult("completed", "")

    def safe_stop(self, reason: str) -> None:
        self.stopped.append(reason)

    def joints(self) -> dict:
        return {}

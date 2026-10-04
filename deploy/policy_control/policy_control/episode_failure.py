"""에피소드 Step 2 — 노드 결과 → 실패 코드(규칙) → 복구 결정. 순수(ROS 없음).

가이드 2-2 ~ 2-8 을 이 저장소 신호에 맞췄다. 실행기(episode_ros)가 노드 결과(NodeResult)에 실은 신호만 본다:
  status  completed | timeout | refused | aborted | error      (정책 노드 에피소드 이벤트 · 서비스 응답 · 경로 재생 rc)
  reason  사람이 읽는 끝난 이유(노드 이벤트 reasons · 거부 사유)
  signals attached · dropped(aglt 컵이 붙었다가 떨어짐) · setting_err_m · cup_in_holder · hand_empty · pd_fault …
안전 규칙(pd HOLD · estop · 발열 · 컨트롤러 오류)은 복구 없이 곧바로 safe stop — VLM(Step 3)이 대신하지 않는다(가이드 3-6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from policy_control.episode_spec import Binding, Episode, Node

SETTING_TOL_M = 0.04          # 손에 든 컵 원점 ↔ SETTING 점(aglt 도착 tol 0.02 · 인계 뱅크 분포 ±1.5 cm 를 덮는다)
HOLDER_XY_TOL_M = 0.035       # 홀더 구멍 반지름(hdgp place_geom HOLDER_BORE_R)
HOLDER_Z_TOL_M = 0.03


class FailureCode(str, Enum):
    NONE = "none"
    POLICY_TIMEOUT = "policy_timeout"
    TARGET_NOT_FOUND = "target_not_found"
    GRASP_FAILED_LEFT = "grasp_failed_left"
    GRASP_FAILED_RIGHT = "grasp_failed_right"
    OBJECT_DROPPED_LEFT = "object_dropped_left"
    OBJECT_DROPPED_RIGHT = "object_dropped_right"
    POUR_FAILED = "pour_failed"
    POUR_PARTIAL = "pour_partial"
    PLACE_FAILED = "place_failed"
    HOLDER_NOT_FOUND = "holder_not_found"
    LOCK_FAILED = "lock_failed"
    LID_NOT_FOUND = "lid_not_found"
    SHAKE_ABORTED = "shake_aborted"
    POSE_MISMATCH = "pose_mismatch"
    CONTROLLER_ERROR = "controller_error"
    OBSERVATION_ERROR = "observation_error"
    OPERATOR_DECLINED = "operator_declined"
    UNKNOWN = "unknown"


F = FailureCode
SAFETY = (F.CONTROLLER_ERROR, F.OBSERVATION_ERROR, F.UNKNOWN, F.OPERATOR_DECLINED)


@dataclass(frozen=True)
class NodeResult:
    status: str                    # completed | timeout | refused | aborted | error
    reason: str = ""
    duration_s: float = 0.0
    role: str = ""                 # 병렬 노드에서 어느 정책인지
    signals: Mapping = field(default_factory=dict)


def _side_code(side: str, right: FailureCode, left: FailureCode) -> FailureCode:
    return left if side == "left" else right


#: 거부 · 중단 사유 문구 → 실패(정책 노드 · pd · 노드 측정의 실제 문구, 소문자로 찾는다)
_REASON_RULES = (
    (("holder pose missing",), "holder"),
    (("컵 자세가 없다", "cup pose missing", "goal comes from the cup"), "target"),
    (("not grasped",), "grasp"),
    (("held target", "training start pose", "rad from", "start pose"), "pose"),
    (("estop", "thermal", "controller", "pd phase", "watchdog"), "controller"),
    (("source", "stale", "missing", "ticks without a valid step"), "observation"),
)


def _from_reason(reason: str, b: Binding | None) -> FailureCode:
    r = reason.lower()
    for keys, kind in _REASON_RULES:
        if any(k.lower() in r for k in keys):
            if kind == "holder":
                return F.HOLDER_NOT_FOUND
            if kind == "target":
                return F.LID_NOT_FOUND if b is not None and b.kind == "lock" else F.TARGET_NOT_FOUND
            if kind == "grasp":
                return _side_code(b.side if b else "right", F.GRASP_FAILED_RIGHT, F.GRASP_FAILED_LEFT)
            if kind == "pose":
                return F.POSE_MISMATCH
            return F.CONTROLLER_ERROR if kind == "controller" else F.OBSERVATION_ERROR
    return F.UNKNOWN


class FailureDetector:
    """노드 결과 → 실패 코드. 병렬 노드는 결과마다 보고 첫 실패를 낸다."""

    def evaluate(self, ep: Episode, node: Node, results: list) -> tuple[FailureCode, str]:
        for res in results:
            code = self._one(ep, node, res)
            if code is not F.NONE:
                return code, res.role or node.name
        return F.NONE, ""

    def _one(self, ep: Episode, node: Node, res: NodeResult) -> FailureCode:
        b = ep.policies.get(res.role) if res.role else None
        s = res.signals
        if s.get("pd_fault"):
            return F.CONTROLLER_ERROR
        if res.status == "error":
            return F.CONTROLLER_ERROR if node.type == "trajectory" else F.OBSERVATION_ERROR
        if res.status in ("refused", "aborted"):
            return _from_reason(res.reason, b)
        if res.status == "timeout":
            return F.POLICY_TIMEOUT
        if res.status != "completed":
            return F.UNKNOWN
        if node.type == "trajectory" or b is None:
            return F.NONE
        side = b.side
        if b.kind == "aglt":
            if s.get("dropped"):
                return _side_code(side, F.OBJECT_DROPPED_RIGHT, F.OBJECT_DROPPED_LEFT)
            if s.get("attached") is False:
                return _side_code(side, F.GRASP_FAILED_RIGHT, F.GRASP_FAILED_LEFT)
            err = s.get("setting_err_m")
            if err is not None and float(err) > SETTING_TOL_M:
                return F.POSE_MISMATCH
        elif b.kind == "place":
            if s.get("cup_in_holder") is False or s.get("hand_empty") is False:
                return F.PLACE_FAILED
        elif b.kind == "pour":
            if s.get("dropped"):
                return F.OBJECT_DROPPED_RIGHT
            if s.get("pour_ok") is False:
                return F.POUR_PARTIAL if s.get("partial") else F.POUR_FAILED
        elif b.kind == "lock":
            if s.get("locked") is False:
                return F.LOCK_FAILED
        elif b.kind == "shake":
            if s.get("dropped"):
                return F.OBJECT_DROPPED_LEFT
        return F.NONE


# ---------------------------------------------------------------- 복구
RETRY, ROLLBACK, SAFE_STOP = "retry", "rollback_retry", "safe_stop"


@dataclass(frozen=True)
class Recovery:
    action: str                     # retry | rollback_retry | safe_stop
    reason: str
    pre: tuple = ()                 # 재시도 전 준비(refresh_perception · open_hand)
    checkpoint: str | None = None   # rollback 이 돌아갈 checkpoint


#: 가이드 2-6 표 — 이 저장소에 있는 수단만 쓴다(재파지 기술 · 장면 재평가가 없는 것은 safe stop)
_TABLE = {
    F.TARGET_NOT_FOUND: (RETRY, ("refresh_perception",)),
    F.HOLDER_NOT_FOUND: (RETRY, ("refresh_perception",)),
    F.LID_NOT_FOUND: (RETRY, ("refresh_perception",)),
    F.GRASP_FAILED_LEFT: (ROLLBACK, ("open_hand",)),
    F.GRASP_FAILED_RIGHT: (ROLLBACK, ("open_hand",)),
    F.POSE_MISMATCH: (ROLLBACK, ()),
    F.POUR_FAILED: (RETRY, ()),
    F.LOCK_FAILED: (RETRY, ()),
    F.POLICY_TIMEOUT: (ROLLBACK, ("open_hand",)),       # aglt 만(쥔 게 없을 때) — 아래에서 거른다
}


class RecoveryManager:
    """실패 → 복구 결정. 재시도는 노드마다 episode.failure_policy 상한까지(무한 재시도 금지, 가이드 2-8)."""

    def decide(self, ep: Episode, node: Node, code: FailureCode, attempts: int, world, checkpoints: Mapping) -> Recovery:
        if code in SAFETY:
            return Recovery(SAFE_STOP, f"{code.value} — 안전 규칙: 복구 없이 정지")
        if attempts >= ep.max_retry(node):
            return Recovery(SAFE_STOP, f"{code.value} — 재시도 {attempts}/{ep.max_retry(node)} 다 씀")
        action, pre = _TABLE.get(code, (SAFE_STOP, ()))
        kinds = {ep.policies[j.role].kind for j in node.jobs if j.role in ep.policies}
        if code is F.POLICY_TIMEOUT and kinds - {"aglt"}:
            return Recovery(SAFE_STOP, "policy_timeout — 컵을 든 채 끝났을 수 있다(aglt 밖) · 정지")
        if action == ROLLBACK:
            cp = rollback_point(ep, node, world, checkpoints)
            if cp is None:
                return Recovery(SAFE_STOP, f"{code.value} — 손에 든 물체가 같은 되돌아갈 checkpoint(trajectory)가 없다")
            return Recovery(ROLLBACK, f"{code.value} — {cp} 로 되돌아가 다시", pre, cp)
        if action == SAFE_STOP:
            return Recovery(SAFE_STOP, f"{code.value} — 이 저장소에 복구 수단이 없다(장면 재평가 · 재파지는 Step 3)")
        return Recovery(RETRY, f"{code.value} — 같은 노드 다시", pre)


def rollback_point(ep: Episode, node: Node, world, checkpoints: Mapping) -> str | None:
    """되돌아갈 checkpoint — 이 노드 앞, trajectory 노드가 만든 것, 두 손의 물체가 지금 기대와 같은 가장 가까운 것.

    실패한 노드 직전 상태(손 물체)는 노드가 바꾸기 전이므로 world(실패 전)와 같아야 한다 — 물체를 든 채 돌아가는 것은 막는다."""
    here = ep.node_index(node.id)
    for i in range(here - 1, -1, -1):
        n = ep.nodes[i]
        cp = checkpoints.get(n.checkpoint) if n.checkpoint else None
        if cp is None or n.type != "trajectory":
            continue
        if dict(cp["world"].hands) == dict(world.hands):
            return n.checkpoint
    return None

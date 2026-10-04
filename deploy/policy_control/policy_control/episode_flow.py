"""상황판 에피소드 흐름 그림의 데이터 — 노드 상자(순서) + 피드백 연결(재시도 · 되돌아가기). 순수(ROS 없음).

10.04 사용자: "상황판이 연결창처럼 순서대로 보였으면 해(동작 2번 반복하는 부분은 피드백 연결처럼)".
피드백은 episode_failure 의 복구 표와 같은 규칙으로 미리 그린다 — 실제로 쓴 횟수는 실행기 view 의 attempts 가 채운다.
  rollback  집기(aglt) 실패 · 시간 초과 · 자세 어긋남 → 손 펴고 손 상태가 같은 홈 checkpoint 로 → 배치 재기록 → 다시
  retry     컵 · 홀더를 못 찾음 → 인지 갱신 후 같은 노드 다시
재시도 상한(failure_policy)이 0 인 노드(놓기)는 피드백이 없다 — 실패면 안전 정지.
"""
from __future__ import annotations

from policy_control.episode_spec import EMPTY, HAND_KEYS, Episode, Node
from policy_control.episode_world import World

TYPE_LABEL = {"snapshot": "배치 기록", "trajectory": "궤적", "policy": "정책", "parallel_policy": "병렬 정책", "terminal": "끝"}
ROLLBACK_LABEL = "집기 실패 · 시간 초과 → 손 펴고 빈손 홈으로 → 배치 다시 기록 → 다시"
RETRY_LABEL = "컵 · 홀더를 못 찾으면 인지 갱신 후 다시"


def _expect_text(expect) -> str:
    out = []
    if expect.get("pose"):
        out.append(str(expect["pose"]))
    for key, side in HAND_KEYS.items():
        if key in expect:
            out.append(f"{'오른손' if side == 'right' else '왼손'} {'빈손' if expect[key] == EMPTY else expect[key]}")
    out += [f"{k} ✓" for k, v in expect.items() if k not in ("pose", *HAND_KEYS) and v]
    return " · ".join(out)


def _lines(ep: Episode, n: Node) -> list[str]:
    if n.type == "trajectory":
        t = ep.trajectories[n.name]
        return [f"{n.name} · 팔 {', '.join(t.sides)}", "실측 → 경로 재계획 → 재생 → 정착" if t.kind == "rehome" else t.kind]
    if n.type == "snapshot":
        return [f"FP++ {', '.join(ep.objects)}", "정지 · 신선 1.5 s 중앙값"]
    if n.type == "terminal":
        return [str(n.result or "SUCCESS")]
    out = []
    for j in n.jobs:
        b = ep.policies[j.role]
        obj = j.target_object or j.source_object
        where = f" → {j.target_holder}({ep.holders[j.target_holder]})" if j.target_holder else ""
        out.append(f"{j.role} · {b.policy or '정책 없음'}")
        if obj or where:
            out.append(f"{obj or ''}{where}".strip())
    return out


def flow_of(ep: Episode) -> dict:
    """{"nodes": [{id, type, title, lines, expect, checkpoint}], "feedback": [{from, to, kind, max, label}]}."""
    world, before, after = World(), [], []
    for n in ep.nodes:
        before.append(dict(world.hands))
        world = world.apply(n.expect)
        after.append(dict(world.hands))
    nodes = [{"id": n.id, "type": n.type, "title": TYPE_LABEL.get(n.type, n.type), "lines": _lines(ep, n),
              "expect": _expect_text(n.expect), "checkpoint": n.checkpoint} for n in ep.nodes]
    feedback = []
    for i, n in enumerate(ep.nodes):
        if n.type not in ("policy", "parallel_policy"):
            continue
        mx = ep.max_retry(n)
        if mx <= 0:
            continue
        if any(ep.policies[j.role].kind == "aglt" for j in n.jobs):
            target = next((ep.nodes[k].id for k in range(i - 1, -1, -1)
                           if ep.nodes[k].type == "trajectory" and ep.nodes[k].checkpoint and after[k] == before[i]), None)
            if target is not None:
                feedback.append({"from": n.id, "to": target, "kind": "rollback", "max": mx, "label": ROLLBACK_LABEL})
        feedback.append({"from": n.id, "to": n.id, "kind": "retry", "max": mx, "label": RETRY_LABEL})
    return {"nodes": nodes, "feedback": feedback}

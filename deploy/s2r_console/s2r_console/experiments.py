"""policy 탭 '실험' 목록(10.21 사용자 "정책을 찾기 어렵고 실험 진행이 어려움 · 깔끔하고 명확하게") — 읽기만 한다.

실험 = 에피소드 파일 하나(팔마다 하나 또는 두 팔 하나). 화면은 실험을 고르면 그 실험이 쓰는 정책 사슬
(접근 정책 → 넘김 → 파지 정책 · 스크립트 접근 → 확인 → 쥐기 …)을 보이고, 실행은 이미 있는 길로 보낸다:
  파지 · 들기(한 팔)  ctl_<에피소드 머리>_<팔> 관리 작업(actions.CONTROL_EPISODES) — 승인 창 → 실행기 · 정책 노드 · 기록
  tilt(두 팔)         미션 단계 episode_tilt_base_<경로>(▶ 실행) — 진행은 '지금 에피소드' [연속 실행]
사슬 · 컵 · 쓸 수 있는지는 에피소드 파일과 정책 등록부에서 읽는다(손으로 적은 제목이 파일과 어긋나지 않게 — 10.21 화면의
'tilt 경로 D = ap05 → rcg03 넘김' 은 파일(1-B 접근 → rcg03)과 달랐다). 파일은 mtime 이 같으면 다시 읽지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import tilt_view as TV

ARMS = ("right", "left")
#: (화면 키, 에피소드 머리, 이름) — 한 팔 파지 · 들기. 머리는 actions.CONTROL_EPISODES 의 키와 같다(시험이 맞춘다).
GRASP = (("1-B", "grasp_b", "1-B"), ("C", "grasp_c", "경로 C"), ("D", "grasp_d", "경로 D"),
         ("E", "grasp_e", "경로 E"), ("F", "grasp_f", "경로 F"))
CONTROL_PREFIX = "ctl_"
SIDE_KO = {"right": "오른팔", "left": "왼팔"}
#: 사슬에서 빼는 노드 — 홈 출발 · 배치 사진 · 끝 · 손 펴기는 모든 실험에 같다
_SKIP_TYPES = ("snapshot", "terminal")


def short_name(policy_id: str | None) -> str:
    """rh56f1/aglt/right_approach_ap05 → ap05 · rh56f1/aglt/left_rcg04 → rcg04."""
    if not policy_id:
        return "정책 없음"
    name = policy_id.rsplit("/", 1)[-1]
    for head in ("right_", "left_", "both_"):
        name = name.removeprefix(head)
    return name.removeprefix("approach_")


def _handoff_text(ag: dict) -> str:
    """넘김 조건 — approach_grasp 값(ApproachGraspCfg)에서. 학습 접근 성공의 pos · still 과 같은 식."""
    from policy_control import approach_grasp as AG
    cfg = AG.ApproachGraspCfg.parse(AG.ApproachGraspCfg.to_param(dict(ag or {})))
    return (f"사전 파지 자리 {AG.HANDOFF_TOL_M * 100:.0f} cm 안 · 손바닥 ≤ {cfg.still_mps:g} m/s "
            f"{cfg.still_s:g} s · 못 오면 {cfg.approach_max_s:g} s 에 멈춤")


def _sides_of(node, ep) -> list[str]:
    if node.type in ("policy", "parallel_policy"):
        return sorted({ep.policies[j.role].side for j in node.jobs}, key=ARMS.index)
    if node.type == "trajectory":
        return list(ep.trajectories[node.name].sides)
    if node.type == "hand":
        return list(node.sides)
    for side in ARMS:                                 # confirm — 이름 끝(confirm_cup_right)
        if node.id.endswith(side) or node.id.endswith("_" + side[0]):
            return [side]
    return []


def _steps(ep) -> list[dict]:
    """에피소드 노드 → 사슬 칸. 칸 = {kind, text, sub, node, sides, id?}. kind: move · confirm · hand · policy · handoff · script."""
    out: list[dict] = []
    for i, n in enumerate(ep.nodes):
        if n.type in _SKIP_TYPES or (n.type == "hand" and n.action != "grasp"):
            continue
        sides = _sides_of(n, ep)
        base = {"node": n.id, "index": i, "sides": sides}
        if n.type == "trajectory":
            t = ep.trajectories[n.name]
            if t.kind == "rehome":
                continue
            text, sub = {"reach": ("1-B 접근", "사전 파지 자리에서 멈춤" if t.stop == "pregrasp" else "손 안 컵 자리까지"),
                         "lift": ("들기", "10 cm"), "carry": ("이송", "tilt 시작 자세로"),
                         "tilt": ("tilt", "붓고 세팅 자세로 복귀")}.get(t.kind, (t.kind, t.name))
            out.append({**base, "kind": "move", "text": text, "sub": sub})
        elif n.type == "confirm":
            out.append({**base, "kind": "confirm", "text": "확인", "sub": n.text})
        elif n.type == "hand":
            out.append({**base, "kind": "hand", "text": "손 쥐기", "sub": "스크립트 닫기"})
        else:
            for job in n.jobs:
                b = ep.policies[job.role]
                one = {**base, "sides": [b.side]}
                ag = b.approach_grasp
                if b.handoff:
                    out += [{**one, "kind": "policy", "id": b.policy, "text": short_name(b.policy), "sub": "접근"},
                            {**one, "kind": "handoff", "text": "넘김", "sub": _handoff_text(ag)},
                            {**one, "kind": "policy", "id": b.handoff, "text": short_name(b.handoff), "sub": "파지 · 들기"}]
                elif ag is not None:
                    stop = "사전 파지 자리에서 멈춤" if ag.get("stop_ready") else "손바닥–컵 가까이"
                    out += [{**one, "kind": "policy", "id": b.policy, "text": short_name(b.policy), "sub": f"접근 · {stop}"},
                            {**one, "kind": "script", "text": "스크립트 쥐기", "sub": "닫기 → 10 cm 들기 → 3 s 유지"}]
                else:
                    out.append({**one, "kind": "policy", "id": b.policy, "text": short_name(b.policy), "sub": "파지 · 들기"})
    return out


def _summary(steps: list[dict]) -> str:
    """목록 한 줄 — 칸 글자를 순서대로, 같은 글자는 한 번(두 팔 실험의 R · L 반복)."""
    seen = list(dict.fromkeys(s["text"] for s in steps if s["kind"] not in ("handoff",)))
    return " → ".join(seen)


def _episode(repo: Path, name: str, entries: dict) -> dict:
    from policy_control import episode_spec as S
    path = repo / "config" / "episodes" / f"{name}.yaml"
    try:
        ep = S.load(path)
    except Exception as exc:                          # noqa: BLE001 — 화면은 이유를 보여 준다
        return {"episode": name, "error": f"{type(exc).__name__}: {exc}", "steps": [], "problems": [], "cups": {}}
    av = S.availability(ep, entries)
    hands = S.hand_problems(ep, repo / "deploy" / "policies")
    problems = [f"{r}: {w}" for r, w in av.items() if w] + [f"손 {n}: {w}" for n, w in hands.items() if w]
    cups = {k: str(v.get("topic", "")).split("/")[2] if str(v.get("topic", "")).startswith("/objects/") else str(v.get("topic", ""))
            for k, v in ep.objects.items()}
    steps = _steps(ep)
    return {"episode": name, "steps": steps, "summary": _summary(steps), "cups": cups, "problems": problems,
            "confirms": sum(n.type == "confirm" for n in ep.nodes)}


def _policy_meta(repo: Path, e) -> dict:
    card = dict(e.card or {})
    ep = re.search(r"ep_?(\d+)", e.checkpoint or "")
    return {"id": e.id, "short": short_name(e.id), "side": card.get("side", ""), "status": card.get("status", ""),
            "ep": int(ep.group(1)) if ep else None, "summary": str(card.get("summary", "")), "eval": str(card.get("eval", "")),
            "checkpoint": e.checkpoint, "contract": e.contract, "issues": list(e.issues),
            "path": str(Path(e.path).relative_to(repo)) if Path(e.path).is_relative_to(repo) else str(e.path)}


def _build(repo: Path, stage_ids: frozenset) -> dict:
    from policy_control import policy_registry as R
    entries = {e.id: e for e in R.scan(repo / "deploy" / "policies", deep=False)}
    items = []
    for key, head, label in GRASP:
        sides = {}
        for side in ARMS:
            name = f"{head}_{side}"
            if f"episode_{name}" not in stage_ids:
                continue
            sides[side] = {**_episode(repo, name, entries), "stage": f"episode_{name}",
                           "action": f"{CONTROL_PREFIX}{head}_{side}", "runner": f"episode_runner_{side}"}
        if sides:
            first = next(iter(sides.values()))
            items.append({"id": head, "group": "grasp", "key": key, "label": label, "summary": first.get("summary", ""),
                          "sides": sides})
    for key, name, _title, when in TV.PATHS:
        if f"episode_{name}" not in stage_ids:
            continue
        one = {**_episode(repo, name, entries), "stage": f"episode_{name}", "action": None, "runner": "episode_runner"}
        items.append({"id": name, "group": "tilt", "key": key, "label": f"tilt {key}", "summary": one.get("summary", ""),
                      "when": when, "sides": {"both": one}})
    used = {s["id"] for it in items for side in it["sides"].values() for s in side["steps"] if s["kind"] == "policy" and s.get("id")}
    policies = {pid: _policy_meta(repo, entries[pid]) for pid in sorted(used) if pid in entries}
    return {"items": items, "policies": policies}


def view(repo: Path, stage_ids) -> dict:
    """상황판 스냅샷 session.experiments — 실패해도 상황판은 돈다(이유는 error)."""
    ids = frozenset(stage_ids)
    files = sorted((repo / "config" / "episodes").glob("*.yaml"))
    pol = repo / "deploy" / "policies" / "rh56f1" / "aglt"
    try:
        return TV._cached(f"xp:{hash(ids)}", files + [pol, *sorted(pol.glob("*/policy.yaml"))], lambda: _build(repo, ids))
    except Exception as exc:                          # noqa: BLE001 — 표시용 집계가 상황판을 멈추지 않게
        return {"items": [], "policies": {}, "error": f"{type(exc).__name__}: {exc}"}

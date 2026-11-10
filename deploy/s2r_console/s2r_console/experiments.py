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
#: ★11.09 사용자 "c,d 제거" — 화면에서 경로 C · D 를 뺐다(에피소드 · 관리 작업은 남는다). "F주황 · F분홍" 칸도 F 안 물체 버튼으로
GRASP = (("1-B", "grasp_b", "1-B"), ("E", "grasp_e", "경로 E"), ("F", "grasp_f", "경로 F"),
         ("FP", "pick_place_f", "경로 F + 놓기"),     # ★10.22 잡고 들기 → newton_05 놓기
         ("S", "grasp_s", "잡기 학생"),               # ★11.11 eval 탭에서 옮김 — 마스크 학생 A 홈 출발 잡고 들기, 오른손 색 버튼
         ("PM", "place_m", "놓기 학생"))              # ★11.11 경로 F 잡고 들기 → 놓기 학생(place_m) — 오른손은 색 · 홀더 버튼
#: ★11.09 사용자 "경로 F 안에서 버튼으로 컵종류와 색깔을 선택 … 왼손,오른손 모두" — (FP++ 물체, 컵, 색). 순서 = 버튼 순서.
#:   생성기 make_rh56f1_missions.F_OBJECTS 와 같다(시험이 맞춘다). 팔 기본 물체는 grasp_f_<팔>, 나머지는 grasp_f_<물체>_<팔>
F_OBJECTS = (("source240_blue", "source240", "파랑"), ("source240_orange", "source240", "주황"), ("source240_pink", "source240", "분홍"),
             ("source200_orange", "source200", "주황"), ("source200_pink", "source200", "분홍"), ("shaker_c_orange", "shaker", "주황"))
F_DEFAULT = {"right": "source240_blue", "left": "shaker_c_orange"}


def f_head(obj: str, side: str) -> str:
    """경로 F 물체 → 에피소드 머리(ctl_<머리>_<팔>)."""
    return "grasp_f" if F_DEFAULT[side] == obj else f"grasp_f_{obj}"
#: ★11.11 사용자 "놓기 학생(place_m) 시험에서 컵 색과 홀더를 콘솔 카드에서 고르게" — (색 키, FP++ 물체, 색). 오른손만(왼손
#:   left_n09_e3800 은 홀더 0 · shaker 하나). 생성기 make_rh56f1_missions.PLACE_M_COLORS · PLACE_M_HOLDERS 와 같다(시험이 맞춘다).
#:   기본(파랑 · 홀더 1)은 place_m_right, 나머지는 place_m_<색>_h<홀더>_right
PLACE_M_COLORS = (("blue", "source240_blue", "파랑"), ("pink", "source240_pink", "분홍"))
PLACE_M_HOLDERS = (1, 2)
PLACE_M_DEFAULT = ("blue", 1)
#: 홀더 2 = sim 계약(right_n10_e4000 target_holders [1, 2]) 안이지만 학생이 배웠는지 확인 전 — 버튼 · 카드에 띄운다
PLACE_M_WARN = {2: "홀더 2 = 학습 범위 확인 전"}


def _cups_right_cup(repo: Path) -> str:
    """★11.11 deploy 리뷰 MEDIUM: cups 단계가 다시 찍는 오른손 컵 — 미션 생성기와 같은 config/fpp_active.yaml(sides.right).

    고른 색이 이것과 다르면 cups 가 그 컵을 찍지 않으니 카드가 Perception Refresh 를 먼저 하라고 띄운다. 못 읽으면 "" (늘 경고).
    """
    import yaml

    try:
        raw = yaml.safe_load((repo / "config" / "fpp_active.yaml").read_text(encoding="utf-8")) or {}
        return str((raw.get("sides") or {}).get("right") or "")
    except (OSError, yaml.YAMLError, AttributeError):
        return ""


def place_m_head(color: str, holder: int) -> str:
    """놓기 학생 (색, 홀더) → 에피소드 머리(ctl_<머리>_right)."""
    return "place_m" if (color, holder) == PLACE_M_DEFAULT else f"place_m_{color}_h{holder}"


#: ★11.11 사용자 "STUDENT 정책들도 POLICY 탭으로 옮기면서 FP 나 PM 처럼" — 잡기 학생(경로 S) (색 키, FP++ 물체, 색).
#:   팔마다 학생 ckpt one_obj 가 허용하는 색만(오른손 파랑 · 분홍, 왼손 주황). 생성기 make_rh56f1_missions.GRASP_S_COLORS ·
#:   GRASP_S_SIDE_COLORS · GRASP_S_DEFAULT 와 같다(시험이 맞춘다). 기본 색은 grasp_s_<팔>, 나머지는 grasp_s_<색>_<팔>
GRASP_S_COLORS = (("blue", "source240_blue", "파랑"), ("pink", "source240_pink", "분홍"), ("orange", "shaker_c_orange", "주황"))
GRASP_S_SIDE_COLORS = {"right": ("blue", "pink"), "left": ("orange",)}
GRASP_S_DEFAULT = {"right": "blue", "left": "orange"}


def grasp_s_head(color: str, side: str) -> str:
    """잡기 학생 색 → 에피소드 머리(ctl_<머리>_<팔>)."""
    return "grasp_s" if GRASP_S_DEFAULT[side] == color else f"grasp_s_{color}"


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
    return (f"사전 파지 자리 {cfg.ready_tol_m * 100:g} cm 안 · 손바닥 {cfg.still_s:g} s 평균 ≤ {cfg.still_mps:g} m/s "
            f"· 못 오면 {cfg.approach_max_s:g} s 에 멈춤")


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
                    sub = "놓기 · 손 빼기" if b.kind == "place" else "파지 · 들기"     # ★10.22 경로 F + 놓기(n05)
                    out.append({**one, "kind": "policy", "id": b.policy, "text": short_name(b.policy), "sub": sub})
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
        if head == "grasp_f":                         # ★11.09 팔마다 물체 버튼 — 고른 물체의 에피소드 · 관리 작업
            for side in ARMS:
                objs = []
                for obj, cup, color in F_OBJECTS:
                    name = f"{f_head(obj, side)}_{side}"
                    if f"episode_{name}" in stage_ids:
                        objs.append({**_episode(repo, name, entries), "stage": f"episode_{name}",
                                     "action": f"{CONTROL_PREFIX}{name}", "runner": f"episode_runner_{side}",
                                     "object": obj, "cup": cup, "color": color, "default": obj == F_DEFAULT[side]})
                if side in sides and objs:
                    sides[side]["objects"] = objs
        if head == "grasp_s":                         # ★11.11 팔마다 색 버튼 — 고른 색의 에피소드 · 관리 작업(홀더 없음)
            cups_cup = _cups_right_cup(repo)
            for side in [a for a in ARMS if a in sides]:
                objs = []
                for color, fpp, color_ko in GRASP_S_COLORS:
                    name = f"{grasp_s_head(color, side)}_{side}"
                    if color in GRASP_S_SIDE_COLORS[side] and f"episode_{name}" in stage_ids:
                        objs.append({**_episode(repo, name, entries), "stage": f"episode_{name}",
                                     "action": f"{CONTROL_PREFIX}{name}", "runner": f"episode_runner_{side}",
                                     "object": color, "fpp": fpp, "cup": fpp, "color": color_ko,
                                     "rescan": side == "right" and fpp != cups_cup,   # place_m 과 같은 cups 경고
                                     "default": color == GRASP_S_DEFAULT[side]})
                if len(objs) > 1:
                    sides[side]["objects"] = objs
        if head == "place_m" and "right" in sides:    # ★11.11 오른손 색 · 홀더 버튼 — 고른 조합의 에피소드 · 관리 작업
            objs, cups_cup = [], _cups_right_cup(repo)
            for color, fpp, color_ko in PLACE_M_COLORS:
                for holder in PLACE_M_HOLDERS:
                    name = f"{place_m_head(color, holder)}_right"
                    if f"episode_{name}" in stage_ids:
                        objs.append({**_episode(repo, name, entries), "stage": f"episode_{name}",
                                     "action": f"{CONTROL_PREFIX}{name}", "runner": "episode_runner_right",
                                     "object": f"{color}_h{holder}", "fpp": fpp, "cup": fpp, "color": color_ko,
                                     "holder": holder, "warn": PLACE_M_WARN.get(holder, ""), "rescan": fpp != cups_cup,
                                     "default": (color, holder) == PLACE_M_DEFAULT})
            if len(objs) > 1:
                sides["right"]["objects"] = objs
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
    used = {s["id"] for it in items for side in it["sides"].values() for one in (side, *side.get("objects", ()))
            for s in one["steps"] if s["kind"] == "policy" and s.get("id")}
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

"""에피소드 정의(config/episodes/*.yaml) — 정책을 차례로 잇는 순서 · 기대 상태 · 역할 → 등록부 정책. 순수(ROS 없음).

10.04 사용자: sim2real_episode_implementation_guide.md(Step 1 정상 파이프라인 · Step 2 실패 처리, VLM 전까지)를 deploy 에.
순서는 YAML 만 바꿔 바꾼다 — 실행기(episode_runner)에 정책 이름을 박지 않는다. 정책 역할(rh_aglt_r …)은 이 파일의
`policies:` 가 등록부(deploy/policies/<id>) 정책 · 계열(kind) · 팔에 묶는다.

    episode: {name, version}
    setting: {right: [x, y, z], left: [...]}      # 공통 중간 자세 = aglt 목표 = place · pour 인계 시작(손에 든 컵 원점, base m)
    objects: {CUP: {topic: /objects/cyl60/pose}}   # 물체 이름 → FP++ 자세 토픽 — 에피소드 처음 snapshot 이 한 번에 기록한다
    holders: {CENTER_HOLDER: 1}                    # 홀더 이름 → 마커 id — 홀더는 고정, 자세는 holder_poses 파일
    holder_poses: config/cup_holder_poses_5090_v1.yaml   # cup_holder_pose_node --write 가 쓴 고정 자세(10.04 사용자 · 10.13 5090 자리)
    policies: {rh_aglt_r: {policy: rh56f1/aglt/right_cyl60g, kind: aglt, side: right}, ...}   # policy 없음 = 아직 없는 정책
    sequence 의 aglt 단계(또는 parallel_policy 항목)에 policy_object: source240_real — 다물체 정책(rg3)의 물체 one-hot(10.12)
    trajectories: {go_home: {kind: rehome, sides: [right, left]}}
    failure_policy: {default: 1, rh_aglt_r: 2}     # 노드당 재시도 상한(Step 2)
    sequence: [ {id, type: snapshot|trajectory|policy|parallel_policy|confirm|hand|terminal, name, ..., expect: {...}, checkpoint: NAME} ]

★10.17 경로 B(수동 파지, docs/PLAN_GRASP_TILT_EVAL_2026-10-17.md) 노드 둘:
    confirm  {id, type: confirm, text: "운영자가 확인할 것"} — 로봇을 움직이지 않는다. 연속 실행도 여기서 멈추고 이 노드 이름의
             승인을 따로 받는다(사용자 "확인 두 번 — 컵 대기 전, 닫기 전").
    hand     {id, type: hand, action: open, sides: [right, left]} — 팔은 제자리, 손을 계약 홈 손(편 손)으로(pd hand_release).
             {id, type: hand, action: grasp, sides: [right], hand_contract: <aglt 정책 id>, expect: {right_hand: SRC}}
             — 팔은 제자리, 손을 닿을 때까지 닫고 막힘 래치로 쥔 판정(tools/hand_grasp.py · 경로 C 와 같은 규칙). 빈손에서만.

snapshot = 10.04 사용자 "처음에 FP++ 로 각 컵들 배치를 한번에 기록" — objects 의 자세를 한 번에 재서(정지 · 신선) 기록하고,
그 뒤 정책은 그 기록을 본다(컵은 잡히기 전까지 움직이지 않는다 · 잡은 뒤는 손바닥 FK).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import yaml

NODE_TYPES = ("snapshot", "trajectory", "policy", "parallel_policy", "confirm", "hand", "terminal")
HAND_ACTIONS = ("open", "grasp")
KINDS = ("aglt", "place", "pour", "lock", "shake")
SIDES = ("right", "left", "both")
TRAJ_KINDS = ("rehome", "noop", "carry", "tilt", "reach", "lift")
#: 10.17 실측에서 계획 · 검사한 뒤 재생하는 궤적(tools/plan_<kind>.py → replay_to_pd).
#: carry = 잡기 뒤 ready_carry 로 붓기 시작 자세(SETTING)까지 — 한 팔씩(sides: [right]) 또는 두 팔(계획 경로 A "한 팔씩").
#:   cups: false = 빈손으로(경로 B, 사람이 컵을 손에 댄다 — 컵 없이 같은 경로).
#: tilt = D4 기본 궤적(잔차 0) + 복귀, 두 팔 14 관절 한 시각표 · 두 손이 컵을 쥔 때만. 자산 = config/<asset>/policy.yaml
#: ★10.19 1-B(정책 없이 제어 모듈만): reach = ready_carry 로 컵 뒤까지 → 편 손으로 곧게 들어가 tilt 기준 손 안 컵 자리
#:   (tools/plan_reach.py — 빈손, 한 팔, object = 그 팔이 쥘 episode 물체 이름) · lift = 쥔 뒤 손바닥 방향 고정 수직 10 cm
#:   + 3 s 유지(tools/plan_lift.py — 컵을 쥔 손, 한 팔, object 가 있으면 끝에 FP++ 로 컵이 들렸는지 본다).
PLANNED_TRAJ = ("carry", "tilt", "reach", "lift")
ONE_ARM_TRAJ = ("reach", "lift")
TRAJ_ASSETS = Path(__file__).resolve().parents[1] / "config"
POSES = ("HOME", "SETTING")
EMPTY = "EMPTY"
HAND_KEYS = {"right_hand": "right", "left_hand": "left"}


class EpisodeSpecError(ValueError):
    pass


@dataclass(frozen=True)
class Binding:
    role: str
    kind: str
    side: str
    policy: str | None            # 등록부 id — None 이면 아직 없는 정책(dry-run 에서만 돈다)
    #: ★10.17 경로 C(approach_grasp.py) — aglt 노드가 접근 뒤 스크립트 닫기 · 쥠 확인 · 들기 · 유지를 한다. None = 끔,
    #:   {} = 기본값, 키 = ApproachGraspCfg 필드 덮어쓰기(모르는 키 거부)
    approach_grasp: Mapping | None = None
    #: ★10.20 경로 D — approach_grasp {handoff: 1} 일 때 접근 정책이 사전 파지 자리에 멈추면 같은 노드에서 넘겨받는 정책의
    #:   등록부 id(aglt · arm_start ready, 예 rh56f1/aglt/right_rcg03). None = 넘김 없음
    handoff: str | None = None


@dataclass(frozen=True)
class Job:
    role: str
    source_object: str | None = None
    target_object: str | None = None
    target_holder: str | None = None
    #: ★10.12 다물체 aglt 정책(rh_aglt object_names)의 물체 이름(one-hot id, 예 source240_real) — 실기 물체는 단계마다 고정
    policy_object: str | None = None


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    name: str = ""                # trajectory 이름 또는 policy 역할
    jobs: tuple = ()              # policy · parallel_policy 의 Job 들
    expect: Mapping = field(default_factory=dict)
    checkpoint: str | None = None
    result: str | None = None     # terminal
    text: str = ""                # confirm — 운영자가 확인할 것
    action: str = ""              # hand — open | grasp
    sides: tuple = ()             # hand — 팔들(grasp 는 하나)
    hand_contract: str = ""       # hand grasp — 손 열림 · 쥠 자세 · 막힘 명목값을 줄 계약(등록부 id 또는 저장소 기준 .json, 10.20)
    cfg: Mapping = field(default_factory=dict)   # hand grasp — 닫기 · 판정 값 덮어쓰기(tools/hand_grasp.GraspCfg)


@dataclass(frozen=True)
class Trajectory:
    name: str
    kind: str
    sides: tuple
    asset: str = ""              # carry · tilt: config/ 기준 자산 폴더(policy.yaml · ref/)
    rate_scale: float = 1.0      # carry · tilt 재생 배속(0, 1]
    cups: bool = True            # carry · tilt: 그 팔 손이 컵을 쥐었을 때만(False = 빈손일 때만, carry 만)
    object: str = ""             # reach · lift: episode objects 의 이름(컵 자세 FP++ 토픽)


@dataclass(frozen=True)
class Episode:
    name: str
    version: int
    nodes: tuple
    policies: Mapping[str, Binding]
    trajectories: Mapping[str, Trajectory]
    objects: Mapping[str, Mapping]
    holders: Mapping[str, int]
    setting: Mapping[str, tuple]
    failure_policy: Mapping[str, int]
    holder_poses: str = ""
    path: str = ""

    def max_retry(self, node: Node) -> int:
        roles = [j.role for j in node.jobs] or [node.name]
        return min(int(self.failure_policy.get(r, self.failure_policy.get("default", 1))) for r in roles)

    def node_index(self, node_id: str) -> int:
        return next(i for i, n in enumerate(self.nodes) if n.id == node_id)


def _job(raw: Mapping, where: str) -> Job:
    unknown = set(raw) - {"name", "source_object", "target_object", "target_holder", "policy_object"}
    if unknown:
        raise EpisodeSpecError(f"{where}: 모르는 키 {sorted(unknown)}")
    if not raw.get("name"):
        raise EpisodeSpecError(f"{where}: name(정책 역할)이 없다")
    return Job(role=str(raw["name"]), source_object=raw.get("source_object"), target_object=raw.get("target_object"),
               target_holder=raw.get("target_holder"),
               policy_object=None if raw.get("policy_object") is None else str(raw["policy_object"]))


def _node(raw: Mapping, i: int) -> Node:
    where = f"sequence[{i}] {raw.get('id', '?')}"
    t = raw.get("type")
    if t not in NODE_TYPES:
        raise EpisodeSpecError(f"{where}: type {t!r} 는 {NODE_TYPES} 중 하나")
    if not raw.get("id"):
        raise EpisodeSpecError(f"{where}: id 가 없다")
    jobs: tuple = ()
    if t == "policy":
        jobs = (_job({k: raw[k] for k in ("name", "source_object", "target_object", "target_holder", "policy_object") if k in raw}, where),)
    elif t == "parallel_policy":
        jobs = tuple(_job(j, f"{where}.policies[{k}]") for k, j in enumerate(raw.get("policies") or []))
        if len(jobs) < 2:
            raise EpisodeSpecError(f"{where}: parallel_policy 는 정책 둘 이상")
    elif t == "trajectory" and not raw.get("name"):
        raise EpisodeSpecError(f"{where}: trajectory name 이 없다")
    elif t == "confirm" and not str(raw.get("text") or "").strip():
        raise EpisodeSpecError(f"{where}: confirm 은 text(운영자가 확인할 것)가 있어야 한다")
    elif t == "hand":
        _check_hand(raw, where)
    return Node(id=str(raw["id"]), type=t, name=str(raw.get("name") or ""), jobs=jobs, expect=dict(raw.get("expect") or {}),
                checkpoint=raw.get("checkpoint"), result=raw.get("result"), text=str(raw.get("text") or ""),
                action=str(raw.get("action") or ""), sides=tuple(raw.get("sides") or ()),
                hand_contract=str(raw.get("hand_contract") or ""), cfg=dict(raw.get("cfg") or {}))


def _check_hand(raw: Mapping, where: str) -> None:
    action, sides = raw.get("action"), list(raw.get("sides") or [])
    if action not in HAND_ACTIONS:
        raise EpisodeSpecError(f"{where}: hand action 은 {HAND_ACTIONS}")
    if not sides or any(s not in ("right", "left") for s in sides) or len(set(sides)) != len(sides):
        raise EpisodeSpecError(f"{where}: hand sides 는 right · left 중 겹치지 않게")
    if action == "grasp":
        if len(sides) != 1:
            raise EpisodeSpecError(f"{where}: hand grasp 는 한 손씩 — sides: [right] 또는 [left]")
        if not raw.get("hand_contract"):
            raise EpisodeSpecError(f"{where}: hand grasp 는 hand_contract(손 자세 · 막힘 명목값을 줄 aglt 정책 id)가 필요하다")
        key = f"{sides[0]}_hand"
        if not (raw.get("expect") or {}).get(key) or (raw.get("expect") or {}).get(key) == EMPTY:
            raise EpisodeSpecError(f"{where}: hand grasp 는 expect.{key} 에 쥘 물체 이름을 둔다")
        if not isinstance(raw.get("cfg") or {}, Mapping):
            raise EpisodeSpecError(f"{where}: hand grasp cfg 는 {{키: 값}}")
        from policy_control.hand_grasp import GraspCfg, HandGraspError     # 무거운 의존은 쓸 때만
        try:
            GraspCfg.from_mapping(raw.get("cfg") or {})
        except HandGraspError as exc:
            raise EpisodeSpecError(f"{where}: {exc}") from exc
    elif raw.get("hand_contract") or raw.get("cfg"):
        raise EpisodeSpecError(f"{where}: hand open 은 hand_contract · cfg 를 받지 않는다")


#: 실기 FP++ 활성 물체(scripts/ops/fpp_object.py activate) — 에피소드 objects 의 {active: <팔>} 이 이것을 따른다(10.09)
ACTIVE_FPP = "fpp_active.yaml"
_REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"


def resolve_objects(raw_objs: Mapping, path: str) -> dict[str, dict]:
    """{이름: {topic: ..}} 그대로, {이름: {active: right|left}} 는 config/fpp_active.yaml 의 그 팔 물체 토픽으로."""
    out: dict[str, dict] = {}
    active = None
    for k, v in (raw_objs or {}).items():
        o = dict(v or {})
        if "active" in o:
            side = o["active"]
            if side not in SIDES:
                raise EpisodeSpecError(f"objects.{k}: active 는 {SIDES} 중 하나")
            if active is None:
                cfg = (Path(path).resolve().parent.parent if path else _REPO_CONFIG) / ACTIVE_FPP
                if not cfg.is_file():
                    cfg = _REPO_CONFIG / ACTIVE_FPP
                active = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("sides") or {}
            if side not in active:
                raise EpisodeSpecError(f"objects.{k}: active {side} — {ACTIVE_FPP} 에 그 팔 물체가 없다")
            o["topic"] = f"/objects/{active[side]}/pose"
        out[str(k)] = o
    return out


def parse(raw: Mapping, *, path: str = "") -> Episode:
    if not isinstance(raw, Mapping) or "sequence" not in raw:
        raise EpisodeSpecError(f"{path}: episode · sequence 가 있어야 한다")
    meta = raw.get("episode") or {}
    pols = {}
    for role, b in (raw.get("policies") or {}).items():
        if b.get("kind") not in KINDS or b.get("side") not in SIDES:
            raise EpisodeSpecError(f"policies.{role}: kind {KINDS} · side {SIDES} 가 필요하다")
        ag = b.get("approach_grasp")
        if ag is not None:
            if b.get("kind") != "aglt" or not isinstance(ag, Mapping):
                raise EpisodeSpecError(f"policies.{role}: approach_grasp 는 aglt 정책의 {{키: 값}}(빈 {{}} = 기본값)")
            from policy_control.approach_grasp import ApproachGraspCfg, ApproachGraspError   # 무거운 의존은 쓸 때만
            try:
                ApproachGraspCfg.to_param(dict(ag))
            except (ApproachGraspError, TypeError, ValueError) as exc:
                raise EpisodeSpecError(f"policies.{role}: approach_grasp — {exc}") from exc
            ag = dict(ag)
        handoff = b.get("handoff")
        if (handoff is not None) != bool(ag and int(ag.get("handoff", 0))):
            raise EpisodeSpecError(f"policies.{role}: handoff(넘겨받을 정책 id)와 approach_grasp {{handoff: 1}} 은 같이 쓴다")
        if handoff is not None and not (isinstance(handoff, str) and handoff.strip()):
            raise EpisodeSpecError(f"policies.{role}: handoff 는 등록부 정책 id 문자열")
        pols[str(role)] = Binding(role=str(role), kind=str(b["kind"]), side=str(b["side"]), policy=b.get("policy"),
                                  approach_grasp=ag, handoff=handoff)
    trajs = {}
    for name, t in (raw.get("trajectories") or {}).items():
        if t.get("kind") not in TRAJ_KINDS:
            raise EpisodeSpecError(f"trajectories.{name}: kind 는 {TRAJ_KINDS}")
        trajs[str(name)] = _trajectory(str(name), t)
    ep = Episode(name=str(meta.get("name", Path(path).stem)), version=int(meta.get("version", 1)),
                 nodes=tuple(_node(n, i) for i, n in enumerate(raw["sequence"])), policies=pols, trajectories=trajs,
                 objects=resolve_objects(raw.get("objects") or {}, path),
                 holders={str(k): int(v) for k, v in (raw.get("holders") or {}).items()},
                 setting={str(k): tuple(float(x) for x in v) for k, v in (raw.get("setting") or {}).items()},
                 failure_policy={str(k): int(v) for k, v in (raw.get("failure_policy") or {"default": 1}).items()},
                 holder_poses=str(raw.get("holder_poses") or ""), path=path)
    validate(ep)
    return ep


def _trajectory(name: str, t: Mapping) -> Trajectory:
    cups = t.get("cups", True)
    if not isinstance(cups, bool):
        raise EpisodeSpecError(f"trajectories.{name}: cups 는 true/false")
    tr = Trajectory(name=name, kind=str(t["kind"]), sides=tuple(t.get("sides") or ("right", "left")),
                    asset=str(t.get("asset") or ""), rate_scale=float(t.get("rate_scale", 1.0)), cups=cups,
                    object=str(t.get("object") or ""))
    if tr.object and tr.kind not in ONE_ARM_TRAJ:
        raise EpisodeSpecError(f"trajectories.{name}: object 는 {ONE_ARM_TRAJ} 에만")
    if tr.kind not in PLANNED_TRAJ:
        if tr.asset or "cups" in t:
            raise EpisodeSpecError(f"trajectories.{name}: asset · cups 는 {PLANNED_TRAJ} 에만")
        return tr
    if tr.kind in ONE_ARM_TRAJ:
        if tuple(tr.sides) not in (("right",), ("left",)):
            raise EpisodeSpecError(f"trajectories.{name}: {tr.kind} 는 한 팔 — sides: [right] · [left]")
        if "cups" not in t or tr.cups != (tr.kind == "lift"):
            raise EpisodeSpecError(f"trajectories.{name}: reach 는 빈손(cups: false) · lift 는 쥔 손(cups: true) — 적어 둘 것")
        if tr.kind == "reach" and not tr.object:
            raise EpisodeSpecError(f"trajectories.{name}: reach 는 object(쥘 episode 물체)가 있어야 한다")
    if tr.kind == "tilt" and (tuple(tr.sides) != ("right", "left") or not tr.cups):
        raise EpisodeSpecError(f"trajectories.{name}: tilt 는 두 팔 한 번에, 두 손이 컵을 쥔 채 — sides: [right, left] · cups: true")
    if tr.kind == "carry" and tuple(tr.sides) not in (("right",), ("left",), ("right", "left")):
        raise EpisodeSpecError(f"trajectories.{name}: carry sides 는 [right] · [left] · [right, left]")
    if not tr.asset or not (TRAJ_ASSETS / tr.asset / "policy.yaml").is_file():
        raise EpisodeSpecError(f"trajectories.{name}: asset {tr.asset!r} 에 policy.yaml 이 없다({TRAJ_ASSETS})")
    if not 0.0 < tr.rate_scale <= 1.0:
        raise EpisodeSpecError(f"trajectories.{name}: rate_scale 은 (0, 1] — {tr.rate_scale}")
    return tr


def episode_sides(ep: Episode) -> list[str]:
    """에피소드가 움직이는 팔(정책 · 실행 노드의 궤적 · 손 노드) — 미션 단계가 그 팔 홈 · pd 를 기다린다. 우 → 좌 순."""
    used = {ep.policies[j.role].side for n in ep.nodes for j in n.jobs if j.role in ep.policies}
    used |= {s for n in ep.nodes if n.type == "trajectory" and n.name in ep.trajectories for s in ep.trajectories[n.name].sides}
    used |= {s for n in ep.nodes if n.type == "hand" for s in n.sides}
    out = [s for s in ("right", "left") if s in used or "both" in used]
    return out


def load(path: str | Path) -> Episode:
    return parse(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}, path=str(path))


#: ★10.12 물체 one-hot 을 받는 정책 종류 — rh_aglt 다물체(rg3) · rh_place 다물체(rg3nt, 잡기와 같은 물체)
OBJECT_KINDS = ("aglt", "place")


def override_grasp_object(ep: Episode, *, policy_object: str = "", cup_object: str = "") -> Episode:
    """★10.12 실행기 파라미터로 물체를 바꾼 새 에피소드 — policy_object 는 aglt 단계의 물체 one-hot 이름,
    cup_object 는 FP++ 물체 이름(/objects/<이름>/pose). 빈 값은 YAML 기본을 둔다. 원본은 바꾸지 않는다.
    계약에 없는 이름은 여기서 막지 않고 availability 가 이유를 댄다(시작 검사)."""
    po, co = policy_object.strip(), cup_object.strip()
    nodes = tuple(
        replace(n, jobs=tuple(replace(j, policy_object=po) if po and _takes_object(ep, j) else j for j in n.jobs))
        for n in ep.nodes)
    objects = {k: ({**o, "topic": f"/objects/{co}/pose"} if co else o) for k, o in ep.objects.items()}
    if co and len(objects) != 1:
        raise EpisodeSpecError(f"cup_object 는 물체가 하나인 에피소드에만({list(ep.objects)})")
    return replace(ep, nodes=nodes, objects=objects)


def validate(ep: Episode) -> None:
    errs: list[str] = []
    ids = [n.id for n in ep.nodes]
    if len(set(ids)) != len(ids):
        errs.append(f"노드 id 가 겹친다: {[i for i in ids if ids.count(i) > 1]}")
    if not ep.nodes or ep.nodes[-1].type != "terminal":
        errs.append("마지막 노드는 terminal 이어야 한다")
    known_objects = set(ep.objects) | {EMPTY}
    topics = [o.get("topic") for o in ep.objects.values()]
    if any(not t for t in topics) or len(set(topics)) != len(topics):
        errs.append("objects 마다 서로 다른 topic 이 있어야 한다(같은 토픽 둘 = FP++ 가 못 가른다)")
    if ep.holders and not ep.holder_poses:
        errs.append("holders 를 쓰면 holder_poses(고정 자세 파일)가 필요하다")
    if any(n.type in ("policy", "parallel_policy") for n in ep.nodes) and ep.objects:
        first_policy = next(i for i, n in enumerate(ep.nodes) if n.type in ("policy", "parallel_policy"))
        if not any(n.type == "snapshot" for n in ep.nodes[:first_policy]):
            errs.append("첫 정책 노드 앞에 snapshot(물체 배치 기록)이 있어야 한다")
    for n in ep.nodes:
        for j in n.jobs:
            if j.role not in ep.policies:
                errs.append(f"{n.id}: 정책 역할 {j.role!r} 가 policies 에 없다")
            for o in (j.source_object, j.target_object):
                if o is not None and o not in ep.objects:
                    errs.append(f"{n.id}: 물체 {o!r} 가 objects 에 없다")
            if j.target_holder is not None and j.target_holder not in ep.holders:
                errs.append(f"{n.id}: 홀더 {j.target_holder!r} 가 holders 에 없다")
            b = ep.policies.get(j.role)
            if j.policy_object is not None and (b is None or b.kind not in OBJECT_KINDS or not j.policy_object):
                errs.append(f"{n.id}: policy_object {j.policy_object!r} 는 {OBJECT_KINDS} 정책 단계에만(물체 one-hot 이름)")
        if n.type == "trajectory" and n.name not in ep.trajectories:
            errs.append(f"{n.id}: trajectory {n.name!r} 가 trajectories 에 없다")
        elif n.type == "trajectory" and ep.trajectories[n.name].object and ep.trajectories[n.name].object not in ep.objects:
            errs.append(f"{n.id}: trajectory {n.name!r} 의 object {ep.trajectories[n.name].object!r} 가 objects 에 없다")
        if n.type in ("confirm", "terminal") and (n.expect or n.checkpoint):
            errs.append(f"{n.id}: {n.type} 는 expect · checkpoint 를 받지 않는다")
        if n.type == "hand" and n.action == "open" and any(k in n.expect for k in HAND_KEYS):
            errs.append(f"{n.id}: hand open 은 빈손에서만 한다 — expect 에 손 물체를 두지 않는다")
        if n.type == "hand" and n.action == "grasp":
            i = ep.nodes.index(n)
            # 사용자 "확인 두 번(컵 대기 전, 닫기 전)" — 바로 앞 두 노드가 confirm 이어야 한다(YAML 편집 실수로 빠지지 않게)
            # ★10.19 1-B: 로봇이 reach 로 컵에 다가간 경우(사람이 컵을 대지 않는다)는 사용자 결정 "승인 = 닫기 전만" —
            #   바로 앞이 confirm 하나이고 그 앞이 같은 팔 reach 궤적이어야 한다.
            prev2 = ep.nodes[i - 2] if i >= 2 else None
            reached = (prev2 is not None and prev2.type == "trajectory" and prev2.name in ep.trajectories
                       and ep.trajectories[prev2.name].kind == "reach" and tuple(ep.trajectories[prev2.name].sides) == tuple(n.sides))
            if i < 1 or ep.nodes[i - 1].type != "confirm":
                errs.append(f"{n.id}: hand grasp 바로 앞에 confirm(닫기 전)이 있어야 한다")
            elif not reached and (prev2 is None or prev2.type != "confirm"):
                errs.append(f"{n.id}: hand grasp 바로 앞에 confirm 두 개(컵 대기 전 · 닫기 전)가 있어야 한다"
                            f"(같은 팔 reach 바로 뒤면 닫기 전 하나)")
            # 재시도 · 되돌아가기는 확인 없이 다시 닫거나 사람이 컵을 든 채 팔을 옮긴다(리뷰) — 실패면 언제나 정지
            if ep.max_retry(n) != 0:
                errs.append(f"{n.id}: hand grasp 는 재시도 상한 0 이어야 한다(failure_policy default 0) — 실패면 정지")
        pose = n.expect.get("pose")
        if pose is not None and pose not in POSES:
            errs.append(f"{n.id}: expect.pose {pose!r} 는 {POSES}")
        for key, side in HAND_KEYS.items():
            if key in n.expect and n.expect[key] not in known_objects:
                errs.append(f"{n.id}: expect.{key} {n.expect[key]!r} 는 objects 이름 또는 {EMPTY}")
        flags = {k: v for k, v in n.expect.items() if k not in ("pose", *HAND_KEYS)}
        bad = [k for k, v in flags.items() if not isinstance(v, bool)]
        if bad:
            errs.append(f"{n.id}: 진행 플래그는 true/false: {bad}")
        if n.checkpoint is not None and n.type not in ("snapshot", "trajectory", "policy", "parallel_policy", "hand"):
            errs.append(f"{n.id}: checkpoint 는 실행 노드에만")
        sides = [ep.policies[j.role].side for j in n.jobs if j.role in ep.policies]
        if n.type == "parallel_policy" and len(set(sides)) != len(sides):
            errs.append(f"{n.id}: 같은 팔 정책을 동시에 돌릴 수 없다 {sides}")
    if errs:
        raise EpisodeSpecError(f"{ep.path}: " + " · ".join(errs))


def availability(ep: Episode, entries: Mapping[str, object]) -> dict[str, str]:
    """역할 → 실기에서 못 쓰는 이유(쓸 수 있으면 ""). entries = policy_registry.scan 의 {id: Entry}."""
    used = {j.role for n in ep.nodes for j in n.jobs}
    out = {}
    for role in sorted(used):
        b = ep.policies[role]
        out[role] = ""
        for pid in (b.policy, *((b.handoff,) if b.handoff else ())):   # ★10.20 경로 D 는 넘겨받는 정책도 써야 한다
            e = entries.get(pid) if pid else None
            if pid is None:
                out[role] = "정책이 아직 없다(policy 칸 비움)"
            elif e is None:
                out[role] = f"등록부에 {pid} 가 없다"
            elif not getattr(e, "contract", ""):
                out[role] = f"{pid}: 계약이 없다"
            elif getattr(e, "status", "") == "hold":
                out[role] = f"{pid}: status hold"
            elif getattr(e, "issues", ()):
                out[role] = f"{pid}: " + " / ".join(e.issues)
            else:
                out[role] = _object_problem(ep, role, e)
            if out[role]:
                break
    return out


def hand_contract_file(hand_contract: str, policies_root: Path) -> Path:
    """hand grasp 손 계약 파일. ★10.20 1-B 는 정책과 떼어 고정 파일(config/hand/rh56f1_<팔>_hand.json — 옛 접근 정책 rg3wc 계약
    그대로, 정책을 지워도 손 닫기가 안 바뀐다)을 쓴다: 값이 .json 이면 저장소(policies_root 의 두 단계 위) 기준 파일,
    아니면 등록 정책 id(<policies_root>/<id>/rh_aglt_contract.json). 순수."""
    if str(hand_contract).endswith(".json"):
        return Path(policies_root).resolve().parents[1] / hand_contract
    return Path(policies_root) / hand_contract / "rh_aglt_contract.json"


def hand_problems(ep: Episode, policies_root: Path) -> dict[str, str]:
    """★10.17 hand grasp 노드 → 못 쓰는 이유(쓸 수 있으면 ""). 계약 파일이 있고 그 손 계약이어야 한다(tools/hand_grasp.py 가 읽는다)."""
    out = {}
    for n in ep.nodes:
        if n.type != "hand" or n.action != "grasp":
            continue
        p = hand_contract_file(n.hand_contract, policies_root)
        if not p.is_file():
            out[n.id] = f"{p} 가 없다"
            continue
        try:
            hand = list(((json.loads(p.read_text()).get("sides") or {}).get("arm") or {}).get("hand_joints") or [])
        except (OSError, ValueError) as exc:
            out[n.id] = f"{p}: 읽지 못함({exc})"
            continue
        out[n.id] = "" if hand and hand[0].startswith(n.sides[0][0] + "_") else f"{n.hand_contract} 는 {n.sides[0]} 손 계약이 아니다"
    return out


def _object_problem(ep: Episode, role: str, e: object) -> str:
    """★10.12 다물체 aglt 계약(object_names)이면 그 역할의 단계마다 policy_object 가 계약 물체 중 하나여야 한다."""
    path = getattr(e, "path", None)
    cpath = Path(path) / e.contract if path is not None else None
    try:
        names = list(json.loads(cpath.read_text()).get("object_names") or []) if cpath and cpath.is_file() else []
    except (OSError, ValueError) as exc:
        return f"{e.contract}: 읽지 못함({exc})"
    given = [j.policy_object for n in ep.nodes for j in n.jobs if j.role == role]
    if not names:
        return f"policy_object {sorted({g for g in given if g})} — 단일 물체 정책(계약에 object_names 없음)" if any(given) else ""
    bad = sorted({str(g) for g in given if g not in names})
    return f"다물체 정책 {names}: 단계 policy_object {bad} — 계약 물체 중 하나를 단계마다 준다" if bad else ""


def _takes_object(ep: Episode, j) -> bool:
    """실행기 파라미터 policy_object 를 받는 단계 — aglt 는 언제나, place 는 YAML 에 policy_object 가 적힌 단계만
    (단일 물체 place 정책(right_i09 등)에 물체를 억지로 넣지 않는다)."""
    kind = ep.policies[j.role].kind
    return kind == "aglt" or (kind == "place" and j.policy_object is not None)


def holder_problems(ep: Episode, poses: Mapping[int, object]) -> list[str]:
    """고정 홀더 자세 파일(holder_poses → {마커 id: 자세})에 이 에피소드가 쓰는 홀더가 다 있는가. 빈 목록 = 된다."""
    if not ep.holders:
        return []
    if not poses:
        return [f"{ep.holder_poses or '홀더 자세 파일'} 이 없거나 비었다 — 미션 cup_holders 단계(--write)로 만든다"]
    return [f"{name}(마커 {hid})이 {ep.holder_poses} 에 없다" for name, hid in ep.holders.items() if hid not in poses]

"""에피소드 정의(config/episodes/*.yaml) — 정책을 차례로 잇는 순서 · 기대 상태 · 역할 → 등록부 정책. 순수(ROS 없음).

10.04 사용자: sim2real_episode_implementation_guide.md(Step 1 정상 파이프라인 · Step 2 실패 처리, VLM 전까지)를 deploy 에.
순서는 YAML 만 바꿔 바꾼다 — 실행기(episode_runner)에 정책 이름을 박지 않는다. 정책 역할(rh_aglt_r …)은 이 파일의
`policies:` 가 등록부(deploy/policies/<id>) 정책 · 계열(kind) · 팔에 묶는다.

    episode: {name, version}
    setting: {right: [x, y, z], left: [...]}      # 공통 중간 자세 = aglt 목표 = place · pour 인계 시작(손에 든 컵 원점, base m)
    objects: {CUP: {topic: /objects/cyl60/pose}}   # 물체 이름 → FP++ 자세 토픽 — 에피소드 처음 snapshot 이 한 번에 기록한다
    holders: {CENTER_HOLDER: 1}                    # 홀더 이름 → 마커 id — 홀더는 고정, 자세는 holder_poses 파일
    holder_poses: config/cup_holder_poses_arm4090.yaml   # cup_holder_pose_node --write 가 쓴 고정 자세(10.04 사용자)
    policies: {rh_aglt_r: {policy: right_rh_aglt_cyl60g, kind: aglt, side: right}, ...}   # policy 없음 = 아직 없는 정책
    trajectories: {go_home: {kind: rehome, sides: [right, left]}}
    failure_policy: {default: 1, rh_aglt_r: 2}     # 노드당 재시도 상한(Step 2)
    sequence: [ {id, type: snapshot|trajectory|policy|parallel_policy|terminal, name, ..., expect: {...}, checkpoint: NAME} ]

snapshot = 10.04 사용자 "처음에 FP++ 로 각 컵들 배치를 한번에 기록" — objects 의 자세를 한 번에 재서(정지 · 신선) 기록하고,
그 뒤 정책은 그 기록을 본다(컵은 잡히기 전까지 움직이지 않는다 · 잡은 뒤는 손바닥 FK).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

NODE_TYPES = ("snapshot", "trajectory", "policy", "parallel_policy", "terminal")
KINDS = ("aglt", "place", "pour", "lock", "shake")
SIDES = ("right", "left", "both")
TRAJ_KINDS = ("rehome", "noop")
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


@dataclass(frozen=True)
class Job:
    role: str
    source_object: str | None = None
    target_object: str | None = None
    target_holder: str | None = None


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    name: str = ""                # trajectory 이름 또는 policy 역할
    jobs: tuple = ()              # policy · parallel_policy 의 Job 들
    expect: Mapping = field(default_factory=dict)
    checkpoint: str | None = None
    result: str | None = None     # terminal


@dataclass(frozen=True)
class Trajectory:
    name: str
    kind: str
    sides: tuple


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
    unknown = set(raw) - {"name", "source_object", "target_object", "target_holder"}
    if unknown:
        raise EpisodeSpecError(f"{where}: 모르는 키 {sorted(unknown)}")
    if not raw.get("name"):
        raise EpisodeSpecError(f"{where}: name(정책 역할)이 없다")
    return Job(role=str(raw["name"]), source_object=raw.get("source_object"), target_object=raw.get("target_object"),
               target_holder=raw.get("target_holder"))


def _node(raw: Mapping, i: int) -> Node:
    where = f"sequence[{i}] {raw.get('id', '?')}"
    t = raw.get("type")
    if t not in NODE_TYPES:
        raise EpisodeSpecError(f"{where}: type {t!r} 는 {NODE_TYPES} 중 하나")
    if not raw.get("id"):
        raise EpisodeSpecError(f"{where}: id 가 없다")
    jobs: tuple = ()
    if t == "policy":
        jobs = (_job({k: raw[k] for k in ("name", "source_object", "target_object", "target_holder") if k in raw}, where),)
    elif t == "parallel_policy":
        jobs = tuple(_job(j, f"{where}.policies[{k}]") for k, j in enumerate(raw.get("policies") or []))
        if len(jobs) < 2:
            raise EpisodeSpecError(f"{where}: parallel_policy 는 정책 둘 이상")
    elif t == "trajectory" and not raw.get("name"):
        raise EpisodeSpecError(f"{where}: trajectory name 이 없다")
    return Node(id=str(raw["id"]), type=t, name=str(raw.get("name") or ""), jobs=jobs, expect=dict(raw.get("expect") or {}),
                checkpoint=raw.get("checkpoint"), result=raw.get("result"))


def parse(raw: Mapping, *, path: str = "") -> Episode:
    if not isinstance(raw, Mapping) or "sequence" not in raw:
        raise EpisodeSpecError(f"{path}: episode · sequence 가 있어야 한다")
    meta = raw.get("episode") or {}
    pols = {}
    for role, b in (raw.get("policies") or {}).items():
        if b.get("kind") not in KINDS or b.get("side") not in SIDES:
            raise EpisodeSpecError(f"policies.{role}: kind {KINDS} · side {SIDES} 가 필요하다")
        pols[str(role)] = Binding(role=str(role), kind=str(b["kind"]), side=str(b["side"]), policy=b.get("policy"))
    trajs = {}
    for name, t in (raw.get("trajectories") or {}).items():
        if t.get("kind") not in TRAJ_KINDS:
            raise EpisodeSpecError(f"trajectories.{name}: kind 는 {TRAJ_KINDS}")
        trajs[str(name)] = Trajectory(name=str(name), kind=str(t["kind"]), sides=tuple(t.get("sides") or ("right", "left")))
    ep = Episode(name=str(meta.get("name", Path(path).stem)), version=int(meta.get("version", 1)),
                 nodes=tuple(_node(n, i) for i, n in enumerate(raw["sequence"])), policies=pols, trajectories=trajs,
                 objects={str(k): dict(v or {}) for k, v in (raw.get("objects") or {}).items()},
                 holders={str(k): int(v) for k, v in (raw.get("holders") or {}).items()},
                 setting={str(k): tuple(float(x) for x in v) for k, v in (raw.get("setting") or {}).items()},
                 failure_policy={str(k): int(v) for k, v in (raw.get("failure_policy") or {"default": 1}).items()},
                 holder_poses=str(raw.get("holder_poses") or ""), path=path)
    validate(ep)
    return ep


def load(path: str | Path) -> Episode:
    return parse(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}, path=str(path))


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
        if n.type == "trajectory" and n.name not in ep.trajectories:
            errs.append(f"{n.id}: trajectory {n.name!r} 가 trajectories 에 없다")
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
        if n.checkpoint is not None and n.type not in ("snapshot", "trajectory", "policy", "parallel_policy"):
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
        e = entries.get(b.policy) if b.policy else None
        if b.policy is None:
            out[role] = "정책이 아직 없다(policy 칸 비움)"
        elif e is None:
            out[role] = f"등록부에 {b.policy} 가 없다"
        elif not getattr(e, "contract", ""):
            out[role] = f"{b.policy}: 계약이 없다"
        elif getattr(e, "status", "") == "hold":
            out[role] = f"{b.policy}: status hold"
        elif getattr(e, "issues", ()):
            out[role] = f"{b.policy}: " + " / ".join(e.issues)
        else:
            out[role] = ""
    return out


def holder_problems(ep: Episode, poses: Mapping[int, object]) -> list[str]:
    """고정 홀더 자세 파일(holder_poses → {마커 id: 자세})에 이 에피소드가 쓰는 홀더가 다 있는가. 빈 목록 = 된다."""
    if not ep.holders:
        return []
    if not poses:
        return [f"{ep.holder_poses or '홀더 자세 파일'} 이 없거나 비었다 — 미션 cup_holders 단계(--write)로 만든다"]
    return [f"{name}(마커 {hid})이 {ep.holder_poses} 에 없다" for name, hid in ep.holders.items() if hid not in poses]

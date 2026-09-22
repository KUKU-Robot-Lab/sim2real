"""배포 프로파일 — 콘솔이 **무엇을** 띄우는가를 한 파일로 고정한다.

프로파일은 계약을 복제하지 않는다. 미션 yaml 과 정책 카드를 **가리키기만** 하고,
콘솔이 스스로 소유하는 사실은 둘뿐이다: 어느 DDS 도메인인가, 그것이 실기인가 fake 인가.

도메인 방어: 실기 도메인(126)과 fake 가 섞이는 조합은 **로드 시점에** 거부한다.
기존 launch 의 `_refuse_real_domain` 은 ""/"0" 만 막는다 — 실기가 126 에서 도는 이 작업장에서는
그것만으로는 fake 가 실기 그래프에 붙는 것을 막지 못한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .diagram_spec import Diagram, parse_diagram
from .errors import ProfileError

SCHEMA = "s2r_console/profile/v1"
#: 이 작업장의 실기 DDS 도메인 (docs/ROBOT_MATERIALS.md, INSTALL.md).
REAL_DOMAIN = 126
CLASSES = ("real", "fake")


@dataclass(frozen=True)
class StackTopic:
    name: str
    #: 마지막 도착 뒤 이 시간이 지나면 stale. None 이면 나이로 판정하지 않는다(드문 토픽).
    stale_ms: float | None = None


@dataclass(frozen=True)
class StackManager:
    name: str                       # controller_manager 노드 전체 이름
    active: tuple[str, ...] = ()    # active 여야 하는 컨트롤러. 나머지는 상태만 보인다


@dataclass(frozen=True)
class Stack:
    """policy_control 바깥에서 붙어 있어야 하는 것 — robot_control bringup, 인지 등."""

    title: str
    nodes: tuple[str, ...]          # 그래프에 보여야 하는 노드 전체 이름
    topics: tuple[StackTopic, ...]
    managers: tuple[StackManager, ...]

    def as_dict(self) -> dict:
        return {"title": self.title, "nodes": list(self.nodes),
                "topics": [{"name": t.name, "stale_ms": t.stale_ms} for t in self.topics],
                "managers": [{"name": m.name, "active": list(m.active)} for m in self.managers]}


@dataclass(frozen=True)
class Profile:
    id: str
    title: str
    mission: Path          # 미션 yaml (repo 상대 → 절대)
    domain: int
    domain_class: str      # real | fake
    status_nodes: tuple[str, ...]
    policy_dir: Path | None  # deploy/policies/<id> — 없으면 카드 검사를 하지 않는다
    policy_dt: float | None
    path: Path
    #: 지연을 재는 두 끝 (먼저 도는 노드, 마지막 노드). 없으면 status_nodes 의 처음과 끝.
    latency: tuple[str, str] | None = None
    stack: Stack | None = None
    diagram: Diagram | None = None

    @property
    def is_real(self) -> bool:
        return self.domain_class == "real"

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "mission": str(self.mission),
                "domain": self.domain, "domain_class": self.domain_class,
                "status_nodes": list(self.status_nodes),
                "policy_dir": None if self.policy_dir is None else str(self.policy_dir),
                "policy_dt": self.policy_dt, "path": str(self.path),
                "latency": None if self.latency is None else list(self.latency),
                "stack": None if self.stack is None else self.stack.as_dict(),
                "diagram": None if self.diagram is None else self.diagram.as_dict()}


def domain_reasons(domain: int, domain_class: str) -> list[str]:
    """이 (도메인, 종류) 조합이 왜 안 되는가. 비어 있으면 된다."""
    if domain_class not in CLASSES:
        return [f"domain.class 는 {CLASSES} 중 하나여야 한다: {domain_class!r}"]
    if not 0 <= domain <= 232:
        return [f"ROS_DOMAIN_ID 는 0..232 다: {domain}"]
    if domain_class == "fake" and domain in (0, REAL_DOMAIN):
        return [f"fake 프로파일이 도메인 {domain} 을 쓴다 — 0 과 {REAL_DOMAIN}(실기) 은 fake 에 금지다"]
    if domain_class == "real" and domain != REAL_DOMAIN:
        return [f"real 프로파일의 도메인은 {REAL_DOMAIN} 이어야 한다: {domain}"]
    return []


def _names(raw: object, *, where: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not all(isinstance(n, str) for n in raw):
        raise ProfileError(f"{where} 는 문자열 목록이어야 한다")
    return tuple(raw)


def _absolute(name: object, *, where: str) -> str:
    if not isinstance(name, str) or not name.startswith("/"):
        raise ProfileError(f"{where} 는 '/' 로 시작하는 전체 이름이어야 한다: {name!r}")
    return name


def _entry(raw: object, *, keys: set[str], where: str) -> Mapping:
    """문자열이면 {"name": 문자열}. 매핑이면 name 이 있고 모르는 키가 없어야 한다."""
    entry = {"name": raw} if isinstance(raw, str) else raw
    if not isinstance(entry, Mapping) or "name" not in entry or set(entry) - keys:
        raise ProfileError(f"{where} 는 이름이거나 {sorted(keys)} 매핑이어야 한다: {raw!r}")
    return entry


def parse_stack(raw: object, *, path: Path) -> Stack:
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{path}: stack 은 매핑이어야 한다")
    unknown = set(raw) - {"title", "nodes", "topics", "managers"}
    if unknown:
        raise ProfileError(f"{path}: stack 의 모르는 키 {sorted(unknown)}")
    nodes = tuple(_absolute(n, where=f"{path}: stack.nodes") for n in _names(raw.get("nodes", []), where=f"{path}: stack.nodes"))
    if not isinstance(raw.get("topics", []), list) or not isinstance(raw.get("managers", []), list):
        raise ProfileError(f"{path}: stack.topics·managers 는 목록이어야 한다")
    topics = []
    for item in raw.get("topics", []):
        e = _entry(item, keys={"name", "stale_ms"}, where=f"{path}: stack.topics 항목")
        stale = e.get("stale_ms")
        if stale is not None and (isinstance(stale, bool) or not isinstance(stale, (int, float)) or stale <= 0):
            raise ProfileError(f"{path}: stack.topics 의 stale_ms 는 양수여야 한다: {stale!r}")
        topics.append(StackTopic(_absolute(e["name"], where=f"{path}: stack.topics"),
                                 stale_ms=None if stale is None else float(stale)))
    managers = []
    for item in raw.get("managers", []):
        e = _entry(item, keys={"name", "active"}, where=f"{path}: stack.managers 항목")
        managers.append(StackManager(_absolute(e["name"], where=f"{path}: stack.managers"),
                                     active=_names(e.get("active", []), where=f"{path}: stack.managers.active")))
    return Stack(title=str(raw.get("title", "로봇 스택")), nodes=nodes, topics=tuple(topics), managers=tuple(managers))


def parse(raw: Mapping, *, path: Path, repo: Path) -> Profile:
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{path}: 매핑이 아니다")
    if raw.get("schema") != SCHEMA:
        raise ProfileError(f"{path}: schema 가 {SCHEMA} 가 아니다: {raw.get('schema')!r}")
    unknown = set(raw) - {"schema", "id", "title", "mission", "domain", "status_nodes", "policy", "policy_dt", "latency", "stack", "diagram"}
    if unknown:
        raise ProfileError(f"{path}: 모르는 키 {sorted(unknown)}")
    for key in ("id", "mission", "domain", "status_nodes"):
        if key not in raw:
            raise ProfileError(f"{path}: '{key}' 가 없다")

    dom = raw["domain"]
    if not isinstance(dom, Mapping) or set(dom) != {"id", "class"}:
        raise ProfileError(f"{path}: domain 은 {{id, class}} 여야 한다")
    try:
        domain = int(dom["id"])
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{path}: domain.id 가 정수가 아니다") from exc
    reasons = domain_reasons(domain, str(dom["class"]))
    if reasons:
        raise ProfileError(f"{path}: " + "; ".join(reasons))

    mission = (repo / str(raw["mission"])).resolve()
    if not mission.is_file():
        raise ProfileError(f"{path}: 미션 파일이 없다: {mission}")
    nodes = tuple(str(n) for n in raw["status_nodes"])
    if not nodes:
        raise ProfileError(f"{path}: status_nodes 가 비어 있다")

    policy_dir = None
    if raw.get("policy"):
        policy_dir = (repo / str(raw["policy"])).resolve()
        if not policy_dir.is_dir():
            raise ProfileError(f"{path}: 정책 디렉터리가 없다: {policy_dir}")
    dt = raw.get("policy_dt")
    latency = None
    if raw.get("latency") is not None:
        lat = raw["latency"]
        if not isinstance(lat, Mapping) or set(lat) != {"from", "to"}:
            raise ProfileError(f"{path}: latency 는 {{from, to}} 여야 한다")
        latency = (str(lat["from"]), str(lat["to"]))
        for n in latency:
            if n not in nodes:
                raise ProfileError(f"{path}: latency 노드 {n!r} 가 status_nodes 에 없다")
    return Profile(id=str(raw["id"]), title=str(raw.get("title", raw["id"])), mission=mission,
                   domain=domain, domain_class=str(dom["class"]), status_nodes=nodes,
                   policy_dir=policy_dir, policy_dt=None if dt is None else float(dt), path=path, latency=latency,
                   stack=None if raw.get("stack") is None else parse_stack(raw["stack"], path=path),
                   diagram=None if raw.get("diagram") is None else parse_diagram(raw["diagram"], path=path, status_nodes=nodes))


def load(path: Path, *, repo: Path) -> Profile:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"{path}: 읽지 못했다: {exc}") from exc
    return parse(raw, path=path, repo=repo)


def scan(directory: Path, *, repo: Path) -> tuple[list[Profile], dict[str, str]]:
    """디렉터리의 프로파일 전부. 깨진 것은 (파일명 → 사유) 로 따로 낸다 — 조용히 숨기지 않는다."""
    good, bad = [], {}
    for p in sorted(directory.glob("*.yaml")):
        try:
            good.append(load(p, repo=repo))
        except ProfileError as exc:
            bad[p.name] = str(exc)
    return good, bad

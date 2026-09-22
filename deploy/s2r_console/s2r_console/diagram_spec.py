"""연결 그림의 **선언** — 프로파일의 `diagram:` 절. 상자(노드)와 전선(토픽)을 적는다.

그림은 ROS 그래프를 자동으로 그리지 않는다: 그래프는 "지금 떠 있는 것"만 알고 "떠 있어야 하는 것"을 모른다.
꺼진 입력이 그림에서 사라지면 끊긴 곳을 볼 수 없다. 그래서 있어야 할 배선을 선언하고, 상태는 브리지가 본 것으로 칠한다
(`diagram.build`). 선언이 실제 배선과 다르면 전선이 "받는 쪽 없음" 으로 나온다 — 조용히 맞는 척하지 않는다.

틀린 선언은 **로드 시점에** 거부한다 (없는 상자로 가는 전선, 거꾸로 가는 전선, 모르는 키).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .errors import ProfileError

_UNIT = re.compile(r"^[A-Za-z0-9_]+#\d+$")
_BOX_KEYS = {"id", "title", "col", "status", "ros", "unit", "manager", "note", "stages", "host"}
_WIRE_KEYS = {"from", "to", "topic", "label", "inputs", "meter", "on_demand", "stale_ms", "heard_by", "episodic", "muted"}


@dataclass(frozen=True)
class Box:
    id: str
    title: str
    col: int                              # 왼쪽부터 몇 번째 열. 신호는 왼쪽에서 오른쪽으로만 흐른다
    status: str | None = None             # /policy_control/status/<이름> 을 내는 노드면 그 이름
    ros: tuple[str, ...] = ()             # 이 상자에 해당하는 ROS 노드 전체 이름 — 전선의 "받는 쪽" 판정에 쓴다
    unit: str | None = None               # 이 상자를 띄우는 미션 명령 `stage#n` — 스위치가 붙는다
    manager: str | None = None            # controller_manager 이름 — 컨트롤러 상태를 상자 안에 적는다
    note: str = ""
    stages: tuple[str, ...] = ()          # 한 프로세스 안의 단계(obs → policy → fabric IK …) — 상자 안에 작은 칩으로 그린다
    host: str = ""                        # 이 프로세스가 도는 PC. 빈 값은 콘솔과 같은 PC(인지는 vision-3090 에서 돈다)

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "col": self.col, "status": self.status, "ros": list(self.ros),
                "unit": self.unit, "manager": self.manager, "note": self.note, "stages": list(self.stages),
                "host": self.host}


@dataclass(frozen=True)
class Wire:
    src: str
    dst: str
    topic: str
    label: str = ""
    inputs: tuple[str, ...] = ()          # 받는 노드가 status 의 inputs 로 스스로 보고하는 행 이름
    meter: bool = True                    # 거짓이면 브리지가 구독하지 않는다(그래프만 본다) — 주기를 모른다
    on_demand: bool = False               # 가끔 한 번 내는 값(fill_level) — 발행자가 없는 것이 정상이다
    stale_ms: float | None = None
    #: 이 토픽을 받는 ROS 노드 — 비어 있으면 받는 상자의 `ros`. ros2_control 컨트롤러는 제 이름의 노드로 구독한다
    heard_by: tuple[str, ...] = ()
    episodic: bool = False                # 에피소드가 도는 동안만 흐른다(joint_target) — 쉬는 동안 끊긴 것이 아니다
    muted: str = ""                       # 비어 있지 않으면 흐르지 않는 것이 정상이다 — 그 사유 (pd execute:=false)

    def as_dict(self) -> dict:
        return {"from": self.src, "to": self.dst, "topic": self.topic, "label": self.label, "inputs": list(self.inputs),
                "meter": self.meter, "on_demand": self.on_demand, "stale_ms": self.stale_ms,
                "heard_by": list(self.heard_by), "episodic": self.episodic, "muted": self.muted}


@dataclass(frozen=True)
class Diagram:
    boxes: tuple[Box, ...]
    wires: tuple[Wire, ...]

    def box(self, box_id: str) -> Box:
        return next(b for b in self.boxes if b.id == box_id)

    def metered(self) -> tuple[str, ...]:
        """브리지가 구독해서 세는 토픽."""
        return _unique(w.topic for w in self.wires if w.meter and not w.on_demand)

    def watched(self) -> tuple[str, ...]:
        """구독하지 않고 그래프(발행자·구독자)만 보는 토픽."""
        counted = set(self.metered())
        return _unique(w.topic for w in self.wires if w.topic not in counted)

    def units(self) -> tuple[str, ...]:
        return _unique(b.unit for b in self.boxes if b.unit)

    def as_dict(self) -> dict:
        return {"boxes": [b.as_dict() for b in self.boxes], "wires": [w.as_dict() for w in self.wires]}


def _unique(items) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _abs_names(raw: object, where: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not all(isinstance(n, str) and n.startswith("/") for n in raw):
        raise ProfileError(f"{where} 는 '/' 로 시작하는 전체 이름의 목록이어야 한다: {raw!r}")
    return tuple(raw)


def _strings(raw: object, where: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not all(isinstance(s, str) for s in raw):
        raise ProfileError(f"{where} 는 문자열 목록이어야 한다: {raw!r}")
    return tuple(raw)


def _box(raw: object, where: str) -> Box:
    if not isinstance(raw, Mapping) or not {"id", "title", "col"} <= set(raw):
        raise ProfileError(f"{where}: 상자는 id·title·col 을 가진 매핑이어야 한다: {raw!r}")
    unknown = set(raw) - _BOX_KEYS
    if unknown:
        raise ProfileError(f"{where}: 상자의 모르는 키 {sorted(unknown)}")
    col = raw["col"]
    if isinstance(col, bool) or not isinstance(col, int) or col < 0:
        raise ProfileError(f"{where}: col 은 0 이상의 정수여야 한다: {col!r}")
    unit = raw.get("unit")
    if unit is not None and not (isinstance(unit, str) and _UNIT.match(unit)):
        raise ProfileError(f"{where}: unit 은 '단계#번호' 여야 한다: {unit!r}")
    manager = raw.get("manager")
    if manager is not None and not (isinstance(manager, str) and manager.startswith("/")):
        raise ProfileError(f"{where}: manager 는 '/' 로 시작하는 전체 이름이어야 한다: {manager!r}")
    status = raw.get("status")
    return Box(id=str(raw["id"]), title=str(raw["title"]), col=col, status=None if status is None else str(status),
               ros=_abs_names(raw.get("ros", []), f"{where}.ros"), unit=unit, manager=manager, note=str(raw.get("note", "")),
               stages=_strings(raw.get("stages", []), f"{where}.stages"), host=str(raw.get("host", "")))


def _wire(raw: object, cols: Mapping[str, int], where: str) -> Wire:
    if not isinstance(raw, Mapping) or not {"from", "to", "topic"} <= set(raw):
        raise ProfileError(f"{where}: 전선은 from·to·topic 을 가진 매핑이어야 한다: {raw!r}")
    unknown = set(raw) - _WIRE_KEYS
    if unknown:
        raise ProfileError(f"{where}: 전선의 모르는 키 {sorted(unknown)}")
    src, dst, topic = str(raw["from"]), str(raw["to"]), raw["topic"]
    for end in (src, dst):
        if end not in cols:
            raise ProfileError(f"{where}: 없는 상자 {end!r}")
    if cols[src] >= cols[dst]:
        raise ProfileError(f"{where}: {src}→{dst} 가 왼쪽에서 오른쪽으로 가지 않는다 (col {cols[src]} → {cols[dst]})")
    if not isinstance(topic, str) or not topic.startswith("/"):
        raise ProfileError(f"{where}: topic 은 '/' 로 시작하는 전체 이름이어야 한다: {topic!r}")
    stale = raw.get("stale_ms")
    if stale is not None and (isinstance(stale, bool) or not isinstance(stale, (int, float)) or stale <= 0):
        raise ProfileError(f"{where}: stale_ms 는 양수여야 한다: {stale!r}")
    inputs = raw.get("inputs", [])
    if not isinstance(inputs, (list, tuple)) or not all(isinstance(i, str) for i in inputs):
        raise ProfileError(f"{where}: inputs 는 문자열 목록이어야 한다")
    return Wire(src=src, dst=dst, topic=topic, label=str(raw.get("label", "")), inputs=tuple(inputs),
                meter=bool(raw.get("meter", True)), on_demand=bool(raw.get("on_demand", False)),
                stale_ms=None if stale is None else float(stale),
                heard_by=_abs_names(raw.get("heard_by", []), f"{where}.heard_by"),
                episodic=bool(raw.get("episodic", False)), muted=str(raw.get("muted", "")))


def parse_diagram(raw: object, *, path: Path, status_nodes: Sequence[str] | None = None) -> Diagram:
    where = f"{path}: diagram"
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{where} 은 매핑이어야 한다")
    unknown = set(raw) - {"boxes", "wires"}
    if unknown:
        raise ProfileError(f"{where} 의 모르는 키 {sorted(unknown)}")
    if not isinstance(raw.get("boxes"), list) or not raw["boxes"]:
        raise ProfileError(f"{where}.boxes 가 비어 있다")
    if not isinstance(raw.get("wires", []), list):
        raise ProfileError(f"{where}.wires 는 목록이어야 한다")
    boxes = tuple(_box(b, f"{where}.boxes[{i}]") for i, b in enumerate(raw["boxes"]))
    ids = [b.id for b in boxes]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ProfileError(f"{where}: 상자 id 가 겹친다 {dup}")
    if status_nodes is not None:
        stray = sorted({b.status for b in boxes if b.status and b.status not in status_nodes})
        if stray:
            raise ProfileError(f"{where}: status {stray} 가 프로파일의 status_nodes {list(status_nodes)} 에 없다")
    cols = {b.id: b.col for b in boxes}
    wires = tuple(_wire(w, cols, f"{where}.wires[{i}]") for i, w in enumerate(raw.get("wires", [])))
    return Diagram(boxes=boxes, wires=wires)

"""승인 원장 — append-only jsonl. 승인은 **그때 본 파일에 대한** 승인이다.

한 줄 = 한 사건: `approve` 또는 `revoke`. 승인 줄은 그 순간의 basis(계약·체크포인트 다이제스트)를
같이 적는다. 나중에 파일이 바뀌면 그 승인은 집합에서 빠지고 `mission_core.gate()` 는 평소처럼
"승인이 없다" 를 낸다 — 가드를 건드리지 않고 승인만 무효화한다.

순수하다: 파일을 읽고 쓰는 두 함수(`read`, `append`)만 바깥을 본다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

KINDS = ("approve", "revoke")


@dataclass(frozen=True)
class Entry:
    kind: str
    stage: str
    operator: str
    at: str
    basis: Mapping[str, str]
    note: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "stage": self.stage, "operator": self.operator,
                "at": self.at, "basis": dict(self.basis), "note": self.note}


def parse(lines: Iterable[str]) -> list[Entry]:
    """깨진 줄은 조용히 건너뛰지 않는다 — 원장이 손상됐으면 승인을 믿을 수 없다."""
    out = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if d["kind"] not in KINDS:
                raise ValueError(f"kind={d['kind']!r}")
            out.append(Entry(kind=d["kind"], stage=str(d["stage"]), operator=str(d.get("operator", "")),
                             at=str(d.get("at", "")), basis=dict(d.get("basis") or {}), note=str(d.get("note", ""))))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"승인 원장 {n}번째 줄이 깨졌다: {exc}") from exc
    return out


def latest(entries: Sequence[Entry]) -> dict[str, Entry]:
    """단계별 마지막 사건."""
    last: dict[str, Entry] = {}
    for e in entries:
        last[e.stage] = e
    return last


def valid_approvals(entries: Sequence[Entry], current_basis: Mapping[str, Mapping[str, str]]) -> frozenset[str]:
    """마지막 사건이 approve 이고 그 basis 가 지금과 글자 그대로 같은 단계만."""
    return frozenset(
        stage for stage, e in latest(entries).items()
        if e.kind == "approve" and stage in current_basis and dict(e.basis) == dict(current_basis[stage])
    )


def stale_reasons(entries: Sequence[Entry], current_basis: Mapping[str, Mapping[str, str]]) -> dict[str, list[str]]:
    """승인했는데 지금은 무효인 단계 → 무엇이 바뀌었는지. 화면이 "왜 다시 승인하라는가" 에 답한다."""
    out: dict[str, list[str]] = {}
    for stage, e in latest(entries).items():
        if e.kind != "approve":
            continue
        now = dict(current_basis.get(stage, {}))
        was = dict(e.basis)
        if stage not in current_basis:
            out[stage] = ["이 단계가 지금 미션에 없다"]
        elif was != now:
            changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
            out[stage] = [f"{k}: 승인 {(_short(was.get(k)))} → 지금 {_short(now.get(k))}" for k in changed]
    return out


def _short(digest: str | None) -> str:
    return "없음" if digest is None else digest[:8]


# ── 파일 ───────────────────────────────────────────────────────────────────
def read(path: Path) -> list[Entry]:
    if not path.is_file():
        return []
    return parse(path.read_text(encoding="utf-8").splitlines())


def append(path: Path, entry: Entry) -> None:
    if entry.kind not in KINDS:
        raise ValueError(f"kind={entry.kind!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")

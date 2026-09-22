"""`sim2real/policies/` — **쓸 정책만** 모아 두는 곳의 규약과 점검 (rclpy·torch 무의존).

`logs/policy/` 에는 배포본·자산 계약·옛 실험이 20 개 섞여 있어 "지금 무엇을 쓸 수 있나"에 답하지 못한다.
여기는 그 질문 하나에만 답한다. 정책 하나 = 디렉터리 하나:

    policies/<id>/
      policy.yaml            사람이 쓰는 카드 — id · status · task · side · note   (도구가 덮어쓰지 않는다)
      params/env.yaml, agent.yaml                                                 (계약의 모든 숫자의 원천)
      nn/<하나>.pth          정확히 1 개 — contract_build 의 "exactly one .pth" 규칙
      fetch.json             출처 매니페스트 (호스트 · 원격 경로 · sha256 · 학습 커밋)
      deploy_contract.json | pour_contract.json                                   (만들어졌으면)
      trace_meta.json, trace.npz                                                  (pour 계열)

상태(status)는 넷뿐이다. `verified`/`deployed` 는 계약이 있어야 한다 — 계약 없는 "검증됨"은 말이 안 된다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

CARD = "policy.yaml"
MANIFEST = "fetch.json"
INDEX = "INDEX.md"
PARAMS = ("params/env.yaml", "params/agent.yaml")
CONTRACTS = ("deploy_contract.json", "pour_contract.json")

STATUSES = {
    "candidate": "받아만 뒀다 — 계약·체인 검증 전",
    "verified": "계약이 서고 fake 체인을 통과했다 — 실기 승인 전",
    "deployed": "실기에서 승인받아 돌린 적이 있다",
    "hold": "쓰지 않는다 — 이유는 note 에",
}
NEEDS_CONTRACT = ("verified", "deployed")


@dataclass(frozen=True)
class Entry:
    id: str
    path: Path
    card: Mapping
    checkpoint: str          # nn/ 아래 파일명, 없거나 여러 개면 ""
    contract: str            # CONTRACTS 중 하나, 없으면 ""
    issues: tuple[str, ...]

    @property
    def status(self) -> str:
        return str(self.card.get("status", "?"))

    @property
    def ok(self) -> bool:
        return not self.issues


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contract_checkpoint_md5(doc: Mapping) -> str:
    """pour v1 은 최상위, deploy v2 는 `run.` 아래에 적는다. 자산 전용 계약은 빈 문자열이다."""
    if "checkpoint_md5" in doc:
        return str(doc.get("checkpoint_md5") or "")
    return str((doc.get("run") or {}).get("checkpoint_md5") or "")


def _load_card(path: Path, issues: list[str]) -> dict:
    p = path / CARD
    if not p.is_file():
        issues.append(f"{CARD} 가 없다 — 이 정책이 무엇이고 왜 여기 있는지 적어라")
        return {}
    try:
        card = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        issues.append(f"{CARD} 를 읽지 못한다: {exc}")
        return {}
    if not isinstance(card, dict):
        issues.append(f"{CARD} 는 매핑이어야 한다")
        return {}
    if card.get("id") != path.name:
        issues.append(f"카드 id {card.get('id')!r} ≠ 디렉터리 이름 {path.name!r}")
    if card.get("status") not in STATUSES:
        issues.append(f"status {card.get('status')!r} 는 {sorted(STATUSES)} 중 하나여야 한다")
    if card.get("status") == "hold" and not str(card.get("note") or "").strip():
        issues.append("status hold 는 note 에 이유가 있어야 한다")
    return card


def _check_manifest(path: Path, issues: list[str], deep: bool) -> None:
    p = path / MANIFEST
    if not p.is_file():
        issues.append(f"{MANIFEST} 가 없다 — 출처를 모른다 (fetch_run.py 로 받아라)")
        return
    try:
        files = json.loads(p.read_text()).get("files") or []
    except (OSError, ValueError) as exc:
        issues.append(f"{MANIFEST} 를 읽지 못한다: {exc}")
        return
    for f in files:
        local = path / f["local_rel"]
        if not local.is_file():
            issues.append(f"매니페스트의 {f['local_rel']} 가 없다")
        elif "size" in f and local.stat().st_size != f["size"]:
            issues.append(f"{f['local_rel']} 크기가 매니페스트와 다르다")
        elif deep and _hash(local, "sha256") != f["sha256"]:
            issues.append(f"{f['local_rel']} sha256 이 매니페스트와 다르다 — 받은 뒤에 바뀌었다")


def check(path: Path, *, deep: bool = True) -> Entry:
    issues: list[str] = []
    card = _load_card(path, issues)

    pths = sorted(p.name for p in (path / "nn").glob("*.pth")) if (path / "nn").is_dir() else []
    if len(pths) != 1:
        issues.append(f"nn/ 에 .pth 가 {len(pths)} 개다 — 정확히 1 개여야 한다" + (f": {', '.join(pths)}" if pths else ""))
    checkpoint = pths[0] if len(pths) == 1 else ""

    issues += [f"{rel} 가 없다" for rel in PARAMS if not (path / rel).is_file()]
    _check_manifest(path, issues, deep)

    found = [c for c in CONTRACTS if (path / c).is_file()]
    if len(found) > 1:
        issues.append("계약이 둘이다: " + ", ".join(found))
    contract = found[0] if len(found) == 1 else ""
    if contract and checkpoint:
        want = ""
        try:
            want = contract_checkpoint_md5(json.loads((path / contract).read_text()))
        except (OSError, ValueError) as exc:
            issues.append(f"{contract} 를 읽지 못한다: {exc}")
        if want and want != _hash(path / "nn" / checkpoint, "md5"):
            issues.append(f"{contract} 의 체크포인트 md5 가 nn/{checkpoint} 와 다르다 — 계약을 다시 만들어라")
    if not contract and card.get("status") in NEEDS_CONTRACT:
        issues.append(f"status {card.get('status')} 인데 계약이 없다")

    return Entry(path.name, path, card, checkpoint, contract, tuple(issues))


def scan(root: Path, *, deep: bool = True) -> tuple[Entry, ...]:
    if not root.is_dir():
        return ()
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    return tuple(check(d, deep=deep) for d in dirs)


def render_index(entries: Iterable[Entry]) -> str:
    out = ["# policies — 쓸 정책 목록", "",
           "`policy_control/tools/policies.py --write-index` 가 만든다. 손으로 고치지 않는다 — 고칠 것은 각 `policy.yaml` 이다.", "",
           "| id | status | task | side | checkpoint | 계약 | 점검 |", "|---|---|---|---|---|---|---|"]
    for e in entries:
        mark = "ok" if e.ok else "✗ " + " / ".join(e.issues)
        out.append(f"| `{e.id}` | {e.status} | {e.card.get('task', '')} | {e.card.get('side', '')} "
                   f"| {e.checkpoint or '-'} | {e.contract or '-'} | {mark} |")
    out += ["", "## status", ""] + [f"- `{k}` — {v}" for k, v in STATUSES.items()]
    notes = [(e.id, str(e.card.get("note") or "").strip()) for e in entries]
    if any(n for _, n in notes):
        out += ["", "## note", ""] + [f"- `{i}` — {n}" for i, n in notes if n]
    return "\n".join(out) + "\n"

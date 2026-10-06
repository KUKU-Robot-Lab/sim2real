"""`sim2real/deploy/policies/` — **쓸 정책만** 모아 두는 곳의 규약과 점검 (rclpy·torch 무의존).

`logs/policy/` 에는 배포본·자산 계약·옛 실험이 20 개 섞여 있어 "지금 무엇을 쓸 수 있나"에 답하지 못한다.
여기는 그 질문 하나에만 답한다. 정책 하나 = 디렉터리 하나, 경로는 로봇 손 / 정책 과제 / 팔_태그
(10.06 사용자: "로봇핸드, 정책 테스크, tag(i**) 이렇게 구분되어야 함"). id 는 그 상대 경로다:

    deploy/policies/<손>/<과제>/<팔>_<태그>/          예: rh56f1/aglt/right_env17 · dg5f_m/pour_fab/both_i18
      policy.yaml            사람이 쓰는 카드 — id · status · task · side · checkpoint · note  (도구가 덮어쓰지 않는다)
      params/env.yaml, agent.yaml                                                 (계약의 모든 숫자의 원천)
      nn/<하나>.pth          한 개면 그것이 후보다
      fetch.json             출처 매니페스트 (호스트 · 원격 경로 · sha256 · 학습 커밋)

**한 벌(bundle)** — 한 팔의 정책 묶음도 디렉터리 하나다(09.22 사용자 배치). 체크포인트가 여럿이면
카드의 `checkpoint:` 가 **후보 하나**를 가리켜야 한다. `build_deploy_contract.py --checkpoint` 가 그것을 받는다.
그때 `params/` 는 런별로 갈라져 있어도 된다 — `params/<체크포인트 이름>/{env,agent}.yaml` 을 먼저 보고
없으면 `params/{env,agent}.yaml` 을 본다.

손으로 옮긴 한 벌에는 `fetch.json` 이 없다. 그것은 `candidate` 까지만 봐준다 — `verified`·`deployed` 는
출처를 모르면 안 된다.
      deploy_contract.json | pour_contract.json | joint_contract.json                                (만들어졌으면)
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
CONTRACTS = ("deploy_contract.json", "pour_contract.json", "joint_contract.json", "pour_fj_contract.json",
             "rh_aglt_contract.json", "rh_place_contract.json")   # rh_aglt: 손 관측이 이름(프로필) 순이라 순서 실측 규칙이 필요 없다(09.30)

STATUSES = {
    "candidate": "받아만 뒀다 — 계약·체인 검증 전",
    "verified": "계약이 서고 fake 체인을 통과했다 — 실기 승인 전",
    "deployed": "실기에서 승인받아 돌린 적이 있다",
    "hold": "쓰지 않는다 — 이유는 note 에",
}
NEEDS_CONTRACT = ("verified", "deployed")
#: 경로 첫 칸(로봇 손) → 그 손 정책의 hdgp gym id 접두어. 다른 손의 정책이 섞여 들어오면 layout_issues 가 잡는다.
#: 새 손을 들이면 여기에 한 줄 더한다.
HANDS = {"rh56f1": "open-rh_", "dg5f_m": "open-short_"}
SIDES = ("right", "left", "both")


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


def _is_tail(path: Path, pid: str) -> bool:
    """pid(상대 경로)가 path 의 끝 칸들과 같은가 — 등록부 뿌리를 모를 때(콘솔 프로파일의 정책 폴더) 쓴다."""
    parts = Path(pid).parts
    return bool(parts) and path.parts[-len(parts):] == parts


def _load_card(path: Path, issues: list[str], want: str | None = None) -> dict:
    """want = 등록부 뿌리 기준 상대 경로(뿌리를 알 때). 모르면 카드 id 가 폴더 경로의 끝이어야 한다."""
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
    cid = str(card.get("id") or "")
    if want is not None and cid != want:
        issues.append(f"카드 id {cid!r} ≠ 등록부 경로 {want!r}")
    elif want is None and not _is_tail(path, cid):
        issues.append(f"카드 id {card.get('id')!r} ≠ 디렉터리 이름 {path.name!r}")
    if card.get("status") not in STATUSES:
        issues.append(f"status {card.get('status')!r} 는 {sorted(STATUSES)} 중 하나여야 한다")
    if card.get("status") == "hold" and not str(card.get("note") or "").strip():
        issues.append("status hold 는 note 에 이유가 있어야 한다")
    return card


def _pick_checkpoint(path: Path, card: Mapping, issues: list[str]) -> str:
    """쓸 가중치 하나. 여럿이면 카드가 골라야 한다 — 도구가 추측하지 않는다."""
    pths = sorted(p.name for p in (path / "nn").glob("*.pth")) if (path / "nn").is_dir() else []
    named = str(card.get("checkpoint") or "").strip()
    if named:
        if named in pths:
            return named
        issues.append(f"카드의 checkpoint {named!r} 가 nn/ 에 없다" + (f" (있는 것: {', '.join(pths)})" if pths else ""))
        return ""
    if len(pths) == 1:
        return pths[0]
    issues.append(f"nn/ 에 .pth 가 {len(pths)} 개다 — 카드의 `checkpoint:` 로 하나를 고르라"
                  + (f": {', '.join(pths)}" if pths else ""))
    return ""


def _params_of(path: Path, checkpoint: str) -> tuple[str, ...]:
    """런별로 갈라 둔 한 벌이면 그 체크포인트의 폴더를 본다. 아니면 공통 params/."""
    run = checkpoint[:-4] if checkpoint.endswith(".pth") else checkpoint
    if run and (path / "params" / run).is_dir():
        return tuple(f"params/{run}/{Path(rel).name}" for rel in PARAMS)
    return PARAMS


def _check_manifest(path: Path, issues: list[str], deep: bool, *, needed: bool = True) -> None:
    p = path / MANIFEST
    if not p.is_file():
        if needed:
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


def check(path: Path, *, deep: bool = True, root: Path | None = None) -> Entry:
    """정책 폴더 하나. root(등록부 뿌리)를 주면 id = root 기준 상대 경로이고 카드 id 가 그것과 같아야 한다."""
    issues: list[str] = []
    want = path.relative_to(root).as_posix() if root is not None else None
    card = _load_card(path, issues, want)
    cid = str(card.get("id") or "")
    pid = want if want is not None else (cid if cid and _is_tail(path, cid) else path.name)

    checkpoint = _pick_checkpoint(path, card, issues)
    issues += [f"{rel} 가 없다" for rel in _params_of(path, checkpoint) if not (path / rel).is_file()]
    _check_manifest(path, issues, deep, needed=card.get("status") in NEEDS_CONTRACT)

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
    if contract in ("joint_contract.json", "pour_fj_contract.json") and card.get("status") in NEEDS_CONTRACT:
        try:
            src = str(json.loads((path / contract).read_text()).get("hand_obs_order_source", ""))
        except (OSError, ValueError):
            src = ""
        if not src.startswith("measured"):
            # 손 관측 순서를 가정한 계약은 관측이 학습과 같다는 근거가 없다 — 실측 trace 로 다시 만들기 전에는 못 올린다
            issues.append(f"status {card.get('status')} 인데 {contract} 의 손 관측 순서가 실측이 아니다({src or '?'})")

    return Entry(pid, path, card, checkpoint, contract, tuple(issues))


def _is_policy_dir(d: Path) -> bool:
    """정책 폴더 = 카드 · nn/ · params/ 중 하나가 있는 폴더. 셋 다 없으면 손 · 과제로 묶는 폴더다."""
    return any((d / name).exists() for name in (CARD, "nn", "params"))


def _policy_dirs(d: Path) -> list[Path]:
    out: list[Path] = []
    for c in sorted(x for x in d.iterdir() if x.is_dir() and not x.name.startswith(".") and x.name != "__pycache__"):
        out += [c] if _is_policy_dir(c) else _policy_dirs(c)
    return out


def scan(root: Path, *, deep: bool = True) -> tuple[Entry, ...]:
    """등록부 전체 — 손 · 과제 폴더를 따라 내려가 정책 폴더마다 check. 순서는 경로 순(손 → 과제 → 팔_태그)."""
    if not root.is_dir():
        return ()
    return tuple(check(d, deep=deep, root=root) for d in _policy_dirs(root))


def layout_issues(e: Entry) -> tuple[str, ...]:
    """id(등록부 상대 경로)가 <손>/<과제>/<팔>_<태그> 규칙을 따르는가(10.06 사용자). 빈 튜플 = 따른다.

    정책을 못 쓰게 막지는 않는다(콘솔 선택 사유에 안 들어간다) — policies.py · INDEX · 등록부 테스트가 잡는다."""
    parts = e.id.split("/")
    if len(parts) != 3:
        return (f"경로 {e.id!r} 는 <손>/<과제>/<팔>_<태그> 여야 한다 (예: rh56f1/aglt/right_env17)",)
    hand, _task, leaf = parts
    side, task = str(e.card.get("side") or ""), str(e.card.get("task") or "")
    out = []
    if hand not in HANDS:
        out.append(f"모르는 손 {hand!r} — {', '.join(sorted(HANDS))} 중 하나(새 손은 policy_registry.HANDS 에 더한다)")
    elif task and not task.startswith(HANDS[hand]):
        out.append(f"과제 {task} 는 {hand} 정책이 아니다(gym id 가 {HANDS[hand]} 로 시작해야 한다)")
    if side not in SIDES or not leaf.startswith(f"{side}_") or leaf == f"{side}_":
        out.append(f"마지막 칸 {leaf!r} 은 <팔>_<태그> 여야 한다(카드 side {side or '?'})")
    return tuple(out)


def render_index(entries: Iterable[Entry]) -> str:
    out = ["# policies — 쓸 정책 목록", "",
           "`deploy/policy_control/tools/policies.py --write-index` 가 만든다. 손으로 고치지 않는다 — 고칠 것은 각 `policy.yaml` 이다.", "",
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

"""로봇 모듈 — 콘솔 첫 화면의 "어느 로봇인가". 로봇 하나 = `robots/<id>.yaml` 파일 하나.

09.29 사용자: "처음 창을 키면 robot 및 실행 가능한 policy 선택창 · robot setting 및 state ·
기존 dg5f-m-s 모델과 t2r rh56f1 [grasping] · [pouring] 세션에서 쓰는 robot (앞으로도 추가될 예정이므로 모듈화)".

로봇 모듈이 아는 것:
  · 어느 자산인가(`asset`) — 계약의 `asset` 이 같은 정책만 이 로봇에서 고를 수 있다
  · 어느 프로파일(미션 + 도메인)로 여는가(`profiles`) — 실기 · fake
  · 정책을 고르면 미션의 어느 산출물을 바꾸는가(`slots`: 쪽 → 산출물 키)
  · 화면에 보일 설정(`settings`)과 그림(`art`)
로봇 이름으로 분기하는 코드는 없다 — 새 로봇은 yaml 하나를 더한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from .errors import ProfileError

SCHEMA = "s2r_console/robot/v1"
SIDES = ("right", "left", "both")
#: 슬롯이 받는 계약 파일 — 산출물 키의 접두어로 정한다(joint_* 는 joint family 계약만).
SLOT_CONTRACTS = {"joint": ("joint_contract.json",), "pour": ("pour_contract.json",)}
_KEYS = {"schema", "id", "title", "host", "asset", "hand", "profiles", "slots", "task_prefixes", "art", "settings", "note",
         "joint_profile"}


@dataclass(frozen=True)
class Robot:
    id: str
    title: str
    host: str
    asset: str
    hand: Mapping[str, str]
    profiles: tuple[str, ...]
    slots: Mapping[str, str]                 # 쪽 → 미션 산출물 키
    task_prefixes: tuple[str, ...]           # 계약이 아직 없는 정책을 목록에만 보일 때 쓰는 과제 이름 접두어
    art: Mapping[str, str]
    settings: tuple[tuple[str, str], ...]
    note: str
    path: Path
    #: robot_control 관절 프로파일(repo 상대) — 로봇 상태 표의 한계 · 원본 이름. 비면 콘솔 기본(openarm_tesollo)
    joint_profile: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "host": self.host, "asset": self.asset, "hand": dict(self.hand),
                "profiles": list(self.profiles), "slots": dict(self.slots), "art": dict(self.art),
                "settings": [{"label": k, "value": v} for k, v in self.settings], "note": self.note,
                "joint_profile": self.joint_profile}


def slot_contracts(artifact_key: str) -> tuple[str, ...]:
    """산출물 키가 받는 계약 파일 이름들 — `joint_left` → joint_contract.json."""
    return SLOT_CONTRACTS.get(artifact_key.split("_", 1)[0], ())


def parse(raw: object, *, path: Path) -> Robot:
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{path}: 매핑이어야 한다")
    if raw.get("schema") != SCHEMA:
        raise ProfileError(f"{path}: schema 는 {SCHEMA} 여야 한다")
    unknown = set(raw) - _KEYS
    if unknown:
        raise ProfileError(f"{path}: 모르는 키 {sorted(unknown)}")
    for key in ("id", "title", "asset"):
        if not raw.get(key):
            raise ProfileError(f"{path}: '{key}' 가 없다")
    if str(raw["id"]) != path.stem:
        raise ProfileError(f"{path}: id {raw['id']!r} ≠ 파일 이름 {path.stem!r}")
    slots = raw.get("slots") or {}
    if not isinstance(slots, Mapping) or any(s not in SIDES for s in slots):
        raise ProfileError(f"{path}: slots 는 {{{'|'.join(SIDES)}: 산출물 키}} 여야 한다")
    bad = [k for k in slots.values() if not slot_contracts(str(k))]
    if bad:
        raise ProfileError(f"{path}: 슬롯 산출물 키 {bad} — 접두어가 {sorted(SLOT_CONTRACTS)} 중 하나여야 한다")
    settings = []
    for item in raw.get("settings") or ():
        if not isinstance(item, Mapping) or set(item) != {"label", "value"}:
            raise ProfileError(f"{path}: settings 항목은 {{label, value}} 여야 한다: {item!r}")
        settings.append((str(item["label"]), str(item["value"])))
    return Robot(id=str(raw["id"]), title=str(raw["title"]), host=str(raw.get("host", "")), asset=str(raw["asset"]),
                 hand={str(k): str(v) for k, v in (raw.get("hand") or {}).items()},
                 profiles=tuple(str(p) for p in raw.get("profiles") or ()),
                 slots={str(k): str(v) for k, v in slots.items()},
                 task_prefixes=tuple(str(p) for p in raw.get("task_prefixes") or ()),
                 art={str(k): str(v) for k, v in (raw.get("art") or {}).items()},
                 settings=tuple(settings), note=str(raw.get("note", "")), path=path,
                 joint_profile=str(raw.get("joint_profile", "")))


def scan(directory: Path) -> tuple[list[Robot], dict[str, str]]:
    """로봇 모듈 전부. 깨진 것은 (파일명 → 사유) 로 따로 — 조용히 숨기지 않는다."""
    good, bad = [], {}
    for p in sorted(directory.glob("*.yaml")) if directory.is_dir() else ():
        try:
            good.append(parse(yaml.safe_load(p.read_text(encoding="utf-8")), path=p))
        except (OSError, yaml.YAMLError) as exc:
            bad[p.name] = f"{p}: 읽지 못했다: {exc}"
        except ProfileError as exc:
            bad[p.name] = str(exc)
    return good, bad


# ── 정책 고르기 ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PolicyInfo:
    """정책 한 벌을 이 화면이 보는 모양 — `policy_registry.Entry` 에서 필요한 것만."""

    id: str
    status: str
    side: str
    task: str
    contract: str            # 파일 이름, 없으면 ""
    asset: str               # 계약의 asset, 계약이 없으면 ""
    issues: tuple[str, ...]
    note: str = ""


def _belongs(robot: Robot, p: PolicyInfo) -> bool:
    if p.asset:
        return p.asset == robot.asset
    return any(p.task.startswith(pre) for pre in robot.task_prefixes)


def choice_reasons(robot: Robot, p: PolicyInfo, side: str) -> list[str]:
    """이 정책을 이 로봇의 `side` 슬롯에 고를 수 없는 이유. 비어 있으면 고를 수 있다."""
    key = robot.slots.get(side)
    if key is None:
        return [f"{robot.title} 에는 {side} 정책 자리가 없다"]
    out = []
    if p.side != side:
        out.append(f"정책의 쪽이 {p.side} 다({side} 자리)")
    if not p.contract:
        out.append("계약이 없다 — build_deploy_contract.py 로 만든 뒤 고른다")
    elif p.contract not in slot_contracts(key):
        out.append(f"{p.contract} 는 {key} 자리가 받는 계약({', '.join(slot_contracts(key))})이 아니다")
    if p.asset and p.asset != robot.asset:
        out.append(f"계약 자산 {p.asset} ≠ 로봇 자산 {robot.asset}")
    if p.status == "hold":
        out.append("status hold — 쓰지 않는다(카드 note)")
    out += [f"카드 점검: {i}" for i in p.issues]
    return out


def choices(robot: Robot, policies: Iterable[PolicyInfo]) -> list[dict]:
    """이 로봇의 정책 목록 — 자리마다 고를 수 있는지와 그 이유. 못 고르는 정책도 이유와 함께 보인다."""
    out = []
    for p in policies:
        if not _belongs(robot, p):
            continue
        slots = {side: choice_reasons(robot, p, side) for side in robot.slots if side == p.side}
        out.append({"id": p.id, "status": p.status, "side": p.side, "task": p.task, "contract": p.contract,
                    "note": p.note, "why": slots.get(p.side, [f"{robot.title} 에는 {p.side} 정책 자리가 없다"])})
    return out


def overrides(robot: Robot, picked: Mapping[str, str], policies: Mapping[str, PolicyInfo],
              policies_root: Path, repo: Path) -> dict[str, str]:
    """{쪽: 정책 id} → {미션 산출물 키: repo 상대 계약 경로}. 하나라도 못 고르면 ProfileError(사유 전부)."""
    out, why = {}, []
    for side, pid in picked.items():
        p = policies.get(pid)
        if p is None:
            why.append(f"{side}: 그런 정책이 없다: {pid}")
            continue
        reasons = choice_reasons(robot, p, side)
        if reasons:
            why += [f"{side} {pid}: {r}" for r in reasons]
            continue
        out[robot.slots[side]] = str((policies_root / pid / p.contract).resolve().relative_to(repo.resolve()))
    if why:
        raise ProfileError("; ".join(why))
    return out


def policy_info(entry, contract_doc: Mapping | None) -> PolicyInfo:
    """`policy_registry.Entry` + 읽은 계약 → PolicyInfo."""
    card = entry.card or {}
    return PolicyInfo(id=entry.id, status=entry.status, side=str(card.get("side", "")), task=str(card.get("task", "")),
                      contract=entry.contract, asset=str((contract_doc or {}).get("asset", "")),
                      issues=tuple(entry.issues), note=str(card.get("note", "") or "")[:400])

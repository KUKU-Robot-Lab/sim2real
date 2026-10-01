"""실기 관절 이름(source) -> sim 자산 관절 이름(canonical) 사상. 순수 파이썬(ROS·Isaac 무관).

진실원천은 robot_control 프로필 `openarm_tesollo.yaml` 의 `joints:` 표다
(`{canonical, source, sign}`). 여기서 값을 새로 짓지 않고 그 표를 읽어 쓴다.

- 팔: `/joint_states` 의 `openarm_right_joint1..7` -> `r_aj_1..7` (좌팔 `l_aj_*`)
- 손: `/dg5f_right/joint_states` 의 `rj_dg_<f>_<j>` -> `r_hj_<finger>_<j>` (좌손 `lj_dg_*` -> `l_hj_*`)
- 이미 canonical 이름으로 오는 메시지(예: `/head/joint_states` 의 `head_j_pan`)는 그대로 통과시킨다.
- 10.01 RH56F1: 손 종속(mimic) 관절은 구동이 아니라 프로필 `joints:` 에 없다(profile.py 가 구동 그룹 밖 관절을 거부).
  상태 노드는 그것들을 canonical 이름으로 낸다 — 프로필 `asset.urdf` 의 `<mimic>` 관절을 canonical 그대로 통과로 더한다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

#: sim2real/robot/isaacsim_bridge/viewer -> rl_ws
_RL_WS = Path(__file__).resolve().parents[4]
DEFAULT_PROFILE = _RL_WS / "robot_control" / "src" / "robot_control" / "profiles" / "openarm_tesollo.yaml"

#: 프로필 표에 없어도 canonical 로 인정하는 이름(머리 관절은 드라이버가 canonical 로 낸다).
EXTRA_CANONICAL = frozenset({"head_j_pan", "head_j_tilt"})


@dataclass(frozen=True)
class JointRule:
    canonical: str
    sign: float


@dataclass(frozen=True)
class MapResult:
    """한 메시지의 사상 결과. `values` 는 canonical -> rad, `unknown` 은 버린 source 이름."""

    values: Mapping[str, float]
    unknown: tuple[str, ...]
    non_finite: tuple[str, ...]


def build_table(joints: Iterable[Mapping]) -> Mapping[str, JointRule]:
    """프로필 `joints:` 항목 목록 -> source 이름 기준 읽기 전용 표."""
    table: dict[str, JointRule] = {}
    for entry in joints:
        try:
            src, can, sign = str(entry["source"]), str(entry["canonical"]), float(entry.get("sign", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"프로필 joints 항목 형식 오류: {entry!r}") from exc
        if sign not in (1.0, -1.0):
            raise ValueError(f"sign 은 +1/-1 만 허용: {src} sign={sign}")
        if src in table and table[src].canonical != can:
            raise ValueError(f"source 이름 중복(서로 다른 canonical): {src}")
        table[src] = JointRule(canonical=can, sign=sign)
    if not table:
        raise ValueError("프로필 joints 표가 비었다")
    return MappingProxyType(table)


def load_profile_table(path: str | Path = DEFAULT_PROFILE) -> Mapping[str, JointRule]:
    """robot_control 프로필 yaml 을 읽어 source -> JointRule 표를 만든다."""
    import yaml  # 지연 import: 테스트에서 build_table 만 쓸 때 yaml 불필요

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"관절 프로필이 없다: {p}")
    with p.open() as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("joints"), list):
        raise ValueError(f"프로필에 joints 목록이 없다: {p}")
    entries = list(doc["joints"])
    known = {str(e.get("canonical")) for e in entries if isinstance(e, Mapping)}
    for name in asset_mimic_joints(p, doc):
        if name not in known:
            entries.append({"source": name, "canonical": name, "sign": 1})
    return build_table(entries)


def _asset_urdf(profile_path: Path, doc: Mapping) -> Path | None:
    """프로필 asset.urdf — 프로필 위치 기준 상대 경로. 없으면 rl_ws/hdgp 아래에서 같은 상대 경로를 찾는다
    (robot_control profile._resolve_manifest 와 같은 규칙, HDGP_ROOT 우선)."""
    asset = doc.get("asset")
    value = asset.get("urdf") if isinstance(asset, Mapping) else None
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path if path.is_file() else None
    direct = (profile_path.parent / path).resolve()
    if direct.is_file():
        return direct
    if "hdgp" in path.parts:
        rel = Path(*path.parts[path.parts.index("hdgp") + 1:])
        for root in (os.environ.get("HDGP_ROOT"), str(_RL_WS / "hdgp")):
            if root and (Path(root) / rel).is_file():
                return Path(root) / rel
    return None


def asset_mimic_joints(profile_path: Path, doc: Mapping) -> tuple[str, ...]:
    """자산 URDF 에서 `<mimic>` 을 가진 관절 이름(자산 = canonical 이름). URDF 가 없으면 빈 튜플."""
    urdf = _asset_urdf(Path(profile_path), doc)
    if urdf is None:
        return ()
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf).getroot()
    return tuple(j.get("name") for j in root.iter("joint") if j.find("mimic") is not None and j.get("name"))


def canonical_names(table: Mapping[str, JointRule]) -> frozenset[str]:
    return frozenset(r.canonical for r in table.values()) | EXTRA_CANONICAL


def map_joint_state(names: Sequence[str], positions: Sequence[float],
                    table: Mapping[str, JointRule]) -> MapResult:
    """sensor_msgs/JointState 의 (name, position) -> canonical 값.

    source 이름이면 부호를 곱해 canonical 로, 이미 canonical 이면 그대로, 둘 다 아니면 버린다.
    NaN/inf 는 버리고 `non_finite` 에 적는다(뷰어에 쓰레기 값을 쓰지 않는다).
    """
    if len(names) != len(positions):
        raise ValueError(f"name {len(names)} 개 / position {len(positions)} 개 길이 불일치")
    canon = canonical_names(table)
    values: dict[str, float] = {}
    unknown: list[str] = []
    bad: list[str] = []
    for name, pos in zip(names, positions):
        rule = table.get(name)
        if rule is not None:
            target, sign = rule.canonical, rule.sign
        elif name in canon:
            target, sign = name, 1.0
        else:
            unknown.append(name)
            continue
        val = float(pos)
        if not math.isfinite(val):
            bad.append(name)
            continue
        values[target] = sign * val
    return MapResult(values=MappingProxyType(values), unknown=tuple(unknown), non_finite=tuple(bad))


def merge(state: Mapping[str, float], update: Mapping[str, float]) -> Mapping[str, float]:
    """기존 상태에 새 값을 얹은 **새** 읽기 전용 사본(원본 불변)."""
    return MappingProxyType({**state, **update})

"""실기 관절 이름(source) -> sim 자산 관절 이름(canonical) 사상. 순수 파이썬(ROS·Isaac 무관).

진실원천은 robot_control 프로필 `openarm_tesollo.yaml` 의 `joints:` 표다
(`{canonical, source, sign}`). 여기서 값을 새로 짓지 않고 그 표를 읽어 쓴다.

- 팔: `/joint_states` 의 `openarm_right_joint1..7` -> `r_aj_1..7` (좌팔 `l_aj_*`)
- 손: `/dg5f_right/joint_states` 의 `rj_dg_<f>_<j>` -> `r_hj_<finger>_<j>` (좌손 `lj_dg_*` -> `l_hj_*`)
- 이미 canonical 이름으로 오는 메시지(예: `/head/joint_states` 의 `head_j_pan`)는 그대로 통과시킨다.
"""

from __future__ import annotations

import math
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
    return build_table(doc["joints"])


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

#!/usr/bin/env python3
"""robot_control 프로필 yaml 을 **자산 manifest 없이** 읽는다 (ROS 무의존).

★왜 따로 있나. `robot_control.profile.load_builtin_profile` 은 프로필이 가리키는 자산
manifest 를 열고 sha256 까지 대조한다. 2026-09-05 자산 교체(`openarm_tesollo_sensor_rl`
삭제)로 그 파일이 없어져 **로더가 죽는다**. 프로필의 관절 이름·부호·한계·tip·컨트롤러는
자산과 무관하게 yaml 안에 그대로 있으므로, 그것만 필요한 도구는 여기로 읽는다.

2026-09-07 우팔 실기에서 `gravity_comp_node` 가 이 함정으로 못 떠서, 팔을 세워둔 채
중력보상을 켜지 못했다. `tests/policy_control/test_pc_gravity_profile_loader.py` 가 재발을 막는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PROFILE = Path("/home/user/rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml")


@dataclass(frozen=True)
class ProfileGroup:
    """한 팔(그룹)에 대해 자산과 무관한 프로필 사실만."""

    name: str
    canonical: tuple[str, ...]
    sources: tuple[str, ...]
    signs: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    tip: str                    # 중력 체인 끝 = asset_tip_link 우선(없으면 tip_link)
    tip_link: str
    controller: str | None
    effort_controller: str | None


def load_profile_group(path: str | Path = DEFAULT_PROFILE, group: str = "openarm_right_arm") -> ProfileGroup:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    groups = raw.get("groups") or {}
    if group not in groups:
        raise SystemExit(f"프로필 {path.name} 에 group {group!r} 이 없다 (있는 것: {sorted(groups)})")
    g = groups[group]
    by = {j["canonical"]: j for j in raw.get("joints") or []}
    canonical = list(g["joints"])
    missing = [c for c in canonical if c not in by]
    if missing:
        raise SystemExit(f"프로필 {path.name} 의 joints 에 {missing} 가 없다")
    return ProfileGroup(
        name=group,
        canonical=tuple(canonical),
        sources=tuple(by[c]["source"] for c in canonical),
        signs=tuple(int(by[c].get("sign", 1)) for c in canonical),
        lower=tuple(float(by[c]["lower"]) for c in canonical),
        upper=tuple(float(by[c]["upper"]) for c in canonical),
        tip=str(g.get("asset_tip_link") or g["tip_link"]),
        tip_link=str(g["tip_link"]),
        controller=g.get("controller"),
        effort_controller=g.get("effort_controller"),
    )

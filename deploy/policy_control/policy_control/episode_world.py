"""에피소드 진행 상태(WorldState) — 자세 · 두 손에 든 물체 · 진행 플래그. 정책 이름이 아니라 실제 상태를 든다. 순수 · 불변.

가이드 1-5: EpisodeManager 는 object / hand / pose 상태를 함께 관리한다. 값은 노드 `expect` 를 성공 뒤에 적용해 바꾼다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from policy_control.episode_spec import EMPTY, HAND_KEYS

UNKNOWN = "UNKNOWN"
RUNNING = "POLICY_RUNNING"


@dataclass(frozen=True)
class World:
    pose: str = UNKNOWN
    hands: Mapping[str, str] = field(default_factory=lambda: {"right": EMPTY, "left": EMPTY})
    flags: frozenset = frozenset()             # 참인 진행 플래그(blue_poured …)
    objects: Mapping[str, Mapping] = field(default_factory=dict)   # 물체 → {"at": table|right_hand|left_hand|holder:<이름>, "pos"}

    def hand(self, side: str) -> str:
        return self.hands.get(side, EMPTY)

    def running(self) -> "World":
        return replace(self, pose=RUNNING)

    def apply(self, expect: Mapping) -> "World":
        """노드 성공 뒤 기대 상태를 적용한 새 World(expect 에 없는 칸은 그대로)."""
        hands = dict(self.hands)
        for key, side in HAND_KEYS.items():
            if key in expect:
                hands[side] = str(expect[key])
        flags = set(self.flags)
        for k, v in expect.items():
            if k in ("pose", *HAND_KEYS):
                continue
            (flags.add if v else flags.discard)(k)
        return World(pose=str(expect.get("pose", self.pose)), hands=hands, flags=frozenset(flags), objects=self.objects)

    def locate(self, name: str, at: str, pos=None) -> "World":
        """물체 하나의 자리를 바꾼 새 World(snapshot · 집기 · 놓기 뒤)."""
        objects = dict(self.objects)
        objects[name] = {"at": at, "pos": None if pos is None else [round(float(v), 4) for v in pos]}
        return replace(self, objects=objects)

    def as_dict(self) -> dict:
        return {"pose": self.pose, "left_hand": self.hand("left"), "right_hand": self.hand("right"),
                "task_flags": sorted(self.flags), "objects": {k: dict(v) for k, v in self.objects.items()}}

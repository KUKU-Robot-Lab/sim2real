"""FP++ 가 실패한 뒤 이전 pose 를 **새 시각으로** 다시 내는 것을 거른다 — 순수부(ROS 무관).

FP++ 스냅샷 노드는 마지막 대표 pose 를 5 Hz 로 계속 낸다(stamp = 지금, 같은 값). 그래서 목을 돌린 뒤 검출에 실패해도 이전 자세에서 본 값이
신선한 값처럼 나간다(배포 정책의 신선도 검사는 받은 시각만 본다 — 2026-10-14 s2r_1st). 고정된 물체의 카메라 프레임 값은 목이 돌면 바뀌므로,
**값이 비트 단위로 같은데 목이 min_move_deg 넘게 돌았으면** 그 값은 옛 자세에서 본 것이다. 정지 상태의 같은 값(정상 유지)은 통과시킨다.
기준 각도는 그 값을 **처음 받은 순간**의 목 각도다. 노드를 켜기 전에 이미 낡은 값이 나오고 있었다면 첫 값은 알 수 없고, 목이 움직이면 그때 걸린다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Seen:
    pos: tuple[float, float, float]
    head: tuple[float, float] | None


class StaleGuard:
    def __init__(self, min_move_deg: float = 0.5) -> None:
        self.min_move_deg = float(min_move_deg)
        self._seen: dict[str, _Seen] = {}

    def is_stale(self, key: str, raw_pos, head: tuple[float, float] | None) -> bool:
        pos = tuple(float(v) for v in raw_pos)
        seen = self._seen.get(key)
        if seen is None or seen.pos != pos:
            self._seen[key] = _Seen(pos, head)
            return False
        if head is None or seen.head is None:
            return False
        return max(abs(head[0] - seen.head[0]), abs(head[1] - seen.head[1])) > self.min_move_deg

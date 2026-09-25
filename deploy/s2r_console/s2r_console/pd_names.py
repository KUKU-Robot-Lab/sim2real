"""pd status 이름 — 09.23 부터 팔마다 갈린다(`pd_right` · `pd_left`).

사용자 결정 09.23: "pd 를 구분하는 게 맞을 것 같음. 앞으로도 양팔 또는 한팔 정책들이 많을 거고
개별 제어를 하는 게 맞을 것 같음."

콘솔은 status 이름을 **토큰 하나**로 다룬다(프로파일 `status_nodes`, 브리지 `--nodes`, 그림의 `status`).
그래서 쪽은 토큰 안에 넣었다 — `pd_right` — 그 층을 건드리지 않고 팔을 나눌 수 있다.
옛 이름 `pd` 도 읽기로는 받는다: 옛 프로파일·옛 런의 기록이 그 이름으로 남아 있다.
"""
from __future__ import annotations

from typing import Iterable, Mapping

#: 쪽 없는 옛 이름 — 읽기만 한다(새로 쓰지 않는다).
LEGACY = "pd"
SIDES = ("right", "left")


def name(side: str) -> str:
    """그 팔의 status 이름."""
    return f"{LEGACY}_{side}"


def is_pd(node: str) -> bool:
    return node == LEGACY or node.startswith(f"{LEGACY}_")


def side_of(node: str) -> str:
    """`pd_right` → `right`. 쪽 없는 옛 이름이면 빈 문자열."""
    return node.split("_", 1)[1] if node.startswith(f"{LEGACY}_") else ""


def nodes(status_nodes: Iterable[str]) -> tuple[str, ...]:
    """프로파일이 선언한 것 중 pd 인 것들 — 선언 순서를 지킨다."""
    return tuple(n for n in status_nodes if is_pd(n))


def sides(status_nodes: Iterable[str]) -> tuple[str, ...]:
    """선언된 pd 가 맡은 쪽들. 옛 이름 하나뿐이면 빈 튜플(쪽을 모른다)."""
    return tuple(s for s in (side_of(n) for n in nodes(status_nodes)) if s)


def worst(phases: Iterable[str | None], order: tuple[str, ...]) -> str | None:
    """팔들의 phase 를 **한 줄**로 합친다 — 배너·종료 규칙처럼 답이 하나여야 하는 곳에서만 쓴다.

    `order` 의 앞쪽이 더 무겁다(한 팔이라도 잡고 있으면 잡고 있는 것으로 본다).
    팔마다 따로 보는 화면(창)은 이것을 쓰지 않고 제 쪽 status 를 직접 읽는다.
    """
    got = [p for p in phases if p is not None]
    if not got:
        return None
    for p in order:
        if p in got:
            return p
    return got[0]


def pick(status: Mapping[str, object], status_nodes: Iterable[str]) -> dict[str, object]:
    """{쪽 또는 이름: status 본문} — 온 것만."""
    out: dict[str, object] = {}
    for n in nodes(status_nodes):
        body = status.get(n)
        if body is not None:
            out[side_of(n) or n] = body
    return out

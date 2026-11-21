"""경계 러너의 접점 — sim 과 실기가 같은 프로토콜을 구현한다(구현체는 여기 없다)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple, Protocol, Sequence

ADVANCE, LOOK_AGAIN, HANDOFF, STOP = "advance", "look_again", "handoff", "stop"
ACTIONS = (ADVANCE, LOOK_AGAIN, HANDOFF, STOP)
RECORD, PROPOSE, AUTO = "record", "propose", "auto"
MODES = (RECORD, PROPOSE, AUTO)
HEAD, ANY = "head", "any"   # 카메라 이름 — head 는 늘 있다. 다른 카메라(global 등)는 있을 때만


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str = ""
    seq: int = -1          # 경계 번호 — 지금 경계와 다르면 러너가 버린다


class Question(NamedTuple):
    text: str
    ok: str = "yes"        # 노드가 성공했을 때의 답(선택지 이름)
    view: str = HEAD       # 볼 카메라: head · global(카메라 이름) · any(앞뒤 사진이 다 있는 카메라 전부)


class FrameSource(Protocol):
    def frame(self, after: float) -> tuple[Mapping[str, bytes], Mapping[str, float]] | None:
        """after 이후에 찍힌 사진 ({카메라 이름: 비어 있지 않은 jpg bytes}, {카메라 이름: 촬영 시각 stamp}). "head" 는 꼭 있고,
        다른 카메라(global 등)는 그 카메라의 마지막 사진을 넣는다 — 러너가 카메라마다 stamp 를 after 와 비교해 오래된 것은 뺀다.

        계약: stamp 와 after 는 러너 시계(BoundaryRunner 의 clock)와 같은 기준이다 — 다른 시계(ROS 시각 등)면 구현이 바꿔서 준다.
        즉시 반환한다(러너는 이 호출에 예산을 걸지 못한다). 그런 사진이 아직 없으면 None — 러너가 마감까지 다시 묻는다.
        head 의 stamp < after · 유한하지 않은 stamp · head 없음 · 빈 jpg 면 러너가 "사진 없음" 으로 버린다(보조 카메라는 그 카메라만 뺀다).
        """
        ...


class Judge(Protocol):
    """계약: ask · decide 는 판정 스레드에서 불린다. 호출마다 자체 timeout 을 가진다 — 러너는 마감 뒤 답을 버릴 뿐
    스레드를 끊지 못하고, 앞 경계의 호출이 살아 있으면 다음 경계는 판정 없이 handoff 한다."""

    def ask(self, images: Mapping[str, Sequence[bytes]], questions: Sequence[str]) -> Sequence[Mapping]:
        """1단 — {카메라 이름: [전, 후]} 사진과 질문들(러너가 질문의 view 로 카메라를 골라 묶는다). 어느 사진이 어느 카메라인지는
        판정기가 프롬프트에 적는다. 질문마다 {"scores": {선택지: p}} 또는 {"low_mass": True}."""
        ...

    def decide(self, answers: Sequence[Mapping], signal: Mapping, allowed: Sequence[str]) -> tuple[str, str]:
        """2단 — 1단 답과 러너 신호로 (allowed 중 하나, 이유)."""
        ...


class ArmStill(Protocol):
    def arm_still(self) -> bool:
        """팔이 멈췄는가 — 경계 사진을 찍기 전에 러너가 묻는다. 계약: 즉시 반환한다(러너는 이 호출에 예산을 걸지 못한다)."""
        ...


# sink(row) 계약(BoundaryRunner 의 sink 인자): 러너 스레드와 판정 스레드(record 모드)에서 함께 불릴 수 있다 — 구현이 잠근다.
# propose · auto 에서 sink 가 예외를 내면 러너는 진행하지 않고 handoff 한다.

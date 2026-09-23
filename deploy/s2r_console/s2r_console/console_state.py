"""콘솔이 화면에 거는 한 줄짜리 상태 — **저장하지 않고 매번 파생한다**.

주인이 이미 있는 상태를 콘솔이 또 들고 있으면 둘이 어긋난다. pd 의 phase 는 pd_node 가,
미션 단계는 `mission_core` 가, 에피소드는 episode 토픽이 주인이다. 여기서는 그것들을 **읽어서**
운영자가 제일 먼저 알아야 할 것 하나를 고른다(위에서부터 먼저 걸리는 것이 이긴다).

순수하다 — rclpy·파일·시계를 보지 않는다. 나이(age)는 호출자가 재서 넘긴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

#: 이보다 오래된 status 는 "안 보인다" 로 친다 [s].
STALE_S = 2.0

OFFLINE = "OFFLINE"
ESTOP = "ESTOP"
FAULT = "FAULT"
HOLD = "HOLD"
STOPPING = "STOPPING"
RUNNING = "RUNNING"
ARMED = "ARMED"
IDLE = "IDLE"

#: 색. 화면은 이 표만 본다 — JS 에 상태 지식을 두지 않는다.
TONE = {OFFLINE: "mute", ESTOP: "bad", FAULT: "bad", HOLD: "warn",
        STOPPING: "warn", RUNNING: "live", ARMED: "live", IDLE: "ok"}


@dataclass(frozen=True)
class Observed:
    """브리지가 본 것. 없으면 None / 빈 dict."""

    bridge_up: bool = False
    #: 노드 이름 → 마지막 status payload
    status: Mapping[str, Mapping] = field(default_factory=dict)
    #: 노드 이름 → 마지막 status 를 받은 지 몇 초
    age_s: Mapping[str, float] = field(default_factory=dict)
    #: 마지막 episode 이벤트 payload
    episode: Mapping | None = None
    #: 브리지가 보고한 도메인 불일치 등 — 있으면 그대로 FAULT
    bridge_faults: Sequence[str] = ()
    #: 브리지 프로세스가 마지막으로 죽은 이유 — OFFLINE 일 때만 의미가 있다
    bridge_down_why: str | None = None
    #: 브리지가 본 ROS 그래프의 노드 전체 이름 (`/ns/name`)
    graph: Sequence[str] = ()
    #: 관절 이름 → [위치, 속도, 토크] — 브리지가 /joint_states 를 구독해 최신값만 보낸 것(09.23 화면 표)
    joints: Mapping[str, Sequence] = field(default_factory=dict)
    #: 관절 상태를 받은 지 몇 초 — 한 번도 못 받았으면 None
    joints_age_s: float | None = None
    #: 스택 토픽 이름 → {"pubs": 내는 쪽 수, "n": 지난 구간에 받은 수, "age_ms": 마지막 도착 뒤 ms | None}
    topics: Mapping[str, Mapping] = field(default_factory=dict)
    #: topics 보고를 받은 지 몇 초 — 한 번도 못 받았으면 None
    topics_age_s: float | None = None
    #: 도메인에서 실제로 본 연결(`rosgraph.observe`) — {"nodes","topics","edges"}. 한 번도 못 받았으면 None
    rosgraph: Mapping | None = None
    rosgraph_age_s: float | None = None
    #: 인지 런처가 vision-3090 에서 읽어 낸 상태 — {"camera_up","camera_hz","objects","viewer","busy","error"}
    perception: Mapping | None = None
    perception_age_s: float | None = None
    #: controller_manager 이름 → {"ok": bool, "controllers": [...]} 또는 {"ok": False, "reason": str}
    controllers: Mapping[str, Mapping] = field(default_factory=dict)
    #: controller_manager 이름 → 그 보고를 받은 지 몇 초
    controllers_age_s: Mapping[str, float] = field(default_factory=dict)
    #: 컨트롤러 프로브 프로세스가 죽은 이유 — 다시 답하면 None
    probe_down_why: str | None = None


@dataclass(frozen=True)
class Banner:
    state: str
    tone: str
    reasons: tuple[str, ...] = ()
    #: pd 가 실제로 명령을 내보내는 중인가(execute 파라미터 ∧ yaml). None = 모른다.
    armed_for_real: bool | None = None

    def as_dict(self) -> dict:
        return {"state": self.state, "tone": self.tone, "reasons": list(self.reasons),
                "armed_for_real": self.armed_for_real}


def _fresh(obs: Observed, node: str) -> Mapping | None:
    s = obs.status.get(node)
    if s is None or obs.age_s.get(node, 1e9) > STALE_S:
        return None
    return s


def derive(obs: Observed, *, expected_nodes: Sequence[str]) -> Banner:
    """위에서부터 먼저 걸리는 것 하나.

    ESTOP 이 HOLD 를 이긴다(HOLD 의 원인이 estop 일 수 있다). OFFLINE 이 전부를 이긴다 —
    안 보이는 로봇에 대해 "정상" 이라고 말하지 않는다.
    """
    if not obs.bridge_up:
        why = (obs.bridge_down_why,) if obs.bridge_down_why else ()
        return Banner(OFFLINE, TONE[OFFLINE], ("ROS 브리지가 떠 있지 않다", *why))
    if obs.bridge_faults:
        return Banner(FAULT, TONE[FAULT], tuple(obs.bridge_faults))

    pd = _fresh(obs, "pd")
    fresh = {n: _fresh(obs, n) for n in expected_nodes}
    seen = {n: s for n, s in fresh.items() if s is not None}
    if not seen:
        return Banner(OFFLINE, TONE[OFFLINE], ("status 가 하나도 오지 않는다 (노드가 떠 있지 않다)",))

    armed = None if pd is None else bool(pd.get("execute"))

    if pd is not None and pd.get("estop"):
        return Banner(ESTOP, TONE[ESTOP], ("pd 가 estop 래치를 보고한다 — 해제는 콘솔 밖에서 한다",), armed)

    bad = [f"{n}: {r}" for n, s in seen.items() if s.get("ok") is False for r in (s.get("reasons") or ["ok=false"])]
    pd_phase = None if pd is None else str(pd.get("phase", ""))

    if pd is not None and pd_phase == "HOLD":
        reasons = tuple(f"pd: {r}" for r in (pd.get("reasons") or ())) or ("pd 가 HOLD 다",)
        return Banner(HOLD, TONE[HOLD], reasons, armed)
    if bad:
        return Banner(FAULT, TONE[FAULT], tuple(bad), armed)
    if pd_phase == "RELEASING":
        return Banner(STOPPING, TONE[STOPPING], (), armed)

    running = [n for n, s in seen.items() if str(s.get("phase", "")).lower() == "running"]
    if running:
        return Banner(RUNNING, TONE[RUNNING], (), armed)
    if pd_phase in ("RAMPING", "TRACKING"):
        return Banner(ARMED, TONE[ARMED], (f"pd {pd_phase}",), armed)

    missing = tuple(f"{n}: status 없음" for n, s in fresh.items() if s is None)
    return Banner(IDLE, TONE[IDLE], missing, armed)

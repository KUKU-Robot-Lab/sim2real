"""브리지 → 콘솔 봉투(NDJSON 한 줄 = 한 사건)와, 그것을 모아 `Observed` 로 만드는 `Feed`.

순수하다 — rclpy 를 모른다. 시계는 주입받는다. 브리지 프로세스가 stdout 으로 이 줄들을 내고,
콘솔은 파이프에서 읽어 `Feed.ingest` 에 넣는다. 방향은 **브리지 → 콘솔 한쪽뿐**이다:
콘솔은 브리지에게 아무것도 시키지 않고, 브리지는 ROS 에 아무것도 발행하지 않는다.

채널:
  hello   {"ch","domain","nodes"}        브리지가 떴다. domain 은 rclpy 가 실제로 쓴 값
  status  {"ch","node","data"}           /policy_control/status/<node> 의 JSON 그대로
  episode {"ch","data"}                  /policy_control/episode 의 JSON 그대로
  beat    {"ch","graph"}                 1 Hz. graph = 지금 보이는 ROS 노드 이름들
  fault   {"ch","reason"}                브리지가 스스로 거부한 사유(도메인 불일치 등)
  topics  {"ch","data"}                  1 Hz. 스택 토픽별 {"pubs","n","age_ms"} — 브리지가 센 것
  rosgraph {"ch","data"}                 바뀔 때(+5 s 마다). 도메인 전체의 노드·토픽·연결 — 정책을 모른다
  controllers {"ch","manager","ok",...}  컨트롤러 프로브(`ctl_probe.py`)가 낸다. ok 면 "controllers", 아니면 "reason"
"""
from __future__ import annotations

import json
import time
from collections import deque
from typing import Callable, Mapping, Sequence

from . import pd_names as PD
from .console_state import Observed
from .telemetry import Window

#: beat 가 이보다 오래 없으면 브리지가 죽은 것으로 친다 [s].
BEAT_STALE_S = 3.0


def line(ch: str, **fields) -> str:
    return json.dumps({"ch": ch, **fields}, ensure_ascii=False, separators=(",", ":"))


class Feed:
    def __init__(self, nodes: Sequence[str], *, expect_domain: int, latency: tuple[str, str] | None = None,
                 clock: Callable[[], float] = time.monotonic, keep_events: int = 200) -> None:
        self.nodes = tuple(nodes)
        self.expect_domain = expect_domain
        self._clock = clock
        self._status: dict[str, Mapping] = {}
        self._seen_at: dict[str, float] = {}
        self._episode: Mapping | None = None
        self._percept: Mapping | None = None
        self._percept_at: float | None = None
        self._percept_why: str | None = None
        self._beat_at: float | None = None
        self._domain: int | None = None
        self._faults: list[str] = []
        self._down_why: str | None = None
        self._down_n = 0
        self._hold_key: tuple | None = None
        self.graph: tuple[str, ...] = ()
        self._topics: Mapping[str, Mapping] = {}
        self._topics_at: float | None = None
        self._joints: Mapping[str, Sequence] = {}
        self._joints_at: float | None = None
        self._rosgraph: Mapping | None = None
        self._rosgraph_at: float | None = None
        self._ctl: dict[str, Mapping] = {}
        self._ctl_at: dict[str, float] = {}
        self._ctl_key: dict[str, tuple] = {}
        self._probe_why: str | None = None
        self.events: deque[dict] = deque(maxlen=keep_events)
        first, last = latency or (None, None)
        self.window = Window(self.nodes, first=first, last=last)
        self.bad_lines = 0

    # ── 넣기 ────────────────────────────────────────────────────────────
    def ingest(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            msg = json.loads(text)
            ch = msg["ch"]
        except (ValueError, KeyError, TypeError):
            self.bad_lines += 1          # 브리지 stdout 에 섞인 rclpy 경고 등 — 세기만 한다
            return
        now = self._clock()
        if ch == "hello":
            self._beat_at = now
            self._domain = msg.get("domain")
            self._faults = []
            self._down_why, self._down_n = None, 0      # 붙었다 — "거듭 죽는 중" 이 끊겼다
            if self._domain != self.expect_domain:
                self._faults.append(f"브리지가 도메인 {self._domain} 에 붙었다 — 프로파일은 {self.expect_domain} 이다")
            self._event("bridge", f"브리지 연결 (도메인 {self._domain})")
        elif ch == "beat":
            self._beat_at = now
            self.graph = tuple(msg.get("graph") or ())
        elif ch == "fault":
            reason = str(msg.get("reason", "브리지 fault"))
            if reason not in self._faults:
                self._faults.append(reason)
                self._event("fault", reason)
        elif ch == "status":
            node, data = msg.get("node"), msg.get("data")
            if isinstance(node, str) and isinstance(data, Mapping):
                self._status[node] = data
                self._seen_at[node] = now
                self.window.offer(node, data)
                if PD.is_pd(node):                 # pd 는 팔마다 따로 온다(09.23)
                    self._note_hold(data)
        elif ch == "joints":
            data = msg.get("data")
            if isinstance(data, Mapping):
                self._joints, self._joints_at = data, now
        elif ch == "topics":
            data = msg.get("data")
            if isinstance(data, Mapping):
                self._topics, self._topics_at = data, now
        elif ch == "rosgraph":
            data = msg.get("data")
            if isinstance(data, Mapping):
                self._rosgraph, self._rosgraph_at = data, now
        elif ch == "controllers":
            manager = msg.get("manager")
            if isinstance(manager, str):
                self._probe_why = None
                self._ctl[manager] = {k: v for k, v in msg.items() if k not in ("ch", "manager")}
                self._ctl_at[manager] = now
                self._note_controllers(manager, self._ctl[manager])
        elif ch == "perception":
            data = msg.get("data")
            if isinstance(data, Mapping):
                self._percept, self._percept_at = data, now
                why = data.get("error")
                if why and why != self._percept_why:
                    self._event("인지", f"vision-3090: {why}")
                self._percept_why = why or None
        elif ch == "episode":
            data = msg.get("data")
            if isinstance(data, Mapping):
                self._episode = data
                reasons = "; ".join(str(r) for r in (data.get("reasons") or ()))
                self._event("episode", f"#{data.get('episode')} {data.get('event')}" + (f" — {reasons}" if reasons else ""))

    def bridge_died(self, why: str) -> None:
        """재시작 루프는 같은 이유로 몇 초마다 죽는다 — 사건 목록을 그것으로 채우지 않는다."""
        self._beat_at = None
        self._domain = None
        if why == self._down_why and self.events and self.events[-1].get("kind") == "bridge":
            self._down_n += 1
            self.events[-1] = {**self.events[-1], "t": time.time(), "text": f"브리지 종료 ×{self._down_n}: {why}"}
            return
        self._down_why, self._down_n = why, 1
        self._event("bridge", f"브리지 종료: {why}")

    def probe_died(self, why: str) -> None:
        """컨트롤러 프로브도 같은 이유로 거듭 죽는다 — 이유가 바뀔 때만 사건으로 남긴다."""
        if why != self._probe_why:
            self._event("stack", f"컨트롤러 프로브 종료: {why}")
        self._probe_why = why

    # ── 꺼내기 ──────────────────────────────────────────────────────────
    @property
    def domain(self) -> int | None:
        return self._domain

    def bridge_up(self) -> bool:
        return self._beat_at is not None and self._clock() - self._beat_at <= BEAT_STALE_S

    def observed(self) -> Observed:
        now = self._clock()
        return Observed(bridge_up=self.bridge_up(), status=dict(self._status),
                        age_s={n: now - t for n, t in self._seen_at.items()},
                        episode=self._episode, bridge_faults=tuple(self._faults),
                        bridge_down_why=None if self.bridge_up() else self._down_why,
                        graph=self.graph, topics=dict(self._topics),
                        joints=dict(self._joints),
                        joints_age_s=None if self._joints_at is None else now - self._joints_at,
                        topics_age_s=None if self._topics_at is None else now - self._topics_at,
                        rosgraph=self._rosgraph,
                        rosgraph_age_s=None if self._rosgraph_at is None else now - self._rosgraph_at,
                        perception=self._percept,
                        perception_age_s=None if self._percept_at is None else now - self._percept_at,
                        controllers=dict(self._ctl),
                        controllers_age_s={m: now - t for m, t in self._ctl_at.items()},
                        probe_down_why=self._probe_why)

    # ── 안 ──────────────────────────────────────────────────────────────
    def _event(self, kind: str, text: str) -> None:
        self.events.append({"t": time.time(), "kind": kind, "text": text})

    def _note_controllers(self, manager: str, report: Mapping) -> None:
        """프로브는 몇 초마다 같은 답을 낸다 — active 집합(또는 실패 사유)이 바뀔 때만 사건으로 남긴다."""
        if report.get("ok"):
            active = sorted(str(c.get("name")) for c in (report.get("controllers") or ())
                            if isinstance(c, Mapping) and c.get("state") == "active")
            key, text = ("ok", tuple(active)), f"{manager} active: " + (", ".join(active) or "없음")
        else:
            reason = str(report.get("reason", "응답 없음"))
            key, text = ("down", reason), f"{manager} 응답 없음 — {reason}"
        if key != self._ctl_key.get(manager):
            self._ctl_key = {**self._ctl_key, manager: key}
            self._event("stack", text)

    def _note_hold(self, data: Mapping) -> None:
        """HOLD 는 100 Hz 로 같은 사유를 반복한다 — (phase, 사유) 가 바뀔 때만 사건으로 남긴다."""
        phase = str(data.get("phase", ""))
        key = (phase, tuple(str(r) for r in (data.get("reasons") or ()))) if phase == "HOLD" else None
        if key != self._hold_key:
            if key is not None:
                self._event("hold", "pd HOLD — " + ("; ".join(key[1]) or "사유 없음"))
            elif self._hold_key is not None:
                self._event("hold", f"pd HOLD 해제 → {phase}")
            self._hold_key = key

"""연결 사슬 — 브리지 → 입력 → 정책 → 실기 → 가드 가 **지금 이어져 있는가**.

배너(`console_state.derive`)는 제일 급한 것 하나만 말한다. 이 표는 그 아래를 전부 펼친다 —
어느 입력이 살아 있고, 정책 노드가 몇 번째 seq 를 돌고 있고, pd 가 로봇 상태를 받고 있는지.

규칙은 배너와 같다.
- **저장하지 않고 매번 파생한다.** 입력의 live/stale 판정은 그 입력을 받는 노드가 제 문턱으로 내린 것을
  그대로 옮긴다(status 의 `inputs`, pd 의 `state_age_ms`/`state_stale_ms`). 콘솔은 문턱을 따로 두지 않는다.
- **안 보이는 것을 정상이라고 하지 않는다.** 브리지가 죽었거나 보고하던 노드가 조용해지면 그 아래는 unknown 이다.
- 순수하다 — rclpy·파일·시계를 보지 않는다. 색 표(TONE)도 여기 있다(JS 에 상태 지식을 두지 않는다).
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .console_state import STALE_S, Observed

LIVE, HELD, OFF, UNKNOWN = "live", "held", "off", "unknown"
STALE, MISSING, FAULT, DOWN = "stale", "missing", "fault", "down"

#: 흐르고 있으면 초록(ok). 보유는 고장도 흐름도 아니라 회색이다. 파랑(live)은 배너 전용.
TONE = {LIVE: "ok", HELD: "mute", OFF: "mute", UNKNOWN: "mute",
        STALE: "warn", MISSING: "bad", FAULT: "bad", DOWN: "bad"}
#: 블록 상태 = 행 중 가장 나쁜 것. off·held 는 정상 쪽이다(선택 입력이 꺼져 있음 / 래치 값).
_RANK = {OFF: 0, HELD: 1, LIVE: 2, UNKNOWN: 3, STALE: 4, MISSING: 5, FAULT: 6, DOWN: 6}

ROBOT_NODE = "pd"
#: pd 가 누구의 목표를 따르는가 — pd_arm 의 extras["target"] 그대로. 내부 유지는 정상(IDLE·홈·fabric 끈 fake)이라 held.
_TARGET = {"external": (LIVE, None, "체인 목표 추종"),
           "internal": (HELD, None, "내부 유지 목표 (체인 목표 안 받음)"),
           None: (OFF, None, "목표 없음")}


def _role(node: str) -> str:
    if node == ROBOT_NODE:
        return "robot"
    return "guard" if node.endswith("guard") else "policy"


def _row(name: str, state: str, age_ms: float | None = None, note: str = "") -> dict:
    state = state if state in TONE else UNKNOWN          # 노드가 모르는 단어를 보내면 live 로 넘기지 않는다
    return {"name": name, "state": state, "tone": TONE[state], "age_ms": age_ms, "note": note}


def _block(block_id: str, title: str, rows: Sequence[dict], *, detail: str = "", state: str | None = None) -> dict:
    if state is None:
        state = max((r["state"] for r in rows), key=_RANK.__getitem__, default=UNKNOWN)
    return {"id": block_id, "title": title, "state": state, "tone": TONE[state], "detail": detail, "rows": list(rows)}


def _fresh(obs: Observed, node: str) -> Mapping | None:
    s = obs.status.get(node)
    return None if s is None or obs.age_s.get(node, 1e9) > STALE_S else s


def _heartbeat(obs: Observed, node: str) -> dict:
    """노드 한 줄 — status 가 오는가, 오면 뭐라고 하는가."""
    s = obs.status.get(node)
    if s is None:
        return _row(node, MISSING, note="status 가 온 적 없다")
    age_ms = obs.age_s.get(node, 1e9) * 1e3
    if age_ms > STALE_S * 1e3:
        return _row(node, STALE, age_ms, "status 가 끊겼다")
    reasons = [str(r) for r in (s.get("reasons") or ())]
    if s.get("ok") is False or s.get("latched"):
        return _row(node, FAULT, age_ms, "; ".join(reasons) or "ok=false")
    seq = s.get("seq")
    bits = [str(s.get("phase") or ""), "" if seq is None or seq < 0 else f"seq {seq}"]
    return _row(node, LIVE, age_ms, " · ".join(b for b in bits if b))


def _inputs_block(obs: Observed, nodes: Sequence[str]) -> dict:
    rows, quiet, reporters = [], [], []
    for n in nodes:
        s = obs.status.get(n)
        if s is None or "inputs" not in s:
            continue
        if _fresh(obs, n) is None:
            quiet.append(n)
            continue
        reporters.append(n)
        rows += [_row(str(r.get("name")), str(r.get("state")), r.get("age_ms")) for r in s["inputs"]]
    if rows:
        return _block("inputs", "입력", rows, detail=f"{', '.join(reporters)} 가 받은 것")
    detail = (f"{', '.join(quiet)} status 가 끊겨 입력을 알 수 없다" if quiet
              else "입력 상태를 내는 노드가 아직 없다")
    return _block("inputs", "입력", [], detail=detail)


def _robot_block(obs: Observed, title: str) -> dict:
    rows = [_heartbeat(obs, ROBOT_NODE)]
    s = _fresh(obs, ROBOT_NODE)
    if s is None:
        return _block("robot", title, rows)
    for side, arm in (s.get("arms") or {}).items():
        limit = arm.get("state_stale_ms")
        for src, age in (arm.get("state_age_ms") or {}).items():
            state = MISSING if age is None else STALE if limit is not None and age > limit else LIVE
            rows.append(_row(f"{side}:{src}", state, age, "로봇 상태 수신"))
        if "target" in arm:
            rows.append(_row(f"{side}:목표", *_TARGET.get(arm["target"], (UNKNOWN, None, str(arm["target"])))))
    bits = [str(s.get("phase") or ""), "명령 나감" if s.get("execute") else "dry-run (명령 안 나감)"]
    if s.get("estop"):
        return _block("robot", title, rows, detail=" · ".join([*bits, "ESTOP 래치"]), state=FAULT)
    return _block("robot", title, rows, detail=" · ".join(b for b in bits if b))


#: 컨트롤러 조회는 2 s 주기다. 세 번 빠지면 모른다고 한다 [s].
CTL_STALE_S = 6.0
_CONTROLLER = {"active": LIVE, "inactive": OFF, "unconfigured": OFF, "finalized": OFF}


def _topic_row(obs: Observed, topic) -> dict:
    rep = obs.topics.get(topic.name)
    if rep is None or obs.topics_age_s is None or obs.topics_age_s > STALE_S:
        return _row(topic.name, UNKNOWN, note="브리지가 이 토픽을 보고하지 않는다")
    if rep.get("why"):
        return _row(topic.name, UNKNOWN, note=str(rep["why"]))
    age = rep.get("age_ms")
    if not rep.get("pubs"):
        return _row(topic.name, MISSING, age, "발행자가 없다")
    if age is None:
        return _row(topic.name, MISSING, note="발행자는 있는데 온 적 없다")
    if topic.stale_ms is not None and age > topic.stale_ms:
        return _row(topic.name, STALE, age, f"{topic.stale_ms:g} ms 넘게 끊겼다")
    note = f"{rep.get('n', 0)} Hz"
    if rep["pubs"] > 1:  # 같은 토픽을 둘이 내면 Hz 가 합쳐져 보인다 - 몇 명인지 같이 적는다
        note += f" · 발행자 {rep['pubs']}"
    return _row(topic.name, LIVE, age, note)


def _manager_rows(obs: Observed, manager) -> list[dict]:
    head = manager.name + " 컨트롤러"
    rep = obs.controllers.get(manager.name)
    if rep is None or obs.controllers_age_s.get(manager.name, 1e9) > CTL_STALE_S:
        return [_row(head, UNKNOWN, note=obs.probe_down_why or "컨트롤러 조회 결과가 없다")]
    if not rep.get("ok"):
        return [_row(head, MISSING, note=str(rep.get("reason") or "응답 없음"))]
    prefix = manager.name.rsplit("/", 1)[0]
    prefix = prefix + "/" if prefix else ""
    seen = {str(c.get("name")): str(c.get("state")) for c in rep.get("controllers", ())}
    rows = []
    for name in manager.active:                       # 있어야 하는 것 — 판정한다
        state = seen.get(name)
        if state is None:
            rows.append(_row(prefix + name, MISSING, note="로드되지 않았다"))
        elif state != "active":
            rows.append(_row(prefix + name, FAULT, note=f"{state} (active 여야 한다)"))
        else:
            rows.append(_row(prefix + name, LIVE, note="active"))
    for name, state in seen.items():                  # 나머지 — 상태만 보인다
        if name not in manager.active:
            rows.append(_row(prefix + name, _CONTROLLER.get(state, UNKNOWN), note=state))
    return rows


def _stack_block(obs: Observed, stack) -> dict:
    """policy_control 바깥 — robot_control bringup 등이 그래프·토픽·컨트롤러로 붙어 있는가."""
    graph = set(obs.graph)
    rows = [_row(n, LIVE if n in graph else MISSING, note="" if n in graph else "ROS 그래프에 없다") for n in stack.nodes]
    rows += [_topic_row(obs, t) for t in stack.topics]
    for m in stack.managers:
        rows += _manager_rows(obs, m)
    return _block("stack", stack.title, rows)


def chain(obs: Observed, *, expected_nodes: Sequence[str], domain: int | None, domain_class: str,
          stack=None) -> list[dict]:
    """신호가 흐르는 순서대로 블록 목록. 가드 블록은 가드 노드를 기대할 때만 있다."""
    by_role = {r: [n for n in expected_nodes if _role(n) == r] for r in ("policy", "robot", "guard")}
    robot_title = "실기 (pd)" if domain_class == "real" else "fake 플랜트 (pd)"
    titles = [("inputs", "입력"), ("policy", "정책"), ("robot", robot_title)] + [("guard", "가드")] * bool(by_role["guard"])

    stack_title = [("stack", stack.title)] if stack is not None else []
    where = "도메인 ?" if domain is None else f"도메인 {domain}"
    if not obs.bridge_up:
        why = obs.bridge_down_why or "ROS 브리지가 떠 있지 않다"
        return [_block("bridge", "ROS 브리지", [], detail=why, state=DOWN),
                *(_block(i, t, [], detail="브리지가 없어 알 수 없다") for i, t in stack_title + titles)]
    if obs.bridge_faults:
        bridge = _block("bridge", "ROS 브리지", [], detail="; ".join(obs.bridge_faults), state=FAULT)
    else:
        bridge = _block("bridge", "ROS 브리지", [], detail=where, state=LIVE)

    out = [bridge, *([_stack_block(obs, stack)] if stack is not None else []),
           _inputs_block(obs, expected_nodes),
           _block("policy", "정책", [_heartbeat(obs, n) for n in by_role["policy"]]),
           _robot_block(obs, robot_title)]
    if by_role["guard"]:
        out.append(_block("guard", "가드", [_heartbeat(obs, n) for n in by_role["guard"]]))
    return out

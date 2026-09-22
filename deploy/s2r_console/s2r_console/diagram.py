"""연결 그림 — 선언된 상자·전선(`diagram_spec`)을 **지금 본 것**으로 칠한다.

`links.chain` 과 같은 규칙이고 같은 낱말(live·held·off·unknown·stale·missing·fault·down)을 쓴다.
- 저장하지 않고 매번 파생한다. 순수하다 — rclpy·파일·시계를 보지 않는다.
- 안 보이는 것을 정상이라고 하지 않는다: 브리지가 죽었거나 토픽 보고가 늙으면 전부 unknown 이다.
- 전선은 증거 둘로 본다: 브리지가 그래프에서 읽은 발행자·구독자와 도착 주기, 그리고 **받는 노드가 스스로 한 말**
  (status 의 `inputs`). 둘이 다르면 받는 쪽 말이 이긴다 — 토픽이 흘러도 받는 노드가 못 받으면 끊긴 것이다.
- 꺼져 있는 것(`off`)과 죽은 것(`down`)을 가른다. 아무도 켜지 않은 입력은 고장이 아니다.
"""
from __future__ import annotations

from typing import Mapping

from .console_state import STALE_S, Observed
from .diagram_spec import Box, Diagram, Wire
from .links import (CTL_STALE_S, DOWN, FAULT, HELD, LIVE, MISSING, OFF, ROBOT_NODE, STALE, TONE, UNKNOWN, _RANK,
                    _heartbeat, _manager_rows, _robot_block)

_NO_UNIT = "미션에 없는 명령이다"
_CHAIN_NS = "/policy_control/"


def _worst(states) -> str:
    return max(states, key=_RANK.__getitem__, default=UNKNOWN)


def _short(topic: str) -> str:
    """상자 안에 적는 짧은 이름 — 끝 두 마디면 어느 손·어느 컨트롤러인지 남는다."""
    if topic.startswith(_CHAIN_NS):                      # 체인 내부 토픽은 머리가 전부 같다 — 떼어야 좁은 상자에서 구별된다
        return topic[len(_CHAIN_NS):]
    parts = topic.strip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else topic.lstrip("/")


def _receiver_rows(obs: Observed, box: Box) -> dict[str, Mapping]:
    """받는 노드가 status 로 스스로 보고한 입력 행. 조용해진 노드의 말은 쓰지 않는다."""
    if box.status is None:
        return {}
    s = obs.status.get(box.status)
    if s is None or obs.age_s.get(box.status, 1e9) > STALE_S:
        return {}
    rows = {str(r.get("name")): r for r in (s.get("inputs") or ())}
    for side, arm in (s.get("arms") or {}).items():            # pd 는 팔별 수신 나이로 말한다
        limit = arm.get("state_stale_ms")
        for src, age in (arm.get("state_age_ms") or {}).items():
            state = MISSING if age is None else STALE if limit is not None and age > limit else LIVE
            rows[f"{side}:{src}"] = {"name": f"{side}:{src}", "state": state, "age_ms": age}
    return rows


def _topic_state(obs: Observed, wire: Wire, dst: Box) -> tuple[str, str, str]:
    """브리지가 본 것만으로 (state, text, note)."""
    rep = obs.topics.get(wire.topic)
    if rep is None or obs.topics_age_s is None or obs.topics_age_s > STALE_S:
        return UNKNOWN, "", "브리지가 이 토픽을 보고하지 않는다"
    if rep.get("why"):
        return UNKNOWN, "", str(rep["why"])
    pubs = int(rep.get("pubs") or 0)
    ears = wire.heard_by or dst.ros                     # ros2_control 컨트롤러는 매니저가 아니라 제 이름의 노드로 구독한다
    heard = None if not ears else bool(set(ears) & set(rep.get("sub_nodes") or ()))
    if not pubs and heard is False:
        return OFF, "", "양쪽 다 없다 — 내는 쪽도 받는 쪽도 떠 있지 않다"
    if not pubs:
        return MISSING, "", "발행자가 없다"
    if heard is False:
        return OFF, "", f"받는 쪽({dst.title})이 이 토픽을 구독하고 있지 않다"
    many = f" · 발행자 {pubs}" if pubs > 1 else ""
    if not wire.meter:
        return LIVE, "연결됨" + many, "주기는 재지 않는다"
    age, limit = rep.get("age_ms"), wire.stale_ms if wire.stale_ms is not None else STALE_S * 1e3
    if age is None:
        return MISSING, "", "발행자는 있는데 온 적 없다"
    if age > limit:
        return STALE, f"{age / 1e3:.1f} s 전", f"{limit:g} ms 넘게 끊겼다"
    return LIVE, f"{rep.get('n', 0)} Hz{many}", ""


def _muted(obs: Observed, wire: Wire, src: Box, dst: Box) -> bool:
    """`execute:=false` 는 미션 argv 가 한 **선언**이다. 정말 흐르고 있으면 증거가 이긴다 —
    밖에서 pd 를 execute:=true 로 다시 띄웠는데 그림이 "무발행" 이라고 하면 실기가 움직이는 줄 모른다.

    흐른다 = 내는 쪽도 있고 받는 쪽도 듣는다. 받는 쪽 사유로 꺼 둔 전선(fake 손은 JTC 를 안 받는다)은
    내는 쪽이 있다고 해서 선언이 틀린 것이 아니다.
    """
    if not wire.muted:
        return False
    rep = obs.topics.get(wire.topic)
    if rep is None or obs.topics_age_s is None or obs.topics_age_s > STALE_S:
        said = obs.status.get(src.status) if src.status else None
        fresh = said is not None and obs.age_s.get(src.status, 1e9) <= STALE_S
        return not (fresh and said.get("execute") is True)   # 내는 쪽이 "명령 나감" 이라고 하면 그 선언은 낡았다
    ears = wire.heard_by or dst.ros
    heard = bool(ears) and bool(set(ears) & set(rep.get("sub_nodes") or ()))
    return not (int(rep.get("pubs") or 0) and heard)


def _controller_gate(obs: Observed, wire: Wire, dst: Box) -> tuple[str | None, str]:
    """흐르는 구동 전선이 정말 먹히는가. ros2_control 구독은 configure 에서 생겨서 **inactive 여도 구독은 한다** —
    토픽만 보면 초록인데 명령은 버려진다. 판정할 수 없으면 상태는 두고 사유만 적는다."""
    if dst.manager is None or not wire.heard_by or set(wire.heard_by) <= set(dst.ros):
        return None, ""                                  # 상자 제 노드가 듣는다(fake 플랜트) — 컨트롤러 이야기가 아니다
    rep = obs.controllers.get(dst.manager)
    if rep is None or obs.controllers_age_s.get(dst.manager, 1e9) > CTL_STALE_S or not rep.get("ok"):
        return None, "컨트롤러가 active 인지는 모른다 — 조회 결과가 없다"
    seen = {str(c.get("name")): str(c.get("state")) for c in rep.get("controllers") or ()}
    want = sorted({n.rsplit("/", 1)[-1] for n in wire.heard_by} & set(seen))   # 없는 것은 구독도 없다 — 위에서 이미 걸린다
    idle = [n for n in want if seen[n] != "active"]
    if idle:
        return FAULT, f"{', '.join(idle)} 가 active 가 아니다({seen[idle[0]]}) — 명령이 버려진다"
    return None, ""


def _port(obs: Observed, index: int, wire: Wire, spec: Diagram) -> dict:
    dst = spec.box(wire.dst)
    said = [r for name, r in _receiver_rows(obs, dst).items() if name in wire.inputs]
    theirs = _worst(str(r.get("state")) if str(r.get("state")) in TONE else UNKNOWN for r in said) if said else None
    if not obs.bridge_up:
        state, text, note = UNKNOWN, "", "브리지가 없어 알 수 없다"
    elif _muted(obs, wire, spec.box(wire.src), dst):                             # 흐르지 않는 것이 정상이다(pd 무발행) — 끊김으로 치지 않는다
        state, text, note = OFF, "", wire.muted
    elif wire.on_demand:
        # 가끔 한 번 내는 값: 발행자가 없는 것이 정상이다. 받는 쪽이 갖고 있다고 하면 그것이 상태다.
        state, text, note = (theirs, "", f"{dst.title} 가 보고한 값") if theirs else (OFF, "", "아직 들어온 적 없다")
    else:
        state, text, note = _topic_state(obs, wire, dst)
        if wire.muted:                                  # 여기 왔다는 것은 정말 흐른다는 뜻이다 — 선언이 졌다
            note = "; ".join(x for x in (note, f"그림은 무발행으로 본다({wire.muted}) — 그런데 흐르고 있다") if x)
        if theirs is not None and _RANK[theirs] > _RANK[state] and state == LIVE:
            state, note = theirs, f"토픽은 오는데 {dst.status} 가 {theirs} 라고 한다"
        if state == LIVE:
            gated, why = _controller_gate(obs, wire, dst)
            if why:
                state, text, note = (gated or state), ("" if gated else text), why
        if wire.episodic and state in (MISSING, STALE):
            resting = _resting(obs, spec.box(wire.src))
            if resting:
                state, text, note = HELD, "", resting
    return {"id": index, "from": wire.src, "topic": wire.topic, "label": wire.label or _short(wire.topic),
            "state": state, "tone": TONE[state], "text": text, "note": note}


def _resting(obs: Observed, src: Box) -> str:
    """에피소드 동안만 내는 전선이 쉬는 중인가 — 내는 노드가 **살아서** running 이 아니라고 말할 때만. 아니면 빈 문자열."""
    if src.status is None or obs.age_s.get(src.status, 1e9) > STALE_S:
        return ""
    phase = str((obs.status.get(src.status) or {}).get("phase") or "")
    return "" if phase in ("", "running") else f"에피소드 밖({phase}) — {src.title} 는 running 일 때만 낸다"


def _unit_verdict(unit: Mapping | None) -> tuple[str, str] | None:
    """프로세스가 떠 있지 않을 때 그 이유 — (state, detail). 떠 있거나 단위가 없으면 None."""
    if unit is None or unit.get("error") or unit.get("alive"):
        return None
    if unit.get("kind") == "manual":                     # 콘솔이 띄우지 않는 명령이다 — 안 떴다고 말할 근거가 없다
        return None
    if not unit.get("started") or unit.get("stopped"):
        return OFF, "꺼져 있다" if unit.get("started") else "아직 켜지 않았다"
    return DOWN, f"프로세스가 죽었다 (rc={unit.get('rc')}) — 로그를 볼 것"


def _status_box(obs: Observed, box: Box, title: str) -> tuple[str, str, list[dict]]:
    if box.status == ROBOT_NODE:
        block = _robot_block(obs, title)
        lines = [{"text": f"{r['name']} — {r['note']}", "tone": r["tone"]} for r in block["rows"] if r["name"].endswith(":목표")]
        beat = block["rows"][0]
        detail = block["detail"] or beat["note"]
        return block["state"] if block["detail"] else beat["state"], detail, lines
    beat = _heartbeat(obs, box.status)
    s = obs.status.get(box.status) or {}
    # fabric 을 끄고 돌린 체인은 joint_target 을 내지 않는다 — pd 로 가는 전선이 빨간 이유를 상자가 말한다
    lines = ([{"text": "fabric 꺼짐 — joint_target 을 내지 않는다 (pd 는 내부 목표를 따른다)", "tone": "warn"}]
             if beat["state"] == LIVE and s.get("use_fabric") is False else [])
    return beat["state"], beat["note"], lines


def _plain_box(obs: Observed, box: Box, spec: Diagram, ports: Mapping[int, dict]) -> tuple[str, str]:
    """status 를 내지 않는 상자 — 그래프에 노드가 보이는가, 내보내는 토픽에 발행자가 있는가."""
    graph, evidence, missing = set(obs.graph), [], []
    if box.manager is not None:                          # 구동 상자는 제 노드도 내보내는 토픽도 없다 — 매니저의 대답이 유일한 증거다
        rows = _manager_rows(obs, _Named(box.manager))
        thin = len(rows) == 1 and rows[0]["state"] in (UNKNOWN, MISSING)
        evidence.append(rows[0]["state"] if thin else LIVE)
        missing += [f"{box.manager}: {rows[0]['note']}"] if thin else []
    for n in box.ros:
        evidence.append(LIVE if n in graph else MISSING)
        missing += [] if n in graph else [n]
    for w in spec.wires:
        if w.src != box.id or w.on_demand:
            continue
        rep = obs.topics.get(w.topic)
        if rep is None or obs.topics_age_s is None or obs.topics_age_s > STALE_S:
            evidence.append(UNKNOWN)
        else:
            has = bool(rep.get("pubs"))
            evidence.append(LIVE if has else MISSING)
            missing += [] if has else [_short(w.topic)]
    if not evidence:                                   # 판정할 것이 없다(운영자 입력) — 제 전선을 따른다
        own = [ports[i]["state"] for i, w in enumerate(spec.wires) if w.src == box.id]
        return (_worst(own) if own else UNKNOWN), box.note
    state = _worst(evidence)
    return state, ("없는 것: " + ", ".join(missing)) if missing else box.note


#: 인지 런처 상태가 이보다 늙으면 저 PC 의 일은 모른다 [s]. 런처는 1 Hz 로 낸다.
PERCEPT_STALE_S = 6.0
_LAUNCHER = "perception"


def _percept(obs: Observed) -> Mapping | None:
    """인지 런처가 방금 한 말. 늙었거나 없으면 None — 저 PC 의 일은 여기서 알 길이 없다."""
    if obs.perception is None or obs.perception_age_s is None or obs.perception_age_s > PERCEPT_STALE_S:
        return None
    return obs.perception


def _percept_box(obs: Observed, box: Box) -> tuple[str, str, list[dict]] | None:
    """vision-3090 에서 도는 것(카메라·FP++ 컨테이너)은 DDS 가 아니라 **런처의 말**로 판정한다.

    영상 토픽은 일부러 구독하지 않는다(PC 사이로 영상을 끌어오지 않는다) — 그래서 그래프만으로는
    "발행자가 있다" 밖에 못 본다. 런처는 저 PC 에서 `docker ps` 와 프로세스를 직접 본다.
    """
    said = _percept(obs)
    if said is None:
        return None
    objects = said.get("objects") or {}
    if box.id == _LAUNCHER:
        error = said.get("error")
        rows = [{"text": f"카메라: {'up' if said.get('camera_up') else 'down'} ({said.get('camera_hz')} Hz)",
                 "tone": "ok" if said.get("camera_up") else "mute"}]
        rows += [{"text": f"{name}: {info.get('container') or '컨테이너 없음'}",
                  "tone": "ok" if info.get("container") else "mute"} for name, info in objects.items()]
        if error:
            return FAULT, str(error), rows
        return LIVE, f"{box.host or '원격'} 을 보고 있다" + (" · 작업 중" if said.get("busy") else ""), rows
    if box.id == "camera":
        if said.get("camera_up"):
            return LIVE, f"{said.get('camera_hz')} Hz", []
        return OFF, f"{box.host} 에서 카메라가 떠 있지 않다 — 인지 런처로 켤 것", []
    if box.id.startswith("fpp_"):
        info = objects.get(box.id[len("fpp_"):]) or {}
        if info.get("container"):
            age = info.get("pose_age_s")
            return LIVE, f"{info['container']}" + ("" if age is None else f" · 포즈 {age:.1f} s 전"), []
        return OFF, f"컨테이너가 없다 — 인지 런처로 켤 것 ({box.host})", []
    return None


def _box(obs: Observed, box: Box, spec: Diagram, ports: Mapping[int, dict], units: Mapping[str, Mapping]) -> dict:
    unit = None if box.unit is None else dict(units.get(box.unit) or {"key": box.unit, "error": _NO_UNIT})
    lines: list[dict] = []
    said = None if not obs.bridge_up else _percept_box(obs, box)
    if not obs.bridge_up:
        state, detail = UNKNOWN, "브리지가 없어 알 수 없다"
    elif said is not None:
        state, detail, lines = said
    elif box.status is not None:
        state, detail, lines = _status_box(obs, box, box.title)
    else:                                                # 런처가 없으면 토픽만으로 본다 — 약하지만 증거는 증거다
        state, detail = _plain_box(obs, box, spec, ports)
        if box.host and not detail:
            detail = f"{box.host} · 인지 런처가 없어 컨테이너 상태는 모른다"
    verdict = _unit_verdict(unit)
    if obs.bridge_up and verdict is not None and state in (MISSING, STALE, UNKNOWN):
        state, detail = verdict                        # 안 보이는 이유를 안다 — 꺼 둔 것인지 죽은 것인지
    if obs.bridge_up and box.manager is not None:
        lines += _controller_lines(obs, box.manager)
    shares = [b.title for b in spec.boxes if b.unit and b.unit == box.unit and b.id != box.id]
    return {"id": box.id, "title": box.title, "host": box.host,
            "state": state, "tone": TONE[state], "detail": detail, "lines": lines,
            "ports": [ports[i] for i, w in enumerate(spec.wires) if w.dst == box.id], "unit": unit, "shares": shares,
            "stages": list(box.stages)}


def _controller_lines(obs: Observed, manager: str) -> list[dict]:
    """컨트롤러는 많고 이름이 길다 — 상자 안에서는 active / 나머지 두 줄로 접는다. 조회가 안 되면 그 사유 한 줄."""
    rows = _manager_rows(obs, _Named(manager))
    if len(rows) == 1 and rows[0]["state"] in (UNKNOWN, MISSING):
        return [{"text": f"컨트롤러: {rows[0]['note']}", "tone": rows[0]["tone"]}]
    short = lambda r: r["name"].rsplit("/", 1)[-1].removesuffix("_controller")  # noqa: E731
    on = [short(r) for r in rows if r["note"] == "active"]
    rest = [short(r) for r in rows if r["note"] != "active"]
    return ([{"text": "active: " + ", ".join(on), "tone": "ok"}] if on else []) + \
           ([{"text": "inactive: " + ", ".join(rest), "tone": "mute"}] if rest else [])


class _Named:
    """`links._manager_rows` 가 읽는 모양(name, active) — 그림에서는 판정 없이 상태만 적는다."""

    def __init__(self, name: str) -> None:
        self.name, self.active = name, ()


def _summary(boxes: list[dict], spec: Diagram) -> dict:
    """상자가 다 떠 있어도 전선이 끊겨 있을 수 있다 — 둘 다 센다. 못 본 것을 이어졌다고 하지 않는다."""
    title = {b["id"]: b["title"] for b in boxes}
    where = lambda b, p: f"{title[p['from']]} → {b['title']} ({p['label']})"  # noqa: E731
    broken = [b["title"] for b in boxes if b["tone"] in ("warn", "bad")]
    cut = [where(b, p) for b in boxes for p in b["ports"]
           if p["tone"] in ("warn", "bad") and b["title"] not in broken and title[p["from"]] not in broken]
    if broken or cut:
        return {"tone": "bad", "text": "끊긴 곳: " + " · ".join([*broken, *cut])}
    if not all(b["tone"] == "ok" or b["state"] == HELD for b in boxes):
        return {"tone": "mute", "text": "아직 다 켜지지 않았다"}
    # 여기부터는 상자가 전부 초록이다 — 그런데도 확인 못한 전선이 있으면 그것이 머리말이다.
    dark = [where(b, p) for b in boxes for p in b["ports"]
            if p["state"] == UNKNOWN or (p["state"] == OFF and not _expected_off(spec.wires[p["id"]]))]
    if dark:
        return {"tone": "warn", "text": "상자는 다 떠 있는데 확인 못한 전선: " + " · ".join(dark)}
    return {"tone": "ok", "text": "전부 이어짐"}


def _expected_off(wire: Wire) -> bool:
    """흐르지 않는 것이 정상인 전선 — pd 무발행(execute:=false)과 가끔 한 번 오는 값."""
    return bool(wire.muted) or wire.on_demand


_PLUMBING = ("/policy_control/status/",)               # 콘솔이 이미 상자 상태로 읽는 것 — 그림 밖 목록에 다시 적지 않는다


def _extra(obs: Observed, spec: Diagram) -> dict | None:
    """도메인에 실제로 있는데 그림이 선언하지 않은 토픽·노드. 그래프를 못 봤으면 None(빈 목록과 다르다)."""
    if obs.rosgraph is None or not obs.bridge_up:        # 브리지가 죽으면 마지막 스냅샷은 지금이 아니다
        return None
    drawn = {w.topic for w in spec.wires}
    known = {n for b in spec.boxes for n in b.ros} | {n for w in spec.wires for n in w.heard_by}
    topics = [t for t in obs.rosgraph.get("topics") or ()
              if t.get("name") not in drawn and not str(t.get("name")).startswith(_PLUMBING)]
    nodes = sorted({n for t in topics for n in (*(t.get("pubs") or ()), *(t.get("subs") or ()))} - known)
    return {"topics": topics, "nodes": nodes, "age_s": obs.rosgraph_age_s}


def build(obs: Observed, spec: Diagram, *, units: Mapping[str, Mapping]) -> dict:
    """화면이 그대로 그리는 모양. `units` 는 키 → 프로세스 상태(`units.view`)."""
    ports = {i: _port(obs, i, w, spec) for i, w in enumerate(spec.wires)}
    boxes = [_box(obs, b, spec, ports, units) for b in spec.boxes]
    cols = [[b for b, s in zip(boxes, spec.boxes) if s.col == c] for c in sorted({s.col for s in spec.boxes})]
    wires = [{"id": i, "from": w.src, "to": w.dst, "state": ports[i]["state"], "tone": ports[i]["tone"],
              "flow": ports[i]["state"] == LIVE and w.meter and not w.on_demand} for i, w in enumerate(spec.wires)]
    return {"cols": cols, "wires": wires, "summary": _summary(boxes, spec), "extra": _extra(obs, spec)}


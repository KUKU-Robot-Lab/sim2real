"""HTTP 라우터 — 소켓을 모른다. (method, path, body, headers) → (status, payload).

쓰기 규칙:
  · 모든 쓰기는 헤더 `X-S2R-Console: 1` 과 JSON 본문을 요구한다. 브라우저는 다른 출처에서 커스텀 헤더를
    preflight 없이 못 붙이고, 이 서버는 OPTIONS 에 답하지 않는다 → 다른 사이트가 localhost 로 POST 를 쏠 수 없다.
  · `Host` 가 localhost/127.0.0.1 이 아니면 거부한다 (DNS rebinding).
  · 로봇을 **움직이게 하는** 쓰기는 lease 토큰(`X-S2R-Lease`)이 있어야 한다.
    **멈추는** 쓰기(`/api/quick/*`)는 lease 없이 된다 — 화면을 보고 있는 누구든 멈출 수 있어야 한다.
  · argv 는 어떤 경로로도 받지 않는다. 오는 것은 단계 id 와 동작 이름뿐이다.
"""
from __future__ import annotations

import json
from typing import Mapping

from .console import Console, ConsoleError
from .lease import LeaseError

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def host_ok(host: str | None) -> bool:
    if not host:
        return False
    name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
    return name in _LOCAL_HOSTS


def _fail(code: int, message: str, reasons=()) -> tuple[int, dict]:
    return code, {"ok": False, "error": message, "reasons": list(reasons) or [message]}


def handle(console: Console, method: str, path: str, *, query: Mapping[str, str], body: bytes,
           headers: Mapping[str, str], client: str) -> tuple[int, dict]:
    h = {k.lower(): v for k, v in headers.items()}
    if not host_ok(h.get("host")):
        return _fail(403, f"Host {h.get('host')!r} 는 허용되지 않는다")
    try:
        if method == "GET":
            return _get(console, path, query)
        if method in ("POST", "DELETE"):
            if h.get("x-s2r-console") != "1":
                return _fail(403, "X-S2R-Console 헤더가 없다")
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except ValueError:
                return _fail(400, "본문이 JSON 이 아니다")
            if not isinstance(data, dict):
                return _fail(400, "본문이 객체가 아니다")
            return _write(console, method, path, data, h.get("x-s2r-lease"), client)
        return _fail(405, f"{method} 는 지원하지 않는다")
    except LeaseError as exc:
        return _fail(409, str(exc))
    except ConsoleError as exc:
        return _fail(exc.code, str(exc), exc.reasons)


def _get(console: Console, path: str, query: Mapping[str, str]) -> tuple[int, dict]:
    if path == "/api/state":
        return 200, console.snapshot()
    if path == "/api/log":
        return 200, {"ok": True, "key": query.get("key", ""), "text": console.log_tail(query.get("key", ""))}
    return _fail(404, f"없는 경로: {path}")


def _write(console: Console, method: str, path: str, data: dict, token: str | None, client: str) -> tuple[int, dict]:
    ok = {"ok": True}
    # ── lease ───────────────────────────────────────────────────────────
    if path == "/api/lease":
        if method == "DELETE":
            console.lease.release(token)
            return 200, ok
        held = console.lease.acquire(str(data.get("operator", "")), force=bool(data.get("force")))
        if data.get("force") and console.session is not None:
            console.session.event("lease", f"{held.holder} 가 lease 를 강제로 가져갔다 ({client})")
        return 200, {"ok": True, "token": held.token, "holder": held.holder}
    if path == "/api/lease/renew":
        console.lease.renew(token or "")
        return 200, ok

    # ── 멈추기: lease 없이 ──────────────────────────────────────────────
    if path.startswith("/api/quick/"):
        console.quick(path.rsplit("/", 1)[1], client=client)
        return 202, ok

    # ── 나머지는 lease 가 있어야 ────────────────────────────────────────
    held = console.lease.check(token)
    console.lease.renew(held.token)
    who = held.holder
    if path == "/api/run/open":
        s = console.open(str(data.get("profile", "")), operator=who)
        return 200, {"ok": True, "run_id": s.run_id}
    if path == "/api/run/end":
        return 200, {"ok": True, "stopped": console.end(force=bool(data.get("force")))}
    if path == "/api/approve":
        console.approve(str(data.get("stage", "")), operator=who, typed=str(data.get("typed", "")), note=str(data.get("note", "")))
        return 200, ok
    if path == "/api/revoke":
        console.revoke(str(data.get("stage", "")), operator=who)
        return 200, ok
    if path == "/api/stage/run":
        console.run_stage(str(data.get("stage", "")), operator=who)
        return 202, ok
    if path == "/api/stage/rewind":
        console.rewind(str(data.get("stage", "")), operator=who)
        return 200, ok
    if path == "/api/stage/skip":
        console.skip_stage(str(data.get("stage", "")), operator=who)
        return 200, ok
    if path == "/api/stage/ack":
        console.ack(int(data.get("index", -1)), bool(data.get("ok")))
        return 200, ok
    if path == "/api/stage/abort":
        console.abort_stage()
        return 202, ok
    if path == "/api/unit":
        # 오는 것은 키와 켬/끔뿐이다. argv 는 미션 yaml 에서만 온다 — 본문의 다른 필드는 읽지 않는다.
        if not isinstance(data.get("on"), bool):
            return _fail(400, "'on' 은 true/false 여야 한다")
        console.toggle_unit(str(data.get("key", "")), data["on"], operator=who)
        return 202, ok
    return _fail(404, f"없는 경로: {path}")

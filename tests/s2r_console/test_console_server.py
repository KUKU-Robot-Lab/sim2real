"""HTTP 서버 경계 — 읽지 않은 본문이 다음 요청으로 해석되면 안 된다(요청 밀반입)."""
from __future__ import annotations

import socket
import threading

import pytest

from s2r_console import server


class _Stub:
    """api.handle 이 부르는 면만 흉내 낸다 — 무엇이 불렸는지 적는다."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def quick(self, name, **kw):  # noqa: ANN001, ANN003
        self.calls.append(("quick", name))
        return {"ok": True}

    def snapshot(self) -> dict:
        return {}


@pytest.fixture()
def live():
    stub = _Stub()
    httpd = server.serve(stub, bind="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield stub, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _exchange(port: int, raw: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
        sock.sendall(raw)
        chunks = []
        while True:
            try:
                got = sock.recv(65536)
            except socket.timeout:
                break
            if not got:
                break
            chunks.append(got)
    return b"".join(chunks)


def _inner() -> bytes:
    return (b"POST /api/quick/pd_release HTTP/1.1\r\nHost: 127.0.0.1\r\nX-S2R-Console: 1\r\n"
            b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}")


def test_an_oversized_body_is_never_parsed_as_the_next_request(live):
    stub, port = live
    body = _inner() + b"x" * (server._MAX_BODY + 100)
    outer = (b"POST /x HTTP/1.1\r\nHost: evil.example\r\nContent-Type: text/plain\r\n"
             + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)

    reply = _exchange(port, outer)

    assert reply.startswith(b"HTTP/1.1 413")
    assert reply.count(b"HTTP/1.1 ") == 1
    assert stub.calls == []


@pytest.mark.parametrize("length", ["-1", "abc", "1e3"])
def test_a_malformed_length_is_refused_and_the_connection_closed(live, length):
    stub, port = live
    raw = (b"POST /api/quick/pd_release HTTP/1.1\r\nHost: 127.0.0.1\r\nX-S2R-Console: 1\r\n"
           + f"Content-Length: {length}\r\n\r\n".encode() + _inner())

    reply = _exchange(port, raw)

    assert reply.startswith(b"HTTP/1.1 400")
    assert reply.count(b"HTTP/1.1 ") == 1
    assert stub.calls == []


def test_a_chunked_body_is_refused_and_the_connection_closed(live):
    stub, port = live
    raw = (b"POST /api/quick/pd_release HTTP/1.1\r\nHost: 127.0.0.1\r\nX-S2R-Console: 1\r\n"
           b"Transfer-Encoding: chunked\r\n\r\n" + _inner())

    reply = _exchange(port, raw)

    assert reply.startswith(b"HTTP/1.1 400")
    assert reply.count(b"HTTP/1.1 ") == 1
    assert stub.calls == []


def test_an_error_reply_closes_the_connection(live):
    _stub, port = live
    raw = b"POST /api/nope HTTP/1.1\r\nHost: evil.example\r\nContent-Length: 0\r\n\r\n" + _inner()

    reply = _exchange(port, raw)

    assert reply.count(b"HTTP/1.1 ") == 1
    assert b"Connection: close" in reply

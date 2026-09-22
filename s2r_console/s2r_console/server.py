"""stdlib HTTP 서버 — 의존성 0. 127.0.0.1 에만 묶는다. 원격에서는 ssh -L 로 본다.

    ssh -L 8091:127.0.0.1:8091 <robot-pc>   →   http://127.0.0.1:8091

`scripts/vision/cup_view_stream.py` 가 이미 같은 방식(ThreadingHTTPServer + 터널)으로 운영되고 있다.
"""
from __future__ import annotations

import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

from . import _paths
from .api import handle, host_ok
from .console import Console

STREAM_PERIOD_S = 0.5
_MAX_BODY = 1 << 16
_CLOSE_FROM = 400   # 오류 응답 뒤에는 연결을 끊는다 — 읽지 않은 본문이 다음 요청이 되지 않게


def make_handler(console: Console):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "s2r_console"

        def log_message(self, fmt, *args):  # noqa: A003, ANN001 — 접속 로그는 쓰지 않는다 (2 Hz 로 쌓인다)
            return

        # ── 읽기 ────────────────────────────────────────────────────────
        def do_GET(self):  # noqa: N802
            url = urlsplit(self.path)
            if url.path == "/api/stream":
                return self._stream()
            if url.path.startswith("/api/"):
                return self._api("GET", url, b"")
            return self._static(url.path)

        def do_POST(self):  # noqa: N802
            self._write("POST")

        def do_DELETE(self):  # noqa: N802
            self._write("DELETE")

        # ── 안 ──────────────────────────────────────────────────────────
        def _write(self, method: str) -> None:
            """본문을 끝까지 읽지 못하는 경로는 전부 연결을 끊는다 — 남은 바이트가 다음 요청으로 해석되면
            공격자가 Host·X-S2R-Console 을 직접 적은 요청을 밀어 넣을 수 있다(요청 밀반입)."""
            if self.headers.get("Transfer-Encoding"):
                return self._send(400, {"ok": False, "error": "Transfer-Encoding 은 받지 않는다"})
            text = (self.headers.get("Content-Length") or "0").strip()
            if not (text.isascii() and text.isdigit()):
                return self._send(400, {"ok": False, "error": "Content-Length 가 음이 아닌 정수가 아니다"})
            n = int(text)
            if n > _MAX_BODY:
                return self._send(413, {"ok": False, "error": "본문이 너무 크다"})
            self._api(method, urlsplit(self.path), self.rfile.read(n) if n else b"")

        def _api(self, method: str, url, body: bytes) -> None:
            code, payload = handle(console, method, url.path, query=dict(parse_qsl(url.query)), body=body,
                                   headers=dict(self.headers.items()), client=f"{self.client_address[0]}:{self.client_address[1]}")
            self._send(code, payload)

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            if code >= _CLOSE_FROM:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(raw)

        def _static(self, path: str) -> None:
            name = "index.html" if path in ("/", "") else path.removeprefix("/static/")
            target = (_paths.WEB / name).resolve()
            if _paths.WEB.resolve() not in target.parents or not target.is_file():
                return self._send(404, {"ok": False, "error": "없는 파일"})
            raw = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", (mimetypes.guess_type(target.name)[0] or "application/octet-stream") + "; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _stream(self) -> None:
            if not host_ok(self.headers.get("Host")):
                return self._send(403, {"ok": False, "error": "Host"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    raw = json.dumps(console.snapshot(), ensure_ascii=False)
                    self.wfile.write(f"event: state\ndata: {raw}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(STREAM_PERIOD_S)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    return Handler


def serve(console: Console, *, bind: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((bind, port), make_handler(console))
    server.daemon_threads = True
    return server

"""S2R 배포 콘솔.

    source /opt/ros/humble/setup.bash
    python3 -m s2r_console                       # 프로파일을 화면에서 고른다
    python3 -m s2r_console --profile pour_i18_fake --operator me
    python3 -m s2r_console --window              # 브라우저 대신 자기 창으로(GTK + WebKit)
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading

from .console import Console, ConsoleError
from .server import serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S2R 배포 콘솔 (브라우저 UI)")
    ap.add_argument("--bind", default="127.0.0.1", help="바꾸지 말 것 — 인증이 없다. 원격은 ssh -L 로 본다")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--profile", help="시작하자마자 열 프로파일 id")
    ap.add_argument("--operator", default="", help="--profile 과 함께: run 을 여는 운영자 이름")
    ap.add_argument("--no-bridge", action="store_true", help="ROS 브리지를 띄우지 않는다 (rclpy 없는 PC 에서 화면만 볼 때)")
    ap.add_argument("--window", action="store_true",
                    help="브라우저 대신 자기 창으로 연다. 창을 닫으면 콘솔이 내려간다(서버는 그동안 그대로 떠 있어 ssh -L 도 된다)")
    args = ap.parse_args(argv)
    if args.window:
        try:
            from . import window
            window.load()                         # 서버를 띄우기 전에 창을 열 수 있는지부터 본다
        except ImportError as exc:
            print(f"[s2r_console] 창을 열 수 없다: {exc}", file=sys.stderr)
            return 2
    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        print(f"[s2r_console] ★{args.bind} 에 묶는다 — 이 서버는 인증이 없다. 같은 망의 누구든 로봇 명령을 낼 수 있다.", file=sys.stderr)

    console = Console(bridge=not args.no_bridge)
    if args.profile:
        try:
            console.open(args.profile, operator=args.operator or "cli")
        except ConsoleError as exc:
            print(f"[s2r_console] {exc}", file=sys.stderr)
            return 2

    server = serve(console, bind=args.bind, port=args.port)

    def _term(signum, frame):  # noqa: ANN001, ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    print(f"[s2r_console] http://{args.bind}:{args.port}   (원격: ssh -L {args.port}:127.0.0.1:{args.port} <이 PC>)", flush=True)
    kept: list[str] = []
    try:
        if args.window:
            threading.Thread(target=server.serve_forever, name="s2r-http", daemon=True).start()
            window.run(f"http://127.0.0.1:{args.port}/", title="S2R 콘솔", close_warning=lambda: close_warning(console))
            print("[s2r_console] 창이 닫혔다 — 내려간다", flush=True)
            server.shutdown()
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\n[s2r_console] 내려간다 — 띄운 자식을 정지한다", flush=True)
    finally:
        kept = console.shutdown()
        server.server_close()
        if kept:
            print("[s2r_console] ★실기 프로세스는 남겼다 — 끄면 모터 토크가 풀린다: " + ", ".join(kept), flush=True)
            print("[s2r_console]   콘솔을 다시 띄워도 이 프로세스들을 넘겨받지 않는다. 팔을 안전 자세에 둔 뒤 PID 로 정리할 것", flush=True)
    if args.window and kept:
        window.notice("실기 프로세스를 남겼다", KEPT_NOTICE.format(keys=", ".join(kept)))
    return 0


KEPT_NOTICE = ("끄면 모터 토크가 풀리므로 남겼다: {keys}\n\n"
               "콘솔을 다시 띄워도 이 프로세스들을 넘겨받지 않는다. 팔을 안전 자세에 둔 뒤 PID 로 정리할 것.")


def close_warning(console: Console) -> str:
    """창을 닫기 전에 알릴 것 — 도는 단계와 살아 있는 자식. 없으면 빈 문자열(바로 닫는다)."""
    s = console.session
    if s is None:
        return ""
    busy = sorted(lane for lane, r in s.runners.items() if r.active)
    live = sorted(p.key for p in s.supervisor.alive())
    if not busy and not live:
        return ""
    lines = []
    if busy:
        lines.append("진행 중인 레인: " + ", ".join(busy))
    if live:
        lines.append("살아 있는 프로세스: " + ", ".join(live))
    what = "실기 팔·손 드라이버는 남기고 나머지는 정지한다" if s.profile.is_real else "띄운 자식을 모두 정지한다"
    return "\n".join(lines) + f"\n\n닫으면 콘솔이 내려가며 {what}."


if __name__ == "__main__":
    sys.exit(main())

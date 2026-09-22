"""S2R 배포 콘솔.

    source /opt/ros/humble/setup.bash
    python3 -m s2r_console                       # 프로파일을 화면에서 고른다
    python3 -m s2r_console --profile pour_i18_fake --operator me
"""
from __future__ import annotations

import argparse
import signal
import sys

from .console import Console, ConsoleError
from .server import serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S2R 배포 콘솔 (브라우저 UI)")
    ap.add_argument("--bind", default="127.0.0.1", help="바꾸지 말 것 — 인증이 없다. 원격은 ssh -L 로 본다")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--profile", help="시작하자마자 열 프로파일 id")
    ap.add_argument("--operator", default="", help="--profile 과 함께: run 을 여는 운영자 이름")
    ap.add_argument("--no-bridge", action="store_true", help="ROS 브리지를 띄우지 않는다 (rclpy 없는 PC 에서 화면만 볼 때)")
    args = ap.parse_args(argv)
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[s2r_console] 내려간다 — 띄운 자식을 정지한다", flush=True)
    finally:
        kept = console.shutdown()
        server.server_close()
        if kept:
            print("[s2r_console] ★실기 프로세스는 남겼다 — 끄면 모터 토크가 풀린다: " + ", ".join(kept), flush=True)
            print("[s2r_console]   콘솔을 다시 띄워도 이 프로세스들을 넘겨받지 않는다. 팔을 안전 자세에 둔 뒤 PID 로 정리할 것", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

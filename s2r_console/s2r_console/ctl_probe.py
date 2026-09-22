"""컨트롤러 프로브 — `<manager>/list_controllers` 를 주기로 물어 stdout 에 NDJSON 으로 낸다(`feed.py`).

브리지와 **따로** 둔다: 브리지는 DDS 에 아무것도 쓰지 않는다는 약속을 지키고, 서비스 요청을 보내는 쪽은
이 프로세스 하나뿐이다. 보내는 것은 읽기 전용 `list_controllers` 뿐 — switch·load·configure 는 코드에 없다.
프로파일이 `stack.managers` 를 선언할 때만 뜬다.

    python3 -m s2r_console.ctl_probe --domain 97 --managers /controller_manager
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable

from .bridge import domain_refusal
from .feed import line

PERIOD_S = 2.0
WAIT_S = 0.5
CALL_S = 1.5


def _emit(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def summarize(controllers: Iterable) -> list[dict]:
    """ControllerState 에서 화면이 쓰는 것만. 순수."""
    return [{"name": str(c.name), "state": str(c.state), "type": str(c.type)} for c in controllers]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", type=int, required=True)
    ap.add_argument("--managers", nargs="+", required=True)
    args = ap.parse_args(argv)

    refusal = domain_refusal(os.environ.get("ROS_DOMAIN_ID"), args.domain)
    if refusal:
        _emit(line("fault", reason=refusal))
        return 2

    import rclpy  # noqa: PLC0415 — 거부 검사 뒤에 올린다
    from controller_manager_msgs.srv import ListControllers
    from rclpy.executors import ExternalShutdownException

    rclpy.init(domain_id=args.domain)
    node = rclpy.create_node("s2r_console_ctl_probe", enable_rosout=False, start_parameter_services=False)
    clients = {m: node.create_client(ListControllers, m.rstrip("/") + "/list_controllers") for m in args.managers}

    def ask(manager: str) -> None:
        client = clients[manager]
        if not client.wait_for_service(timeout_sec=WAIT_S):
            _emit(line("controllers", manager=manager, ok=False, reason="list_controllers 서비스가 없다"))
            return
        future = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=CALL_S)
        if not future.done():
            future.cancel()
            _emit(line("controllers", manager=manager, ok=False, reason=f"{CALL_S} s 안에 답이 없다"))
            return
        if future.exception() is not None:
            _emit(line("controllers", manager=manager, ok=False, reason=f"호출 실패: {future.exception()}"))
            return
        _emit(line("controllers", manager=manager, ok=True, controllers=summarize(future.result().controller)))

    try:
        while rclpy.ok():
            began = time.monotonic()
            for m in args.managers:
                ask(m)
            time.sleep(max(0.0, PERIOD_S - (time.monotonic() - began)))
    except (KeyboardInterrupt, ExternalShutdownException, BrokenPipeError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

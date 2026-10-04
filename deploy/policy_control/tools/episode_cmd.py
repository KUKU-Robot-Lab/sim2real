#!/usr/bin/env python3
"""에피소드 실행기 명령 — 승인 한 줄을 쓰고 episode_runner 서비스를 부른다. rc 0 = 받음, 1 = 거부.

    python3 deploy/policy_control/tools/episode_cmd.py status                          # 지금 노드 · 다음 승인 이름(읽기만)
    python3 deploy/policy_control/tools/episode_cmd.py next --approve pick_cup --execute   # 구분 실행: 노드 하나
    python3 deploy/policy_control/tools/episode_cmd.py run --approve episode:pick_place_right --execute   # 연속 실행
    python3 deploy/policy_control/tools/episode_cmd.py stop --execute                  # 멈춤(승인 없음 — 로봇을 덜 움직인다)
    python3 deploy/policy_control/tools/episode_cmd.py reset --execute

규약(trigger.py 와 같다): `--execute` 없이는 아무것도 부르지 않는다 · ROS_DOMAIN_ID 0/unset 거부.
next · run 은 `--approve` 가 실행기가 물을 이름(status 의 next_action · run 은 episode:<이름>)과 같아야 한다 — 다르면
여기서 거부한다(실행기는 승인 없이 아무것도 움직이지 않는다). 상황판은 확인 입력 뒤 이 도구를 부른다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trigger import domain_refusal  # noqa: E402

from policy_control.episode_runner_node import APPROVALS, NAME, NS, write_approval  # noqa: E402

ACTIONS = ("status", "next", "run", "stop", "reset")


def expected(action: str, status: dict) -> str | None:
    """이 명령이 요구하는 승인 이름(없으면 None)."""
    if action == "next":
        return status.get("next_action")
    if action == "run":
        return f"episode:{status.get('episode')}"
    return None


def refusal(action: str, approve: str | None, status: dict) -> str | None:
    if action not in ("next", "run"):
        return None
    if status.get("busy"):
        return f"실행기가 돌고 있다({status.get('node')}) — 끝나거나 stop 뒤에"
    want = expected(action, status)
    if want is None:
        return f"할 노드가 없다(상태 {status.get('phase')})"
    if approve != want:
        return f"--approve {approve!r} ≠ 실행기가 물을 이름 {want!r}"
    return None


def read_status(timeout: float = 3.0) -> dict:
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node("episode_cmd_probe")
    got: dict = {}
    node.create_subscription(String, f"{NS}/status/{NAME}", lambda m: got.update(json.loads(m.data)), 10)
    try:
        t0 = time.monotonic()
        while not got and time.monotonic() - t0 < timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
        return got
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def call(action: str, timeout: float = 5.0) -> tuple[bool, list]:
    import rclpy
    from std_srvs.srv import Trigger
    rclpy.init()
    node = rclpy.create_node("episode_cmd")
    cli = node.create_client(Trigger, f"{NS}/{NAME}/{action}")
    try:
        if not cli.wait_for_service(timeout_sec=3.0):
            return False, [f"{NS}/{NAME}/{action}: 서비스 없음 — episode_runner_node 가 떠 있는가"]
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
        if not fut.done():
            return False, ["응답 없음"]
        res = fut.result()
        body = json.loads(res.message) if res.message.startswith("{") else {"ok": res.success, "reasons": [res.message]}
        return bool(body.get("ok")), list(body.get("reasons", []))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("action", choices=ACTIONS)
    ap.add_argument("--approve", default=None, help="next · run 의 승인 이름(status 의 next_action · episode:<이름>)")
    ap.add_argument("--operator", default=os.environ.get("USER", "operator"))
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--allow-domain-0", action="store_true")
    args = ap.parse_args(argv)
    why = domain_refusal(os.environ, args.allow_domain_0)
    if why:
        print(f"[episode] {why}")
        return 1
    status = read_status()
    if not status:
        print("[episode] episode_runner 상태가 안 온다 — 노드가 떠 있는가")
        return 1
    print(f"[episode] {status.get('episode')} · {status.get('phase')} · 노드 {status.get('node')} · "
          f"다음 승인 {status.get('next_action')} · busy {status.get('busy')}")
    if args.action == "status":
        print(json.dumps({k: status.get(k) for k in ("world", "last", "pending", "nodes")}, ensure_ascii=False, indent=1))
        return 0
    bad = refusal(args.action, args.approve, status)
    if bad:
        print(f"[episode] 거부: {bad}")
        return 1
    if not args.execute:
        print(f"[episode] 계획만 — {args.action}" + (f" (승인 {args.approve})" if args.approve else "") + " · --execute 로 실행")
        return 0
    if args.approve:
        write_approval(args.approve, args.operator, APPROVALS)
    ok, reasons = call(args.action)
    print(f"[episode] {args.action}: {'받음' if ok else '거부'} {reasons}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

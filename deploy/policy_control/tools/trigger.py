#!/usr/bin/env python3
"""Trigger 서비스 **하나**를 부르고 응답을 판정한다 — rc 0 = ok, 1 = 거부/실패.

`ros2 service call` 은 응답이 success=False 여도 rc 0 이라 스크립트·콘솔이 실패를 못 본다.
이 도구는 `episode_ctl.parse_trigger` 와 같은 규칙({"ok","reasons"} JSON)으로 읽는다.

    python3 deploy/policy_control/tools/trigger.py pd/engage                              # 계획만
    python3 deploy/policy_control/tools/trigger.py pd/goto_home --expect-pd TRACKING --execute
    python3 deploy/policy_control/tools/trigger.py episode/stop --execute

규약
  · `--execute` 없이는 아무 서비스도 부르지 않는다.
  · 부를 수 있는 서비스는 아래 SERVICES 의 7개뿐이다(임의 서비스 이름을 받지 않는다).
  · `--expect-pd` 를 주면 호출 뒤 /policy_control/status/pd 의 phase 가 그 값이 될 때까지 기다린다.
  · ROS_DOMAIN_ID 0/unset 은 거부한다 — `--allow-domain-0` 로만 통과한다(`palm_cmd.py` 와 같다).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_ctl import parse_trigger  # noqa: E402

NS = "/policy_control"
#: pd 는 팔마다 서비스가 따로다 — `pd/engage --side right` → `/policy_control/pd_right/engage` (09.23).
PD_SERVICES = ("pd/engage", "pd/goto_home", "pd/release", "pd/hand_home", "pd/hand_rest", "pd/hand_path")
SERVICES = PD_SERVICES + ("episode/reset", "episode/start", "episode/stop", "episode/abort")
#: 로봇을 **덜** 움직이게 하는 쪽 — 승인 없이 언제든 불러도 되는 것들.
DESCENDING = ("episode/stop", "episode/abort", "pd/release")


def domain_refusal(env: Mapping[str, str], allow_zero: bool) -> str | None:
    domain = str(env.get("ROS_DOMAIN_ID", "")).strip()
    if domain in ("", "0") and not allow_zero:
        return f"ROS_DOMAIN_ID={domain or 'unset'} 은 실기 기본 도메인이다 — --allow-domain-0 없이는 거부"
    return None


def read_pd(timeout: float, side: str) -> str:
    """`status/pd` 의 phase 한 줄. **구독만 한다** — 아무 서비스도 부르지 않고 아무것도 발행하지 않는다.

    09.23: 이미 잡고 있는 pd 에 engage 를 부르면 거부된다(`phase TRACKING is not IDLE`). 부르기 전에
    상태를 볼 수단이 없어서 도구들이 거부를 실패로 다뤘다.
    """
    import rclpy  # noqa: PLC0415
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node(f"policy_control_pd_probe_{side}")
    pd: dict = {}

    def on_pd(msg: String) -> None:
        try:
            pd.update(json.loads(msg.data))
        except ValueError:
            pass

    node.create_subscription(String, f"{NS}/status/pd_{side}", on_pd, 10)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and "phase" not in pd:
            rclpy.spin_once(node, timeout_sec=0.1)
        return str(pd.get("phase", ""))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def resolve(service: str, side: str) -> str:
    """사용자가 적는 이름 → 실제 서비스 경로. pd 는 쪽이 붙는다."""
    if service in PD_SERVICES:
        return f"{NS}/pd_{side}/{service.split('/', 1)[1]}"
    return f"{NS}/{service}"


def call(service: str, *, side: str, expect_pd: Sequence[str], service_timeout: float, phase_timeout: float) -> tuple[bool, list[str]]:
    import rclpy  # noqa: PLC0415
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node("policy_control_trigger")
    pd: dict = {}

    def on_pd(msg: String) -> None:
        try:
            pd.update(json.loads(msg.data))
        except ValueError:
            pass

    node.create_subscription(String, f"{NS}/status/pd_{side}", on_pd, 10)
    try:
        path = resolve(service, side)
        client = node.create_client(Trigger, path)
        if not client.wait_for_service(timeout_sec=service_timeout):
            return False, [f"service {path} unavailable ({service_timeout:.0f}s)"]
        future = client.call_async(Trigger.Request())
        # goto_home 같은 서비스는 램프가 끝나야 응답한다 — 응답 자체를 phase 타임아웃만큼 기다린다.
        rclpy.spin_until_future_complete(node, future, timeout_sec=max(service_timeout, phase_timeout))
        if not future.done() or future.result() is None:
            return False, [f"service {path} timeout"]
        resp = future.result()
        print(f"  ← {path}: success={resp.success} message={resp.message}", flush=True)
        ok, reasons = parse_trigger(resp.success, resp.message)
        if not ok or not expect_pd:
            return ok, reasons
        deadline = time.monotonic() + phase_timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if pd.get("phase") in expect_pd:
                print(f"  ✓ pd phase {pd.get('phase')}", flush=True)
                return True, reasons
        return False, [f"pd phase {pd.get('phase')!r} not in {list(expect_pd)} within {phase_timeout:.0f}s"]
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("service", nargs="?", choices=SERVICES,
                    help="--read-pd 만 쓸 때는 생략한다")
    ap.add_argument("--read-pd", action="store_true",
                    help="pd phase 한 줄을 찍고 끝낸다 — 구독 전용이라 --execute 가 필요 없다")
    ap.add_argument("--side", choices=("right", "left"), default="right",
                    help="pd 서비스·status 는 팔마다 따로다 — 어느 팔인가")
    ap.add_argument("--expect-pd", nargs="*", default=[], help="호출 뒤 기다릴 pd phase (여러 개면 그중 하나)")
    ap.add_argument("--execute", action="store_true", help="★실제로 서비스를 부른다")
    ap.add_argument("--allow-domain-0", action="store_true")
    ap.add_argument("--service-timeout", type=float, default=20.0)
    ap.add_argument("--phase-timeout", type=float, default=90.0)
    args = ap.parse_args(argv)

    if args.read_pd:
        refusal = domain_refusal(os.environ, args.allow_domain_0)
        if refusal:
            print(f"  ✗ {refusal}", file=sys.stderr)
            return 2
        print(read_pd(args.service_timeout, args.side))
        return 0
    if not args.service:
        ap.error("service 가 필요하다 (또는 --read-pd)")
    print(f"trigger {resolve(args.service, args.side)} · 기대 pd {args.expect_pd or '-'} · execute {args.execute}", flush=True)
    if not args.execute:
        print("DRY RUN — 아무 서비스도 부르지 않았다.")
        return 0
    refusal = domain_refusal(os.environ, args.allow_domain_0)
    if refusal:
        print(f"  ✗ {refusal}")
        return 2
    ok, reasons = call(args.service, side=args.side, expect_pd=args.expect_pd,
                       service_timeout=args.service_timeout, phase_timeout=args.phase_timeout)
    for r in reasons:
        print(f"  {'·' if ok else '✗'} {r}")
    print("  결과: " + ("ok" if ok else "거부/실패"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

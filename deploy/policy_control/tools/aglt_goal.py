#!/usr/bin/env python3
"""rh_aglt 정책에 목표 하나를 준다(base 좌표 m) — /policy_control/<side>/goal 발행 → goal_result 판정. rc 0 = 받음, 1 = 거부/무응답.

    python3 deploy/policy_control/tools/aglt_goal.py --side right --xyz 0.25 -0.12 0.41              # 계획만
    python3 deploy/policy_control/tools/aglt_goal.py --side right --handoff --execute                # 놓기 인계 자리로

--handoff = rh_place 학습 시작 자세(aglt cyl60n 이 컵을 쥐고 멈춘 고정 목표 (0.25, ∓0.12, +0.12) — z 는 선 컵 원점
0.290 + 0.12, 10.04 PLACE 세션). 정책 노드가 학습 목표 박스 밖이면 거부하고, 먼 목표는 0.08 m 중간 목표로 나눈다.
`--execute` 없이는 아무것도 발행하지 않는다. ROS_DOMAIN_ID 0/unset 은 거부한다(trigger.py 와 같다).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trigger import NS, domain_refusal  # noqa: E402

#: hdgp rh_place_r 시작 뱅크(bank_{r,l}_cyl60_keep)를 만든 aglt 고정 목표 — y 는 오른팔 −, 왼팔 +
HANDOFF = {"right": (0.25, -0.12, 0.41), "left": (0.25, 0.12, 0.41)}


def plan(side: str, xyz) -> dict:
    return {"topic": f"{NS}/{side}/goal", "result": f"{NS}/{side}/goal_result", "xyz": [float(v) for v in xyz]}


def send(p: dict, timeout: float) -> tuple[bool, dict]:
    import rclpy  # noqa: PLC0415
    from geometry_msgs.msg import Point
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("policy_control_aglt_goal")
    got: dict = {}
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, p["result"], lambda m: got.update(json.loads(m.data)), latched)
    pub = node.create_publisher(Point, p["topic"], QoSProfile(depth=10))
    try:
        deadline = time.monotonic() + timeout
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        got.clear()                                   # 예전 결과(latched)는 버린다 — 이번 요청의 답만
        x, y, z = p["xyz"]
        pub.publish(Point(x=x, y=y, z=z))
        while time.monotonic() < deadline and [round(v, 4) for v in got.get("request", [])] != [round(v, 4) for v in p["xyz"]]:
            rclpy.spin_once(node, timeout_sec=0.05)
        return bool(got.get("ok")), got
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", required=True, choices=sorted(HANDOFF))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    g.add_argument("--handoff", action="store_true", help="rh_place 인계 자리(학습 시작 뱅크의 aglt 목표)")
    ap.add_argument("--execute", action="store_true", help="★실제로 발행한다 — 팔이 움직인다")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--allow-domain-0", action="store_true")
    args = ap.parse_args(argv)
    p = plan(args.side, HANDOFF[args.side] if args.handoff else args.xyz)
    print(f"[aglt_goal] {p['topic']} ← {p['xyz']}")
    if not args.execute:
        print("[aglt_goal] 계획만 — --execute 로 발행")
        return 0
    why = domain_refusal(os.environ, args.allow_domain_0)
    if why:
        print(f"[aglt_goal] {why}")
        return 1
    ok, body = send(p, args.timeout)
    print(f"[aglt_goal] {'받음' if ok else '거부/무응답'} {json.dumps(body, ensure_ascii=False)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

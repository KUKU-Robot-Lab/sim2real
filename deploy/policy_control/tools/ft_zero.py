#!/usr/bin/env python3
"""DG-5F 손끝 F/T 센서 영점 — 드라이버 서비스 `set_ft_sensor_offset`(std_srvs/Trigger)을 부른다. 로봇은 움직이지 않는다.

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/ft_zero.py --side left

09.28 사용자: "ft 센서 키는것 까진 좋은데 처음에 bias 잡아야해. 0으로 세팅해야함 home 자세로 갔을때 기준으로".
정책 단계가 팔을 정책 시작 자세(home)에 세우고 손을 초기 자세로 둔 뒤, 아무것도 닿지 않은 때 부른다 — 반복할 때마다
다시 잡는다. 서비스는 드라이버가 `fingertip_sensor:=true` 로 떴을 때만 생긴다(`delto_hardware_interface_node`).
이름은 네임스페이스가 붙을 수 있어 서비스 목록에서 `set_ft_sensor_offset` 로 끝나는 것 중 이 손의 것을 고른다.
--if-present: 서비스가 없으면(센서를 켜지 않은 드라이버) 경고만 하고 0 으로 끝난다 — 정책을 막지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SUFFIX = "set_ft_sensor_offset"


def pick_service(names: list[str], side: str) -> tuple[str | None, str]:
    """(서비스 이름 | None, 이유). 이 손의 네임스페이스(/dg5f_<side>/)가 붙은 것을 먼저, 없으면 하나뿐일 때만."""
    cands = sorted(n for n in names if n.endswith("/" + SUFFIX))
    mine = [n for n in cands if n.startswith(f"/dg5f_{side}/")]
    if len(mine) == 1:
        return mine[0], "namespace match"
    if len(mine) > 1:
        return None, f"이 손의 영점 서비스가 여러 개다 {mine}"
    if len(cands) == 1:
        return cands[0], "only one offset service"
    if not cands:
        return None, "영점 서비스가 없다 — 손 드라이버가 fingertip_sensor:=true 로 떠 있는가"
    return None, f"어느 손의 서비스인지 모른다 {cands}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--wait-s", type=float, default=5.0, help="서비스를 찾는 시간")
    ap.add_argument("--settle-s", type=float, default=0.5, help="영점 뒤 기다리는 시간(펌웨어가 오프셋을 잡는 동안)")
    ap.add_argument("--if-present", action="store_true", help="서비스가 없으면 경고만 하고 성공으로 끝낸다")
    args = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        raise SystemExit("✗ ROS_DOMAIN_ID 가 비었거나 0 — 거부")

    import rclpy
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node(f"ft_zero_{args.side}")
    try:
        name, why = None, ""
        t0 = time.time()
        while time.time() - t0 < args.wait_s:
            names = [n for n, _ in node.get_service_names_and_types()]
            name, why = pick_service(names, args.side)
            if name is not None:
                break
            rclpy.spin_once(node, timeout_sec=0.2)
        if name is None:
            print(f"{'⚠' if args.if_present else '✗'} [ft_zero] {args.side}: {why}", flush=True)
            return 0 if args.if_present else 1
        cli = node.create_client(Trigger, name)
        if not cli.wait_for_service(timeout_sec=args.wait_s):
            print(f"✗ [ft_zero] {name} 가 응답하지 않는다", flush=True)
            return 1
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=args.wait_s)
        res = fut.result()
        if res is None or not res.success:
            print(f"✗ [ft_zero] {name}: {None if res is None else res.message}", flush=True)
            return 1
        time.sleep(args.settle_s)
        print(json.dumps({"ok": True, "service": name, "message": res.message}, ensure_ascii=False), flush=True)
        print(f"✓ [ft_zero] {args.side} 손끝 F/T 영점 — 지금 자세(정책 시작 · 무접촉)가 0", flush=True)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())

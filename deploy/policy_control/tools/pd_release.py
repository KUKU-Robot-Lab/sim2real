#!/usr/bin/env python3
"""pd 노드 밖의 복구 CLI — forward 컨트롤러에 0 을 보내고 JTC 로 되돌린다.

pd 노드가 죽으면 하드웨어는 마지막 (q*, q̇*, τ_ff) 를 **영구 유지**한다(워치독 없음). 이
스크립트는 (1) velocity/effort 에 0 을 5회 송출하고 (2) switch_controller 로 JTC 를 다시
활성화한다. position 은 건드리지 않는다 — JTC 가 현재 위치를 이어받아 홀딩한다.

    python3 deploy/policy_control/tools/pd_release.py --side left --execute
    python3 deploy/policy_control/tools/pd_release.py --side left --execute --ensure

--ensure(★11.09 tuning [pd 끄기]): list_controllers 로 먼저 보고 JTC active · forward 3종 inactive 면 아무것도
보내지 않는다. 아니면 위 복구를 한 뒤 다시 보고, 여전히 아니면 종료 코드 4.
"""
from __future__ import annotations

import argparse
import time

ZERO_TICKS = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", required=True, help="left | right | both | left,right (팔마다 차례로)")
    ap.add_argument("--execute", action="store_true", help="★없으면 계획만 출력한다")
    ap.add_argument("--ensure", action="store_true", help="이미 JTC 면 그대로, 아니면 복구 뒤 확인")
    args = ap.parse_args()
    sides = parse_sides(args.side)
    plans = {s: ([f"{s}_forward_{k}_controller" for k in ("position", "velocity", "effort")],
                 f"{s}_joint_trajectory_controller") for s in sides}
    for s, (fwd, jtc) in plans.items():
        print(f"[pd_release] {s}: 0 ×{ZERO_TICKS} → /{fwd[1]}/commands, /{fwd[2]}/commands · switch → {jtc}")
    if not args.execute:
        print("DRY RUN — 실제로 보내려면 --execute")
        return 0

    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("pd_release")
    rc = 0
    for s, (fwd, jtc) in plans.items():
        rc = max(rc, ensure_one(node, fwd, jtc) if args.ensure else release_one(node, fwd, jtc))
    node.destroy_node()
    rclpy.shutdown()
    return rc


def parse_sides(text: str) -> list[str]:
    t = text.strip().lower()
    sides = ["right", "left"] if t in ("both", "all") else [s.strip() for s in t.split(",") if s.strip()]
    bad = [s for s in sides if s not in ("left", "right")]
    if bad or not sides or len(set(sides)) != len(sides):
        raise SystemExit(f"--side 는 left | right | both | left,right — 받은 값 {text!r}")
    return sides


NO_RESPONSE = "list_controllers 응답 없음(서비스 없음 · 시간 초과) — 상태를 모른다"


def jtc_problem(states: dict | None, fwd: list[str], jtc: str) -> str:
    """'' = JTC active · forward 전부 inactive. None = 서비스 응답 없음. 아니면 무엇이 어긋났는지 한 줄."""
    if states is None:
        return NO_RESPONSE
    bad = [f"{n}={states.get(n, '없음')}" for n in fwd if states.get(n) == "active"]
    if states.get(jtc) != "active":
        bad.insert(0, f"{jtc}={states.get(jtc, '없음')}")
    return ", ".join(bad)


def list_states(node) -> dict | None:
    """{이름: 상태}. 서비스가 없거나 응답이 오지 않으면 None(빈 목록 {} 과 구분)."""
    import rclpy
    from controller_manager_msgs.srv import ListControllers

    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    if not client.wait_for_service(3.0):
        return None
    fut = client.call_async(ListControllers.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    return {c.name: c.state for c in fut.result().controller} if fut.result() is not None else None


def ensure_one(node, fwd: list[str], jtc: str) -> int:
    problem = jtc_problem(list_states(node), fwd, jtc)
    if not problem:
        print(f"[pd_release] {jtc} active · forward inactive — 보낸 것 없음")
        return 0
    if problem == NO_RESPONSE:        # 어긋남이 아니라 모름 — 확인 못 한 채 switch 하지 않는다
        print(f"[pd_release] {problem}")
        return 5
    print(f"[pd_release] 어긋남: {problem} — 0 송출 · switch")
    rc = release_one(node, fwd, jtc)
    problem = jtc_problem(list_states(node), fwd, jtc)
    if problem == NO_RESPONSE:
        print(f"[pd_release] 복구 뒤 확인 못 함: {problem}")
        return 5
    if problem:
        print(f"[pd_release] 복구 뒤에도 어긋남: {problem}")
        return max(rc, 4)
    print(f"[pd_release] 복구 확인: {jtc} active · forward inactive")
    return 0


def release_one(node, fwd: list[str], jtc: str) -> int:
    import rclpy
    from controller_manager_msgs.srv import SwitchController
    from std_msgs.msg import Float64MultiArray

    pubs = [node.create_publisher(Float64MultiArray, f"/{n}/commands", 10) for n in fwd[1:]]
    time.sleep(0.3)
    for _ in range(ZERO_TICKS):
        for p in pubs:
            p.publish(Float64MultiArray(data=[0.0] * 7))
        rclpy.spin_once(node, timeout_sec=0.02)
    client = node.create_client(SwitchController, "/controller_manager/switch_controller")
    if not client.wait_for_service(3.0):
        print("[pd_release] controller_manager 없음 — 0 송출만 했다")
        return 2
    req = SwitchController.Request()
    req.activate_controllers, req.deactivate_controllers = [jtc], fwd
    req.strictness, req.activate_asap = req.BEST_EFFORT, True
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    ok = bool(fut.result() and fut.result().ok)
    print(f"[pd_release] switch → {jtc}: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

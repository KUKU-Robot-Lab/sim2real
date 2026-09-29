#!/usr/bin/env python3
"""RH56F1 한 축 확인 — 한 손가락(축)만 작게 움직여 **슬롯 대응 · 방향**을 눈과 실측으로 확인한다. 실기 명령.

09.29 사용자: rh56f1 제어 연결. 변환표(config/rh56f1_hand_map.yaml)의 엄지 두 축은 행정 길이로 맞춘 추정이라
verified: false 이고, pd 는 그 축에 -1(움직이지 않음)을 보낸다. 이 도구로 한 축씩:
  1) 지금 레지스터를 읽는다(angle_actual)
  2) 그 축만 `--to` rad 에 해당하는 레지스터로 보낸다(다른 다섯 축은 -1 — 드라이버가 그대로 둔다)
  3) --hold 초 뒤 다시 읽어 그 슬롯만 움직였는가 · 레지스터가 어느 쪽으로 갔는가를 보고
  4) --back 이면 원래 레지스터로 되돌린다
사람이 **어느 손가락이 어느 쪽(굽힘/펴짐 · 벌림)으로** 움직였는지 보고, 맞으면 변환표 verified 를 true 로 바꾼다.

  python3 tools/rh56f1_axis_probe.py --side right --axis index_1 --to 0.3 --back            # 계획만(발행 없음)
  python3 tools/rh56f1_axis_probe.py --side right --axis index_1 --to 0.3 --back --execute  # ★실기 — 승인 뒤
  python3 tools/rh56f1_axis_probe.py --side right --axis thumb_1 --by -0.2 --back --execute  # 지금에서 −0.2 rad
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import rh56f1_map  # noqa: E402

#: 한 번에 움직일 수 있는 최대 [rad] — 확인용이다. 크게 움직이면 옆 손가락 · 손바닥 · 물건에 닿는다
MAX_STEP_RAD = 0.5
READ_TIMEOUT_S = 3.0


def plan(hmap: rh56f1_map.HandMap, axis: str, to_rad: float, now_reg: list[int]) -> tuple[list[int], int, str]:
    """(보낼 SetAngle1 값, 목표 레지스터, 설명). 순수 — 한 축만, 나머지 -1. 너무 크면 ValueError."""
    if axis not in hmap.joint_order:
        raise ValueError(f"축 {axis!r} — {list(hmap.joint_order)} 중 하나")
    a = hmap.axes[hmap.joint_order.index(axis)]
    now_rad = a.to_rad(now_reg[a.slot])
    if abs(to_rad - now_rad) > MAX_STEP_RAD:
        raise ValueError(f"{axis}: 지금 {now_rad:.3f} → {to_rad:.3f} rad 는 {abs(to_rad - now_rad):.3f} rad — "
                         f"한 번에 {MAX_STEP_RAD} rad 까지")
    target = a.to_reg(to_rad)
    cmd = [rh56f1_map.LEAVE] * 6
    cmd[a.slot] = target
    how = (f"{axis}(슬롯 {a.slot}) 레지스터 {now_reg[a.slot]} → {target}  "
           f"(변환표상 {now_rad:.3f} → {to_rad:.3f} rad, {'확인됨' if a.verified else '★방향 미확인'})")
    return cmd, target, how


def moved(before: list[int], after: list[int], slot: int, tol: int = 5) -> tuple[bool, list[int]]:
    """(그 슬롯이 움직였나, 함께 움직인 다른 슬롯들). 순수."""
    others = [i for i in range(6) if i != slot and abs(after[i] - before[i]) > tol]
    return abs(after[slot] - before[slot]) > tol, others


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--axis", required=True, help="thumb_1 · thumb_2 · index_1 · middle_1 · ring_1 · pinky_1")
    to = ap.add_mutually_exclusive_group(required=True)
    to.add_argument("--to", type=float, help="목표 [rad] (자산 기준, 0 = 편 손)")
    to.add_argument("--by", type=float, help="지금에서 이만큼 [rad] — 지금 값을 모르는 축(엄지 회전)에")
    ap.add_argument("--hold", type=float, default=2.0, help="보낸 뒤 기다리는 시간 [s]")
    ap.add_argument("--back", action="store_true", help="끝나면 원래 레지스터로 되돌린다")
    ap.add_argument("--execute", action="store_true", help="★실기에 보낸다(없으면 계획만)")
    ap.add_argument("--map", default=str(rh56f1_map.DEFAULT_PATH))
    args = ap.parse_args(argv)
    hmap = rh56f1_map.load(args.map)

    import rclpy
    from rclpy.node import Node
    from rh56f1_interfaces.msg import GetAngleAct1, SetAngle1

    rclpy.init()
    node = Node(f"rh56f1_axis_probe_{args.side}")
    ns = f"/hand_{args.side}"
    box: dict = {"reg": None}
    node.create_subscription(GetAngleAct1, f"{ns}/angle_actual", lambda m: box.update(reg=list(m.joint_values)), 10)
    pub = node.create_publisher(SetAngle1, f"{ns}/angle_set", 10) if args.execute else None

    def read() -> list[int] | None:
        box["reg"] = None
        t_end = time.monotonic() + READ_TIMEOUT_S
        while box["reg"] is None and time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        return box["reg"]

    def matched() -> bool:
        """드라이버가 angle_set 을 구독하는가 — 연결 전에 한 번 보내면 조용히 사라진다(09.29 fake: thumb_1 이 안 움직였다)."""
        t_end = time.monotonic() + READ_TIMEOUT_S
        while pub.get_subscription_count() < 1 and time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        return pub.get_subscription_count() >= 1

    def send(values: list[int]) -> None:
        if pub is None:
            return
        msg = SetAngle1()
        msg.hand_id = 0
        msg.joint_values = [int(v) for v in values]
        pub.publish(msg)

    def wait(sec: float) -> None:
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)

    rc = 0
    try:
        before = read()
        if before is None:
            print(f"✗ {ns}/angle_actual 이 {READ_TIMEOUT_S} s 안에 안 온다 — 드라이버를 먼저")
            return 1
        a = hmap.axes[hmap.joint_order.index(args.axis)] if args.axis in hmap.joint_order else None
        goal = args.to if args.by is None or a is None else a.to_rad(before[a.slot]) + args.by
        cmd, target, how = plan(hmap, args.axis, goal, before)
        print(f"[{args.side}] 지금 {before}\n  계획: {how}\n  보낼 값 {cmd}")
        if not args.execute:
            print("  (계획만 — --execute 없이는 보내지 않는다)")
            return 0
        if not matched():
            print(f"✗ {ns}/angle_set 을 구독하는 드라이버가 {READ_TIMEOUT_S} s 안에 안 보인다 — 보내지 않았다")
            return 1
        send(cmd)
        wait(args.hold)
        after = read() or before
        slot = hmap.axes[hmap.joint_order.index(args.axis)].slot
        ok, others = moved(before, after, slot)
        print(f"  뒤 {after}")
        print(f"  {'✓' if ok else '✗'} 슬롯 {slot} {before[slot]} → {after[slot]} (목표 {target})"
              + (f" · ★다른 슬롯도 움직였다 {others}" if others else ""))
        print("  ★눈으로: 어느 손가락이 어느 쪽으로 움직였나 — 변환표와 맞으면 rh56f1_hand_map.yaml 의 verified 를 true 로")
        rc = 0 if ok and not others else 1
        if args.back:
            back = [rh56f1_map.LEAVE] * 6
            back[slot] = before[slot]
            send(back)
            wait(args.hold)
            print(f"  되돌림 → {read()}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

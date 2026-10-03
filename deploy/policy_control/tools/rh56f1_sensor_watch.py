#!/usr/bin/env python3
"""RH56F1 손 센서 실시간 보기 — 손끝 촉각 · 손바닥 · 관절 힘 · 전류 · 온도 · 상태가 바뀌면 그 칸을 찍는다. 구독만(명령 없음).

10.03 사용자: 손에 들어오는 센서 데이터 확인. 사람이 손끝 · 손바닥을 하나씩 눌러 어느 칸이 바뀌는지로 손가락 대응 ·
단위를 확인한다(촉각 1024 = 10.24 N, 중문 매뉴얼 V1.0.0 2.5.22).

    python3 deploy/policy_control/tools/rh56f1_sensor_watch.py --side right            # 바뀔 때마다 한 줄
    python3 deploy/policy_control/tools/rh56f1_sensor_watch.py --side right --seconds 30 --summary   # 끝에 최대값 표
"""
from __future__ import annotations

import argparse
import json
import time

FINGERS = ("새끼", "약지", "중지", "검지", "엄지")            # 벤더 촉각 순(새끼부터)
SLOTS = ("새끼", "약지", "중지", "검지", "엄지굽힘", "엄지회전")  # 관절 슬롯 순
TOUCH_N_PER_RAW = 0.01                                        # 1024 = 10.24 N
NO_CONTACT = 65535


def touch_changes(prev: dict | None, now: dict, thr: int) -> list[str]:
    """손끝 법선 · 접선 · 손바닥 중 thr 넘게 바뀐 칸. 순수."""
    out = []
    for k, label in (("finger_forces", "법선"), ("finger_tangentials", "접선")):
        for i, v in enumerate(now[k]):
            p = prev[k][i] if prev else 0
            if abs(v - p) >= thr:
                out.append(f"{FINGERS[i]} {label} {v} ({v * TOUCH_N_PER_RAW:.2f} N)")
    for i, v in enumerate(now["palm_data"]):
        if i % 3 == 2:                                        # 3 영역 × (법선 · 접선 · 방향) — 방향은 따로 보지 않는다
            continue
        p = prev["palm_data"][i] if prev else 0
        if v >= 0 and abs(v - p) >= thr:
            out.append(f"손바닥{i // 3 + 1} {'법선' if i % 3 == 0 else '접선'} {v}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = Ctrl-C 까지")
    ap.add_argument("--thr", type=int, default=20, help="촉각 변화 문턱(원시값, 20 = 0.2 N)")
    ap.add_argument("--summary", action="store_true", help="끝날 때 칸별 최대값")
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from rh56f1_interfaces.msg import GetCurrentAct1, GetForceAct1, TouchData1

    rclpy.init()
    node = Node(f"rh56f1_sensor_watch_{args.side}")
    ns = f"/hand_{args.side}"
    st = {"touch": None, "force": None, "cur": None, "peak": {}, "status": None}

    def peak(key, v):
        st["peak"][key] = max(st["peak"].get(key, v), v)

    def on_touch(m):
        now = {k: list(getattr(m, k)) for k in ("finger_forces", "finger_tangentials", "finger_angles",
                                                 "finger_proximity", "palm_data")}
        ch = touch_changes(st["touch"], now, args.thr)
        if ch:
            dirs = " ".join(f"{FINGERS[i]}{'-' if a == NO_CONTACT else a}°" for i, a in enumerate(now["finger_angles"]))
            print(f"[촉각] {' · '.join(ch)}   | 방향 {dirs}", flush=True)
            st["touch"] = now
        elif st["touch"] is None:
            st["touch"] = now
        for i in range(5):
            peak(f"{FINGERS[i]} 법선", now["finger_forces"][i])
            peak(f"{FINGERS[i]} 접선", now["finger_tangentials"][i])

    def on_six(kind, m):
        v = list(m.joint_values)
        prev = st[kind]
        thr = 30 if kind == "force" else 5
        if prev is None or any(abs(a - b) >= thr for a, b in zip(v, prev)):
            label = "관절 힘" if kind == "force" else "전류"
            print(f"[{label}] " + " · ".join(f"{s} {x}" for s, x in zip(SLOTS, v)), flush=True)
            st[kind] = v
        for s, x in zip(SLOTS, v):
            peak(f"{'힘' if kind == 'force' else '전류'} {s}", x)

    def on_status(m):
        d = json.loads(m.data)
        key = (tuple(d.get("error", [])), tuple(d.get("status", [])))
        if key != st["status"]:
            print(f"[상태] 오류 {d.get('error')} · 상태 {d.get('status_text')} · 온도 {d.get('temperature')} °C", flush=True)
            st["status"] = key

    node.create_subscription(TouchData1, f"{ns}/touch_data", on_touch, 10)
    node.create_subscription(GetForceAct1, f"{ns}/force_actual", lambda m: on_six("force", m), 10)
    node.create_subscription(GetCurrentAct1, f"{ns}/current_actual", lambda m: on_six("cur", m), 10)
    node.create_subscription(String, f"{ns}/ecat_status", on_status, 10)
    print(f"{ns} 센서 보기 — 손끝 · 손바닥을 하나씩 눌러 보세요(변화 문턱 {args.thr} = {args.thr * TOUCH_N_PER_RAW:.2f} N)", flush=True)
    t_end = time.monotonic() + args.seconds if args.seconds > 0 else None
    try:
        while rclpy.ok() and (t_end is None or time.monotonic() < t_end):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if args.summary:
            print("최대값:", " · ".join(f"{k} {v}" for k, v in sorted(st["peak"].items())), flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

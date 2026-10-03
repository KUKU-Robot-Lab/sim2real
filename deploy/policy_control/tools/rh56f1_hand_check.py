#!/usr/bin/env python3
"""RH56F1 손 점검 — **읽기만**. 드라이버 · 상태 노드가 살아 있고 값이 말이 되는가.

09.29 사용자: rh56f1 제어 연결(arm4090). 명령을 내지 않는다 — 콘솔 미션 hand_check_<side> 단계가 부른다.

  /hand_<side>/angle_actual  (벤더 레지스터 0.1°)  주기 · 값 · 슬롯 이름
  /hand_<side>/joint_states  (상태 노드, rad)       주기 · 한계 안인가 · 레지스터 → rad 가 변환표와 맞는가
  /hand_<side>/touch_data    (촉각)                  주기(없어도 실패는 아니다 — 촉각 없는 손)

    python3 tools/rh56f1_hand_check.py --side right [--seconds 3]
rc 0 = 통과, 1 = 끊김 · 한계 밖 · 변환 불일치.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import rh56f1_map  # noqa: E402

MIN_HZ = 20.0            # RS485 벤더 기본 50 Hz — 응답이 없으면 ~13 Hz 로 떨어진다(robot_control README)
PORTS = Path(__file__).resolve().parents[1] / "config" / "rh56f1_ports.yaml"


def min_hz_for(ports: dict, side: str) -> float:
    """기대 최소 주기 — EtherCAT 이면 설정 state_hz 의 80 %(10.03 200 Hz → 160), RS485 는 MIN_HZ."""
    if (ports.get(side) or {}).get("transport") == "ethercat":
        return 0.8 * float((ports.get("ethercat") or {}).get("state_hz", 100))
    return MIN_HZ
LIMIT_TOL = 0.02         # rad


def verdict(hmap: rh56f1_map.HandMap, side: str, reg: list[int] | None, names: list[str] | None,
            js: dict[str, float] | None, hz: dict[str, float], lower: dict, upper: dict,
            min_hz: float = MIN_HZ) -> list[str]:
    """문제 목록 — 비면 통과. 순수."""
    bad = []
    if reg is None:
        return [f"/hand_{side}/angle_actual 이 안 온다 — 드라이버(rh56f1_driver.py — EtherCAT 노드 · 마스터)가 떠 있는가 · 손 전원 · 랜 케이블"]
    if hz.get("angle", 0.0) < min_hz:
        bad.append(f"angle_actual {hz.get('angle', 0.0):.1f} Hz < {min_hz:.0f} — 손이 응답하지 않는 틱이 있다(/hand_{side}/ecat_status 의 WKC · OP 확인)")
    want = [""] * 6
    for a in hmap.axes:
        want[a.slot] = f"{side[0]}_hj_{a.name}"
    if names and [n for n in names if n] and list(names) != want:
        bad.append(f"드라이버 슬롯 이름 {list(names)} ≠ 변환표 {want}")
    if js is None:
        bad.append(f"/hand_{side}/joint_states 가 안 온다 — rh56f1_state_node 가 떠 있는가")
        return bad
    q = hmap.to_rad(reg, side)
    for n, v in zip(hmap.names(side), q):
        got = js.get(n)
        if got is None:
            bad.append(f"joint_states 에 {n} 이 없다")
        elif abs(got - v) > 0.02:
            bad.append(f"{n}: joint_states {got:.3f} ≠ 변환표 {v:.3f} rad(레지스터 {reg})")
        if got is not None and not (lower[n] - LIMIT_TOL <= got <= upper[n] + LIMIT_TOL):
            bad.append(f"{n} = {got:.3f} rad 가 한계 [{lower[n]:.3f}, {upper[n]:.3f}] 밖")
    return bad


def rate_hz(stamps: list) -> float:
    """받은 시각들의 평균 주기 → Hz(첫 · 끝 메시지 사이). 두 개 미만이면 0."""
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        return 0.0
    return (len(stamps) - 1) / (stamps[-1] - stamps[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--map", default=str(rh56f1_map.DEFAULT_PATH))
    args = ap.parse_args(argv)
    hmap = rh56f1_map.load(args.map)
    axes = hmap.axes_of(args.side)                                      # 이 손의 보정 범위(09.30 스윕)
    lower = {n: min(a.rad) for n, a in zip(hmap.names(args.side), axes)}
    upper = {n: max(a.rad) for n, a in zip(hmap.names(args.side), axes)}

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from rh56f1_interfaces.msg import GetAngleAct1, TouchData1

    rclpy.init()
    node = Node(f"rh56f1_hand_check_{args.side}")
    ns = f"/hand_{args.side}"
    seen = {"angle": [], "js": [], "touch": []}
    last: dict = {"reg": None, "names": None, "js": None}

    def on_angle(m):
        seen["angle"].append(time.monotonic())
        last["reg"], last["names"] = list(m.joint_values), list(m.joint_names)

    def on_js(m):
        seen["js"].append(time.monotonic())
        last["js"] = dict(zip(m.name, m.position))

    node.create_subscription(GetAngleAct1, f"{ns}/angle_actual", on_angle, 10)
    node.create_subscription(JointState, f"{ns}/joint_states", on_js, 10)
    node.create_subscription(TouchData1, f"{ns}/touch_data", lambda _m: seen["touch"].append(time.monotonic()), 10)
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()

    # 10.01 fake e2e: 개수 / 창 길이는 발견(discovery) 지연을 주기로 쳐 50 Hz 손을 9.3 Hz 로 읽었다 — 받은 메시지 사이 간격으로 잰다
    hz = {k: rate_hz(v) for k, v in seen.items()}
    print(f"[{args.side}] angle_actual {hz['angle']:.1f} Hz · joint_states {hz['js']:.1f} Hz · touch_data {hz['touch']:.1f} Hz")
    if last["reg"] is not None:
        print(f"  레지스터(슬롯 순 새끼 · 약지 · 중지 · 검지 · 엄지 굽힘 · 엄지 회전) {last['reg']}")
        print("  rad(자산 순) " + " · ".join(f"{n.split('_hj_')[1]} {v:+.3f}"
                                            for n, v in zip(hmap.names(args.side), hmap.to_rad(last["reg"], args.side))))
    if hmap.unverified(args.side):
        print(f"  ★방향 확인 전 축(pd 가 -1 로 둔다): {', '.join(hmap.unverified(args.side))} — tools/rh56f1_axis_probe.py")
    import yaml
    want_hz = min_hz_for(yaml.safe_load(PORTS.read_text()) or {}, args.side) if PORTS.is_file() else MIN_HZ
    bad = verdict(hmap, args.side, last["reg"], last["names"], last["js"], hz, lower, upper, want_hz)
    for b in bad:
        print(f"  ✗ {b}")
    print("  ✓ 통과" if not bad else f"  ✗ 문제 {len(bad)} 개")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

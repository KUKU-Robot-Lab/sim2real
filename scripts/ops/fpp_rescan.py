#!/usr/bin/env python3
"""컵 좌표를 지금 카메라로 다시 찍는다 — FP++ 묶음 컨테이너(fpp_cups)를 끄지 않고(10.08 사용자).

    python3 scripts/ops/fpp_rescan.py cyl60 cyl60_blue cyl60_pink [--wait 120]

미션 cups 단계가 FP++ 를 켠 뒤 부른다. 처음 실행이면 컨테이너가 켜지며 찍는 한 번을 기다리고(다시 명령하지 않는다),
단계를 다시 실행하면(상황판 '↶ 여기서 다시') /perception_plus_plus/snapshot/cmd 로 다시 찍게 하고 그 회차가 끝나길
기다린다. 다시 찍는 동안 컨테이너는 그 컵의 옛 좌표를 내지 않는다 — 옛 값이 정책 입력으로 새지 않는다.
끝나면 컵마다 base 좌표(/objects/<컵>/pose)와 판정을 찍는다. 못 찾은 컵이 있으면 rc 1.
로봇은 움직이지 않는다 — 카메라 영상만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

STATUS_TOPIC = "/perception_plus_plus/snapshot/status"
CMD_TOPIC = "/perception_plus_plus/snapshot/cmd"
TABLE_TOP_Z = 0.205
CUP_ORIGIN_Z = 0.085            # cyl60 원점 = 원통 중심(바닥 위)
Z_TOL = 0.008
TILT_MAX_DEG = 3.0
X_RANGE = (0.1, 0.4)


def first_action(status: dict | None) -> str:
    """처음 할 일 — 상태가 아직 없으면 기다리고, 첫 회차(generation 0)가 안 끝났으면 그것을 기다리고, 아니면 다시 찍게 한다."""
    if status is None:
        return "wait_status"
    if int(status.get("generation", 0)) == 0:
        return "wait_round"
    return "command"


def round_done(status: dict, after: int, names: list[str]) -> bool:
    """after 보다 새 회차(찍기 한 바퀴)가 끝났고 물어본 컵을 다 찾았는가. 못 찾은 컵은 노드가 retry_s 마다 다시 찾는다."""
    objs = status.get("objects", {})
    return int(status.get("generation", 0)) > after and all(objs.get(n, {}).get("found") for n in names)


def missing(status: dict, names: list[str]) -> list[str]:
    objs = status.get("objects", {})
    return [f"{n}: {objs.get(n, {}).get('why', '상태 없음')}" for n in names if not objs.get(n, {}).get("found")]


def check_cup(name: str, xyz: tuple[float, float, float], tilt_deg: float, origin_above_bottom: float = CUP_ORIGIN_Z) -> list[str]:
    """테이블 위에 서 있는 물체 기준의 경고들(빈 목록 = 통과). 원점 높이는 물체마다(레지스트리, 쉐이커 0.065)."""
    x, _, z = xyz
    warn = []
    z_want = TABLE_TOP_Z + origin_above_bottom
    if abs(z - z_want) > Z_TOL:
        warn.append(f"z {z:.3f} — 기대 {z_want:.3f} ± {Z_TOL * 1e3:.0f} mm")
    if tilt_deg > TILT_MAX_DEG:
        warn.append(f"기울기 {tilt_deg:.1f}° > {TILT_MAX_DEG:.0f}°")
    if not X_RANGE[0] <= x <= X_RANGE[1]:
        warn.append(f"x {x:.3f} — {X_RANGE[0]}~{X_RANGE[1]} 밖")
    return warn


def _origins(names: list[str]) -> dict[str, float]:
    """물체마다 원점 높이(바닥 위) — 레지스트리를 못 읽으면 비워 둔다(cyl60 값으로 판정)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from object_registry import load_registry
        reg = load_registry()
        return {n: float(reg.get(n).origin_above_bottom_m) for n in names}
    except (OSError, ValueError, KeyError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("cups", nargs="+")
    ap.add_argument("--wait", type=float, default=120.0, help="전체 대기 상한(s)")
    args = ap.parse_args(argv)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("fpp_rescan")
    box: dict = {"status": None, "poses": {}}
    node.create_subscription(String, STATUS_TOPIC, lambda m: box.update(status=json.loads(m.data)), 10)
    pub = node.create_publisher(String, CMD_TOPIC, 10)
    t_end = time.monotonic() + args.wait

    def spin_until(pred) -> bool:
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if pred():
                return True
        return False

    try:
        spin_until(lambda: box["status"] is not None)
        action = first_action(box["status"])
        if action == "wait_status":
            print(f"[rescan] {STATUS_TOPIC} 가 {args.wait:.0f} s 안에 안 온다 — 묶음 FP++ 컨테이너(fpp_<묶음>)가 떠 있는가 · docker logs 로 확인", file=sys.stderr)
            return 1
        after = int(box["status"].get("generation", 0))
        if action == "command":
            spin_until(lambda: pub.get_subscription_count() > 0)
            pub.publish(String(data=",".join(args.cups)))
            print(f"[rescan] 다시 찍기 요청 — {', '.join(args.cups)} (지난 회차 {after})")
        else:
            print("[rescan] FP++ 가 켜지며 찍는 첫 회차를 기다린다")
        if not spin_until(lambda: round_done(box["status"], after, args.cups)):
            st = box["status"] or {}
            print(f"[rescan] {args.wait:.0f} s 안에 다 못 찾았다 — {' · '.join(missing(st, args.cups)) or st.get('pending')} · "
                  f"오류 {st.get('error') or '-'}", file=sys.stderr)
            return 1
        st = box["status"]
        lost = missing(st, args.cups)          # 끝났으면 빈 목록 — 시간 초과 쪽은 위에서 끝난다
        t_round = time.time()

        def on_pose(name):
            def f(m):
                box["poses"][name] = m
            return f
        for n in args.cups:
            node.create_subscription(PoseStamped, f"/objects/{n}/pose", on_pose(n), 10)
        found = [n for n in args.cups if n not in {x.split(":")[0] for x in lost}]
        origin = _origins(args.cups)
        spin_until(lambda: all(n in box["poses"] for n in found))
        print(f"[rescan] 회차 {st.get('generation')} · {time.strftime('%H:%M:%S', time.localtime(t_round))}")
        for n in args.cups:
            info = st.get("objects", {}).get(n, {})
            m = box["poses"].get(n)
            if m is None:
                print(f"  {n:11s} 없음 — {info.get('why', 'base 자세가 안 온다(object_pose_node)')}")
                continue
            p, q = m.pose.position, m.pose.orientation
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))
            warn = check_cup(n, (p.x, p.y, p.z), tilt, origin.get(n, CUP_ORIGIN_Z))
            print(f"  {n:11s} x {p.x:.3f} y {p.y:+.3f} z {p.z:.3f} 기울기 {tilt:.1f}° · 색 {info.get('color_score')} · "
                  f"흔들림 {info.get('spread_mm')} mm  {'✓' if not warn else '⚠ ' + ' · '.join(warn)}")
        return 1 if lost else 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())

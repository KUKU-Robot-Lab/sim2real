#!/usr/bin/env python3
"""FP++ 로 본 컵 자리 → 방안 2 장면 JSON(hdgp scripts/tools/replay_chain/search_scene.py 의 --scene).

    # 로봇 PC(카메라 · FP++ 켠 상태). 잡을 컵 하나 + (있으면) 피할 컵들
    python3 scripts/ops/fpp_scene.py --cup source200_pink --holder 2 --out /tmp/scene.json \\
        [--obstacle source240_orange:source240_real ...] [--samples 20 --timeout 8]

좌표: /objects/<물체>/pose 는 로봇 base 프레임이고, 로봇 base = 학습 env-local 이다(policy_control rh_aglt ·
pour_obs 와 같은 약속). 그래서 xy 를 그대로 장면에 적는다. z 는 쓰지 않는다(sim 은 컵을 테이블 위에 세운다) —
대신 테이블 위에 서 있는지(기울기)만 검사해 넘어진 컵으로 궤적을 만들지 않게 한다.
수신을 못 하면 멈춘다 — 기본값으로 장면을 만들면 엉뚱한 자리의 궤적이 나온다.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from object_registry import output_topic  # noqa: E402

MAX_TILT_DEG = 10.0      # 이보다 기울면 서 있는 컵이 아니다(넘어짐 · 오인식)
MAX_SPREAD_M = 0.005     # 표본 xy 흩어짐(중앙값 둘레 최대) — 넘으면 추적이 흔들리는 중


def tilt_deg(q_xyzw: tuple[float, float, float, float]) -> float:
    """물체 z 축과 base z 축 사이 각."""
    x, y = q_xyzw[0], q_xyzw[1]
    zz = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def summarize(samples: list[tuple[float, float, float, float, float, float, float]]) -> dict:
    """표본(x, y, z, qx, qy, qz, qw) → 중앙값 xy · z · 기울기 · 흩어짐."""
    if not samples:
        raise ValueError("표본 없음")
    x = st.median(s[0] for s in samples)
    y = st.median(s[1] for s in samples)
    z = st.median(s[2] for s in samples)
    tilt = st.median(tilt_deg(s[3:7]) for s in samples)
    spread = max(math.hypot(s[0] - x, s[1] - y) for s in samples)
    return {"xy": [round(x, 4), round(y, 4)], "z": round(z, 4), "tilt_deg": round(tilt, 1),
            "spread_m": round(spread, 4), "n": len(samples)}


def check(name: str, s: dict) -> None:
    if s["tilt_deg"] > MAX_TILT_DEG:
        raise SystemExit(f"[fpp_scene] {name} 기울기 {s['tilt_deg']}° > {MAX_TILT_DEG} — 서 있는 컵이 아니다")
    if s["spread_m"] > MAX_SPREAD_M:
        raise SystemExit(f"[fpp_scene] {name} xy 흩어짐 {s['spread_m'] * 1000:.1f} mm > "
                         f"{MAX_SPREAD_M * 1000:.0f} — 추적이 흔들린다(가림 · 조명), 다시 찍는다")


def build_scene(cup: dict, holder: int, obstacles: list[tuple[str, dict]]) -> dict:
    """장면 JSON — search_scene 형식. obstacles 는 (sim 물체 이름, summarize 결과)."""
    return {"cup": cup["xy"], "holder": int(holder),
            "obstacles": [{"object": obj, "pos": s["xy"]} for obj, s in obstacles],
            "fpp": {"cup": cup, "obstacles": [s for _, s in obstacles], "t": time.strftime("%Y-%m-%dT%H:%M:%S")}}


def read_poses(names: list[str], samples: int, timeout: float) -> dict[str, list]:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = rclpy.create_node("fpp_scene")
    got: dict[str, list] = {n: [] for n in names}

    def cb_for(n: str):
        def cb(m: PoseStamped) -> None:
            p, q = m.pose.position, m.pose.orientation
            got[n].append((p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        return cb

    for n in names:
        node.create_subscription(PoseStamped, output_topic(n), cb_for(n), qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < timeout and min(len(v) for v in got.values()) < samples:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cup", required=True, help="잡을 컵의 FP++ 물체 이름(예: source200_pink)")
    ap.add_argument("--holder", type=int, required=True, help="sim 홀더 번호(왼손 0 · 오른손 2)")
    ap.add_argument("--obstacle", action="append", default=[],
                    help="피할 컵 '<FP++ 이름>:<sim 물체 이름>' (예: source240_orange:source240_real)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=8.0)
    a = ap.parse_args(argv)

    obs = []
    for spec in a.obstacle:
        if ":" not in spec:
            raise SystemExit(f"[fpp_scene] --obstacle '{spec}' — '<FP++ 이름>:<sim 물체 이름>' 형식")
        obs.append(tuple(spec.split(":", 1)))
    got = read_poses([a.cup] + [f for f, _ in obs], a.samples, a.timeout)
    missing = [n for n, v in got.items() if not v]
    if missing:
        raise SystemExit(f"[fpp_scene] 수신 없음: {', '.join(output_topic(n) for n in missing)} — FP++ 켜졌는지 확인")
    cup = summarize(got[a.cup])
    check(a.cup, cup)
    obstacles = []
    for fname, sim_name in obs:
        s = summarize(got[fname])
        check(fname, s)
        obstacles.append((sim_name, s))
    scene = build_scene(cup, a.holder, obstacles)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(scene, ensure_ascii=False, indent=1))
    print(f"[fpp_scene] 컵 {a.cup} xy {cup['xy']} · z {cup['z']} · 기울기 {cup['tilt_deg']}° · 흩어짐 "
          f"{cup['spread_m'] * 1000:.1f} mm · 장애물 {len(obstacles)} → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""hand_grip_replay — 실기 bag 한 에피소드의 손 목표(/hand_<s>/angle_target)를 손에만 다시 보낸다(팔 없이).

10.13 s2r_1st: rg3 오른손 쥠 실패의 제어 쪽 원인(어드민턴스 힘 입력 배율, 마스터 --adm-fscale)을 정책 없이 확인한다.
운영자가 컵을 손 안 쥐는 자리에 대고 있는 동안, 정책이 그 에피소드에서 보낸 목표를 같은 시각 간격으로 재생한다.

  단계  lead   : --from-s 시각의 목표(쥐기 직전 편 자세)를 --lead-s 동안 유지 — 이때 운영자가 컵을 댄다
        replay : --from-s ~ --to-s(기본 그 에피소드 stop) 목표를 원래 간격으로
        hold   : 마지막 목표를 --hold-s 동안 — 운영자가 컵을 당기거나 들어 본다
        release: lead 목표로 되돌린다(--no-release 면 마지막 목표 유지로 끝)
  중단  손가락 전류가 --abort-ma 를 --abort-s 넘게 넘거나 Ctrl-C · SIGTERM → lead 목표(편 자세)를 보내고 끝낸다.

    python3 deploy/policy_control/tools/hand_grip_replay.py --bag logs/bags/20261013_175321_episode_grasp_rg3_right \\
        --side right --episode 2 --from-s 3.5                       # 계획만 보인다(손 무동작)
    … --execute                                                     # 실기 — 실행 직전 사용자 허락(sim2real CLAUDE.md)

손은 rh56f1_ecat_node 가 떠 있어야 한다(같은 도메인). 기록은 따로 ros2 bag record 로 남긴다.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palm_cmd import check_domain  # noqa: E402

PUB_HZ = 60.0            # 정책 발행 주기(bag 중앙 16.7 ms)
STATE_TIMEOUT_S = 3.0    # 손 상태가 이만큼 안 오면 드라이버가 없는 것으로 본다
ABORT_MA = 750.0         # 계약 current_limit 800 mA 바로 아래(마스터 current_hold 650 mA 는 이보다 먼저 막는다)
ABORT_S = 0.3


@dataclass(frozen=True)
class Plan:
    hand_id: int
    lead: np.ndarray            # (6,) 레지스터
    times: np.ndarray           # (N,) 재생 시작 기준 s
    targets: np.ndarray         # (N, 6)
    t_from: float
    t_to: float


def episode_window(events: list[tuple[float, dict]], episode: int) -> tuple[float, float]:
    """episode 토픽 (시각, json) → episode 번째(1부터) start 와 그 뒤 첫 stop. 없으면 ValueError."""
    starts = [t for t, e in events if e.get("event") == "start"]
    if not 1 <= episode <= len(starts):
        raise ValueError(f"에피소드 {episode} 없음(start {len(starts)} 개)")
    t0 = starts[episode - 1]
    t1 = next((t for t, e in events if e.get("event") == "stop" and t > t0), None)
    if t1 is None:
        raise ValueError(f"에피소드 {episode} 의 stop 이 없다")
    return t0, t1


def make_plan(stamps: np.ndarray, values: np.ndarray, hand_id: int, t0: float, t1: float,
              t_from: float, t_to: float | None) -> Plan:
    """bag 목표(시각, 6) → 재생 계획. t_from · t_to = 에피소드 start 기준 s(t_to 없으면 stop)."""
    stamps = np.asarray(stamps, float)
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != 6 or len(stamps) != len(values):
        raise ValueError(f"목표 모양이 (N, 6) 이 아니다: {values.shape}")
    end = (t1 - t0) if t_to is None else float(t_to)
    if not 0.0 <= t_from < end <= (t1 - t0) + 1e-6:
        raise ValueError(f"구간 0 ≤ from {t_from} < to {end} ≤ 에피소드 길이 {t1 - t0:.2f} s")
    i0 = int(np.searchsorted(stamps, t0 + t_from, "right")) - 1
    if i0 < 0:
        raise ValueError("from 시각 전에 보낸 목표가 없다")
    sel = (stamps > t0 + t_from) & (stamps <= t0 + end)
    if not sel.any():
        raise ValueError("구간 안에 목표가 없다")
    if not np.all(np.isfinite(values[sel])) or not np.all(np.isfinite(values[i0])):
        raise ValueError("목표에 유한하지 않은 값")
    return Plan(int(hand_id), values[i0].copy(), stamps[sel] - (t0 + t_from), values[sel].copy(), float(t_from), float(end))


def over_current(history: list[tuple[float, float]], now: float, limit_ma: float, span_s: float) -> bool:
    """(시각, 그 시각 손가락 최대 전류) 기록에서 최근 span_s 동안 줄곧 limit 넘었나."""
    recent = [c for t, c in history if now - t <= span_s]
    covered = bool(history) and now - history[0][0] >= span_s
    return covered and bool(recent) and min(recent) > limit_ma


def load_plan(bag: str, side: str, episode: int, t_from: float, t_to: float | None) -> Plan:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    def read(uri: str, topic: str) -> list:
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
        types = {t.name: t.type for t in r.get_all_topics_and_types()}
        if topic not in types:
            raise ValueError(f"{uri} 에 {topic} 이 없다")
        out = []
        while r.has_next():
            t, d, ts = r.read_next()
            if t == topic:
                out.append((ts * 1e-9, deserialize_message(d, get_message(types[t]))))
        return out

    ev = [(t, json.loads(m.data)) for t, m in read(f"{bag}/arm", f"/policy_control/{side}/episode")]
    t0, t1 = episode_window(ev, episode)
    tg = read(f"{bag}/hand", f"/hand_{side}/angle_target")
    stamps = np.array([t for t, _ in tg])
    values = np.array([list(m.joint_values) for _, m in tg], float)
    return make_plan(stamps, values, tg[0][1].hand_id, t0, t1, t_from, t_to)


def run(plan: Plan, side: str, lead_s: float, hold_s: float, release: bool, abort_ma: float, abort_s: float) -> int:
    import rclpy
    from rclpy.node import Node
    from rh56f1_interfaces.msg import GetAngleAct1, GetCurrentAct1, SetAngle1

    rclpy.init()
    node = Node(f"hand_grip_replay_{side}")
    pub = node.create_publisher(SetAngle1, f"/hand_{side}/angle_target", 10)
    seen = {"angle": 0.0}
    cur: list[tuple[float, float]] = []
    node.create_subscription(GetAngleAct1, f"/hand_{side}/angle_actual", lambda m: seen.__setitem__("angle", time.monotonic()), 10)
    node.create_subscription(GetCurrentAct1, f"/hand_{side}/current_actual",
                             lambda m: cur.append((time.monotonic(), float(max(list(m.joint_values)[:5])))), 50)
    stop: dict[str, str | None] = {"why": None}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("why", "신호"))

    def send(v: np.ndarray) -> None:
        m = SetAngle1()
        m.header.stamp = node.get_clock().now().to_msg()
        m.hand_id = plan.hand_id
        m.joint_values = [int(round(x)) for x in v]
        pub.publish(m)

    def spin_until(t_end: float) -> bool:
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.002)
            now = time.monotonic()
            del cur[:max(0, len(cur) - 2000)]
            if stop["why"] is None and over_current(cur, now, abort_ma, abort_s):
                stop["why"] = f"전류 > {abort_ma:.0f} mA {abort_s:.1f} s"
            if stop["why"] is None and now - seen["angle"] > STATE_TIMEOUT_S:
                stop["why"] = "손 상태 끊김"
            if stop["why"]:
                return False
        return True

    rc = 0
    t_wait = time.monotonic() + STATE_TIMEOUT_S
    while time.monotonic() < t_wait and seen["angle"] == 0.0:
        rclpy.spin_once(node, timeout_sec=0.05)
    if seen["angle"] == 0.0:
        print(f"✗ /hand_{side}/angle_actual 이 안 온다 — 드라이버 · 도메인 확인", flush=True)
        node.destroy_node(); rclpy.shutdown()
        return 2
    try:
        print(f"[lead] 편 자세 {plan.lead.astype(int).tolist()} {lead_s:.0f} s — 컵을 손에 댄다", flush=True)
        t = time.monotonic()
        ok = True
        while ok and time.monotonic() - t < lead_s:
            send(plan.lead)
            ok = spin_until(time.monotonic() + 1.0 / PUB_HZ)
        if ok:
            print(f"[replay] {plan.t_from:.2f}~{plan.t_to:.2f} s ({len(plan.times)} 목표)", flush=True)
            t = time.monotonic()
            for dt, v in zip(plan.times, plan.targets):
                if not (ok := spin_until(t + dt)):
                    break
                send(v)
        if ok:
            print(f"[hold] 마지막 목표 {plan.targets[-1].astype(int).tolist()} {hold_s:.0f} s — 컵을 당기거나 들어 본다", flush=True)
            t = time.monotonic()
            while ok and time.monotonic() - t < hold_s:
                send(plan.targets[-1])
                ok = spin_until(time.monotonic() + 1.0 / PUB_HZ)
        if not ok:
            print(f"✗ 중단({stop['why']}) — 편 자세로", flush=True)
            rc = 1
        if release or not ok:
            for _ in range(int(PUB_HZ)):
                send(plan.lead)
                rclpy.spin_once(node, timeout_sec=1.0 / PUB_HZ)
            print("[release] 편 자세 보냄", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--bag", required=True)
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--from-s", type=float, default=0.0, help="에피소드 start 기준 재생 시작 s")
    ap.add_argument("--to-s", type=float, default=None, help="재생 끝 s(기본 stop)")
    ap.add_argument("--lead-s", type=float, default=10.0)
    ap.add_argument("--hold-s", type=float, default=10.0)
    ap.add_argument("--no-release", action="store_true")
    ap.add_argument("--abort-ma", type=float, default=ABORT_MA)
    ap.add_argument("--abort-s", type=float, default=ABORT_S)
    ap.add_argument("--execute", action="store_true", help="실기 손에 보낸다(없으면 계획만)")
    ap.add_argument("--allow-domain-0", action="store_true")
    a = ap.parse_args(argv)
    plan = load_plan(a.bag, a.side, a.episode, a.from_s, a.to_s)
    print(f"계획 {a.side} ep{a.episode}: lead {plan.lead.astype(int).tolist()} {a.lead_s:.0f} s → "
          f"재생 {plan.t_from:.2f}~{plan.t_to:.2f} s({len(plan.times)} 목표, 끝 {plan.targets[-1].astype(int).tolist()}) → "
          f"유지 {a.hold_s:.0f} s → {'유지로 끝' if a.no_release else '편 자세'}. 중단: 전류 > {a.abort_ma:.0f} mA {a.abort_s} s")
    if not a.execute:
        print("(계획만 — 실기는 --execute)")
        return 0
    why = check_domain(dict(os.environ), a.allow_domain_0)
    if why:
        print(f"✗ {why}")
        return 2
    return run(plan, a.side, a.lead_s, a.hold_s, not a.no_release, a.abort_ma, a.abort_s)


if __name__ == "__main__":
    sys.exit(main())

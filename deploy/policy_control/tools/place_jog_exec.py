#!/usr/bin/env python3
"""고정 놓기 jog 실행기 — plan_place_fixed --jog 의 끝 관절(place_cmd.npz 팔 마지막 행)로 board_jog --control precise 방식으로 간다.

11.18 실기(blue_tune_jog_06): replay_traj_rh 로 실측 → 실측 + Δq 를 흘리면 pd 세트포인트 ↔ 실측 정상 오차 안이라 1 mm jog 가 사라졌다.
여기서는 board_jog precise(기본 유지 보정)와 같게 한다:
  c = pd 세트포인트 − 정지 실측(touch_probe.start_e0: 멈춤 · max|c| ≤ E0_START_MAX, 발행 시작에 고정 — 마찰을 쫓아 적분하지 않는다).
  명령 = 실측 → 실측 + Δq(precise_jog.QuinticPath, 120 Hz · ≤ MAX_VEL(0.25) rad/s · ≤ 0.60 rad/s²), 발행 = 명령 + c = q_set → q_set + Δq.
손 = pd applied 손 6 고정. 발행 토픽 · pd TRACKING/estop 감시 · 끝 episode stop 은 replay_traj_rh 와 같다.
--execute 없으면 검사 · 계산만(발행 안 함).

    python3 deploy/policy_control/tools/place_jog_exec.py --npz <out>/plan/place_cmd.npz [--execute]
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_control import place_teach as P  # noqa: E402
from precise_jog import QuinticPath  # noqa: E402
from touch_probe import PD_VEL_RAD_S  # noqa: E402

PUB_HZ = 120.0
MAX_VEL, MAX_ACC = min(0.30, PD_VEL_RAD_S), 0.60   # board_jog precise 상한, pd 세트포인트 전진 상한보다 빠르지 않게(rad/s · rad/s²)
MAX_JOG_DQ_RAD = 0.15                  # jog 누적 상한이 40 mm · 15°(0.26 rad)라 한 번 jog 는 이보다 작다 — 넘으면 계획 오류
WAIT_S = 5.0
STOP_REPUB_S = 0.1                     # 끝 episode stop 재발행 시간
SPIN_MAX = 5                           # 발행 루프 한 번에 콜백 비우기 최대 횟수(timeout 0)


def names_fault(side: str, arm_names: list[str], hand_names: list[str]) -> str | None:
    """npz meta 이름이 그 팔(순서까지)·그 쪽 손과 다르면 이유. 순수."""
    if tuple(arm_names) != P.arm_names(side):
        return f"meta 팔 이름 {list(arm_names)} ≠ {list(P.arm_names(side))}"
    if tuple(hand_names) != P.hand_names(side):
        return f"meta 손 이름 {list(hand_names)} ≠ {list(P.hand_names(side))}"
    return None


def dq_fault(dq: np.ndarray) -> str | None:
    """jog Δq 가 상한을 넘으면 이유. 순수."""
    m = float(np.abs(dq).max())
    return f"max|Δq| {m:.4f} rad > {MAX_JOG_DQ_RAD} — jog 크기가 아님" if m > MAX_JOG_DQ_RAD else None


def jog_stream(meas: np.ndarray, q_set: np.ndarray, goal: np.ndarray, hz: float = PUB_HZ) -> tuple[np.ndarray, np.ndarray]:
    """발행할 팔 (q, qd) 행들. 명령 = 실측 → goal 5차 경로, 발행 = 명령 + c(c = q_set − 실측, 고정). 첫 행 = q_set(튐 없음). 순수."""
    meas, q_set, goal = (np.asarray(v, float) for v in (meas, q_set, goal))
    path, c = QuinticPath(meas, goal, max_vel=MAX_VEL, max_acc=MAX_ACC), q_set - meas
    n = int(np.ceil(path.duration * hz)) + 1
    rows = [path.sample(k / hz) for k in range(n)]
    return np.array([q for q, _ in rows]) + c, np.array([qd for _, qd in rows])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--max-start-gap", type=float, default=0.05, help="팔 실측 ↔ 계획 첫 행 상한(rad)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    d = np.load(a.npz)
    side = str(d["meta_side"])
    arm = np.asarray(d["arm_q"], float)
    arm_n, hand_n = [*map(str, d["meta_arm_names"])], [*map(str, d["meta_hand_names"])]
    if why := names_fault(side, arm_n, hand_n) or dq_fault(arm[-1] - arm[0]):
        print(f"[place_jog] ✗ {why} — 발행 안 함 · rc 3", flush=True)
        return 3
    return run(side, [*arm_n, *hand_n], arm[0], arm[-1], a)


def run(side: str, names: list[str], q_first: np.ndarray, q_last: np.ndarray, a) -> int:
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    from hand_grasp import arm_hold_target
    from replay_to_pd import EPISODE_TOPIC_SIDE, PD_STATUS, STATE_TOPIC, STOP_WAIT_S, TOPIC, replay_episode_id
    from teach_place import MOVING, arm_limits, pd_fault
    from touch_probe import E0_WIN_S, start_e0

    hand_names = names[7:]
    arm_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]   # /joint_states 실측 이름(발행은 계획 이름 names[:7])
    rclpy.init()
    node = rclpy.create_node("place_jog_exec")
    arm_m: dict = {}
    applied: dict = {}
    pd: dict = {}
    win: list = []          # (t, 팔 실측 7) — 정지 처짐 창
    collect = [True]        # 사전 검사 뒤에는 창에 쌓지 않는다

    def on_arm(m) -> None:
        arm_m.update(zip(m.name, m.position))
        if collect[0] and all(n in arm_m for n in arm_names):
            win.append((time.monotonic(), np.array([arm_m[n] for n in arm_names])))

    def on_pd(m) -> None:
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        pd.clear()
        pd.update(d, _t=time.monotonic())

    node.create_subscription(JointState, STATE_TOPIC, on_arm, qos_profile_sensor_data)
    node.create_subscription(String, PD_STATUS.format(side=side), on_pd, 10)
    node.create_subscription(JointState, f"/policy_control/pd_{side}/applied", lambda m: applied.update(zip(m.name, m.position)), 10)
    pub = node.create_publisher(JointState, TOPIC, 10)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ep_pub = node.create_publisher(String, EPISODE_TOPIC_SIDE.format(side=side), latched)

    def bye(rc: int, why: str) -> int:
        print(f"[place_jog] {why} · rc {rc}", flush=True)
        node.destroy_node()
        rclpy.shutdown()
        return rc

    t0 = time.monotonic()
    while time.monotonic() - t0 < WAIT_S and not (win and win[-1][0] - win[0][0] >= E0_WIN_S + 0.1 and pd
                                                 and all(n in applied for n in hand_names)):
        rclpy.spin_once(node, timeout_sec=0.02)
    now = time.monotonic()
    if why := pd_fault(pd, now):        # 발행 전은 TRACKING 만(RAMPING 거부)
        return bye(3, f"✗ {why} — 발행 안 함")
    if not win or not all(n in applied for n in hand_names):
        miss = [n for n in hand_names if n not in applied]
        return bye(3, f"✗ 팔 실측 {len(win)} 행 · pd applied 손 없음 {miss} — 발행 안 함")
    collect[0] = False
    W = [(t, q) for t, q in win if t >= win[-1][0] - E0_WIN_S]
    meas = W[-1][1]
    gap = float(np.abs(meas - q_first).max())
    if gap > a.max_start_gap:
        return bye(3, f"✗ 팔 실측 ↔ 계획 첫 행 {gap:.3f} rad > {a.max_start_gap} — 발행 안 함")
    q_set, why = arm_hold_target(pd, side, meas)
    if q_set is None:
        return bye(3, f"✗ {why}")
    e0, why = start_e0([t for t, _ in W], [q for _, q in W], q_set)
    if e0 is None:
        return bye(3, f"✗ 유지 보정 거부: {why}")
    meas = q_set + e0                       # 창 중앙값 — c = q_set − meas = −e0
    goal = meas + (q_last - q_first)        # 계획 Δq 를 지금 실측에 얹는다(계획 시점과 같은 자리 · ≤ max-start-gap)
    q_pub, qd_pub = jog_stream(meas, q_set, goal)
    lo, hi = arm_limits(side)
    if np.any(q_pub < lo + P.LIMIT_MARGIN_RAD) or np.any(q_pub > hi - P.LIMIT_MARGIN_RAD):
        return bye(2, "✗ 발행값(유지 보정 c 포함)이 관절 한계(여유 포함) 밖 — 발행 안 함")
    hand = np.array([applied[n] for n in hand_names])
    dq = q_last - q_first
    print(f"[place_jog] {side} · Δq max {np.abs(dq).max():.4f} rad · c {np.round(q_set - meas, 4).tolist()} · "
          f"{len(q_pub)} 행 @ {PUB_HZ:g} Hz ({(len(q_pub) - 1) / PUB_HZ:.2f} s) · 손 {np.round(hand, 3).tolist()}", flush=True)
    if not a.execute:
        return bye(0, "드라이런(--execute 로 발행)")

    ep_id = replay_episode_id()

    def event(kind: str, **kw) -> None:
        ep_pub.publish(String(data=json.dumps({"episode": ep_id, "event": kind, "t_ns": time.time_ns(),
                                               "object_anchor": None, "home_q": {}, **kw})))

    def send(k: int, q: np.ndarray, qd: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = f"{ep_id}:{k}"
        msg.name, msg.position = list(names), [*q.tolist(), *hand.tolist()]
        msg.velocity, msg.effort = [*qd.tolist(), *[0.0] * len(hand)], [0.0] * len(names)
        pub.publish(msg)

    def spin_drain() -> None:
        for _ in range(SPIN_MAX):
            rclpy.spin_once(node, timeout_sec=0.0)

    def on_term(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)   # SIGTERM 도 Ctrl-C 와 같은 정리 경로
    rc, why = 0, "done"
    t_start = time.monotonic()
    try:
        event("hand_mode", hand_command="position", reasons=["place_jog_exec 손 = pd applied 고정"])
        for k in range(len(q_pub)):
            send(k, q_pub[k], qd_pub[k])
            spin_drain()
            if f := pd_fault(pd, time.monotonic(), MOVING):
                rc, why = 3, f"{f} (행 {k}/{len(q_pub)})"
                break
            delay = t_start + (k + 1) / PUB_HZ - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        rc, why = 130, "Ctrl-C · SIGTERM"
    except Exception as e:  # noqa: BLE001 — 어떤 예외든 stop 을 내고 정리한다
        rc, why = 1, f"예외 {e!r}"
    finally:
        wall = time.monotonic() - t_start
        t1 = time.monotonic()
        while True:                         # rc 와 무관하게 stop 을 몇 번 다시 낸다
            event("stop", reasons=[f"place_jog_exec {why}"])
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.monotonic() - t1 >= STOP_REPUB_S:
                break
    t1 = time.monotonic()
    while time.monotonic() - t1 < STOP_WAIT_S and pd.get("target") != "internal" and rc == 0:
        send(len(q_pub), q_pub[-1], np.zeros(7))
        rclpy.spin_once(node, timeout_sec=1.0 / PUB_HZ)
    return bye(rc, f"{why} · {len(q_pub)} 행 {wall:.2f} s(평균 {len(q_pub) / max(wall, 1e-9):.1f} Hz) · pd {pd.get('phase')}")


if __name__ == "__main__":
    sys.exit(main())

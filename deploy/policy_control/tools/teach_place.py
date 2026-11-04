#!/usr/bin/env python3
"""place 직접 교시 — pd 를 engage 한 채 한 팔을 손으로 끌어 홀더까지 옮기고, 키로 손을 쥐고 편다. 녹화는 npz 하나.

11.04 사용자: 틸트 시작자세에서 컵을 쥐고 → 팔을 손으로 옮겨 홀더 위에 맞추고 → 손가락을 펴 떨어뜨린다. 홀더 1·2 = 오른팔,
홀더 0 = 왼팔. 콘솔 · pd 를 오가지 않는다: pd 가 그 팔을 TRACKING 으로 붙든 상태에서 이 도구가 /policy_control/joint_target
(팔 7 + 손 6)을 policy_hz 로 낸다. 법칙은 policy_control/place_teach.py(robot_control.teaching 상태기계).

    키: g = 쥠 · o = 폄 · space = 잠금(밀어도 안 따라감) · q = 끝(녹화 저장) · x = 버림(저장 안 함)

순서(팔마다): 콘솔 bringup → pd engage → home → 틸트 시작자세(pour_start) → 이 도구.
  python3 deploy/policy_control/tools/teach_place.py --side right --out logs/place_teach/holder1.npz [--execute]
--execute 없이는 측정 · pd 상태만 읽고 계획을 출력한다(발행 0). ★실기 동작 — 사용자 승인 뒤에만 --execute.
끝나면 그 팔 episode stop → pd 가 마지막 팔 · 손 목표를 붙든다. 재생: replay_traj_rh.py(팔 + 손) 또는 replay_to_pd.py(팔만).

종료 코드: 0 = 저장 · 2 = 저장할 움직임 없음 · 3 = pd 이상(HOLD · estop · 단계) · 4 = 측정 없음 · 5 = 정지 신호 · 1 = 그 밖.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import place_teach as P  # noqa: E402
from policy_control import _paths  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_grasp import HAND_STATE, MEAS_WAIT_S, STALE_S, arm_hold_target  # noqa: E402
from replay_to_pd import EPISODE_TOPIC_SIDE, PD_STATUS, STATE_TOPIC, STOP_WAIT_S, TOPIC, replay_episode_id  # noqa: E402

MOVING = ("RAMPING", "TRACKING")
PROFILE = "openarm_rh56f1"


def arm_limits(side: str) -> tuple[np.ndarray, np.ndarray]:
    from robot_control.profile import load_builtin_profile

    joints = {j.canonical: j for j in load_builtin_profile(PROFILE).joints}
    names = P.arm_names(side)
    return np.array([joints[n].lower for n in names]), np.array([joints[n].upper for n in names])


def pd_fault(pd: dict, now: float, phases=("TRACKING",)) -> str | None:
    """pd 가 그 팔을 붙들고 있지 않으면 이유. 순수."""
    if not pd or now - float(pd.get("_t", 0.0)) > 1.0:
        return "pd status 없음(1 s 넘게)"
    if pd.get("estop"):
        return f"pd estop {pd.get('reasons', '')}"
    if pd.get("phase") not in phases:
        return f"pd {pd.get('phase')} — engage 해서 TRACKING 인 팔에서만 교시한다 {pd.get('reasons', '')}"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--out", required=True, help="녹화 npz")
    ap.add_argument("--grip", help=f"쥠 자세 rad, {','.join(P.HAND_AXES)} (기본 {','.join(map(str, P.GRIP_RAD))})")
    ap.add_argument("--open", dest="open_", help=f"폄 자세 rad (기본 {','.join(map(str, P.OPEN_RAD))})")
    ap.add_argument("--max-vel", type=float, default=P.MAX_VEL_RAD_S, help="팔 목표 속도 상한 rad/s(pd reduced 0.25 의 0.8)")
    ap.add_argument("--seconds", type=float, default=300.0, help="교시 세션 최대 길이(손목 발열)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    try:
        if Path(a.out).suffix != ".npz":
            raise P.TeachError(f"--out 은 .npz: {a.out}")
        if Path(a.out).exists():
            raise P.TeachError(f"{a.out} 가 이미 있다 — 덮어쓰지 않는다")
        lo, hi = arm_limits(a.side)
        teacher = P.Teacher(a.side, lo, hi, P.parse_pose(a.grip, P.GRIP_RAD, "--grip"),
                            P.parse_pose(a.open_, P.OPEN_RAD, "--open"), max_vel=a.max_vel)
    except (P.TeachError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    rc, info = run(a.side, teacher, Path(a.out), a.seconds, a.execute)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return rc


def run(side: str, teacher: P.Teacher, out: Path, seconds: float, execute: bool) -> tuple[int, dict]:
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    from robot_control.keys import KeyReader

    arm_names, hand_names = list(P.arm_names(side)), list(P.hand_names(side))
    arm_src = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    rclpy.init()
    node = rclpy.create_node(f"teach_place_{side}")
    arm, hand, pd = {}, {}, {}

    def on_arm(m) -> None:
        arm.update({n: (p, time.monotonic()) for n, p in zip(m.name, m.position)})

    def on_hand(m) -> None:
        hand.update({n: (p, time.monotonic()) for n, p in zip(m.name, m.position)})

    def on_pd(m) -> None:
        try:
            pd.clear()
            pd.update(json.loads(m.data), _t=time.monotonic())
        except ValueError:
            pass

    node.create_subscription(JointState, STATE_TOPIC, on_arm, qos_profile_sensor_data)
    node.create_subscription(JointState, HAND_STATE.format(side=side), on_hand, qos_profile_sensor_data)
    node.create_subscription(String, PD_STATUS.format(side=side), on_pd, 10)

    def fresh(store: dict, names: list) -> np.ndarray | None:
        now = time.monotonic()
        if not all(n in store and now - store[n][1] < STALE_S for n in names):
            return None
        return np.array([store[n][0] for n in names], float)

    stop_sig = {"on": False}

    def on_term(_signum, _frame) -> None:
        stop_sig["on"] = True

    prev_term = signal.signal(signal.SIGTERM, on_term)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < MEAS_WAIT_S:
            rclpy.spin_once(node, timeout_sec=0.05)
            if fresh(arm, arm_src) is not None and fresh(hand, hand_names) is not None and pd:
                break
        q_arm, q_hand = fresh(arm, arm_src), fresh(hand, hand_names)
        if q_arm is None or q_hand is None:
            return 4, {"phase": "failed", "why": f"측정 없음 — {STATE_TOPIC} 또는 {HAND_STATE.format(side=side)}"}
        why = pd_fault(pd, time.monotonic())
        if why:
            return 3, {"phase": "failed", "why": f"시작 전 {why} — 아무것도 보내지 않았다"}
        q_set, why = arm_hold_target(pd, side, q_arm)
        if q_set is None:
            return 3, {"phase": "failed", "why": f"시작 전 {why} — 아무것도 보내지 않았다"}
        plan = {"side": side, "out": str(out), "rate_hz": round(1.0 / teacher.dt, 1), "max_vel": teacher.max_vel,
                "q_set": np.round(q_set, 4).tolist(), "hand_now": np.round(q_hand, 3).tolist(),
                "grip": teacher.grip.tolist(), "open": teacher.open.tolist()}
        if not execute:
            return 0, {"phase": "dry_run", "plan": plan, "why": "--execute 없이 발행하지 않았다"}
        if not sys.stdin.isatty():
            return 1, {"phase": "failed", "why": "터미널(tty)에서 실행해야 키를 읽는다 — 아무것도 보내지 않았다"}

        pub = node.create_publisher(JointState, TOPIC, 10)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        ep_pub = node.create_publisher(String, EPISODE_TOPIC_SIDE.format(side=side), latched)
        ep_id = replay_episode_id()

        def event(name: str, **kw) -> None:
            ep_pub.publish(String(data=json.dumps({"episode": ep_id, "event": name, "t_ns": time.time_ns(), **kw})))

        def send(k: int, qa: np.ndarray, qh: np.ndarray) -> None:
            m = JointState()
            m.header.stamp = node.get_clock().now().to_msg()
            m.header.frame_id = f"{ep_id}:{k}"
            m.name = arm_names + hand_names
            m.position = [float(v) for v in np.concatenate([qa, qh])]
            m.velocity = [0.0] * len(m.name)
            m.effort = [0.0] * len(m.name)
            pub.publish(m)

        event("hand_mode", hand_command="position", reasons=["teach_place 손 위치 명령"])
        start = time.monotonic()
        teacher.start(q_set, q_arm, q_hand, 0.0)
        print(f"교시 시작({side}): g 쥠 · o 폄 · space 잠금 · q 끝(저장) · x 버림 — 최대 {seconds:g} s", flush=True)
        k, rc, why, keep = 0, 0, "q", True
        qa, qh = teacher.command, teacher.hand
        t_next = time.monotonic()
        with KeyReader() as keys:
            while True:
                rclpy.spin_once(node, timeout_sec=0.0)
                now = time.monotonic()
                if stop_sig["on"]:
                    rc, why = 5, "정지 신호(SIGTERM)"
                    break
                if now - start > seconds:
                    why = f"{seconds:g} s 지남"
                    break
                meas = fresh(arm, arm_src)
                fault = pd_fault(pd, now, MOVING)
                if meas is None or fault:
                    rc, why = (3, fault) if fault else (4, "팔 측정이 끊겼다")
                    break
                key = keys.poll()
                if key in ("q", "x"):
                    keep, why = key == "q", "q" if key == "q" else "x(버림)"
                    break
                if key is not None:
                    note = teacher.key(key, now - start)
                    if note:
                        print(f"{now - start:7.2f} s  {note}", flush=True)
                qa, qh, ev = teacher.tick(meas, now - start)
                if ev:
                    print(f"{now - start:7.2f} s  팔: {ev}", flush=True)
                send(k, qa, qh)
                k += 1
                t_next += teacher.dt
                time.sleep(max(0.0, t_next - time.monotonic()))
        event("stop", object_anchor=None, home_q={}, reasons=[f"teach_place 끝: {why}"])
        t1 = time.monotonic()
        while time.monotonic() - t1 < STOP_WAIT_S and pd.get("target") != "internal":
            send(k, qa, qh)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(teacher.dt)
        info = {"phase": "stopped", "why": why, "ticks": k, "pd_target": pd.get("target"), "plan": plan,
                "events": [{"name": e.name, "t": round(e.t, 2)} for e in teacher.events]}
        if not keep:
            return rc, {**info, "saved": None}
        try:
            fields = P.build_recording(teacher)
        except P.TeachError as error:
            return (rc or 2), {**info, "saved": None, "why": f"{why}; 저장 안 함: {error}"}
        P.save_recording(out, fields)
        segs = json.loads(str(fields["meta_segments"]))
        return rc, {**info, "saved": str(out), "frames": int(fields["arm_q"].shape[0]),
                    "seconds": round(float(fields["arm_q"].shape[0] * fields["meta_step_dt"]), 2), "segments": segs}
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""방안 2 실기 재생 — sim 성공 궤적(s2r_replay_v1: 팔 7 + 손 6)을 pd 입력(/policy_control/joint_target)으로 흘린다.

궤적은 hdgp `scripts/tools/replay_chain/stitch.py` 가 만든다(sim 에서 잡기 → 놓기를 이어 굴려 성공한 env 만).
계획(열 선택 · 보간 · 손 모드 사건 · 한계 · 속도)은 policy_control.rh_replay 가 정하고, 여기서는 확인과 발행만 한다.

    # 무발행(기본): 계획 · 한계 · 속도만 본다
    python3 deploy/policy_control/tools/replay_traj_rh.py --npz <후보>.npz \
        --contract deploy/policies/rh56f1/aglt/left_shk21/rh_aglt_contract.json --rate-scale 0.2
    # ★실기 — 사용자 승인 뒤에만
    ... --object-topic /objects/shaker_c_orange/pose --execute

발행 전에 거부: 이름·측·모양 · 계약 한계 밖 · 단계 속도(pd max_vel × 0.8) 초과 · 팔 실측이 첫 프레임에서 --max-start-gap 넘음
· 물체가 sim 출발 자리에서 --object-tol 넘게 벗어남(open-loop 라 보정이 없다) · pd 가 TRACKING/RAMPING 이 아님.
실행 중 pd 가 HOLD·IDLE 로 가면 발행을 멈추고 끝낸다. 끝(또는 Ctrl-C)에는 이 팔 episode stop → pd 가 마지막 목표를 붙든다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402,F401
from policy_control import rh_replay as R  # noqa: E402

TOPIC = "/policy_control/joint_target"
EPISODE_TOPIC_SIDE = "/policy_control/{side}/episode"
PD_STATUS = "/policy_control/status/pd_{side}"
ARM_STATE = "/joint_states"
HAND_STATE = "/hand_{side}/joint_states"
MOVING = ("RAMPING", "TRACKING")
STOP_WAIT_S = 1.0
#: pd_rh56f1.yaml max_vel(단계) · hand.max_vel_track — 여기 값의 0.8 배까지만 쓴다
STAGE_ARM_VMAX = {"reduced": 0.25, "full": 2.0}
HAND_VMAX = 2.1
SPEED_MARGIN = 0.8


def contract_limits(path: Path, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    from policy_control import rh_aglt

    c = rh_aglt.load_contract(path)
    s = next(iter(c.sides.values()))
    lim = dict(zip(list(s.arm_joints) + list(s.hand_joints),
                   zip(list(s.arm_lo) + list(s.hand_lim_lo), list(s.arm_hi) + list(s.hand_lim_hi))))
    missing = [n for n in names if n not in lim]
    if missing:
        raise SystemExit(f"[replay_rh] 계약에 관절이 없다(다른 팔 계약?): {missing}")
    return np.array([lim[n][0] for n in names]), np.array([lim[n][1] for n in names])


def summarize(traj: R.ReplayTraj, plan: R.Plan, args) -> list[str]:
    """사람이 읽을 요약과 거부 사유 목록."""
    arm_vmax = STAGE_ARM_VMAX[args.stage] * SPEED_MARGIN
    hand_vmax = HAND_VMAX * SPEED_MARGIN
    errs = R.speed_violations(plan, traj.names, arm_vmax=arm_vmax, hand_vmax=hand_vmax)
    if args.contract:
        lo, hi = contract_limits(args.contract, traj.names)
        errs += R.limit_violations(traj.q, lo, hi, traj.names, margin=args.limit_margin)
    segs = " → ".join(f"{s['label']}({(s['end'] - s['start'] + 1) * traj.step_dt / args.rate_scale:.1f}s·{s['hand_command']})"
                      for s in traj.segments)
    print(f"[replay_rh] {traj.side} · {traj.n_frames} 프레임({traj.n_frames * traj.step_dt:.2f} s sim) × {args.rate_scale:g}"
          f" → 램프 {plan.n_ramp * plan.pub_dt:.1f} s + 본 {plan.seconds - plan.n_ramp * plan.pub_dt:.1f} s"
          f" · 팔 {traj.arm_src} · 손 {traj.hand_src} · 발행 {1 / plan.pub_dt:.0f} Hz")
    print(f"[replay_rh] 구간 {segs}")
    print(f"[replay_rh] 손 모드 사건 {list(plan.mode_at)} · 허용 최대 rate_scale "
          f"{R.max_rate_scale(traj, arm_vmax=arm_vmax, hand_vmax=hand_vmax):.3f} (단계 {args.stage})")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--contract", type=Path, default=None, help="한계 검사용 rh_aglt 계약(그 팔)")
    ap.add_argument("--arm-src", default="arm_q", choices=R.ARM_SOURCES)
    ap.add_argument("--hand-src", default="hand_q_cmd", choices=R.HAND_SOURCES)
    ap.add_argument("--rate-scale", type=float, default=0.2)
    ap.add_argument("--pub-hz", type=float, default=60.0)
    ap.add_argument("--stage", default="reduced", choices=tuple(STAGE_ARM_VMAX))
    ap.add_argument("--limit-margin", type=float, default=0.0)
    ap.add_argument("--hand-ramp-speed", type=float, default=0.5, help="진입 램프 손 속도 [rad/s](팔은 0.1 고정)")
    ap.add_argument("--max-start-gap", type=float, default=0.15, help="팔 실측 ↔ 첫 프레임 허용 [rad]")
    ap.add_argument("--object-topic", default=None, help="물체 pose(base) — sim 출발 자리와 비교")
    ap.add_argument("--object-tol", type=float, default=0.01, help="물체 xy 허용 [m]")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    traj = R.load_replay(args.npz, arm_src=args.arm_src, hand_src=args.hand_src)
    dry = R.build_plan(traj, traj.q[0], rate_scale=args.rate_scale, pub_hz=args.pub_hz)
    errs = summarize(traj, dry, args)
    if errs:
        print("[replay_rh] 거부:\n  " + "\n  ".join(errs))
        return 2
    if not args.execute:
        print("[replay_rh] 무발행 — 실기는 --execute(사용자 승인 뒤)")
        return 0
    return run(traj, args)


def run(traj: R.ReplayTraj, args) -> int:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    from replay_to_pd import replay_episode_id

    side = traj.side
    rclpy.init()
    node = rclpy.create_node("replay_traj_rh")
    arm_m: dict = {}
    hand_m: dict = {}
    pd = {"phase": None, "target": None}
    obj = {"p": None}
    node.create_subscription(JointState, ARM_STATE, lambda m: arm_m.update(zip(m.name, m.position)), qos_profile_sensor_data)
    node.create_subscription(JointState, HAND_STATE.format(side=side),
                             lambda m: hand_m.update(zip(m.name, m.position)), qos_profile_sensor_data)

    def on_pd(m) -> None:
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        pd.update(phase=d.get("phase"), target=d.get("target"))

    node.create_subscription(String, PD_STATUS.format(side=side), on_pd, 10)
    if args.object_topic:
        node.create_subscription(PoseStamped, args.object_topic,
                                 lambda m: obj.update(p=(m.pose.position.x, m.pose.position.y, m.pose.position.z)), 10)
    pub = node.create_publisher(JointState, TOPIC, 10)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ep_pub = node.create_publisher(String, EPISODE_TOPIC_SIDE.format(side=side), latched)

    arm_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    hand_names = list(traj.names[7:])
    t0 = time.time()
    while time.time() - t0 < 5.0 and not (all(n in arm_m for n in arm_names) and all(n in hand_m for n in hand_names)
                                          and pd["phase"] and (obj["p"] or not args.object_topic)):
        rclpy.spin_once(node, timeout_sec=0.1)
    missing = [n for n in arm_names if n not in arm_m] + [n for n in hand_names if n not in hand_m]
    if missing or not pd["phase"]:
        raise SystemExit(f"[replay_rh] 수신 없음: 실측 {missing[:4]} · pd status {pd['phase']!r}")
    if pd["phase"] not in MOVING:
        raise SystemExit(f"[replay_rh] pd {side} phase {pd['phase']} — TRACKING/RAMPING 에서만 시작한다(먼저 engage)")
    start = np.array([arm_m[n] for n in arm_names] + [hand_m[n] for n in hand_names])
    gap = float(np.abs(start[:7] - traj.q[0, :7]).max())
    if gap > args.max_start_gap:
        raise SystemExit(f"[replay_rh] 팔 실측이 첫 프레임에서 {gap:.3f} rad(> {args.max_start_gap}) — 진입 램프는 "
                         f"관절공간 직선이라 그 길을 검사하지 않았다. 홈(계약 arm_home)에서 시작한다")
    if args.object_topic:
        if traj.cup is None:
            raise SystemExit("[replay_rh] 궤적에 cup 열이 없다 — --object-topic 비교 불가")
        dxy = float(np.linalg.norm(np.asarray(obj["p"][:2]) - traj.cup[0, :2]))
        print(f"[replay_rh] 물체 {np.round(obj['p'], 3)} · sim 출발 {np.round(traj.cup[0, :3], 3)} · xy 차 {dxy * 1000:.1f} mm")
        if dxy > args.object_tol:
            raise SystemExit(f"[replay_rh] 물체가 sim 출발 자리에서 {dxy * 1000:.1f} mm(> {args.object_tol * 1000:.0f}) — "
                             f"open-loop 재생은 보정하지 않는다. 물체를 옮기거나 그 자리로 궤적을 다시 만든다")
    plan = R.build_plan(traj, start, rate_scale=args.rate_scale, pub_hz=args.pub_hz, hand_ramp_speed=args.hand_ramp_speed)
    ep_id = replay_episode_id()
    modes = dict(plan.mode_at)

    def event(kind: str, **kw) -> None:
        ep_pub.publish(String(data=json.dumps({"episode": ep_id, "event": kind, "t_ns": time.time_ns(),
                                               "object_anchor": None, "home_q": {}, **kw})))

    def send(k: int, q: np.ndarray, qd: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = f"{ep_id}:{k}"
        msg.name, msg.position = list(traj.names), q.tolist()
        msg.velocity, msg.effort = qd.tolist(), [0.0] * len(q)
        pub.publish(msg)

    rc, why = 0, "done"
    t_start = time.monotonic()
    try:
        for k in range(plan.q.shape[0]):
            if k in modes:
                event("hand_mode", hand_command=modes[k], reasons=[f"replay_traj_rh 구간 손 입력 {modes[k]}"])
            send(k, plan.q[k], plan.qd[k])
            rclpy.spin_once(node, timeout_sec=0.0)
            if pd["phase"] not in MOVING:
                rc, why = 3, f"pd phase {pd['phase']} (프레임 {k}/{plan.q.shape[0]})"
                break
            delay = t_start + (k + 1) * plan.pub_dt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        rc, why = 130, "Ctrl-C"
    event("stop", reasons=[f"replay_traj_rh {why}"])
    t1 = time.time()
    while time.time() - t1 < STOP_WAIT_S and pd["target"] != "internal" and rc == 0:
        send(plan.q.shape[0], plan.q[-1], np.zeros(plan.q.shape[1]))
        rclpy.spin_once(node, timeout_sec=plan.pub_dt)
    print(f"[replay_rh] {why} · rc {rc} · 램프 {plan.n_ramp} + 본 {plan.q.shape[0] - plan.n_ramp} 프레임 · pd {pd}")
    node.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""joint family 입력 점검 — 실기 토픽을 **구독만** 해서 정책 관측이 조립되는지 본다. 아무것도 발행하지 않는다.

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/joint_obs_probe.py \
        --contract deploy/policies/dg5f_m/cup_pick/right_m15/joint_contract.json --robot dg5f_m_right_real --seconds 5

보는 것: 소스별 수신 속도 · 결손 · stale, 계약 관절이 소스에 다 있는가, 손 속도가 오는가, 관측 조각(팔 · 손 ·
palm · 손끝)의 값, 지금 팔이 정책 시작 자세에서 얼마나 먼가. 컵 자세가 오면 키포인트 칸까지 채운다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from policy_control import codec  # noqa: E402
from policy_control.joint_contract import load_contract  # noqa: E402
from policy_control.joint_node import build_fk  # noqa: E402
from policy_control.joint_obs import Pose, keypoints  # noqa: E402
from policy_control.sources import SourceSet, load_robot_cfg, select_side  # noqa: E402

ROBOTS = HERE.parents[1] / "config" / "robots"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--robot", default="dg5f_m_right_real")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    import os
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        raise SystemExit("✗ ROS_DOMAIN_ID 가 비었거나 0 — 거부")

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    c = load_contract(args.contract)
    rpath = Path(args.robot) if Path(args.robot).is_file() else ROBOTS / f"{args.robot}.yaml"
    cfg = select_side(load_robot_cfg(rpath), c.side)
    src = SourceSet(cfg)
    counts: dict = {}

    rclpy.init()
    node = rclpy.create_node("joint_obs_probe")
    msgs = {"joint_state": JointState, "pose": PoseStamped}

    def cb_for(s):
        def cb(msg):
            now = time.monotonic()
            counts[s.name] = counts.get(s.name, 0) + 1
            if s.type == "joint_state":
                src.update_from_joint_state(s.name, codec.decode_joint_state(msg), now)
            else:
                src.update_from_pose(s.name, codec.decode_pose(msg), now)
        return cb

    for s in cfg.sources.values():
        if s.type in msgs and (s.role or s.name) in ("arm", "ee", "object"):
            node.create_subscription(msgs[s.type], s.topic, cb_for(s), qos_profile_sensor_data)
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    st = src.snapshot(time.monotonic())
    node.destroy_node()
    rclpy.shutdown()

    print(f"[probe] {c.task} · {c.side} · robot {rpath.name} · {args.seconds:.0f} s 구독(발행 없음)")
    for s in cfg.sources.values():
        if (s.role or s.name) in ("arm", "ee", "object"):
            n = counts.get(s.name, 0)
            flag = "stale" if s.name in st.stale else "missing" if s.name in st.missing else "ok"
            print(f"  {s.name:7s} {s.topic:40s} {n / args.seconds:6.1f} Hz  {flag}")
    problems = []
    if st.arm_q is None or st.ee_q is None:
        print("  ✗ 팔 또는 손 관절이 안 들어온다 — 드라이버를 볼 것")
        return 1
    arm_names = tuple(cfg.sources["arm"].joints)
    pos = {**dict(zip(arm_names, st.arm_q)), **dict(zip(st.ee_names, st.ee_q))}
    vel = {**dict(zip(arm_names, st.arm_qd if st.arm_qd is not None else [np.nan] * 7)),
           **dict(zip(st.ee_names, st.ee_qd if st.ee_qd is not None else [np.nan] * len(st.ee_names)))}
    missing = [n for n in (*c.arm_joints, *c.hand_joints, *c.welded) if n not in pos]
    if missing:
        problems.append(f"계약 관절이 소스에 없다: {missing}")
    if st.ee_qd is None or not np.all(np.isfinite(st.ee_qd)):
        problems.append("손 관절 속도가 안 온다(hand_qd 는 정책 입력)")
    arm = np.array([pos[n] for n in c.arm_joints])
    hand = np.array([pos[n] for n in c.hand_joints])
    fk = build_fk(c)
    pose = fk.palm_pose(arm, hand)
    err = np.abs(arm - np.asarray(c.arm_reset))
    print(f"  팔 관절    {np.round(arm, 3).tolist()}")
    print(f"  팔 속도    {np.round([vel[n] for n in c.arm_joints], 3).tolist()}")
    print(f"  손(관측순) {np.round([pos[n] for n in c.hand_obs_order], 3).tolist()}")
    print(f"  손 속도    max |qd| {np.nanmax(np.abs([vel[n] for n in c.hand_obs_order])):.3f} rad/s")
    for n, w in c.welded.items():
        print(f"  용접 {n} 실측 {pos.get(n, float('nan')):+.3f} (보낼 값 {w:+.3f})")
    print(f"  palm {np.round(pose.palm_pos, 4).tolist()} · 손끝−palm 최대 {np.linalg.norm(pose.tips - pose.palm_pos, axis=1).max():.3f} m")
    print(f"  시작 자세까지 최대 {err.max():.3f} rad ({c.arm_joints[int(err.argmax())]}) — start 허용 0.15")
    if st.object_pos is not None:
        kp = keypoints(c, Pose(np.asarray(st.object_pos), np.asarray(st.object_quat)))
        print(f"  컵 {np.round(st.object_pos, 4).tolist()} · 키포인트−palm 최소 {np.linalg.norm(kp - pose.palm_pos, axis=1).min():.3f} m")
    else:
        problems.append("컵 자세(object)가 안 온다 — 인지(카메라 · FP++ · 수신기 · object_pose_node · 목 상태)가 필요하다")
    for p in problems:
        print(f"  ✗ {p}")
    if not problems:
        print("  ✓ 정책 입력 소스가 모두 들어온다")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""fake rh_cap 플랜트(10.12 locking — rh_cap_node 리허설 전용, fake 도메인만). 실기 손 · 팔 · FP++ 대신 같은 토픽으로 선다.

  구독  /policy_control/joint_target (JointState) — 오른팔 7 · 오른손 6 목표
  발행  /joint_states(오른팔, 목표로 1차 지연 추종) · /hand_right/joint_states(손 6, 같은 추종) · /hand_right/tip_forces(5) ·
        /hand_right/joint_forces(6) · /objects/cap/pose · /objects/receiver/pose(latched)
  쥠 흉내: 네 손가락 평균 닫힘이 --grip-close 를 넘으면 엄지 · 검지 손끝 3 N(쥔 판정 참). 쥔 뒤 캡은 손바닥 아래에 붙어 따라간다
  (seat 모드는 처음부터 쥔 채). 손바닥 위치는 policy_control FK(UrdfChainFK)로 푼다.

    ROS_DOMAIN_ID=97 python3 scripts/fakes/fake_rh_cap_plant.py --contract <rh_cap_contract.json> --arm-q -0.6,0.2,0.65,1.76,0.38,0.2,0.66
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_S2R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_S2R / "deploy" / "policy_control"))

from policy_control import rh_cap as C  # noqa: E402
from policy_control.ros_domain import REAL_DOMAIN  # noqa: E402  config/ros_domain.env 한 곳(10.30)
from policy_control.fk_numpy import UrdfChainFK  # noqa: E402
from policy_control.contract_assets import ASSETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract", required=True)
    ap.add_argument("--arm-q", required=True, help="시작 팔 7 (rad, 쉼표)")
    ap.add_argument("--hz", type=float, default=250.0)
    ap.add_argument("--tau", type=float, default=0.05, help="1차 추종 시정수 [s]")
    ap.add_argument("--grip-close", type=float, default=0.5, help="네 손가락 평균 닫힘(rad) — 넘으면 쥠")
    ap.add_argument("--cap-below", type=float, default=0.06, help="시작 캡 = 손바닥 아래 이만큼 [m]")
    args = ap.parse_args()
    dom = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    if dom in (0, REAL_DOMAIN):
        raise SystemExit(f"fake 도메인에서만(지금 ROS_DOMAIN_ID={dom})")
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    c = C.load_contract(args.contract)
    s = c.side()
    fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
    arm = np.array([float(v) for v in args.arm_q.split(",")])
    hand = np.asarray(s.hand_grip if c.cap_phase == "seat" else s.hand_open, float) * (0.8 if c.cap_phase == "seat" else 1.0)
    tgt = {"arm": arm.copy(), "hand": hand.copy()}
    pose = fk.palm_pose(arm, hand)
    R0 = pose.extra["palm_rot"]
    cap = pose.palm_pos + np.array([0.0, 0.0, -args.cap_below])
    rel = R0.T @ (cap - pose.palm_pos)
    rcv = pose.palm_pos + np.array([0.0, 0.0, -args.cap_below - c.seat_offset]) if c.cap_phase == "seat" else np.array([0.30, 0.12, 0.41])
    held = c.cap_phase == "seat"

    rclpy.init()
    node = rclpy.create_node("fake_rh_cap_plant")
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    p_arm = node.create_publisher(JointState, "/joint_states", 10)
    p_hand = node.create_publisher(JointState, "/hand_right/joint_states", 10)
    p_tip = node.create_publisher(Float64MultiArray, "/hand_right/tip_forces", 10)
    p_jf = node.create_publisher(Float64MultiArray, "/hand_right/joint_forces", 10)
    p_cap = node.create_publisher(PoseStamped, "/objects/cap/pose", 10)
    p_rcv = node.create_publisher(PoseStamped, "/objects/receiver/pose", latched)

    def on_target(msg) -> None:
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in s.arm_joints):
            tgt["arm"] = np.array([d[j] for j in s.arm_joints])
        if all(j in d for j in s.hand_joints):
            tgt["hand"] = np.array([d[j] for j in s.hand_joints])

    node.create_subscription(JointState, "/policy_control/joint_target", on_target, 50)

    def pose_msg(p, q=(1.0, 0.0, 0.0, 0.0)) -> PoseStamped:
        m = PoseStamped()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in p)
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (float(v) for v in q)
        return m

    p_rcv.publish(pose_msg(rcv))
    st = {"arm": arm, "hand": hand, "cap": cap, "held": held, "n": 0}
    dt = 1.0 / args.hz
    a = dt / (args.tau + dt)
    fingers = [i for i, j in enumerate(s.hand_joints) if "thumb" not in j]

    def tick() -> None:
        st["arm"] = st["arm"] + a * (tgt["arm"] - st["arm"])
        if not st["held"]:                         # 쥔 뒤 손가락은 캡에 막혀 멈춘다(실기 막힘 신호 쥠 멈춤 · 쥔 판정이 이걸 본다)
            st["hand"] = st["hand"] + a * (tgt["hand"] - st["hand"])
        close = float(np.mean(np.abs(st["hand"][fingers] - np.asarray(s.hand_open)[fingers])))
        grip = close > args.grip_close
        P = fk.palm_pose(st["arm"], st["hand"])
        if grip and not st["held"]:
            st["held"] = True
            node.get_logger().info(f"fake 쥠 — 닫힘 {close:.2f} rad, 캡이 손바닥에 붙는다")
        if st["held"]:
            st["cap"] = P.palm_pos + P.extra["palm_rot"] @ rel
        now = node.get_clock().now().to_msg()
        js = JointState(name=[f"openarm_right_joint{i}" for i in range(1, 8)],   # 실기 소스 이름(robot_control 프로필 source)
                         position=[float(v) for v in st["arm"]], velocity=[0.0] * 7)
        js.header.stamp = now
        hs = JointState(name=list(s.hand_joints), position=[float(v) for v in st["hand"]], velocity=[0.0] * 6)
        hs.header.stamp = now
        p_arm.publish(js)
        p_hand.publish(hs)
        p_tip.publish(Float64MultiArray(data=[3.0, 3.0, 0.0, 0.0, 0.0] if st["held"] else [0.0] * 5))
        p_jf.publish(Float64MultiArray(data=[0.0] * 6))
        st["n"] += 1
        if st["n"] % 4 == 0:                       # 약 60 Hz
            p_cap.publish(pose_msg(st["cap"]))
        if st["n"] % int(args.hz) == 0:
            p_rcv.publish(pose_msg(rcv))
            node.get_logger().info(f"palm z {P.palm_pos[2]:.3f} · cap {np.round(st['cap'], 3).tolist()} · 쥠 {st['held']} · 닫힘 {close:.2f}")

    node.create_timer(dt, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

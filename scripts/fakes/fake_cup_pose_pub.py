#!/usr/bin/env python3
"""합성 /cup_pose 발행 (카메라 무관 obs 검증용).

정지: --x 0.40 --y -0.15 --z 0.38
원운동(움직이는 컵 모사): --orbit 0.05,0.2  (반경 5cm, 0.2Hz, 중심=xyz)
★10.20 손 안 컵 따라가기(1-B 리허설): --follow-side right — /fake/hand_<side>/cup(Bool, fake_rh56f1_hand 의 '손 안 컵' 과 같은
  토픽)이 참이 되는 순간의 손바닥(배포 계약 FK, /joint_states 실측) 기준 자리를 잡아 두고, 참인 동안 손바닥을 따라 옮긴다.
  거짓이 되면 그 자리에 둔다(놓은 컵). 들기 판정(FP++ 컵 상승)을 fake 에서도 그대로 시험한다.
"""
from __future__ import annotations
import argparse, math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class Follow:
    """손 안 컵: 참이 되는 순간 컵 위치를 손바닥 틀로 옮겨 두고, 참인 동안 손바닥 · 그 오프셋으로 컵 위치를 낸다."""

    def __init__(self, node, args):
        import sys
        from pathlib import Path
        import numpy as np
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
        repo = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo / "deploy" / "policy_control"))
        from policy_control import rh_aglt as A
        from policy_control.contract_assets import ASSETS
        from policy_control.fk_numpy import UrdfChainFK
        cpath = args.contract or str(repo / f"config/hand/rh56f1_{args.follow_side}_hand.json")   # ★10.20 옛 approach 계약 고정본(1-B 와 같은 것)
        c = A.load_contract(cpath)
        s = c.sides[next(iter(c.sides))]
        self.np, self.log = np, node.get_logger()
        self.fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
        self.arm, self.hand_open = list(s.arm_joints), np.asarray(s.hand_open, float)
        # /joint_states 는 컨트롤러 이름 · 부호(openarm_right_joint1 …) — fake_arm_bridge 와 같은 로봇 프로필로 canonical 에 맞춘다
        from policy_control.sources import load_profile, load_robot_cfg
        prof = load_profile(load_robot_cfg(repo / args.robot_yaml).joint_profiles)
        self.src = [str(prof[j]["source"]) for j in self.arm]
        self.sign = np.array([float(prof[j]["sign"]) for j in self.arm])
        self.q = None
        self.on = False
        self.off = None                # 손바닥 틀의 컵 위치
        self.last = None               # 마지막으로 낸 위치(놓으면 그 자리)
        node.create_subscription(JointState, "/joint_states", self._js, 10)
        node.create_subscription(Bool, f"/fake/hand_{args.follow_side}/cup", self._cup, 10)

    def _js(self, msg) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(j in idx for j in self.src):
            self.q = self.np.array([msg.position[idx[j]] for j in self.src], float) * self.sign
        elif all(j in idx for j in self.arm):
            self.q = self.np.array([msg.position[idx[j]] for j in self.arm], float)

    def _palm(self):
        np = self.np
        P = self.fk.palm_pose(self.q, self.hand_open)
        w, x, y, z = (float(v) for v in P.palm_quat)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return R, np.asarray(P.palm_pos, float)

    def _cup(self, msg) -> None:
        on = bool(msg.data)
        if on and not self.on and self.q is not None and self.last is not None:
            R, p = self._palm()
            self.off = R.T @ (self.np.asarray(self.last, float) - p)
            self.log.info(f"손 안 컵 켬 — 손바닥 틀 오프셋 {self.np.round(self.off, 3).tolist()}")
        elif not on and self.on:
            self.log.info("손 안 컵 끔 — 그 자리에 둔다")
        self.on = on

    def position(self, static):
        if self.on and self.off is not None and self.q is not None:
            R, p = self._palm()
            self.last = p + R @ self.off
        elif self.last is None or self.off is None:
            self.last = self.np.asarray(static, float)
        return self.last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.40)
    ap.add_argument("--y", type=float, default=-0.15)
    ap.add_argument("--z", type=float, default=0.38)
    ap.add_argument("--frame", default="base_link")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--orbit", default="", help="r,hz (예: 0.05,0.2). 빈값=정지")
    ap.add_argument("--latched", action="store_true", help="transient_local(실기 cup_holder_pose_node 와 같은 QoS — 홀더용)")
    ap.add_argument("--topic", default="/cup_pose",
                    help="발행 토픽. policy_control 은 /objects/<name>/pose 를 구독한다")
    ap.add_argument("--follow-side", choices=("right", "left"), default=None,
                    help="/fake/hand_<side>/cup 이 참인 동안 그 손바닥을 따라간다(손 안 컵)")
    ap.add_argument("--contract", default="", help="손바닥 FK 계약(기본 config/hand/rh56f1_<side>_hand.json)")
    ap.add_argument("--robot-yaml", default="deploy/policy_control/config/robots/rh56f1_bi_fake.yaml",
                    help="/joint_states 이름 · 부호 프로필(저장소 기준)")
    args = ap.parse_args()
    r = hz = 0.0
    if args.orbit:
        r, hz = (float(v) for v in args.orbit.split(","))

    rclpy.init()
    node = Node("fake_cup_pose_pub")
    if args.latched:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    else:
        qos = 10
    pub = node.create_publisher(PoseStamped, args.topic, qos)
    t0 = node.get_clock().now()
    follow = Follow(node, args) if args.follow_side else None

    def tick():
        t = (node.get_clock().now() - t0).nanoseconds * 1e-9
        dx = r * math.cos(2 * math.pi * hz * t)
        dy = r * math.sin(2 * math.pi * hz * t)
        pos = (args.x + dx, args.y + dy, args.z)
        if follow is not None:
            pos = follow.position(pos)
        m = PoseStamped()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = args.frame
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
        m.pose.orientation.w = 1.0
        pub.publish(m)

    node.create_timer(1.0 / args.rate, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

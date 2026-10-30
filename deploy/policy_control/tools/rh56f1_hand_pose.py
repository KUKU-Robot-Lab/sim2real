#!/usr/bin/env python3
"""RH56F1 한 손을 pd 설정의 저장 자세로 — 팔 pd 없이 손 드라이버(/hand_<side>/angle_set)에 바로 보낸다. 실기 명령.

10.13 사용자: "드라이버 켰다가 주먹 쥐게 하고, 바로 다시 꺼야해. 바로 콘솔로 확인하는 절차 진행해야하니까"
— 콘솔 전에 손만 주먹(pd yaml `hand_path_pose`, 홈 경로를 계획한 손)으로 만들어 두고 드라이버를 내린다.
pd/hand_path(episode_ctl --only pd_hand_path)는 팔 engage 가 있어야 해서 이 순간에는 못 쓴다.

  bash deploy/policy_control/tools/rh56f1_hand_up.sh right                                   # 드라이버(OP)
  ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/rh56f1_hand_pose.py --side right             # 계획만
  ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/rh56f1_hand_pose.py --side right --execute   # ★보낸다
  bash deploy/policy_control/tools/rh56f1_hand_down.sh right                                 # 드라이버 내림

보내는 값은 변환표(config/rh56f1_hand_map.yaml)로 rad → 레지스터. 확인 안 된 축은 -1(그대로).
보낸 뒤 --hold 초 기다려 angle_actual 을 읽고 축마다 목표와의 차이를 찍는다(막히면 실패로 돌려준다).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import rh56f1_map  # noqa: E402

PD_CONFIG = Path(__file__).resolve().parents[1] / "config" / "pd_rh56f1.yaml"
READ_TIMEOUT_S = 3.0
REACH_TOL_RAD = 0.15          # 도착 판정 — hand_to_path_pose 와 같다


def pose_from_pd(path: Path, side: str, key: str) -> dict[str, float]:
    """pd yaml 의 <key>.<side> → {축 이름(thumb_1 …): rad}."""
    raw = yaml.safe_load(Path(path).read_text())
    if key not in raw or side not in raw[key]:
        raise SystemExit(f"{path} 에 {key}.{side} 가 없다")
    prefix = f"{side[0]}_hj_"
    out = {}
    for name, v in raw[key][side].items():
        if not name.startswith(prefix):
            raise SystemExit(f"{key}.{side} 의 {name} 가 {prefix}* 가 아니다")
        out[name[len(prefix):]] = float(v)
    return out


def plan(hmap, side: str, pose: dict[str, float]) -> list[int]:
    missing = [j for j in hmap.joint_order if j not in pose]
    if missing:
        raise SystemExit(f"자세에 축이 빠졌다: {missing}")
    return hmap.to_register([pose[j] for j in hmap.joint_order], side=side)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--key", default="hand_path_pose", help="pd yaml 의 자세 키(기본 hand_path_pose = 주먹)")
    ap.add_argument("--pd-config", type=Path, default=PD_CONFIG)
    ap.add_argument("--hold", type=float, default=3.0, help="보낸 뒤 기다리는 시간 [s]")
    ap.add_argument("--execute", action="store_true", help="★실기에 보낸다(없으면 계획만)")
    ap.add_argument("--map", default=str(rh56f1_map.DEFAULT_PATH))
    args = ap.parse_args(argv)
    hmap = rh56f1_map.load(args.map)
    pose = pose_from_pd(args.pd_config, args.side, args.key)
    cmd = plan(hmap, args.side, pose)
    print(f"[{args.side}] {args.key} {pose}\n  보낼 레지스터(슬롯 순) {cmd}")
    if not args.execute:
        print("  (계획만 — --execute 없이는 보내지 않는다)")
        return 0

    import rclpy
    from rclpy.node import Node
    from rh56f1_interfaces.msg import GetAngleAct1, SetAngle1

    rclpy.init()
    node = Node(f"rh56f1_hand_pose_{args.side}")
    ns = f"/hand_{args.side}"
    box: dict = {"reg": None}
    node.create_subscription(GetAngleAct1, f"{ns}/angle_actual", lambda m: box.update(reg=list(m.joint_values)), 10)
    pub = node.create_publisher(SetAngle1, f"{ns}/angle_set", 10)

    def spin(sec: float, until=lambda: False) -> None:
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end and not until():
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        spin(READ_TIMEOUT_S, lambda: box["reg"] is not None)
        if box["reg"] is None:
            print(f"✗ {ns}/angle_actual 이 {READ_TIMEOUT_S} s 안에 안 온다 — 드라이버를 먼저(rh56f1_hand_up.sh)")
            return 1
        print(f"  지금 {box['reg']}")
        spin(READ_TIMEOUT_S, lambda: pub.get_subscription_count() >= 1)
        if pub.get_subscription_count() < 1:
            print(f"✗ {ns}/angle_set 을 구독하는 드라이버가 안 보인다 — 보내지 않았다")
            return 1
        msg = SetAngle1()
        msg.hand_id = 0
        msg.joint_values = [int(v) for v in cmd]
        pub.publish(msg)
        box["reg"] = None
        spin(args.hold)
        spin(READ_TIMEOUT_S, lambda: box["reg"] is not None)
        after = box["reg"]
        if after is None:
            print("✗ 보낸 뒤 angle_actual 이 안 온다")
            return 1
        got = hmap.to_rad(after, side=args.side)
        rc = 0
        for j, a, q in zip(hmap.joint_order, hmap.axes_of(args.side), got):
            if cmd[a.slot] == rh56f1_map.LEAVE:
                print(f"  {j:9s} 그대로(확인 안 된 축)")
                continue
            err = q - pose[j]
            bad = abs(err) > REACH_TOL_RAD
            rc |= bad
            print(f"  {'✗' if bad else '✓'} {j:9s} 목표 {pose[j]:+.3f} 실측 {q:+.3f} rad (레지스터 {after[a.slot]})")
        print("  ✓ 도착" if not rc else f"  ✗ {REACH_TOL_RAD} rad 밖 축이 있다 — 막혔거나 덜 갔다")
        return int(rc)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())

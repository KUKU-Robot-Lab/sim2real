#!/usr/bin/env python3
"""목 드라이버 — 목 시리얼 포트를 **혼자** 맡아 각도를 발행하고 이동 요청도 받는다(2026-10-15 사용자 결정).

`global_camera_extrinsics.yaml` 의 `camera:` 블록은 **한 목 자세의 정적 스냅샷**이라
목이 돌면 통째로 틀린다(pan 15° 면 카메라가 11 mm 이동 + 큰 회전 → 0.6 m 컵에서 수 cm).
이 노드가 목 각도를 흘려 주면 릴레이가

    T_base_cam(pan, tilt) = T_base_neck(pan, tilt) ∘ T_neck_cam

로 매번 다시 만든다.

★**URDF 규약(라디안)으로 발행한다.** 토픽이 `head_j_pan`/`head_j_tilt` 라는 URDF 관절
이름을 쓰므로 그게 표준 의미다. 인코더 각을 그대로 실으면 pan 이 반대로 흐른다.
변환은 `head_fk_chain.urdf_from_encoder` 가 한다.

★**포트 단독 소유자다.** 읽기(20 Hz 발행)와 쓰기(이동)를 한 스레드에서 번갈아 하므로 "Incorrect status packet" 이
나지 않는다. 다른 프로세스가 포트를 열면 `port_lock` 이 거절한다 — `head_home.py --execute` 같은 직접 쓰기 도구를
같이 돌리면 안 된다(예전 문구 "같이 돌려도 안전" 은 틀렸다: 읽기도 같은 버스를 쓴다).
목을 움직이려면 `scripts/head_move.py` 가 `/head/move_request` 로 요청한다(규약은 `head_motion.py`).
이동 중에도 각도 발행은 끊기지 않는다. 한 번에 한 요청만 받고 이동 중 새 요청은 거절한다.

    python head_joint_publisher.py                       # 20 Hz 로 /head/joint_states + 이동 요청 대기
    python head_joint_publisher.py --dry-run             # 하드웨어 없이 계획만
"""

from __future__ import annotations

import argparse
import math
import sys

from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from head_compliant_hold import (
    DEFAULT_BAUD,
    CompliantController,
    autodetect_port,
    discover_ports,
    parse_motor_ids,
    tick_to_deg,
)
from head_driver_core import HeadDriver
from head_fk_chain import urdf_from_encoder
from head_home import DEFAULT_CONFIG, load_head_home
import head_motion

JOINT_NAMES = ["head_j_pan", "head_j_tilt"]
DEFAULT_TOPIC = "/head/joint_states"
DEFAULT_RATE_HZ = 20.0


def joint_state_fields(pan_encoder_deg: float, tilt_encoder_deg: float
                       ) -> tuple[list[str], list[float]]:
    """(이름, 위치[rad]) — **URDF 규약**으로 변환해 실는다."""
    pan_urdf, tilt_urdf = urdf_from_encoder(pan_encoder_deg, tilt_encoder_deg)
    return list(JOINT_NAMES), [math.radians(pan_urdf), math.radians(tilt_urdf)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--ids", default="1,2", help="pan,tilt 순서")
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="기준 자세·게인(head_home.yaml). 이동 요청이 이 게인 순서로 적용된다")
    p.add_argument("--max-tilt", type=float, default=head_motion.MAX_TILT_DEG, help="이동 요청에서 허용하는 tilt 상한(인코더 deg)")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        ids = parse_motor_ids(args.ids)
        if len(ids) != 2:
            raise ValueError(f"pan,tilt 두 개여야 한다: {ids}")
        port = args.port or autodetect_port(discover_ports())
    except (ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    print(f"포트 {port} @ {args.baud} · ids {ids} → {args.topic} @ {args.rate_hz} Hz")
    print(f"관절 이름 {JOINT_NAMES} · 단위 라디안 · **URDF 규약**(pan 부호 반전 적용)")
    print(f"포트 단독 소유 · 이동 요청 {head_motion.REQUEST_TOPIC} → 상태 {head_motion.STATUS_TOPIC} · tilt 상한 {args.max_tilt}°")
    if args.dry_run:
        names, pos = joint_state_fields(0.0, -20.0)
        print(f"\n예시(인코더 pan 0 / tilt -20): {names} = "
              f"{[round(v, 6) for v in pos]} rad")
        print("--dry-run — 포트를 열지 않았다")
        return 0

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    try:
        config = load_head_home(Path(args.config))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"❌ 설정을 읽을 수 없다: {exc}", file=sys.stderr)
        return 2
    try:
        controller = CompliantController(port, args.baud)
    except Exception as exc:  # noqa: BLE001 - PortBusy 포함: 다른 프로세스가 포트를 쥐고 있다
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    rclpy.init()
    node = Node("head_joint_publisher")
    pub = node.create_publisher(JointState, args.topic, 10)
    status_pub = node.create_publisher(String, head_motion.STATUS_TOPIC, 10)
    count = {"n": 0}

    def publish_angles(pan: float, tilt: float) -> None:
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name, msg.position = joint_state_fields(pan, tilt)
        pub.publish(msg)
        count["n"] += 1
        if count["n"] % (round(args.rate_hz) * 10) == 1:
            node.get_logger().info(f"#{count['n']} 인코더 pan {pan:+.2f}° tilt {tilt:+.2f}°")

    driver = HeadDriver(controller, config, pan_id=ids[0], tilt_id=ids[1], publish_angles=publish_angles,
                        publish_status=lambda text: status_pub.publish(String(data=text)), max_tilt_deg=args.max_tilt,
                        log=node.get_logger().info)
    node.create_subscription(String, head_motion.REQUEST_TOPIC, lambda m: driver.on_request(m.data), 10)
    node.create_timer(1.0 / args.rate_hz, driver.tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        controller.close()
        print("포트 닫음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

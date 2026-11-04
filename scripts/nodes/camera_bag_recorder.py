#!/usr/bin/env python3
"""머리 카메라 토픽을 낮은 빈도로 rosbag2(sqlite3)에 쓴다 — 에피소드 bag 의 세 번째 묶음 camera(11.04 사용자).

    python3 scripts/nodes/camera_bag_recorder.py -o logs/bags/<시각>_episode_<이름>/camera --hz 10 <토픽...>
    (deploy/policy_control/tools/rh56f1_record.sh 가 CAMERA='<토픽...>' 일 때 팔 · 손 bag 옆에 띄운다. SIGINT 에 마무리)

ros2 bag record(humble)는 빈도를 못 낮추고 topic_tools 도 깔려 있지 않다. 그래서 받은 직렬화 바이트를 RateGate 로 걸러
그대로 쓴다(디코드 · 재인코딩 없음 → 이 프로세스 CPU 는 작다). 압축(JPEG · PNG)은 카메라 노드의 image_transport 가 한다 —
구독자가 하나라도 있으면 그 노드가 매 프레임(30 Hz) 인코딩하므로, 여기서 빈도를 낮춰도 카메라 쪽 인코딩은 줄지 않는다.
bag 시각 = 받은 시각(system clock, ros2 bag record 와 같은 규약)이라 같은 PC 의 arm · hand bag 과 바로 맞는다.
구독은 best effort — 이 기록기가 느려도 카메라(reliable 발행)에 재전송 부담을 주지 않고 프레임만 버린다.
아직 안 보이는 토픽(예: /head_depth/image)은 1 s 마다 다시 찾아 나타나면 붙는다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_recorder import RateGate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("-o", "--out", type=Path, required=True, help="bag 폴더(없어야 한다 — rosbag2 가 만든다)")
    ap.add_argument("--hz", type=float, default=10.0, help="토픽마다 남길 최대 빈도")
    ap.add_argument("topics", nargs="+")
    args = ap.parse_args(argv)
    if args.hz <= 0:
        ap.error("--hz 는 0 보다 커야 한다")

    import rclpy
    import rosbag2_py
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message

    rclpy.init()
    node = Node("camera_bag_recorder")
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.out), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    wanted, subs, counts = set(args.topics), {}, {t: 0 for t in args.topics}

    def on_msg(topic: str):
        gate = RateGate(args.hz)

        def cb(data: bytes) -> None:
            t = node.get_clock().now().nanoseconds
            if gate.take(t * 1e-9):
                writer.write(topic, data, t)
                counts[topic] += 1
        return cb

    def discover() -> None:
        for name, types in node.get_topic_names_and_types():
            if name in wanted and name not in subs:
                writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=types[0], serialization_format="cdr"))
                subs[name] = node.create_subscription(get_message(types[0]), name, on_msg(name),
                                                      qos_profile_sensor_data, raw=True)
                print(f"[camera_bag] 구독 {name} ({types[0]})", flush=True)

    discover()
    node.create_timer(1.0, discover)
    print(f"[camera_bag] 기록 시작 → {args.out} · {args.hz:g} Hz · 토픽 {len(wanted)}개", flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        writer = None                   # humble SequentialWriter 는 close() 가 없다 — 놓을 때 metadata.yaml 을 쓴다
        rclpy.try_shutdown()
    print("[camera_bag] 끝 — " + " · ".join(f"{t} {n}" for t, n in counts.items()), flush=True)
    missing = [t for t in args.topics if t not in subs]
    if missing:
        print(f"[camera_bag] 한 번도 안 보인 토픽: {missing}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""팔 A/B 분석용 구독 전용 기록기. 명령 발행·서비스 호출·CAN 송신 없음.

ROS 환경/domain/lan_profile을 호출자가 설정한 뒤 실행한다.
  .venv/bin/python logs/analysis/fk_board_jog_1105/right_2_audit_1106/live_probe/record_ros_readonly.py \
      --side right --seconds 30 --out logs/analysis/fk_board_jog_1105/right_2_audit_1106/live_probe/runs/ab_off

각 수신 메시지마다 host wall/monotonic ns, 원본 header stamp/frame_id,
전체 ROS 메시지를 JSONL과 pickle 없는 NPZ로 보존한다. 수신시각은 동기 샘플이
아니므로 분석 시 stamp/recv 시각으로 정렬해야 한다. 알려진 토픽이 없어도 대기한다.
--selftest는 ROS 없이 직렬화·저장 계약만 검사한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np


def topic_spec(side: str) -> dict[str, str]:
    arm = f"/{side}_forward"
    hand = f"/hand_{side}"
    return {
        "/joint_states": "sensor_msgs/msg/JointState",
        "/robot_description": "std_msgs/msg/String",
        "/head/joint_states": "sensor_msgs/msg/JointState",
        "/hand_left/joint_states": "sensor_msgs/msg/JointState",
        "/hand_right/joint_states": "sensor_msgs/msg/JointState",
        "/pd_state_broadcaster/joint_states": "sensor_msgs/msg/JointState",
        "/pd_temp_broadcaster/dynamic_joint_states": "control_msgs/msg/DynamicJointState",
        f"{hand}/joint_states": "sensor_msgs/msg/JointState",
        f"{hand}/joint_forces_zeroed": "std_msgs/msg/Float64MultiArray",
        f"{hand}/joint_forces": "std_msgs/msg/Float64MultiArray",
        f"{hand}/touch_data": "rh56f1_interfaces/msg/TouchData1",
        f"{hand}/tip_forces_zeroed": "std_msgs/msg/Float64MultiArray",
        f"{hand}/admittance_offset": "std_msgs/msg/Float64MultiArray",
        f"{hand}/ecat_status": "std_msgs/msg/String",
        "/policy_control/joint_target": "sensor_msgs/msg/JointState",
        f"/policy_control/pd_{side}/applied": "sensor_msgs/msg/JointState",
        f"/policy_control/status/pd_{side}": "std_msgs/msg/String",
        f"/policy_control/pd_{side}/diagnostics": "std_msgs/msg/String",
        f"/policy_control/pd_{side}/contact": "std_msgs/msg/Bool",
        f"/policy_control/pd_{side}/contact_guard": "std_msgs/msg/String",
        f"{arm}_position_controller/commands": "std_msgs/msg/Float64MultiArray",
        f"{arm}_velocity_controller/commands": "std_msgs/msg/Float64MultiArray",
        f"{arm}_effort_controller/commands": "std_msgs/msg/Float64MultiArray",
        # 에피소드 경계·비상 정지(pd_node.py:221-226 구독과 같은 이름, 래치 QoS)
        "/policy_control/episode": "std_msgs/msg/String",
        f"/policy_control/{side}/episode": "std_msgs/msg/String",
        "/policy_control/estop": "std_msgs/msg/Bool",
    }


# 발행 쪽이 TRANSIENT_LOCAL 로 래치하는 토픽. 늦게 붙어도 마지막 값을 받는다.
LATCHED_TOPICS = ("/robot_description", "/policy_control/episode", "/policy_control/estop")


def is_latched(topic: str) -> bool:
    return topic in LATCHED_TOPICS or topic.endswith("/episode")


def event(topic: str, message: dict, *, wall_ns: int, mono_ns: int) -> dict:
    header = message.get("header", {})
    stamp = header.get("stamp")
    stamp_ns = None if stamp is None else int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])
    return {"topic": topic, "recv_wall_ns": wall_ns, "recv_mono_ns": mono_ns,
            "header_stamp_ns": stamp_ns, "frame_id": header.get("frame_id"), "message": message}


def save_npz(path: Path, rows: list[dict]) -> None:
    # Unicode JSON retains arbitrary ROS message fields without object arrays/pickle.
    with path.open("xb") as stream:
        np.savez(stream, recv_wall_ns=np.array([r["recv_wall_ns"] for r in rows], dtype=np.int64),
                 recv_mono_ns=np.array([r["recv_mono_ns"] for r in rows], dtype=np.int64),
                 header_stamp_ns=np.array([-1 if r["header_stamp_ns"] is None else r["header_stamp_ns"]
                                           for r in rows], dtype=np.int64),
                 topic=np.array([r["topic"] for r in rows], dtype=str),
                 event_json=np.array([json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows], dtype=str))


def selftest() -> None:
    r = event("/joint_states", {"header": {"stamp": {"sec": 7, "nanosec": 12}, "frame_id": "base"},
                                "name": ["j1"], "position": [0.2]}, wall_ns=100, mono_ns=50)
    assert r["header_stamp_ns"] == 7_000_000_012 and r["frame_id"] == "base"
    r2 = event("/status", {"data": "ok"}, wall_ns=101, mono_ns=51)
    assert r2["header_stamp_ns"] is None
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / "test.npz"
        save_npz(p, [r, r2])
        with np.load(p, allow_pickle=False) as data:
            assert data["header_stamp_ns"].tolist() == [7_000_000_012, -1]
            assert json.loads(data["event_json"][0]) == r
        try:
            save_npz(p, [])
        except FileExistsError:
            pass
        else:
            raise AssertionError("must refuse overwriting")
    assert topic_spec("right")["/policy_control/pd_right/applied"] == "sensor_msgs/msg/JointState"
    print("selftest ok (no ROS initialized)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", type=Path, help="new output directory; existing directory refused")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if args.out is None or not 0 < args.seconds <= 3600:
        parser.error("--out and 0 < --seconds <= 3600 required")
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        parser.error("ROS_DOMAIN_ID must be explicitly nonzero")
    out = args.out.resolve()
    workspace = Path.home() / "rl_ws"
    if not out.is_relative_to(workspace):
        parser.error(f"output must remain under {workspace}")

    # Imports are intentionally after --selftest and argument checks.
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rosidl_runtime_py.utilities import get_message

    spec = topic_spec(args.side)
    message_types = {topic: get_message(typename) for topic, typename in spec.items()}
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": "readonly_ros_probe_v1", "side": args.side, "seconds": args.seconds,
                "start_wall_ns": time.time_ns(), "ROS_DOMAIN_ID": os.environ["ROS_DOMAIN_ID"],
                "FASTRTPS_DEFAULT_PROFILES_FILE": os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"),
                "topics": spec, "commands_sent": 0,
                "time_note": "per-topic callbacks, not synchronized sampling; align timestamps in analysis"}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(f"readonly_probe_{args.side}_{os.getpid()}", enable_rosout=False,
                            start_parameter_services=False)
    qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
    rows: list[dict] = []
    counts: Counter = Counter()
    try:
        with (out / "events.jsonl").open("x", buffering=1_048_576) as stream:
            def callback(topic, msg):
                wall, mono = time.time_ns(), time.monotonic_ns()
                row = event(topic, message_to_ordereddict(msg), wall_ns=wall, mono_ns=mono)
                rows.append(row)
                counts[topic] += 1
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

            for topic, message_type in message_types.items():
                selected_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                          durability=DurabilityPolicy.TRANSIENT_LOCAL) if is_latched(topic) else qos
                node.create_subscription(message_type, topic, lambda msg, t=topic: callback(t, msg), selected_qos)
            print(f"read-only recording -> {out}", flush=True)
            deadline, last_flush = time.monotonic() + args.seconds, time.monotonic()
            while rclpy.ok() and not stop and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if time.monotonic() - last_flush >= 0.5:
                    stream.flush()
                    last_flush = time.monotonic()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        save_npz(out / "events.npz", rows)
        metadata.update(end_wall_ns=time.time_ns(), counts=dict(counts), n_events=len(rows),
                        missing_topics=[t for t in spec if not counts[t]])
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"out": str(out), "counts": counts, "commands_sent": 0}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

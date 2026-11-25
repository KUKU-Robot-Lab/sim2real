#!/usr/bin/env python3
"""머리 카메라 토픽을 낮은 빈도로 rosbag2(sqlite3)에 쓴다 — 에피소드 bag 의 세 번째 묶음 camera(11.04 사용자).

    python3 scripts/nodes/camera_bag_recorder.py -o logs/runs/<YYMMDD>/bags/<시각>_episode_<이름>/camera --hz 10 <토픽...>
    (deploy/policy_control/tools/rh56f1_record.sh 가 CAMERA='<토픽...>' 일 때 팔 · 손 bag 옆에 띄운다. SIGINT 에 마무리)

ros2 bag record(humble)는 빈도를 못 낮추고 topic_tools 도 깔려 있지 않다. 그래서 받은 직렬화 바이트를 RateGate 로 걸러
그대로 쓴다(디코드 · 재인코딩 없음 → 이 프로세스 CPU 는 작다). 압축(JPEG · PNG)은 카메라 노드의 image_transport 가 한다 —
구독자가 하나라도 있으면 그 노드가 매 프레임(30 Hz) 인코딩하므로, 여기서 빈도를 낮춰도 카메라 쪽 인코딩은 줄지 않는다.
bag 시각 = 받은 시각(system clock, ros2 bag record 와 같은 규약)이라 같은 PC 의 arm · hand bag 과 바로 맞는다.
구독은 best effort — 이 기록기가 느려도 카메라(reliable 발행)에 재전송 부담을 주지 않고 프레임만 버린다.
아직 안 보이는 토픽(예: /head_depth/image)은 1 s 마다 다시 찾아 나타나면 붙는다.

--encode-raw(11.05): 드라이버 압축을 끄고 여기서 압축한다. 드라이버는 압축 토픽 구독자가 있으면 30 Hz 전부를 인코딩해 원본
color · depth 발행이 p50 40 → 54 ms 로 밀렸다(logs/analysis/camera_robot_1105). 이 모드는 …/compressed · …/compressedDepth 대신
원본(rgb8 · 16UC1)을 구독해 --hz 로 솎은 것만 작업 스레드에서 압축하고, 같은 토픽 이름 · 같은 메시지(CompressedImage,
compressedDepth = 12 바이트 ConfigHeader + PNG)로 쓴다 — episode_video_overlay · student_bag_replay 가 그대로 읽는다.
압축 값은 드라이버와 같다(11.05 `ros2 param get /camera/camera`: jpeg_quality 95 · png_level 3). 압축이 밀리면 오래된 프레임을 버린다.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_recorder import RateGate  # noqa: E402

JPEG_QUALITY = 95       # 드라이버 .camera.color.image_raw.jpeg_quality(11.05)
PNG_LEVEL = 3           # 드라이버 .camera.aligned_depth_to_color.image_raw.png_level(11.05)
DEPTH_FMT = "16UC1; compressedDepth"   # humble compressed_depth_image_transport 가 쓰는 문자열(11.05 bag 실측, " png" 없음)
DEPTH_HEADER = struct.pack("<iff", 0, 0.0, 0.0)   # ConfigHeader(int32 format + float depthParam[2]) — 16UC1 은 값을 안 쓴다


def raw_source(topic: str) -> str | None:
    """압축 토픽 → 원본 토픽(압축 토픽이 아니면 None)."""
    for suffix in ("/compressed", "/compressedDepth"):
        if topic.endswith(suffix):
            return topic[: -len(suffix)]
    return None


def encode_image(topic: str, encoding: str, height: int, width: int, step: int, data) -> tuple[str, bytes]:
    """원본 Image 필드 → (CompressedImage.format, data). 드라이버 image_transport 와 같은 형식."""
    import cv2
    import numpy as np

    rows = np.frombuffer(data, np.uint8).reshape(height, step)
    if topic.endswith("/compressedDepth"):
        if encoding != "16UC1":
            raise ValueError(f"{topic}: 원본 encoding {encoding!r} ≠ 16UC1")
        img = rows[:, : width * 2].copy().view(np.uint16)   # step 여백을 잘라낸 뒤 16 비트로 본다(little endian, D435i)
        ok, png = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
        if not ok:
            raise ValueError(f"{topic}: PNG 인코딩 실패")
        return DEPTH_FMT, DEPTH_HEADER + png.tobytes()
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"{topic}: 원본 encoding {encoding!r} — rgb8 · bgr8 만")
    img = rows[:, : width * 3].reshape(height, width, 3)
    if encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # image_transport 도 bgr8 로 바꿔 압축한다
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"{topic}: JPEG 인코딩 실패")
    return f"{encoding}; jpeg compressed bgr8", jpg.tobytes()


#: runner 상태가 이 값이면 그 회는 끝났다(episode_runner SUCCESS · FAILURE · STOPPED)
_ENDED = ("SUCCESS", "FAILURE", "STOPPED")


class EpisodeGate:
    """★11.21 사용자: 에피소드 영상은 scene 진입부터 done(끝)까지만 — runner 상태 JSON(/policy_control/status/<runner>, 0.5 s)으로 연다.

    열기: 이 회의 runs 에 시작 노드가 생겼다(에피소드에 그 노드가 없으면 첫 노드). 끝난 회의 상태(지난 SUCCESS 등)로는 열지 않는다.
    끝: 연 뒤 상태가 _ENDED 이거나 episode_id 가 바뀌었다([새 에피소드]). 한 번 끝나면 다시 열지 않는다(기록기도 끝낸다).
    상태는 0.5 s 마다 오므로 시작 · 끝이 그만큼 늦을 수 있다. 두 PC 시계를 비교하지 않으려고 받은 때 연다."""

    def __init__(self, start_node: str) -> None:
        self.start_node = start_node
        self.open = False
        self.finished = False
        self.episode_id: str | None = None

    def update(self, data: str) -> None:
        if self.finished:
            return
        try:
            s = json.loads(data)
            status, eid, runs = s.get("status"), s.get("episode_id"), s.get("runs") or {}
            ids = [n.get("id") for n in s.get("nodes") or []]
        except (ValueError, AttributeError):
            return
        if self.open:
            if status in _ENDED or eid != self.episode_id:
                self.open, self.finished = False, True
            return
        started = self.start_node in runs if self.start_node in ids else bool(runs)
        if started and status not in _ENDED:
            self.open, self.episode_id = True, eid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("-o", "--out", type=Path, required=True, help="bag 폴더(없어야 한다 — rosbag2 가 만든다)")
    ap.add_argument("--hz", type=float, default=10.0, help="토픽마다 남길 최대 빈도")
    ap.add_argument("--encode-raw", action="store_true",
                    help="…/compressed · …/compressedDepth 는 원본을 구독해 이 프로세스에서 압축한다(드라이버 압축 안 함)")
    ap.add_argument("--gate", default="",
                    help="에피소드 runner 상태 토픽(/policy_control/status/<runner>) — 주면 --gate-start 노드 진입부터 그 회가 끝날 때까지만 쓰고 스스로 끝난다")
    ap.add_argument("--gate-start", default="scene", help="--gate 기록을 여는 노드 id(에피소드에 없으면 첫 노드)")
    ap.add_argument("topics", nargs="+")
    args = ap.parse_args(argv)
    if args.hz <= 0:
        ap.error("--hz 는 0 보다 커야 한다")

    import rclpy
    import rosbag2_py
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message

    rclpy.init()
    node = Node("camera_bag_recorder")
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.out), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    wanted, subs, counts = set(args.topics), {}, {t: 0 for t in args.topics}
    # --encode-raw: 원본 토픽 → 쓸 압축 토픽
    encode = {raw_source(t): t for t in args.topics if raw_source(t)} if args.encode_raw else {}
    lock = threading.Lock()             # SequentialWriter 를 콜백 스레드와 압축 스레드가 같이 쓴다
    jobs: collections.deque = collections.deque(maxlen=2)   # 밀리면 가장 오래된 작업이 빠진다
    wake, stop = threading.Event(), threading.Event()
    dropped = [0]                       # 압축이 밀려 버린 프레임 수
    ep_gate = EpisodeGate(args.gate_start) if args.gate else None

    def write(topic: str, data: bytes, t: int) -> None:
        with lock:
            writer.write(topic, data, t)
            counts[topic] += 1

    def on_msg(topic: str):
        gate = RateGate(args.hz)

        def cb(data: bytes) -> None:
            if ep_gate is not None and not ep_gate.open:      # 에피소드 구간 밖 — 솎기 전에 버린다
                return
            t = node.get_clock().now().nanoseconds
            if not gate.take(t * 1e-9):
                return
            if topic in encode:
                if len(jobs) == jobs.maxlen:
                    dropped[0] += 1
                jobs.append((encode[topic], data, t))   # 디코드 · 압축은 솎은 뒤 작업 스레드에서
                wake.set()
            else:
                write(topic, data, t)
        return cb

    def encoder() -> None:
        from rclpy.serialization import deserialize_message, serialize_message
        from sensor_msgs.msg import CompressedImage, Image

        bad: set[str] = set()
        while not (stop.is_set() and not jobs):
            try:
                topic, data, t = jobs.popleft()
            except IndexError:
                wake.wait(0.1)
                wake.clear()
                continue
            try:                            # 어떤 오류든 스레드가 조용히 죽어 기록이 멈추지 않게 프레임만 버린다
                m = deserialize_message(data, Image)
                fmt, out = encode_image(topic, m.encoding, m.height, m.width, m.step, m.data)
                write(topic, serialize_message(CompressedImage(header=m.header, format=fmt, data=out)), t)
            except Exception as e:  # noqa: BLE001
                if topic not in bad:        # 같은 토픽 오류는 한 번만 알린다(프레임마다 찍지 않게)
                    bad.add(topic)
                    print(f"[camera_bag] 압축 · 기록 못 함 — 이 토픽 프레임을 버린다: {e!r}", flush=True)

    def discover() -> None:
        for name, types in node.get_topic_names_and_types():
            if name in encode and name not in subs and types[0] == "sensor_msgs/msg/Image":
                with lock:                  # 압축 스레드가 다른 토픽을 쓰는 중일 수 있다
                    writer.create_topic(rosbag2_py.TopicMetadata(name=encode[name], serialization_format="cdr",
                                                                 type="sensor_msgs/msg/CompressedImage"))
                subs[name] = node.create_subscription(get_message(types[0]), name, on_msg(name), raw_qos, raw=True)
                print(f"[camera_bag] 구독 {name} → 압축해 {encode[name]}", flush=True)
            elif name in wanted and name not in subs and name not in encode.values():
                writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=types[0], serialization_format="cdr"))
                subs[name] = node.create_subscription(get_message(types[0]), name, on_msg(name),
                                                      qos_profile_sensor_data, raw=True)
                print(f"[camera_bag] 구독 {name} ({types[0]})", flush=True)

    worker, raw_qos = None, None
    if encode:
        import cv2
        cv2.setNumThreads(1)            # 정책 · FP++ 와 코어를 다투지 않게
        raw_qos = QoSProfile(depth=2, reliability=qos_profile_sensor_data.reliability,   # sensor_data, 깊이만 2
                             durability=qos_profile_sensor_data.durability)
        worker = threading.Thread(target=encoder, name="camera_bag_encode", daemon=True)
        worker.start()
    if ep_gate is not None:
        from std_msgs.msg import String

        def on_status(m) -> None:
            was = ep_gate.open
            ep_gate.update(m.data)
            if ep_gate.open and not was:
                print(f"[camera_bag] 에피소드 {ep_gate.episode_id} {args.gate_start} 진입 — 기록 시작", flush=True)
            elif ep_gate.finished:
                print(f"[camera_bag] 에피소드 {ep_gate.episode_id} 끝 — 기록 마무리", flush=True)

        node.create_subscription(String, args.gate, on_status, QoSProfile(depth=10))
    discover()
    node.create_timer(1.0, discover)
    print(f"[camera_bag] 기록 시작 → {args.out} · {args.hz:g} Hz · 토픽 {len(wanted)}개"
          + (f" · {args.gate} 의 {args.gate_start} 진입부터 그 회 끝까지" if ep_gate else ""), flush=True)
    try:
        while rclpy.ok() and not (ep_gate is not None and ep_gate.finished):
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if worker is not None:          # 큐에 남은 프레임까지 쓰고 끝낸다
            stop.set()
            wake.set()
            worker.join(timeout=5.0)
            if worker.is_alive():
                print("[camera_bag] 압축 스레드가 5 s 안에 안 끝남 — 마지막 프레임이 빠질 수 있다", flush=True)
        writer = None                   # humble SequentialWriter 는 close() 가 없다 — 놓을 때 metadata.yaml 을 쓴다
        rclpy.try_shutdown()
    print("[camera_bag] 끝 — " + " · ".join(f"{t} {n}" for t, n in counts.items()), flush=True)
    if encode:
        print(f"[camera_bag] 압축이 밀려 버린 프레임 {dropped[0]}", flush=True)
    missing = [t for t in args.topics if t not in subs and t not in encode.values()]
    missing += [t for r, t in encode.items() if r not in subs]
    if missing:
        print(f"[camera_bag] 한 번도 안 보인 토픽: {missing}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

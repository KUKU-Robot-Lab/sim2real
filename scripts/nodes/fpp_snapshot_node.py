#!/usr/bin/env python3
"""FP++ 컨테이너 하나에서 같은 모양 · 다른 색 물체 여럿을 한 번 찍는다(10.08 사용자). 컨테이너 안에서 돈다.

    python3 /opt/s2r/scripts/nodes/fpp_snapshot_node.py --config /opt/params/<그룹>.yaml
    (scripts/vision/fpp_group_up.sh 가 띄운다 — 인지 런처가 objects.yaml 의 fpp.group 으로 부른다)

왜: 컵은 가만히 있고 정책도 시작 때 컵 자세를 한 번 붙잡는다(에피소드 snapshot · 단독 aglt cup_latch, 10.04 사용자
"FPP 는 정지 상태만, 잡은 뒤는 FK"). 물체마다 컨테이너를 두면 모델을 따로 올리고(하나 약 1.4 GB) 첫 등록이 겹쳐
GPU 가 모자랐다(10.08 두 컨테이너 동시 등록 → 2.6 GB 더 요구 · OOM).

하는 일:
  1. 최신 컬러 · 정렬 깊이 한 장 → 후보 = YOLO(classes, 문턱 yolo_conf) + YOLO 가 못 잡은 색의 색 덩어리 → 깊이로 base 에
     옮겨 작업 영역(workspace, camera_to_base 가 있을 때) 밖은 버림 → 마스크 색 비율로 물체마다 후보 하나(fpp_color_pick.assign).
  2. 물체마다 차례로: 이어지는 register_frames 장에서 따로따로 FP++ 등록 → 위치 중앙값에 가까운 장(대표 자세).
     추적(track)은 쓰지 않는다 — 10.08 실측: 둘째 물체부터 추적 9 장이 262 · 111 mm 흔들렸다(첫 물체 0.5 mm). 마스크
     추적기(Cutie)가 앞 물체를 기억한 채 넘어가는 것으로 본다. 등록은 매번 그 물체 마스크로 새로 시작한다.
     모델 · CUDA 컨텍스트는 하나를 재사용한다(재등록 = reset_object). 그래서 최대 메모리는 물체 수와 무관하다.
  3. 대표 자세를 /perception_plus_plus/<물체>/pose 에 republish_hz 로 계속 낸다(stamp = 지금, 같은 값) —
     object_pose_node 가 base 로 바꾼다. 신선도를 보는 하류(정책 노드 · 콘솔)가 그대로 돈다.
  4. 못 찾은 물체는 retry_s 마다 다시 찾는다. /perception_plus_plus/snapshot/cmd(String "all" | "이름,이름")로 다시 찍는다
     (컵을 옮긴 뒤 — 미션 cups 단계를 다시 실행하면 scripts/ops/fpp_rescan.py 가 보낸다). 다시 찍는 컵은 옛 좌표를 내지 않는다.
     generation = 끝낸 찍기 바퀴 수(상태에 실린다).
  /perception_plus_plus/snapshot/status(String JSON): 물체별 found · 색 비율 · 흔들림 mm · 장 수 · 오류, 후보별 색 비율.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fpp_color_pick as C  # noqa: E402

STATUS_TOPIC = "/perception_plus_plus/snapshot/status"
CMD_TOPIC = "/perception_plus_plus/snapshot/cmd"


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    objs = raw.get("objects")
    if not isinstance(objs, list) or not objs:
        raise ValueError(f"{path}: objects 목록이 없다")
    for o in objs:
        for k in ("name", "color", "mesh_path", "mesh_scale_to_meters", "pose_topic"):
            if k not in o:
                raise ValueError(f"{path}: 물체 {o.get('name')} 에 {k} 가 없다")
        try:
            C.hue_ranges(o["color"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: 물체 {o['name']} 색 {o['color']!r} — {C.COLORS} 또는 [lo, hi]") from exc
    return raw


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    from perception_plus_plus_core.config import TrackingConfig
    from perception_plus_plus_core.detection.yolo import YoloCupDetector
    from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import FoundationPosePlusPlusAdapter
    from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, MeshSpec
    from perception_plus_plus_core.validation.depth import depth_to_meters
    from perception_plus_plus_core.validation.quality import evaluate_quality

    objects = {o["name"]: o for o in cfg["objects"]}
    register_frames = int(cfg.get("register_frames", 3))
    classes = {int(c) for c in cfg.get("classes", [39, 41, 75])}       # bottle · cup · vase
    yolo_conf = float(cfg.get("yolo_conf", 0.05))
    blob_min_area = int(cfg.get("blob_min_area", 1500))
    T_base_cam = np.asarray(cfg["camera_to_base"], float) if cfg.get("camera_to_base") is not None else None
    workspace = cfg.get("workspace", {"x": [0.05, 0.47], "y": [-0.45, 0.45], "z": [0.15, 0.50]})
    retry_s = float(cfg.get("retry_s", 3.0))
    tconf = TrackingConfig.from_yaml(cfg.get("tracking_config", "config/cup_tracking.yaml"))

    class Snapshot(Node):
        def __init__(self) -> None:
            super().__init__("fpp_snapshot")
            self.bridge = CvBridge()
            self.lock = threading.Lock()
            self.frame: FrameBundle | None = None
            self.frame_seq = 0
            self.header = None
            self.poses: dict[str, np.ndarray] = {}
            self.info: dict[str, dict] = {n: {"found": False} for n in objects}
            self.candidates: list = []
            self.pending: set[str] = set(objects)
            self.generation = 0                     # 끝낸 찍기 바퀴 수 — fpp_rescan 이 '새 회차'를 가린다
            self.wake = threading.Event()
            self.pubs = {n: self.create_publisher(PoseStamped, o["pose_topic"], 10) for n, o in objects.items()}
            self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
            self.create_subscription(String, CMD_TOPIC, self._on_cmd, 10)
            subs = [Subscriber(self, Image, cfg.get("rgb_topic", "/camera/camera/color/image_raw"),
                               qos_profile=qos_profile_sensor_data),
                    Subscriber(self, Image, cfg.get("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"),
                               qos_profile=qos_profile_sensor_data),
                    Subscriber(self, CameraInfo, cfg.get("camera_info_topic", "/camera/camera/color/camera_info"),
                               qos_profile=qos_profile_sensor_data)]
            self.sync = ApproximateTimeSynchronizer(subs, 10, 0.04)
            self.sync.registerCallback(self._on_frame)
            self.create_timer(1.0 / float(cfg.get("republish_hz", 5.0)), self._republish)
            self.create_timer(1.0, self._publish_status)
            self.adapter = FoundationPosePlusPlusAdapter()
            self.detector = YoloCupDetector(cfg.get("yolo_weights", "models/yolo/yolov8m-seg.pt"),
                                            int(cfg.get("cup_class_id", 41)), yolo_conf, pick="confidence")
            self.error = ""
            threading.Thread(target=self._worker, name="fpp-snapshot", daemon=True).start()
            listing = ", ".join(f"{n}({o['color']})" for n, o in objects.items())
            self.get_logger().info(f"물체 {listing} · 한 번 찍기")

        # ── 입력 ──
        def _on_frame(self, rgb_msg, depth_msg, info_msg) -> None:
            rgb = np.asarray(self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8"))
            depth = depth_to_meters(np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")), depth_msg.encoding)
            k = info_msg.k
            frame = FrameBundle(rgb, depth, CameraIntrinsics(k[0], k[4], k[2], k[5], info_msg.width, info_msg.height),
                                rclpy.time.Time.from_msg(rgb_msg.header.stamp).nanoseconds, rgb_msg.header.frame_id)
            with self.lock:
                self.frame, self.header = frame, rgb_msg.header
                self.frame_seq += 1

        def _on_cmd(self, msg) -> None:
            text = msg.data.strip()
            names = set(objects) if text in ("", "all") else {n.strip() for n in text.split(",")} & set(objects)
            with self.lock:
                for n in names:
                    self.poses.pop(n, None)
                    self.info[n] = {"found": False}
                self.pending |= names
            self.get_logger().info(f"다시 찍기: {sorted(names)}")
            self.wake.set()

        def _next_frame(self, after: int, timeout: float = 2.0):
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                with self.lock:
                    if self.frame is not None and self.frame_seq > after:
                        return self.frame, self.frame_seq
                time.sleep(0.01)
            return None, after

        # ── 찍기 ──
        def _worker(self) -> None:
            seq = 0
            while rclpy.ok():
                with self.lock:
                    todo = sorted(self.pending)
                if not todo:
                    self.wake.wait(timeout=1.0)
                    self.wake.clear()
                    continue
                frame, seq = self._next_frame(seq, timeout=5.0)
                if frame is None:
                    self.error = "카메라 영상이 없다"
                    continue
                try:
                    self._snapshot(frame, seq, todo)
                    with self.lock:
                        self.generation += 1
                    self.error = ""
                except Exception as exc:  # noqa: BLE001 — 노드는 살아서 오류를 상태로 낸다
                    self.error = f"{type(exc).__name__}: {exc}"[:200]
                    self.get_logger().error(self.error)
                with self.lock:
                    left = bool(self.pending)
                if left:
                    self.wake.wait(timeout=retry_s)
                    self.wake.clear()

        def _candidates(self, frame, colors: list) -> list[dict]:
            """YOLO(classes 의 아무 클래스, 낮은 문턱) + YOLO 가 못 잡은 색의 색 덩어리 → 작업 영역 안 후보만.
            10.09: YOLO 가 핑크 병을 vase 0.08 로만 잡았다(병은 bottle 39 · cyl60 은 cup 41)."""
            out = []
            res = self.detector.model(frame.rgb, conf=yolo_conf, verbose=False)
            for r in res:
                if r.boxes is None or r.masks is None:
                    continue
                for cls, conf, box, mk in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist(),
                                              r.masks.data.cpu().numpy()):
                    if int(cls) not in classes:
                        continue
                    m = mk.astype(bool)
                    if m.shape != frame.rgb.shape[:2]:
                        import cv2
                        m = cv2.resize(m.astype(np.uint8), frame.rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                    out.append({"mask": m, "src": f"yolo {int(cls)}", "conf": round(float(conf), 3),
                                "box": [round(float(v)) for v in box]})
            for c in colors:
                if any(C.color_fraction(frame.rgb, d["mask"], c) >= C.MIN_SCORE for d in out):
                    continue
                for m in C.color_blobs(frame.rgb, c, min_area=blob_min_area):
                    ys, xs = np.nonzero(m)
                    out.append({"mask": m, "src": f"blob {c}", "conf": None,
                                "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
            K = frame.intrinsics.matrix
            if T_base_cam is not None:
                # 영상에서 붙어 보이는 물체들을 3D 로 나눈다(10.09 바깥으로 옮긴 핑크 병 + 뒤에 누운 핑크 병이 한 덩어리)
                split = []
                for d in out:
                    parts = C.split_by_depth(d["mask"], frame.depth, K, T_base_cam, min_px=blob_min_area // 2)
                    split += [{**d, "mask": q, "src": d["src"] + (f" /3D{k}" if len(parts) > 1 else "")}
                              for k, q in enumerate(parts)] or [d]
                out = split
            for d in out:
                d["inside"] = T_base_cam is None or C.in_workspace(T_base_cam, C.mask_point(d["mask"], frame.depth, K), workspace)
                d["points"] = (C.mask_points_base(d["mask"], frame.depth, K, T_base_cam)
                               if T_base_cam is not None and d["inside"] else None)
            return out

        def _snapshot(self, frame, seq: int, todo: list[str]) -> None:
            cands = self._candidates(frame, C.unique_colors(objects[n]["color"] for n in todo))
            self.candidates = [{"src": d["src"], "conf": d["conf"], "box": d["box"], "inside": d["inside"], **sc}
                               for d, sc in zip(cands, C.scores(frame.rgb, [d["mask"] for d in cands]))]
            dets = [d for d in cands if d["inside"]]
            masks = [d["mask"] for d in dets]
            # 서 있는 모양(수평 퍼짐 ≤ 물체 바닥)인 후보만 — 누운 같은 색 물체를 버린다(10.09 핑크 병). aabb · 카메라 자세가 있을 때
            allowed = {n: {i for i, d in enumerate(dets) if d["points"] is None or C.fits_footprint(d["points"], objects[n]["aabb"])}
                       for n in todo if objects[n].get("aabb")}
            pick = C.assign(frame.rgb, masks, {n: objects[n]["color"] for n in todo}, allowed=allowed)
            for name in todo:
                if name not in pick:
                    self.info[name] = {"found": False, "why": "그 색 후보가 없다"}
                    continue
                det = dets[pick[name]]
                det_mask = det["mask"]
                o = objects[name]
                mesh = MeshSpec(o["mesh_path"], float(o["mesh_scale_to_meters"]))
                t0 = time.monotonic()
                poses, why, f, s = [], "", frame, seq
                for k in range(register_frames):
                    if k:
                        f, s = self._next_frame(s)
                        if f is None:
                            break
                    result = self.adapter.initialize(f, det_mask, mesh)      # 컵은 그대로 — 첫 장 마스크를 쓴다
                    self.adapter.reset()
                    q = evaluate_quality(f, result, None, tconf)
                    if q.valid:
                        poses.append(result.object_to_camera)
                    else:
                        why = q.reason
                if not poses:
                    self.info[name] = {"found": False, "why": f"등록 품질 {why}"}
                    continue
                T, spread = C.representative(poses)
                with self.lock:
                    self.poses[name] = T
                    self.pending.discard(name)
                self.info[name] = {"found": True, "color_score": round(C.color_fraction(frame.rgb, det_mask, o["color"]), 3),
                                   "src": det["src"], "yolo_conf": det["conf"], "frames": len(poses),
                                   "spread_mm": round(spread, 1), "ms": round((time.monotonic() - t0) * 1e3),
                                   "box": det["box"]}
                self.get_logger().info(f"{name}: {self.info[name]}")

        # ── 출력 ──
        def _republish(self) -> None:
            with self.lock:
                if self.header is None:
                    return
                frame_id = self.header.frame_id
                items = list(self.poses.items())
            now = self.get_clock().now().to_msg()
            for name, T in items:
                msg = PoseStamped()
                msg.header.frame_id, msg.header.stamp = frame_id, now
                p, qx = T[:3, 3], _quat_xyzw(T[:3, :3])
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
                (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                 msg.pose.orientation.w) = map(float, qx)
                self.pubs[name].publish(msg)

        def _publish_status(self) -> None:
            with self.lock:
                body = {"ok": not self.pending and not self.error, "generation": self.generation,
                        "pending": sorted(self.pending), "error": self.error,
                        "objects": dict(self.info), "candidates": self.candidates}
            self.status_pub.publish(String(data=json.dumps(body, ensure_ascii=False)))

    rclpy.init()
    node = Snapshot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _quat_xyzw(R: np.ndarray) -> np.ndarray:
    """회전 행렬 → 사원수(x, y, z, w). 대각합이 작을 때도 안정한 분기."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


if __name__ == "__main__":
    main()

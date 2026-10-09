#!/usr/bin/env python3
"""실기 런 동안 카메라 JPEG(기본 5 Hz) · FP++ 실시간 좌표(/perception_plus_plus/<물체>/live_pose)를 남긴다(10.09 사용자).

    ROS_LOCALHOST_ONLY=1 python3 scripts/nodes/vision_recorder.py --out logs/vision_runs/<이름> --objects source200_pink
    (미션 정책 단계가 bag 기록과 같이 띄운다 — SIGINT · SIGTERM 에 마무리)

카메라 · FP++ 는 localhost 전용 DDS(ROS_LOCALHOST_ONLY=1)라 팔 bag 기록기(로봇 쪽 DDS)가 못 본다 — 이 노드가 그쪽에서 돈다.
남기는 것: <out>/frames/<시각 ns>.jpg · <out>/live_poses.csv(t · 물체 · 카메라 좌표 · base 좌표 · 기울기) · <out>/meta.json.
정책은 시작 때 찍은 고정 좌표를 쓴다 — 이 기록은 가림 · 좌표 튐 · 물체가 밀린 순간을 보는 분석용이다(파인튜닝 근거).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

FIELDS = ["t", "name", "cx", "cy", "cz", "qx", "qy", "qz", "qw", "bx", "by", "bz", "tilt_deg"]


class RateGate:
    """주어진 주기로만 통과 — 카메라 30 Hz 에서 5 Hz 만 남긴다."""

    def __init__(self, hz: float) -> None:
        self.dt, self.next = 1.0 / float(hz), float("-inf")

    def take(self, t: float) -> bool:
        if t + 1e-9 < self.next:
            return False
        self.next = t + self.dt
        return True


def pose_row(t: float, name: str, p_cam, q_xyzw, T_base_cam) -> dict:
    """카메라 프레임 자세 → CSV 한 줄(base 위치 · 물체 z 축과 base z 사이 기울기)."""
    x, y, z, w = (float(v) for v in q_xyzw)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.asarray(T_base_cam, float)
    pb = T[:3, :3] @ np.asarray(p_cam, float) + T[:3, 3]
    axis = T[:3, :3] @ R[:, 2]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(axis[2]))))))   # 위아래 대칭 물체 — 축 부호 무시
    return {"t": t, "name": name, "cx": p_cam[0], "cy": p_cam[1], "cz": p_cam[2], "qx": x, "qy": y, "qz": z, "qw": w,
            "bx": float(pb[0]), "by": float(pb[1]), "bz": float(pb[2]), "tilt_deg": tilt}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="기록 폴더(없으면 --out-root/<날짜_시각>_<이름>)")
    ap.add_argument("--out-root", type=Path, default=_ROOT / "logs" / "vision_runs")
    ap.add_argument("--name", default="run")
    ap.add_argument("--objects", nargs="*", default=[])
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--camera-extrinsics", type=Path, default=_ROOT / "config" / "global_camera_extrinsics_arm4090.yaml")
    args = ap.parse_args(argv)
    import cv2
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from object_registry import _base_from_camera

    out = args.out or args.out_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.name}"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    T = np.asarray(_base_from_camera(args.camera_extrinsics))
    (out / "meta.json").write_text(json.dumps({"objects": args.objects, "hz": args.hz, "T_base_cam": T.tolist(),
                                               "camera_extrinsics": str(args.camera_extrinsics), "start": time.time()},
                                              ensure_ascii=False, indent=1))
    fh = (out / "live_poses.csv").open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    stop = {"now": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(now=True))
    rclpy.init()
    node = rclpy.create_node("vision_recorder")
    gate, counts = RateGate(args.hz), {"frames": 0, "poses": 0}

    def on_image(m) -> None:
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if not gate.take(t):
            return
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / "frames" / f"{m.header.stamp.sec}{m.header.stamp.nanosec:09d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        counts["frames"] += 1

    def on_pose(name):
        def f(m) -> None:
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            p, q = m.pose.position, m.pose.orientation
            w.writerow(pose_row(t, name, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), T))
            counts["poses"] += 1
        return f

    node.create_subscription(Image, "/camera/camera/color/image_raw", on_image, qos_profile_sensor_data)
    for n in args.objects:
        node.create_subscription(PoseStamped, f"/perception_plus_plus/{n}/live_pose", on_pose(n), 50)
    last = time.time()
    while not stop["now"] and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - last > 10:
            fh.flush()
            last = time.time()
    fh.close()
    node.destroy_node()
    rclpy.try_shutdown()
    print(f"[vision] {out} · 영상 {counts['frames']} 장 · 실시간 좌표 {counts['poses']} 줄")
    return 0


if __name__ == "__main__":
    sys.exit(main())

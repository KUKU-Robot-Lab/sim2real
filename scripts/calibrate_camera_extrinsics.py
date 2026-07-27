"""QR 정적 extrinsics 캘리브 CLI.

카메라 1프레임(라이브 ROS 또는 NPZ) → QR 4모서리 → solvePnP → T_cam_qr →
T_base_cam = T_base_qr ∘ inv(T_cam_qr) → global_camera_extrinsics.yaml 의
camera.position/orientation_wxyz 갱신(다른 블록·주석 보존).

목 관절은 캘리브한 home 자세에 고정 전제. 동적 목/핸드-아이는 범위 밖.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from extrinsics_calib import (  # noqa: E402
    compose_base_cam, mat_to_pos_quat_wxyz, reprojection_residual_px, solve_qr_pose,
)

DEFAULT_EXTRINSICS = _DIR.parent / "config" / "global_camera_extrinsics.yaml"


def detect_qr_corners(rgb: np.ndarray):
    """cv2.QRCodeDetector로 QR 4모서리 (4,2). 실패 시 None."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    detector = cv2.QRCodeDetector()
    ok, points = detector.detect(gray)
    if not ok or points is None:
        return None
    return np.asarray(points, np.float64).reshape(4, 2)


def _fmt(vals):
    return "[" + ", ".join(f"{v:.9g}" for v in vals) + "]"


def update_camera_extrinsics_yaml(path: str, pos, quat_wxyz) -> str:
    """`camera:` 블록의 position/orientation_wxyz 두 줄만 교체, 나머지 보존.

    두 치환이 모두 일어나지 않으면(`camera:` 블록 부재 또는 필드 누락)
    조용히 원문을 반환하지 않고 ValueError를 낸다.
    """
    text = Path(path).read_text()
    lines = text.splitlines(keepends=True)
    out, in_camera = [], False
    position_replaced = False
    orientation_replaced = False
    for line in lines:
        stripped = line.rstrip("\n")
        # 최상위 키 시작 판정 (들여쓰기 없음 + `key:`). 주석 줄(#로 시작)은
        # `key:`로 끝나도 블록 경계로 취급하지 않는다.
        if not stripped.lstrip().startswith("#") and re.match(r"^\S.*:\s*$", stripped):
            in_camera = stripped.strip() == "camera:"
        if in_camera and re.match(r"^\s+position:\s*\[", line):
            out.append(re.sub(r"\[[^\]]*\]", _fmt(pos), line, count=1))
            position_replaced = True
            continue
        if in_camera and re.match(r"^\s+orientation_wxyz:\s*\[", line):
            out.append(re.sub(r"\[[^\]]*\]", _fmt(quat_wxyz), line, count=1))
            orientation_replaced = True
            continue
        out.append(line)
    if not position_replaced or not orientation_replaced:
        missing = []
        if not position_replaced:
            missing.append("camera.position")
        if not orientation_replaced:
            missing.append("camera.orientation_wxyz")
        raise ValueError(
            f"'camera:' 블록에서 다음 항목을 찾지 못해 갱신 불가: {', '.join(missing)}"
        )
    return "".join(out)


def _load_t_base_qr(arg: str) -> np.ndarray:
    """xyzrpy(6값, m·rad) 또는 yaml 경로 → 4x4."""
    import cv2
    if Path(arg).exists():
        d = yaml.safe_load(Path(arg).read_text())
        pos = d["position"]; rpy = d["rpy"]
    else:
        vals = [float(v) for v in arg.split(",")]
        if len(vals) != 6:
            raise ValueError(
                f"--t-base-qr 값은 정확히 6개(x,y,z,r,p,y)여야 함, 받은 개수: {len(vals)}"
            )
        pos, rpy = vals[:3], vals[3:6]
    R, _ = cv2.Rodrigues(np.array(rpy))   # 근사(작은각) — 정밀 필요시 yaml에 quat 사용
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    return T


def average_corners(corners_list: list) -> np.ndarray:
    """유효 검출된 (4,2) QR 코너 배열들의 평균 (4,2)."""
    return np.mean(np.stack([np.asarray(c, np.float64) for c in corners_list], axis=0), axis=0)


def _frame_from_npz(path: str):
    d = np.load(path)
    return np.asarray(d["rgb"]), np.asarray(d["K"], np.float64)


def _frame_from_ros(rgb_topic, info_topic, n_frames: int = 1):
    """ROS에서 n_frames개의 (rgb, K) 프레임을 캡처해 리스트로 반환."""
    import rclpy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init()
    node = rclpy.create_node("extrinsics_calib_capture")
    bridge = CvBridge()
    frames = []
    while rclpy.ok() and len(frames) < n_frames:
        got = {}
        def rgb_cb(m): got["rgb"] = np.asarray(bridge.imgmsg_to_cv2(m, "rgb8"))
        def info_cb(m): got["K"] = np.array([[m.k[0], 0, m.k[2]], [0, m.k[4], m.k[5]], [0, 0, 1]])
        sub_rgb = node.create_subscription(Image, rgb_topic, rgb_cb, 10)
        sub_info = node.create_subscription(CameraInfo, info_topic, info_cb, 10)
        while rclpy.ok() and ("rgb" not in got or "K" not in got):
            rclpy.spin_once(node, timeout_sec=1.0)
        node.destroy_subscription(sub_rgb)
        node.destroy_subscription(sub_info)
        frames.append((got["rgb"], got["K"]))
    node.destroy_node(); rclpy.shutdown()
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="QR 정적 extrinsics 캘리브")
    ap.add_argument("--source", choices=["npz", "ros"], default="ros")
    ap.add_argument("--npz", help="--source npz 일 때 rgb+K NPZ")
    ap.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    ap.add_argument("--qr-size", type=float, required=True, help="QR 한 변(m)")
    ap.add_argument("--t-base-qr", required=True,
                    help="QR의 base 기준 pose: 'x,y,z,r,p,y'(m,rad) 또는 yaml 경로")
    ap.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    ap.add_argument("--write", action="store_true", help="yaml 실제 갱신(없으면 dry-run)")
    ap.add_argument("--frames", type=int, default=1, help="ROS 캡처 시 평균낼 프레임 수(코너 평균)")
    args = ap.parse_args(argv)

    if args.source == "npz" and not args.npz:
        ap.error("--source npz 사용 시 --npz 가 필수입니다")

    if args.source == "npz":
        if args.frames > 1:
            print("참고: --source npz 는 저장된 단일 프레임만 사용함(--frames 무시)",
                  file=sys.stderr)
        rgb, K = _frame_from_npz(args.npz)
        corners = detect_qr_corners(rgb)
        if corners is None:
            print("QR 미검출 — 조명/거리/크기 확인", file=sys.stderr)
            return 1
    else:
        frames = _frame_from_ros(args.rgb_topic, args.info_topic, args.frames)
        K = frames[0][1]  # K는 프레임 간 고정
        corners_list = [c for rgb, _ in frames if (c := detect_qr_corners(rgb)) is not None]
        if not corners_list:
            print(f"QR 미검출 — {len(frames)}개 프레임 전부 실패(조명/거리/크기 확인)",
                  file=sys.stderr)
            return 1
        corners = average_corners(corners_list)

    T_cam_qr = solve_qr_pose(corners, args.qr_size, K, None)
    residual = reprojection_residual_px(corners, T_cam_qr, args.qr_size, K, None)
    T_base_qr = _load_t_base_qr(args.t_base_qr)
    T_base_cam = compose_base_cam(T_base_qr, T_cam_qr)
    pos, quat = mat_to_pos_quat_wxyz(T_base_cam)
    print(f"재투영 잔차: {residual:.3f}px")
    print(f"T_base_cam position: {pos}")
    print(f"T_base_cam orientation_wxyz: {quat}")
    if residual > 2.0:
        print("경고: 잔차 큼(>2px) — QR 검출/크기 확인", file=sys.stderr)
    if args.write:
        updated = update_camera_extrinsics_yaml(args.extrinsics, pos, quat)
        Path(args.extrinsics).write_text(updated)
        print(f"갱신됨: {args.extrinsics}")
    else:
        print("dry-run (--write 로 yaml 갱신)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

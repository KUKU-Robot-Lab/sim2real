#!/usr/bin/env python3
"""Isaac ROS FoundationPose 출력 → /cup_pose 릴레이 (P1 비전 노드).

Isaac ROS FoundationPose 노드는 카메라 optical 프레임 기준 컵 CAD 프레임
pose를 vision_msgs/Detection3DArray 로 발행한다. 이 노드가 글로벌 카메라
extrinsics(T_base_cam)와 CAD↔body 정합(T_cad_body)을 합성해 소비자 규약
(geometry_msgs/PoseStamped, robot base 프레임)으로 /cup_pose 를 발행한다.

    T_base_body = T_base_cam ∘ T_cam_cad ∘ T_cad_body

소비자: sim2real_inference.py (grasp 단계), pour_inference.py (capture 1회).
pour 루프는 FK라 이 노드가 죽어도 pouring은 계속된다.

사용:
    python3 cup_pose_relay.py \
        --extrinsics ../config/global_camera_extrinsics.yaml \
        --in-topic /poses --out-topic /cup_pose --min-score 0.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pour_obs_builder import compose_pose  # noqa: E402

DEFAULT_EXTRINSICS = _SCRIPT_DIR.parent / "config" / "global_camera_extrinsics.yaml"
QUAT_NORM_TOL = 1e-3


# ---------------------------------------------------------------------------
# 순수 로직 (ROS 불필요 — test_cup_pose_relay.py 대상)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Extrinsics:
    """base←camera 변환 + CAD→body 보정 (쿼터니언 wxyz)."""

    cam_pos: np.ndarray
    cam_quat: np.ndarray
    cad_to_body_pos: np.ndarray
    cad_to_body_quat: np.ndarray
    base_frame: str


def _validated_quat(raw, name: str) -> np.ndarray:
    q = np.asarray(raw, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"{name}: expected 4 values (wxyz), got shape {q.shape}")
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > QUAT_NORM_TOL:
        raise ValueError(f"{name}: quaternion not normalized (|q|={norm:.4f})")
    return q / norm


def _validated_pos(raw, name: str) -> np.ndarray:
    p = np.asarray(raw, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"{name}: expected 3 values, got shape {p.shape}")
    return p


def load_extrinsics(path: str | Path) -> Extrinsics:
    """yaml 로드 + 스키마 검증. 잘못된 값이면 즉시 ValueError."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: yaml root must be a mapping")
    for key in ("camera", "cad_to_body", "base_frame"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key '{key}'")
    cam = cfg["camera"]
    cad = cfg["cad_to_body"]
    return Extrinsics(
        cam_pos=_validated_pos(cam["position"], "camera.position"),
        cam_quat=_validated_quat(cam["orientation_wxyz"], "camera.orientation_wxyz"),
        cad_to_body_pos=_validated_pos(cad["position"], "cad_to_body.position"),
        cad_to_body_quat=_validated_quat(
            cad["orientation_wxyz"], "cad_to_body.orientation_wxyz"
        ),
        base_frame=str(cfg["base_frame"]),
    )


def cad_pose_to_base_body(
    ext: Extrinsics,
    cad_pos_cam: np.ndarray,
    cad_quat_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """카메라 프레임 CAD pose → base 프레임 body pose (pos, quat wxyz)."""
    base_cad_pos, base_cad_quat = compose_pose(
        ext.cam_pos, ext.cam_quat, cad_pos_cam, cad_quat_cam
    )
    return compose_pose(
        base_cad_pos, base_cad_quat, ext.cad_to_body_pos, ext.cad_to_body_quat
    )


def select_best_detection(
    candidates: list[tuple[float, np.ndarray, np.ndarray]],
    min_score: float,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """(score, pos, quat_wxyz) 후보 중 min_score 이상 최고점. 없으면 None."""
    passing = [c for c in candidates if c[0] >= min_score]
    if not passing:
        return None
    return max(passing, key=lambda c: c[0])


# ---------------------------------------------------------------------------
# ROS 노드
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FoundationPose → /cup_pose 릴레이")
    parser.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    parser.add_argument("--in-topic", default="/poses",
                        help="Isaac ROS FoundationPose Detection3DArray 출력 토픽")
    parser.add_argument("--out-topic", default="/cup_pose")
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    ext = load_extrinsics(args.extrinsics)

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from vision_msgs.msg import Detection3DArray

    class CupPoseRelay(Node):
        def __init__(self) -> None:
            super().__init__("cup_pose_relay")
            self._pub = self.create_publisher(PoseStamped, args.out_topic, 10)
            self.create_subscription(
                Detection3DArray, args.in_topic, self._detections_cb, 10
            )
            self._published = 0
            self.get_logger().info(
                f"relay: {args.in_topic} → {args.out_topic} "
                f"(extrinsics={args.extrinsics}, min_score={args.min_score})"
            )

        def _detections_cb(self, msg) -> None:
            candidates = []
            for det in msg.detections:
                for res in det.results:
                    p = res.pose.pose.position
                    q = res.pose.pose.orientation  # ROS xyzw
                    candidates.append((
                        float(res.hypothesis.score),
                        np.array([p.x, p.y, p.z]),
                        np.array([q.w, q.x, q.y, q.z]),
                    ))
            best = select_best_detection(candidates, args.min_score)
            if best is None:
                return
            _, cad_pos, cad_quat = best
            pos, quat = cad_pose_to_base_body(ext, cad_pos, cad_quat)

            out = PoseStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = ext.base_frame
            out.pose.position.x, out.pose.position.y, out.pose.position.z = pos
            out.pose.orientation.w = quat[0]
            out.pose.orientation.x = quat[1]
            out.pose.orientation.y = quat[2]
            out.pose.orientation.z = quat[3]
            self._pub.publish(out)
            self._published += 1
            if self._published % 30 == 1:
                self.get_logger().info(
                    f"cup_pose #{self._published}: pos={np.round(pos, 3).tolist()}"
                )

    rclpy.init()
    node = CupPoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

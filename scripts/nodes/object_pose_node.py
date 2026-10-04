#!/usr/bin/env python3
"""노드 3 — 카메라 프레임 물체 pose → base_link `/objects/<name>/pose`.

vision-3090 의 FP++ 가 내는 `/perception_plus_plus/<name>/pose`(camera optical) 를
DDS 로 직접 구독하고, cup_pose_relay 의 변환(T_base_cam ∘ T_cam_cad ∘ T_cad_body)으로
base_link 로 바꿔 발행한다. camera 블록은 레지스트리가 가리키는 공유 extrinsics,
cad_to_body 는 물체 항목. 소비자: ROS 정책 노드(다음 스펙).

  python3 object_pose_node.py [--objects shaker_closed cup_big_s100] [--head-joint-topic /head/joint_states]
  python3 object_pose_node.py --objects cup_big_s100 --camera-extrinsics config/global_camera_extrinsics_arm4090.yaml

10.01: 카메라 extrinsics 는 로봇마다 다르다(arm4090 은 테이블 CAD 캘리브 전용 파일). `--camera-extrinsics` 가
레지스트리의 공유 파일을 대신하고, 그 파일의 `base_z_bias_m`(없으면 0 — depth 치우침 보정)을 출력 z 에 더한다.
10.04: 그 파일에 `depth_bias`(offset_m · per_m · valid_z_m)가 있으면 FP++ 위치를 카메라 광선 방향으로 늘린다(z 만 보정의 대신).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))


from cup_pose_relay import (  # noqa: E402
    cad_pose_to_base_body, extrinsics_at_head, head_state_is_usable,
)
from object_registry import extrinsics_for, input_topic, load_registry, output_topic  # noqa: E402
from pose_symmetry import point_axis_up, quat_axis_direction, quat_conj, remove_twist  # noqa: E402


def base_z_bias(camera_yaml: str | Path) -> float:
    """camera extrinsics yaml 의 선택 키 `base_z_bias_m`(없으면 0)."""
    with open(camera_yaml) as fh:
        raw = yaml.safe_load(fh) or {}
    value = float(raw.get("base_z_bias_m", 0.0))
    if abs(value) > 0.05:
        raise ValueError(f"{camera_yaml}: base_z_bias_m {value} 가 ±0.05 m 밖 — 보정이 아니라 캘리브 오류다")
    return value


DEPTH_BIAS_MAX_M = 0.05


def depth_bias(camera_yaml: str | Path) -> tuple[float, float, float, float] | None:
    """camera yaml 의 선택 블록 `depth_bias`(offset_m · per_m · valid_z_m) — 없으면 None. 10.04 arm4090."""
    with open(camera_yaml) as fh:
        raw = (yaml.safe_load(fh) or {}).get("depth_bias")
    if raw is None:
        return None
    off, per = float(raw["offset_m"]), float(raw["per_m"])
    lo, hi = (float(v) for v in raw.get("valid_z_m", (0.3, 1.5)))
    if not 0.0 < lo < hi:
        raise ValueError(f"{camera_yaml}: depth_bias.valid_z_m [{lo}, {hi}] 가 이상하다")
    worst = max(abs(off + per * lo), abs(off + per * hi))
    if worst > DEPTH_BIAS_MAX_M:
        raise ValueError(f"{camera_yaml}: depth_bias 가 범위 끝에서 {worst:.3f} m — 보정이 아니라 캘리브 오류다")
    return off, per, lo, hi


def correct_depth(pos_cam: np.ndarray, bias: tuple[float, float, float, float] | None) -> np.ndarray:
    """카메라 프레임 점을 광선 방향으로 늘린다: 측정 깊이 z 가 e(z) 만큼 짧다 → 참 깊이 z − e(z). 순수."""
    p = np.asarray(pos_cam, float)
    if bias is None or p[2] <= 0.0:
        return p
    off, per, lo, hi = bias
    z = float(p[2])
    e = off + per * min(max(z, lo), hi)
    return p * ((z - e) / z)


class PoseConverter:
    """순수부: 물체별 Extrinsics 를 미리 조립해 두고 변환만 한다."""

    def __init__(self, registry, names: list[str], camera_yaml: str | Path | None = None,
                 z_bias: float | None = None) -> None:
        camera_yaml = Path(camera_yaml) if camera_yaml is not None else registry.camera_extrinsics
        self.z_bias = float(z_bias) if z_bias is not None else base_z_bias(camera_yaml)
        self.depth_bias = depth_bias(camera_yaml)
        if self.depth_bias is not None and self.z_bias:
            raise ValueError(f"{camera_yaml}: depth_bias(광선 보정)와 z 보정 {self.z_bias:+.4f} 를 같이 쓰지 않는다 — 하나만")
        self.names: list[str] = []
        for raw in names:
            canon = registry.resolve(raw)
            if canon not in self.names:
                self.names.append(canon)
        self._ext = {n: extrinsics_for(registry.get(n), camera_yaml) for n in self.names}
        # 대칭축을 body 프레임으로 옮겨 둔다: a_body = R(q_cad_body)ᵀ · a_cad
        self._axis_body = {}
        for n in self.names:
            spec = registry.get(n)
            self._axis_body[n] = (None if spec.symmetry_axis is None else
                                  quat_axis_direction(quat_conj(spec.cad_to_body_quat),
                                                      np.asarray(spec.symmetry_axis, float)))
        self._flip = {n: bool(registry.get(n).symmetry_flip) for n in self.names}
        self.base_frame = next(iter(self._ext.values())).base_frame if self._ext else "base_link"

    def convert(self, name: str, pos_cam: np.ndarray, quat_cam: np.ndarray,
                head: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        ext = self._ext[name]
        if head is not None:
            ext = extrinsics_at_head(ext, *head)
        pos, quat = cad_pose_to_base_body(ext, correct_depth(pos_cam, self.depth_bias), np.asarray(quat_cam, float))
        if self._axis_body[name] is not None:
            if self._flip[name]:
                # ★위아래도 대칭인 원통은 FP++ 가 뒤집어 잡는다(10.04 cyl60 기울기 179°) — 서 있는 컵이라는 가정으로 축을 위로
                quat = point_axis_up(quat, self._axis_body[name])
            # ★출력(base) 프레임에서 body 대칭축 둘레 twist 를 뺀다 — 축 방향(기울기)은 보존,
            #   축 둘레 회전(추적기 자유 방향)은 0. 정립이면 base 기준 항등에 가까운 자세가 된다.
            quat = remove_twist(quat, self._axis_body[name])
        if self.z_bias:
            pos = pos + np.array([0.0, 0.0, self.z_bias])
        return pos, quat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="*", default=None, help="기본: 레지스트리 전체")
    ap.add_argument("--head-joint-topic", default=None,
                    help="주면 T_base_cam 을 목 각도로 매번 계산(정적 camera 블록은 pan0/tilt-20 전용)")
    ap.add_argument("--head-max-age", type=float, default=1.0)
    ap.add_argument("--camera-extrinsics", default=None,
                    help="camera extrinsics yaml(기본: 레지스트리 공유 파일). 로봇 전용 캘리브 파일을 준다")
    ap.add_argument("--z-bias", type=float, default=None, help="출력 z 보정 [m](기본: 그 yaml 의 base_z_bias_m)")
    args = ap.parse_args()
    registry = load_registry()
    conv = PoseConverter(registry, args.objects or registry.names(), args.camera_extrinsics, args.z_bias)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    class ObjectPoseNode(Node):
        def __init__(self) -> None:
            super().__init__("object_pose_node")
            self._pubs = {n: self.create_publisher(PoseStamped, output_topic(n), 10) for n in conv.names}
            for n in conv.names:
                self.create_subscription(PoseStamped, input_topic(n), lambda m, n=n: self._on_pose(n, m), 10)
            self._head: tuple[float, float] | None = None
            self._head_stamp: float | None = None
            if args.head_joint_topic:
                from sensor_msgs.msg import JointState
                self.create_subscription(JointState, args.head_joint_topic, self._on_head, 10)
            self._count = {n: 0 for n in conv.names}
            self.create_timer(10.0, self._report)
            self.get_logger().info(f"objects {conv.names} → {[output_topic(n) for n in conv.names]} · "
                                   f"camera {args.camera_extrinsics or registry.camera_extrinsics} · z 보정 {conv.z_bias:+.4f} m"
                                   f" · 깊이 광선 보정 {conv.depth_bias}")

        def _on_head(self, msg) -> None:
            names = list(msg.name)
            try:
                pan = float(msg.position[names.index("head_j_pan")])
                tilt = float(msg.position[names.index("head_j_tilt")])
            except (ValueError, IndexError):
                return
            self._head = (np.degrees(pan), np.degrees(tilt))
            self._head_stamp = self.get_clock().now().nanoseconds * 1e-9

        def _on_pose(self, name: str, msg: PoseStamped) -> None:
            head = None
            if args.head_joint_topic:
                now = self.get_clock().now().nanoseconds * 1e-9
                if not head_state_is_usable(self._head_stamp, now, args.head_max_age):
                    self.get_logger().warning("목 각도가 없거나 오래됐다 — 발행 보류", throttle_duration_sec=5.0)
                    return
                head = self._head
            p, q = msg.pose.position, msg.pose.orientation
            pos, quat = conv.convert(name, np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z]), head)
            out = PoseStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = conv.base_frame
            out.pose.position.x, out.pose.position.y, out.pose.position.z = map(float, pos)
            (out.pose.orientation.w, out.pose.orientation.x,
             out.pose.orientation.y, out.pose.orientation.z) = map(float, quat)
            self._pubs[name].publish(out)
            self._count[name] += 1

        def _report(self) -> None:
            self.get_logger().info(" · ".join(f"{n} {c}" for n, c in self._count.items()))

    rclpy.init()
    node = ObjectPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

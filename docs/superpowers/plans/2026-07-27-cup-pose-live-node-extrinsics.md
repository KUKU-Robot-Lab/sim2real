# cup_pose 라이브 노드 + extrinsics 캘리브 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cup 하나를 카메라 → 견고한 6D pose → `/cup_pose`(base) → 정책 obs 까지 라이브로 완성한다(SP1 견고한 ROS 노드 + SP2 QR extrinsics 캘리브).

**Architecture:** 검증된 `live_fp_demo.py` 루프(detection-anchor + in-place re-anchor + fd-level stdout 억제)를 새 ROS 노드로 승격한다(FP++ 서브모듈·기존 `CupTrackingNode`/`TrackingManager` 무수정). 노드는 `PoseStamped`(camera frame)를 발행하고 기존 `cup_pose_relay.py`가 `T_base_cam ∘ T_cam_cad ∘ T_cad_body`로 `/cup_pose`(base)를 낸다. `T_base_cam`은 테이블 QR 1회 정적 캘리브로 채운다.

**Tech Stack:** Python, ROS 2 Humble(rclpy, message_filters, tf2_ros), FoundationPose++ 어댑터, ultralytics YOLO(bbox-only), OpenCV(QRCodeDetector, solvePnP), numpy, pytest, PyYAML.

## Global Constraints

- **FP++ upstream 서브모듈(`external/foundationpose_plus_plus`, pin 58aa715)은 절대 수정 금지.**
- **기존 `CupTrackingNode`(`node.py`)·`TrackingManager`·어댑터 클래스 수정 금지.** 신규 파일만 추가하며, 유일한 예외는 새 노드를 등록하는 `setup.py`의 `console_scripts` **한 줄 추가**(기존 항목 유지).
- **in-place 재-앵커는 `adapter.engine`의 메서드를 호출만 한다**(수정 아님).
- **머신 분리:** SP1(Task 1–2) = vision-3090 `~/rl_ws/perception_plus_plus`(자체 git repo). SP2(Task 3–4) = pc5090 `~/rl_ws/sim2real`. SP1 파일 편집·테스트는 vision-3090에서(로컬 clone+push 또는 ssh 편집). SP2는 pc5090 로컬.
- **쿼터니언은 전부 wxyz 순서.**
- **DDS:** ROS_DOMAIN_ID=126, vision-3090 ↔ 로봇 PC 동일 LAN.
- **테스트 실행 환경:** perception_plus_plus 코어 pytest는 `~/rl_ws/perception_plus_plus/.venv` 활성화 후(cpu). sim2real pytest는 pc5090 기존 환경.
- **검출기:** cup = COCO class 41, bbox-only(yolo11n.pt). 세그 마스크 미사용 → 마스크는 depth 밴드로 생성.
- 참조 원본(검증된 프로토타입): `sim2real` 스크래치의 `live_fp_demo.py`(현재 vision-3090 데모 이미지에 baked). 본 계획의 순수 로직·재-앵커 코드는 이 데모에서 유래.

---

### Task 1: SP1 순수 anchor 기하 헬퍼 (ROS 비의존)

새 노드의 프레임별 판정 로직을 ROS 없이 단위 테스트 가능한 순수 함수로 분리한다.

**Files:**
- Create: `perception_plus_plus_core/tracking/anchor_geometry.py` (vision-3090)
- Test: `tests/test_anchor_geometry.py` (vision-3090)

**Interfaces:**
- Produces:
  - `reproject(pose: np.ndarray(4,4), K: np.ndarray(3,3)) -> tuple[float,float] | None`
  - `inside_box(uv: tuple[float,float], xyxy: tuple[float,float,float,float], margin: float) -> bool`
  - `anchor_valid(pose, K, det_xyxy: tuple|None, z_min: float, z_max: float, margin: float) -> bool`
  - `bbox_depth_mask(depth: np.ndarray, xyxy, band: float=0.08, lo: float=0.1, hi: float=3.0) -> np.ndarray(bool)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_anchor_geometry.py
import numpy as np
from perception_plus_plus_core.tracking.anchor_geometry import (
    reproject, inside_box, anchor_valid, bbox_depth_mask,
)

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def _pose(x, y, z):
    p = np.eye(4)
    p[:3, 3] = (x, y, z)
    return p


def test_reproject_center():
    uv = reproject(_pose(0.0, 0.0, 1.0), K)
    assert uv is not None
    assert abs(uv[0] - 320.0) < 1e-6 and abs(uv[1] - 240.0) < 1e-6


def test_reproject_behind_camera_is_none():
    assert reproject(_pose(0.0, 0.0, -0.5), K) is None


def test_inside_box_margin():
    box = (300.0, 220.0, 340.0, 260.0)
    assert inside_box((320.0, 240.0), box, 0.0)
    assert not inside_box((360.0, 240.0), box, 0.0)      # 밖
    assert inside_box((360.0, 240.0), box, 0.5)          # margin 확장으로 안


def test_anchor_valid_z_gate():
    assert not anchor_valid(_pose(0, 0, 0.05), K, None, 0.15, 1.6, 0.35)   # 너무 가까움
    assert not anchor_valid(_pose(0, 0, 2.0), K, None, 0.15, 1.6, 0.35)    # 너무 멀음
    assert anchor_valid(_pose(0, 0, 0.6), K, None, 0.15, 1.6, 0.35)        # z OK, det 없음 → 유효


def test_anchor_valid_off_cup():
    # pose가 화면 중앙(320,240)에 투영되는데 det bbox는 우측에 있음 → 벗어남
    det = (500.0, 220.0, 560.0, 300.0)
    assert not anchor_valid(_pose(0, 0, 0.6), K, det, 0.15, 1.6, 0.35)
    det_center = (300.0, 220.0, 340.0, 260.0)
    assert anchor_valid(_pose(0, 0, 0.6), K, det_center, 0.15, 1.6, 0.35)


def test_bbox_depth_mask_selects_near_band():
    depth = np.full((480, 640), 2.5, np.float32)   # 배경 원거리
    depth[230:250, 310:330] = 0.6                  # bbox 안 근거리 컵
    mask = bbox_depth_mask(depth, (300, 220, 340, 260))
    assert mask.dtype == bool and mask.shape == (480, 640)
    assert mask[240, 320]            # 근거리 픽셀 포함
    assert not mask[0, 0]            # bbox 밖 제외
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/rl_ws/perception_plus_plus && source .venv/bin/activate && pytest tests/test_anchor_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: ...anchor_geometry`

- [ ] **Step 3: Write minimal implementation**

```python
# perception_plus_plus_core/tracking/anchor_geometry.py
"""ROS 비의존 anchor 기하 헬퍼 — 새 anchored 노드의 프레임별 판정 로직.

live_fp_demo.py에서 유래(검증됨). 기존 tracking 로직은 건드리지 않는다.
"""
from __future__ import annotations

import numpy as np


def reproject(pose: np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    p = pose[:3, 3]
    if p[2] <= 1e-6:
        return None
    return (float(K[0, 0] * p[0] / p[2] + K[0, 2]),
            float(K[1, 1] * p[1] / p[2] + K[1, 2]))


def inside_box(uv: tuple[float, float], xyxy, margin: float) -> bool:
    x0, y0, x1, y1 = xyxy
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (x0 - mx <= uv[0] <= x1 + mx) and (y0 - my <= uv[1] <= y1 + my)


def anchor_valid(pose: np.ndarray, K: np.ndarray, det_xyxy,
                 z_min: float, z_max: float, margin: float) -> bool:
    """z 범위 + (검출 있으면) bbox 재투영 포함 여부. 검출 없으면 z만 본다."""
    z = float(pose[2, 3])
    if z < z_min or z > z_max:
        return False
    if det_xyxy is None:
        return True
    uv = reproject(pose, K)
    if uv is None:
        return False
    return inside_box(uv, det_xyxy, margin)


def bbox_depth_mask(depth: np.ndarray, xyxy, band: float = 0.08,
                    lo: float = 0.1, hi: float = 3.0) -> np.ndarray:
    """bbox ∩ 유효깊이의 near-percentile35 ± band. 2000px 미만이면 유효깊이 전체."""
    h, w = depth.shape
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    box = np.zeros((h, w), bool)
    box[y0:y1, x0:x1] = True
    valid = box & (depth > lo) & (depth < hi)
    vals = depth[valid]
    if vals.size == 0:
        return box
    centre = float(np.percentile(vals, 35))
    mask = valid & (depth > centre - band) & (depth < centre + band)
    return mask if mask.sum() >= 2000 else valid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/rl_ws/perception_plus_plus && source .venv/bin/activate && pytest tests/test_anchor_geometry.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
cd ~/rl_ws/perception_plus_plus
git add perception_plus_plus_core/tracking/anchor_geometry.py tests/test_anchor_geometry.py
git commit -m "feat: ROS 비의존 anchor 기하 헬퍼(reproject/inside_box/anchor_valid/bbox_depth_mask)"
```

---

### Task 2: SP1 견고한 anchored ROS 노드

Task 1 헬퍼 + FP++ 어댑터로 detection-anchor 추종·in-place 재-앵커 노드를 만든다. `PoseStamped`(camera frame) + TF + 최소 `TrackingStatus` 발행. 기존 노드/코어 무수정, 신규 파일 + setup.py 한 줄.

**Files:**
- Create: `ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/anchored_node.py` (vision-3090)
- Create: `ros_ws/src/perception_plus_plus_ros/test/test_anchored_node_import.py` (vision-3090)
- Modify: `ros_ws/src/perception_plus_plus_ros/setup.py` (console_scripts 항목 **추가**, 기존 유지)

**Interfaces:**
- Consumes (Task 1): `anchor_valid`, `bbox_depth_mask` from `perception_plus_plus_core.tracking.anchor_geometry`.
- Consumes (기존, import만): `FoundationPosePlusPlusAdapter`, `YoloCupDetector`는 **쓰지 않음**(세그 필요). 대신 `ultralytics.YOLO` 직접(bbox-only). `perception_plus_plus_core.types.{CameraIntrinsics,FrameBundle,MeshSpec}`, `perception_plus_plus_core.validation.depth.depth_to_meters`, `perception_plus_plus_ros.conversion.fill_transform`, `perception_plus_plus_msgs.msg.TrackingStatus`.
- Adapter 계약(기존): `adapter.initialize(FrameBundle, mask_bool, MeshSpec) -> PoseResult(.object_to_camera(4,4), .mask)`; `adapter.track(FrameBundle) -> PoseResult`; 재-앵커는 `adapter.engine.{estimator.register, cutie.initialize, kalman.initiate, get_pose_array, est_iter, mask, kf_mean, kf_covariance}`.
- Produces: 토픽 `/perception_plus_plus/cup/pose`(geometry_msgs/PoseStamped, camera optical frame), TF child `cup`, `/perception_plus_plus/cup/tracking_status`. → `cup_pose_relay.py --in-type posestamped --in-topic /perception_plus_plus/cup/pose` 소비.

- [ ] **Step 1: Write the failing test** (임포트/구조 스모크 — 라이브 하드웨어 없이)

```python
# ros_ws/src/perception_plus_plus_ros/test/test_anchored_node_import.py
import numpy as np
from perception_plus_plus_ros import anchored_node as an


def test_cup_bbox_picks_highest_conf_cup():
    class _Box:
        def __init__(self, cls, conf, xyxy):
            self.cls = _T([cls]); self.conf = _T([conf]); self.xyxy = _T([xyxy])
    class _T:
        def __init__(self, v): self._v = np.asarray(v, float)
        def cpu(self): return self
        def numpy(self): return self._v
    class _Res:
        def __init__(self, boxes): self.boxes = boxes
    class _Boxes:
        def __init__(self, rows):
            self.cls = _T([r[0] for r in rows])
            self.conf = _T([r[1] for r in rows])
            self.xyxy = _T([r[2] for r in rows])
    class _Model:
        def __call__(self, rgb, verbose=False):
            return [_Res(_Boxes([(41, 0.9, [10, 10, 40, 40]),
                                 (41, 0.5, [50, 50, 60, 60]),
                                 (0, 0.99, [0, 0, 5, 5])]))]
    best = an.cup_bbox(_Model(), np.zeros((80, 80, 3), np.uint8), conf=0.25, class_id=41)
    assert best is not None
    assert abs(best[0] - 0.9) < 1e-6                       # 최고 conf cup
    assert tuple(best[1]) == (10.0, 10.0, 40.0, 40.0)


def test_reanchor_uses_engine_without_reinit():
    calls = []
    class _Est:
        def register(self, **kw): calls.append("register"); return np.eye(4)
    class _Cutie:
        def initialize(self, rgb, d): calls.append("cutie")
    class _Kal:
        def initiate(self, arr): calls.append("kalman"); return (arr, np.eye(len(arr)))
    class _Eng:
        estimator = _Est(); cutie = _Cutie(); kalman = _Kal(); est_iter = 5
        mask = None; kf_mean = None; kf_covariance = None
        def get_pose_array(self, pose): return np.zeros(7)
    class _Adapter:
        engine = _Eng()
    rgb = np.zeros((48, 64, 3), np.uint8)
    depth = np.full((48, 64), 0.6, np.float32)
    K = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    pose, mask = an.reanchor(_Adapter(), rgb, depth, K,
                             np.ones((48, 64), bool), np.ones((48, 64), np.uint8))
    assert calls == ["register", "cutie", "kalman"]        # 재init(initialize) 호출 없음
    assert pose.shape == (4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/rl_ws/perception_plus_plus && source .venv/bin/activate && pytest ros_ws/src/perception_plus_plus_ros/test/test_anchored_node_import.py -v`
Expected: FAIL — `ModuleNotFoundError: ...anchored_node` (또는 import 시 rclpy 미가용). rclpy 미설치 환경이면 Step 3 후 `ros_ws` colcon 환경에서 재실행.

- [ ] **Step 3: Write minimal implementation**

```python
# ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/anchored_node.py
"""Detection-anchored FP++ cup 추종 ROS 노드 (신규).

live_fp_demo.py의 검증된 루프를 ROS로 승격: 매 프레임 YOLO cup bbox(bbox-only) +
FP++ track, 추정 pose가 bbox+margin 밖으로 patience 프레임 벗어나면 in-place
재-앵커(기존 engine 재사용, Hydra 재init 회피). FP는 프레임당 C/CUDA 프린트가
있어 fd 1을 /dev/null로 억제(상태는 stderr).

기존 CupTrackingNode/TrackingManager/어댑터는 수정하지 않는다.
"""
from __future__ import annotations

import os
import sys

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import (
    FoundationPosePlusPlusAdapter,
)
from perception_plus_plus_core.tracking.anchor_geometry import anchor_valid, bbox_depth_mask
from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, MeshSpec
from perception_plus_plus_core.validation.depth import depth_to_meters
from perception_plus_plus_msgs.msg import TrackingStatus
from .conversion import fill_transform


def cup_bbox(model, rgb, conf, class_id):
    """bbox-only YOLO에서 최고 conf cup의 (conf, xyxy). 없으면 None."""
    best = None
    for res in model(rgb, verbose=False):
        if res.boxes is None:
            continue
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cfd = res.boxes.conf.cpu().numpy()
        box = res.boxes.xyxy.cpu().numpy()
        for i, (c, p) in enumerate(zip(cls, cfd)):
            if c == class_id and p >= conf and (best is None or p > best[0]):
                best = (float(p), box[i])
    return best


def _as_mat(pose):
    p = pose.detach().cpu().numpy() if hasattr(pose, "detach") else np.asarray(pose)
    return p.reshape(4, 4)


def reanchor(adapter, rgb, depth, K, mask_bool, mask_u8):
    """기존 engine에서 pose 재추정 + Cutie 마스크 재시드 + Kalman 재초기화."""
    eng = adapter.engine
    pose = eng.estimator.register(K=K, rgb=rgb, depth=depth,
                                  ob_mask=mask_u8 * 255, iteration=eng.est_iter)
    eng.cutie.initialize(rgb, {"mask": mask_u8})
    eng.mask = mask_bool
    eng.kf_mean, eng.kf_covariance = eng.kalman.initiate(eng.get_pose_array(pose))
    return _as_mat(pose), mask_bool


def _silence_stdout():
    sys.stdout.flush()
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)


class AnchoredCupNode(Node):
    def __init__(self) -> None:
        super().__init__("anchored_cup_tracking")
        defaults = {
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "pose_topic": "/perception_plus_plus/cup/pose",
            "status_topic": "/perception_plus_plus/cup/tracking_status",
            "child_frame_id": "cup",
            "mesh_path": "assets/meshes/cup.obj",
            "mesh_scale_to_meters": 1.0,
            "yolo_weights": "models/yolo/yolo11n.pt",
            "cup_class_id": 41,
            "yolo_confidence": 0.25,
            "margin": 0.35,
            "z_min": 0.15,
            "z_max": 1.6,
            "patience": 3,
            "sync_slop_seconds": 0.04,
            "sync_queue_size": 10,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        g = lambda n: self.get_parameter(n).value

        from ultralytics import YOLO
        self.yolo = YOLO(g("yolo_weights"))
        self.class_id = int(g("cup_class_id"))
        self.conf = float(g("yolo_confidence"))
        self.margin = float(g("margin"))
        self.z_min, self.z_max = float(g("z_min")), float(g("z_max"))
        self.patience = int(g("patience"))
        self.adapter = FoundationPosePlusPlusAdapter()
        self.mesh = MeshSpec(g("mesh_path"), float(g("mesh_scale_to_meters")))
        self.bridge = CvBridge()
        self.child_frame_id = g("child_frame_id")
        self.pose_pub = self.create_publisher(PoseStamped, g("pose_topic"), 10)
        self.status_pub = self.create_publisher(TrackingStatus, g("status_topic"), 10)
        self.tf = TransformBroadcaster(self)

        self._started = False
        self._bad = 0
        self._valid = self._invalid = 0

        rgb = message_filters.Subscriber(self, Image, g("rgb_topic"),
                                         qos_profile=qos_profile_sensor_data)
        depth = message_filters.Subscriber(self, Image, g("depth_topic"),
                                           qos_profile=qos_profile_sensor_data)
        info = message_filters.Subscriber(self, CameraInfo, g("camera_info_topic"),
                                          qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth, info], int(g("sync_queue_size")), float(g("sync_slop_seconds")))
        self.sync.registerCallback(self._callback)
        self.get_logger().info("anchored_cup_tracking 준비 (FP 로그 억제 시작)")
        _silence_stdout()

    def _callback(self, rgb_msg, depth_msg, info_msg):
        rgb = np.asarray(self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8"))
        depth = depth_to_meters(
            np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")),
            depth_msg.encoding)
        k = info_msg.k
        K = np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]])
        intr = CameraIntrinsics(k[0], k[4], k[2], k[5], info_msg.width, info_msg.height)
        frame = FrameBundle(rgb, depth, intr,
                            rclpy.time.Time.from_msg(rgb_msg.header.stamp).nanoseconds,
                            rgb_msg.header.frame_id)

        det = cup_bbox(self.yolo, rgb, self.conf, self.class_id)
        pose = None
        if not self._started:
            if det is not None:
                mask = bbox_depth_mask(depth, det[1])
                try:
                    r = self.adapter.initialize(frame, mask, self.mesh)
                    pose = np.asarray(r.object_to_camera).reshape(4, 4)
                    self._started, self._bad = True, 0
                    self._log_stderr("initialized")
                except BaseException as e:                       # noqa: BLE001
                    self._log_stderr(f"init failed: {e}")
        else:
            r = self.adapter.track(frame)
            pose = np.asarray(r.object_to_camera).reshape(4, 4)
            det_xyxy = tuple(det[1]) if det is not None else None
            if anchor_valid(pose, K, det_xyxy, self.z_min, self.z_max, self.margin):
                self._bad, self._valid, self._invalid = 0, self._valid + 1, 0
            else:
                self._bad += 1
                self._valid, self._invalid = 0, self._invalid + 1
                if det is not None and self._bad >= self.patience:
                    mask = bbox_depth_mask(depth, det[1])
                    try:
                        pose, _ = reanchor(self.adapter, rgb, depth, K,
                                           mask, mask.astype(np.uint8))
                        self._bad = 0
                        self._log_stderr("re-anchored")
                    except BaseException as e:                   # noqa: BLE001
                        self._log_stderr(f"reanchor failed: {e}")

        if pose is not None:
            self._publish(rgb_msg, pose)
        self._publish_status(rgb_msg)

    def _publish(self, image, matrix):
        pose = PoseStamped()
        pose.header = image.header
        fill_transform(pose.pose, matrix)
        self.pose_pub.publish(pose)
        tfm = TransformStamped()
        tfm.header = image.header
        tfm.child_frame_id = self.child_frame_id
        fill_transform(tfm.transform, matrix)
        self.tf.sendTransform(tfm)

    def _publish_status(self, image):
        s = TrackingStatus()
        s.header = image.header
        s.state = 1 if self._started else 0
        s.failure_reason = "" if self._bad == 0 else "OFF_CUP_OR_Z"
        s.failure_detail = ""
        s.consecutive_valid = int(self._valid)
        s.consecutive_invalid = int(self._invalid)
        s.fatal = False
        self.status_pub.publish(s)

    @staticmethod
    def _log_stderr(msg):
        print(f"[anchored_cup] {msg}", file=sys.stderr, flush=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnchoredCupNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
```

- [ ] **Step 4: setup.py에 노드 등록(한 줄 추가)**

`ros_ws/src/perception_plus_plus_ros/setup.py`의 `entry_points`를 아래처럼 항목 추가(기존 `cup_tracking_node` 유지):

```python
    entry_points={"console_scripts": [
        "cup_tracking_node = perception_plus_plus_ros.node:main",
        "anchored_cup_tracking_node = perception_plus_plus_ros.anchored_node:main",
    ]},
```

- [ ] **Step 5: 스모크 테스트 통과 확인**

Run: `cd ~/rl_ws/perception_plus_plus && source .venv/bin/activate && pytest ros_ws/src/perception_plus_plus_ros/test/test_anchored_node_import.py -v`
Expected: PASS (2 passed). rclpy import가 .venv에 없으면 `source /opt/ros/humble/setup.bash && PYTHONNOUSERSITE=1 pytest ...`로 실행(기존 노드 테스트와 동일 방식).

- [ ] **Step 6: colcon 빌드로 노드 등록 확인**

Run:
```bash
cd ~/rl_ws/perception_plus_plus/ros_ws && source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-select perception_plus_plus_ros
source install/setup.bash
ros2 pkg executables perception_plus_plus_ros | grep anchored_cup_tracking_node
```
Expected: `perception_plus_plus_ros anchored_cup_tracking_node` 출력.

- [ ] **Step 7: Commit**

```bash
cd ~/rl_ws/perception_plus_plus
git add ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/anchored_node.py \
        ros_ws/src/perception_plus_plus_ros/test/test_anchored_node_import.py \
        ros_ws/src/perception_plus_plus_ros/setup.py
git commit -m "feat: detection-anchored FP++ cup 추종 ROS 노드(in-place 재-앵커)"
```

- [ ] **Step 8: 라이브 검증(실기, 하드웨어 필요 — 문서화 게이트)**

RealSense + 데모 이미지(또는 colcon 환경)에서:
```bash
ros2 run perception_plus_plus_ros anchored_cup_tracking_node
ros2 topic echo /perception_plus_plus/cup/pose --once      # PoseStamped(camera frame) 확인
```
Expected: cup 추종 중 손으로 컵을 들어 손 드리프트를 유발 → `[anchored_cup] re-anchored`(stderr) 후 pose가 컵으로 복귀. (하드웨어 미가용 시 이 스텝은 보류 표시.)

---

### Task 3: SP2 QR 캘리브 순수 수학 (ROS 비의존)

QR 4모서리 + K → `T_cam_qr`(solvePnP), 그리고 `T_base_cam = T_base_qr ∘ inv(T_cam_qr)`, wxyz 직렬화, 재투영 잔차. 전부 순수 함수로 단위 테스트.

**Files:**
- Create: `scripts/extrinsics_calib.py` (pc5090 sim2real)
- Test: `scripts/test_extrinsics_calib.py` (pc5090 sim2real)

**Interfaces:**
- Produces:
  - `qr_object_points(qr_size: float) -> np.ndarray(4,3)` (QR 평면 z=0, 코너 순서 TL,TR,BR,BL)
  - `solve_qr_pose(corners_2d: np.ndarray(4,2), qr_size: float, K: np.ndarray(3,3), dist: np.ndarray | None) -> np.ndarray(4,4)` (T_cam_qr)
  - `compose_base_cam(T_base_qr: np.ndarray(4,4), T_cam_qr: np.ndarray(4,4)) -> np.ndarray(4,4)` (T_base_cam)
  - `mat_to_pos_quat_wxyz(T: np.ndarray(4,4)) -> tuple[list[float], list[float]]`
  - `reprojection_residual_px(corners_2d, T_cam_qr, qr_size, K, dist) -> float`

- [ ] **Step 1: Write the failing test**

```python
# scripts/test_extrinsics_calib.py
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from extrinsics_calib import (
    qr_object_points, solve_qr_pose, compose_base_cam,
    mat_to_pos_quat_wxyz, reprojection_residual_px,
)

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def _rt(rvec, t):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def _project(T_cam_qr, qr_size):
    obj = qr_object_points(qr_size)
    cam = (T_cam_qr[:3, :3] @ obj.T + T_cam_qr[:3, 3:4]).T
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def test_solve_recovers_known_pose():
    T_true = _rt([0.05, -0.1, 0.02], [0.03, -0.02, 0.7])
    corners = _project(T_true, 0.08)
    T_est = solve_qr_pose(corners, 0.08, K, None)
    assert np.allclose(T_est[:3, 3], T_true[:3, 3], atol=2e-3)
    assert np.allclose(T_est[:3, :3], T_true[:3, :3], atol=2e-3)


def test_reprojection_residual_small_for_true_pose():
    T_true = _rt([0.0, 0.0, 0.0], [0.0, 0.0, 0.6])
    corners = _project(T_true, 0.08)
    assert reprojection_residual_px(corners, T_true, 0.08, K, None) < 0.5


def test_compose_base_cam():
    # base=qr(단위), camera가 qr를 z=0.7에서 봄 → base_cam translation = -inv 적용 결과
    T_base_qr = np.eye(4)
    T_cam_qr = _rt([0, 0, 0], [0.0, 0.0, 0.7])
    T_base_cam = compose_base_cam(T_base_qr, T_cam_qr)
    # T_base_cam ∘ T_cam_qr == T_base_qr
    assert np.allclose(T_base_cam @ T_cam_qr, T_base_qr, atol=1e-9)


def test_mat_to_pos_quat_wxyz_identity():
    pos, quat = mat_to_pos_quat_wxyz(np.eye(4))
    assert np.allclose(pos, [0, 0, 0])
    assert np.allclose(quat, [1, 0, 0, 0])       # wxyz
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/rl_ws/sim2real/scripts && python -m pytest test_extrinsics_calib.py -v`
Expected: FAIL — `ModuleNotFoundError: extrinsics_calib`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/extrinsics_calib.py
"""QR 정적 extrinsics 캘리브 순수 수학 (ROS 비의존).

T_base_cam = T_base_qr ∘ inv(T_cam_qr),  T_cam_qr = solvePnP(QR 코너).
쿼터니언은 wxyz. 코너 순서는 TL,TR,BR,BL (OpenCV QRCodeDetector 순서).
"""
from __future__ import annotations

import numpy as np


def qr_object_points(qr_size: float) -> np.ndarray:
    """QR 중심 원점, 평면 z=0, 코너 TL,TR,BR,BL (x우/y하)."""
    h = qr_size / 2.0
    return np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])


def solve_qr_pose(corners_2d: np.ndarray, qr_size: float, K: np.ndarray,
                  dist: np.ndarray | None) -> np.ndarray:
    import cv2
    obj = qr_object_points(qr_size).astype(np.float64)
    img = np.asarray(corners_2d, np.float64).reshape(-1, 1, 2)
    d = np.zeros((5, 1)) if dist is None else np.asarray(dist, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float64), d,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("solvePnP failed for QR corners")
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def compose_base_cam(T_base_qr: np.ndarray, T_cam_qr: np.ndarray) -> np.ndarray:
    return T_base_qr @ np.linalg.inv(T_cam_qr)


def mat_to_pos_quat_wxyz(T: np.ndarray) -> tuple[list[float], list[float]]:
    m = T[:3, :3]
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return [float(v) for v in T[:3, 3]], [float(v) for v in q]


def reprojection_residual_px(corners_2d: np.ndarray, T_cam_qr: np.ndarray,
                             qr_size: float, K: np.ndarray,
                             dist: np.ndarray | None) -> float:
    import cv2
    obj = qr_object_points(qr_size).astype(np.float64)
    R = T_cam_qr[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = T_cam_qr[:3, 3].reshape(3, 1)
    d = np.zeros((5, 1)) if dist is None else np.asarray(dist, np.float64)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), d)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - np.asarray(corners_2d, float), axis=1)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/rl_ws/sim2real/scripts && python -m pytest test_extrinsics_calib.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd ~/rl_ws/sim2real
git add scripts/extrinsics_calib.py scripts/test_extrinsics_calib.py
git commit -m "feat: QR extrinsics 캘리브 순수 수학(solvePnP/compose/wxyz/residual)"
```

---

### Task 4: SP2 캘리브 CLI + yaml 갱신

Task 3 수학 + OpenCV QR 검출 + yaml 갱신을 묶은 CLI. 오프라인(NPZ) 경로와 yaml 갱신은 단위 테스트, 라이브(ROS) 경로는 문서화 게이트.

**Files:**
- Create: `scripts/calibrate_camera_extrinsics.py` (pc5090 sim2real)
- Test: `scripts/test_calibrate_camera_extrinsics.py` (pc5090 sim2real)

**Interfaces:**
- Consumes (Task 3): `solve_qr_pose`, `compose_base_cam`, `mat_to_pos_quat_wxyz`, `reprojection_residual_px`.
- Produces:
  - `detect_qr_corners(rgb: np.ndarray) -> np.ndarray(4,2) | None` (cv2.QRCodeDetector)
  - `update_camera_extrinsics_yaml(path: str, pos: list[float], quat_wxyz: list[float]) -> str` — `camera:` 블록의 `position`/`orientation_wxyz` 두 줄만 교체, 나머지(주석·cad_to_body·base_frame)는 보존. 갱신된 텍스트 반환.
  - `main()` — CLI: `--source {npz,ros}`, `--qr-size`, `--t-base-qr`(6값 xyzrpy 또는 yaml), `--extrinsics`, `--frames N`(평균).

- [ ] **Step 1: Write the failing test**

```python
# scripts/test_calibrate_camera_extrinsics.py
from pathlib import Path
import numpy as np
import pytest

from calibrate_camera_extrinsics import update_camera_extrinsics_yaml

SAMPLE = """\
# 주석 헤더 보존 확인
camera:
  frame: camera_color_optical_frame
  position: [0.0, 0.0, 0.0]
  orientation_wxyz: [1.0, 0.0, 0.0, 0.0]

# cad_to_body 주석 보존
cad_to_body:
  position: [0.0, 0.0, 0.0]
  orientation_wxyz: [0.707107, 0.707107, 0.0, 0.0]

base_frame: base_link
"""


def test_update_preserves_other_blocks(tmp_path):
    p = tmp_path / "ext.yaml"
    p.write_text(SAMPLE)
    out = update_camera_extrinsics_yaml(str(p), [0.1, -0.2, 0.9],
                                        [0.5, 0.5, -0.5, 0.5])
    assert "0.707107" in out                         # cad_to_body 보존
    assert "# cad_to_body 주석 보존" in out           # 주석 보존
    assert "base_frame: base_link" in out
    # camera 블록만 갱신
    import yaml
    data = yaml.safe_load(out)
    assert np.allclose(data["camera"]["position"], [0.1, -0.2, 0.9])
    assert np.allclose(data["camera"]["orientation_wxyz"], [0.5, 0.5, -0.5, 0.5])
    assert np.allclose(data["cad_to_body"]["orientation_wxyz"],
                       [0.707107, 0.707107, 0.0, 0.0])


def test_update_only_touches_camera_block(tmp_path):
    p = tmp_path / "ext.yaml"
    p.write_text(SAMPLE)
    out = update_camera_extrinsics_yaml(str(p), [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    # cad_to_body.position은 그대로 [0,0,0]
    import yaml
    data = yaml.safe_load(out)
    assert np.allclose(data["cad_to_body"]["position"], [0.0, 0.0, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/rl_ws/sim2real/scripts && python -m pytest test_calibrate_camera_extrinsics.py -v`
Expected: FAIL — `ModuleNotFoundError: calibrate_camera_extrinsics`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/calibrate_camera_extrinsics.py
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
    """`camera:` 블록의 position/orientation_wxyz 두 줄만 교체, 나머지 보존."""
    text = Path(path).read_text()
    lines = text.splitlines(keepends=True)
    out, in_camera = [], False
    for line in lines:
        stripped = line.rstrip("\n")
        # 최상위 키 시작 판정 (들여쓰기 없음 + `key:`)
        if re.match(r"^\S.*:\s*$", stripped):
            in_camera = stripped.strip() == "camera:"
        if in_camera and re.match(r"^\s+position:\s*\[", line):
            out.append(re.sub(r"\[.*\]", _fmt(pos), line))
            continue
        if in_camera and re.match(r"^\s+orientation_wxyz:\s*\[", line):
            out.append(re.sub(r"\[.*\]", _fmt(quat_wxyz), line))
            continue
        out.append(line)
    return "".join(out)


def _load_t_base_qr(arg: str) -> np.ndarray:
    """xyzrpy(6값, m·rad) 또는 yaml 경로 → 4x4."""
    import cv2
    if Path(arg).exists():
        d = yaml.safe_load(Path(arg).read_text())
        pos = d["position"]; rpy = d["rpy"]
    else:
        vals = [float(v) for v in arg.split(",")]
        pos, rpy = vals[:3], vals[3:6]
    R, _ = cv2.Rodrigues(np.array(rpy))   # 근사(작은각) — 정밀 필요시 yaml에 quat 사용
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    return T


def _frame_from_npz(path: str):
    d = np.load(path)
    return np.asarray(d["rgb"]), np.asarray(d["K"], np.float64)


def _frame_from_ros(rgb_topic, info_topic):
    import rclpy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init()
    node = rclpy.create_node("extrinsics_calib_capture")
    bridge = CvBridge()
    got = {}
    def rgb_cb(m): got["rgb"] = np.asarray(bridge.imgmsg_to_cv2(m, "rgb8"))
    def info_cb(m): got["K"] = np.array([[m.k[0], 0, m.k[2]], [0, m.k[4], m.k[5]], [0, 0, 1]])
    node.create_subscription(Image, rgb_topic, rgb_cb, 10)
    node.create_subscription(CameraInfo, info_topic, info_cb, 10)
    while rclpy.ok() and ("rgb" not in got or "K" not in got):
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node(); rclpy.shutdown()
    return got["rgb"], got["K"]


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
    args = ap.parse_args(argv)

    rgb, K = (_frame_from_npz(args.npz) if args.source == "npz"
              else _frame_from_ros(args.rgb_topic, args.info_topic))
    corners = detect_qr_corners(rgb)
    if corners is None:
        print("QR 미검출 — 조명/거리/크기 확인", file=sys.stderr)
        return 1
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/rl_ws/sim2real/scripts && python -m pytest test_calibrate_camera_extrinsics.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd ~/rl_ws/sim2real
git add scripts/calibrate_camera_extrinsics.py scripts/test_calibrate_camera_extrinsics.py
git commit -m "feat: QR extrinsics 캘리브 CLI(QR 검출·yaml 갱신·잔차 리포트)"
```

- [ ] **Step 6: 라이브 캘리브(실기, 하드웨어 필요 — 문서화 게이트)**

**카메라·QR은 vision-3090에 연결됨.** T_base_cam yaml은 pc5090 sim2real가 소비하므로,
프레임은 vision-3090에서 캡처(NPZ) → pc5090으로 scp → pc5090에서 `--source npz`로 calib.
목 home 자세에서 QR을 크게 보이게 두고, 테이블 QR의 base 기준 pose 실측 후:
```bash
# 1) vision-3090: RGB+K 1프레임 NPZ 캡처(기존 capture_frame.py 재사용)
ssh vision-3090 'cd ~/rl_ws/perception_plus_plus && \
    python scripts/capture_frame.py --out /tmp/qr_frame.npz'   # rgb, K 포함
scp vision-3090:/tmp/qr_frame.npz /tmp/qr_frame.npz
# 2) pc5090: NPZ로 calib → sim2real yaml 갱신
cd ~/rl_ws/sim2real/scripts
python calibrate_camera_extrinsics.py --source npz --npz /tmp/qr_frame.npz \
    --qr-size 0.10 --t-base-qr "0.5,0.0,0.0,0,0,0" --write
```
Expected: 재투영 잔차 < 2px, `global_camera_extrinsics.yaml`의 `camera` 블록만 갱신. 이후
`cup_pose_relay.py --in-type posestamped --in-topic /perception_plus_plus/cup/pose` 기동 →
`/cup_pose`가 테이블 위 그럴듯한 base 좌표에 떨어지는지 sanity. (라이브 `--source ros`는
카메라 토픽이 DDS로 pc5090까지 보일 때만 대안. 기본은 NPZ 핸드오프.) 하드웨어 미가용 시 보류 표시.

---

## Self-Review

**Spec coverage:**
- SP1 견고한 노드(detection-anchor + in-place 재-앵커 + fd 억제) → Task 1(순수 헬퍼) + Task 2(노드). ✓
- 기존 노드/코어/FP++ 무수정, setup.py 한 줄 → Global Constraints + Task 2 Step 4. ✓
- SP2 QR 정적 캘리브(solvePnP + T_base_qr 합성 + yaml 갱신 + 잔차) → Task 3(수학) + Task 4(CLI). ✓
- 목 home 고정 전제, 동적 목/핸드-아이/SP3 out-of-scope → Global Constraints + 계획 서두. ✓
- 라이브 검증(노드→relay→/cup_pose) → Task 2 Step 8, Task 4 Step 6(문서화 게이트). ✓

**Placeholder scan:** 실제 코드/명령/기대값 모두 기입. `--t-base-qr`의 rpy는 작은각 Rodrigues 근사임을 코드 주석에 명시(정밀 필요 시 yaml quat) — 의도된 설계 선택. TODO/TBD 없음.

**Type consistency:** `solve_qr_pose`/`compose_base_cam`/`mat_to_pos_quat_wxyz`/`reprojection_residual_px`(Task 3) 시그니처가 Task 4에서 동일하게 소비됨. `anchor_valid`/`bbox_depth_mask`(Task 1) 시그니처가 Task 2에서 동일. `cup_bbox`/`reanchor` 반환형(Task 2 테스트 ↔ 구현) 일치. 쿼터니언 wxyz 전 구간 일관.

## Execution Handoff

**주의(실행 로지스틱스):** Task 1–2는 vision-3090 `perception_plus_plus`(원격 repo), Task 3–4는 pc5090 `sim2real`(로컬). 서브에이전트/인라인 실행 시 SP1 파일 편집은 vision-3090에서 수행(ssh 편집 또는 로컬 clone→push→pull). Task 2 Step 8·Task 4 Step 6은 하드웨어(카메라+로봇 PC DDS) 필요 게이트.

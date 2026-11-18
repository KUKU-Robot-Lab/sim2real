"""camera_bag_to_mp4: 가짜 camera bag(JPEG 3 장, 10 Hz) → mp4 3 프레임 · 기록 빈도 fps, 기록 중 bag 은 오류."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))


def _camera_bag(path, n=3, pose=None):
    cv2 = pytest.importorskip("cv2")
    rosbag2_py = pytest.importorskip("rosbag2_py")
    import numpy as np
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CompressedImage
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    w.create_topic(rosbag2_py.TopicMetadata(name="/camera/camera/color/image_raw/compressed",
                                            type="sensor_msgs/msg/CompressedImage", serialization_format="cdr"))
    if pose is not None:      # FP++ live 추적(카메라 프레임) — 첫 프레임 stamp 에만 한 번
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo
        for name, tp in (("/camera/camera/color/camera_info", "sensor_msgs/msg/CameraInfo"),
                         ("/perception_plus_plus/source240_blue/live_pose", "geometry_msgs/msg/PoseStamped")):
            w.create_topic(rosbag2_py.TopicMetadata(name=name, type=tp, serialization_format="cdr"))
        w.write("/camera/camera/color/camera_info",
                serialize_message(CameraInfo(k=[100.0, 0.0, 64.0, 0.0, 100.0, 48.0, 0.0, 0.0, 1.0], d=[0.0] * 5)), 1_700_000_000_000_000_000)
        m = PoseStamped()
        m.header.stamp.sec, (m.pose.position.x, m.pose.position.y, m.pose.position.z) = 1_700_000_000, pose
        m.pose.orientation.w = 1.0
        w.write("/perception_plus_plus/source240_blue/live_pose", serialize_message(m), 1_700_000_000_000_000_000)
    for i in range(n):
        m = CompressedImage(format="jpeg")
        m.header.stamp.sec, m.header.stamp.nanosec = 1_700_000_000, i * 300_000_000   # 영상 stamp 0 · 0.3 · 0.6 s
        m.data = cv2.imencode(".jpg", np.full((96, 128, 3), 40 * i, np.uint8))[1].tobytes()
        w.write("/camera/camera/color/image_raw/compressed", serialize_message(m), 1_700_000_000_000_000_000 + i * 100_000_000)
    del w   # 닫혀야 metadata.yaml 이 생긴다


def test_bag_becomes_mp4_with_one_frame_per_message(tmp_path):
    cv2 = pytest.importorskip("cv2")
    _camera_bag(tmp_path / "ep" / "camera")
    import camera_bag_to_mp4 as M
    assert M.main([str(tmp_path / "ep" / "camera")]) == 0
    cap = cv2.VideoCapture(str(tmp_path / "ep" / "camera.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3 and abs(cap.get(cv2.CAP_PROP_FPS) - 10) < 0.1
    assert not (tmp_path / "ep" / "camera.part.mp4").exists()


def test_recording_bag_without_metadata_exits_1(tmp_path):
    pytest.importorskip("cv2")
    import camera_bag_to_mp4 as M
    (tmp_path / "camera").mkdir()
    (tmp_path / "camera" / "camera_0.db3").write_bytes(b"")
    assert M.main([str(tmp_path / "camera")]) == 1


def test_fpp_live_pose_draws_box_in_separate_mp4_and_grays_when_stale(tmp_path):
    """★11.18 bbox: 카메라 프레임 자세 + K → 상자 꼭짓점 픽셀, 0.5 s 넘게 늙으면 회색. camera.mp4 는 상자 없이 그대로."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    import camera_bag_to_mp4 as M
    K = np.array([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]])
    uv = M.box_px(K, np.zeros(5), (0, 0, 1.0), (1, 0, 0, 0), ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)))
    assert len(M.EDGES) == 12 and np.allclose(uv[0], [64 - 100 * 0.1 / 0.9, 48 - 100 * 0.1 / 0.9])   # 앞 모서리(z 0.9 m)
    _camera_bag(tmp_path / "ep" / "camera", pose=(0.0, 0.0, 0.5))
    assert M.main([str(tmp_path / "ep" / "camera")]) == 0
    frames = []
    for name in ("camera.mp4", "camera_bbox.mp4"):
        cap = cv2.VideoCapture(str(tmp_path / "ep" / name))
        frames.append([cap.read()[1] for _ in range(3)])

    def sat(img):                       # 색 있는 픽셀 수 — 시각 · 이름 글자(위쪽) 아래만, 상자 y ≈ 41~55
        px = img[40:].reshape(-1, 3).astype(int)
        return (px.max(1) - px.min(1) > 60).sum()
    assert sat(frames[0][0]) == 0 and sat(frames[1][0]) > 30            # 원본엔 없고 bbox 판 첫 프레임(나이 0)엔 색 상자
    assert sat(frames[1][2]) < 5 and np.abs(frames[1][2].astype(int) - frames[0][2]).max() > 60   # 0.6 s 늙음 → 회색 선

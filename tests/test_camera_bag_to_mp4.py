"""camera_bag_to_mp4: 가짜 camera bag(JPEG 3 장, 10 Hz) → mp4 3 프레임 · 기록 빈도 fps, 기록 중 bag 은 오류."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))


def _camera_bag(path, n=3):
    cv2 = pytest.importorskip("cv2")
    rosbag2_py = pytest.importorskip("rosbag2_py")
    import numpy as np
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CompressedImage
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    w.create_topic(rosbag2_py.TopicMetadata(name="/camera/camera/color/image_raw/compressed",
                                            type="sensor_msgs/msg/CompressedImage", serialization_format="cdr"))
    for i in range(n):
        m = CompressedImage(format="jpeg")
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

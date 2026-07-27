# scripts/test_calibrate_camera_extrinsics.py
from pathlib import Path
import numpy as np
import pytest

from calibrate_camera_extrinsics import average_corners, update_camera_extrinsics_yaml

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

SAMPLE_CAD_FIRST = """\
cad_to_body:
  position: [0.1, 0.2, 0.3]
  orientation_wxyz: [0.707107, 0.707107, 0.0, 0.0]

camera:
  frame: camera_color_optical_frame
  position: [0.0, 0.0, 0.0]
  orientation_wxyz: [1.0, 0.0, 0.0, 0.0]

base_frame: base_link
"""

SAMPLE_NO_CAMERA = """\
cad_to_body:
  position: [0.0, 0.0, 0.0]
  orientation_wxyz: [1.0, 0.0, 0.0, 0.0]

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


def test_update_camera_after_cad_to_body_block(tmp_path):
    """cad_to_body: 가 camera: 보다 먼저 나오고 둘 다 동일 필드를 가질 때
    camera 블록만 갱신되고 cad_to_body는 그대로여야 한다."""
    p = tmp_path / "ext.yaml"
    p.write_text(SAMPLE_CAD_FIRST)
    out = update_camera_extrinsics_yaml(str(p), [9.0, 8.0, 7.0], [0.1, 0.2, 0.3, 0.4])
    import yaml
    data = yaml.safe_load(out)
    assert np.allclose(data["camera"]["position"], [9.0, 8.0, 7.0])
    assert np.allclose(data["camera"]["orientation_wxyz"], [0.1, 0.2, 0.3, 0.4])
    # cad_to_body (camera보다 먼저 나오는 블록)는 변경되지 않아야 함
    assert np.allclose(data["cad_to_body"]["position"], [0.1, 0.2, 0.3])
    assert np.allclose(data["cad_to_body"]["orientation_wxyz"],
                       [0.707107, 0.707107, 0.0, 0.0])


def test_update_raises_when_no_camera_block(tmp_path):
    p = tmp_path / "ext.yaml"
    p.write_text(SAMPLE_NO_CAMERA)
    with pytest.raises(ValueError):
        update_camera_extrinsics_yaml(str(p), [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])


def test_average_corners_mean_of_two():
    c1 = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    c2 = np.array([[2.0, 2.0], [12.0, 2.0], [12.0, 12.0], [2.0, 12.0]])
    avg = average_corners([c1, c2])
    expected = np.array([[1.0, 1.0], [11.0, 1.0], [11.0, 11.0], [1.0, 11.0]])
    assert avg.shape == (4, 2)
    assert np.allclose(avg, expected)


def test_update_preserves_comments_with_brackets(tmp_path):
    """verify that inline comments containing brackets are preserved during update.

    regression test for: re.sub without count=1 was replacing all bracket groups,
    including those in trailing comments, e.g., '# calibrated [rig A]'.
    """
    sample_with_comment = """\
camera:
  frame: camera_color_optical_frame
  position: [0.0, 0.0, 0.0]  # calibrated [rig A]
  orientation_wxyz: [1.0, 0.0, 0.0, 0.0]  # quat from [baseline]
"""
    p = tmp_path / "ext.yaml"
    p.write_text(sample_with_comment)
    out = update_camera_extrinsics_yaml(str(p), [1.5, 2.5, 3.5], [0.7, 0.7, 0.0, 0.0])

    # verify the values are updated
    import yaml
    data = yaml.safe_load(out)
    assert np.allclose(data["camera"]["position"], [1.5, 2.5, 3.5])
    assert np.allclose(data["camera"]["orientation_wxyz"], [0.7, 0.7, 0.0, 0.0])

    # verify the comments are preserved
    assert "[rig A]" in out, "Comment '[rig A]' should be preserved in position line"
    assert "[baseline]" in out, "Comment '[baseline]' should be preserved in orientation_wxyz line"

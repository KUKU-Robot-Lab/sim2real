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

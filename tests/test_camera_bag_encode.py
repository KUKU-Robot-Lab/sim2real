"""camera_bag_recorder --encode-raw: 기록기가 만든 JPEG · compressedDepth 를 student_bag_replay 디코더가 그대로 푼다."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
import camera_bag_recorder as C  # noqa: E402
import student_bag_replay as R  # noqa: E402

H, W = 720, 1280


def test_raw_source_maps_only_compressed_topics():
    assert C.raw_source(R.T_COLOR) == "/camera/camera/color/image_raw"
    assert C.raw_source(R.T_DEPTH) == "/camera/camera/aligned_depth_to_color/image_raw"
    assert C.raw_source("/head_depth/image") is None


def test_depth_png_round_trips_exactly_with_row_padding():
    d = (np.arange(H * W, dtype=np.uint32) * 7 % 65535).astype(np.uint16).reshape(H, W)
    step = W * 2 + 64                                   # 행 끝 여백이 있어도 잘라낸다
    buf = np.zeros((H, step), np.uint8)
    buf[:, : W * 2] = d.view(np.uint8)
    fmt, data = C.encode_image(R.T_DEPTH, "16UC1", H, W, step, buf.tobytes())
    np.testing.assert_array_equal(R.decode_compressed_depth(fmt, data), d)


def test_color_jpeg_decodes_to_bgr_with_high_psnr():
    y, x = np.mgrid[0:H, 0:W]
    rgb = np.stack([x * 255 // W, y * 255 // H, (x + y) * 255 // (W + H)], -1).astype(np.uint8)
    fmt, data = C.encode_image(R.T_COLOR, "rgb8", H, W, W * 3, rgb.tobytes())
    bgr = R.decode_jpeg_bgr(fmt, data)
    mse = np.mean((bgr.astype(float) - rgb[..., ::-1].astype(float)) ** 2)
    assert 10 * np.log10(255**2 / max(mse, 1e-9)) > 40      # 채널 순서가 틀리면 PSNR 이 10 dB 안팎으로 떨어진다


def test_unexpected_encoding_is_refused():
    with pytest.raises(ValueError, match="16UC1"):
        C.encode_image(R.T_DEPTH, "32FC1", 2, 2, 8, bytes(16))
    with pytest.raises(ValueError, match="rgb8"):
        C.encode_image(R.T_COLOR, "mono8", 2, 2, 2, bytes(4))

"""student_bag_replay — compressedDepth 풀기(12 바이트 헤더 + PNG 16UC1). bag 전체 재생은 11.05 실제 bag 으로 따로 확인했다."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
import student_bag_replay as R  # noqa: E402


def _msg(d: np.ndarray, header: bytes = b"\x00" * R.CD_HEADER) -> bytes:
    ok, png = cv2.imencode(".png", d)
    assert ok
    return header + png.tobytes()


def test_compressed_depth_round_trips_16bit_mm_after_the_12_byte_header():
    d = (np.arange(720 * 1280, dtype=np.uint32) % 65535).astype(np.uint16).reshape(720, 1280)
    np.testing.assert_array_equal(R.decode_compressed_depth("16UC1; compressedDepth", _msg(d)), d)


def test_compressed_depth_refuses_other_formats_and_a_shifted_png():
    d = np.zeros((4, 4), np.uint16)
    with pytest.raises(ValueError, match="format"):
        R.decode_compressed_depth("32FC1; compressedDepth", _msg(d))
    with pytest.raises(ValueError, match="PNG"):
        R.decode_compressed_depth("16UC1; compressedDepth", _msg(d, b"\x00" * 8))
    with pytest.raises(ValueError, match="16 비트"):
        R.decode_compressed_depth("16UC1; compressedDepth", _msg(d.astype(np.uint8)))

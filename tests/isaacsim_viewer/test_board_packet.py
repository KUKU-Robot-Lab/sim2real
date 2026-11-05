"""board_viewer UDP 계약(board_packet) — 정상 왕복과 거부 경우(ROS·Isaac 불필요)."""

from __future__ import annotations

import json

import pytest

import board_packet

pytestmark = pytest.mark.unit

NAMES = [f"r_aj_{i}" for i in range(1, 8)] + ["r_hj_thumb_1", "r_hj_thumb_2", "r_hj_index_1", "r_hj_middle_1",
                                               "r_hj_ring_1", "r_hj_pinky_1"]
T = [[1, 0, 0, 0.24], [0, -1, 0, -0.24], [0, 0, -1, 0.208], [0, 0, 0, 1]]


def _obj(**kw) -> dict:
    o = json.loads(board_packet.encode(1.0, NAMES, [0.1] * 13, None, T, (0.3, -0.3, 0.208), (0.3, -0.3, 0.25), False, "dry"))
    return o | kw


def test_roundtrip():
    p = board_packet.decode(board_packet.encode(1.0, NAMES, [0.1] * 13, [0.2] * 13, T, None, (0, 0, 0), True, "pub"))
    assert p.names == tuple(NAMES) and p.q_cmd == (0.2,) * 13 and p.target is None and p.contact and p.mode == "pub"
    assert p.T_base_board[2] == (0.0, 0.0, -1.0, 0.208)


def test_board_optional():
    assert board_packet.decode(json.dumps(_obj(boards={})).encode()).T_base_board is None


@pytest.mark.parametrize("bad", [
    {"q_show": [0.1] * 12},                                            # 길이
    {"q_cmd": [float("nan")] * 13},                                    # NaN(json 은 NaN 을 쓴다)
    {"mode": "live"},
    {"contact": 1},
    {"names": NAMES[:-1] + NAMES[:1]},                                 # 중복
    {"tip_xyz_base": None},
    {"boards": {"charuco": {"T_base_board": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]}}},   # 거울(det<0)
    {"boards": {"charuco": {"T_base_board": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}},    # 비직교
    {"boards": {"charuco": {}}},
])
def test_rejects(bad):
    with pytest.raises(ValueError):
        board_packet.decode(json.dumps(_obj(**bad)).encode())

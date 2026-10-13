"""rh56f1_hand_pose — pd hand_path_pose(주먹)를 손 드라이버 레지스터로(10.13 콘솔 전 손만 쥐기)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"
pytestmark = pytest.mark.unit


def _load():
    spec = importlib.util.spec_from_file_location("rh56f1_hand_pose", TOOLS / "rh56f1_hand_pose.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rh56f1_hand_pose"] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load()


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_fist_maps_to_closed_registers(side):
    from policy_control import rh56f1_map
    hmap = rh56f1_map.load()
    pose = H.pose_from_pd(H.PD_CONFIG, side, "hand_path_pose")
    reg = H.plan(hmap, side, pose)
    order = list(hmap.joint_order)
    for f in ("index_1", "middle_1", "ring_1", "pinky_1"):        # 매뉴얼: 클수록 펴짐 — 1.45 rad 는 닫힌 쪽(900 근처)
        r = reg[hmap.axes_of(side)[order.index(f)].slot]
        assert 900 <= r < 1000
    assert all(v != rh56f1_map.LEAVE for v in reg)                 # 여섯 축 모두 확인된 축


def test_a_missing_key_is_refused(tmp_path):
    p = tmp_path / "pd.yaml"
    p.write_text("hand_path_pose: {right: {r_hj_thumb_1: 0.8}}\n")
    with pytest.raises(SystemExit):
        H.pose_from_pd(p, "left", "hand_path_pose")
    from policy_control import rh56f1_map
    with pytest.raises(SystemExit):
        H.plan(rh56f1_map.load(), "right", H.pose_from_pd(p, "right", "hand_path_pose"))

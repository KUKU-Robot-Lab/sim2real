"""bag 묶음 · 누적기 — 촉각은 정책 묶음과 따로, 배열 모양은 joint_trace 가 읽는 그대로."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy/policy_control/tools"))


@pytest.fixture(scope="module")
def cc():
    from policy_control.joint_contract import load_contract
    from policy_control.sources import load_robot_cfg, select_side
    c = load_contract(REPO / "deploy/policies/dg5f_m/cup_grasp/left_i01/joint_contract.json")
    return c, select_side(load_robot_cfg(REPO / "deploy/policy_control/config/robots/dg5f_m_left_real.yaml"), "left")


def test_sensors_are_bagged_apart_from_the_policy_stream(cc):
    from policy_control.trace_acc import topics
    t = topics(*cc)
    assert t["/dg5f_left/tactile/finger_3"] == ("tac3", "sensor_msgs/msg/Image", "sensors")
    assert t["/policy_control/joint_target"][2] == "policy" and t["/objects/cup_big_s100/pose"][0] == "obj"
    assert {g for _, _, g in t.values()} == {"policy", "sensors"}


def test_tactile_images_become_18_cells_and_arrays_have_the_report_shape(cc):
    from policy_control.trace_acc import TraceAccumulator
    acc = TraceAccumulator(*cc)
    img = SimpleNamespace(data=bytes(range(15)), encoding="mono8")
    acc.add("tac2", 1.0, img)
    acc.add("jn", 1.0, SimpleNamespace(data='{"phase": "running"}'))
    a = acc.to_arrays({})
    assert a["tac"].shape == (1, 18) and a["tac"][0, 14] == 14 and np.isnan(a["tac"][0, 15]) and a["tac_idx"][0] == 2
    assert a["jn_json"][0] == '{"phase": "running"}' and a["obs"].shape == (0, 133)


def test_the_newest_record_is_a_bag_folder_or_an_old_npz(tmp_path):
    from joint_trace_report import newest
    (tmp_path / "20260928_170931__t__left.npz").write_bytes(b"")
    (tmp_path / "20260928_180053__t__left").mkdir()
    assert newest(tmp_path, "left").name == "20260928_180053__t__left"
    assert newest(tmp_path, "right") is None

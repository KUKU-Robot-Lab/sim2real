"""실기 런 기록 도구 — 프로세스별 CPU(proc_cpu_record) · 팔 지연(arm_latency_report). 10.08 실기 전 세팅."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"
BAG = Path(__file__).resolve().parents[2] / "logs/bags/20261003_144536_home_return/arm"


def _tool(name: str):
    spec = importlib.util.spec_from_file_location("_tool_" + name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_processes_are_labelled_by_their_command_line():
    C = _tool("proc_cpu_record")
    assert C.label_of("/x/.venv/bin/python deploy/policy_control/policy_control/rh_aglt_node.py --ros-args") == "rh_aglt"
    assert C.label_of("python3 pd_node.py") == "pd"
    assert C.label_of("tools/ethercat/rh56f1_ecat_master --ifname enp6s0") == "hand_ecat_master"
    assert C.label_of("bash -lc true") is None
    # 10.08 리뷰: 팔 기록기는 EXTRA 토픽에 정책 status 이름을 들고 있다 — 정책으로 세면 안 된다
    rec = "/usr/bin/python3 /opt/ros/humble/bin/ros2 bag record -o x/arm /joint_states /policy_control/status/rh_aglt_node_right"
    assert C.label_of(rec) == "bag"
    assert C.label_of("bash -lc EXTRA='/policy_control/status/rh_place_node_right' bash rh56f1_record.sh start x") is None
    assert C.label_of("/x/python /y/rh_place_node.py --ros-args -p cup_topic:=/objects/cyl60/pose") == "rh_place"


def test_the_summary_sums_processes_of_a_label_per_sample():
    C = _tool("proc_cpu_record")
    rows = [{"t": t, "label": "pd", "pid": p, "cores": c, "rss_mb": 50} for t in (1, 2) for p, c in ((1, 0.2), (2, 0.1))]
    rows += [{"t": 1, "label": "rh_aglt", "pid": 3, "cores": 0.5, "rss_mb": 400}]
    s = C.summarize(rows)
    assert s["pd"]["mean"] == pytest.approx(0.3) and s["pd"]["n"] == 2 and s["rh_aglt"]["rss_mb"] == 400
    assert "합계 평균 0.80" in C.render(s)


def test_the_arm_latency_tool_finds_the_pd_pickup_and_a_zero_order_hold():
    L = _tool("arm_latency_report")
    ta = np.array([0.0, 0.01, 0.02]); qa = np.array([[0.0] * 7, [1.0] * 7, [2.0] * 7])
    assert L.zoh(ta, qa, np.array([0.005, 0.015, 0.5]))[:, 0].tolist() == [0.0, 1.0, 2.0]
    tt, qt = np.array([0.0, 0.1]), np.array([[0.0] * 7, [1.0] * 7])
    ta, qa = np.array([0.0, 0.105, 0.11]), np.array([[0.0] * 7, [0.5] * 7, [1.0] * 7])
    assert L.pickup_ms((tt, qt), (ta, qa)).tolist() == pytest.approx([10.0])


def test_the_arm_latency_tool_reproduces_the_10_03_measurement():
    """10.07 Grasping 회신 값(오른팔 외부 목표 구간 중앙: j1 76 · j4 64 · j6 230 ms, 집어감 중앙 8 ms)."""
    if not (BAG / "metadata.yaml").is_file():
        pytest.skip("10.03 bag 없음(git 밖) — arm4090 logs/bags 에서 복사")
    pytest.importorskip("rosbag2_py")
    rep = _tool("arm_latency_report").report(BAG, "right")
    assert rep["stage_cfg"] == ["full"] and rep["pickup_ms"]["p50"] == pytest.approx(8.0, abs=1.0)
    ext = rep["external"]
    assert (ext["j1"]["p50"], ext["j4"]["p50"], ext["j6"]["p50"]) == pytest.approx((76, 64, 230), abs=6)

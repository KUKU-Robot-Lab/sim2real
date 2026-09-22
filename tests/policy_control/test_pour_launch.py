"""pour_chain.launch.py pure part (no ROS graph)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SIM2REAL = Path(__file__).resolve().parents[2]
LAUNCH = SIM2REAL / "policy_control" / "launch" / "pour_chain.launch.py"


@pytest.fixture(scope="module")
def mod():
    pytest.importorskip("launch_ros")
    spec = importlib.util.spec_from_file_location("pour_chain_launch_t", LAUNCH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["pour_chain_launch_t"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def cfg(tmp_path):
    c = tmp_path / "pour_contract.json"
    c.write_text("{}")
    return {"contract": str(c), "robot": "dg5f_m_bi_real", "device": "cuda:0", "fake": "false",
            "use_source": "true", "src_cup_topic": "/objects/src_cup/pose", "rcv_cup_topic": "/objects/rcv_cup/pose",
            "use_fabric": "true", "params_file": ""}


def test_description_type(mod):
    from launch import LaunchDescription
    assert isinstance(mod.generate_launch_description(), LaunchDescription)


def test_one_pour_node_with_params(mod, cfg):
    nodes = mod.pour_nodes(cfg)
    assert len(nodes) == 1
    params = mod.pour_params(cfg)
    assert params["contract"] == cfg["contract"] and params["src_cup_topic"] == "/objects/src_cup/pose"
    assert params["use_fabric"] is True and params["robot"].endswith("dg5f_m_bi_real.yaml")


def test_cup_topics_required(mod, cfg):
    with pytest.raises(RuntimeError, match="rcv_cup_topic"):
        mod.pour_params({**cfg, "rcv_cup_topic": ""})


def test_missing_contract_refused(mod, cfg):
    with pytest.raises(RuntimeError):
        mod.pour_params({**cfg, "contract": "/nope/pour_contract.json"})

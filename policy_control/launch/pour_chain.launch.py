"""Bimanual pour chain launch: ONE pour_node (obs -> policy -> decoder -> two fabrics). pd_node is launched
separately (pd_controller.launch.py) with a control-only DeployContract of the same asset; do NOT start
episode_master next to this (pour_node is the episode master).

    ros2 launch policy_control pour_chain.launch.py contract:=logs/policy/pour_i11/pour_contract.json \
        robot:=dg5f_m_bi_real src_cup_topic:=/objects/<src>/pose rcv_cup_topic:=/objects/<rcv>/pose use_source:=true

args: contract (pour_contract.json, required) · robot · src_cup_topic/rcv_cup_topic (PoseStamped, robot base frame,
required) · device · use_fabric · fake (true -> refuse ROS_DOMAIN_ID 0) · use_source · params_file
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def _load_chain_helpers():
    path = Path(__file__).resolve().parent / "policy_chain.launch.py"
    spec = importlib.util.spec_from_file_location("policy_chain_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chain = _load_chain_helpers()
POUR_NODE = "pour_node"


def pour_params(cfg: dict) -> dict:
    _chain.check_domain(_chain.is_true(cfg.get("fake", "false")))
    contract = _chain.require_file(_chain.resolve_path(cfg["contract"]), "pour contract")
    robot = _chain.require_file(_chain.resolve_robot(cfg["robot"]), "robot yaml")
    for key in ("src_cup_topic", "rcv_cup_topic"):
        if not str(cfg.get(key, "")).strip():
            raise RuntimeError(f"{key} is required (PoseStamped of that cup in the robot base frame)")
    return {"contract": str(contract), "robot": str(robot), "device": str(cfg.get("device", "cuda:0")),
            "src_cup_topic": str(cfg["src_cup_topic"]).strip(), "rcv_cup_topic": str(cfg["rcv_cup_topic"]).strip(),
            "use_fabric": _chain.is_true(cfg.get("use_fabric", "true"))}


def pour_nodes(cfg: dict) -> list:
    params: list = [pour_params(cfg)]
    if cfg.get("params_file", ""):
        params.append(str(_chain.require_file(_chain.resolve_path(cfg["params_file"]), "params_file")))
    return [_chain.make_node(POUR_NODE, params, use_source=_chain.is_true(cfg.get("use_source", "false")))]


def _opaque(context):
    return pour_nodes(dict(context.launch_configurations))


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("contract", description="pour_contract.json"),
        DeclareLaunchArgument("robot", default_value="dg5f_m_bi_real"),
        DeclareLaunchArgument("src_cup_topic", default_value=""),
        DeclareLaunchArgument("rcv_cup_topic", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("use_fabric", default_value="true"),
        DeclareLaunchArgument("fake", default_value="false"),
        DeclareLaunchArgument("use_source", default_value="false"),
        DeclareLaunchArgument("params_file", default_value=""),
        OpaqueFunction(function=_opaque),
    ])

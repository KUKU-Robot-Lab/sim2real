"""joint family chain launch: ONE joint_node (obs -> policy -> decoder, no fabric) for one arm. pd_node is launched
separately (pd_controller.launch.py) with a control-only DeployContract of the deploy asset; do NOT start
episode_master next to this (joint_node is the episode master).

    ros2 launch policy_control joint_chain.launch.py contract:=deploy/policies/dg5f_m/cup_pick/right_m15/joint_contract.json \
        robot:=dg5f_m_right_real use_source:=true

args: contract (joint_contract.json, required) · robot · device · goal_offset ('x,y,z' m, '' = contract default) ·
publish_target (false -> compute only, pd follows its own target) · fake (true -> refuse ROS_DOMAIN_ID 0) ·
use_source · params_file
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
JOINT_NODE = "joint_node"


def _offset(text: str) -> list[float] | None:
    text = str(text).strip()
    if not text:
        return None
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 3:
        raise RuntimeError(f"goal_offset needs 3 comma-separated metres, got {text!r}")
    return vals


def joint_params(cfg: dict) -> dict:
    _chain.check_domain(_chain.is_true(cfg.get("fake", "false")))
    contract = _chain.require_file(_chain.resolve_path(cfg["contract"]), "joint contract")
    robot = _chain.require_file(_chain.resolve_robot(cfg["robot"]), "robot yaml")
    off = _offset(cfg.get("goal_offset", ""))
    return {"contract": str(contract), "robot": str(robot), "device": str(cfg.get("device", "cuda:0")),
            "use_goal_offset": off is not None, "goal_offset": off or [0.0, 0.0, 0.0],
            "publish_target": _chain.is_true(cfg.get("publish_target", "true")),
            # 09.28 에피소드 끝 — 학습 규칙(키포인트 도달 tol · 연속 스텝 · 에피소드 길이). 0 = 끔
            "success_tol_m": float(cfg.get("success_tol_m", "0.0")), "success_steps": int(cfg.get("success_steps", "10")),
            "max_episode_s": float(cfg.get("max_episode_s", "0.0"))}


def joint_nodes(cfg: dict) -> list:
    params: list = [joint_params(cfg)]
    if cfg.get("params_file", ""):
        params.append(str(_chain.require_file(_chain.resolve_path(cfg["params_file"]), "params_file")))
    return [_chain.make_node(JOINT_NODE, params, use_source=_chain.is_true(cfg.get("use_source", "false")))]


def _opaque(context):
    return joint_nodes(dict(context.launch_configurations))


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("contract", description="joint_contract.json"),
        DeclareLaunchArgument("robot", default_value="dg5f_m_right_real"),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("goal_offset", default_value=""),
        DeclareLaunchArgument("publish_target", default_value="true"),
        DeclareLaunchArgument("success_tol_m", default_value="0.0", description="목표 도달 키포인트 거리 [m] (0 = 끔)"),
        DeclareLaunchArgument("success_steps", default_value="10"),
        DeclareLaunchArgument("max_episode_s", default_value="0.0", description="에피소드 최대 길이 [s] (0 = 끔)"),
        DeclareLaunchArgument("fake", default_value="false"),
        DeclareLaunchArgument("use_source", default_value="false"),
        DeclareLaunchArgument("params_file", default_value=""),
        OpaqueFunction(function=_opaque),
    ])

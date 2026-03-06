"""
Apply per-joint Isaac Sim drive settings from a JSON config file.

Use in Isaac Sim Script Editor after the robot USD is loaded:

    Open this file in Isaac Sim Script Editor and run it.

By default this reads:
    /tmp/isaacsim_next_joint_drive_config.json

To use another file, edit CONFIG_PATH below before running.
"""

from __future__ import annotations

import json


CONFIG_PATH = "/tmp/isaacsim_next_joint_drive_config.json"


def _find_joint_prim(stage, robot_root: str, joint_name: str):
    root = stage.GetPrimAtPath(robot_root)
    if not root:
        return None

    root_path = str(root.GetPath())
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_path + "/") and prim.GetName() == joint_name:
            return prim
    return None


def _set_drive(stage, robot_root: str, joint_name: str, drive_type: str, stiffness: float, damping: float) -> bool:
    from pxr import UsdPhysics

    prim = _find_joint_prim(stage, robot_root, joint_name)
    if prim is None:
        print(f"Joint not found under {robot_root}: {joint_name}")
        return False

    drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
    if not drive:
        print(f"No {drive_type} drive on: {prim.GetPath()}")
        return False

    if not drive.GetStiffnessAttr():
        drive.CreateStiffnessAttr(stiffness)
    else:
        drive.GetStiffnessAttr().Set(stiffness)

    if not drive.GetDampingAttr():
        drive.CreateDampingAttr(damping)
    else:
        drive.GetDampingAttr().Set(damping)

    print(
        f"Updated {prim.GetPath()} ({drive_type}) "
        f"stiffness={stiffness:g} damping={damping:g}"
    )
    return True


def main() -> None:
    import omni.usd

    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    robot_root = str(config.get("robot_root", "/openarm_dual_modular"))
    joints = config.get("joints", {})
    if not isinstance(joints, dict):
        raise ValueError("Config JSON must contain an object at 'joints'")

    stage = omni.usd.get_context().get_stage()

    updated = 0
    for joint_name, joint_config in sorted(joints.items()):
        if not isinstance(joint_config, dict):
            continue

        updated += int(
            _set_drive(
                stage=stage,
                robot_root=robot_root,
                joint_name=joint_name,
                drive_type=str(joint_config.get("drive_type", "angular")),
                stiffness=float(joint_config.get("stiffness", 0.0)),
                damping=float(joint_config.get("damping", 0.0)),
            )
        )

        offset_delta = float(joint_config.get("recommended_offset_delta", 0.0))
        if abs(offset_delta) > 0.0:
            print(
                f"Offset hint for {joint_name}: apply {offset_delta:+.6f} "
                "in the sim-to-real mapping or zero calibration path"
            )

    print(f"Applied drive config from {CONFIG_PATH} to {updated} joints under {robot_root}")


main()

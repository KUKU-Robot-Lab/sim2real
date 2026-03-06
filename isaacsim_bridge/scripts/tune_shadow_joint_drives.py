"""
Increase Isaac Sim joint drive stiffness/damping so the shadow robot holds commanded poses more tightly.

Use in Isaac Sim Script Editor after the robot USD is loaded:

    Open this file in Isaac Sim Script Editor and run it.
"""

from __future__ import annotations


ROBOT_ROOT = "/openarm_dual_modular"

LEFT_ARM_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

RIGHT_ARM_JOINTS = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

RIGHT_HAND_JOINTS = [
    "rj_dg_1_1",
    "rj_dg_1_2",
    "rj_dg_1_3",
    "rj_dg_1_4",
    "rj_dg_2_1",
    "rj_dg_2_2",
    "rj_dg_2_3",
    "rj_dg_2_4",
    "rj_dg_3_1",
    "rj_dg_3_2",
    "rj_dg_3_3",
    "rj_dg_3_4",
    "rj_dg_4_1",
    "rj_dg_4_2",
    "rj_dg_4_3",
    "rj_dg_4_4",
    "rj_dg_5_1",
    "rj_dg_5_2",
    "rj_dg_5_3",
    "rj_dg_5_4",
]

LEFT_GRIPPER_JOINT = "openarm_left_finger_joint1"

ARM_STIFFNESS = 150000.0
ARM_DAMPING = 15000.0
HAND_STIFFNESS = 40000.0
HAND_DAMPING = 4000.0
GRIPPER_STIFFNESS = 40000.0
GRIPPER_DAMPING = 4000.0


def _find_joint_prim(stage, joint_name: str):
    root = stage.GetPrimAtPath(ROBOT_ROOT)
    if not root:
        return None
    root_path = str(root.GetPath())
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_path + "/") and prim.GetName() == joint_name:
            return prim
    return None


def _set_drive(stage, joint_name: str, drive_type: str, stiffness: float, damping: float) -> bool:
    from pxr import UsdPhysics

    prim = _find_joint_prim(stage, joint_name)
    if prim is None:
        print(f"Joint not found: {joint_name}")
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

    stage = omni.usd.get_context().get_stage()

    updated = 0
    for joint_name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
        updated += int(_set_drive(stage, joint_name, "angular", ARM_STIFFNESS, ARM_DAMPING))

    for joint_name in RIGHT_HAND_JOINTS:
        updated += int(_set_drive(stage, joint_name, "angular", HAND_STIFFNESS, HAND_DAMPING))

    updated += int(_set_drive(stage, LEFT_GRIPPER_JOINT, "linear", GRIPPER_STIFFNESS, GRIPPER_DAMPING))

    print(f"Updated drives for {updated} joints under {ROBOT_ROOT}")


main()

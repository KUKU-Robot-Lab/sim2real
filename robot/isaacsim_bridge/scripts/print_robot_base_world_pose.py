"""
Print the current world-space pose of the loaded robot prim in Isaac Sim.

Use in Isaac Sim Script Editor after the robot USD is loaded:
    Open this file in Isaac Sim Script Editor and run it.
"""

from __future__ import annotations


ROBOT_PRIMS = [
    "/openarm_dual_modular",
    "/openarm_dual_modular/root_joint",
]


def _format_vec3(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def main() -> None:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()

    for path in ROBOT_PRIMS:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            print(f"Prim not found: {path}")
            continue

        xformable = UsdGeom.Xformable(prim)
        world_m = xformable.ComputeLocalToWorldTransform(0.0)
        translation = world_m.ExtractTranslation()

        print(f"{path} world_translation: {_format_vec3(translation)}")


main()

"""
List articulation root prims in the current Isaac Sim stage.

Use in Isaac Sim Script Editor:
    Open this file in Isaac Sim Script Editor and run it.
"""

import omni.usd
from pxr import UsdPhysics


def main():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("No USD stage is open.")
        return

    found = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            found.append(str(prim.GetPath()))

    if not found:
        print("No articulation roots found in the current stage.")
        return

    print("Articulation roots:")
    for path in found:
        print(f"  {path}")


main()

"""
Compare joint/frame/state-defining USD physics data between a source USD and a target USD.

This is for diagnosing cases where a joint has the same drive settings but a different
"zero pose" because the joint frame itself was authored differently.

Default comparison:
    source: /home/user/rl_ws/openarm_bimanual/openarm_bimanual.usd
    target: /home/user/rl_ws/urdf/openarm_modular_dual/openarm_modular_dual.usd
    joints: openarm_left_joint1

Run inside Isaac Sim Script Editor:

    import sys, runpy
    sys.argv = ["compare_joint_frames.py"]
    runpy.run_path(
        "/home/user/rl_ws/sim2real_control/isaacsim_bridge/scripts/compare_joint_frames.py",
        run_name="__main__",
    )

Or with specific joints:

    import sys, runpy
    sys.argv = ["compare_joint_frames.py", "--joint", "openarm_left_joint1", "--joint", "openarm_left_joint2"]
    runpy.run_path(
        "/home/user/rl_ws/sim2real_control/isaacsim_bridge/scripts/compare_joint_frames.py",
        run_name="__main__",
    )
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap_pxr_from_isaacsim() -> None:
    isaac_roots = [
        Path("/home/user/isaacsim/5.1.0"),
        Path("/home/user/isaacsim/5.0.0"),
    ]
    package_globs = (
        "extscache/omni.usd.libs-*",
        "extscache/omni.usd.schema.physx-*",
    )

    for root in isaac_roots:
        if not root.exists():
            continue
        for pattern in package_globs:
            for package_dir in sorted(root.glob(pattern)):
                package_path = str(package_dir)
                if package_path not in sys.path:
                    sys.path.append(package_path)


try:
    from pxr import Usd, UsdPhysics
except ModuleNotFoundError:
    _bootstrap_pxr_from_isaacsim()
    from pxr import Usd, UsdPhysics


DEFAULT_SOURCE_USD = Path("/home/user/rl_ws/openarm_bimanual/openarm_bimanual.usd")
DEFAULT_TARGET_USD = Path("/home/user/rl_ws/urdf/openarm_modular_dual/openarm_modular_dual.usd")
DEFAULT_JOINTS = ["openarm_left_joint1"]
COMPARE_ATTRS = [
    "physics:axis",
    "physics:localPos0",
    "physics:localPos1",
    "physics:localRot0",
    "physics:localRot1",
    "physics:jointEnabled",
    "physics:lowerLimit",
    "physics:upperLimit",
    "physics:jointEquivalentInertia",
    "state:angular:physics:position",
    "state:angular:physics:velocity",
    "state:linear:physics:position",
    "state:linear:physics:velocity",
    "drive:angular:type",
    "drive:angular:targetPosition",
    "drive:angular:targetVelocity",
    "drive:angular:stiffness",
    "drive:angular:damping",
    "drive:angular:maxForce",
    "drive:linear:type",
    "drive:linear:targetPosition",
    "drive:linear:targetVelocity",
    "drive:linear:stiffness",
    "drive:linear:damping",
    "drive:linear:maxForce",
]
COMPARE_RELS = [
    "physics:body0",
    "physics:body1",
]
COMPARE_XFORM_ATTRS = [
    "xformOp:translate",
    "xformOp:orient",
    "xformOp:rotateXYZ",
    "xformOp:scale",
    "xformOpOrder",
]


def _iter_joint_prims(stage: Usd.Stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            yield prim


def _find_joint_by_name(stage: Usd.Stage, joint_name: str):
    for prim in _iter_joint_prims(stage):
        if prim.GetName() == joint_name:
            return prim
    return None


def _format_value(value) -> str:
    if value is None:
        return "<none>"
    return str(value)


def _get_attr_value(prim, attr_name: str):
    attr = prim.GetAttribute(attr_name)
    if not attr:
        return None, False
    return attr.Get(), attr.HasAuthoredValueOpinion()


def _get_rel_targets(prim, rel_name: str):
    rel = prim.GetRelationship(rel_name)
    if not rel:
        return [], False
    targets = [str(path) for path in rel.GetTargets()]
    return targets, rel.HasAuthoredTargets()


def _report_xform_compare(source_stage: Usd.Stage, target_stage: Usd.Stage, source_path: str, target_path: str, label: str) -> None:
    source_prim = source_stage.GetPrimAtPath(source_path)
    target_prim = target_stage.GetPrimAtPath(target_path)
    print(f"\n  [{label}]")
    print(f"  source link: {source_path}")
    print(f"  target link: {target_path}")
    if not source_prim or not source_prim.IsValid():
        print("  source link prim not found")
        return
    if not target_prim or not target_prim.IsValid():
        print("  target link prim not found")
        return

    mismatch_count = 0
    for attr_name in COMPARE_XFORM_ATTRS:
        source_value, source_authored = _get_attr_value(source_prim, attr_name)
        target_value, target_authored = _get_attr_value(target_prim, attr_name)
        matches = source_value == target_value
        state = "MATCH" if matches else "DIFF"
        if not matches:
            mismatch_count += 1
        print(
            f"  {state:>5} {attr_name}\n"
            f"    source ({'authored' if source_authored else 'fallback'}): {_format_value(source_value)}\n"
            f"    target ({'authored' if target_authored else 'fallback'}): {_format_value(target_value)}"
        )
    if mismatch_count == 0:
        print("  result: link xform matches")
    else:
        print(f"  result: {mismatch_count} differing xform attributes")


def _report_joint(source_stage: Usd.Stage, target_stage: Usd.Stage, joint_name: str) -> None:
    source_prim = _find_joint_by_name(source_stage, joint_name)
    target_prim = _find_joint_by_name(target_stage, joint_name)

    print(f"\n=== {joint_name} ===")
    if source_prim is None:
        print(f"source: joint not found")
        return
    if target_prim is None:
        print(f"target: joint not found")
        return

    print(f"source prim: {source_prim.GetPath()}")
    print(f"target prim: {target_prim.GetPath()}")
    print(f"source type: {source_prim.GetTypeName()}")
    print(f"target type: {target_prim.GetTypeName()}")

    mismatch_count = 0
    for attr_name in COMPARE_ATTRS:
        source_value, source_authored = _get_attr_value(source_prim, attr_name)
        target_value, target_authored = _get_attr_value(target_prim, attr_name)
        matches = source_value == target_value
        state = "MATCH" if matches else "DIFF"
        if not matches:
            mismatch_count += 1
        print(
            f"{state:>5} {attr_name}\n"
            f"  source ({'authored' if source_authored else 'fallback'}): {_format_value(source_value)}\n"
            f"  target ({'authored' if target_authored else 'fallback'}): {_format_value(target_value)}"
        )

    source_bodies: dict[str, list[str]] = {}
    target_bodies: dict[str, list[str]] = {}
    for rel_name in COMPARE_RELS:
        source_targets, source_authored = _get_rel_targets(source_prim, rel_name)
        target_targets, target_authored = _get_rel_targets(target_prim, rel_name)
        source_bodies[rel_name] = source_targets
        target_bodies[rel_name] = target_targets
        matches = source_targets == target_targets
        state = "MATCH" if matches else "DIFF"
        if not matches:
            mismatch_count += 1
        print(
            f"{state:>5} {rel_name}\n"
            f"  source ({'authored' if source_authored else 'fallback'}): {_format_value(source_targets)}\n"
            f"  target ({'authored' if target_authored else 'fallback'}): {_format_value(target_targets)}"
        )

    if mismatch_count == 0:
        print("result: joint frame attributes match")
    else:
        print(f"result: {mismatch_count} differing attributes")

    for rel_name, label in [("physics:body0", "body0 xform"), ("physics:body1", "body1 xform")]:
        source_targets = source_bodies.get(rel_name, [])
        target_targets = target_bodies.get(rel_name, [])
        if len(source_targets) == 1 and len(target_targets) == 1:
            _report_xform_compare(source_stage, target_stage, source_targets[0], target_targets[0], label)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-usd",
        type=Path,
        default=DEFAULT_SOURCE_USD,
        help=f"Source USD. Default: {DEFAULT_SOURCE_USD}",
    )
    parser.add_argument(
        "--target-usd",
        type=Path,
        default=DEFAULT_TARGET_USD,
        help=f"Target USD. Default: {DEFAULT_TARGET_USD}",
    )
    parser.add_argument(
        "--joint",
        action="append",
        dest="joints",
        help="Joint name to compare. Repeat to compare multiple joints.",
    )
    parser.add_argument(
        "--quit-app",
        action="store_true",
        help="If running inside Isaac Sim app startup, request the app to quit after printing results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    joints = args.joints or DEFAULT_JOINTS

    source_stage = Usd.Stage.Open(str(args.source_usd))
    if source_stage is None:
        raise RuntimeError(f"Failed to open source USD: {args.source_usd}")

    target_stage = Usd.Stage.Open(str(args.target_usd))
    if target_stage is None:
        raise RuntimeError(f"Failed to open target USD: {args.target_usd}")

    print(f"source usd: {args.source_usd}")
    print(f"target usd: {args.target_usd}")

    for joint_name in joints:
        _report_joint(source_stage, target_stage, joint_name)

    if args.quit_app:
        try:
            import omni.kit.app

            omni.kit.app.get_app().post_quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()

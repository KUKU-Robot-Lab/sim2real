"""
Copy per-joint USD physics drive attributes from source USDs into a target robot USD.

This is intended for the case where the target robot was assembled from a custom
URDF/Xacro, but you want to reuse the drive tuning from existing OpenArm and DG5F
USD assets.

Run this with Isaac Sim's Python environment (or any Python with `pxr` schemas):

    ./python.sh copy_joint_physics_from_usd.py

By default it copies joint drive attributes from the composed source assets:
    - /home/user/rl_ws/openarm_bimanual/openarm_bimanual.usd
    - /home/user/rl_ws/urdf/delto_m_ros2/dg_isaacsim/dg5f_right/dg5f_right.usd

Into:
    - /home/user/rl_ws/urdf/openarm_modular_dual/openarm_modular_dual.usd

Matching is done by joint prim name, not by prim path. That allows the target USD
to keep a different hierarchy as long as the relevant joint names are preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys


def _bootstrap_pxr_from_isaacsim() -> None:
    isaac_roots = [
        Path("/home/user/isaacsim/5.1.0"),
        Path("/home/user/isaacsim/5.0.0"),
    ]
    package_globs = (
        "extscache/omni.usd.libs-*/",
        "extscache/omni.usd.schema.physx-*/",
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
    from pxr import PhysxSchema, Usd, UsdPhysics
except ModuleNotFoundError:
    _bootstrap_pxr_from_isaacsim()
    from pxr import PhysxSchema, Usd, UsdPhysics


DEFAULT_TARGET_USD = Path("/home/user/rl_ws/urdf/openarm_modular_dual/openarm_modular_dual.usd")
DEFAULT_SOURCE_USDS = [
    Path("/home/user/rl_ws/openarm_bimanual/openarm_bimanual.usd"),
    Path("/home/user/rl_ws/urdf/delto_m_ros2/dg_isaacsim/dg5f_right/dg5f_right.usd"),
]
ATTR_PREFIXES = ("drive:angular:", "drive:linear:", "physxJoint:")
DEFAULT_SOURCE_ALIASES = {
    "openarm_left_finger_joint2": "openarm_left_finger_joint1",
}


@dataclass
class JointPhysicsSpec:
    source_file: str
    joint_name: str
    attr_values: dict[str, tuple[object, object, bool]] = field(default_factory=dict)
    has_angular_drive: bool = False
    has_linear_drive: bool = False
    has_physx_joint_api: bool = False


def _iter_joint_prims(stage: Usd.Stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            yield prim


def _collect_joint_specs(source_usd: Path) -> dict[str, JointPhysicsSpec]:
    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open source USD: {source_usd}")

    specs: dict[str, JointPhysicsSpec] = {}
    for prim in _iter_joint_prims(stage):
        joint_name = prim.GetName()
        spec = JointPhysicsSpec(
            source_file=str(source_usd),
            joint_name=joint_name,
            has_angular_drive=bool(UsdPhysics.DriveAPI(prim, "angular")),
            has_linear_drive=bool(UsdPhysics.DriveAPI(prim, "linear")),
            has_physx_joint_api=bool(PhysxSchema.PhysxJointAPI(prim)),
        )

        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            if not attr.HasAuthoredValueOpinion():
                continue
            if not attr_name.startswith(ATTR_PREFIXES):
                continue
            spec.attr_values[attr_name] = (attr.Get(), attr.GetTypeName(), attr.IsCustom())

        if spec.attr_values:
            specs[joint_name] = spec

    return specs


def _ensure_required_apis(target_prim, spec: JointPhysicsSpec) -> None:
    if spec.has_angular_drive:
        UsdPhysics.DriveAPI.Apply(target_prim, "angular")
    if spec.has_linear_drive:
        UsdPhysics.DriveAPI.Apply(target_prim, "linear")
    if spec.has_physx_joint_api:
        PhysxSchema.PhysxJointAPI.Apply(target_prim)


def _copy_joint_attrs(target_prim, spec: JointPhysicsSpec, dry_run: bool) -> int:
    _ensure_required_apis(target_prim, spec)

    copied = 0
    for attr_name, (value, type_name, is_custom) in sorted(spec.attr_values.items()):
        if value is None:
            continue

        target_attr = target_prim.GetAttribute(attr_name)
        if not target_attr:
            target_attr = target_prim.CreateAttribute(attr_name, type_name, is_custom)

        if not dry_run:
            target_attr.Set(value)
        copied += 1

    return copied


def _build_source_map(source_usds: list[Path]) -> tuple[dict[str, JointPhysicsSpec], dict[str, list[str]]]:
    source_map: dict[str, JointPhysicsSpec] = {}
    collisions: dict[str, list[str]] = {}

    for source_usd in source_usds:
        specs = _collect_joint_specs(source_usd)
        for joint_name, spec in specs.items():
            if joint_name in source_map:
                collisions.setdefault(joint_name, [source_map[joint_name].source_file]).append(spec.source_file)
            source_map[joint_name] = spec

    return source_map, collisions


def copy_joint_physics(
    target_usd: Path,
    source_usds: list[Path],
    dry_run: bool,
    source_aliases: dict[str, str] | None = None,
) -> None:
    source_map, collisions = _build_source_map(source_usds)
    if not source_map:
        raise RuntimeError("No joint physics attributes were found in the source USDs.")
    source_aliases = source_aliases or {}

    target_stage = Usd.Stage.Open(str(target_usd))
    if target_stage is None:
        raise RuntimeError(f"Failed to open target USD: {target_usd}")

    matched_joints = 0
    copied_attrs = 0
    missing_target_specs = set(source_map)
    unmatched_target_joints: list[str] = []

    for target_prim in _iter_joint_prims(target_stage):
        joint_name = target_prim.GetName()
        spec = source_map.get(joint_name)
        source_joint_name = joint_name
        if spec is None:
            source_joint_name = source_aliases.get(joint_name, joint_name)
            spec = source_map.get(source_joint_name)
        if spec is None:
            unmatched_target_joints.append(joint_name)
            continue

        copied_here = _copy_joint_attrs(target_prim, spec, dry_run=dry_run)
        if copied_here:
            matched_joints += 1
            copied_attrs += copied_here
            missing_target_specs.discard(source_joint_name)
            alias_note = "" if source_joint_name == joint_name else f" via alias {source_joint_name}"
            print(
                f"[copied] {joint_name} <- {spec.source_file} "
                f"({copied_here} attrs to {target_prim.GetPath()}{alias_note})"
            )

    if collisions:
        print("[warning] duplicate joint names existed across source USDs; later source won:")
        for joint_name, source_files in sorted(collisions.items()):
            print(f"  - {joint_name}: {', '.join(source_files)}")

    if missing_target_specs:
        print("[warning] source joint specs with no matching target joint:")
        for joint_name in sorted(missing_target_specs):
            spec = source_map[joint_name]
            print(f"  - {joint_name} (from {spec.source_file})")

    if unmatched_target_joints:
        print("[info] target joints with no source physics match:")
        for joint_name in sorted(unmatched_target_joints):
            print(f"  - {joint_name}")

    if dry_run:
        print(
            f"[dry-run] would copy {copied_attrs} attrs across {matched_joints} joints into {target_usd}"
        )
        return

    target_stage.GetRootLayer().Save()
    print(f"[saved] copied {copied_attrs} attrs across {matched_joints} joints into {target_usd}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-usd",
        type=Path,
        default=DEFAULT_TARGET_USD,
        help=f"Target USD to modify in place. Default: {DEFAULT_TARGET_USD}",
    )
    parser.add_argument(
        "--source-usd",
        type=Path,
        action="append",
        dest="source_usds",
        help=(
            "Source physics USD to copy from. Repeat this option to add multiple sources. "
            "If omitted, the built-in OpenArm + DG5F defaults are used."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect matches and print what would be copied without saving the target USD.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_usds = args.source_usds or DEFAULT_SOURCE_USDS
    copy_joint_physics(
        target_usd=args.target_usd,
        source_usds=source_usds,
        dry_run=bool(args.dry_run),
        source_aliases=DEFAULT_SOURCE_ALIASES,
    )


if __name__ == "__main__":
    main()

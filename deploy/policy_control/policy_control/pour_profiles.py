"""Build-time loader for the hdgp bimanual pair profiles (pure data, no Isaac).

``openarm/__init__`` pulls in Isaac Lab, so the three pure-data modules are loaded by file
path under stub parent packages. Used only by ``contract_build_pour``; the runtime nodes read
the resulting contract and never import hdgp.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_PKGS = ("openarm", "openarm.agnostic", "openarm.agnostic.modules")
_MODULES = (
    ("openarm.agnostic.modules.vendor_gains", "modules/vendor_gains.py"),
    ("openarm.agnostic.modules.robot_profiles", "modules/robot_profiles.py"),
    ("_pour_fabric_bimanual", "tasks/pour_fabric/bimanual.py"),
)


class ProfileLoadError(RuntimeError):
    pass


def load_pair(hdgp_root: Path, pair_name: str):
    """Return the hdgp ``BimanualPair`` (source/receiver RobotProfile) named ``pair_name``."""
    def pick(mods):
        try:
            return mods[-1].get_pair(pair_name)
        except KeyError as exc:
            raise ProfileLoadError(str(exc)) from exc
    return _with_modules(hdgp_root, _MODULES, pick)


_MIMIC_PKGS = ("openarm.agnostic.tasks", "openarm.agnostic.tasks.pour_fabric_mimic")
_MIMIC_MODULES = (
    ("openarm.agnostic.modules.vendor_gains", "modules/vendor_gains.py"),
    ("openarm.agnostic.modules.robot_profiles", "modules/robot_profiles.py"),
    ("openarm.agnostic.tasks.pour_fabric_mimic.robot_profiles", "tasks/pour_fabric_mimic/robot_profiles.py"),
    ("openarm.agnostic.tasks.pour_fabric_mimic.bimanual", "tasks/pour_fabric_mimic/bimanual.py"),
)


def load_mimic_pair(hdgp_root: Path, pair_name: str = "rh"):
    """RH56F1 양팔 쌍(hdgp tasks/pour_fabric_mimic/bimanual.py) — pour_fj 계약 빌드용(09.29). 패키지 상대 import 를 쓰므로
    task 패키지까지 스텁을 둔다."""
    def pick(mods):
        try:
            return mods[-1].get_pair(pair_name)
        except KeyError as exc:
            raise ProfileLoadError(str(exc)) from exc
    return _with_modules(hdgp_root, _MIMIC_MODULES, pick, extra_pkgs=_MIMIC_PKGS)


def load_profile(hdgp_root: Path, profile_name: str):
    """Return one hdgp ``RobotProfile`` (``robot_profiles.PROFILES[name]``) — joint family 계약 빌드용."""
    def pick(mods):
        profiles = mods[-1].PROFILES
        if profile_name not in profiles:
            raise ProfileLoadError(f"hdgp profile {profile_name!r} not found (have {sorted(profiles)})")
        return profiles[profile_name]
    return _with_modules(hdgp_root, _MODULES[:2], pick)


def _with_modules(hdgp_root: Path, modules: tuple, pick, extra_pkgs: tuple = ()):
    base = Path(hdgp_root) / "source" / "openarm" / "openarm" / "agnostic"
    names = _PKGS + tuple(extra_pkgs) + tuple(n for n, _ in modules)
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for pkg in _PKGS + tuple(extra_pkgs):
            stub = types.ModuleType(pkg)
            stub.__path__ = []
            sys.modules[pkg] = stub
        loaded = []
        for name, rel in modules:
            path = base / rel
            if not path.is_file():
                raise ProfileLoadError(f"hdgp profile module missing: {path}")
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            loaded.append(mod)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], leaf, mod)
        return pick(loaded)
    finally:
        for n, old in saved.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old

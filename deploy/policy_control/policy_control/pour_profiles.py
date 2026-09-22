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
    base = Path(hdgp_root) / "source" / "openarm" / "openarm" / "agnostic"
    names = _PKGS + tuple(n for n, _ in _MODULES)
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for pkg in _PKGS:
            stub = types.ModuleType(pkg)
            stub.__path__ = []
            sys.modules[pkg] = stub
        mod = None
        for name, rel in _MODULES:
            path = base / rel
            if not path.is_file():
                raise ProfileLoadError(f"hdgp profile module missing: {path}")
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], leaf, mod)
        try:
            return mod.get_pair(pair_name)
        except KeyError as exc:
            raise ProfileLoadError(str(exc)) from exc
    finally:
        for n, old in saved.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old

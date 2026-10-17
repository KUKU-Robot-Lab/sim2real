"""배포가 hdgp 저장소 코드를 그대로 부를 때(tilt 기본 궤적 · ready_carry 이송) 그 파일들의 md5 를 고정한다.

10.17: 실기 tilt 궤적은 학습과 같은 계산(hdgp scripts/tools/pour_bank/tilt_real_start.py)으로, 잡기 뒤 이송은
ready_carry(hdgp agnostic/modules/ready_carry)로 한다. hdgp 공유 작업 트리는 다른 세션이 계속 고치므로, 자산
policy.yaml 의 hdgp.md5 와 지금 파일이 하나라도 다르면 계획하지 않는다(움직이기 전에 멈춘다). 핀을 바꾸는 것은
사람이 diff 를 보고 policy.yaml 을 고치는 일이다.

hdgp 위치 = 환경변수 HDGP_DIR, 없으면 ~/rl_ws/hdgp.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Mapping

import yaml

READY_CARRY_DIR = "source/openarm/openarm/agnostic/modules/ready_carry"
READY_CARRY_PKG = "ready_carry_hdgp"      # openarm 패키지 초기화(학습 과제 설정 import)를 거치지 않게 따로 이름을 붙인다


class PinError(RuntimeError):
    pass


def hdgp_dir() -> Path:
    return Path(os.environ.get("HDGP_DIR", "~/rl_ws/hdgp")).expanduser().resolve()


def md5(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_policy(asset_dir: str | Path) -> dict:
    p = Path(asset_dir) / "policy.yaml"
    if not p.is_file():
        raise PinError(f"{p} 가 없다")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    pins = (cfg.get("hdgp") or {}).get("md5")
    if not isinstance(pins, Mapping) or not pins:
        raise PinError(f"{p}: hdgp.md5 (hdgp 루트 기준 상대 경로 → md5) 가 없다")
    return cfg


def verify(root: Path, pins: Mapping[str, str]) -> None:
    """핀 파일이 모두 있고 md5 가 같아야 한다. 다르면 무엇이 다른지 모아서 PinError."""
    bad = []
    for rel, want in pins.items():
        p = root / rel
        if not p.is_file():
            bad.append(f"{rel}: 없음")
        elif md5(p) != str(want):
            bad.append(f"{rel}: {md5(p)} ≠ 고정 {want}")
    if bad:
        raise PinError(f"hdgp({root}) 코드가 자산에 고정한 것과 다르다 — " + "; ".join(bad))


def verify_refs(asset_dir: Path, refs: Mapping[str, str]) -> None:
    bad = [f"{name}: {'없음' if not (asset_dir / 'ref' / name).is_file() else md5(asset_dir / 'ref' / name)} ≠ {want}"
           for name, want in refs.items()
           if not (asset_dir / "ref" / name).is_file() or md5(asset_dir / "ref" / name) != str(want)]
    if bad:
        raise PinError("기준 파일이 자산 policy.yaml 의 md5 와 다르다 — " + "; ".join(bad))


def load_ready_carry(root: Path) -> ModuleType:
    """hdgp ready_carry 폴더를 독립 패키지(READY_CARRY_PKG)로 읽는다 → 패키지 모듈(.mover · .posture · .chain 하위 모듈 포함)."""
    if READY_CARRY_PKG in sys.modules:
        return sys.modules[READY_CARRY_PKG]
    d = root / READY_CARRY_DIR
    spec = importlib.util.spec_from_file_location(READY_CARRY_PKG, d / "__init__.py", submodule_search_locations=[str(d)])
    if spec is None or spec.loader is None:
        raise PinError(f"{d} 를 읽지 못했다")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[READY_CARRY_PKG] = pkg
    spec.loader.exec_module(pkg)
    for sub in ("chain", "posture", "mover"):
        importlib.import_module(f"{READY_CARRY_PKG}.{sub}")
    return pkg

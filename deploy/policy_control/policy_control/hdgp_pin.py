"""배포가 hdgp 저장소 코드를 그대로 부를 때(tilt 기본 궤적 · ready_carry 이송) 그 파일들의 md5 를 고정한다.

10.17: 실기 tilt 궤적은 학습과 같은 계산(hdgp scripts/tools/pour_bank/tilt_real_start.py)으로, 잡기 뒤 이송은
ready_carry(hdgp agnostic/modules/ready_carry)로 한다. hdgp 공유 작업 트리는 다른 세션이 계속 고치므로, 자산
policy.yaml 의 hdgp.md5 와 지금 파일이 하나라도 다르면 계획하지 않는다(움직이기 전에 멈춘다). 핀을 바꾸는 것은
사람이 diff 를 보고 policy.yaml 을 고치는 일이다.

hdgp 위치 = 환경변수 HDGP_DIR, 없으면 ~/rl_ws/hdgp.

★10.20 커밋 고정 읽기(사용자 "권장방법으로 진행"): 10.20 02:03 aglt_new 가 공유 작업 트리의 ready_approach.py 를 고친 순간
실기 1-B reach 가 md5 핀에 걸려 두 번 멈췄다. 이제 다른 세션이 자주 고치는 코드(config/hdgp_snapshot.yaml 의 paths)는
작업 트리가 아니라 고정 커밋에서 꺼낸 사본(~/.cache/s2r_hdgp/<커밋>)을 읽는다 — 작업 트리를 고쳐도(커밋 전이라도) 실기는
그대로다. 그 밖의 경로(URDF · 메쉬 · tilt 스크립트)는 예전처럼 작업 트리 + md5 핀. md5 핀은 사본에도 그대로 건다(이중 확인).
새 코드로 올리기 = 옛 · 새 커밋으로 같은 자리를 계획해 비교(HDGP_SNAPSHOT_COMMIT 로 바꿔 돌린다) → 커밋 · md5 핀을 같이 고친다.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Mapping

import yaml

READY_CARRY_DIR = "source/openarm/openarm/agnostic/modules/ready_carry"
READY_CARRY_PKG = "ready_carry_hdgp"      # openarm 패키지 초기화(학습 과제 설정 import)를 거치지 않게 따로 이름을 붙인다


class PinError(RuntimeError):
    pass


SNAPSHOT_CFG = Path(__file__).resolve().parents[1] / "config" / "hdgp_snapshot.yaml"
#: 사본 안에서 작업 트리로 잇는 꼭대기 폴더 — ready_carry chain.default_urdf 는 자기 위 폴더에서 assets/ 를 찾는다
SNAPSHOT_LINKS = ("assets",)


def hdgp_tree() -> Path:
    """hdgp 작업 트리(공유) 루트."""
    return Path(os.environ.get("HDGP_DIR", "~/rl_ws/hdgp")).expanduser().resolve()


def hdgp_dir() -> "HdgpView":
    """배포가 읽는 hdgp — 고정 경로는 커밋 사본, 나머지는 작업 트리(HdgpView). Path 처럼 `/` · os.fspath 를 쓴다."""
    root = hdgp_tree()
    cfg = load_snapshot_cfg()
    commit = os.environ.get("HDGP_SNAPSHOT_COMMIT") or cfg["commit"]
    return HdgpView(root, snapshot(root, commit, cfg["paths"]), cfg["paths"], commit)


def load_snapshot_cfg(path: Path = SNAPSHOT_CFG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    paths = cfg.get("paths")
    if not isinstance(cfg.get("commit"), str) or not isinstance(paths, list) or not paths:
        raise PinError(f"{path}: commit(문자열) · paths(hdgp 기준 경로 목록)가 필요하다")
    if any(not isinstance(p, str) or p.startswith("/") or ".." in Path(p).parts for p in paths):
        raise PinError(f"{path}: paths 는 hdgp 루트 기준 상대 경로여야 한다")
    return cfg


def _under(rel: str, prefixes) -> bool:
    rel = str(rel).strip("/")
    return any(rel == p.strip("/") or rel.startswith(p.strip("/") + "/") for p in prefixes)


class HdgpView(os.PathLike):
    """hdgp 작업 트리 + 고정 커밋 사본. `view / rel` 은 rel 이 고정 경로(paths) 아래면 사본, 아니면 작업 트리.
    os.fspath · str · Path(view) 는 작업 트리 루트(자산 · 기록용). 그 밖의 Path 속성(is_dir 등)도 작업 트리에 넘긴다."""

    def __init__(self, root: Path, snap: Path, paths, commit: str):
        self.root, self.snap, self.paths, self.commit = Path(root), Path(snap), tuple(paths), str(commit)

    def __truediv__(self, rel) -> Path:
        return (self.snap if _under(str(rel), self.paths) else self.root) / rel

    def __fspath__(self) -> str:
        return str(self.root)

    def __str__(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:
        return f"HdgpView({self.root}, 고정 {self.commit[:9]} {list(self.paths)})"

    def __getattr__(self, name):
        return getattr(self.root, name)


def snapshot(root: Path, commit: str, paths, cache: Path | None = None) -> Path:
    """고정 커밋의 paths 를 `git archive` 로 꺼낸 읽기 전용 사본(있으면 그대로) → 사본 루트.
    작업 트리를 건드리지 않는다(index · HEAD 그대로). 커밋이 없으면 PinError(움직이기 전에 멈춘다)."""
    cache = Path(cache or os.environ.get("S2R_HDGP_CACHE", "~/.cache/s2r_hdgp")).expanduser()
    root = Path(root)

    def git(*args, **kw):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False, **kw)

    rev = git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}", text=True)
    if rev.returncode != 0:
        raise PinError(f"hdgp({root}) 에 고정 커밋 {commit} 이 없다 — 로컬 hdgp 에서 git fetch 뒤 다시")
    sha = rev.stdout.strip()
    key = sha + "-" + hashlib.md5("\n".join(sorted(paths)).encode()).hexdigest()[:8]
    dest = cache / key
    if (dest / ".complete").is_file():
        return dest
    cache.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache))
    try:
        tar = git("archive", "--format=tar", sha, "--", *paths)
        if tar.returncode != 0:
            raise PinError(f"hdgp {sha[:9]} 에서 {list(paths)} 를 꺼내지 못했다: {tar.stderr.decode(errors='replace')[-300:]}")
        x = subprocess.run(["tar", "-x", "-C", str(tmp)], input=tar.stdout, capture_output=True, check=False)
        if x.returncode != 0:
            raise PinError(f"사본 풀기 실패: {x.stderr.decode(errors='replace')[-300:]}")
        for link in SNAPSHOT_LINKS:
            if not (tmp / link).exists():
                (tmp / link).symlink_to(root / link)
        for f in tmp.rglob("*"):
            if f.is_file() and not f.is_symlink():
                f.chmod(0o444)
        (tmp / ".complete").write_text(f"{sha}\n" + "\n".join(paths) + "\n")
        try:
            tmp.rename(dest)
        except OSError:                                    # 다른 프로세스가 먼저 만들었다 — 그것을 쓴다
            if not (dest / ".complete").is_file():
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
    return dest


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

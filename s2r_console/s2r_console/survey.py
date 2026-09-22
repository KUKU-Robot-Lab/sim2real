"""바깥 세계 조사 — 파일이 있는지, 다이제스트가 무엇인지. 판정은 `mission_core` 가 한다.

`mission_run.gather_evidence` 와 **같은 `Evidence` 를 낸다**(테스트가 잠근다). 다른 점은 하나:
콘솔은 같은 파일을 몇 초마다 다시 보므로 (크기, mtime) 이 그대로면 해시를 다시 세지 않는다.

여기서 하나를 더 낸다 — 단계별 **승인의 근거(basis)**. 승인은 "이 계약·이 체크포인트로 움직여도 된다"는
말이므로, 그 파일이 바뀌면 승인도 죽어야 한다(`ledger.valid_approvals`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from . import _paths  # noqa: F401
from mission_core import Evidence, Mission  # noqa: E402


class DigestCache:
    """경로 → 다이제스트. (크기, mtime_ns) 가 같으면 다시 세지 않는다."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], tuple[int, int, str]] = {}

    def digest(self, path: Path, algo: str) -> str | None:
        try:
            st = path.stat()
        except OSError:
            return None
        if not path.is_file():
            return None
        key = (str(path), algo)
        hit = self._seen.get(key)
        if hit is not None and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return hit[2]
        h = hashlib.new(algo)  # noqa: S324 — 무결성 대조용이지 보안용이 아니다
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        self._seen[key] = (st.st_size, st.st_mtime_ns, h.hexdigest())
        return self._seen[key][2]


def gather(mission: Mission, *, repo: Path, approvals: frozenset[str], cache: DigestCache) -> Evidence:
    present = frozenset(k for k, rel in mission.artifacts.items() if (repo / rel).exists())
    digests, params = {}, set()
    for key, spec in mission.checkpoints.items():
        md5 = cache.digest(repo / spec["path"], "md5")
        if md5 is not None:
            digests[key] = md5
        if (repo / spec["params"]).is_dir():
            params.add(key)
    return Evidence(present_artifacts=present, checkpoint_digests=digests,
                    present_params=frozenset(params), approvals=approvals)


def stage_basis(mission: Mission, stage_id: str, *, repo: Path, cache: DigestCache) -> dict[str, str]:
    """이 단계의 승인이 기대고 있는 파일들의 다이제스트. 없는 파일은 `missing`, 디렉터리는 `dir`."""
    stage = next(s for s in mission.stages if s.id == stage_id)
    basis: dict[str, str] = {}
    for key in stage.artifacts:
        path = repo / mission.artifacts[key]
        if path.is_dir():
            basis[f"artifact:{key}"] = "dir"
        else:
            basis[f"artifact:{key}"] = cache.digest(path, "sha256") or "missing"
    for key in stage.checkpoints:
        basis[f"checkpoint:{key}"] = cache.digest(repo / mission.checkpoints[key]["path"], "md5") or "missing"
    return basis


def all_basis(mission: Mission, *, repo: Path, cache: DigestCache) -> Mapping[str, dict[str, str]]:
    return {s.id: stage_basis(mission, s.id, repo=repo, cache=cache) for s in mission.stages}

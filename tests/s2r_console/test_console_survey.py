"""survey.gather 는 mission_run.gather_evidence 와 **같은 Evidence** 를 낸다 — 콘솔과 CLI 의 가드가 같은 것을 본다."""
from __future__ import annotations

from pathlib import Path

import yaml

import mission_run
from mission_core import load_mission
from s2r_console import survey

SIM2REAL = Path(__file__).resolve().parents[2]


def _mission(name):
    return load_mission(yaml.safe_load((SIM2REAL / "config" / name).read_text()))


def test_same_evidence_as_the_cli_on_the_real_missions():
    for name in ("mission_pour_fake.yaml", "mission_policy_control.yaml"):
        mission = _mission(name)
        approvals = frozenset({mission.stages[-1].id})
        want = mission_run.gather_evidence(mission, repo=SIM2REAL, approvals=approvals)
        got = survey.gather(mission, repo=SIM2REAL, approvals=approvals, cache=survey.DigestCache())
        assert got == want, name


def test_digest_cache_rehashes_only_when_the_file_changes(tmp_path, monkeypatch):
    f = tmp_path / "a.bin"
    f.write_bytes(b"one")
    cache = survey.DigestCache()
    first = cache.digest(f, "md5")
    calls = []
    real_new = survey.hashlib.new
    monkeypatch.setattr(survey.hashlib, "new", lambda algo: calls.append(algo) or real_new(algo))
    assert cache.digest(f, "md5") == first and calls == []
    f.write_bytes(b"two!")
    assert cache.digest(f, "md5") != first and calls == ["md5"]


def test_digest_of_a_missing_file_or_a_directory_is_none(tmp_path):
    cache = survey.DigestCache()
    assert cache.digest(tmp_path / "none", "md5") is None
    assert cache.digest(tmp_path, "md5") is None


def test_stage_basis_names_every_file_the_stage_leans_on():
    mission = _mission("mission_pour_fake.yaml")
    basis = survey.stage_basis(mission, "chain", repo=SIM2REAL, cache=survey.DigestCache())
    assert set(basis) == {"artifact:contract_pour", "artifact:robot_fake", "checkpoint:pour"}
    assert all(v not in ("missing", "dir") for v in basis.values())


def test_a_missing_artifact_shows_up_as_missing_not_as_an_exception(tmp_path):
    mission = load_mission({"name": "m", "artifacts": {"c": "nope.json"},
                            "stages": [{"id": "s", "title": "t", "artifacts": ["c"]}]})
    assert survey.stage_basis(mission, "s", repo=tmp_path, cache=survey.DigestCache()) == {"artifact:c": "missing"}

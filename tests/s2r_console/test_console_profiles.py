"""프로파일 — 도메인 방어는 로드 시점에 걸린다."""
from __future__ import annotations

from pathlib import Path

import pytest

from s2r_console.profiles import REAL_DOMAIN, ProfileError, domain_reasons, parse, scan

SIM2REAL = Path(__file__).resolve().parents[2]


def raw(**over):
    base = {"schema": "s2r_console/profile/v1", "id": "p", "mission": "m.yaml",
            "domain": {"id": 97, "class": "fake"}, "status_nodes": ["obs", "pd"]}
    return {**base, **over}


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "m.yaml").write_text("name: x\n")
    return tmp_path


@pytest.mark.parametrize(("domain", "cls", "ok"), [
    (97, "fake", True), (REAL_DOMAIN, "real", True),
    (REAL_DOMAIN, "fake", False),   # ★fake 가 실기 그래프에 붙는다
    (0, "fake", False),             # 기본 도메인 — 누가 있을지 모른다
    (97, "real", False),            # 실기라면서 다른 도메인
    (300, "fake", False), (97, "sim", False),
])
def test_domain_table(domain, cls, ok):
    assert (domain_reasons(domain, cls) == []) is ok


def test_a_fake_profile_on_the_real_domain_does_not_load(repo):
    with pytest.raises(ProfileError, match="fake 에 금지"):
        parse(raw(domain={"id": REAL_DOMAIN, "class": "fake"}), path=repo / "p.yaml", repo=repo)


@pytest.mark.parametrize("over", [
    {"schema": "v0"}, {"extra": 1}, {"domain": {"id": 97}}, {"domain": {"id": "x", "class": "fake"}},
    {"status_nodes": []}, {"mission": "nope.yaml"}, {"policy": "no/such/dir"},
    {"latency": {"from": "obs", "to": "ghost"}}, {"latency": {"from": "obs"}},
])
def test_malformed_profiles_are_refused(repo, over):
    with pytest.raises(ProfileError):
        parse(raw(**over), path=repo / "p.yaml", repo=repo)


def test_missing_required_key(repo):
    bad = raw()
    del bad["status_nodes"]
    with pytest.raises(ProfileError, match="status_nodes"):
        parse(bad, path=repo / "p.yaml", repo=repo)


def test_scan_reports_broken_files_instead_of_hiding_them(repo):
    d = repo / "profiles"
    d.mkdir()
    (d / "good.yaml").write_text("schema: s2r_console/profile/v1\nid: good\nmission: m.yaml\ndomain: {id: 97, class: fake}\nstatus_nodes: [pd]\n")
    (d / "bad.yaml").write_text("schema: nope\n")
    good, bad = scan(d, repo=repo)
    assert [p.id for p in good] == ["good"] and list(bad) == ["bad.yaml"]


def test_every_committed_profile_loads_and_its_mission_parses():
    """저장소에 든 프로파일은 전부 읽혀야 하고, 가리키는 미션도 mission_core 가 받아야 한다."""
    import yaml
    from mission_core import load_mission
    from mission_stages import load_runbook

    good, bad = scan(SIM2REAL / "s2r_console" / "profiles", repo=SIM2REAL)
    assert bad == {} and good
    for p in good:
        doc = yaml.safe_load(p.mission.read_text())
        mission = load_mission(doc)
        load_runbook(doc.get("run", {}), mission)
        assert p.id == p.path.stem, "파일명과 id 가 같아야 화면에서 찾는다"


def test_fake_profiles_never_point_at_a_mission_that_touches_real():
    import yaml
    from mission_core import load_mission

    good, _ = scan(SIM2REAL / "s2r_console" / "profiles", repo=SIM2REAL)
    for p in good:
        if p.domain_class == "fake":
            mission = load_mission(yaml.safe_load(p.mission.read_text()))
            assert [s.id for s in mission.stages if s.touches_real] == [], p.id

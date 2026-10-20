"""★10.20 배포가 hdgp 코드를 고정 커밋 사본에서 읽는다(hdgp_pin.HdgpView · snapshot).

10.20 02:06 · 02:08 실기 1-B reach 가 두 번 멈췄다 — aglt_new 가 공유 작업 트리의 ready_approach.py 를 고친 순간 md5 핀에 걸렸다.
이제 작업 트리를 고쳐도(커밋 전이라도) 실기 계획은 고정 커밋 사본을 읽는다. 자산(URDF · 메쉬)은 작업 트리 + md5 핀 그대로.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest
import yaml

from policy_control import hdgp_pin as H

REPO = Path(__file__).resolve().parents[2]
CODE = "src/pkg/code.py"


def _git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    """작은 hdgp 흉내 — 코드 하나(src/pkg) · 자산 하나(assets) · 커밋 하나."""
    root = tmp_path / "hdgp"
    (root / "src/pkg").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / CODE).write_text("VALUE = 1\n")
    (root / "assets/robot.urdf").write_text("<robot/>\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c1")
    return root, _git(root, "rev-parse", "HEAD")


def test_view_reads_pinned_paths_from_the_commit_and_the_rest_from_the_tree(repo, tmp_path):
    root, sha = repo
    snap = H.snapshot(root, sha, ["src/pkg"], cache=tmp_path / "cache")
    (root / CODE).write_text("VALUE = 2  # 다른 세션이 작업 트리를 고쳤다(커밋 전)\n")
    v = H.HdgpView(root, snap, ["src/pkg"], sha)
    assert (v / CODE).read_text() == "VALUE = 1\n"                     # 실기는 커밋 사본
    assert v / "assets/robot.urdf" == root / "assets/robot.urdf"         # 자산은 작업 트리
    assert os.fspath(v) == str(v) == str(root) and Path(v) == root and v.is_dir()
    assert (snap / "assets").resolve() == (root / "assets").resolve()   # 사본 안 코드가 자기 위에서 assets 를 찾는다


def test_md5_pins_hold_on_the_snapshot_while_the_tree_changes(repo, tmp_path):
    root, sha = repo
    pins = {CODE: H.md5(root / CODE)}
    v = H.HdgpView(root, H.snapshot(root, sha, ["src/pkg"], cache=tmp_path / "c"), ["src/pkg"], sha)
    (root / CODE).write_text("VALUE = 3\n")
    H.verify(v, pins)                                                    # 예전엔 여기서 PinError → reach 정지
    with pytest.raises(H.PinError):
        H.verify(root, pins)                                             # 작업 트리를 직접 보면 걸린다(예전 동작)


def test_snapshot_is_read_only_and_reused(repo, tmp_path):
    root, sha = repo
    a = H.snapshot(root, sha, ["src/pkg"], cache=tmp_path / "c")
    assert not os.access(a / CODE, os.W_OK)
    assert (a / ".complete").read_text().startswith(sha)
    assert H.snapshot(root, sha, ["src/pkg"], cache=tmp_path / "c") == a
    assert len([p for p in (tmp_path / "c").iterdir()]) == 1             # 임시 폴더를 남기지 않는다


def test_a_new_commit_gets_its_own_snapshot(repo, tmp_path):
    root, sha1 = repo
    (root / CODE).write_text("VALUE = 4\n")
    _git(root, "commit", "-qam", "c2")
    sha2 = _git(root, "rev-parse", "HEAD")
    a = H.snapshot(root, sha1, ["src/pkg"], cache=tmp_path / "c")
    b = H.snapshot(root, sha2, ["src/pkg"], cache=tmp_path / "c")
    assert a != b and (a / CODE).read_text() == "VALUE = 1\n" and (b / CODE).read_text() == "VALUE = 4\n"


def test_missing_commit_refuses_before_planning(repo, tmp_path):
    root, _ = repo
    with pytest.raises(H.PinError, match="고정 커밋"):
        H.snapshot(root, "0" * 40, ["src/pkg"], cache=tmp_path / "c")


def test_missing_path_in_commit_refuses(repo, tmp_path):
    root, sha = repo
    with pytest.raises(H.PinError):
        H.snapshot(root, sha, ["src/nope"], cache=tmp_path / "c")
    assert not list((tmp_path / "c").iterdir())


def test_snapshot_cfg_rejects_absolute_or_parent_paths(tmp_path):
    for bad in (["/etc"], ["../x"], []):
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump({"commit": "abc", "paths": bad}))
        with pytest.raises(H.PinError):
            H.load_snapshot_cfg(p)


def test_env_commit_override_is_for_side_by_side_comparison(repo, tmp_path, monkeypatch):
    root, sha1 = repo
    (root / CODE).write_text("VALUE = 5\n")
    _git(root, "commit", "-qam", "c2")
    cfg = tmp_path / "snap.yaml"
    cfg.write_text(yaml.safe_dump({"commit": _git(root, "rev-parse", "HEAD"), "paths": ["src/pkg"]}))
    monkeypatch.setattr(H, "SNAPSHOT_CFG", cfg)
    monkeypatch.setenv("HDGP_DIR", str(root))
    monkeypatch.setenv("S2R_HDGP_CACHE", str(tmp_path / "c"))
    monkeypatch.setattr(H, "load_snapshot_cfg", lambda path=cfg: yaml.safe_load(Path(path).read_text()))
    assert (H.hdgp_dir() / CODE).read_text() == "VALUE = 5\n"
    monkeypatch.setenv("HDGP_SNAPSHOT_COMMIT", sha1)
    assert (H.hdgp_dir() / CODE).read_text() == "VALUE = 1\n"


def test_deploy_snapshot_commit_satisfies_every_pin():
    """실제 설정 — 고정 커밋 사본이 ready_approach · ready_carry md5 핀과 맞고, 자산 핀(URDF)은 작업 트리에서 맞는다."""
    if not H.hdgp_tree().is_dir():
        pytest.skip("hdgp 없음")
    v = H.hdgp_dir()
    ra = yaml.safe_load((REPO / "deploy/policy_control/config/ready_approach/rh56f1.yaml").read_text())
    H.verify(v, ra["hdgp_md5"])
    tilt = H.load_policy(REPO / "deploy/policy_control/config/tilt/base_arc6")
    try:
        H.verify(v, tilt["hdgp"]["md5"])
    except H.PinError as exc:                     # tilt 스크립트 · URDF 는 작업 트리 — 그쪽 세션이 고친 것이면 tilt 핀 문제다
        assert "ready_carry" not in str(exc) and "ready_approach" not in str(exc), exc
    for rel in ra["hdgp_md5"]:
        assert v.snap in (v / rel).parents

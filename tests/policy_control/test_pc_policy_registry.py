"""policies/ 규약 — 정책 하나가 '쓸 수 있는 한 벌'인지 점검하는 순수 로직."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from policy_control import policy_registry as R

SIM2REAL = Path(__file__).resolve().parents[2]
PTH = b"weights-v1"


def make(root: Path, pid: str = "p1", *, status: str = "candidate", note: str = "", contract: str = "",
         contract_md5: str | None = None) -> Path:
    d = root / pid
    (d / "nn").mkdir(parents=True)
    (d / "params").mkdir()
    (d / "nn" / "a.pth").write_bytes(PTH)
    for rel in R.PARAMS:
        (d / rel).write_text("x: 1\n")
    files = [{"local_rel": rel, "remote_rel": rel, "sha256": hashlib.sha256((d / rel).read_bytes()).hexdigest(),
              "size": (d / rel).stat().st_size} for rel in (*R.PARAMS, "nn/a.pth")]
    (d / R.MANIFEST).write_text(json.dumps({"host": "server", "files": files}))
    (d / R.CARD).write_text(yaml.safe_dump({"id": pid, "status": status, "task": "t", "side": "right", "note": note}))
    if contract:
        md5 = hashlib.md5(PTH).hexdigest() if contract_md5 is None else contract_md5
        doc = {"checkpoint_md5": md5} if contract == "pour_contract.json" else {"run": {"checkpoint_md5": md5}}
        (d / contract).write_text(json.dumps(doc))
    return d


def test_a_complete_candidate_has_no_issues(tmp_path):
    e = R.check(make(tmp_path))
    assert e.ok and e.checkpoint == "a.pth" and e.contract == "" and e.status == "candidate"


def test_two_checkpoints_is_an_issue_because_the_contract_builder_refuses_to_guess(tmp_path):
    d = make(tmp_path)
    (d / "nn" / "b.pth").write_bytes(b"other")
    e = R.check(d)
    assert e.checkpoint == "" and any("2 개" in i for i in e.issues)


def test_no_checkpoint_is_an_issue(tmp_path):
    d = make(tmp_path)
    (d / "nn" / "a.pth").unlink()
    assert any("0 개" in i for i in R.check(d).issues)


@pytest.mark.parametrize("rel", R.PARAMS)
def test_missing_params_is_an_issue(tmp_path, rel):
    d = make(tmp_path)
    (d / rel).unlink()
    assert any(rel in i for i in R.check(d).issues)


def test_missing_card_is_an_issue(tmp_path):
    d = make(tmp_path)
    (d / R.CARD).unlink()
    assert any(R.CARD in i for i in R.check(d).issues)


def test_card_id_must_match_the_directory(tmp_path):
    d = make(tmp_path)
    (d / R.CARD).write_text(yaml.safe_dump({"id": "other", "status": "candidate"}))
    assert any("디렉터리 이름" in i for i in R.check(d).issues)


def test_unknown_status_is_an_issue(tmp_path):
    assert any("status" in i for i in R.check(make(tmp_path, status="great")).issues)


def test_hold_needs_a_reason(tmp_path):
    assert any("note" in i for i in R.check(make(tmp_path, status="hold")).issues)
    assert R.check(make(tmp_path, "p2", status="hold", note="ckpt_gate REJECT")).ok


def test_a_file_changed_after_fetch_is_caught(tmp_path):
    d = make(tmp_path)
    (d / "params" / "env.yaml").write_text("x: 2\n")      # 같은 크기, 다른 내용
    assert any("sha256" in i for i in R.check(d).issues)
    assert R.check(d, deep=False).ok                        # 얕은 점검은 크기만 본다


def test_missing_manifest_is_an_issue(tmp_path):
    d = make(tmp_path)
    (d / R.MANIFEST).unlink()
    assert any(R.MANIFEST in i for i in R.check(d).issues)


@pytest.mark.parametrize("contract", R.CONTRACTS)
def test_contract_md5_is_read_from_both_schemas(tmp_path, contract):
    assert R.check(make(tmp_path, status="verified", contract=contract)).ok
    e = R.check(make(tmp_path, "p2", status="verified", contract=contract, contract_md5="0" * 32))
    assert any("md5" in i for i in e.issues)


def test_an_asset_only_contract_with_empty_md5_is_not_compared(tmp_path):
    assert R.check(make(tmp_path, contract="deploy_contract.json", contract_md5="")).ok


@pytest.mark.parametrize("status", R.NEEDS_CONTRACT)
def test_verified_or_deployed_without_a_contract_is_an_issue(tmp_path, status):
    assert any("계약이 없다" in i for i in R.check(make(tmp_path, status=status)).issues)


def test_two_contracts_is_an_issue(tmp_path):
    d = make(tmp_path, contract="pour_contract.json")
    (d / "deploy_contract.json").write_text("{}")
    assert any("계약이 둘" in i for i in R.check(d).issues)


def test_scan_is_sorted_and_skips_hidden_and_files(tmp_path):
    make(tmp_path, "b")
    make(tmp_path, "a")
    (tmp_path / ".staging_x").mkdir()
    (tmp_path / R.INDEX).write_text("")
    assert [e.id for e in R.scan(tmp_path)] == ["a", "b"]
    assert R.scan(tmp_path / "nope") == ()


def test_index_lists_every_policy_with_its_issues_and_notes(tmp_path):
    make(tmp_path, "good", status="hold", note="틸트 155°")
    (make(tmp_path, "bad") / R.CARD).unlink()
    text = R.render_index(R.scan(tmp_path))
    assert "| `good` | hold |" in text and "| ok |" in text
    assert "`bad`" in text and "✗" in text
    assert "- `good` — 틸트 155°" in text
    assert all(f"`{s}`" in text for s in R.STATUSES)


def test_the_real_policies_directory_is_clean():
    root = SIM2REAL / "policies"
    if not root.is_dir():
        pytest.skip("policies/ 가 아직 없다")
    bad = {e.id: e.issues for e in R.scan(root, deep=False) if not e.ok}
    assert not bad, bad

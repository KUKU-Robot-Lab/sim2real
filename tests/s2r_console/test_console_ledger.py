"""승인 원장 — 승인은 그때 본 파일에 대한 승인이다."""
from __future__ import annotations

import pytest

from s2r_console import ledger
from s2r_console.ledger import Entry

BASIS = {"move": {"artifact:contract": "aaaa1111"}}


def approve(stage="move", basis=None):
    return Entry("approve", stage, "op", "2026-09-21T10:00:00", BASIS["move"] if basis is None else basis)


def test_an_approval_with_the_same_basis_is_valid():
    assert ledger.valid_approvals([approve()], BASIS) == frozenset({"move"})


def test_a_changed_file_kills_the_approval_and_says_which():
    now = {"move": {"artifact:contract": "bbbb2222"}}
    assert ledger.valid_approvals([approve()], now) == frozenset()
    assert ledger.stale_reasons([approve()], now) == {"move": ["artifact:contract: 승인 aaaa1111 → 지금 bbbb2222"]}


def test_revoke_after_approve_wins():
    entries = [approve(), Entry("revoke", "move", "op", "t", {})]
    assert ledger.valid_approvals(entries, BASIS) == frozenset()
    assert ledger.stale_reasons(entries, BASIS) == {}


def test_approve_after_revoke_wins():
    entries = [approve(), Entry("revoke", "move", "op", "t", {}), approve()]
    assert ledger.valid_approvals(entries, BASIS) == frozenset({"move"})


def test_an_approval_for_a_stage_no_longer_in_the_mission_is_not_valid():
    assert ledger.valid_approvals([approve("gone")], BASIS) == frozenset()
    assert ledger.stale_reasons([approve("gone")], BASIS) == {"gone": ["이 단계가 지금 미션에 없다"]}


def test_a_corrupt_line_is_an_error_not_a_skipped_line():
    with pytest.raises(ValueError, match="2번째 줄"):
        ledger.parse(['{"kind":"approve","stage":"a"}', "{not json"])


def test_an_unknown_kind_is_an_error():
    with pytest.raises(ValueError, match="kind"):
        ledger.parse(['{"kind":"bless","stage":"a"}'])


def test_file_round_trip_is_append_only(tmp_path):
    path = tmp_path / "run" / "approvals.jsonl"
    ledger.append(path, approve())
    ledger.append(path, Entry("revoke", "move", "console", "t2", {}, "실행에 쓰였다"))
    got = ledger.read(path)
    assert [e.kind for e in got] == ["approve", "revoke"] and got[0].basis == BASIS["move"]
    assert len(path.read_text().splitlines()) == 2


def test_missing_file_reads_as_empty(tmp_path):
    assert ledger.read(tmp_path / "none.jsonl") == []

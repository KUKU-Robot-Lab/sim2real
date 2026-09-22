"""status_join 승격이 숫자를 바꾸지 않았는가 — 기록된 실행을 오라클로 쓴다.

`logs/policy_control/<run>/` 에는 같은 실행의 `status.jsonl`(원본 메시지)·`status.csv`(join 결과)·
`status_summary.txt`(요약 한 줄)가 함께 남아 있다. jsonl 을 새 코드로 다시 join 해서 csv·summary 와
같아야 한다. `fake_plant_run.sh` 가 그 요약 문구로 합격을 판정하므로 이 동등성은 계약이다.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from policy_control.status_join import (NODES, StatusJoiner, join_jsonl, nodes_from_fields,
                                        row_fields, summarize)

RUNS = Path(__file__).resolve().parents[2] / "logs" / "policy_control"


def _golden_runs():
    if not RUNS.is_dir():
        return []
    out = []
    for d in sorted(RUNS.iterdir()):
        if (d / "status.jsonl").is_file() and (d / "status.csv").is_file() \
                and (d / "status_summary.txt").is_file():
            out.append(d)
    return out


GOLDEN = _golden_runs()
pytestmark = pytest.mark.skipif(not GOLDEN, reason="기록된 fake 실행이 없다")


def _recorded_nodes(run: Path) -> tuple[str, ...]:
    """이 실행이 어떤 `--nodes` 로 찍혔는가 — 기록된 CSV 헤더가 유일한 출처다."""
    with (run / "status.csv").open(newline="") as fh:
        header = next(csv.reader(fh), [])
    return nodes_from_fields(header) or NODES


def _as_text(v) -> str:
    """CSV 는 전부 문자열이다. None → "" 로, 숫자는 str() 로 — csv.DictWriter 와 같은 규칙."""
    return "" if v is None else str(v)


@pytest.mark.golden
@pytest.mark.parametrize("run", GOLDEN, ids=lambda p: p.name)
def test_rejoining_the_jsonl_reproduces_the_committed_csv(run: Path):
    rows = join_jsonl((run / "status.jsonl").read_text(errors="replace").splitlines(),
                      _recorded_nodes(run))
    with (run / "status.csv").open(newline="") as fh:
        want = list(csv.DictReader(fh))
    assert row_fields(rows) == list(want[0].keys()) if want else True
    assert len(rows) == len(want), (len(rows), len(want))
    for got, ref in zip(rows, want):
        assert {k: _as_text(v) for k, v in got.items()} == ref


@pytest.mark.golden
@pytest.mark.parametrize("run", GOLDEN, ids=lambda p: p.name)
def test_summary_line_is_unchanged(run: Path):
    rows = join_jsonl((run / "status.jsonl").read_text(errors="replace").splitlines(),
                      _recorded_nodes(run))
    ref = (run / "status_summary.txt").read_text().strip().split(" → ")[0]  # 뒤는 csv 경로다
    # 기록 당시 --policy-dt 를 준 실행은 예산 문구가 붙어 있다. 그 값을 되살려 같은 조건으로 만든다.
    m = re.search(r"budget 0\.5·dt = ([\d.]+) ms", ref)
    policy_dt = float(m[1]) / 500 if m else None
    assert summarize(rows, policy_dt) == ref


def test_offer_ignores_messages_without_a_seq():
    j = StatusJoiner()
    assert j.offer("obs", {"node": "obs"}) is None
    assert len(j) == 0
    assert j.offer("obs", {"seq": 3, "ok": True}) == 3
    assert len(j) == 1


def test_latency_needs_both_ends():
    j = StatusJoiner()
    j.offer("obs", {"seq": 1, "t_pub_ns": 1_000_000})
    assert j.rows()[0]["latency_ms"] is None
    j.offer("pd", {"seq": 1, "t_pub_ns": 3_000_000})
    assert j.rows()[0]["latency_ms"] == pytest.approx(2.0)


def test_rows_are_sorted_by_seq_whatever_the_arrival_order():
    j = StatusJoiner()
    for s in (7, 2, 5):
        j.offer("obs", {"seq": s, "ok": True})
    assert [r["seq"] for r in j.rows()] == [2, 5, 7]


def test_reasons_are_joined_with_a_pipe_and_default_to_empty():
    j = StatusJoiner()
    j.offer("pd", {"seq": 1, "reasons": ["watchdog", "tracking error"]})
    row = j.rows()[0]
    assert row["pd_reasons"] == "watchdog|tracking error"
    assert row["obs_reasons"] == ""


def test_latest_takes_the_highest_seq_per_node():
    j = StatusJoiner()
    j.offer("obs", {"seq": 1, "phase": "idle"})
    j.offer("obs", {"seq": 9, "phase": "running"})
    j.offer("pd", {"seq": 4, "phase": "TRACKING"})
    latest = j.latest()
    assert latest["obs"]["phase"] == "running" and latest["pd"]["phase"] == "TRACKING"


def test_summarize_reports_missing_seqs():
    rows = [{"seq": 1, "latency_ms": None}, {"seq": 5, "latency_ms": None}]
    assert "seq missing 3" in summarize(rows, None)
    assert summarize([], None) == "no rows"


def test_summarize_shows_the_budget_when_a_policy_dt_is_given():
    rows = [{"seq": 1, "latency_ms": 4.0}]
    assert "budget 0.5·dt = 8.3 ms" in summarize(rows, 1 / 60)


def test_nodes_order_is_the_wire_order():
    assert NODES == ("obs", "policy", "fabric", "pd")


def test_nodes_are_read_back_from_the_header():
    j = StatusJoiner()
    j.offer("pour_node", {"seq": 1, "ok": True})
    assert nodes_from_fields(row_fields(j.rows(("pour_node",)))) == ("pour_node",)
    assert nodes_from_fields(row_fields(j.rows())) == NODES
    assert nodes_from_fields(["seq"]) == ()


def test_a_recording_of_other_nodes_is_not_read_as_empty():
    """회귀: pour 기록(`--nodes pour_node`)을 4노드 기본값으로 되읽으면 "no rows" 가 된다."""
    lines = ['{"topic": "pour_node", "seq": 1, "ok": true}', '{"topic": "pour_node", "seq": 2, "ok": true}']
    assert join_jsonl(lines) == []
    assert [r["seq"] for r in join_jsonl(lines, ("pour_node",))] == [1, 2]

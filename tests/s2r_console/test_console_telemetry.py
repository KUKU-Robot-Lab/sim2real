"""화면의 지연 숫자는 CSV 도구(status_join)의 숫자와 같아야 한다."""
from __future__ import annotations

from pathlib import Path

import pytest

from policy_control.status_join import NODES, iter_jsonl, join_jsonl, summarize
from s2r_console.telemetry import Window, bucket_max, p50, p95

SIM2REAL = Path(__file__).resolve().parents[2]
def _has_latency(path: Path) -> bool:
    summary = path.with_name("status_summary.txt")
    return summary.is_file() and "latency p50 nan" not in summary.read_text(errors="replace") and "latency p50" in summary.read_text(errors="replace")


#: 지연이 실제로 기록된 run 만 — 나머지(t_pub_ns 이전의 기록)로는 이 대조가 공허하다.
RUNS = sorted(p for p in (SIM2REAL / "logs" / "policy_control").glob("fake_run*/status.jsonl") if _has_latency(p))


def _payload(node, seq, t_ms, ep=1, proc=0.5):
    return {"node": node, "episode": ep, "seq": seq, "t_pub_ns": int(t_ms * 1e6), "proc_ms": proc}


def test_latency_is_last_minus_first_in_ms():
    w = Window(("obs", "pd"))
    for seq in range(3):
        w.offer("obs", _payload("obs", seq, 100 * seq))
        w.offer("pd", _payload("pd", seq, 100 * seq + 2 + seq))
    m = w.metrics(policy_dt=0.02)
    assert m["latency_ms"] == {"p50": 3.0, "p95": 3.0, "max": 4.0} and m["budget_ms"] == 10.0 and m["seq_missing"] == 0


def test_seq_gap_is_counted():
    w = Window(("obs", "pd"))
    for seq in (0, 1, 4):
        w.offer("obs", _payload("obs", seq, seq))
    assert w.metrics(policy_dt=None)["seq_missing"] == 2


def test_negative_seq_from_pd_outside_an_episode_is_ignored():
    w = Window(("obs", "pd"))
    w.offer("pd", {"seq": -1, "t_pub_ns": 1})
    assert w.metrics(policy_dt=None)["rows"] == 0


def test_a_new_episode_on_the_first_node_clears_the_window():
    w = Window(("obs", "pd"))
    w.offer("obs", _payload("obs", 500, 1, ep=1))
    w.offer("obs", _payload("obs", 0, 2, ep=2))
    m = w.metrics(policy_dt=None)
    assert (m["rows"], m["episode"], m["seq_missing"]) == (1, 2, 0)


def test_the_window_is_bounded():
    w = Window(("obs", "pd"), keep=50)
    for seq in range(500):
        w.offer("obs", _payload("obs", seq, seq))
    assert w.metrics(policy_dt=None)["rows"] == 50


def test_latency_ends_can_be_named_when_the_last_node_has_no_seq():
    w = Window(("pour_node", "pd", "guard"), first="pour_node", last="pd")
    w.offer("pour_node", _payload("pour_node", 0, 10))
    w.offer("pd", _payload("pd", 0, 13))
    assert w.metrics(policy_dt=None)["latency_ms"]["max"] == 3.0
    with pytest.raises(ValueError):
        Window(("a", "b"), last="ghost")


def test_downsampling_keeps_the_spike():
    values = [1.0] * 1000
    values[617] = 40.0
    out = bucket_max(values, 90)
    assert len(out) == 90 and max(out) == 40.0


def test_bucket_max_edges():
    assert bucket_max([], 10) == [] and bucket_max([1, 2], 10) == [1, 2] and bucket_max([1, 2], 0) == []


@pytest.mark.golden
@pytest.mark.skipif(not RUNS, reason="fake run 기록이 없다")
@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.parent.name)
def test_percentiles_match_the_csv_tool_on_recorded_runs(run):
    """같은 기록 → 같은 p50/p95·seq missing. 규칙이 갈라지면 운영자는 화면과 CSV 중 무엇을 믿을지 모른다.

    한계: 기록된 run 은 전부 에피소드가 하나다. 에피소드가 여럿이면 CSV 는 seq 가 겹치는 행을 덮어쓰고 창은
    새 에피소드에서 비운다 — 그 경우의 숫자는 여기서 잠기지 않는다(`test_a_new_episode_on_the_first_node_clears_the_window` 가 창 쪽 동작만 고정한다).
    """
    lines = run.read_text(errors="replace").splitlines()
    rows = join_jsonl(lines)
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    assert lat, "RUNS 는 지연이 기록된 run 만 골랐다"
    text = summarize(rows, 0.02)
    assert f"p50 {p50(lat):.2f}" in text and f"p95 {p95(lat):.2f}" in text

    w = Window(NODES, keep=10**6)
    for topic, payload in iter_jsonl(lines):
        w.offer(topic, payload)
    got = w.metrics(policy_dt=0.02)
    # 갈라지는 곳은 하나이고 의도된 것이다: CSV 는 pd/fabric 이 에피소드 밖에서 내는 seq −1/−2 센티널도 행으로
    # 세고, 창은 버린다(틱이 아니다). 그 행에는 obs 가 없어 지연·결손 숫자에는 영향이 없다.
    assert got["rows"] == sum(1 for r in rows if r["seq"] >= 0)
    assert got["latency_ms"]["p50"] == pytest.approx(p50(lat)) and got["latency_ms"]["p95"] == pytest.approx(p95(lat))
    assert f"seq missing {got['seq_missing']}" in text


def test_the_golden_runs_are_not_vacuous():
    assert len(RUNS) >= 3, "지연이 기록된 fake run 이 사라졌다 — 위 대조 테스트가 아무것도 잠그지 않는다"


def test_latency_note_explains_a_pd_that_holds_internally():
    """fabric 을 끈 fake 실행: pd 는 seq −2 로만 돈다. 빈 지연 칸은 이유와 함께 나와야 한다."""
    w = Window(("obs", "pd"))
    for seq in range(5):
        w.offer("obs", {"seq": seq, "episode": 1, "t_pub_ns": seq * 1_000_000, "proc_ms": 1.0})
        w.offer("pd", {"seq": -2, "episode": 1, "t_pub_ns": seq * 1_000_000 + 5, "proc_ms": 0.1})
    m = w.metrics(policy_dt=0.02)
    assert m["latency_ms"]["p50"] is None
    assert "seq -2" in m["latency_note"] and "내부 유지" in m["latency_note"]


def test_latency_note_clears_once_pd_follows_the_chain():
    w = Window(("obs", "pd"))
    w.offer("pd", {"seq": -1, "episode": 1, "t_pub_ns": 1, "proc_ms": 0.1})
    assert w.metrics(policy_dt=0.02)["latency_note"] is None      # 아직 아무 seq 도 없다 — 할 말이 없다
    w.offer("obs", {"seq": 0, "episode": 1, "t_pub_ns": 1_000_000, "proc_ms": 1.0})
    assert "목표 없음" in w.metrics(policy_dt=0.02)["latency_note"]
    w.offer("pd", {"seq": 0, "episode": 1, "t_pub_ns": 3_000_000, "proc_ms": 0.1})
    m = w.metrics(policy_dt=0.02)
    assert m["latency_note"] is None and m["latency_ms"]["p50"] == 2.0


def test_latency_note_is_silent_for_a_single_node_window():
    w = Window(("obs",))
    w.offer("obs", {"seq": 0, "episode": 1, "t_pub_ns": 1, "proc_ms": 1.0})
    assert w.metrics(policy_dt=0.02)["latency_note"] is None

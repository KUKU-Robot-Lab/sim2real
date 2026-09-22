"""`/policy_control/status/*` 4 노드를 `seq` 로 join 하고 요약한다 — rclpy 무의존 순수 로직.

`tools/status_to_csv.py` 에 있던 것을 패키지로 올렸다. 이유는 소비자가 둘이 되었기 때문이다:
"N 초 구독 후 CSV 로 쓰고 끝나는" 도구와, 같은 숫자를 라이브로 보여주는 상태판.
**계산이 두 벌이 되면 둘이 다른 말을 하기 시작한다** — 그래서 한 곳에 둔다.

CSV 열 순서와 `summarize()` 문구는 바꾸지 않는다. `fake_plant_run.sh` 가 그 출력을 파싱한다.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable, Iterator, Mapping, Sequence

NODES = ("obs", "policy", "fabric", "pd")


class StatusJoiner:
    """노드 이름 → status dict 를 받아 `seq` 로 모은다. 도착 순서는 상관없다."""

    def __init__(self) -> None:
        self._by_seq: dict[int, dict[str, dict]] = defaultdict(dict)

    def offer(self, node: str, payload: Mapping) -> int | None:
        """한 건을 넣는다. `seq` 가 없는 메시지는 버리고 None 을 돌려준다."""
        seq = payload.get("seq")
        if seq is None:
            return None
        self._by_seq[int(seq)][node] = dict(payload)
        return int(seq)

    def __len__(self) -> int:
        return len(self._by_seq)

    def rows(self, nodes: Sequence[str] = NODES) -> list[dict]:
        """seq 오름차순 행. 열 구성은 예전 status_to_csv 와 글자 그대로 같다."""
        out = []
        for seq in sorted(self._by_seq):
            d = self._by_seq[seq]
            row: dict = {"seq": seq}
            for n in nodes:
                s = d.get(n, {})
                row[f"{n}_ok"] = s.get("ok")
                row[f"{n}_proc_ms"] = s.get("proc_ms")
                row[f"{n}_t_pub_ns"] = s.get("t_pub_ns")
                row[f"{n}_reasons"] = "|".join(s.get("reasons", []))
            t_obs = d.get("obs", {}).get("t_pub_ns")
            t_pd = d.get("pd", {}).get("t_pub_ns")
            row["latency_ms"] = None if (t_obs is None or t_pd is None) else (t_pd - t_obs) * 1e-6
            out.append(row)
        return out

    def latest(self) -> dict[str, dict]:
        """노드별 가장 큰 seq 의 status — 상태판이 '지금'을 그릴 때 쓴다."""
        best: dict[str, tuple[int, dict]] = {}
        for seq, d in self._by_seq.items():
            for n, s in d.items():
                if n not in best or seq > best[n][0]:
                    best[n] = (seq, s)
        return {n: s for n, (_, s) in best.items()}


def row_fields(rows: Sequence[Mapping]) -> list[str]:
    return list(rows[0].keys()) if rows else ["seq"]


def nodes_from_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """CSV 헤더 → 기록 당시의 `--nodes`. row_fields 의 역함수다.

    기록은 자기가 어떤 노드 구성으로 찍혔는지 헤더로만 말한다(pour 는 `pour_node` 하나, 4노드 체인은 NODES).
    되읽는 쪽이 기본값을 가정하면 다른 구성의 기록이 전부 "no rows" 로 읽힌다.
    """
    out = []
    for f in fields:
        if f.endswith("_t_pub_ns"):
            n = f[: -len("_t_pub_ns")]
            if n not in out:
                out.append(n)
    return tuple(out)


def summarize(rows: Sequence[Mapping], policy_dt: float | None) -> str:
    if not rows:
        return "no rows"
    lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    seqs = sorted(r["seq"] for r in rows)
    missing = (seqs[-1] - seqs[0] + 1 - len(seqs)) if seqs else 0
    p50 = lat[len(lat) // 2] if lat else float("nan")
    p95 = lat[int(0.95 * (len(lat) - 1))] if lat else float("nan")
    budget = "" if policy_dt is None else f" · budget 0.5·dt = {500 * policy_dt:.1f} ms"
    return f"rows {len(rows)} · latency p50 {p50:.2f} / p95 {p95:.2f} ms{budget} · seq missing {missing}"


def iter_jsonl(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """`--jsonl` 덤프를 (topic, payload) 로 되돌린다. 깨진 줄은 조용히 건너뛴다(기록은 디버그물이다)."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        topic = d.get("topic")
        if topic:
            yield str(topic), d


def join_jsonl(lines: Iterable[str], nodes: Sequence[str] = NODES) -> list[dict]:
    """기록된 status 덤프 → 행. 라이브 구독과 같은 경로를 태워 숫자가 갈라지지 않게 한다."""
    j = StatusJoiner()
    wanted = set(nodes)
    for topic, payload in iter_jsonl(lines):
        if topic in wanted:
            j.offer(topic, payload)
    return j.rows(nodes)

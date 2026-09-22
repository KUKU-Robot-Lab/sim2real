"""status 흐름을 화면용 숫자로 줄인다 — 순수, 유한 창.

`status_join.StatusJoiner` 는 끝없이 쌓는다(CSV 도구에는 그것이 맞다). 콘솔은 며칠씩 떠 있으므로
최근 `keep` 개 seq 만 든다. 지연의 정의는 그쪽과 같다: 같은 seq 의 (마지막 노드 t_pub − 첫 노드 t_pub).

★그래프용 다운샘플은 **버킷의 최댓값**이다 — 평균을 내면 운영자가 보러 온 바로 그 스파이크가 지워진다.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence


def p50(sorted_values: Sequence[float]) -> float | None:
    """status_join.summarize 와 글자 그대로 같은 규칙 — 콘솔과 CSV 도구의 숫자가 달라지면 안 된다."""
    return sorted_values[len(sorted_values) // 2] if sorted_values else None


def p95(sorted_values: Sequence[float]) -> float | None:
    return sorted_values[int(0.95 * (len(sorted_values) - 1))] if sorted_values else None


def bucket_max(values: Sequence[float], n: int) -> list[float]:
    if n <= 0 or not values:
        return []
    if len(values) <= n:
        return [round(v, 3) for v in values]
    out = []
    for i in range(n):
        lo, hi = i * len(values) // n, (i + 1) * len(values) // n
        out.append(round(max(values[lo:hi]), 3))
    return out


class Window:
    def __init__(self, nodes: Sequence[str], *, first: str | None = None, last: str | None = None,
                 keep: int = 600) -> None:
        if not nodes:
            raise ValueError("노드가 없다")
        self.nodes = tuple(nodes)
        #: 지연 = last.t_pub_ns − first.t_pub_ns. seq 를 내지 않는 노드(가드 등)가 끝에 있을 수 있어 따로 받는다.
        self.first, self.last = first or self.nodes[0], last or self.nodes[-1]
        for n in (self.first, self.last):
            if n not in self.nodes:
                raise ValueError(f"지연 기준 노드 {n!r} 가 노드 목록에 없다: {self.nodes}")
        self.keep = keep
        self._by_seq: OrderedDict[int, dict[str, tuple[int | None, float | None]]] = OrderedDict()
        self._episode: int | None = None
        #: 음수 seq 를 마지막으로 낸 노드 → 그 값. pd 의 −1(목표 없음)/−2(내부 유지)가 여기 남는다.
        self._off_chain: dict[str, int] = {}

    def offer(self, node: str, payload: Mapping) -> None:
        if node not in self.nodes:
            return
        seq = payload.get("seq")
        if not isinstance(seq, int):
            return
        if seq < 0:                               # pd 는 체인 목표를 안 받을 때 seq −1/−2 를 낸다
            self._off_chain[node] = seq
            return
        self._off_chain.pop(node, None)
        episode = payload.get("episode")
        if node == self.first and episode != self._episode:
            self._episode = episode              # 새 에피소드 — seq 가 0 부터 다시 시작한다
            self._by_seq.clear()
        row = self._by_seq.setdefault(seq, {})
        row[node] = (payload.get("t_pub_ns"), payload.get("proc_ms"))
        while len(self._by_seq) > self.keep:
            self._by_seq.popitem(last=False)

    def metrics(self, *, policy_dt: float | None, points: int = 90) -> dict:
        first, last = self.first, self.last
        seqs = sorted(self._by_seq)
        lat, proc = [], {n: [] for n in self.nodes}
        for s in seqs:
            row = self._by_seq[s]
            a, b = row.get(first, (None, None))[0], row.get(last, (None, None))[0]
            if a is not None and b is not None and first != last:
                lat.append((b - a) * 1e-6)
            for n in self.nodes:
                ms = row.get(n, (None, None))[1]
                if ms is not None:
                    proc[n].append(float(ms))
        ls = sorted(lat)
        return {
            "episode": self._episode,
            "rows": len(seqs),
            "seq_missing": (seqs[-1] - seqs[0] + 1 - len(seqs)) if seqs else 0,
            "budget_ms": None if policy_dt is None else round(500.0 * policy_dt, 3),
            "latency_ms": {"p50": p50(ls), "p95": p95(ls), "max": ls[-1] if ls else None},
            "latency_note": self._latency_note(bool(ls)),
            "proc_ms": {n: {"p50": p50(sorted(v)), "p95": p95(sorted(v))} for n, v in proc.items()},
            "series": {"latency_ms": bucket_max(lat, points), "proc_ms": {n: bucket_max(v, points) for n, v in proc.items()}},
        }

    def _latency_note(self, measured: bool) -> str | None:
        """지연 칸이 비었을 때 **왜** 비었는지. 빈 칸만 보이면 고장으로 읽힌다."""
        if measured or self.first == self.last or not self._by_seq:
            return None                       # 체인이 아직 seq 를 안 냈다 — 에피소드 전이라 할 말이 없다
        seq = self._off_chain.get(self.last)
        if seq is not None:
            what = {-2: "내부 유지 목표", -1: "목표 없음"}.get(seq, "체인 밖 목표")
            return (f"{self.last} 가 {what}(seq {seq})로 돌고 있다 — 체인 목표를 받지 않아 "
                    f"{self.first}→{self.last} 지연을 잴 수 없다 (fabric 을 끈 fake 실행이면 정상)")
        return f"{self.last} 의 status 가 {self.first} 와 같은 seq 로 아직 오지 않았다"

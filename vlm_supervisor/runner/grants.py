"""사람 승인 줄(vlm_grants.jsonl) 읽기 전용 — 이 꾸러미에는 그 파일을 고치는 코드를 두지 않는다(시험이 검사한다).

줄: {"operator", "episode_id", "scope": "episode"|"node", "node", "action", "since"}. since = 사람이 승인한 시각(wall).
읽기는 예외를 올리지 않는다 — 못 읽은 파일 · 줄은 "승인 없음" 이다. 종료(end)는 읽기에 기대지 않고 시각만 적는다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

SKEW_S = 5.0    # since 가 지금보다 이만큼 넘게 앞서면 거부한다(승인을 쓰는 쪽과의 시계 오차 허용)


def _no_constant(name: str):
    raise ValueError(name)      # NaN · Infinity · -Infinity


class Grants:
    def __init__(self, path, episode_id: str, wall: Callable[[], float]) -> None:
        self.path, self.episode_id, self.wall = Path(path), episode_id, wall
        self.start = wall()                 # 이 실행보다 앞선 줄은 지난 실행의 것이다
        self._revoked_at = -math.inf        # end() 시각 + SKEW_S — since 가 이 이하인 줄은 전부 죽었다
        self._used: set[str] = set()        # 한 번 쓴 node 줄

    def _lines(self) -> list[tuple[str, dict]]:
        try:
            raws = self.path.read_bytes().decode("utf-8", errors="replace").splitlines()
            now = self.wall()
        except Exception:    # noqa: BLE001 — 읽기 실패 = 승인 없음
            return []
        out = []
        for raw in raws:
            try:
                g = json.loads(raw, parse_constant=_no_constant)
                op, since = g.get("operator"), g.get("since")
                if (raw in self._used or not isinstance(op, str) or not op.strip() or op.strip().lower().startswith("vlm")
                        or g.get("episode_id") != self.episode_id or isinstance(since, bool)
                        or not isinstance(since, (int, float)) or not math.isfinite(since)
                        or since < self.start or since <= self._revoked_at or since > now + SKEW_S):
                    continue
            except Exception:    # noqa: BLE001 — 깨진 줄(RecursionError · OverflowError 포함)은 건너뛴다
                continue
            out.append((raw, g))
        return out

    def episode(self) -> dict | None:
        """살아 있는 에피소드 1회 승인(없으면 None)."""
        return next((g for _, g in self._lines() if g.get("scope") == "episode"), None)

    def node(self, node_id: str, action: str, since: float) -> dict | None:
        """그 결정(노드 id · 동작) 자체를 since 뒤에 승인한 줄 — 한 줄은 한 번만 쓴다."""
        for raw, g in self._lines():
            if g.get("scope") == "node" and g.get("node") == node_id and g.get("action") == action and g["since"] >= since:
                self._used.add(raw)
                return g
        return None

    def end(self) -> None:
        """handoff · stop · FAILURE · 러너 오류 — 지금까지의 줄은 다시 못 쓴다(파일을 읽지 않는다). 다시 가려면 사람이 새 줄을 더한다."""
        # + SKEW_S: 쓰는 쪽 시계가 앞서 있어 종료 시각보다 조금 뒤로 찍힌 옛 줄까지 죽인다(새 줄은 종료 SKEW_S 뒤부터 통한다)
        self._revoked_at = max(self._revoked_at, self.wall() + SKEW_S)

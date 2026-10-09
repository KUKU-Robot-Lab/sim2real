"""trigger — 정책 노드가 막 떴을 때 reset 이 '측정 없음'으로 거부되면 잠깐 다시 부른다(10.09 실기)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools" / "trigger.py"
_spec = importlib.util.spec_from_file_location("trigger_tool", _PATH)
T = importlib.util.module_from_spec(_spec)
sys.modules["trigger_tool"] = T
_spec.loader.exec_module(T)


def test_only_a_reset_refused_for_a_not_yet_received_source_is_retried():
    """10.09: 왼팔 정책 노드가 뜬 지 0.25 s 만에 reset 이 불려 "source 'arm' is missing" 으로 거부됐다(오른팔은 1.4 s 뒤라 됐다).
    측정 · 컵이 아직 안 왔거나 늦은 것만 다시 — 진짜 거부(자세 · 기울기 · 박스 밖)는 그대로 실패."""
    assert T.retry_reason("episode/reset", ["reset: left: source 'arm' is missing"])
    assert T.retry_reason("episode/reset", ["reset: right: source 'cup' is stale (0.8 s)"])
    assert not T.retry_reason("episode/reset", ["start: arm 0.31 rad from start pose > 0.15"])
    assert not T.retry_reason("episode/start", ["start: left: source 'arm' is missing"])   # start 는 사람이 보고 다시
    assert not T.retry_reason("episode/reset", [])

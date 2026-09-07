"""pd_selftest 의 구간 길이는 벽시계로 지켜져야 한다 (ROS 없음).

2026-09-07 실기: `rclpy.spin_once(node, timeout_sec=dt)` 를 sleep 대신 써서 dwell 이
전혀 지켜지지 않았다. spin_once 는 처리할 메시지가 있으면 **즉시 반환**하고, 실기
`/joint_states` 는 745 Hz 로 들어온다 — 루프가 946 Hz 로 돌아 "2.0 s dwell" 이 0.106 s 가
됐다(19배). 팔이 움직일 시간이 없어 추종률 15~44 % 로 측정됐고 셀프테스트가 오판했다.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = SIM2REAL / f"policy_control/tools/{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _BusyNode:
    """항상 처리할 메시지가 있는 노드 — spin_once 가 절대 기다리지 않는다(실기와 같다)."""

    def __init__(self):
        self.spins = 0

    def spin_once(self, timeout_sec=None):
        self.spins += 1


def test_spin_until_honours_wall_clock_even_when_spin_never_blocks():
    mod = _load("pd_selftest")
    node = _BusyNode()

    dur = 0.20
    t0 = time.monotonic()
    mod.spin_until(node, t0 + dur, spin=lambda n, timeout_sec=None: n.spin_once(timeout_sec))
    elapsed = time.monotonic() - t0

    assert elapsed >= dur * 0.95, (
        f"구간이 {elapsed:.3f} s 로 끝났다 — {dur} s 를 안 지킨다. "
        "spin_once 를 sleep 처럼 쓰면 메시지가 많을 때 즉시 반환한다")
    assert elapsed < dur * 3, f"{elapsed:.3f} s 는 너무 오래 걸렸다"
    assert node.spins > 0, "기다리는 동안에도 콜백은 돌아야 한다"

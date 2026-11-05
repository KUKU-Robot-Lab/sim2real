"""rclpy 는 로그 호출 위치마다 심각도를 고정한다. 한 줄에서 info/warning 을 번갈아 부르면 ValueError 가 나고
서비스 콜백 안이면 노드가 죽는다(11.05 왼손 rh56f1_state_node: tip_zero 거부 → 손 측정 끊김 → pd 오차 NaN)."""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEVELS = {"debug", "info", "warn", "warning", "error", "fatal"}


def _is_level_attr(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr in LEVELS


def _severity_switches(tree: ast.AST):
    for node in ast.walk(tree):
        # (log.info if ok else log.warning)(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.IfExp):
            if _is_level_attr(node.func.body) and _is_level_attr(node.func.orelse):
                yield node.lineno


def test_no_ros_node_switches_log_severity_on_one_line():
    bad = []
    for root in ("deploy", "scripts"):
        for path in (REPO / root).rglob("*.py"):
            if any(p in path.parts for p in ("build", "install", ".venv", "__pycache__")):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            bad += [f"{path.relative_to(REPO)}:{n}" for n in _severity_switches(tree)]
    assert not bad, f"한 줄에서 로그 심각도를 바꾸면 rclpy 가 ValueError 를 낸다: {bad}"


def test_the_scan_catches_the_11_05_pattern():
    src = "(node.get_logger().info if ok else node.get_logger().warning)('x')\n"
    assert list(_severity_switches(ast.parse(src))) == [1]


def test_rclpy_really_refuses_a_severity_change_at_one_call_site():
    import pytest
    rclpy_logging = pytest.importorskip("rclpy.logging")
    log = rclpy_logging.get_logger("severity_probe")

    def say(ok: bool) -> None:
        (log.info if ok else log.warning)("probe")

    say(True)
    with pytest.raises(ValueError, match="severity"):
        say(False)

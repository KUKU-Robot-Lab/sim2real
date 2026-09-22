"""스냅샷 배선 검사 — 프로파일이 선언한 것이 화면 payload 까지 실제로 닿는가.

links.chain() 자체는 test_console_links / test_console_stack 이 본다. 여기서는 Console 이 그것을
**빠짐없이 넘기는지** 만 본다 (09.21: stack= 을 안 넘겨 로봇 스택 블록이 화면에서 통째로 빠졌다).
"""
from __future__ import annotations

import textwrap

STACK = """
stack:
  title: 테스트 스택
  nodes: [/fake_arm_bridge]
  topics:
    - {name: /joint_states, stale_ms: 500}
  managers:
    - {name: /controller_manager, active: [joint_state_broadcaster]}
"""


def _block_ids(console) -> list[str]:
    return [b["id"] for b in console.snapshot()["session"]["links"]]


def test_a_profile_without_a_stack_shows_no_stack_block(console):
    # Arrange / Act
    console.open("t_fake", operator="pytest")

    # Assert
    assert "stack" not in _block_ids(console)


def test_a_declared_stack_reaches_the_snapshot(console, tiny_repo):
    # Arrange
    path = tiny_repo / "profiles" / "t_fake.yaml"
    path.write_text(path.read_text() + textwrap.dedent(STACK))

    # Act
    console.open("t_fake", operator="pytest")
    links = console.snapshot()["session"]["links"]

    # Assert
    stack = next((b for b in links if b["id"] == "stack"), None)
    assert stack is not None, [b["id"] for b in links]
    assert stack["title"] == "테스트 스택"
    # 이 fixture 는 브리지 없이 돈다 — 행은 비고 "알 수 없다" 로 나온다. 행 내용은 test_console_stack 이 본다.
    assert stack["rows"] == []
    assert [b["id"] for b in links][:2] == ["bridge", "stack"]

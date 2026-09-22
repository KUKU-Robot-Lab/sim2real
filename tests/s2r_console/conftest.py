"""s2r_console 테스트 — 패키지 루트를 sys.path 에 넣는다 (colcon 설치 없이 돈다).

★여기 테스트는 **실기 도메인(126)에 아무것도 띄우지 않는다**. 흐름 테스트의 프로파일은 전부 fake/도메인 98 이고,
정지 동작(trigger.py)은 아무 일도 하지 않는 스크립트로 바꿔 끼운다.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

SIM2REAL = Path(__file__).resolve().parents[2]
PKG_ROOT = SIM2REAL / "s2r_console"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import s2r_console._paths  # noqa: E402, F401 — mission_core · mission_run · policy_control 을 올린다

MISSION = """
name: 테스트 미션
loop_to: move
artifacts:
  contract: data/contract.json
stages:
  - id: check
    title: 전경 명령 하나
  - id: up
    title: 배경 프로세스
    needs: [check]
  - id: move
    title: 실기를 움직이는 단계
    needs: [up]
    artifacts: [contract]
    touches_real: true
run:
  check:
    - note: 통과
      argv: ["true"]
  up:
    - note: 오래 사는 것
      argv: [sleep, "30"]
      background: true
  move:
    - note: 움직임
      argv: ["true"]
"""

PROFILE = """
schema: s2r_console/profile/v1
id: t_fake
title: 테스트
mission: mission.yaml
domain: {id: 98, class: fake}
policy_dt: 0.02
status_nodes: [obs, pd]
"""


@pytest.fixture()
def tiny_repo(tmp_path, monkeypatch):
    """미션 · 프로파일 · 계약 파일 하나가 든 가짜 repo. run 기록도 tmp 로 간다."""
    import mission_run  # noqa: PLC0415 — _paths 가 올린 뒤라야 보인다

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "data" / "contract.json").write_text('{"v": 1}')
    (repo / "mission.yaml").write_text(textwrap.dedent(MISSION))
    (repo / "profiles").mkdir()
    (repo / "profiles" / "t_fake.yaml").write_text(textwrap.dedent(PROFILE))
    monkeypatch.setattr(mission_run, "MISSION_LOG_DIR", tmp_path / "logs")
    return repo


@pytest.fixture()
def console(tiny_repo, monkeypatch, tmp_path):
    from s2r_console import console as mod  # noqa: PLC0415

    noop = tmp_path / "noop_trigger.py"
    noop.write_text("import sys; sys.exit(0)\n")
    monkeypatch.setattr(mod, "_TRIGGER", noop)
    c = mod.Console(repo=tiny_repo, profiles_dir=tiny_repo / "profiles", bridge=False)
    yield c
    c.shutdown()

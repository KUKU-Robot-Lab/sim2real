"""묶음 · 건너뛰기 · 정지 명령 — 09.22 실기 재구성의 흐름을 fake 도메인에서 잠근다.

실기에서 드러난 것: selftest 가 engage 앞에 있었고, 발행 pd 를 띄우는 단계가 없었고, 끄는 길이 없어 kill 로 내렸다.
여기서는 미션 하나로 ① 묶음이 화면까지 가는지 ② 정지 명령이 앞 단계의 프로세스를 적힌 순서로 내리는지
③ 건너뛰기가 선언된 단계에서만, pd 가 팔을 잡지 않을 때만 되는지를 본다.
"""
from __future__ import annotations

import textwrap

import pytest

MISSION = """
name: 흐름 테스트
groups:
  - {id: connect, title: 연결}
  - {id: prepare, title: 준비}
  - {id: finish, title: 정리}
stages:
  - id: up
    group: connect
    title: 배경 둘
  - id: swap
    group: prepare
    title: 하나를 내리고 다른 것을 띄운다
    needs: [up]
  - id: extra
    group: prepare
    title: 건너뛸 수 있는 단계
    skippable: true
  - id: down
    group: finish
    title: 전부 내린다
run:
  up:
    - {note: 첫째, argv: [sleep, "30"], background: true}
    - {note: 둘째, argv: [sleep, "31"], background: true}
  swap:
    - {note: 첫째를 내린다, stop: ["up#0"]}
    - {note: 셋째, argv: [sleep, "32"], background: true}
  extra:
    - {note: 돌면 안 된다, argv: ["false"]}
  down:
    - {note: 남은 것 전부, stop: ["swap#1", "up#1", "up#0"]}
"""


@pytest.fixture()
def flow(console, tiny_repo):
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(MISSION))
    console.open("t_fake", operator="pytest")
    return console


def _run(console, stage, outcome="DONE"):
    console.run_stage(stage, operator="pytest")
    console.session.runner.join(15)
    assert console.session.runner.outcome == outcome, console.session.runner.view()


def _alive(console, key):
    return console.session.supervisor.is_alive(key)


def test_groups_and_skip_flags_reach_the_mission_view(flow):
    m = flow.snapshot()["session"]["mission"]
    assert [g["id"] for g in m["groups"]] == ["connect", "prepare", "finish"]
    rows = {r["id"]: r for r in m["rows"]}
    assert rows["up"]["group"] == "connect" and rows["extra"]["skippable"] is True and rows["up"]["skippable"] is False
    assert [c["kind"] for c in rows["swap"]["commands"]] == ["stop", "background"]
    assert rows["swap"]["commands"][0]["stop"] == ["up#0"]


def test_a_stop_step_brings_down_what_an_earlier_stage_started(flow):
    _run(flow, "up")
    assert _alive(flow, "up#0") and _alive(flow, "up#1")
    _run(flow, "swap")
    assert not _alive(flow, "up#0") and _alive(flow, "up#1") and _alive(flow, "swap#1")
    step = flow.session.runner.view()["steps"][0]
    assert step["kind"] == "stop" and "up#0" in step["detail"]


def test_skip_only_where_declared_and_the_skipped_stage_never_runs(flow):
    from s2r_console.console import ConsoleError
    _run(flow, "up")
    with pytest.raises(ConsoleError):
        flow.skip_stage("swap", operator="pytest")              # skippable 이 아니다
    _run(flow, "swap")
    assert flow.snapshot()["session"]["mission"]["rows"][2]["can_skip"] is True
    flow.skip_stage("extra", operator="pytest")                 # 명령이 `false` — 돌았다면 FAILED 였다
    m = flow.snapshot()["session"]["mission"]
    assert m["stage"] == "down" and "extra" in [r["id"] for r in m["rows"] if r["done"]]
    assert "extra#0" not in {p["key"] for p in flow.session.supervisor.table()}


def test_skip_is_refused_while_pd_holds_the_arm(flow, monkeypatch):
    from s2r_console.console import ConsoleError
    _run(flow, "up")
    _run(flow, "swap")
    monkeypatch.setattr(type(flow), "_pd_phase", lambda self, s, obs, procs: "TRACKING")
    with pytest.raises(ConsoleError) as exc:
        flow.skip_stage("extra", operator="pytest")
    assert any("TRACKING" in r for r in exc.value.reasons)


def test_the_last_stage_can_bring_everything_down_in_order(flow):
    _run(flow, "up")
    _run(flow, "swap")
    flow.skip_stage("extra", operator="pytest")
    _run(flow, "down")
    assert not any(_alive(flow, k) for k in ("up#0", "up#1", "swap#1"))
    assert "swap#1, up#1" in flow.session.runner.view()["steps"][0]["detail"]   # up#0 은 이미 없었다


def test_rewind_goes_back_to_a_finished_stage_and_keeps_its_processes(flow):
    # 09.22 사용자: "gui 창에서 잘못되면 되돌아가서 진행할 수가 없네"
    from s2r_console.console import ConsoleError
    _run(flow, "up")
    _run(flow, "swap")
    flow.skip_stage("extra", operator="pytest")
    with pytest.raises(ConsoleError):
        flow.rewind("down", operator="pytest")                     # 아직 끝내지 않은 단계로는 못 간다
    flow.rewind("swap", operator="pytest")
    m = flow.snapshot()["session"]["mission"]
    assert m["stage"] == "swap" and m["status"] == "PENDING"
    rows = {r["id"]: r for r in m["rows"]}
    assert rows["up"]["done"] and not rows["swap"]["done"] and not rows["extra"]["done"] and not rows["extra"]["skipped"]
    assert _alive(flow, "swap#1") and _alive(flow, "up#1")          # 떠 있는 것은 그대로
    _run(flow, "swap")                                              # 다시 실행 — 이미 떠 있는 것은 넘긴다
    assert flow.session.runner.view()["steps"][1]["status"] == "kept"


def test_rewind_is_refused_while_a_stage_runs(flow, tiny_repo):
    from s2r_console.console import ConsoleError
    _run(flow, "up")
    flow.session.runner = type("R", (), {"active": True, "stage_id": "swap"})()
    with pytest.raises(ConsoleError, match="실행 중"):
        flow.rewind("up", operator="pytest")
    flow.session.runner = None


DYING = """
name: 자식이 죽는 launch
stages:
  - id: up
    title: launch 흉내 — 자식이 죽었다고 찍고 자기는 산다
run:
  up:
    - note: 반쯤 죽은 launch
      argv: [bash, -c, "echo '[ERROR] [fake_arm_bridge-1]: process has died [pid 1, exit code 1]'; sleep 30"]
      background: true
"""


def test_a_launch_whose_child_died_fails_the_stage(console, tiny_repo):
    # 09.22 fake: 팔 브리지가 뜨자마자 죽었는데 ros2 launch 는 살아 있어 drivers 가 완료로 넘어갔다.
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(DYING))
    console.open("t_fake", operator="pytest")
    _run(console, "up", outcome="FAILED")
    assert "process has died" in console.session.state.note

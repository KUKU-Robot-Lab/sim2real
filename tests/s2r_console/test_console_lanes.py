"""창(lane) — 따로 도는 장치를 동시에 다룬다. 09.23 사용자 결정.

"차라리 robot stat 창 아래에 / 드라이버 창, vision 창 / 오른팔 제어 창, 왼팔 제어창 / … 이렇게 분리되어야
한번에 진행하면서, 되지? 따로 분리되어 있는 시스템이 순서대로 하는게 이상함"

여기서 잠그는 것:
  ① 창이 다르면 **차례를 기다리지 않는다** — 커서가 사라지고 `needs` 만 남는다
  ② 같은 창 안에서는 여전히 한 번에 하나다
  ③ 한 창이 실패해도 다른 창은 제 단계를 계속한다
  ④ 되돌리기는 **그 단계에 기대는 것**만 지운다 — 다른 창의 끝낸 단계를 잊지 않는다
"""
from __future__ import annotations

import textwrap
import time

import pytest

MISSION = """
name: 창 테스트
lanes:
  - {id: rig, title: 드라이버}
  - {id: arm_right, title: 오른팔, side: right}
  - {id: arm_left, title: 왼팔, side: left}
groups:
  - {id: connect, title: 연결}
  - {id: motion, title: 자세 이동, motion: true}
stages:
  - id: drivers
    lane: rig
    group: connect
    title: 드라이버
  - id: right_load
    lane: arm_right
    group: connect
    title: 오른팔 pd
    needs: [drivers]
  - id: right_home
    lane: arm_right
    group: motion
    title: 오른팔 홈
    needs: [right_load]
  - id: left_load
    lane: arm_left
    group: connect
    title: 왼팔 pd
    needs: [drivers]
  - id: left_home
    lane: arm_left
    group: motion
    title: 왼팔 홈
    needs: [left_load]
run:
  drivers:
    - {note: 드라이버, argv: [sleep, "30"], background: true}
  right_load:
    - {note: 오른팔, argv: [sleep, "31"], background: true}
  right_home:
    - {note: 느린 것, argv: [sleep, "3"]}
  left_load:
    - {note: 왼팔, argv: [sleep, "32"], background: true}
  left_home:
    - {note: 실패한다, argv: ["false"]}
"""


@pytest.fixture()
def lanes(console, tiny_repo):
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(MISSION))
    console.open("t_fake", operator="pytest")
    return console


def _run(console, stage, outcome="DONE"):
    console.run_stage(stage, operator="pytest")
    runner = console._runner_of(console.session, stage)
    assert runner is not None
    runner.join(15)
    assert runner.outcome == outcome, runner.view()


def _mission(console):
    return console.snapshot()["session"]["mission"]


def _rows(console):
    return {r["id"]: r for r in _mission(console)["rows"]}


def test_a_stage_in_another_lane_does_not_wait_its_turn(lanes):
    _run(lanes, "drivers")
    # 왼팔을 먼저 해도 된다 — yaml 에서는 오른팔이 앞이지만 창이 다르면 차례가 없다.
    _run(lanes, "left_load")
    _run(lanes, "right_load")
    assert set(lanes.session.state.completed) == {"drivers", "left_load", "right_load"}


def test_two_lanes_run_at_the_same_time(lanes):
    _run(lanes, "drivers")
    _run(lanes, "right_load")
    _run(lanes, "left_load")
    lanes.run_stage("right_home", operator="pytest")          # 3 초짜리
    lanes.run_stage("left_home", operator="pytest")           # 돌고 있는 중에 바로 뜬다
    busy = lanes._busy(lanes.session)
    assert busy.get("arm_right") == "right_home"
    assert "arm_left" in busy or "left_home" in lanes.session.state.completed
    for r in list(lanes.session.runners.values()):
        r.join(20)


def test_the_same_lane_still_runs_one_at_a_time(lanes):
    from s2r_console.console import ConsoleError

    _run(lanes, "drivers")
    _run(lanes, "right_load")
    lanes.run_stage("right_home", operator="pytest")
    with pytest.raises(ConsoleError, match="오른팔 창에서 right_home 가 실행 중"):
        lanes.run_stage("right_home", operator="pytest")
    lanes.session.runners["arm_right"].join(20)


def test_one_lane_failing_does_not_stop_the_other(lanes):
    _run(lanes, "drivers")
    _run(lanes, "right_load")
    _run(lanes, "left_load")
    _run(lanes, "left_home", outcome="FAILED")
    rows = _rows(lanes)
    assert rows["left_home"]["can_run"]                        # 다시 실행할 수 있다
    assert rows["right_home"]["can_run"]                       # 오른팔은 실패를 모른다
    lane = {x["id"]: x for x in _mission(lanes)["lanes"]}
    assert lane["arm_left"]["last"]["outcome"] == "FAILED"
    assert lane["arm_right"]["last"]["stage"] == "right_load"


def test_rewind_only_forgets_what_depends_on_the_stage(lanes):
    _run(lanes, "drivers")
    _run(lanes, "right_load")
    _run(lanes, "left_load")
    lanes.rewind("right_load", operator="pytest")
    done = set(lanes.session.state.completed)
    assert "left_load" in done and "drivers" in done          # 다른 창은 그대로
    assert "right_load" not in done


def test_rewinding_the_shared_stage_forgets_both_arms(lanes):
    _run(lanes, "drivers")
    _run(lanes, "right_load")
    _run(lanes, "left_load")
    lanes.rewind("drivers", operator="pytest")
    assert lanes.session.state.completed == ()               # 둘 다 drivers 에 기댄다


def test_each_lane_reports_its_own_next_stage(lanes):
    _run(lanes, "drivers")
    _run(lanes, "right_load")
    view = {x["id"]: x for x in _mission(lanes)["lanes"]}
    assert view["arm_right"]["next"] == "right_home"
    assert view["arm_left"]["next"] == "left_load"
    assert view["rig"]["next"] is None                        # 이 창은 끝났다


def test_a_stage_that_stops_another_lane_is_marked(console, tiny_repo):
    mission = MISSION.replace('''  left_load:
    - {note: 왼팔, argv: [sleep, "32"], background: true}''',
                              '''  left_load:
    - {note: 오른팔 pd 를 내린다, stop: ["right_load#0"]}
    - {note: 왼팔, argv: [sleep, "32"], background: true}''')
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(mission))
    console.open("t_fake", operator="pytest")
    rows = _rows(console)
    assert rows["left_load"]["stops_lanes"] == ["arm_right"]
    assert rows["right_load"]["stops_lanes"] == []


def test_ack_and_abort_name_the_stage_when_two_lanes_run(lanes):
    from s2r_console.console import ConsoleError

    _run(lanes, "drivers")
    _run(lanes, "right_load")
    _run(lanes, "left_load")
    lanes.run_stage("right_home", operator="pytest")
    lanes.run_stage("left_home", operator="pytest")
    time.sleep(0.2)
    live = [r.stage_id for r in lanes.session.runners.values() if r.active]
    if len(live) > 1:
        with pytest.raises(ConsoleError, match="동시에 돈다"):
            lanes.abort_stage()
    lanes.abort_stage("right_home")
    for r in list(lanes.session.runners.values()):
        r.join(20)

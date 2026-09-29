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


OPTIONAL = MISSION.replace(
    "  - {id: motion, title: 자세 이동, motion: true}\n",
    "  - {id: motion, title: 자세 이동, motion: true}\n  - {id: diagnose, title: 진단 (선택), motion: true, optional: true}\n",
).replace(
    "  - id: left_home\n",
    "  - id: right_selftest\n    lane: arm_right\n    group: diagnose\n    title: 셀프테스트\n    needs: [right_load]\n"
    "  - id: left_home\n",
).replace(
    "run:\n",
    "run:\n  right_selftest:\n    - {note: 셀프테스트, argv: [\"true\"]}\n",
)


def test_an_optional_group_is_never_the_next_stage_but_can_still_be_run(console, tiny_repo):
    """09.28 사용자: "selftest_left 이거 맨날 실패하는 것 같은데 따로 빼두던가" — 차례에 끼지 않고, 고르면 돈다."""
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(OPTIONAL))
    console.open("t_fake", operator="pytest")
    _run(console, "drivers")
    _run(console, "right_load")
    _run(console, "right_home")
    view = {x["id"]: x for x in _mission(console)["lanes"]}
    assert view["arm_right"]["next"] is None                  # 진단은 차례가 아니다
    assert _rows(console)["right_selftest"]["can_run"]
    _run(console, "right_selftest")


TOGGLE = """
name: 켜기 끄기 · focus
lanes:
  - {id: head, title: 비전}
  - {id: arm_right, title: 오른팔, side: right, focus: right_policy}
groups:
  - {id: connect, title: 연결}
  - {id: policy, title: 정책}
  - {id: finish, title: 정리}
stages:
  - id: sensors
    lane: head
    group: connect
    title: 인지 켜기
    undoes: [sensors_off]
  - id: right_home
    lane: arm_right
    group: connect
    title: 홈
  - id: right_policy
    lane: arm_right
    group: policy
    title: 정책
    needs: [right_home, sensors]
  - id: right_rehome
    lane: arm_right
    group: finish
    title: 다시 홈
    needs: [right_home]
  - id: right_return
    lane: arm_right
    group: finish
    title: 차렷
    needs: [right_home]
    undoes: [right_home]
  - id: sensors_off
    lane: head
    group: finish
    title: 인지 끄기
    skippable: true
    undoes: [sensors]
run:
  sensors:
    - {note: 켜기, argv: ["true"]}
  right_home:
    - {note: 홈, argv: ["true"]}
  right_policy:
    - {note: 정책, argv: ["true"]}
  right_rehome:
    - {note: 다시 홈, argv: ["true"]}
  right_return:
    - {note: 차렷, argv: ["true"]}
  sensors_off:
    - {note: 끄기, argv: ["true"]}
"""


def _lane(console, lane_id):
    return {x["id"]: x for x in _mission(console)["lanes"]}[lane_id]


def test_sensors_can_be_turned_on_again_after_sensors_off(console, tiny_repo):
    """09.29 사용자: "sensor-off 를 하면 두 번 다시 sensor 를 킬 수 없음" — 끄면 켬 완료가 지워져 다시 차례가 된다."""
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(TOGGLE))
    console.open("t_fake", operator="pytest")
    _run(console, "sensors")
    _run(console, "sensors_off")
    assert "sensors" not in console.session.state.completed
    assert _lane(console, "head")["next"] == "sensors"
    assert _rows(console)["sensors"]["can_run"]
    _run(console, "sensors")
    assert "sensors_off" not in console.session.state.completed and "sensors" in console.session.state.completed


def test_skipping_sensors_off_undoes_nothing(console, tiny_repo):
    """건너뜀은 실행하지 않았다 — 인지는 켜진 채이므로 켬 완료를 지우지 않는다."""
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(TOGGLE))
    console.open("t_fake", operator="pytest")
    _run(console, "sensors")
    t_end = time.time() + 5.0
    while console._busy(console.session) and time.time() < t_end:           # 러너가 끝난 뒤 창이 비기까지(경합)
        time.sleep(0.02)
    console.skip_stage("sensors_off", operator="pytest")
    assert {"sensors", "sensors_off"} <= set(console.session.state.completed)


def test_the_arm_card_returns_to_the_policy_after_home(console, tiny_repo):
    """09.29 사용자: "home 이후에는 정책 창이 디폴트" — 정책을 돌린 뒤 · rehome 뒤에도 카드는 정책이다."""
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(TOGGLE))
    console.open("t_fake", operator="pytest")
    _run(console, "right_home")
    assert _lane(console, "arm_right")["next"] == "right_policy"      # 인지 전에도 차례는 정책(막힘 사유가 보인다)
    _run(console, "sensors")
    _run(console, "right_policy")
    assert _lane(console, "arm_right")["next"] == "right_policy"
    _run(console, "right_rehome")
    assert _lane(console, "arm_right")["next"] == "right_policy"
    _run(console, "right_return")                                      # 차렷으로 — 정책 시작 자세가 아니다
    assert "right_home" not in console.session.state.completed
    assert _lane(console, "arm_right")["next"] == "right_home"

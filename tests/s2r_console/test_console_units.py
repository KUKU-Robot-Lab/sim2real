"""단위 스위치 — 그림의 상자에서 미션의 배경 명령 하나를 켜고 끈다.

argv 는 여전히 미션 yaml 에서만 온다(HTTP 로는 키 `stage#n` 만 온다). 켜고 끄는 규칙은 순수 함수다.
"""
from __future__ import annotations

import json
import time

import pytest

import s2r_console._paths  # noqa: F401
from s2r_console import api
from s2r_console.units import PD_UNKNOWN, UnitCmd, off_reasons, on_reasons, pd_phase_of

BG = UnitCmd(key="up#0", stage="up", index=0, note="오래 사는 것", argv=("sleep", "30"), kind="background",
             touches_real=False, needs=("check",))


# ── 켜기 규칙 ───────────────────────────────────────────────────────────
def test_a_background_unit_whose_needs_are_done_may_be_switched_on():
    assert on_reasons(BG, alive=False, busy_stage=None, completed={"check"}) == []


@pytest.mark.parametrize(("unit", "kw", "needle"), [
    (BG, dict(alive=True, busy_stage=None, completed={"check"}), "이미"),
    (BG, dict(alive=False, busy_stage="up", completed={"check"}), "실행 중"),          # 제 단계의 러너가 같은 키를 띄우는 중일 수 있다
    (BG, dict(alive=False, busy_stage=None, completed=set()), "check"),
    (UnitCmd(**{**BG.__dict__, "kind": "manual"}), dict(alive=False, busy_stage=None, completed={"check"}), "수동"),
    (UnitCmd(**{**BG.__dict__, "kind": "foreground"}), dict(alive=False, busy_stage=None, completed={"check"}), "배경"),
    (UnitCmd(**{**BG.__dict__, "touches_real": True}), dict(alive=False, busy_stage=None, completed={"check"}), "승인"),
])
def test_switching_on_is_refused_with_a_reason(unit, kw, needle):
    reasons = on_reasons(unit, **kw)
    assert reasons and any(needle in r for r in reasons)


# ── 끄기 규칙 ───────────────────────────────────────────────────────────
def test_a_live_unit_may_be_switched_off_while_pd_is_idle():
    assert off_reasons(BG, alive=True, busy_stage=None, pd_phase="IDLE", robot_unit=True, real=True) == []
    assert off_reasons(BG, alive=True, busy_stage=None, pd_phase=None, robot_unit=True, real=True) == []


@pytest.mark.parametrize(("kw", "needle"), [
    (dict(alive=False, busy_stage=None, pd_phase="IDLE", robot_unit=False, real=False), "떠 있지"),
    (dict(alive=True, busy_stage="up", pd_phase="IDLE", robot_unit=False, real=False), "실행 중"),
    (dict(alive=True, busy_stage=None, pd_phase="TRACKING", robot_unit=True, real=False), "PD 해제"),   # pd 를 죽이면 팔이 떨어진다
    (dict(alive=True, busy_stage=None, pd_phase="TRACKING", robot_unit=False, real=True), "PD 해제"),   # 실기는 전부 막는다
])
def test_switching_off_is_refused_with_a_reason(kw, needle):
    reasons = off_reasons(BG, **kw)
    assert reasons and any(needle in r for r in reasons)


# ── pd phase 읽기: 조용한 pd 는 "자유" 가 아니다 ────────────────────────────
PHASE_KW = dict(stale_s=2.0, pd_alive=False, real=False, pd_in_graph=False)


def test_a_fresh_status_is_the_phase():
    assert pd_phase_of({"phase": "TRACKING"}, 0.4, **PHASE_KW) == "TRACKING"
    assert pd_phase_of({"phase": "IDLE"}, 0.4, **{**PHASE_KW, "pd_alive": True}) == "IDLE"


@pytest.mark.parametrize("kw", [
    dict(pd_alive=True),                       # 콘솔이 띄운 pd 가 살아 있는데 상태가 안 온다
    dict(pd_in_graph=True),                    # 콘솔 밖에서 띄운 pd 가 그래프에 보인다
    dict(real=True, pd_in_graph=None),         # 실기인데 그래프를 못 봤다
    dict(real=True, pd_alive=None),            # 실기인데 어느 단위가 pd 인지 모른다
])
def test_a_silent_pd_is_unknown_not_free(kw):
    for status, age in ((None, None), ({"phase": "IDLE"}, 9.0)):      # 없거나, 묵은 IDLE
        assert pd_phase_of(status, age, **{**PHASE_KW, **kw}) == PD_UNKNOWN


def test_pd_is_absent_only_when_that_is_proven():
    assert pd_phase_of(None, None, **PHASE_KW) is None                                  # 안 띄웠고 그래프에도 없다
    assert pd_phase_of(None, None, **{**PHASE_KW, "pd_alive": None, "pd_in_graph": None}) is None   # fake 는 잃을 팔이 없다


def test_a_pd_that_spoke_and_fell_silent_is_unknown_until_it_is_proven_gone():
    blind = {**PHASE_KW, "pd_alive": None, "pd_in_graph": None}                   # fake · 어느 단위가 pd 인지 모른다 · 그래프도 못 봤다
    assert pd_phase_of({"phase": "TRACKING"}, 30.0, **blind) == PD_UNKNOWN
    assert pd_phase_of({"phase": "TRACKING"}, 30.0, **{**blind, "pd_in_graph": False}) is None   # 그래프에 없다 — 사라졌다


def test_an_unknown_pd_locks_guarded_units_only():
    kw = dict(alive=True, busy_stage=None, pd_phase=PD_UNKNOWN)
    assert any("모른다" in r for r in off_reasons(BG, **kw, robot_unit=True, real=False))
    assert any("모른다" in r for r in off_reasons(BG, **kw, robot_unit=False, real=True))
    assert off_reasons(BG, **kw, robot_unit=False, real=False) == []


def test_a_running_stage_locks_guarded_units():
    # 도는 단계가 곧 pd 를 걸 수 있다 — IDLE 스냅샷만 믿고 pd 단위를 끄지 않는다.
    reasons = off_reasons(BG, alive=True, busy_stage="episode", pd_phase="IDLE", robot_unit=True, real=False)
    assert any("episode" in r for r in reasons)


def test_another_stage_running_does_not_lock_the_switch():
    # episode 단계는 수십 초 돈다. 그동안 입력을 껐다 켤 수 없으면 스위치가 쓸모없다 (09.21 fake 실행에서 막혔다).
    assert on_reasons(BG, alive=False, busy_stage="episode", completed={"check"}) == []
    assert off_reasons(BG, alive=True, busy_stage="episode", pd_phase="IDLE", robot_unit=False, real=False) == []


def test_a_fake_input_may_be_switched_off_while_pd_tracks():
    # 입력이 끊기면 체인이 abort → pd HOLD 로 간다. fake 에서는 그 경로를 일부러 밟아 볼 수 있어야 한다.
    assert off_reasons(BG, alive=True, busy_stage=None, pd_phase="TRACKING", robot_unit=False, real=False) == []


# ── 콘솔 흐름 ───────────────────────────────────────────────────────────
def _finish(console, stage):
    console.run_stage(stage, operator="pytest")
    console.session.runner.join(10)
    assert console.session.runner.outcome == "DONE"


def _unit(console, key):
    return console.snapshot()["session"]["units"][key]


def test_units_are_listed_with_what_blocks_them(console):
    console.open("t_fake", operator="pytest")
    u = _unit(console, "up#0")
    assert u["alive"] is False and u["can_on"] is False and any("check" in r for r in u["why_on"])
    assert "check#0" not in console.snapshot()["session"]["units"]      # 전경 명령은 단위가 아니다


def test_switching_on_spawns_the_missions_own_argv_and_off_stops_it(console):
    console.open("t_fake", operator="pytest")
    _finish(console, "check")

    console.toggle_unit("up#0", True, operator="pytest")
    u = _unit(console, "up#0")
    assert u["alive"] is True and u["can_off"] is True and u["argv"] == ["sleep", "30"]

    console.toggle_unit("up#0", False, operator="pytest")
    deadline = time.time() + 8
    while _unit(console, "up#0")["alive"] and time.time() < deadline:
        time.sleep(0.05)
    u = _unit(console, "up#0")
    assert u["alive"] is False and u["stopped"] is True and u["can_on"] is True

    intents = [json.loads(ln) for ln in console.session.intents_path.read_text().splitlines()]
    assert [i["action"] for i in intents if i["action"].startswith("unit/")] == ["unit/on", "unit/off"]
    assert intents[-1]["key"] == "up#0" and intents[-2]["argv"] == ["sleep", "30"]


def test_a_crash_after_switching_back_on_is_not_mistaken_for_the_operators_off(console):
    # 끈 표시는 그때의 pid 에만 붙는다 — 같은 키로 다시 뜬 프로세스가 혼자 죽으면 "끈 것" 이 아니라 "죽은 것" 이다.
    console.open("t_fake", operator="pytest")
    _finish(console, "check")
    s = console.session
    console.toggle_unit("up#0", True, operator="pytest")
    s.units_stopped["up#0"] = -1                     # 예전에 끈 기록(다른 pid)이 남아 있다
    assert _unit(console, "up#0")["alive"] is True

    s.supervisor.stop(["up#0"])                      # 운영자를 거치지 않은 죽음
    u = _unit(console, "up#0")
    assert u["alive"] is False and u["stopped"] is False


def test_a_refused_switch_changes_nothing_and_says_why(console):
    from s2r_console.console import ConsoleError

    console.open("t_fake", operator="pytest")
    with pytest.raises(ConsoleError) as exc:
        console.toggle_unit("up#0", True, operator="pytest")
    assert any("check" in r for r in exc.value.reasons)
    assert _unit(console, "up#0")["alive"] is False


def test_an_unknown_or_foreground_key_is_not_found(console):
    from s2r_console.console import ConsoleError

    console.open("t_fake", operator="pytest")
    for key in ("nope#0", "check#0", "up#9", "up"):
        with pytest.raises(ConsoleError) as exc:
            console.toggle_unit(key, True, operator="pytest")
        assert exc.value.code == 404


def test_a_unit_of_a_real_stage_cannot_be_switched_on_from_the_picture(console, tiny_repo):
    from s2r_console.console import ConsoleError

    mission = tiny_repo / "mission.yaml"
    # conftest 의 미션은 `run.move` 목록으로 끝난다 — 같은 들여쓰기로 배경 명령 하나를 덧붙인다.
    mission.write_text(mission.read_text().rstrip("\n") + "\n"
                       "    - note: 실기 쪽 배경\n"
                       '      argv: [sleep, "30"]\n'
                       "      background: true\n")
    console.open("t_fake", operator="pytest")
    assert "move#1" in console.snapshot()["session"]["units"]
    with pytest.raises(ConsoleError) as exc:
        console.toggle_unit("move#1", True, operator="pytest")
    assert any("승인" in r for r in exc.value.reasons)


# ── 살아 있는 관측 → 끄기 규칙 (콘솔 배선) ─────────────────────────────
def _pd_says(console, phase, *, age_s=0.0):
    from s2r_console.feed import line

    s = console.session
    with s.feed_lock:
        s.feed.ingest(line("hello", domain=98, nodes=["obs", "pd"]))
        s.feed.ingest(line("status", node="pd", data={"node": "pd", "phase": phase, "seq": 1, "ok": True, "reasons": []}))
        s.feed._seen_at["pd"] -= age_s


def _up(console):
    console.open("t_fake", operator="pytest")
    _finish(console, "check")
    console.toggle_unit("up#0", True, operator="pytest")
    assert _unit(console, "up#0")["alive"] is True


def test_a_tracking_pd_locks_the_switch_through_the_console(console):
    _up(console)
    _pd_says(console, "TRACKING")
    u = _unit(console, "up#0")
    assert u["can_off"] is False and any("PD 해제" in r for r in u["why_off"])


def test_a_pd_that_went_quiet_is_unknown_not_free(console):
    # 브리지가 죽거나 status 가 멎으면 마지막 말이 늙는다 — 그때 "자유" 로 치면 토크를 쥔 pd 를 죽이게 된다.
    from s2r_console.console import ConsoleError

    _up(console)
    _pd_says(console, "TRACKING", age_s=30.0)
    u = _unit(console, "up#0")
    assert u["can_off"] is False and any("모른다" in r for r in u["why_off"])
    with pytest.raises(ConsoleError):
        console.toggle_unit("up#0", False, operator="pytest")
    assert any("모른다" in r for r in console.end_reasons())


def test_an_idle_pd_leaves_the_switch_free(console):
    _up(console)
    _pd_says(console, "IDLE")
    assert _unit(console, "up#0")["can_off"] is True and console.end_reasons() == []


# ── HTTP ────────────────────────────────────────────────────────────────
def _post(console, path, body, token=None):
    headers = {"Host": "127.0.0.1:8091", "X-S2R-Console": "1", **({"X-S2R-Lease": token} if token else {})}
    return api.handle(console, "POST", path, query={}, body=json.dumps(body).encode(), headers=headers, client="pytest")


def test_the_switch_needs_the_lease_and_never_takes_argv(console):
    console.open("t_fake", operator="pytest")
    code, _ = _post(console, "/api/unit", {"key": "up#0", "on": True})
    assert code == 409                                                     # lease 없음

    token = _post(console, "/api/lease", {"operator": "pytest"})[1]["token"]
    code, out = _post(console, "/api/unit", {"key": "up#0", "on": True, "argv": ["rm", "-rf", "/"]}, token)
    assert code == 409 and any("check" in r for r in out["reasons"])       # 규칙은 그대로 — 본문의 argv 는 읽지 않는다
    assert _post(console, "/api/unit", {"key": "up#0", "on": "yes"}, token)[0] == 400


def test_every_pd_unit_is_guarded_when_a_mission_launches_more_than_one():
    # mission_dg5f_m_control 은 pd 를 좌·우 두 번 띄운다. 한쪽만 지키면 나머지 하나는 토크를 쥔 채 꺼진다.
    from s2r_console.units import views
    units = {k: UnitCmd(key=k, stage=f"pd_load_{k}", index=0, note="", argv=("sleep", "1"), kind="background",
                        touches_real=True, needs=()) for k in ("pd_l", "pd_r")}
    procs = {k: {"key": k, "alive": True, "pid": 1, "rc": None, "age_s": 1.0} for k in units}
    out = views(units, procs, stopped=(), busy_stage=None, completed=(), pd_phase="TRACKING",
                robot_keys={"pd_l", "pd_r"}, real=False)
    assert all(any("PD 해제" in r for r in v["why_off"]) for v in out.values()), out


def test_a_locked_switch_says_what_to_do_in_few_words():
    # 좁은 상자에서 사유가 잘려 hover 해야만 읽혔다. 실기 중 운영자가 스위치 앞에서 묻는 것은 "뭘 해야 켜지나" 다.
    real = UnitCmd(key="bringup#3", stage="bringup", index=3, note="", argv=("ros2", "launch", "x.launch.py"),
                   kind="background", touches_real=True, needs=("preflight",))
    waiting = UnitCmd(key="sensors#0", stage="sensors", index=0, note="", argv=("python3", "a.py"),
                      kind="background", touches_real=False, needs=("preflight",))
    (why_real,) = on_reasons(real, alive=False, busy_stage=None, completed=())
    (why_wait,) = on_reasons(waiting, alive=False, busy_stage=None, completed=())
    assert "bringup" in why_real and "실행" in why_real and len(why_real) <= 28, why_real
    assert "preflight" in why_wait and len(why_wait) <= 28, why_wait


def test_a_locked_switch_names_the_stage_that_turns_it_on():
    # 운영자가 "노드를 어떻게 켜나" 를 못 찾았다(09.22 실기). 잠금 줄이 그 단계로 데려가려면 서버가 단계를 알려 줘야 한다.
    from s2r_console.units import views
    real = UnitCmd(key="bringup#3", stage="bringup", index=3, note="", argv=("ros2", "launch", "x.launch.py"),
                   kind="background", touches_real=True, needs=("preflight",))
    head = UnitCmd(key="sensors#1", stage="sensors", index=1, note="", argv=("python3", "h.py"),
                   kind="background", touches_real=False, needs=("preflight",))
    kw = dict(stopped=(), busy_stage=None, pd_phase=None, robot_keys=set(), real=True)
    v = views({u.key: u for u in (real, head)}, {}, completed=(), **kw)
    assert v["bringup#3"]["goto"] == "bringup"          # 실기 단계의 명령 — 그 단계를 실행해야 켜진다
    assert v["sensors#1"]["goto"] == "preflight"        # 선행 단계가 먼저다
    v = views({head.key: head}, {}, completed=("preflight",), **kw)
    assert v["sensors#1"]["goto"] is None               # 스위치로 바로 켤 수 있다


# ── 끝낼 때 팔을 떨어뜨리지 않는다 (OpenArm on_deactivate → disable_all) ──────
def test_ending_a_real_run_with_the_arm_bringup_alive_is_refused_with_the_reason():
    # 브링업을 콘솔이 감독하게 된 뒤(09.22) "run 끝내기" 가 PD 해제 뒤 브링업까지 내렸다 — 모터 토크가 전부 풀린다.
    from s2r_console.units import stack_end_reason
    procs = {"bringup#3": {"alive": True}, "bringup#5": {"alive": False}}
    why = stack_end_reason(real=True, stack_keys={"bringup#3", "bringup#5"}, procs=procs)
    assert why and "bringup#3" in why and "토크" in why and "bringup#5" not in why
    assert stack_end_reason(real=False, stack_keys={"bringup#3"}, procs=procs) is None      # fake 플랜트는 떨어질 팔이 없다
    assert stack_end_reason(real=True, stack_keys={"bringup#3"}, procs={"bringup#3": {"alive": False}}) is None


def test_quitting_the_console_keeps_only_the_real_arm_and_hand_drivers():
    # 콘솔 터미널의 Ctrl+C 는 보호 없이 전부 내렸다(팔이 떨어진다). 그렇다고 전부 남기면 09.22 처럼 다시 띄운 콘솔이
    # 목 퍼블리셔를 한 번 더 띄워 시리얼 포트를 둘이 잡는다 — 드라이버만 남기고 나머지는 정지한다.
    from s2r_console.units import keep_on_exit
    procs = {"bringup#3": {"alive": True}, "bringup#5": {"alive": False}, "sensors#1": {"alive": True}, "pd_load_right#0": {"alive": True}}
    assert keep_on_exit(real=True, stack_keys={"bringup#3", "bringup#5"}, procs=procs) == ["bringup#3"]
    assert keep_on_exit(real=True, stack_keys={"bringup#5"}, procs=procs) == []
    assert keep_on_exit(real=False, stack_keys={"bringup#3"}, procs=procs) == []


def test_skipping_is_refused_while_the_arm_may_be_held():
    from s2r_console.units import PD_UNKNOWN, skip_reasons
    assert skip_reasons(skippable=True, busy=False, pd_phase=None) == []
    assert skip_reasons(skippable=True, busy=False, pd_phase="IDLE") == []
    assert any("건너뛸 수 없게" in r for r in skip_reasons(skippable=False, busy=False, pd_phase=None))
    assert any("실행 중" in r for r in skip_reasons(skippable=True, busy=True, pd_phase=None))
    assert any("TRACKING" in r for r in skip_reasons(skippable=True, busy=False, pd_phase="TRACKING"))
    assert any("모른다" in r for r in skip_reasons(skippable=True, busy=False, pd_phase=PD_UNKNOWN))

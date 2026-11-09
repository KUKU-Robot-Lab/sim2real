"""★11.09 tuning 탭 [실험 시작] — 승인 한 번이 pd → engage → 붙들기 → 남은 trial 을 ActionManager 길로 돈다.

fake 만: 명령은 tuning_view._sh 를 바꿔 끼워 sleep/exit 로 흉내 낸다(ros2 launch · --execute 를 실제로 띄우지 않는다).
"""
import json
import re
import shlex
import shutil
import time
from pathlib import Path

import pytest

from s2r_console import actions as A
from s2r_console import tuning_view as TU
from s2r_console.actions import ActionManager
from s2r_console.console import ConsoleError

NAME = "261109_F1off_right"
_REAL_SH = TU._sh


def _fake_sh(body):
    if "ros2 launch" in body:                        # pd launch 대신 오래 사는 프로세스
        return ("sleep", "30")
    if "stationary_batch.py" in body:                # 배치 흉내 — 진행 줄을 찍고 trial 별로 멈춘다
        out = []
        for t in re.findall(r"--trial (\S+)", body):
            out += [f"echo '@@ {t} capture'", f"echo '@@ {t} rebase'"]
            if t == "rb":
                out.append("echo 'rebase rb: AssertionError: fixed hand changed; rerun prepare/IK'; exit 1")
            out.append(f"echo '@@ {t} trial'")
            if t == "t2":
                out.append("echo boom; exit 3")
            if t == "slow":
                out.append("sleep 30")
        return ("sh", "-c", "; ".join(out + ["echo '@@ done'"]))
    return ("sh", "-c", ": " + shlex.quote(body))


@pytest.fixture()
def tuning(console, tiny_repo, monkeypatch):
    prof = (tiny_repo / "profiles" / "t_fake.yaml").read_text().replace("id: t_fake", "id: rh56f1_fake")
    (tiny_repo / "profiles" / "rh56f1_fake.yaml").write_text(prof)
    (tiny_repo / "config").mkdir(exist_ok=True)
    shutil.copy(Path(__file__).resolve().parents[2] / "config/objects.yaml", tiny_repo / "config/objects.yaml")
    monkeypatch.setattr(TU, "_sh", _fake_sh)
    console.open("rh56f1_fake", operator="test")
    m = ActionManager(console, settle_s=0)
    calls = []
    monkeypatch.setattr(A, "REAL_DOMAIN", console.session.profile.domain)      # fake 도메인을 실기 도메인처럼
    monkeypatch.setattr(m, "_trigger", lambda s, service, side, **kw: calls.append((service, side, kw.get("expect"))))
    monkeypatch.setattr(m, "_tune_hold_argv",
                        lambda side: calls.append(("pd/hold", side, ("TRACKING",))) or ("true",))
    monkeypatch.setattr(m, "_wait_pd", lambda *a, **kw: None)
    monkeypatch.setattr(m, "_tune_release_argv", lambda side: ("sh", "-c", ": release"))
    monkeypatch.setattr(A, "scan_processes", lambda: [])     # 이 호스트에 실기 pd 가 떠 있어도 시험은 그것을 보지 않는다
    return m, calls


def make_exp(repo, trials, done=(), partial=()):
    e = repo / TU.ROOT / NAME
    (e / "setup" / "plan").mkdir(parents=True)
    (e / "setup" / "plan" / "plan.json").write_text("{}")
    (e / "setup" / "pd_rh56f1_stationary_right_off.yaml").write_text("x: 1\n")
    (e / "setup" / "trials.txt").write_text("\n".join(trials) + "\n")
    for t in done:
        (e / "runs" / t).mkdir(parents=True)
        (e / "runs" / t / "summary.json").write_text(json.dumps({"pass": True}))
    for t in partial:
        (e / "runs" / t).mkdir(parents=True)
    return e


def wait(m, active=False):
    end = time.monotonic() + 8
    while m.view().get("active") != active and time.monotonic() < end:
        time.sleep(.02)
    return m.view()


def trial_argvs(console):
    return [" ".join(p["argv"]) for p in console.session.supervisor.table() if p["stage"] == "tune_batch_right"]


def test_one_approval_runs_pd_engage_hold_then_one_batch_of_remaining_trials(tuning, console, tiny_repo):
    m, calls = tuning
    make_exp(tiny_repo, ["t1", "t3", "t4"], done=["t1"])
    d = m.describe("tune_right", {"folder": NAME})
    assert d["enabled"], d["reasons"]
    assert [c["step"] for c in d["commands"]] == ["pd", "engage", "hold", "batch"]
    real = TU.plan.__globals__  # 실제 명령 줄은 진짜 _sh 로 본다(실행은 fake)
    real["_sh"], fake = _REAL_SH, real["_sh"]
    try:
        argv = " ".join(TU.plan(tiny_repo, "right", NAME)["batch"])
    finally:
        real["_sh"] = fake
    E = f"{TU.ROOT}/{NAME}"
    assert f"--trial t3 {E}/setup/rebase/t3_a01 {E}/runs/t3" in argv and "--trial t4" in argv
    assert "--trial t1" not in argv and argv.endswith("--execute") and "--side right" in argv
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "DONE", v["note"]
    assert calls[0] == ("pd/hold", "right", ("TRACKING",))
    assert len(trial_argvs(console)) == 1                         # 남은 trial 둘이 프로세스 하나
    tu = m.view()["tuning"]
    assert (tu["step"], tu["trial"], tu["trial_i"], tu["trial_n"]) == ("done", "t4", 2, 2)


def test_failed_trial_stops_there_with_last_error_line(tuning, console, tiny_repo):
    m, _ = tuning
    make_exp(tiny_repo, ["t1", "t2", "t3"])
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "FAILED"
    tu = v["tuning"]
    assert (tu["step"], tu["trial"], tu["trial_i"], tu["error"]) == ("trial", "t2", 2, "boom")


def test_rebase_failure_stops_before_run_and_shows_why(tuning, console, tiny_repo):
    m, _ = tuning
    make_exp(tiny_repo, ["rb", "t1"])
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "FAILED"
    assert (v["tuning"]["step"], v["tuning"]["trial"]) == ("rebase", "rb")
    assert "fixed hand changed" in v["tuning"]["error"]


def test_partial_run_folder_is_never_rewritten(tuning, tiny_repo):
    m, _ = tuning
    make_exp(tiny_repo, ["t1", "t2"], partial=["t1"])
    d = m.describe("tune_right", {"folder": NAME})
    assert not d["enabled"] and any("끊긴 trial t1" in r for r in d["reasons"])
    assert not m.describe("tune_left", {"folder": NAME})["enabled"]      # 다른 팔 폴더


def test_stop_halts_trial_keeps_pd_then_pd_off_stops_only_its_launch(tuning, console, tiny_repo, monkeypatch):
    m, calls = tuning
    make_exp(tiny_repo, ["slow"])
    s = console.session
    s.supervisor.spawn("other#0", stage="other", note="남의 것", argv=["sleep", "30"], background=True, manual=False)
    m.execute("tune_right", {"folder": NAME}, operator="test")
    end = time.monotonic() + 5
    while not any(p["stage"] == "tune_batch_right" and p["alive"] for p in s.supervisor.table()) and time.monotonic() < end:
        time.sleep(.02)
    m.stop_tuning("right")
    v = wait(m)
    assert v["outcome"] == "ABORTED"
    assert s.supervisor.is_alive(A.tune_pd_key("right"))
    with pytest.raises(ConsoleError):
        m.stop_tuning("right")                       # 도는 실험이 없다
    monkeypatch.setattr(m, "_tune_phase", lambda s, side: "HOLD")      # 정지 = pd/hold 뒤
    m.execute("tune_pd_off_right", operator="test")
    assert wait(m)["outcome"] == "DONE"
    assert not s.supervisor.is_alive(A.tune_pd_key("right"))
    assert s.supervisor.is_alive("other#0")


def test_new_experiment_preview_is_the_script_with_a_fresh_name(tuning, tiny_repo):
    m, _ = tuning
    make_exp(tiny_repo, ["t1"])
    d = m.describe("tune_new", {"side": "right", "test": "F1off"})
    argv = d["commands"][0]["argv"]
    assert argv[1].endswith("common/scripts/new_experiment.sh") and argv[3:] == ["right", "off", "F1", "2.0", "3"]
    assert d["enabled"] and not m.describe("tune_new", {"side": "up", "test": "F1off"})["enabled"]


def test_stop_between_steps_never_starts_the_next_trial(tuning, console, tiny_repo, monkeypatch):
    """정지가 루프 확인과 새 runner 사이에 와도(_lock 안에서) 다음 trial 은 뜨지 않는다."""
    m, _ = tuning
    make_exp(tiny_repo, ["t1", "t3"])
    real = A.StageRunner

    class Racing(real):
        def __init__(self, sid, commands, *a, **kw):
            super().__init__(sid, commands, *a, **kw)
            if any("@@ t3" in " ".join(c.argv) for c in commands):
                m.stop_tuning("right")              # 같은 스레드 · RLock — 플래그가 runner 시작 전에 선다
    monkeypatch.setattr(A, "StageRunner", Racing)
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "ABORTED", v["note"]
    assert not any("@@ t3" in a for a in trial_argvs(console))


def test_trial_over_its_time_limit_fails(tuning, console, tiny_repo, monkeypatch):
    m, _ = tuning
    make_exp(tiny_repo, ["slow"])
    monkeypatch.setattr(TU, "TRIAL_FALLBACK_S", 0.3)
    monkeypatch.setattr(TU, "PREP_TIMEOUT_S", 0.0)
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "FAILED" and "시간 한도" in v["note"]


def test_live_tune_pd_blocks_other_work_on_that_arm_and_domain_gate(tuning, console, tiny_repo, monkeypatch):
    m, _ = tuning
    s = console.session
    specs = m._specs
    monkeypatch.setattr(m, "_specs", lambda ss: {**specs(ss), "pd_load_right": ("pd", "setting", ()),
                                                 "pd_load_left": ("pd", "setting", ())})
    s.supervisor.spawn(A.tune_pd_key("right"), stage="tune_pd_right", note="pd", argv=["sleep", "30"],
                       background=True, manual=False)
    d = m.describe("pd_load_right")
    assert not d["enabled"] and any("마찰 실험 pd" in r for r in d["reasons"])
    assert not any("마찰 실험 pd" in r for r in m.describe("pd_load_left")["reasons"])
    assert m.describe("tune_pd_off_right")["enabled"]
    make_exp(tiny_repo, ["t1"])
    monkeypatch.setattr(A, "REAL_DOMAIN", 50)
    assert any("도메인" in r for r in m.describe("tune_right", {"folder": NAME})["reasons"])


def test_pd_launch_sides_and_bash_wrapped_launch_target():
    from s2r_console.supervisor import launch_target
    assert A.pd_launch_sides("ros2 launch x/pd_controller.launch.py sides:=right") == {"right"}
    assert A.pd_launch_sides("ros2 launch x/pd_controller.launch.py execute:=true") == {"right", "left"}
    assert A.pd_launch_sides("ros2 launch x/pd_controller.launch.py sides:=both") == {"right", "left"}
    assert A.pd_launch_sides("ros2 launch x/other.launch.py sides:=right") == set()
    argv = TU.plan.__globals__["_sh"]("ros2 launch deploy/policy_control/launch/pd_controller.launch.py sides:=right")
    assert launch_target(argv) == "pd_controller.launch.py"


def _pd_up(m, console, tiny_repo):
    make_exp(tiny_repo, ["slow"])
    m.execute("tune_right", {"folder": NAME}, operator="test")
    sup, end = console.session.supervisor, time.monotonic() + 5
    while not any(p["stage"] == "tune_batch_right" and p["alive"] for p in sup.table()) and time.monotonic() < end:
        time.sleep(.02)
    m.stop_tuning("right")
    assert wait(m)["outcome"] == "ABORTED"
    assert console.session.supervisor.is_alive(A.tune_pd_key("right"))


def test_pd_off_releases_then_checks_jtc_around_the_stop(tuning, console, tiny_repo, monkeypatch):
    m, _ = tuning
    _pd_up(m, console, tiny_repo)
    d = m.describe("tune_pd_off_right")
    assert d["enabled"], d["reasons"]
    assert [c["step"] for c in d["commands"]] == ["release", "jtc", "pd_off", "jtc"]
    assert "pd_release.py --side right --execute --ensure" in " ".join(A.ActionManager._tune_jtc_argv("right"))
    seen = []
    monkeypatch.setattr(m, "_tune_phase", lambda s, side: "TRACKING")
    real_step = m._tune_step
    monkeypatch.setattr(m, "_tune_step", lambda s, side, c: seen.append(c["step"]) or real_step(s, side, c))
    m.execute("tune_pd_off_right", operator="test")
    v = wait(m)
    assert v["outcome"] == "DONE", v["note"]
    assert seen == ["release", "jtc", "jtc"] and v["tuning"]["step"] == "done"
    assert not console.session.supervisor.is_alive(A.tune_pd_key("right"))


def test_pd_off_skips_release_when_idle_and_stops_where_the_jtc_check_fails(tuning, console, tiny_repo, monkeypatch):
    m, _ = tuning
    _pd_up(m, console, tiny_repo)
    monkeypatch.setattr(m, "_tune_phase", lambda s, side: "IDLE")
    monkeypatch.setattr(A.ActionManager, "_tune_jtc_argv",
                        staticmethod(lambda side: ("sh", "-c", "echo '어긋남: right_forward_effort_controller=active'; exit 4")))
    m.execute("tune_pd_off_right", operator="test")
    v = wait(m)
    assert v["outcome"] == "FAILED"
    assert v["tuning"]["step"] == "jtc" and "forward_effort" in v["tuning"]["error"]
    assert console.session.supervisor.is_alive(A.tune_pd_key("right"))      # pd 를 내리기 전에 멈췄다


def test_pd_off_stops_when_the_pd_phase_is_unknown(tuning, console, tiny_repo, monkeypatch):
    m, _ = tuning
    _pd_up(m, console, tiny_repo)
    monkeypatch.setattr(m, "_tune_phase", lambda s, side: "")              # stale — pd 는 떠 있다
    seen = []
    monkeypatch.setattr(m, "_tune_step", lambda s, side, c: seen.append(c["step"]))
    m.execute("tune_pd_off_right", operator="test")
    v = wait(m)
    assert v["outcome"] == "FAILED" and seen == []
    assert v["tuning"]["step"] == "release" and "pd 상태를 읽지 못함" in v["tuning"]["error"]
    assert console.session.supervisor.is_alive(A.tune_pd_key("right"))

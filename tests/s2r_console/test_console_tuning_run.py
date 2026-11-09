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


def _fake_sh(body):
    if "ros2 launch" in body:                        # pd launch 대신 오래 사는 프로세스
        return ("sleep", "30")
    if "rebase_trial.py" in body and "--trial rb" in body:
        return ("sh", "-c", "echo 'AssertionError: fixed hand changed; rerun prepare/IK'; exit 1")
    if "stationary_tracking.py run" in body:         # trial 별 흉내는 run 에만 — capture · rebase 는 통과
        if "--trial t2" in body:
            return ("sh", "-c", "echo boom; exit 3")
        if "--trial slow" in body:
            return ("sleep", "30")
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
    return [" ".join(p["argv"]) for p in console.session.supervisor.table() if p["stage"] == "tune_trial_right"]


def test_one_approval_runs_pd_engage_hold_then_only_remaining_trials(tuning, console, tiny_repo):
    m, calls = tuning
    make_exp(tiny_repo, ["t1", "t3", "t4"], done=["t1"])
    d = m.describe("tune_right", {"folder": NAME})
    assert d["enabled"], d["reasons"]
    assert [c["step"] for c in d["commands"]] == ["pd", "engage", "hold", *["capture", "rebase", "trial"] * 2]
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "DONE", v["note"]
    assert calls[0] == ("pd/hold", "right", ("TRACKING",))
    assert len(trial_argvs(console)) == 1          # 같은 키로 차례로 — 마지막 것만 표에 남는다
    assert "--trial t4" in trial_argvs(console)[0]
    assert console.session.supervisor.is_alive(A.tune_pd_key("right"))   # pd 는 끄지 않는다
    assert v["tuning"]["trial_i"] == v["tuning"]["trial_n"] == 2


def test_failed_trial_stops_there_with_last_error_line(tuning, console, tiny_repo):
    m, _ = tuning
    make_exp(tiny_repo, ["t1", "t2", "t3"], done=["t1"])
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "FAILED"
    assert v["tuning"]["trial"] == "t2" and v["tuning"]["error"] == "boom"
    assert not any("--trial t3" in a for a in trial_argvs(console))


def test_each_trial_captures_then_rebases_then_runs_the_rebased_plan(tuning, console, tiny_repo):
    m, _ = tuning
    make_exp(tiny_repo, ["t1"])
    d = m.describe("tune_right", {"folder": NAME})
    cap, reb, run = (" ".join(c["argv"]) for c in d["commands"][3:])
    rdir = re.search(r"--out (\S+)/capture", cap).group(1)
    assert f"/{NAME}/setup/rebase/t1_" in rdir and "--side right --seconds 2" in cap
    assert f"--snapshot {rdir}/capture/snapshot.json" in reb and f"--out {rdir}/plan" in reb
    assert f"--plan {rdir}/plan/plan.json" in run and f"--out {TU.ROOT}/{NAME}/runs/t1" in run


def test_rebase_failure_stops_before_run_and_shows_why(tuning, console, tiny_repo):
    m, _ = tuning
    make_exp(tiny_repo, ["rb", "t3"])
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "FAILED"
    assert v["tuning"]["step"] == "rebase" and v["tuning"]["trial"] == "rb"
    assert "fixed hand changed" in v["tuning"]["error"]
    assert trial_argvs(console) == []                # run 은 하나도 안 떴다(t3 도)


def test_partial_run_folder_is_never_rewritten(tuning, tiny_repo):
    m, _ = tuning
    make_exp(tiny_repo, ["t1", "t2"], partial=["t1"])
    d = m.describe("tune_right", {"folder": NAME})
    assert not d["enabled"] and any("끊긴 trial t1" in r for r in d["reasons"])
    assert not m.describe("tune_left", {"folder": NAME})["enabled"]      # 다른 팔 폴더


def test_stop_halts_trial_keeps_pd_then_pd_off_stops_only_its_launch(tuning, console, tiny_repo):
    m, calls = tuning
    make_exp(tiny_repo, ["slow"])
    s = console.session
    s.supervisor.spawn("other#0", stage="other", note="남의 것", argv=["sleep", "30"], background=True, manual=False)
    m.execute("tune_right", {"folder": NAME}, operator="test")
    end = time.monotonic() + 5
    while not any(p["stage"] == "tune_trial_right" and p["alive"] for p in s.supervisor.table()) and time.monotonic() < end:
        time.sleep(.02)
    m.stop_tuning("right")
    v = wait(m)
    assert v["outcome"] == "ABORTED"
    assert s.supervisor.is_alive(A.tune_pd_key("right"))
    with pytest.raises(ConsoleError):
        m.stop_tuning("right")                       # 도는 실험이 없다
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
            if any("--trial t3" in " ".join(c.argv) for c in commands):
                m.stop_tuning("right")              # 같은 스레드 · RLock — 플래그가 runner 시작 전에 선다
    monkeypatch.setattr(A, "StageRunner", Racing)
    m.execute("tune_right", {"folder": NAME}, operator="test")
    v = wait(m)
    assert v["outcome"] == "ABORTED", v["note"]
    assert not any("--trial t3" in a for a in trial_argvs(console))


def test_trial_over_its_time_limit_fails(tuning, console, tiny_repo, monkeypatch):
    m, _ = tuning
    make_exp(tiny_repo, ["slow"])
    monkeypatch.setattr(TU, "TRIAL_FALLBACK_S", 0.3)
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

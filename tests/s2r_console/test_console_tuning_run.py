"""★11.09 tuning 탭 [실험 시작] — 승인 한 번이 pd → engage → 붙들기 → 남은 trial 을 ActionManager 길로 돈다.

fake 만: 명령은 tuning_view._sh 를 바꿔 끼워 sleep/exit 로 흉내 낸다(ros2 launch · --execute 를 실제로 띄우지 않는다).
"""
import json
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
    monkeypatch.setattr(m, "_trigger", lambda s, service, side, **kw: calls.append((service, side, kw.get("expect"))))
    monkeypatch.setattr(m, "_wait_pd", lambda *a, **kw: None)
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
    assert [c["step"] for c in d["commands"]] == ["pd", "engage", "hold", "trial", "trial"]
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

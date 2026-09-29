"""RH56F1 로봇(arm4090) 미션 — 실기 · fake 가 한 정의에서 나오고, 손 명령은 승인 단계에서만 나간다. 09.29 사용자.

"sim2real 과 robot_control 쪽에서 rh56f1 제어 part 연결" · 손마다 개별 포트 · USB RS485 / CANFD 를 상황에 따라.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from mission_core import load_mission
from mission_stages import commands_for, load_runbook

REPO = Path(__file__).resolve().parents[2]
PC = REPO / "deploy" / "policy_control"
sys.path.insert(0, str(PC / "tools"))


def _load(name):
    raw = yaml.safe_load((REPO / "config" / name).read_text(encoding="utf-8"))
    m = load_mission(raw)
    return raw, m, load_runbook(raw["run"], m)


REAL_RAW, REAL, REAL_BOOK = _load("mission_rh56f1_control.yaml")
FAKE_RAW, FAKE, FAKE_BOOK = _load("mission_rh56f1_fake.yaml")


def _cmds(m, book, sid):
    return commands_for(book, m, sid, repo=REPO, execute=True)


def test_the_committed_missions_come_from_the_generator():
    r = subprocess.run([sys.executable, str(REPO / "scripts/ops/make_rh56f1_missions.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_real_and_fake_have_the_same_stages_and_order():
    shape = lambda m: [(s.id, s.group, s.lane, s.needs, s.undoes) for s in m.stages]  # noqa: E731
    assert shape(REAL) == shape(FAKE)
    assert not any(s.touches_real for s in FAKE.stages)
    real_touch = {s.id for s in REAL.stages if s.touches_real}
    assert {"drivers", "hand_right", "hand_left", "probe_right", "probe_left", "pd_load_right", "pd_arm_right",
            "release_right", "shutdown"} <= real_touch
    assert not {"preflight", "hand_check_right", "hand_check_left"} & real_touch          # 읽기만


def test_hands_launch_the_driver_on_the_real_robot_and_the_fake_hand_in_fake():
    for s in ("right", "left"):
        real = " ".join(" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, f"hand_{s}"))
        fake = " ".join(" ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, f"hand_{s}"))
        assert f"rh56f1_driver.py --side {s}" in real and "fake_rh56f1_hand" not in real
        assert f"fake_rh56f1_hand.py --side {s}" in fake and "rh56f1_driver.py" not in fake
        assert f"rh56f1_state_node.py --side {s}" in real and f"rh56f1_state_node.py --side {s}" in fake
        assert "rh56f1_ports" in REAL.stages[[x.id for x in REAL.stages].index(f"hand_{s}")].artifacts


def test_the_probe_moves_one_axis_at_a_time_only_with_execute_and_waits_for_the_eye():
    for s in ("right", "left"):
        cmds = _cmds(REAL, REAL_BOOK, f"probe_{s}")
        plan = commands_for(REAL_BOOK, REAL, f"probe_{s}", repo=REPO, execute=False)
        probes = [i for i, c in enumerate(cmds) if any("rh56f1_axis_probe.py" in a for a in c.argv)]
        axes = [cmds[i].argv[cmds[i].argv.index("--axis") + 1] for i in probes]
        assert sorted(axes) == sorted(["index_1", "middle_1", "ring_1", "pinky_1", "thumb_2", "thumb_1"])
        assert all("--execute" in cmds[i].argv and "--execute" not in plan[i].argv and "--back" in cmds[i].argv
                   for i in probes)
        assert all(cmds[i + 1].manual for i in probes)                                      # 눈으로 확인
        assert cmds[0].manual                                                               # 손 주변 확인이 먼저


def test_pd_uses_the_rh56f1_configs_and_fake_launches_refuse_domain_zero():
    for s in ("right", "left"):
        load = " ".join(_cmds(REAL, REAL_BOOK, f"pd_load_{s}")[0].argv)
        arm = " ".join(_cmds(REAL, REAL_BOOK, f"pd_arm_{s}")[1].argv)
        assert "pd_rh56f1.yaml" in load and "execute:=false" in load and f"rh56f1_{s}_real.yaml" in load
        assert "pd_rh56f1_exec.yaml" in arm and "execute:=true" in arm
        for sid in (f"pd_load_{s}", f"pd_arm_{s}"):
            for c in _cmds(FAKE, FAKE_BOOK, sid):
                if c.argv and c.argv[0] == "ros2":
                    assert "fake:=true" in c.argv and any("_fake.yaml" in a for a in c.argv)


def test_every_stop_names_a_background_command():
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        bg = {f"{s.id}#{i}" for s in m.stages for i, c in enumerate(_cmds(m, book, s.id)) if c.background}
        stops = {k for s in m.stages for c in _cmds(m, book, s.id) for k in c.stop}
        assert stops <= bg, sorted(stops - bg)


def test_every_execute_flag_is_one_the_tool_takes():
    for m, book in ((REAL, REAL_BOOK),):
        for s in m.stages:
            for spec in book.commands.get(s.id, ()):
                if not spec.execute_args:
                    continue
                tool = next(a for a in spec.argv if a.endswith(".py"))
                path = Path(tool.replace("{repo}", str(REPO)))
                text = path.read_text(encoding="utf-8")
                assert all(f'"{f}"' in text for f in spec.execute_args if f.startswith("--")), (s.id, tool)


def test_the_driver_launcher_reads_the_port_file_and_refuses_one_port_for_two_hands():
    import rh56f1_driver as D
    cfg = yaml.safe_load((PC / "config" / "rh56f1_ports.yaml").read_text())
    argv = D.argv_for(cfg, "right")
    assert argv[:4] == ["ros2", "launch", "rh56f1_driver", "rh56f1_right_driver.launch.py"]
    assert f"port:={cfg['right']['port']}" in argv and f"transport:={cfg['right']['transport']}" in argv
    with pytest.raises(ValueError, match="같은 포트"):
        D.argv_for({"right": cfg["right"], "left": dict(cfg["left"], port=cfg["right"]["port"])}, "right")
    with pytest.raises(ValueError, match="transport"):
        D.argv_for({"right": dict(cfg["right"], transport="can"), "left": cfg["left"]}, "right")


def test_the_console_opens_the_rh56f1_fake_profile_with_its_robot_module(tmp_path, monkeypatch):
    import mission_run
    from s2r_console.console import Console
    monkeypatch.setattr(mission_run, "MISSION_LOG_DIR", tmp_path / "logs")
    c = Console(bridge=False)
    try:
        snap = c.snapshot()
        (rh,) = [r for r in snap["robots"] if r["id"] == "openarm_rh56f1"]
        assert [p["id"] for p in rh["profiles"]] == ["rh56f1_real", "rh56f1_fake"]
        s = c.open("rh56f1_fake", operator="pytest")
        assert s.robot is not None and s.robot.id == "openarm_rh56f1"
        assert s.joint_limits.get("r_hj_thumb_1") == pytest.approx((0.0, 2.0944))          # 로봇 모듈의 프로필
        view = c.snapshot()["session"]
        assert view["robot_module"]["id"] == "openarm_rh56f1"
        assert {g["title"] for g in view["robot"]["groups"]} >= {"오른손", "왼손"} or view["robot"]["groups"] == []
    finally:
        c.shutdown()


def test_home_and_cups_are_blocked_on_the_real_robot_until_they_exist():
    """실기: RH56F1 홈 경로 · 두 컵 인지가 아직 없다 — 이유와 함께 막는다. fake 는 goto_home · fake 컵으로 돈다."""
    by = {s.id: s for s in REAL.stages}
    for sid in ("cups", "home_right", "home_left"):
        assert by[sid].blocked, sid
    assert "홈 경로" in by["home_right"].blocked and "인지" in by["cups"].blocked
    assert not any(s.blocked for s in FAKE.stages)
    fake_home = " ".join(" ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, "home_right"))
    assert "--only pd_engage" in fake_home and "--only pd_goto_home" in fake_home


def test_the_pour_fj_policy_stage_runs_the_picked_contract_with_the_venv_python():
    real = _cmds(REAL, REAL_BOOK, "policy_pourfj")
    node = next(c for c in real if any("pour_fj_node.py" in a for a in c.argv))
    assert node.background and node.argv[0].endswith(".venv/bin/python")
    assert any("contract:=" in a and "pour_fj_contract.json" in a for a in node.argv)
    assert any(a == "max_episode_s:=15.0" for a in node.argv)                      # 정책 카드 deploy
    by = {s.id: s for s in REAL.stages}
    assert {"home_right", "home_left", "cups"} <= set(by["policy_pourfj"].needs) and by["policy_pourfj"].touches_real
    assert [c for c in real if "episode/start" in " ".join(c.argv)][0].argv[-1] == "--execute"   # execute 일 때만
    assert {la.id: la.focus for la in REAL.lanes}["both"] == "policy_pourfj"


def test_the_rh56f1_robot_offers_the_pour_fj_policy_for_both_arms():
    from s2r_console import robots as RB
    (rh,) = [r for r in RB.scan(REPO / "deploy/s2r_console/robots")[0] if r.id == "openarm_rh56f1"]
    assert rh.slots["both"] == "pourfj_both" and RB.slot_contracts("pourfj_both") == ("pour_fj_contract.json",)
    assert REAL.artifacts["pourfj_both"].endswith("both_rh_pourfj_f01/pour_fj_contract.json")

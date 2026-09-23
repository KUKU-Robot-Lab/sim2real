"""감독자의 정지 — 그룹 **전체**가 끝나야 끝난 것이다.

`ros2 launch` 는 SIGTERM 을 받으면 제 자식(pd_node)을 기다리지 않고 먼저 끝날 수 있다. 그룹 리더만 기다리면
토크를 쥔 노드가 고아로 남고, 화면에는 "정지됨" 으로 나온다.
"""
from __future__ import annotations

import os
import sys
import time

import s2r_console._paths  # noqa: F401
from s2r_console import supervisor as SV

# 리더는 TERM 에 바로 끝나고, 자식은 TERM 을 무시한다 — launch 가 먼저 죽고 노드가 남는 모양.
LEADER = r"""
import os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time, sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('up', flush=True); time.sleep(60)"],
    stdout=open(sys.argv[1], "w"))
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
open(sys.argv[2], "w").write(str(child.pid))
time.sleep(60)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:                                              # 좀비는 끝난 것이다
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_stop_does_not_return_while_a_child_of_the_group_survives(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "KILL_GRACE_S", 0.6)
    sup = SV.Supervisor(cwd=tmp_path, run_dir=tmp_path, env=dict(os.environ))
    ready, pidfile = tmp_path / "ready", tmp_path / "child.pid"
    sup.spawn("up#0", stage="up", note="launch 흉내", argv=[sys.executable, "-c", LEADER, str(ready), str(pidfile)],
              background=True, manual=False)
    deadline = time.time() + 10
    while time.time() < deadline and not (pidfile.exists() and ready.exists() and ready.read_text()):
        time.sleep(0.05)
    child = int(pidfile.read_text())
    assert _alive(child)

    sup.stop(["up#0"])

    assert not _alive(child)                          # 리더가 먼저 죽었어도 그룹에 남은 자식까지 끝낸다


def test_a_launch_already_running_outside_the_console_is_found_and_ours_is_not():
    # 09.22 실기: 3 시간 전 run 이 남긴 팔 브링업이 살아 있는데 새 콘솔이 한 번 더 띄웠다 — 같은 CAN 에 둘.
    from s2r_console.supervisor import foreign_launches, launch_target
    t = launch_target(["ros2", "launch", "openarm_bringup", "openarm.bimanual.launch.py", "use_rviz:=false"])
    assert t == "openarm.bimanual.launch.py"
    assert launch_target(["ros2", "launch", "/r/deploy/policy_control/launch/pd_controller.launch.py"]) == "pd_controller.launch.py"
    assert launch_target(["python3", "x.py"]) is None
    procs = [(10, 10, "/usr/bin/python3 /opt/ros/humble/bin/ros2 launch openarm_bringup openarm.bimanual.launch.py a:=1"),
             (11, 10, "/opt/ros/humble/lib/controller_manager/ros2_control_node --ros-args"),        # launch 의 자식 — 따로 세지 않는다
             (20, 20, "/usr/bin/python3 /opt/ros/humble/bin/ros2 launch openarm_bringup openarm.bimanual.launch.py"),
             (30, 30, "/usr/bin/python3 /opt/ros/humble/bin/ros2 launch dg5f_driver dg5f_right_driver.launch.py")]
    assert [p for p, _ in foreign_launches(t, procs, own_pgids={20})] == [10]      # 20 은 콘솔이 띄운 것
    assert foreign_launches("dg5f_left_driver.launch.py", procs, set()) == []


# ── 자식 죽음 판정 (09.23 실기에서 둘 다 멀쩡한 드라이버를 실패로 만들었다) ──────────────
DIED_JSB = ("[ERROR] [spawner-3]: process has died [pid 1183517, exit code 1, "
            "cmd '/opt/ros/humble/lib/controller_manager/spawner joint_state_broadcaster "
            "-c /dg5f_right/controller_manager --ros-args'].")
DIED_ARM = ("[ERROR] [spawner-4]: process has died [pid 1149944, exit code 1, "
            "cmd '/opt/ros/humble/lib/controller_manager/spawner left_joint_trajectory_controller "
            "right_joint_trajectory_controller -c /controller_manager --ros-args'].")
ALREADY = "[WARN] [spawner_joint_state_broadcaster]: Controller already loaded, skipping load_controller"
ACTIVE = "[ERROR] [ctl]: Controller 'joint_state_broadcaster' can not be configured from 'active' state."


def test_a_spawner_that_died_because_the_controller_was_already_active_is_not_a_death():
    assert SV.benign_spawner_death(DIED_JSB, [ALREADY, ACTIVE, DIED_JSB])


def test_a_spawner_death_without_that_evidence_is_a_real_death():
    assert not SV.benign_spawner_death(DIED_JSB, [DIED_JSB])
    # 두 컨트롤러 중 하나만 '이미 로드' 면 진짜 죽음으로 본다(보수적)
    half = "[WARN] [spawner_left_joint_trajectory_controller]: Controller already loaded, skipping load_controller"
    assert not SV.benign_spawner_death(DIED_ARM, [half, DIED_ARM])


def test_a_node_death_that_is_not_a_spawner_is_a_real_death():
    line = "[ERROR] [pd_node-1]: process has died [pid 5, exit code 1, cmd '/x/pd_node --ros-args']."
    assert not SV.benign_spawner_death(line, [ALREADY, line])


def test_child_deaths_ignores_lines_the_previous_launch_of_the_same_stage_wrote(tmp_path):
    sup = SV.Supervisor(cwd=tmp_path, run_dir=tmp_path, env=dict(os.environ))
    log = tmp_path / "proc" / "drivers_3.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(DIED_ARM + "\n")                       # 지난 기동이 남긴 죽음 줄
    proc = sup.spawn("drivers#3", stage="drivers", note="n", argv=[sys.executable, "-c", "pass"],
                     background=True, manual=False)
    proc.popen.wait(timeout=10)
    assert sup.child_deaths("drivers#3") == []            # 이번 기동이 쓴 것은 없다
    with log.open("a") as fh:
        fh.write(DIED_JSB + "\n")
    assert sup.child_deaths("drivers#3") == [DIED_JSB]    # 이번 기동의 줄은 잡는다
    sup.stop()

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

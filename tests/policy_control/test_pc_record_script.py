"""rh56f1_record.sh — 이름마다 따로 기록한다(10.09 실기: 양팔 정책을 동시에 돌리자 왼팔 기록 시작이 오른팔 기록을
'남은 기록'으로 끄고, 오른팔 단계의 기록 끝이 왼팔 기록을 껐다). ros2 는 가짜(SIGINT 를 받으면 끝나는 sleep)."""
import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools" / "rh56f1_record.sh"


@pytest.fixture
def env(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    # 파이썬 가짜 — 배경(&)으로 뜬 bash 는 SIGINT 가 무시된 채 시작해 trap 을 못 건다(진짜 ros2 는 signal() 로 다시 건다)
    (fake / "ros2").write_text("#!/usr/bin/env python3\nimport signal, sys, time\n"
                               "signal.signal(signal.SIGINT, lambda *a: sys.exit(0))\nwhile True:\n    time.sleep(0.1)\n")
    (fake / "ros2").chmod(0o755)
    e = dict(os.environ)
    e.update({"PATH": f"{fake}:{e['PATH']}", "RH56F1_RECORD_DIR": str(tmp_path / "run"),
              "RH56F1_RECORD_ENV": "true", "RH56F1_RECORD_OUT": str(tmp_path / "bags"), "RH56F1_RECORD_NO_INFO": "1"})
    yield e
    subprocess.run(["bash", str(SCRIPT), "stop"], env=e, capture_output=True, timeout=30)


def _pids(e, name):
    d = Path(e["RH56F1_RECORD_DIR"]) / name
    return [int((d / f"{k}.pid").read_text()) for k in ("arm", "hand") if (d / f"{k}.pid").exists()]


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_two_arms_record_at_once_and_each_stop_ends_only_its_own(env):
    for name in ("aglt_right", "aglt_left"):
        r = subprocess.run(["bash", str(SCRIPT), "start", name], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
    right, left = _pids(env, "aglt_right"), _pids(env, "aglt_left")
    assert len(right) == 2 and len(left) == 2 and all(map(_alive, right + left))       # 왼팔 시작이 오른팔을 안 껐다
    subprocess.run(["bash", str(SCRIPT), "stop", "aglt_right"], env=env, capture_output=True, timeout=30)
    time.sleep(0.3)
    assert not any(map(_alive, right)) and all(map(_alive, left))                       # 오른팔 끝이 왼팔을 안 껐다
    subprocess.run(["bash", str(SCRIPT), "stop"], env=env, capture_output=True, timeout=30)   # 이름 없이 = 전부(shutdown)
    time.sleep(0.3)
    assert not any(map(_alive, left))


def test_restarting_the_same_name_finishes_its_leftover_first(env):
    subprocess.run(["bash", str(SCRIPT), "start", "aglt_right"], env=env, capture_output=True, timeout=30)
    old = _pids(env, "aglt_right")
    subprocess.run(["bash", str(SCRIPT), "start", "aglt_right"], env=env, capture_output=True, timeout=30)
    time.sleep(0.3)
    assert not any(map(_alive, old)) and all(map(_alive, _pids(env, "aglt_right")))

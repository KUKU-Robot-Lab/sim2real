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


def test_stop_forgets_the_finished_recording(env):
    # 폴더가 logs/ 에 남으므로 이름 없는 stop 이 끝난 기록마다 bag info 를 다시 돌리면 안 된다
    subprocess.run(["bash", str(SCRIPT), "start", "aglt_right"], env=env, capture_output=True, timeout=30)
    last_out = Path(env["RH56F1_RECORD_DIR"]) / "aglt_right" / "last_out"
    assert last_out.exists()
    subprocess.run(["bash", str(SCRIPT), "stop", "aglt_right"], env=env, capture_output=True, timeout=30)
    assert not last_out.exists()


def test_camera_gets_its_own_third_recorder_and_stops_with_the_name(env, tmp_path):
    # 11.04 사용자: 머리 카메라 영상은 세 번째 묶음 — arm 에 섞으면 큰 메시지가 /joint_states 기록을 늦춘다
    fake = tmp_path / "bin" / "camera_bag_recorder.py"
    fake.write_text((tmp_path / "bin" / "ros2").read_text())
    fake.chmod(0o755)
    e = dict(env, CAMERA="/camera/camera/color/image_raw/compressed /head/joint_states", CAMERA_HZ="10",
             RH56F1_CAMERA_CMD=str(fake))
    r = subprocess.run(["bash", str(SCRIPT), "start", "episode_x"], env=e, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    d = Path(env["RH56F1_RECORD_DIR"]) / "episode_x"
    cam = int((d / "camera.pid").read_text())
    assert _alive(cam) and len(_pids(env, "episode_x")) == 2
    argv = Path(f"/proc/{cam}/cmdline").read_bytes().split(b"\0")
    assert b"--hz" in argv and b"10" in argv and b"/head/joint_states" in argv and any(a.endswith(b"/camera") for a in argv)
    subprocess.run(["bash", str(SCRIPT), "stop", "episode_x"], env=env, capture_output=True, timeout=30)
    time.sleep(0.3)
    assert not _alive(cam) and not (d / "camera.pid").exists()


def test_without_camera_there_is_no_third_recorder(env):
    subprocess.run(["bash", str(SCRIPT), "start", "episode_y"], env=env, capture_output=True, timeout=30)
    assert not (Path(env["RH56F1_RECORD_DIR"]) / "episode_y" / "camera.pid").exists()


def test_hand_bag_keeps_the_raw_tips_and_the_tip_zero_topics(env):
    # 11.04 사용자: home · rehome 에서 잡은 손끝 bias 와 영점 뺀 값도 기록 — 원값 tip_forces 는 그대로 같이
    subprocess.run(["bash", str(SCRIPT), "start", "episode_z"], env=env, capture_output=True, timeout=30)
    hand = int((Path(env["RH56F1_RECORD_DIR"]) / "episode_z" / "hand.pid").read_text())
    argv = Path(f"/proc/{hand}/cmdline").read_bytes().split(b"\0")
    for s in ("right", "left"):
        for t in ("tip_forces", "tip_forces_zeroed", "tip_bias", "tip_zero_status", "joint_forces", "joint_forces_zeroed"):
            assert f"/hand_{s}/{t}".encode() in argv, (s, t)


def test_pd_bag_keeps_policy_obs_action_of_grasp_and_place_without_doubling_extra(env):
    # 11.10 사용자: 칵테일 놓기 입력이 bag 에 없었다 — 잡기(ns <팔>) · 놓기(ns <팔>_place) 정책 입출력을 기본으로. EXTRA(arm bag)와 겹치면 뺀다
    e = dict(env, EXTRA="/policy_control/joint_target /policy_control/right/obs")
    subprocess.run(["bash", str(SCRIPT), "start", "episode_p"], env=e, capture_output=True, timeout=30)
    pd = int((Path(env["RH56F1_RECORD_DIR"]) / "episode_p" / "pd.pid").read_text())
    argv = Path(f"/proc/{pd}/cmdline").read_bytes().split(b"\0")
    for s in ("right", "left"):
        for n in (s, f"{s}_place"):
            for t in ("obs", "action", "hand_target", "policy_action"):
                if (n, t) != ("right", "obs"):
                    assert f"/policy_control/{n}/{t}".encode() in argv, (n, t)
        assert f"/policy_control/status/rh_place_node_{s}".encode() in argv
    assert b"/policy_control/right/obs" not in argv and b"/policy_control/joint_target" not in argv

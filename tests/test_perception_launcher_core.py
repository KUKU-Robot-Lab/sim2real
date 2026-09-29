"""perception_launcher_core 순수 로직 검증. ROS·ssh 불필요."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402
from perception_launcher_core import (  # noqa: E402
    Command, RemoteState, build_status, last_crash, parse_command, parse_remote_status, plan_actions,
)

REG = load_registry(DEFAULT_REGISTRY)


def test_parse_start_resolves_aliases_and_dedups():
    cmd = parse_command(json.dumps({"op": "start", "objects": ["cup_big_s080", "shaker_closed",
                                                                "cup_big_s100"], "viewer": True}), REG)
    assert cmd == Command(op="start", objects=("cup_big_s100", "shaker_closed"), viewer=True, camera=False)


def test_parse_rejects_bad_input():
    with pytest.raises(ValueError, match="op"):
        parse_command(json.dumps({"objects": ["shaker_closed"]}), REG)
    with pytest.raises(ValueError, match="unknown object"):
        parse_command(json.dumps({"op": "start", "objects": ["teapot"]}), REG)
    with pytest.raises(ValueError, match="objects"):
        parse_command(json.dumps({"op": "start", "objects": []}), REG)
    with pytest.raises(ValueError, match="JSON"):
        parse_command("not json", REG)


def test_parse_stop_and_viewer():
    assert parse_command('{"op":"stop","camera":true}', REG) == Command("stop", (), None, True)
    assert parse_command('{"op":"viewer","on":false}', REG) == Command("viewer", (), False, False)


def test_parse_remote_status():
    st = parse_remote_status('{"camera_up": true, "containers": {"fpp_shaker_closed": "Up 3 minutes"},'
                             ' "viewer_up": false}')
    assert st == RemoteState(camera_up=True, containers={"fpp_shaker_closed": "Up 3 minutes"},
                             viewer_up=False)
    assert st.pose_tx_up is False                                    # 옛 status.sh(키 없음)는 꺼진 것으로
    st = parse_remote_status('{"camera_up": true, "containers": {}, "viewer_up": false, "pose_tx_up": true}')
    assert st.pose_tx_up is True
    with pytest.raises(ValueError, match="status"):
        parse_remote_status("garbage")


def test_plan_start_is_idempotent_and_prunes_extras():
    state = RemoteState(camera_up=True, containers={"fpp_shaker_closed": "Up 1 minute",
                                                    "fpp_old": "Up 9 minutes"}, viewer_up=False, pose_tx_up=True)
    cmd = Command("start", ("shaker_closed", "cup_big_s100"), viewer=True, camera=False)
    assert plan_actions(cmd, state) == [("fpp_down", "fpp_old"), ("fpp_up", "cup_big_s100"),
                                        ("viewer_up",)]


def test_plan_start_cold_brings_camera_first():
    state = RemoteState(camera_up=False, containers={}, viewer_up=False)
    cmd = Command("start", ("shaker_closed",), viewer=None, camera=False)
    assert plan_actions(cmd, state) == [("camera_up",), ("fpp_up", "shaker_closed"), ("pose_tx_up",)]


def test_the_pose_sender_is_started_once_and_stopped_with_the_containers():
    """09.26: 영상 · FP++ 는 vision-3090 안에서만 돌고 자세만 UDP 로 넘어온다 — 송신기 없이는 정책 입력이 없다."""
    running = RemoteState(camera_up=True, containers={"fpp_shaker_closed": "Up 1 minute"}, viewer_up=False,
                          pose_tx_up=True)
    assert plan_actions(Command("start", ("shaker_closed",), None, False), running) == []
    stopping = plan_actions(Command("stop", (), None, False), running)
    assert stopping == [("fpp_down", "fpp_shaker_closed"), ("pose_tx_down",)]
    idle = RemoteState(camera_up=True, containers={}, viewer_up=False)
    assert ("pose_tx_down",) not in plan_actions(Command("stop", (), None, False), idle)


def test_plan_stop_tears_down_everything_camera_only_when_asked():
    state = RemoteState(camera_up=True, containers={"fpp_a": "Up", "fpp_b": "Exited (1)"}, viewer_up=True)
    assert plan_actions(Command("stop", (), None, False), state) == [
        ("viewer_down",), ("fpp_down", "fpp_a"), ("fpp_down", "fpp_b")]
    assert plan_actions(Command("stop", (), None, True), state)[-1] == ("camera_down",)


def test_plan_viewer_toggle():
    up = RemoteState(True, {}, True)
    assert plan_actions(Command("viewer", (), True, False), up) == []
    assert plan_actions(Command("viewer", (), False, False), up) == [("viewer_down",)]


def test_build_status_shape():
    st = RemoteState(True, {"fpp_shaker_closed": "Up 2 minutes"}, False)
    out = build_status(st, camera_hz=29.9, pose_ages={"shaker_closed": 0.05, "cup_big_s100": None},
                       busy=False, error=None)
    assert out["camera_hz"] == 29.9 and out["camera_up"] is True
    assert out["objects"]["shaker_closed"] == {"container": "Up 2 minutes", "pose_age_s": 0.05, "crash": None}
    assert out["objects"]["cup_big_s100"] == {"container": None, "pose_age_s": None, "crash": None}
    assert out["viewer"] is False and out["busy"] is False and out["error"] is None
    assert build_status(None, 0.0, {}, True, "ssh failed")["error"] == "ssh failed"


OOM_LOG = """[cup_tracking_node-1] [INFO] tracking started
[cup_tracking_node-1] Traceback (most recent call last):
[cup_tracking_node-1]   File "/workspace/x.py", line 3, in f
[cup_tracking_node-1]     return self.reciprocal() * other
[cup_tracking_node-1] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 296.00 MiB. GPU 0 has a total capacity of 23.53 GiB of which 157.12 MiB is free.
"""


def test_a_tracker_that_died_inside_an_up_container_is_reported():
    """09.29: 컨테이너는 Up 인데 추적 노드가 CUDA OOM 으로 죽어 자세가 0 이었다."""
    assert last_crash(OOM_LOG).startswith("torch.OutOfMemoryError: CUDA out of memory")
    assert last_crash("[cup_tracking_node-1] [INFO] ok\n") is None
    st = parse_remote_status('{"camera_up": true, "containers": {"fpp_cup_big_s100": "Up 2 minutes"}, "viewer_up": false,'
                             ' "crashes": {"fpp_cup_big_s100": "torch.OutOfMemoryError: x"},'
                             ' "gpu": {"used_mib": 23934, "total_mib": 24576}}')
    out = build_status(st, 30.0, {"cup_big_s100": None}, False, None)
    assert out["objects"]["cup_big_s100"]["crash"] == "torch.OutOfMemoryError: x"
    assert out["gpu"] == {"used_mib": 23934, "total_mib": 24576}
    old = build_status(parse_remote_status('{"camera_up": true, "containers": {}, "viewer_up": false}'), 0.0, {}, False, None)
    assert old["gpu"] is None                                       # 옛 status.sh 는 모른다

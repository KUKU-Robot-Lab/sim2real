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
    # ★11.05 perception_ctl --add 는 modes 에 add 가 없으면 보내지 않는다 — 런처가 늘 알린다
    assert out["modes"] == ["add"] and build_status(None, 0.0, {}, False, None)["modes"] == ["add"]


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


def _group_of(name):
    return REG.get(name).fpp.get("group")


def test_grouped_cups_start_one_container_and_status_points_each_cup_at_it():
    """10.08: 색 다른 cyl60 셋은 fpp_cups 하나 — 물체마다 컨테이너를 띄우지 않는다. 옛 물체별 컨테이너는 내린다."""
    cmd = parse_command(json.dumps({"op": "start", "objects": ["cyl60", "cyl60_blue", "cyl60_pink"]}), REG)
    state = RemoteState(camera_up=True, containers={"fpp_cyl60": "Up 1 h", "fpp_cyl60_blue": "Up 1 h"},
                        viewer_up=False, pose_tx_up=True)
    acts = plan_actions(cmd, state, group_of=_group_of)
    assert acts == [("fpp_down", "fpp_cyl60"), ("fpp_down", "fpp_cyl60_blue"), ("fpp_group_up", "cups")]
    up = RemoteState(camera_up=True, containers={"fpp_cups": "Up 5 s"}, viewer_up=False, pose_tx_up=True)
    assert plan_actions(cmd, up, group_of=_group_of) == []
    st = build_status(up, 30.0, {"cyl60": 0.1, "cyl60_pink": None}, False, None, group_of=_group_of)
    assert st["objects"]["cyl60"]["container"] == "Up 5 s" and st["objects"]["cyl60_pink"]["container"] == "Up 5 s"


def test_one_object_container_failure_does_not_stop_the_others():
    """★10.13 사용자 실기: 오렌지 · 블루 중 하나가 막히면 단계 전체가 실패했다 — 실패를 모으고 나머지는 계속."""
    from perception_launcher_core import run_actions
    done = []

    def do(action):
        if action == ("fpp_group_up", "shaker_c"):
            raise RuntimeError("rc=1")
        done.append(action)

    acts = [("camera_up",), ("fpp_group_up", "shaker_c"), ("fpp_group_up", "source240b"), ("pose_tx_up",)]
    assert run_actions(acts, do) == ["shaker_c: rc=1"]
    assert done == [("camera_up",), ("fpp_group_up", "source240b"), ("pose_tx_up",)]


def test_all_object_containers_failing_or_a_shared_action_failing_still_raises():
    from perception_launcher_core import run_actions

    def fail_objects(action):
        if action[0] == "fpp_group_up":
            raise RuntimeError("rc=1")

    with pytest.raises(RuntimeError, match="하나도"):
        run_actions([("fpp_group_up", "shaker_c"), ("fpp_group_up", "source240b")], fail_objects)

    def fail_camera(action):
        if action[0] == "camera_up":
            raise RuntimeError("no camera")

    with pytest.raises(RuntimeError, match="no camera"):
        run_actions([("camera_up",), ("fpp_group_up", "shaker_c")], fail_camera)
    assert run_actions([], fail_camera) == []


def test_status_carries_the_partial_failure_warning():
    st = parse_remote_status('{"camera_up": true, "containers": {}, "viewer_up": false}')
    assert build_status(st, 0.0, {}, False, None)["warning"] is None
    assert build_status(st, 0.0, {}, False, None, warning="fpp_shaker_c: rc=1")["warning"] == "fpp_shaker_c: rc=1"


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


def test_next_group_starts_only_after_previous_first_round():
    """★11.03 실기: 묶음 셋이 동시에 첫 회차를 하며 4090 24 GB 가 차 셋 다 warp alloc 실패 — 하나씩 띄운다."""
    from perception_launcher_core import FIRST_ROUND_MARK, run_actions, wait_first_round
    clock, events = _Clock(), []
    logs = {"source240b": ["true\nloading", "true\nloading", f"true\n{FIRST_ROUND_MARK} · generation 1 · 오류 없음"],
            "shaker_c": ["true\n", f"true\n{FIRST_ROUND_MARK} · generation 0 · 오류 PerceptionError: x"]}
    notes = []

    def probe(group):
        events.append(("probe", group))
        return logs[group].pop(0)

    def do(action):
        events.append(("up", action[1]))
        notes.append(wait_first_round(lambda: probe(action[1]), clock=clock, sleep=clock.sleep))

    assert run_actions([("fpp_group_up", "source240b"), ("fpp_group_up", "shaker_c")], do) == []
    assert events == [("up", "source240b"), *[("probe", "source240b")] * 3, ("up", "shaker_c"), *[("probe", "shaker_c")] * 2]
    assert notes[0] is None and "PerceptionError: x" in notes[1]          # 첫 회차 오류는 참고로만(다음은 계속)


def test_first_round_wait_warns_after_limit_and_raises_when_container_stops():
    from perception_launcher_core import FIRST_ROUND_S, wait_first_round
    clock = _Clock()
    warn = wait_first_round(lambda: "true\nloading", clock=clock, sleep=clock.sleep)
    assert warn and "90 s" in warn and FIRST_ROUND_S <= clock.t < FIRST_ROUND_S + 2.1
    clock = _Clock()
    with pytest.raises(RuntimeError, match="멈췄다.*RuntimeError: boom"):
        wait_first_round(lambda: "false\nTraceback (most recent call last):\nRuntimeError: boom", clock=clock,
                         sleep=clock.sleep)
    assert clock.t == 0.0


def test_probe_errors_are_unknown_and_polling_continues():
    """리뷰 H1: ssh 오류 · 시간 초과로 대기를 건너뛰면 다음 묶음이 바로 떠 다시 동시 첫 회차가 된다."""
    import subprocess
    from perception_launcher_core import FIRST_ROUND_MARK, wait_first_round
    clock, calls = _Clock(), []
    seq = [RuntimeError("ssh rc=255"), subprocess.TimeoutExpired("ssh", 15), f"true\n{FIRST_ROUND_MARK} · generation 1 · 오류 없음"]

    def probe():
        calls.append(1)
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    assert wait_first_round(probe, clock=clock, sleep=clock.sleep) is None and len(calls) == 3

    def always_fail():
        raise RuntimeError("ssh down")

    clock = _Clock()
    warn = wait_first_round(always_fail, clock=clock, sleep=clock.sleep)
    assert warn and "90 s" in warn and clock.t >= 90.0


def test_probe_cmd_finds_the_mark_even_after_long_logs(tmp_path):
    """리뷰 M4: --tail 로 읽으면 표시 줄이 밀려난다 — grep -m1 로 첫 줄을 찾는다. docker 를 흉내 내는 스크립트로 bash 에서 돌린다."""
    import os
    import subprocess
    from perception_launcher_core import FIRST_ROUND_MARK, first_round_probe_cmd
    fake = tmp_path / "docker"
    fake.write_text('#!/bin/bash\nif [ "$1" = inspect ]; then [ -f "$STATE" ] && cat "$STATE" || exit 1; exit 0; fi\n'
                    'cat "$LOG"\n')
    fake.chmod(0o755)
    log, state = tmp_path / "log", tmp_path / "state"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "LOG": str(log), "STATE": str(state)}
    run = lambda: subprocess.run(["bash", "-c", first_round_probe_cmd("fpp_x")], env=env,  # noqa: E731
                                 capture_output=True, text=True, check=True).stdout
    state.write_text("true\n")
    log.write_text(f"{FIRST_ROUND_MARK} · generation 1 · 오류 없음\n" + "추적 줄\n" * 500)
    assert run().splitlines() == ["true", f"{FIRST_ROUND_MARK} · generation 1 · 오류 없음"]
    log.write_text("loading\n")
    assert run().splitlines() == ["true", "loading"]
    state.unlink()                                                        # 컨테이너 없음 = false
    assert run().splitlines()[0] == "false"


def test_start_wait_covers_each_group_first_round():
    from perception_launcher_core import FIRST_ROUND_S, start_wait_s
    assert start_wait_s(3) >= 3 * FIRST_ROUND_S and start_wait_s(3) > start_wait_s(2)


def test_snapshot_node_logs_the_first_round_mark_also_without_camera():
    """리뷰 M3: 카메라 영상이 없으면 찍기에 못 들어간다 — 그때도 표시 줄을 찍어 런처가 상한까지 기다리지 않는다."""
    from perception_launcher_core import FIRST_ROUND_MARK
    src = (Path(__file__).resolve().parents[1] / "scripts/nodes/fpp_snapshot_node.py").read_text(encoding="utf-8")
    assert src.count(f'f"{FIRST_ROUND_MARK} · generation') == 2


def test_notes_are_kept_apart_from_the_warning():
    """리뷰 M1: 첫 회차 지연 · 오류는 참고 — warning 에 섞으면 --require-all 이 떠 있는 컨테이너를 실패로 본다."""
    st = parse_remote_status('{"camera_up": true, "containers": {}, "viewer_up": false}')
    out = build_status(st, 0.0, {}, False, None, notes=["shaker_c: 첫 회차가 90 s 안에 안 끝났다"])
    assert out["warning"] is None and out["notes"] == ["shaker_c: 첫 회차가 90 s 안에 안 끝났다"]


def test_add_mode_keeps_other_containers_and_starts_only_missing_ones():
    """★11.05 에피소드가 자기 물체를 확보할 때(perception_ctl --add) — 다른 물체 컨테이너를 내리지 않는다."""
    cmd = parse_command(json.dumps({"op": "start", "objects": ["shaker_closed"], "add": True}), REG)
    assert cmd.add
    other = RemoteState(camera_up=True, containers={"fpp_cup_big_s100": "Up 1 minute"}, viewer_up=False, pose_tx_up=True)
    assert plan_actions(cmd, other) == [("fpp_up", "shaker_closed")]
    assert ("fpp_down", "fpp_cup_big_s100") in plan_actions(Command("start", ("shaker_closed",), None, False), other)
    assert not parse_command(json.dumps({"op": "stop", "add": True}), REG).add          # stop 은 덧붙이기가 없다
    with pytest.raises(ValueError, match="add"):
        parse_command(json.dumps({"op": "start", "objects": ["shaker_closed"], "add": "yes"}), REG)

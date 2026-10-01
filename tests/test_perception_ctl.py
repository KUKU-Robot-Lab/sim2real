"""perception_ctl 순수부(build_payload) — start 는 --viewer 없이 뷰어를 건드리지 않는다."""
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402
from perception_ctl import build_payload  # noqa: E402
from perception_launcher_core import Command, RemoteState, parse_command, plan_actions  # noqa: E402

REG = load_registry(DEFAULT_REGISTRY)


def test_start_without_viewer_flag_omits_key_and_keeps_viewer_running():
    payload = build_payload(Namespace(op="start", objects=["cup_big_s080"], viewer=False), REG)
    assert payload == {"op": "start", "objects": ["cup_big_s100"]}
    import json
    cmd = parse_command(json.dumps(payload), REG)
    assert cmd.viewer is None
    state = RemoteState(camera_up=True, containers={"fpp_cup_big_s100": "Up 1 minute"}, viewer_up=True,
                        pose_tx_up=True)
    assert plan_actions(cmd, state) == []


def test_start_with_viewer_flag_and_stop_payloads():
    assert build_payload(Namespace(op="start", objects=["shaker_closed"], viewer=True), REG) == {
        "op": "start", "objects": ["shaker_closed"], "viewer": True}
    assert build_payload(Namespace(op="stop", camera=True), REG) == {"op": "stop", "camera": True}
    assert build_payload(Namespace(op="viewer", on="off"), REG) == {"op": "viewer", "on": False}
    assert build_payload(Namespace(op="status"), REG) is None


def test_wait_verdict_accepts_an_immediate_no_change_and_catches_errors():
    # 09.22 fake: 런처가 "변경 없음" 으로 즉시 끝나 busy 가 안 켜졌는데 150 s 를 기다렸다 · 실기에서는 실패를 못 보고 통과했다
    from perception_ctl import wait_verdict
    assert wait_verdict(False, False, 1.0, None) is None                 # 아직 이르다
    assert wait_verdict(False, False, 3.5, None) == "ok"                 # 변경 없음 — 끝
    assert wait_verdict(True, True, 30.0, None) is None                  # 일하는 중
    assert wait_verdict(True, False, 12.0, None) == "ok"
    assert wait_verdict(True, False, 12.0, "camera did not publish") == "fail"
    assert wait_verdict(False, False, 4.0, "ssh rc=1") == "fail"


def test_stop_without_the_launcher_calls_the_same_down_scripts_over_ssh():
    """09.28 실기: shutdown 이 런처를 먼저 내려 sensors_off 가 실패했다 — 런처 없이도 저 PC 에서 직접 내린다."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("perception_ctl_direct",
                                                  Path(__file__).resolve().parents[1] / "scripts/ops/perception_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.direct_stop_scripts(False) == [("pose_tx_down.sh",), ("fpp_down.sh", "all")]
    assert mod.direct_stop_scripts(True)[-1] == ("camera_down.sh",)


def test_stop_can_wait_until_the_launcher_is_done():
    # 재등록은 stop 이 끝난 뒤 start 를 보내야 한다 — 런처가 바쁜 동안 start 가 섞이지 않게
    from perception_ctl import parser
    assert parser().parse_args(["stop", "--wait", "60"]).wait == 60.0
    assert parser().parse_args(["stop"]).wait == 0.0


def test_local_host_runs_the_same_scripts_on_this_pc_without_ssh():
    """10.01 arm4090: 로봇 PC 가 카메라 · FP++ 를 같이 돌린다 — 런처 · 직접 내리기 모두 --host local 이면 bash 로."""
    from perception_launcher_core import shell_argv
    assert shell_argv("local", "bash rl_ws/sim2real/scripts/vision/status.sh") == [
        "bash", "-c", "bash rl_ws/sim2real/scripts/vision/status.sh"]
    assert shell_argv("localhost", "x")[0] == "bash"
    assert shell_argv("vision-3090", "x")[:2] == ["ssh", "-o"] and shell_argv("vision-3090", "x")[-2:] == ["vision-3090", "x"]


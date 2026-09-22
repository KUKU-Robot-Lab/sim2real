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
    state = RemoteState(camera_up=True, containers={"fpp_cup_big_s100": "Up 1 minute"}, viewer_up=True)
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

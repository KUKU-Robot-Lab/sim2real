"""★11.19 act_tilt(ACT 정책 붓기) — 미션 등록 · 카드 설정(ckpt · max_vel) 치환 · 범위 거부."""
from pathlib import Path

import pytest
import yaml

from s2r_console import actions as A
from s2r_console.console import ConsoleError

REPO = Path(__file__).resolve().parents[2]


def _run(kind):
    return yaml.safe_load((REPO / "config" / f"mission_rh56f1_{kind}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["control", "control_cam5090", "fake"])
def test_act_tilt_stages_are_registered_with_dry_run_never_touching_real(kind):
    m = _run(kind)
    stages = {s["id"]: s for s in m["stages"]}
    assert stages["act_tilt_dry"].get("touches_real", False) is False   # fake 는 이 키를 쓰지 않는다
    assert stages["act_tilt"].get("touches_real", False) is (kind != "fake")
    for sid in A.ACT_TILT:
        assert set(stages[sid]["needs"]) == {"pd_arm_right", "pd_arm_left"}


def test_real_act_tilt_publishes_only_with_execute_after_a_manual_check():
    run = _run("control")["run"]
    dry, = run["act_tilt_dry"]
    assert not dry.get("execute_args") and "--execute" not in dry["argv"]
    check, go = run["act_tilt"]
    assert check.get("manual") is True
    assert go["execute_args"] == ["--execute"] and go["argv"] == dry["argv"]
    assert go["argv"][3:] == ["act_tilt", A.ACT_TILT_CKPTS[0], "--max-vel", "0.5"]


def test_card_parameters_swap_ckpt_and_max_vel_only_in_the_bridge_command():
    argv = tuple(_run("control")["run"]["act_tilt"][1]["argv"]) + ("--execute",)
    out = A.act_tilt_argv(argv, {"ckpt": "010000", "max_vel": 0.3})
    assert out[:3] == argv[:3] and out[3:] == ("act_tilt", "010000", "--max-vel", "0.3", "--execute")
    assert A.act_tilt_argv(("bash", "-lc", "true"), {"ckpt": "010000"}) == ("bash", "-lc", "true")
    assert A.act_tilt_argv(argv, {}) == argv


@pytest.mark.parametrize("bad", [{"ckpt": "020000"}, {"ckpt": "../x"}, {"max_vel": 0}, {"max_vel": 1.6},
                                 {"max_vel": "0.5"}, {"max_vel": True}])
def test_out_of_range_card_parameters_are_refused(bad):
    with pytest.raises(ConsoleError):
        A.act_tilt_parameters(bad)

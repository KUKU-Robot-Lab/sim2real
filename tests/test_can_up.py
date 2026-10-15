"""can_up — 이미 맞게 올라온 CAN 은 건드리지 않고, 다시 올릴 때는 sudoers 가 허용한 줄과 글자까지 같은 명령만 쓴다."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("can_up", ROOT / "scripts/setup/can_up.py")
can_up = importlib.util.module_from_spec(spec)
spec.loader.exec_module(can_up)


def link(flags=("NOARP", "UP", "LOWER_UP"), bitrate=1_000_000, dbitrate=5_000_000, fd=True, state="ERROR-ACTIVE"):
    return {"flags": list(flags), "linkinfo": {"info_data": {
        "ctrlmode": ["FD"] if fd else [], "state": state,
        "bittiming": {"bitrate": bitrate}, "data_bittiming": {"bitrate": dbitrate}}}}


def test_a_correct_link_is_left_alone():
    assert can_up.why_reconfigure(link()) is None


def test_each_wrong_setting_is_named():
    assert "DOWN" in can_up.why_reconfigure(link(flags=("NOARP",)))
    assert "bitrate" in can_up.why_reconfigure(link(bitrate=500_000))
    assert "dbitrate" in can_up.why_reconfigure(link(dbitrate=2_000_000))
    assert "FD" in can_up.why_reconfigure(link(fd=False))
    assert "BUS-OFF" in can_up.why_reconfigure(link(state="BUS-OFF"))
    assert can_up.why_reconfigure(None)


def test_commands_match_the_sudoers_lines_exactly():
    allowed = {m.group(1) for m in re.finditer(r"NOPASSWD: (.+)$", (ROOT / "scripts/setup/sudoers-s2r-can").read_text(), re.M)}
    issued = {" ".join(cmd) for name in can_up.ALLOWED_IFS for cmd in can_up.link_commands(name)}
    assert issued == allowed


def test_the_rh56f1_drivers_stage_sets_can_without_a_manual_sudo_step():
    import yaml
    run = yaml.safe_load((ROOT / "config/mission_rh56f1_control.yaml").read_text())["run"]["drivers"]
    assert not any("sudo ip link" in " ".join(c.get("argv", [])) for c in run)
    can = [i for i, c in enumerate(run) if "can_up.py" in " ".join(c.get("argv", []))]
    launch = next(i for i, c in enumerate(run) if "openarm.bimanual.launch.py" in c.get("argv", []))
    assert can and can[0] < launch and not run[can[0]].get("manual")

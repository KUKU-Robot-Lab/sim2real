"""check_host 의 목 · 카메라 위치 판단. 10.30 사용자 arm_v2: 5090 = 로봇 제어 · 정책 실행, 4090 = 카메라 · 목 · FP++.
★11.03 사용자: 카메라 · 목 · FP++ 도 5090 — arm_v2 에 hosts.vision 이 없어 5090 이 목 · 카메라를 직접 검사한다."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "deploy/policy_control/config/sites"
spec = importlib.util.spec_from_file_location("check_host", ROOT / "scripts/setup/check_host.py")
assert spec is not None and spec.loader is not None
CH = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = CH        # dataclass 가 모듈을 sys.modules 에서 찾는다
spec.loader.exec_module(CH)
ARM_V2 = yaml.safe_load((SITES / "arm_v2.yaml").read_text())


#: 10.30~11.03 의 두 PC 판(목 · 카메라가 4090) — 판단 규칙 자체는 남아 있어 그대로 본다
SPLIT = {"hosts": {"control": {"ip": "192.168.0.50"}, "vision": {"name": "arm4090", "ip": "192.168.0.40"}}}


def test_arm_v2_puts_head_and_camera_on_arm4090_again_since_1111():
    """11.03~11.10 은 5090 이 목 · 카메라를 직접 봤고, ★11.11 다시 4090(hosts.vision) — 5090 은 저 PC 를 기대한다."""
    away = CH.vision_elsewhere(ARM_V2, {"127.0.0.1", "192.168.0.50", "100.103.21.126"})
    assert away is not None and away["ip"] == ARM_V2["hosts"]["arm4090"]["ip"]
    assert CH.vision_elsewhere(ARM_V2, {"127.0.0.1", ARM_V2["hosts"]["arm4090"]["ip"]}) is None


def test_control_pc_expects_head_and_camera_on_the_vision_host():
    away = CH.vision_elsewhere(SPLIT, {"127.0.0.1", "192.168.0.50", "100.103.21.126"})
    assert away is not None
    assert (away["name"], away["ip"]) == ("arm4090", "192.168.0.40")


def test_vision_pc_still_checks_its_own_head_and_camera():
    assert CH.vision_elsewhere(SPLIT, {"127.0.0.1", "192.168.0.40"}) is None


def test_single_pc_sites_keep_the_local_check():
    v1 = yaml.safe_load((SITES / "arm_v1.yaml").read_text())
    assert CH.vision_elsewhere(v1, {"192.168.0.50"}) is None
    assert CH.vision_elsewhere(None, set()) is None


def test_head_config_found_inside_ssh_command_string_list_argv_or_missing():
    ssh = [{"argv": ["ssh", "-tt", "arm4090-lan",
                     "cd rl_ws/sim2real && exec python3 scripts/nodes/head_joint_publisher.py --config config/head_camset_rh56f1.yaml --ids 1,2"]}]
    assert CH.head_config_path(ssh, Path("/r")) == Path("/r/config/head_camset_rh56f1.yaml")
    local = [{"argv": ["python3", "{repo}/scripts/head_home.py", "--config", "{repo}/config/head_home.yaml"]}]
    assert CH.head_config_path(local, Path("/r")) == Path("/r/config/head_home.yaml")
    assert CH.head_config_path([{"argv": ["python3", "{repo}/scripts/head_move.py"]}], Path("/r")) is None
    run = yaml.safe_load((ROOT / "config/mission_rh56f1_control.yaml").read_text())["run"]
    assert CH.head_config_path(run["head_home"], ROOT).exists()     # 지금 미션이 가리키는 머리 설정

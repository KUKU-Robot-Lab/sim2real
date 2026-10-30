"""check_host 의 목 · 카메라 위치 판단. 10.30 사용자 arm_v2: 5090 = 로봇 제어 · 정책 실행, 4090 = 카메라 · 목 · FP++."""
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


def test_control_pc_expects_head_and_camera_on_the_vision_host():
    away = CH.vision_elsewhere(ARM_V2, {"127.0.0.1", "192.168.0.50", "100.103.21.126"})
    assert away is not None
    assert (away["name"], away["ip"]) == ("arm4090", "192.168.0.40")


def test_vision_pc_still_checks_its_own_head_and_camera():
    assert CH.vision_elsewhere(ARM_V2, {"127.0.0.1", "192.168.0.40"}) is None


def test_single_pc_sites_keep_the_local_check():
    v1 = yaml.safe_load((SITES / "arm_v1.yaml").read_text())
    assert CH.vision_elsewhere(v1, {"192.168.0.50"}) is None
    assert CH.vision_elsewhere(None, set()) is None

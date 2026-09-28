"""손끝 F/T 영점 서비스 고르기 — 이 손의 네임스페이스를 먼저, 애매하면 고르지 않는다."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
TOOL = Path(__file__).resolve().parents[2] / "deploy/policy_control/tools/ft_zero.py"
spec = importlib.util.spec_from_file_location("ft_zero", TOOL)
F = importlib.util.module_from_spec(spec)
sys.modules["ft_zero"] = F
spec.loader.exec_module(F)


def test_the_service_under_this_hands_namespace_wins():
    names = ["/dg5f_left/delto_hardware_interface_node/set_ft_sensor_offset",
             "/dg5f_right/delto_hardware_interface_node/set_ft_sensor_offset", "/other/srv"]
    assert F.pick_service(names, "left")[0] == names[0]
    assert F.pick_service(names, "right")[0] == names[1]


def test_a_single_unnamespaced_service_is_used_and_two_are_refused():
    one = ["/delto_hardware_interface_node/set_ft_sensor_offset"]
    assert F.pick_service(one, "left")[0] == one[0]
    two = one + ["/x/delto_hardware_interface_node/set_ft_sensor_offset"]
    assert F.pick_service(two, "left")[0] is None
    assert F.pick_service([], "left") == (None, "영점 서비스가 없다 — 손 드라이버가 fingertip_sensor:=true 로 떠 있는가")

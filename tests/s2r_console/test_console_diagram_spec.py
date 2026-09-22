"""프로파일의 `diagram:` — 상자와 전선 선언. 틀린 그림은 로드 시점에 거부한다."""
from __future__ import annotations

from pathlib import Path

import pytest

import s2r_console._paths  # noqa: F401
from s2r_console.diagram_spec import Box, Diagram, Wire, parse_diagram
from s2r_console.profiles import ProfileError, parse, scan

SIM2REAL = Path(__file__).resolve().parents[2]
P = Path("p.yaml")

GOOD = {
    "boxes": [
        {"id": "arm", "title": "팔", "col": 0, "unit": "plant#0"},
        {"id": "pour_node", "title": "정책", "col": 1, "status": "pour_node", "ros": ["/pour_node"], "unit": "chain#0"},
        {"id": "pd", "title": "pd", "col": 2, "status": "pd", "ros": ["/pd_node"], "manager": "/controller_manager"},
    ],
    "wires": [
        {"from": "arm", "to": "pour_node", "topic": "/joint_states", "inputs": ["src:arm", "rcv:arm"], "stale_ms": 500},
        {"from": "pour_node", "to": "pd", "topic": "/policy_control/joint_target"},
        {"from": "pour_node", "to": "pd", "topic": "/policy_control/episode", "meter": False},
        {"from": "arm", "to": "pour_node", "topic": "/policy_control/pour/fill_level", "inputs": ["fill"], "on_demand": True},
    ],
}


def test_a_good_diagram_parses_into_frozen_records():
    d = parse_diagram(GOOD, path=P)
    assert isinstance(d, Diagram)
    assert d.boxes[1] == Box(id="pour_node", title="정책", col=1, status="pour_node", ros=("/pour_node",), unit="chain#0")
    assert d.wires[0] == Wire(src="arm", dst="pour_node", topic="/joint_states", inputs=("src:arm", "rcv:arm"), stale_ms=500.0)
    assert d.wires[2].meter is False and d.wires[3].on_demand is True


def test_metered_and_watched_topics_are_split_and_deduplicated():
    d = parse_diagram(GOOD, path=P)
    assert d.metered() == ("/joint_states", "/policy_control/joint_target")
    assert d.watched() == ("/policy_control/episode", "/policy_control/pour/fill_level")   # on_demand 는 세지 않는다


def test_units_lists_each_key_once_in_order():
    assert parse_diagram(GOOD, path=P).units() == ("plant#0", "chain#0")


def _with(**over):
    return {**GOOD, **over}


@pytest.mark.parametrize(("raw", "needle"), [
    ({"boxes": []}, "boxes"),                                                     # 상자가 없다
    (_with(boxez=[]), "boxez"),                                                   # 모르는 키
    (_with(boxes=[*GOOD["boxes"], {"id": "arm", "title": "또", "col": 0}]), "arm"),   # id 중복
    (_with(boxes=[{"id": "a", "title": "t", "col": -1}]), "col"),
    (_with(boxes=[{"id": "a", "title": "t", "col": 0, "color": "red"}]), "color"),
    (_with(boxes=[{"id": "a", "title": "t", "col": 0, "ros": ["pd_node"]}]), "/"),    # 전체 이름이 아니다
    (_with(boxes=[{"id": "a", "title": "t", "col": 0, "unit": "plant"}]), "unit"),    # stage#n 이 아니다
    (_with(wires=[{"from": "arm", "to": "nope", "topic": "/x"}]), "nope"),
    (_with(wires=[{"from": "arm", "to": "pd", "topic": "joint_states"}]), "/"),
    (_with(wires=[{"from": "pd", "to": "arm", "topic": "/x"}]), "왼쪽"),             # 거꾸로 가는 전선
    (_with(wires=[{"from": "arm", "to": "pd", "topic": "/x", "stale_ms": 0}]), "stale_ms"),
    (_with(wires=[{"from": "arm", "to": "pd", "topic": "/x", "colour": 1}]), "colour"),
])
def test_a_malformed_diagram_is_a_load_error(raw, needle):
    with pytest.raises(ProfileError, match=needle):
        parse_diagram(raw, path=P)


def test_profile_carries_the_diagram(tmp_path):
    (tmp_path / "m.yaml").write_text("name: x\n")
    raw = {"schema": "s2r_console/profile/v1", "id": "p", "mission": "m.yaml",
           "domain": {"id": 97, "class": "fake"}, "status_nodes": ["pour_node", "pd"], "diagram": GOOD}
    p = parse(raw, path=tmp_path / "p.yaml", repo=tmp_path)
    assert p.diagram is not None and p.as_dict()["diagram"]["boxes"][0]["id"] == "arm"


def test_a_status_box_must_name_a_status_node_of_the_profile(tmp_path):
    (tmp_path / "m.yaml").write_text("name: x\n")
    raw = {"schema": "s2r_console/profile/v1", "id": "p", "mission": "m.yaml",
           "domain": {"id": 97, "class": "fake"}, "status_nodes": ["pd"], "diagram": GOOD}
    with pytest.raises(ProfileError, match="pour_node"):
        parse(raw, path=tmp_path / "p.yaml", repo=tmp_path)


def test_every_shipped_diagram_unit_is_a_background_command_of_its_mission():
    # 그림은 이제 미션에서 만든다(`wiring.generate`) — 상자에 붙는 스위치는 여전히 그 미션의 배경·수동 명령이어야 한다.
    from s2r_console.console import diagram_of, mission_units

    good, bad = scan(SIM2REAL / "s2r_console" / "profiles", repo=SIM2REAL)
    assert bad == {}
    for p in good:
        units = mission_units(p, repo=SIM2REAL)
        diagram = diagram_of(p, units, repo=SIM2REAL)
        assert diagram.units(), f"{p.id}: 스위치가 붙은 상자가 하나도 없다"
        for key in diagram.units():
            assert key in units, f"{p.id}: {key} 가 미션에 없다"
            # manual 은 그림에 걸 수 있다(스위치는 잠긴다). 전경 명령은 켜 둘 것이 아니라서 안 된다.
            assert units[key].kind in ("background", "manual"), f"{p.id}: {key} 는 배경·수동 명령이 아니다"

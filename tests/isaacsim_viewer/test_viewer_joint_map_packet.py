"""읽기 전용 Isaac 뷰어 — 관절 이름 사상 · UDP 패킷 형식 · 도메인 가드 (ROS·Isaac 불필요)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

import joint_map
import packet
from joint_state_relay import DEFAULT_TOPICS, parse_args, require_ros_domain

pytestmark = pytest.mark.unit

ASSET_URDF = (Path(__file__).resolve().parents[3] / "hdgp" / "assets" / "robot"
              / "openarm_dg5f-m-short-tl_bi_rl" / "openarm_dg5f-m-short-tl_bi_rl.urdf")
POLICY_ENV_YAML = (Path(__file__).resolve().parents[2] / "deploy" / "policies" / "right_aglt"
                   / "params" / "env.yaml")


@pytest.fixture(scope="module")
def table():
    if not joint_map.DEFAULT_PROFILE.is_file():
        pytest.skip(f"robot_control 프로필 없음: {joint_map.DEFAULT_PROFILE}")
    return joint_map.load_profile_table()


# ---------------------------------------------------------------- 이름 사상

def test_arm_source_names_map_to_canonical(table):
    names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    res = joint_map.map_joint_state(names, [0.1 * i for i in range(1, 8)], table)
    assert dict(res.values) == pytest.approx({f"r_aj_{i}": 0.1 * i for i in range(1, 8)})
    assert res.unknown == ()


def test_left_arm_and_hands_map(table):
    res = joint_map.map_joint_state(
        ["openarm_left_joint3", "rj_dg_2_2", "rj_dg_5_1", "lj_dg_1_2"], [0.3, 1.0, 0.5, 0.7], table)
    assert dict(res.values) == pytest.approx(
        {"l_aj_3": 0.3, "r_hj_index_2": 1.0, "r_hj_pinky_1": 0.5, "l_hj_thumb_2": 0.7})


def test_canonical_names_pass_through(table):
    res = joint_map.map_joint_state(["r_aj_1", "head_j_pan", "r_hj_middle_3"], [0.2, -0.1, 0.4], table)
    assert dict(res.values) == pytest.approx({"r_aj_1": 0.2, "head_j_pan": -0.1, "r_hj_middle_3": 0.4})


def test_unknown_and_nonfinite_are_dropped(table):
    res = joint_map.map_joint_state(["bogus", "rj_dg_1_3", "rj_dg_1_4"], [1.0, float("nan"), 0.2], table)
    assert dict(res.values) == pytest.approx({"r_hj_thumb_4": 0.2})
    assert res.unknown == ("bogus",)
    assert res.non_finite == ("rj_dg_1_3",)


def test_length_mismatch_raises(table):
    with pytest.raises(ValueError):
        joint_map.map_joint_state(["rj_dg_1_1"], [], table)


def test_sign_is_applied():
    tbl = joint_map.build_table([{"canonical": "x_can", "source": "x_src", "sign": -1}])
    res = joint_map.map_joint_state(["x_src"], [0.5], tbl)
    assert dict(res.values) == {"x_can": -0.5}


def test_build_table_rejects_bad_sign_and_conflict():
    with pytest.raises(ValueError):
        joint_map.build_table([{"canonical": "a", "source": "s", "sign": 2}])
    with pytest.raises(ValueError):
        joint_map.build_table([{"canonical": "a", "source": "s"}, {"canonical": "b", "source": "s"}])
    with pytest.raises(ValueError):
        joint_map.build_table([])


def test_merge_returns_new_mapping_without_mutating():
    base = joint_map.merge({}, {"a": 1.0})
    new = joint_map.merge(base, {"b": 2.0, "a": 3.0})
    assert dict(base) == {"a": 1.0}
    assert dict(new) == {"a": 3.0, "b": 2.0}
    with pytest.raises(TypeError):
        new["c"] = 1.0  # type: ignore[index]


def test_profile_covers_every_movable_joint_of_the_task_asset(table):
    """right_aglt 자산(short-tl)의 가동 관절 중 팔·손 관절은 전부 프로필 canonical 에 있어야 한다."""
    if not ASSET_URDF.is_file():
        pytest.skip("hdgp 자산 URDF 없음")
    text = ASSET_URDF.read_text()
    movable = set(re.findall(r'<joint\s+name="([^"]+)"\s+type="(?:revolute|prismatic|continuous)"', text))
    arm_hand = {n for n in movable if re.match(r"^[rl]_[ah]j_", n)}
    assert len(arm_hand) == 2 * (7 + 19)        # thumb_1 은 -tl 자산에서 용접(fixed)
    missing = arm_hand - joint_map.canonical_names(table)
    assert missing == set()


def test_task_joint_names_match_profile(table):
    """env.yaml robot_cfg.init_state.joint_pos 의 이름(태스크 관절)도 전부 사상 가능한 canonical 이다."""
    if not POLICY_ENV_YAML.is_file():
        pytest.skip("정책 env.yaml 없음")
    text = POLICY_ENV_YAML.read_text()
    block = text.split("    joint_pos:\n", 1)[1].split("    joint_vel:", 1)[0]
    names = re.findall(r"^\s+([a-z_0-9]+):", block, flags=re.M)
    assert "r_aj_1" in names and "r_hj_thumb_1" not in names
    assert set(names) - joint_map.canonical_names(table) == set()


# ---------------------------------------------------------------- 패킷

def test_packet_roundtrip_and_format():
    data = packet.encode(123.5, {"r_aj_2": 0.25, "r_aj_1": -0.5})
    obj = json.loads(data)
    assert set(obj) == {"t", "names", "positions"}
    assert obj["names"] == ["r_aj_1", "r_aj_2"]          # 이름순(결정론)
    pkt = packet.decode(data)
    assert pkt.t == 123.5 and pkt.as_dict() == {"r_aj_1": -0.5, "r_aj_2": 0.25}


@pytest.mark.parametrize("raw", [
    b"not json",
    b"[1,2]",
    json.dumps({"t": 1.0, "names": ["a"]}).encode(),
    json.dumps({"t": 1.0, "names": ["a", "b"], "positions": [1.0]}).encode(),
    json.dumps({"t": 1.0, "names": ["a"], "positions": ["x"]}).encode(),
    json.dumps({"t": 1.0, "names": ["a", "a"], "positions": [1.0, 2.0]}).encode(),
    json.dumps({"t": "now", "names": ["a"], "positions": [1.0]}).encode(),
    json.dumps({"t": 1.0, "names": [""], "positions": [1.0]}).encode(),
    json.dumps({"t": 1.0, "names": ["a"], "positions": [True]}).encode(),
    b'{"t": 1.0, "names": ["a"], "positions": [NaN]}',
])
def test_decode_rejects_malformed(raw):
    with pytest.raises(ValueError):
        packet.decode(raw)


def test_encode_rejects_nonfinite_and_oversize():
    with pytest.raises(ValueError):
        packet.encode(1.0, {"a": math.inf})
    with pytest.raises(ValueError):
        packet.encode(1.0, {f"j{i}": 0.0 for i in range(packet.MAX_JOINTS + 1)})
    with pytest.raises(ValueError):
        packet.decode(b"x" * (packet.MAX_DATAGRAM_BYTES + 1))


def test_full_robot_packet_fits_one_datagram(table):
    joints = {n: 0.123456789 for n in joint_map.canonical_names(table)}
    assert len(packet.encode(1e9, joints)) < packet.MAX_DATAGRAM_BYTES


# ---------------------------------------------------------------- relay 가드

@pytest.mark.parametrize("env", [{}, {"ROS_DOMAIN_ID": ""}, {"ROS_DOMAIN_ID": "0"},
                                 {"ROS_DOMAIN_ID": "-3"}, {"ROS_DOMAIN_ID": "abc"}])
def test_relay_refuses_empty_or_zero_domain(env):
    with pytest.raises(SystemExit):
        require_ros_domain(env)


def test_relay_accepts_real_and_test_domain():
    assert require_ros_domain({"ROS_DOMAIN_ID": "126"}) == 126
    assert require_ros_domain({"ROS_DOMAIN_ID": " 97 "}) == 97


def test_relay_only_loopback_and_default_topics():
    with pytest.raises(SystemExit):
        parse_args(["--host", "192.168.0.10"])
    a = parse_args([])
    assert a.host == "127.0.0.1" and a.port == packet.DEFAULT_PORT
    assert {"/joint_states", "/dg5f_right/joint_states", "/dg5f_left/joint_states"} <= set(DEFAULT_TOPICS)


def test_relay_source_has_no_publisher_calls():
    """구독 전용 — relay 소스에 create_publisher 호출이 없어야 한다."""
    src = (Path(joint_map.__file__).parent / "joint_state_relay.py").read_text()
    assert "create_publisher(" not in src
    assert "create_client(" not in src and "ActionClient" not in src

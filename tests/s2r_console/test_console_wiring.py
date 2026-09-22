"""연결 그림 **자동 생성** — 미션의 명령(argv) + 계약 json + robot yaml → 상자와 전선.

정책을 바꾼다 = 미션이 다른 계약·다른 launch 를 가리킨다. 그림은 그것을 읽어 따라온다 — 프로파일에 손으로 적지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import s2r_console._paths  # noqa: F401
from s2r_console.units import UnitCmd
from s2r_console.wiring import CAMERA_TOPICS, generate, parse_cmd

SIM2REAL = Path(__file__).resolve().parents[2]

ROBOT_BI = {
    "robot": "bi",
    "sources": {
        "arm_right": {"type": "joint_state", "topic": "/joint_states"},
        "ee_right": {"type": "joint_state", "topic": "/dg5f_right/joint_states"},
        "tip_force_right": {"type": "float_array", "topic": "/dg5f_right/tip_forces_xyz"},
        "decoder_target_right": {"type": "joint_state", "topic": "/policy_control/joint_target", "required": False},
        "arm_left": {"type": "joint_state", "topic": "/joint_states"},
        "ee_left": {"type": "joint_state", "topic": "/dg5f_left/joint_states"},
        "tip_force_left": {"type": "float_array", "topic": "/dg5f_left/tip_forces_xyz"},
        "object": {"type": "pose", "topic": "/objects/cup_big_s100/pose"},
        "head": {"type": "joint_state", "topic": "/head/joint_states", "required": False, "stale_sec": 1.0},
    },
    "groups": {
        "right_arm": {"backend": "arm_forward", "side": "right"},
        "right_hand": {"backend": "dg5f_jtc", "topic": "/dg5f_right/dg5f_right_controller/joint_trajectory"},
        "left_arm": {"backend": "arm_forward", "side": "left"},
        "left_hand": {"backend": "dg5f_jtc", "topic": "/dg5f_left/dg5f_left_controller/joint_trajectory"},
    },
}
ROBOT_LEFT = {
    "robot": "gripper_left",
    "sources": {
        "arm": {"type": "joint_state", "topic": "/joint_states", "joints": ["l_aj_1"]},
        "ee": {"type": "joint_state", "topic": "/joint_states"},
        "object": {"type": "pose", "topic": "/objects/cup_big_s100/pose"},
        "head": {"type": "joint_state", "topic": "/head/joint_states", "required": False},
    },
    "groups": {
        "left_arm": {"backend": "arm_forward", "side": "left"},
        "left_gripper": {"backend": "jtc_single_point", "topic": "/left_gripper_controller/joint_trajectory"},
    },
}
POUR = {"schema": "policy_control/pour_contract/v1", "family": "pour_bimanual", "obs_dim": 223, "action_dim": 18,
        "sides": [{"role": "src", "side": "right"}, {"role": "rcv", "side": "left"}]}
PD_BI = {"schema": "policy_control/deploy_contract/v2", "control_only": True, "primary_side": "right",
         "sides": {"right": {"pd_groups": ["right_arm", "right_hand"]}, "left": {"pd_groups": ["left_arm", "left_hand"]}}}
GRASP_LEFT = {"schema": "policy_control/deploy_contract/v2", "control_only": False, "primary_side": "left",
              "policy": {"obs_dim": 79, "action_dim": 7},
              "sides": {"left": {"pd_groups": ["left_arm", "left_gripper"]}}}


@pytest.fixture()
def repo(tmp_path):
    for name, body in (("bi.yaml", ROBOT_BI), ("left.yaml", ROBOT_LEFT)):
        (tmp_path / name).write_text(yaml.safe_dump(body), encoding="utf-8")
    for name, body in (("pour.json", POUR), ("pd_bi.json", PD_BI), ("grasp_left.json", GRASP_LEFT)):
        (tmp_path / name).write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def unit(key, *argv, kind="background"):
    stage, index = key.split("#")
    return UnitCmd(key=key, stage=stage, index=int(index), note=key, argv=tuple(argv), kind=kind,
                   touches_real=False, needs=())


def pour_units(repo):
    r = str(repo)
    return {u.key: u for u in (
        unit("plant#0", "ros2", "launch", f"{r}/x/fake_plant.launch.py", "side:=both", f"robot:={r}/bi.yaml",
             f"contract:={r}/pd_bi.json"),
        unit("plant#1", "python3", f"{r}/x/fake_cup_pose_pub.py", "--topic", "/objects/pour_src_cup/pose", "--x", "0.3"),
        unit("plant#2", "python3", f"{r}/x/fake_cup_pose_pub.py", "--topic", "/objects/pour_rcv_cup/pose"),
        unit("pd_load#0", "ros2", "launch", f"{r}/x/pd_controller.launch.py", f"contract:={r}/pd_bi.json",
             f"robot:={r}/bi.yaml", "sides:=right,left", "execute:=true"),
        unit("chain#0", "ros2", "launch", f"{r}/x/pour_chain.launch.py", f"contract:={r}/pour.json", f"robot:={r}/bi.yaml",
             "src_cup_topic:=/objects/pour_src_cup/pose", "rcv_cup_topic:=/objects/pour_rcv_cup/pose", "use_fabric:=false"),
        unit("chain#1", "python3", f"{r}/x/pour_guard_node.py", "--ros-args", "-p", "src_cup_topic:=/objects/pour_src_cup/pose",
             "-p", "rcv_cup_topic:=/objects/pour_rcv_cup/pose"),
    )}


def grasp_units(repo, *, execute="false"):
    r = str(repo)
    return {u.key: u for u in (
        unit("bringup#1", "ros2", "launch", "openarm_bringup", "openarm.bimanual.launch.py", kind="manual"),
        unit("pd_load#0", "ros2", "launch", f"{r}/x/pd_controller.launch.py", f"contract:={r}/grasp_left.json",
             f"robot:={r}/left.yaml", f"execute:={execute}"),
        unit("observe_only#0", "ros2", "launch", "policy_control", "policy_chain.launch.py",
             f"contract:={r}/grasp_left.json", f"robot:={r}/left.yaml"),
    )}


def wires_of(d):
    return {(w.topic, w.src, w.dst) for w in d.wires}


# ── 명령 읽기 ───────────────────────────────────────────────────────────
def test_a_launch_command_gives_its_file_and_arguments():
    by_path = parse_cmd(("ros2", "launch", "/r/launch/pd_controller.launch.py", "contract:=/c.json", "execute:=true"))
    by_pkg = parse_cmd(("ros2", "launch", "policy_control", "policy_chain.launch.py", "side:=both"))
    assert (by_path.name, by_path.args) == ("pd_controller.launch.py", {"contract": "/c.json", "execute": "true"})
    assert (by_pkg.name, by_pkg.args) == ("policy_chain.launch.py", {"side": "both"})


def test_a_script_command_gives_its_flags_and_ros_parameters():
    cup = parse_cmd(("python3", "/r/fakes/fake_cup_pose_pub.py", "--topic", "/objects/a/pose", "--x", "0.3"))
    guard = parse_cmd(("python3", "/r/pour_guard_node.py", "--ros-args", "-p", "src_cup_topic:=/objects/a/pose"))
    assert (cup.name, cup.args["topic"]) == ("fake_cup_pose_pub.py", "/objects/a/pose")
    assert guard.args["src_cup_topic"] == "/objects/a/pose"


# ── pour 계열 ───────────────────────────────────────────────────────────
def test_the_pour_chain_is_wired_from_its_contract_roles_and_the_robot_yaml(repo):
    d = generate(pour_units(repo), repo=repo, status_nodes=("pour_node", "pd", "pour_guard"))
    got = wires_of(d)
    for topic, inputs in (("/joint_states", ("src:arm", "rcv:arm")), ("/dg5f_right/joint_states", ("src:hand",)),
                          ("/dg5f_left/tip_forces_xyz", ("rcv:force",)), ("/objects/pour_src_cup/pose", ("src:cup",))):
        wire = next(w for w in d.wires if w.topic == topic and w.dst == "pour_node")
        assert wire.inputs == inputs
    assert ("/objects/pour_rcv_cup/pose", "obj_pour_rcv_cup", "pour_guard") in got
    assert ("/policy_control/joint_target", "pour_node", "pd") in got
    assert ("/policy_control/episode", "pour_node", "pour_guard") in got
    fill = next(w for w in d.wires if w.topic == "/policy_control/pour/fill_level")
    assert fill.on_demand and fill.inputs == ("fill",) and fill.src == "operator"


def test_every_box_is_switched_by_the_mission_command_that_starts_it(repo):
    d = generate(pour_units(repo), repo=repo, status_nodes=("pour_node", "pd", "pour_guard"))
    unit_of = {b.id: b.unit for b in d.boxes}
    assert unit_of["pour_node"] == "chain#0" and unit_of["pour_guard"] == "chain#1" and unit_of["pd"] == "pd_load#0"
    assert unit_of["obj_pour_src_cup"] == "plant#1" and unit_of["arm_state"] == "plant#0"
    assert unit_of["operator"] is None


def test_pd_is_wired_per_side_to_state_and_to_every_drive_topic(repo):
    d = generate(pour_units(repo), repo=repo, status_nodes=("pour_node", "pd", "pour_guard"))
    state = next(w for w in d.wires if w.topic == "/joint_states" and w.dst == "pd")
    assert state.inputs == ("right:arm", "left:arm")
    got = wires_of(d)
    for side in ("right", "left"):
        for kind in ("position", "velocity", "effort"):
            assert (f"/{side}_forward_{kind}_controller/commands", "pd", "arm_drive") in got
        assert (f"/dg5f_{side}/dg5f_{side}_controller/joint_trajectory", "pd", f"hand_{side}_drive") in got
    metered = {w.topic for w in d.wires if w.dst == "arm_drive" and w.meter}
    assert metered == {"/right_forward_position_controller/commands", "/left_forward_position_controller/commands"}


def test_the_fabric_stage_inside_pour_node_is_shown_in_its_box(repo):
    d = generate(pour_units(repo), repo=repo, status_nodes=("pour_node", "pd", "pour_guard"))
    stages = d.box("pour_node").stages
    assert any("fabric" in s.lower() for s in stages) and any("223" in s for s in stages)


# ── 일반 체인 (obs → policy → fabric → pd) ─────────────────────────────
def test_the_generic_chain_gets_one_box_per_node(repo):
    d = generate(grasp_units(repo), repo=repo, status_nodes=("obs", "policy", "fabric", "pd"))
    ids = [b.id for b in d.boxes]
    for want in ("obs", "policy", "fabric", "pd"):
        assert want in ids and d.box(want).unit == ("pd_load#0" if want == "pd" else "observe_only#0")
    got = wires_of(d)
    assert {("/policy_control/obs", "obs", "policy"), ("/policy_control/obs", "obs", "fabric"),
            ("/policy_control/action", "policy", "fabric"), ("/policy_control/joint_target", "fabric", "pd")} <= got
    assert d.box("fabric").ros == ("/fabric_node",) and d.box("obs").status == "obs"


def test_perception_is_drawn_back_to_the_camera_when_nothing_in_the_mission_fakes_the_object(repo):
    d = generate(grasp_units(repo), repo=repo, status_nodes=("obs", "policy", "fabric", "pd"))
    got = wires_of(d)
    assert ("/perception_plus_plus/cup_big_s100/pose", "fpp_cup_big_s100", "object_pose") in got
    assert ("/objects/cup_big_s100/pose", "object_pose", "obs") in got
    assert ("/objects/cup_big_s100/pose", "object_pose", "fabric") in got
    cam = [w for w in d.wires if w.src == "camera"]
    assert {w.topic for w in cam} == set(CAMERA_TOPICS) and not any(w.meter for w in cam)     # 영상은 세지 않는다
    assert d.box("object_pose").ros == ("/object_pose_node",)


def test_a_faked_object_has_no_perception_chain(repo):
    d = generate(pour_units(repo), repo=repo, status_nodes=("pour_node", "pd", "pour_guard"))
    assert not [b for b in d.boxes if b.id in ("camera", "object_pose") or b.id.startswith("fpp_")]


def test_a_pd_that_does_not_publish_mutes_its_drive_wires_instead_of_breaking_them(repo):
    d = generate(grasp_units(repo, execute="false"), repo=repo, status_nodes=("obs", "policy", "fabric", "pd"))
    drive = [w for w in d.wires if w.src == "pd"]
    assert drive and all("execute" in w.muted for w in drive)
    live = generate(grasp_units(repo, execute="true"), repo=repo, status_nodes=("obs", "policy", "fabric", "pd"))
    assert not any(w.muted for w in live.wires)


def test_real_controllers_are_heard_on_their_own_nodes(repo):
    d = generate(grasp_units(repo, execute="true"), repo=repo, status_nodes=("obs", "policy", "fabric", "pd"))
    wire = next(w for w in d.wires if w.topic == "/left_forward_position_controller/commands")
    grip = next(w for w in d.wires if w.topic == "/left_gripper_controller/joint_trajectory")
    assert wire.heard_by == ("/left_forward_position_controller",) and grip.heard_by == ("/left_gripper_controller",)


def test_a_control_only_contract_runs_fabric_in_direct_mode(repo):
    r = str(repo)
    units = {"fab#0": unit("fab#0", "ros2", "launch", "policy_control", "policy_chain.launch.py",
                           f"contract:={r}/pd_bi.json", f"robot:={r}/bi.yaml", "side:=both")}
    d = generate(units, repo=repo, status_nodes=("fabric", "pd"))
    ids = {b.id for b in d.boxes}
    assert {"episode_master", "fabric_right", "fabric_left"} <= ids and "obs" not in ids and "policy" not in ids
    assert d.box("fabric_left").ros == ("/fabric_node_left",)
    cmd = next(w for w in d.wires if w.topic == "/policy_control/palm_cmd" and w.dst == "fabric_right")
    assert cmd.on_demand and cmd.src == "operator"


# ── 그림의 규칙 ─────────────────────────────────────────────────────────
def test_every_wire_runs_left_to_right_and_feedback_is_left_out(repo):
    for units, nodes in ((pour_units(repo), ("pour_node", "pd", "pour_guard")), (grasp_units(repo), ("obs", "policy", "fabric", "pd"))):
        d = generate(units, repo=repo, status_nodes=nodes)
        col = {b.id: b.col for b in d.boxes}
        assert all(col[w.src] < col[w.dst] for w in d.wires)
        assert not [w for w in d.wires if w.topic == "/policy_control/joint_target" and w.dst in ("obs", "pour_node")]
        assert sorted(set(col.values())) == list(range(len(set(col.values()))))          # 빈 열이 없다


def test_outputs_that_only_flow_during_an_episode_are_marked(repo):
    d = generate(pour_units(repo), repo=repo, status_nodes=("pour_node", "pd", "pour_guard"))
    target = next(w for w in d.wires if w.topic == "/policy_control/joint_target" and w.dst == "pd")
    state = next(w for w in d.wires if w.topic == "/joint_states" and w.dst == "pd")
    assert target.episodic and not state.episodic


def test_a_command_the_generator_does_not_know_is_still_a_box(repo):
    units = {**pour_units(repo), "extra#0": unit("extra#0", "python3", "/r/some_other_node.py")}
    d = generate(units, repo=repo, status_nodes=("pour_node", "pd", "pour_guard"))
    box = d.box("unit_extra_0")
    assert box.unit == "extra#0" and "some_other_node.py" in box.title


def test_a_status_name_the_profile_does_not_listen_to_is_not_claimed(repo):
    d = generate(pour_units(repo), repo=repo, status_nodes=("pd",))
    assert d.box("pour_node").status is None and d.box("pd").status == "pd"


def test_an_unreadable_contract_is_said_in_the_box_not_raised(repo):
    r = str(repo)
    units = {"chain#0": unit("chain#0", "ros2", "launch", f"{r}/x/pour_chain.launch.py", f"contract:={r}/nope.json",
                             f"robot:={r}/bi.yaml")}
    d = generate(units, repo=repo, status_nodes=("pour_node",))
    assert "nope.json" in d.box("unit_chain_0").note


# ── 출고 미션 ───────────────────────────────────────────────────────────
def test_camera_topics_are_the_ones_the_fpp_parameters_name():
    import object_registry as reg

    rendered = reg.render_fpp_yaml(next(iter(reg.load_registry().objects.values())))
    assert all(t in rendered for t in CAMERA_TOPICS)


def test_the_gripper_is_switched_with_the_arm_bringup(repo):
    d = generate(grasp_units(repo), repo=repo, status_nodes=("obs", "policy", "fabric", "pd"))
    assert d.box("hand_left_drive").unit == "bringup#1" and d.box("arm_drive").unit == "bringup#1"


def test_the_fake_plants_object_topic_is_the_one_its_launch_file_publishes():
    from s2r_console.wiring import FAKE_PLANT_OBJECT_TOPIC

    text = (SIM2REAL / "policy_control" / "launch" / "fake_plant.launch.py").read_text(encoding="utf-8")
    assert f'OBJECT_TOPIC = "{FAKE_PLANT_OBJECT_TOPIC}"' in text


def test_shipped_profiles_do_not_hand_write_the_picture():
    from s2r_console.profiles import scan

    good, _ = scan(SIM2REAL / "s2r_console" / "profiles", repo=SIM2REAL)
    assert [p.id for p in good if p.diagram is not None] == []           # 적어 두면 정책을 바꿔도 그림이 안 따라온다


def test_every_shipped_profile_generates_a_valid_diagram():
    from s2r_console.console import mission_units
    from s2r_console.profiles import scan

    good, bad = scan(SIM2REAL / "s2r_console" / "profiles", repo=SIM2REAL)
    assert not bad and good
    for p in good:
        d = generate(mission_units(p, repo=SIM2REAL), repo=SIM2REAL, status_nodes=p.status_nodes)
        assert {b.status for b in d.boxes if b.status} == set(p.status_nodes), p.id
        assert d.wires and all(b.unit is None or b.unit in d.units() for b in d.boxes)


# ── 한 미션이 같은 노드를 두 번 띄울 때 ────────────────────────────────
def test_two_pd_launches_get_two_boxes_each_with_its_own_switch(repo):
    # mission_dg5f_m_control 은 pd 를 좌·우 따로 띄운다. 하나로 합치면 나머지 하나는 스위치도 보호도 없다.
    r = str(repo)
    units = {u.key: u for u in (
        unit("pd_load_right#0", "ros2", "launch", f"{r}/x/pd_controller.launch.py", f"contract:={r}/pd_bi.json",
             f"robot:={r}/bi.yaml", "sides:=right", "execute:=false"),
        unit("pd_load_left#0", "ros2", "launch", f"{r}/x/pd_controller.launch.py", f"contract:={r}/pd_bi.json",
             f"robot:={r}/bi.yaml", "sides:=left", "execute:=false"),
    )}
    units["chain#0"] = unit("chain#0", "ros2", "launch", f"{r}/x/policy_chain.launch.py",
                            f"contract:={r}/pd_bi.json", f"robot:={r}/bi.yaml").__class__(
        **{**unit("chain#0", "x").__dict__, "argv": ("ros2", "launch", f"{r}/x/policy_chain.launch.py",
                                                     f"contract:={r}/pd_bi.json", f"robot:={r}/bi.yaml")})
    d = generate(units, repo=repo, status_nodes=("pd", "episode_master"))
    pd = [b for b in d.boxes if b.id.startswith("pd")]
    assert {b.unit for b in pd} == {"pd_load_right#0", "pd_load_left#0"}, [(b.id, b.unit) for b in pd]
    # 두 pd 모두 에피소드를 받는다 — 한쪽만 이으면 나머지는 그림에서 죽은 가지로 보인다
    got = {w.dst for w in d.wires if w.topic.endswith("/episode")}
    assert {b.id for b in pd} <= got, got


def test_a_handler_that_raises_anything_becomes_a_box_not_a_dead_console(repo):
    # 계약의 sides 가 목록이면 TypeError 다 — generate 가 던지면 Session.__init__ 이 죽어 화면이 아예 안 뜬다.
    (repo / "broken.json").write_text(json.dumps(
        {"schema": "policy_control/deploy_contract/v2", "control_only": True, "sides": ["right"]}), encoding="utf-8")
    r = str(repo)
    units = {u.key: u for u in (unit("pd_load#0", "ros2", "launch", f"{r}/x/pd_controller.launch.py",
                                     f"contract:={r}/broken.json", f"robot:={r}/bi.yaml"),)}
    d = generate(units, repo=repo, status_nodes=("pd",))
    assert [b.unit for b in d.boxes] == ["pd_load#0"] and "읽지 못했다" in d.boxes[0].note


def test_an_unknown_manual_command_is_kept_as_a_box(repo):
    units = {u.key: u for u in (unit("misc#0", "python3", "/r/some_tool.py", kind="manual"),)}
    d = generate(units, repo=repo, status_nodes=())
    assert [b.unit for b in d.boxes] == ["misc#0"]


def test_the_fake_hand_jtc_wire_says_it_is_meant_to_be_silent(repo):
    d = generate(pour_units(repo), repo=repo, status_nodes=("pd", "pour_node"))
    jtc = [w for w in d.wires if "joint_trajectory" in w.topic]
    assert jtc and all(w.muted for w in jtc), [(w.topic, w.muted) for w in jtc]


def test_a_diagram_that_cannot_be_built_leaves_the_console_openable(repo, tmp_path):
    # 그림은 편의다. 그림이 안 만들어진다고 세션이 안 열리면 운영자는 미션 자체를 못 본다.
    from s2r_console.console import diagram_of
    from s2r_console.profiles import Profile
    prof = Profile(id="p", title="p", path=tmp_path / "p.yaml", mission=tmp_path / "m.yaml", domain=97,
                   domain_class="fake", policy_dir=None, policy_dt=0.02, status_nodes=("pd",))
    assert diagram_of(prof, {}, repo=repo) is None


def test_each_fake_hand_box_is_judged_by_its_own_side_node(repo):
    # fake 손 상태 노드는 좌·우가 같은 이름(/fake_hand_state_pub)이다 — 한쪽이 죽어도 이름이 남아 둘 다 초록이었다.
    d = generate(pour_units(repo), repo=repo, status_nodes=("pd", "pour_node"))
    hands = {b.id: b.ros for b in d.boxes if b.id.startswith("hand_") and b.id.endswith("_drive")}
    assert len(hands) == 2 and len({tuple(v) for v in hands.values()}) == 2, hands
    assert all(v and v[0].startswith("/dg5f_") for v in hands.values()), hands


def test_a_real_hand_drive_box_knows_which_controller_manager_it_lives_under(repo):
    d = generate(grasp_units(repo), repo=repo, status_nodes=("pd",))
    grip = next(b for b in d.boxes if b.id.startswith("hand_"))
    assert grip.manager == "/controller_manager"            # 그리퍼 JTC 는 팔 bringup 아래에 있다


def test_an_unmetered_wire_does_not_declare_a_limit_nobody_checks(repo):
    d = generate(grasp_units(repo), repo=repo, status_nodes=("pd",))
    assert [(w.topic, w.stale_ms) for w in d.wires if not w.meter and w.stale_ms is not None] == []


# ── 인지(FP++)는 vision-3090 에서 돈다 ─────────────────────────────────
def test_the_perception_launcher_becomes_a_box_with_its_remote_host(repo):
    # 카메라·FP++ 컨테이너는 vision-3090 에서 뜬다. 그것을 켜는 것은 로컬 런처이고, 그 런처가 유일한 스위치다.
    units = dict(pour_units(repo))
    units["percept#0"] = unit("percept#0", "python3", str(repo / "scripts/nodes/perception_launcher_node.py"),
                              "--host", "vision-3090")
    d = generate(units, repo=repo, status_nodes=("pd", "pour_node"))
    box = next(b for b in d.boxes if b.id == "perception")
    assert box.unit == "percept#0" and box.ros == ("/perception_launcher",)
    assert box.host == "vision-3090" and box.col == 0          # 카메라보다 왼쪽 — 이것이 그 PC 를 켠다


def test_camera_and_tracker_say_which_machine_they_run_on(repo):
    units = dict(pour_units(repo))
    units["percept#0"] = unit("percept#0", "python3", str(repo / "scripts/nodes/perception_launcher_node.py"))
    # 컵을 fake 로 내지 않는 미션이어야 인지 사슬이 그려진다
    units.pop("plant#1")
    units.pop("plant#2")
    d = generate(units, repo=repo, status_nodes=("pd", "pour_node"))
    ids = {b.id: b for b in d.boxes}
    assert ids["camera"].host == "vision-3090"
    assert [b.host for b in d.boxes if b.id.startswith("fpp_")] == ["vision-3090"] * len([b for b in d.boxes if b.id.startswith("fpp_")])
    assert ids["object_pose"].host == ""                        # 이쪽 PC 다


def test_an_unknown_command_is_named_by_what_the_mission_calls_it(repo):
    # "bash" 라고만 적힌 상자는 운영자에게 아무 말도 하지 않는다 — 미션이 붙인 설명이 이름이다.
    u = unit("bringup#0", "bash", "-lc", "sudo ip link set can1 up", kind="manual")
    u = u.__class__(**{**u.__dict__, "note": "★CAN 설정 — sudo 필요, 사용자 셸에서"})
    d = generate({u.key: u}, repo=repo, status_nodes=())
    assert "CAN 설정" in d.boxes[0].note, d.boxes[0].note     # 미션이 붙인 설명이 상자 안에 남는다
    assert d.boxes[0].col == max(b.col for b in d.boxes)     # 체인 사이에 끼지 않는다

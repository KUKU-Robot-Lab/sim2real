"""연결 그림 — 상자 하나 = 노드 하나 (10.05 사용자: "연결창에 노드 상자가 너무 많고 중복된 것도 있다").

RH56F1 실기 미션의 그림이 상자 31개였다: 그림이 모르는 RH56F1 명령 20개가 전선 없는 상자로 오른쪽에 쌓였고,
같은 노드를 띄우는 단계마다 상자가 따로 생겼다(pd 무발행 · 발행, 단독 aglt · 에피소드 aglt …). 같은 이름의 노드는
동시에 둘일 수 없으므로 상자는 하나, 띄우는 명령마다 스위치가 하나다. 한 명령만 쓰는 전선은 그 명령을 적는다.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import s2r_console._paths  # noqa: F401
from s2r_console.console import Console, drawable_units, mission_units
from s2r_console.console_state import Observed
from s2r_console.diagram import build
from s2r_console.diagram_spec import parse_diagram
from s2r_console.profiles import load
from s2r_console.units import UnitCmd
from s2r_console import wiring as W

SIM2REAL = Path(__file__).resolve().parents[2]
_ACTIVE = yaml.safe_load((SIM2REAL / "config" / "fpp_active.yaml").read_text(encoding="utf-8"))   # 10.09 실기 물체 = 이 파일


def _active_groups() -> list[str]:
    """팔별 물체의 묶음들(팔마다 다를 수 있다 — 10.09 오른손 병 · 왼손 쉐이커)."""
    import sys as _s
    _s.path.insert(0, str(SIM2REAL / "scripts"))
    from object_registry import load_registry
    reg = load_registry()
    return list(dict.fromkeys(reg.get(n).fpp["group"] for n in _ACTIVE["sides"].values()))
PROFILES = SIM2REAL / "deploy" / "s2r_console" / "profiles"


def _station():
    """★10.13 로봇 PC 별 인지 호스트 — 미션 생성기의 표를 그대로 읽는다."""
    import importlib.util
    sys.path.insert(0, str(SIM2REAL / "scripts"))
    spec = importlib.util.spec_from_file_location("make_rh56f1_missions", SIM2REAL / "scripts" / "ops" / "make_rh56f1_missions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _station()


def _real():
    import mission_core as MC
    prof = load(PROFILES / "rh56f1_real.yaml", repo=SIM2REAL)
    mission = MC.load_mission(yaml.safe_load(prof.mission.read_text(encoding="utf-8")))
    units = drawable_units(mission, mission_units(prof, repo=SIM2REAL))
    return prof, units, W.generate(units, repo=SIM2REAL, status_nodes=prof.status_nodes)


@pytest.fixture(scope="module")
def real():
    return _real()


# ── 실기 미션 그림 ──────────────────────────────────────────────────────
def test_the_rh56f1_real_mission_draws_every_node_once_and_knows_every_command(real):
    _, _, d = real
    assert not [b.id for b in d.boxes if b.id.startswith("unit_")]           # 모르는 명령 상자가 없다
    ros = Counter(n for b in d.boxes for n in b.ros)
    assert not {n: c for n, c in ros.items() if c > 1}                        # 같은 ROS 노드를 두 상자가 주장하지 않는다
    assert len({b.title for b in d.boxes}) == len(d.boxes)
    assert len(d.boxes) <= 22, [b.id for b in d.boxes]                       # ★10.19 팔마다 실행기 상자 둘(좌우 동시 실행)


def test_no_wire_is_drawn_twice(real):
    _, _, d = real
    ends = Counter((w.src, w.dst, w.topic) for w in d.wires)
    assert not {k: v for k, v in ends.items() if v > 1}


def test_a_blocked_stage_is_left_out_of_the_picture_but_not_out_of_the_units(real):
    prof, units, d = real
    assert not any(k.startswith("policy_pourfj#") for k in units)
    assert "policy_pourfj#1" in mission_units(prof, repo=SIM2REAL)
    assert not [b for b in d.boxes if "/pour_fj_node" in b.ros]
    assert not [b for b in d.boxes if b.id.startswith("fpp_cup_")]            # 막힌 단계의 컵 두 개는 인지 사슬도 없다


def test_a_node_started_by_its_own_stage_and_by_the_episode_is_one_box_with_both_switches(real):
    _, units, d = real
    aglt = d.box("rh_aglt_node_right")
    assert aglt.ros == ("/rh_aglt_node_right", "/rh_aglt_node_right_poll")    # 센서는 폴링 노드가 구독한다
    # ★10.20 영상 기록 기본 끔(VISION_RECORD) — 정책 단계의 aglt 노드가 #4 → #3
    assert set(aglt.units) == {"policy_aglt_right#3", "episode_pick_place_right#3",
                               "episode_grasp_c_right#3",                       # 10.17 경로 C(접근 정책 + 스크립트 파지)
                               "episode_grasp_d_right#3",                       # ★10.20 경로 D(1-B 접근 → rcg03)
                               "episode_grasp_e_right#3",                       # ★10.20 밤 경로 E(ap05 → rcg03 넘김)
                               "episode_grasp_f_right#3",                       # ★10.21 경로 F(ap05 → rcg04 넘김)
                               "episode_tilt_base_d#3",                         # ★10.20 경로 A 폐지 → tilt 경로 D(ap05 → rcg03)
                               "episode_tilt_base_c#3"} & set(units)            # 10.17 tilt 기본 궤적 경로 C
    place = d.box("rh_place_node_right")
    assert set(place.units) == {"policy_place_right#1", "episode_pick_place_right#4"}
    # ★10.19 사용자 "좌우 동시 실행": 한 팔 에피소드는 팔마다 실행기(episode_runner_<팔>), 두 팔 tilt 는 기본 실행기
    runner = d.box("episode_runner")
    assert set(runner.units) == {"episode_tilt_base_d#7",   # 실기: 앞에 CPU 기록 한 줄(10.08) · ★10.20 경로 A 폐지 → D, 정책 노드 둘
                                 "episode_tilt_base_c#7",   # 10.17 경로 C — 정책 노드 둘(D 와 같은 자리)
                                 "episode_tilt_base_b#5"}   # 10.17 경로 B — 정책 노드 없음(사람이 컵을 댄다)
    for side in ("right", "left"):
        assert set(d.box(f"episode_runner_{side}").units) == {f"episode_pick_place_{side}#7",
                                                              f"episode_grasp_c_{side}#6",
                                                              # ★10.19 1-B — 정책 노드 없음, 기록 bag · CPU 뒤(★10.20 영상 기본 끔 → #5)
                                                              f"episode_grasp_b_{side}#5",
                                                              f"episode_grasp_d_{side}#6",   # ★10.20 경로 D(aglt 노드 하나)
                                                              f"episode_grasp_e_{side}#6",   # ★10.20 밤 경로 E(aglt 노드 하나, 넘김)
                                                              f"episode_grasp_f_{side}#6"}   # ★10.21 경로 F(경로 E + rcg04)


def test_one_arms_pd_without_and_with_publishing_is_one_box_and_only_the_quiet_one_mutes_the_drive(real):
    _, _, d = real
    (pd,) = [b for b in d.boxes if b.ros == ("/pd_node_right", "/pd_node_poll_right")]
    assert pd.units == ("pd_load_right#0", "pd_arm_right#1") and pd.stages[-1] == "무발행 → 발행"
    drive = [w for w in d.wires if w.src == pd.id and w.dst in ("arm_drive", "hand_right_drive")]
    assert drive and all(w.muted and w.muted_by == ("pd_load_right#0",) for w in drive)


def test_an_input_only_one_stage_uses_names_that_stage(real):
    _, _, d = real
    into = {(w.src, w.topic): w for w in d.wires if w.dst == "rh_aglt_node_right"}
    # 10.12 사용자(A안): 단독 · 에피소드 모두 쥘 때까지 FP++ live 컵을 직접 받는다 — snapshot 재발행(/episode/objects) 은 안 받는다
    assert not into[("object_pose", f"/objects/{_ACTIVE['sides']['right']}/live_pose")].units                  # 세 단계 모두
    assert not [k for k in into if k[1].startswith("/episode/objects/")]
    assert not into[("hand_right_state", "/hand_right/joint_states")].units                       # 둘 다 쓴다 — 언제나


def test_in_the_episode_the_runner_gives_the_holder_and_goal_and_alone_the_holder_node_does(real):
    _, _, d = real
    holder = {w.src: w.units for w in d.wires if w.dst == "rh_place_node_right" and w.topic == "/objects/cup_holder_1/pose"}
    # ★10.13 원격 인지면 마커 노드가 없다 — 단독 놓기의 홀더는 미션 밖(바깥 상자)에서 온다
    alone = "obj_cup_holder_1" if GEN.REMOTE_PERCEPTION else "cup_holders"
    assert holder == {alone: ("policy_place_right#1",), "episode_runner_right": ("episode_pick_place_right#4",)}
    goal = {w.src: w for w in d.wires if w.dst == "rh_aglt_node_right" and w.topic == "/policy_control/right/goal"}
    # ★10.19 한 팔 에피소드는 오른팔 실행기, 두 팔 tilt 는 기본 실행기가 목표를 낸다
    # ★10.20 옛 정책 정리 뒤: 계약이 없는 정책 노드는 단위에서 빠진다 → 목표를 내는 실행기는 오른손 aglt 노드를 실제로 띄우는 단계의 것만
    assert set(goal) == set(_goal_runners(real[1])) and all(w.on_demand for w in goal.values())


def _goal_runners(units) -> dict[str, str]:
    """오른손 aglt 노드를 띄우는 에피소드 단계의 실행기 노드 → 그림 이름표(우 실행기 · 실행기)."""
    def node(u):
        return next((a.split(":=", 1)[1].lstrip("/") for a in u.argv if a.startswith("__node:=")), "")
    aglt = {u.stage for u in units.values() if node(u) == "rh_aglt_node_right"}
    out = {}
    for u in units.values():
        if u.stage in aglt and any(str(a).endswith("episode_runner_node.py") for a in u.argv):
            n = node(u) or "episode_runner"
            out[n] = {"episode_runner_right": "우 실행기", "episode_runner_left": "좌 실행기"}.get(n, "실행기")
    return out


def test_an_episode_stage_that_launches_the_aglt_node_gets_its_goal_from_that_runner(real, tmp_path):
    """★10.20 실기 미션의 aglt 에피소드가 모두 옛 정책이라 빠졌다 — 같은 규칙을 지금 등록된 aglt 노드 명령으로 따로 본다."""
    from dataclasses import replace
    units = real[1]
    node = next(u for u in units.values() if "__node:=rh_aglt_node_right" in u.argv)
    ep = tmp_path / "ep.yaml"
    ep.write_text(yaml.safe_dump({"policies": {"rh_aglt_r": {"policy": "x", "kind": "aglt", "side": "right"}},
                                  "sequence": [{"id": "n", "type": "policy"}]}), encoding="utf-8")
    runner = UnitCmd(key="ep_x#6", stage="ep_x", index=6, note="", needs=(), kind="background", touches_real=False,
                     argv=("python3", "/x/episode_runner_node.py", "--ros-args", "-r", "__node:=episode_runner_right",
                           "-p", f"episode:={ep}"))
    d = W.generate({"ep_x#3": replace(node, key="ep_x#3", stage="ep_x"), "ep_x#6": runner}, repo=SIM2REAL, status_nodes=())
    goal = [w for w in d.wires if w.dst == "rh_aglt_node_right" and w.topic == "/policy_control/right/goal"]
    assert [(w.src, w.on_demand) for w in goal] == [("episode_runner_right", True)]   # 내는 쪽이 하나면 이름표는 비운다


def test_policy_nodes_feed_only_their_own_arms_pd(real):
    _, _, d = real
    pd_of = {b.ros[0]: b.id for b in d.boxes if b.ros and b.ros[0].startswith("/pd_node_")}
    target = {(w.src, w.dst) for w in d.wires if w.topic == "/policy_control/joint_target"}
    assert ("rh_aglt_node_right", pd_of["/pd_node_right"]) in target
    assert ("rh_aglt_node_right", pd_of["/pd_node_left"]) not in target
    assert ("rh_place_node_left", pd_of["/pd_node_left"]) in target
    episode = {w.topic for w in d.wires if w.src == "rh_place_node_right" and w.dst == pd_of["/pd_node_right"]}
    assert "/policy_control/right/episode" in episode                         # 에피소드 놓기도 그 팔 토픽으로 낸다


def test_the_rh56f1_hand_is_a_driver_box_and_a_state_box_each_with_its_own_switch(real):
    _, _, d = real
    drive, state = d.box("hand_right_drive"), d.box("hand_right_state")
    assert drive.units == ("hand_right#1",) and drive.ros == ("/rh56f1_ecat_right",) and "JTC" not in drive.title
    assert state.units == ("hand_right#2",) and state.ros == ("/rh56f1_state_right",)
    (cmd,) = [w for w in d.wires if w.dst == "hand_right_drive"]
    assert cmd.topic == "/hand_right/angle_set" and (cmd.heard_by or drive.ros) == ("/rh56f1_ecat_right",)
    assert {w.topic for w in d.wires if w.src == "hand_right_state"} >= {"/hand_right/joint_states", "/hand_right/tip_forces",
                                                                         "/hand_right/joint_forces"}


def test_the_perception_chain_runs_on_the_perception_pc_and_the_udp_receiver_switches_the_fpp_box(real):
    _, _, d = real
    host = GEN.PERCEPTION_HOST                                                 # 10.01 arm4090 = local · 10.13 5090 = vision-3090
    assert d.box("perception").host == host and d.box("camera").host == host
    groups, sides = _active_groups(), _ACTIVE["sides"]
    boxes = [d.box(f"fpp_{g}") for g in groups]  # 10.08 색 다른 같은 모양 = 묶음 컨테이너 하나 = 상자 하나(묶음마다)
    assert all(b.units == ("cups#2",) and b.host == host for b in boxes)
    assert sum(b.ros == ("/fpp_pose_rx",) for b in boxes) == 1              # 수신 노드 하나 = 상자 하나
    assert not [b.id for b in d.boxes if b.id in {f"fpp_{n}" for n in sides.values()}]
    # 그림은 받는 쪽이 있는 물체만 잇는다 — 정책이 읽는 팔별 물체
    assert {w.topic for w in d.wires if w.src in {f"fpp_{g}" for g in groups} and w.dst == "object_pose"} == {
        f"/perception_plus_plus/{n}/pose" for n in sides.values()}
    assert d.box("object_pose").units == ("cups#3",)
    holder_feed = {w.topic for w in d.wires if w.src == "camera" and w.dst == "cup_holders"}
    assert holder_feed == (set() if GEN.REMOTE_PERCEPTION else {                # 10.13 원격 인지면 마커 노드가 없다
        "/camera/camera/color/image_raw", "/camera/camera/color/camera_info"})


def test_the_one_shot_goal_tool_is_not_drawn_as_a_node(real):
    # 10.05 사용자: 연결창은 실기 노드와 연결 상태 — 사람이 한 번 내는 도구(aglt_goal.py)는 노드가 아니다(조작판에는 있다)
    _, units, d = real
    assert {"policy_place_right#0", "policy_place_left#0"} <= set(units)
    assert not [b for b in d.boxes if {"policy_place_right#0", "policy_place_left#0"} & set(b.units)]


def test_the_fake_profile_gets_the_same_picture_with_fake_hands():
    prof = load(PROFILES / "rh56f1_fake.yaml", repo=SIM2REAL)
    d = W.generate(mission_units(prof, repo=SIM2REAL), repo=SIM2REAL, status_nodes=prof.status_nodes)
    assert not [b.id for b in d.boxes if b.id.startswith("unit_")]
    assert d.box("hand_right_drive").ros == ("/fake_rh56f1_right",)
    assert d.box("hand_right_state").ros == ("/rh56f1_state_right",)        # hands:=none — fake 플랜트의 손 상자가 없다
    holders = d.box("cup_holders")                                          # fake 홀더 넷 = 실기 홀더 노드 상자 하나
    assert len(holders.units) == 4 and holders.title == W.HOLDER_TITLE and not holders.ros


# ── 노드 소스의 이름 ────────────────────────────────────────────────────
def _src(rel: str) -> str:
    return (SIM2REAL / rel).read_text(encoding="utf-8")


def test_node_names_and_topics_are_the_ones_the_node_sources_use():
    assert 'make_lean_node(f"rh56f1_ecat_{args.side}")' in _src("deploy/policy_control/policy_control/rh56f1_ecat_node.py")
    assert W.HAND_DRIVER_NODE["rh56f1_driver.py"] == "/rh56f1_ecat_{side}"
    assert 'Node(f"fake_rh56f1_{args.side}")' in _src("scripts/fakes/fake_rh56f1_hand.py")
    assert W.HAND_DRIVER_NODE["fake_rh56f1_hand.py"] == "/fake_rh56f1_{side}"
    state = _src("deploy/policy_control/policy_control/rh56f1_state_node.py")
    assert 'make_lean_node(f"rh56f1_state_{args.side}")' in state and W.HAND_STATE_NODE == "/rh56f1_state_{side}"
    assert 'ns = f"/hand_{args.side}"' in state
    for topic in W.HAND_STATE_TOPICS:
        assert f'f"{{ns}}/{topic.rsplit("/", 1)[1]}"' in state, topic
    assert 'super().__init__("fpp_pose_rx")' in _src("scripts/nodes/fpp_pose_rx.py") and W.FPP_RX_NODE == "/fpp_pose_rx"
    assert 'super().__init__("object_pose_node")' in _src("scripts/nodes/object_pose_node.py")
    assert 'super().__init__("cup_holder_pose_node")' in _src("scripts/nodes/cup_holder_pose_node.py")
    assert f'SIM2REAL / "{W.HOLDER_CFG}"' in _src("scripts/calib/cup_holder_pose.py")
    assert 'NAME = "episode_runner"' in _src("deploy/policy_control/policy_control/episode_runner_node.py")
    assert 'make_poll_node(f"{NODE_NAME}_poll_{\'_\'.join(self.sides)}"' in _src("deploy/policy_control/policy_control/pd_node.py")
    assert 'NODE_NAME = "pd_node"' in _src("deploy/policy_control/policy_control/pd_node.py")
    assert W.PD_POLL_NODE.format(sides="right") == "/pd_node_poll_right"
    assert 'make_poll_node(f"{self.node_name}_poll"' in _src("deploy/policy_control/policy_control/pour_fj_node.py")
    assert W.POLICY_POLL_NODE.format(node="rh_aglt_node_right") == "/rh_aglt_node_right_poll"
    ros = _src("deploy/policy_control/policy_control/episode_ros.py")
    assert f'OBJECT_RELAY = "{W.OBJECT_RELAY}"' in ros and 'f"/objects/cup_holder_{hid}/pose"' in ros
    assert 'f"/objects/cup_holder_{self.holder_id}/pose"' in _src("deploy/policy_control/policy_control/rh_place_node.py")


def test_policy_node_cup_defaults_are_the_nodes_families():
    N = pytest.importorskip("policy_control.pour_fj_node")
    for script, (node, _, cups) in W.POLICY_NODES.items():
        fam = N.FAMILIES[node.removesuffix("_node")]
        assert fam.name == node, script
        assert [(param, topic) for _, param, topic in fam.cups] == list(cups), script


# ── 그림의 상태: 여러 명령이 띄우는 상자 ─────────────────────────────────
SPEC = parse_diagram({
    "boxes": [
        {"id": "pose", "title": "물체 자세", "col": 0, "ros": ["/object_pose_node"], "unit": "cups#3"},
        {"id": "runner", "title": "실행기", "col": 1, "ros": ["/episode_runner"], "units": ["ep#6"]},
        {"id": "aglt", "title": "aglt", "col": 2, "ros": ["/rh_aglt_node_right"], "units": ["aglt#1", "ep#3"]},
        {"id": "place", "title": "place", "col": 2, "ros": ["/rh_place_node_right"], "units": ["place#1"]},
        {"id": "pd", "title": "pd", "col": 3, "ros": ["/pd_node_right"], "units": ["pd_load#0", "pd_arm#1"]},
        {"id": "drive", "title": "구동", "col": 4, "ros": ["/rh56f1_ecat_right"], "unit": "hand#1"},
    ],
    "wires": [
        {"from": "pose", "to": "aglt", "topic": "/objects/cyl60/pose", "meter": False, "units": ["aglt#1"]},
        {"from": "runner", "to": "aglt", "topic": "/episode/objects/CUP/pose", "meter": False, "units": ["ep#3"]},
        {"from": "aglt", "to": "pd", "topic": "/policy_control/joint_target", "stale_ms": 500, "episodic": True},
        {"from": "place", "to": "pd", "topic": "/policy_control/joint_target", "stale_ms": 500, "episodic": True},
        {"from": "pd", "to": "drive", "topic": "/hand_right/angle_set", "meter": False, "muted": "무발행 pd",
         "muted_by": ["pd_load#0"]},
    ],
}, path=Path("p.yaml"))


def _topic(pubs, subs, n: float | None = 60, age: float | None = 5.0) -> dict:
    return {"pubs": len(pubs), "n": n, "age_ms": age, "pub_nodes": list(pubs), "sub_nodes": list(subs)}


def _obs(graph, topics):
    return Observed(bridge_up=True, status={}, age_s={}, graph=tuple(graph), topics=topics, topics_age_s=0.5)


def _units(alive=(), started=(), stopped=(), ages=None):
    ages = ages or {}
    keys = {u for b in SPEC.boxes for u in b.units}
    return {k: {"key": k, "kind": "background", "alive": k in alive, "started": k in alive or k in started,
                "stopped": k in stopped, "rc": None if k in alive else -15, "age_s": ages.get(k, 1.0), "pid": 1}
            for k in keys}


def _ports(out):
    return {(p["from"], p["topic"]): p for b in out["cols"] for box in b for p in box["ports"]}


EPISODE_GRAPH = ("/object_pose_node", "/episode_runner", "/rh_aglt_node_right", "/pd_node_right", "/rh56f1_ecat_right")
EPISODE_TOPICS = {
    "/objects/cyl60/pose": _topic(("/object_pose_node",), ("/episode_runner",)),
    "/episode/objects/CUP/pose": _topic(("/episode_runner",), ("/rh_aglt_node_right",)),
    "/policy_control/joint_target": _topic(("/rh_aglt_node_right",), ("/pd_node_right",)),
    "/hand_right/angle_set": _topic(("/pd_node_right",), ("/rh56f1_ecat_right",), n=None, age=None),
}


def test_while_the_episode_runs_the_node_the_single_stage_input_rests_and_is_not_a_break():
    out = build(_obs(EPISODE_GRAPH, EPISODE_TOPICS), SPEC, units=_units(alive=("cups#3", "ep#6", "ep#3", "pd_arm#1", "hand#1")))
    ports = _ports(out)
    alone = ports[("pose", "/objects/cyl60/pose")]
    assert alone["state"] == "off" and alone["expected_off"] and "aglt" in alone["note"]
    assert ports[("runner", "/episode/objects/CUP/pose")]["state"] == "live"
    assert "끊긴 곳" not in out["summary"]["text"]


def test_a_node_nobody_started_does_not_borrow_another_nodes_joint_target():
    # 에피소드 aglt 가 joint_target 을 내는데 place 는 꺼져 있다 — place → pd 전선이 초록이면 안 된다
    out = build(_obs(EPISODE_GRAPH, EPISODE_TOPICS), SPEC, units=_units(alive=("cups#3", "ep#6", "ep#3", "pd_arm#1", "hand#1")))
    place = _ports(out)[("place", "/policy_control/joint_target")]
    assert place["state"] == "off" and place["note"] == "내는 쪽이 꺼져 있다"
    assert _ports(out)[("aglt", "/policy_control/joint_target")]["state"] == "live"


def test_the_quiet_pd_declaration_holds_only_while_the_quiet_pd_runs():
    quiet = build(_obs(EPISODE_GRAPH, {**EPISODE_TOPICS, "/hand_right/angle_set": _topic((), ("/rh56f1_ecat_right",))}),
                  SPEC, units=_units(alive=("pd_load#0", "hand#1")))
    p = _ports(quiet)[("pd", "/hand_right/angle_set")]
    assert p["state"] == "off" and p["note"] == "무발행 pd" and p["expected_off"]
    loud = build(_obs(EPISODE_GRAPH, EPISODE_TOPICS), SPEC, units=_units(alive=("pd_arm#1", "hand#1")))
    p = _ports(loud)[("pd", "/hand_right/angle_set")]
    assert p["state"] == "live" and "무발행" not in p["note"] and not p["expected_off"]


def test_a_box_with_several_commands_gives_a_switch_each_and_the_latest_reason():
    obs = _obs((), {})
    out = build(obs, SPEC, units=_units(started=("pd_load#0", "pd_arm#1"), stopped=("pd_load#0",),
                                         ages={"pd_load#0": 60.0, "pd_arm#1": 5.0}))
    pd = next(b for col in out["cols"] for b in col if b["id"] == "pd")
    assert [u["key"] for u in pd["units"]] == ["pd_load#0", "pd_arm#1"] and pd["unit"]["key"] == "pd_load#0"
    assert pd["state"] == "down" and "rc=-15" in pd["detail"]                # 최근(발행 pd)이 스스로 죽었다
    out = build(obs, SPEC, units=_units(started=("pd_load#0", "pd_arm#1"), stopped=("pd_load#0", "pd_arm#1")))
    pd = next(b for col in out["cols"] for b in col if b["id"] == "pd")
    assert pd["state"] == "off" and pd["detail"] == "꺼져 있다"


# ── 콘솔의 보호 규칙은 상자의 명령 전부를 본다 ────────────────────────────
def test_both_pd_commands_of_one_arm_are_robot_units_and_both_bringup_commands_are_stack_units():
    d = parse_diagram({"boxes": [
        {"id": "pd", "title": "pd", "col": 0, "status": "pd_right", "ros": ["/pd_node_right"], "units": ["pd_load_right#0", "pd_arm_right#1"]},
        {"id": "arm_drive", "title": "팔 구동", "col": 1, "manager": "/controller_manager", "units": ["drivers#3", "drivers#9"]},
    ]}, path=Path("p.yaml"))
    s = SimpleNamespace(diagram=d)
    assert Console._robot_keys(None, s) == {"pd_load_right#0", "pd_arm_right#1"}   # noqa — self 를 쓰지 않는다
    assert Console._stack_keys(s) == {"drivers#3", "drivers#9"}


# ── 감독자가 내린 프로세스는 '죽은' 것이 아니다 ─────────────────────────
def test_a_process_the_supervisor_stopped_is_marked_stopped_and_one_that_exits_by_itself_is_not(tmp_path):
    from s2r_console import supervisor as SV
    sup = SV.Supervisor(cwd=tmp_path, run_dir=tmp_path, env=dict(os.environ))
    sup.spawn("a#0", stage="a", note="잠", argv=[sys.executable, "-c", "import time; time.sleep(30)"], background=True,
              manual=False)
    sup.spawn("b#0", stage="b", note="곧 끝", argv=[sys.executable, "-c", "pass"], background=True, manual=False)
    sup.stop(["a#0"])
    deadline = time.time() + 5
    while time.time() < deadline and sup.is_alive("b#0"):
        time.sleep(0.05)
    rows = {p["key"]: p for p in sup.table()}
    assert rows["a#0"]["stopped"] and not rows["a#0"]["alive"]
    assert not rows["b#0"]["stopped"] and not rows["b#0"]["alive"]


def test_generated_units_keys_are_mission_command_keys(real):
    _, units, d = real
    assert set(d.units()) <= set(units)
    assert json.dumps(d.as_dict(), ensure_ascii=False)                       # 직렬화된다(run.json)


def test_merging_keeps_a_wire_always_on_when_any_command_uses_it_always():
    g = W._Graph(status_nodes=())
    g.box("a", "A", 0, unit="x#0")
    g.box("b", "B", 1, unit="y#0")
    g.wire("a", "b", "/t", units=["y#0"])
    g.wire("a", "b", "/t")
    assert g.wires == [{"from": "a", "to": "b", "topic": "/t"}]


def test_a_unitless_repeat_does_not_rename_the_box_and_a_new_command_merges_the_side_into_the_title():
    g = W._Graph(status_nodes=())
    g.box("fabric", "fabric_node · 역기구학 (오른팔)", 5, ros=["/fabric_node"], unit="r#0")
    g.box("fabric", "fabric_node · 역기구학 (오른팔)", 5, ros=["/fabric_node"])
    assert g.boxes["fabric"]["title"] == "fabric_node · 역기구학 (오른팔)"
    g.box("fabric", "fabric_node · 역기구학 (왼팔)", 5, ros=["/fabric_node"], unit="l#0")
    assert g.boxes["fabric"]["title"] == "fabric_node · 역기구학 (오른팔 | 왼팔)" and g.boxes["fabric"]["units"] == ["r#0", "l#0"]
    other = g.box("fabric", "fabric_node · 역기구학", 5, ros=["/fabric_node_left"], unit="z#0")
    assert other != "fabric" and g.boxes[other]["units"] == ["z#0"]                # 다른 노드는 제 상자


def test_a_handler_that_fails_after_joining_a_box_leaves_that_box_as_it_was(tmp_path):
    # 실행기 둘(우 · 좌 에피소드)이 같은 상자다. 두 번째가 상자에 제 명령을 더한 뒤 실패하면 그 명령은 상자에서 빠져야 한다
    # 10.17 실행기는 배치 기록(snapshot)이 있는 에피소드에서만 컵 토픽을 받는다 — 두 파일 모두 snapshot 을 둔다
    snap = [{"id": "s", "type": "snapshot"}]
    (tmp_path / "good.yaml").write_text(yaml.safe_dump({"objects": {"CUP": {"topic": "/objects/cyl60/pose"}}, "sequence": snap}),
                                        encoding="utf-8")
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"objects": {"CUP": {"name": "no topic"}}, "sequence": snap}), encoding="utf-8")

    def runner(key, path):
        return UnitCmd(key=key, stage=key.split("#")[0], index=6, note="", needs=(), kind="background", touches_real=False,
                       argv=("python3", "/x/episode_runner_node.py", "--ros-args", "-p", f"episode:={path}"))

    d = W.generate({"ep_r#6": runner("ep_r#6", tmp_path / "good.yaml"), "ep_l#6": runner("ep_l#6", tmp_path / "bad.yaml")},
                   repo=tmp_path, status_nodes=())
    assert d.box("episode_runner").units == ("ep_r#6",)
    (bad,) = [b for b in d.boxes if b.units == ("ep_l#6",)]
    assert bad.id.startswith("unit_") and "읽지 못했다" in bad.note


def test_a_topic_heard_from_two_boxes_names_the_sender_in_the_port(real):
    # 두 정책이 같은 pd 로 joint_target 을 낸다 — 포트 이름이 같으면 화면에서 중복처럼 보인다
    _, _, d = real
    labels = {w.label for w in d.wires if w.dst == "pd" and w.topic == "/policy_control/joint_target"}
    assert labels == {"집기 ▸ joint_target", "놓기 ▸ joint_target"}                  # 내는 쪽이 앞 — 좁은 상자에서도 안 잘린다
    # ★10.19 목표는 오른팔 실행기 · 두 팔 실행기 둘이 낸다 — 내는 쪽 이름이 앞
    # ★10.20 aglt 노드를 띄우는 단계의 실행기만 — 둘 이상일 때만 이름이 앞에 붙는다(옛 정책을 내린 뒤 지금은 오른팔 실행기 하나)
    goal_tags = set(_goal_runners(real[1]).values())
    assert {w.label for w in d.wires if w.dst == "rh_aglt_node_right" and w.topic.endswith("/goal")} == (
        {f"{tag} ▸ right/goal" for tag in goal_tags} if len(goal_tags) > 1 else {""})
    holder = "/objects/cup_holder_1/pose (바깥)" if GEN.REMOTE_PERCEPTION else "마커"     # 10.13 원격 인지면 마커 노드가 없다
    assert {w.label for w in d.wires if w.dst == "rh_place_node_right" and w.topic.endswith("cup_holder_1/pose")} == {
        f"{holder} ▸ cup_holder_1/pose", "우 실행기 ▸ cup_holder_1/pose"}        # ★10.19 놓기 에피소드는 오른팔 실행기
    assert not d.box("pd").title.count("무발행")                                # 제목은 노드, 무발행 · 발행은 칩과 스위치가 말한다


def test_a_running_node_that_listens_through_its_poll_node_is_heard():
    # 10.04 CPU: pd · 정책 노드는 센서를 '<노드>_poll' 로 구독한다 — 본 노드 이름만 보면 늘 "받는 쪽이 구독하지 않는다"
    spec = parse_diagram({"boxes": [
        {"id": "hand", "title": "손 상태", "col": 0, "ros": ["/rh56f1_state_right"], "unit": "hand#2"},
        {"id": "pd", "title": "pd", "col": 1, "ros": ["/pd_node_right", "/pd_node_poll_right"], "unit": "pd#1"},
    ], "wires": [{"from": "hand", "to": "pd", "topic": "/hand_right/joint_states", "stale_ms": 500}]}, path=Path("p.yaml"))
    obs = _obs(("/rh56f1_state_right", "/pd_node_right", "/pd_node_poll_right"),
               {"/hand_right/joint_states": _topic(("/rh56f1_state_right",), ("/pd_node_poll_right",), n=250)})
    alive = {k: {"key": k, "kind": "background", "alive": True, "started": True, "stopped": False} for k in ("hand#2", "pd#1")}
    (port,) = _ports(build(obs, spec, units=alive)).values()
    assert port["state"] == "live" and port["text"].startswith("250 Hz")


def test_the_inputs_of_a_node_that_is_off_stay_quiet():
    # 꺼진 노드의 입력마다 사유를 적으면 상자가 사유로 가득 찬다 — 상자 하나가 "아직 켜지 않았다" 고 말한다
    out = build(_obs(("/object_pose_node", "/pd_node_right"), {"/objects/cyl60/pose": _topic(("/object_pose_node",), ())}),
                SPEC, units=_units(alive=("cups#3", "pd_arm#1")))
    port = _ports(out)[("pose", "/objects/cyl60/pose")]
    assert port["state"] == "off" and port["note"] == ""
    aglt = next(b for col in out["cols"] for b in col if b["id"] == "aglt")
    assert aglt["state"] == "off" and aglt["detail"] == "아직 켜지 않았다"


def test_the_fake_hand_publishes_every_topic_the_state_node_reads():
    # 10.05: fake 손이 force_actual 을 안 내 상태 노드가 joint_forces 를 못 냈다 — 정책 입력이 비고 그림이 '끊긴 곳' 이었다
    import re
    state = _src("deploy/policy_control/policy_control/rh56f1_state_node.py")
    fake = _src("scripts/fakes/fake_rh56f1_hand.py")
    read = re.findall(r'create_subscription\(\w+, f"\{ns\}/(\w+)"', state)
    assert set(read) == {"angle_actual", "touch_data", "force_actual"}
    for name in read:
        assert f'f"{{ns}}/{name}"' in fake, name



# ── 연결창은 실기 기준 (10.05 사용자: "노드 중에 FAKE 쪽은 필요 없을 것 같은데 — 실기 값 · 연결 상태 위주로") ──
def _profile_diagram(name):
    import mission_core as MC
    prof = load(PROFILES / f"{name}.yaml", repo=SIM2REAL)
    mission = MC.load_mission(yaml.safe_load(prof.mission.read_text(encoding="utf-8")))
    return prof, W.generate(drawable_units(mission, mission_units(prof, repo=SIM2REAL)), repo=SIM2REAL,
                            status_nodes=prof.status_nodes)


@pytest.mark.parametrize("name", sorted(p.stem for p in PROFILES.glob("*_real.yaml")))
def test_real_profiles_draw_no_fake_node(name):
    _, d = _profile_diagram(name)
    bad = [b.title for b in d.boxes if "fake" in f"{b.title} {' '.join(b.ros)}".lower() or "mock" in b.title.lower()]
    assert not bad


@pytest.mark.parametrize("name", sorted(p.stem for p in PROFILES.glob("*_fake.yaml")))
def test_fake_stand_ins_take_the_place_of_the_real_boxes_instead_of_being_their_own_nodes(name):
    _, d = _profile_diagram(name)
    assert not [b.title for b in d.boxes if "(fake)" in b.title or "mock" in b.title.lower()]
    assert not [b.id for b in d.boxes if b.id.startswith("obj_")]


def test_the_fake_rh56f1_picture_has_the_real_pictures_boxes():
    _, real = _profile_diagram("rh56f1_real")
    _, fake = _profile_diagram("rh56f1_fake")
    titles = {b.title for b in real.boxes}
    # fake 에 없는 것: 인지(카메라 · FP++ · 런처 — 정지 컵을 fake 가 바로 낸다), 목 드라이버(fake 에는 목이 없다). fake 에만 있는 것: 막히지 않은 붓기 노드
    missing = {b.title for b in real.boxes} - {b.title for b in fake.boxes}
    assert missing <= {f"인지 런처 · {GEN.PERCEPTION_HOST}", "카메라 (RealSense)", "목 상태 (head)", "/objects/cup_holder_1/pose (바깥)", *(f"FPP 추적 · {g}" for g in (*_active_groups(), "source240b"))}, missing   # rg3 파지 에피소드의 오른 컵 = source240b
    extra = {b.title for b in fake.boxes} - titles
    # 10.13 원격 인지 실기에는 마커 노드가 없다 — fake 는 그대로 홀더 노드 상자를 그린다
    assert extra <= {"pour_fj_node · tilt 정책 (오른팔 · 왼팔)", *([W.HOLDER_TITLE] if GEN.REMOTE_PERCEPTION else [])}, extra


def test_policy_nodes_and_the_runner_say_their_own_phase_and_the_bridge_listens():
    from s2r_console.console import bridge_argv
    prof, d = _profile_diagram("rh56f1_real")
    assert prof.status_nodes == ("pd_right", "pd_left")                    # 노드 표 · 배너 · 지연 기준은 그대로
    want = {"rh_aglt_node_right", "rh_aglt_node_left", "rh_place_node_right", "rh_place_node_left", "episode_runner"}
    units = mission_units(prof, repo=SIM2REAL)
    if not any('__node:=rh_aglt_node_left' in u.argv for u in units.values()):
        want.discard('rh_aglt_node_left')
    assert {b.id: b.status for b in d.boxes if b.id in want} == {n: n for n in want}
    argv = bridge_argv(prof, d)
    nodes = argv[argv.index("--nodes") + 1:argv.index("--topics")]
    assert set(nodes) >= want | {"pd_right", "pd_left"}


def test_a_policy_box_shows_its_phase_and_its_refusal_reason():
    spec = parse_diagram({"boxes": [
        {"id": "aglt", "title": "aglt", "col": 0, "status": "rh_aglt_node_right", "ros": ["/rh_aglt_node_right"], "unit": "a#1"},
    ]}, path=Path("p.yaml"), status_nodes=("rh_aglt_node_right",))
    up = {"a#1": {"key": "a#1", "kind": "background", "alive": True, "started": True, "stopped": False}}
    ok = Observed(bridge_up=True, status={"rh_aglt_node_right": {"phase": "running", "seq": 812, "ok": True}},
                  age_s={"rh_aglt_node_right": 0.1}, graph=("/rh_aglt_node_right",), topics={}, topics_age_s=0.5)
    box = build(ok, spec, units=up)["cols"][0][0]
    assert box["state"] == "live" and box["detail"] == "running · seq 812"
    refused = Observed(bridge_up=True, status={"rh_aglt_node_right": {"phase": "idle", "seq": 3, "ok": False,
                                                                       "reasons": ["컵 자세가 없다 (0.5 s)"]}},
                       age_s={"rh_aglt_node_right": 0.1}, graph=("/rh_aglt_node_right",), topics={}, topics_age_s=0.5)
    box = build(refused, spec, units=up)["cols"][0][0]
    assert box["state"] == "fault" and "컵 자세가 없다" in box["detail"]


def test_an_idle_policy_does_not_take_credit_for_another_policys_joint_target():
    # 에피소드: aglt 가 running 으로 joint_target 을 내고 place 는 떠서 idle — place → pd 전선은 쉬는 것이지 초록이 아니다
    spec = parse_diagram({"boxes": [
        {"id": "place", "title": "place", "col": 0, "status": "rh_place_node_right", "ros": ["/rh_place_node_right"], "unit": "p#4"},
        {"id": "pd", "title": "pd", "col": 1, "ros": ["/pd_node_right", "/pd_node_poll_right"], "unit": "pd#1"},
    ], "wires": [{"from": "place", "to": "pd", "topic": "/policy_control/joint_target", "stale_ms": 500, "episodic": True}]},
        path=Path("p.yaml"), status_nodes=("rh_place_node_right",))
    obs = Observed(bridge_up=True, status={"rh_place_node_right": {"phase": "idle", "seq": 0, "ok": True}},
                   age_s={"rh_place_node_right": 0.1}, graph=("/rh_place_node_right", "/pd_node_right", "/pd_node_poll_right"),
                   topics={"/policy_control/joint_target": _topic(("/rh_aglt_node_right", "/rh_place_node_right"),
                                                                  ("/pd_node_poll_right",))}, topics_age_s=0.5)
    alive = {k: {"key": k, "kind": "background", "alive": True, "started": True, "stopped": False} for k in ("p#4", "pd#1")}
    (port,) = _ports(build(obs, spec, units=alive)).values()
    assert port["state"] == "held" and port["note"] == "에피소드 밖(idle)"


def test_only_a_real_node_box_says_its_commands_run_one_at_a_time():
    out = build(_obs((), {}), SPEC, units=_units())
    boxes = {b["id"]: b for col in out["cols"] for b in col}
    assert boxes["aglt"]["single"] is True                                      # ROS 노드 하나
    fake = parse_diagram({"boxes": [{"id": "object_pose", "title": "물체 자세", "col": 0, "units": ["c#0", "c#1"]}]},
                         path=Path("p.yaml"))
    (box,) = build(_obs((), {}), fake, units={})["cols"][0]
    assert box["single"] is False                                               # fake 컵 대역 여럿은 같이 뜬다


def test_an_episode_without_a_snapshot_draws_no_fpp_cup_for_the_runner(tmp_path):
    # 10.17 경로 B(tilt_base_b): 사람이 컵을 손에 댄다 — 실행기가 FP++ 컵을 안 받으니 '끊긴 곳'으로 그리지 않는다
    def runner(key, path):
        return UnitCmd(key=key, stage=key.split("#")[0], index=5, note="", needs=(), kind="background", touches_real=False,
                       argv=("python3", "/x/episode_runner_node.py", "--ros-args", "-p", f"episode:={path}"))

    for name, want in (("tilt_base_b", False), ("tilt_base_a", True)):
        d = W.generate({f"ep_{name[-1]}#5": runner(f"ep_{name[-1]}#5", SIM2REAL / "config" / "episodes" / f"{name}.yaml")},
                       repo=SIM2REAL, status_nodes=())
        cups = [w for w in d.wires if w.dst == "episode_runner" and w.topic.startswith("/objects/")]
        assert bool(cups) is want, name

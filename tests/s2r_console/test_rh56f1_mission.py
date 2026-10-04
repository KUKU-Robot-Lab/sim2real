"""RH56F1 로봇(arm4090) 미션 — 실기 · fake 가 한 정의에서 나오고, 손 명령은 승인 단계에서만 나간다. 09.29 사용자.

"sim2real 과 robot_control 쪽에서 rh56f1 제어 part 연결" · 손마다 개별 포트 · USB RS485 / CANFD 를 상황에 따라.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from mission_core import load_mission
from mission_stages import commands_for, load_runbook

REPO = Path(__file__).resolve().parents[2]
PC = REPO / "deploy" / "policy_control"
sys.path.insert(0, str(PC / "tools"))


def _load(name):
    raw = yaml.safe_load((REPO / "config" / name).read_text(encoding="utf-8"))
    m = load_mission(raw)
    return raw, m, load_runbook(raw["run"], m)


REAL_RAW, REAL, REAL_BOOK = _load("mission_rh56f1_control.yaml")
FAKE_RAW, FAKE, FAKE_BOOK = _load("mission_rh56f1_fake.yaml")


def _cmds(m, book, sid):
    return commands_for(book, m, sid, repo=REPO, execute=True)


def test_the_committed_missions_come_from_the_generator():
    r = subprocess.run([sys.executable, str(REPO / "scripts/ops/make_rh56f1_missions.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_real_and_fake_have_the_same_stages_and_order():
    shape = lambda m: [(s.id, s.group, s.lane, s.needs, s.undoes) for s in m.stages]  # noqa: E731
    assert shape(REAL) == shape(FAKE)
    assert not any(s.touches_real for s in FAKE.stages)
    real_touch = {s.id for s in REAL.stages if s.touches_real}
    assert {"drivers", "hand_right", "hand_left", "probe_right", "probe_left", "pd_load_right", "pd_arm_right",
            "release_right", "shutdown"} <= real_touch
    assert not {"preflight", "hand_check_right", "hand_check_left"} & real_touch          # 읽기만


def test_hands_launch_the_driver_on_the_real_robot_and_the_fake_hand_in_fake():
    for s in ("right", "left"):
        real = " ".join(" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, f"hand_{s}"))
        fake = " ".join(" ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, f"hand_{s}"))
        assert f"rh56f1_driver.py --side {s}" in real and "fake_rh56f1_hand" not in real
        assert f"fake_rh56f1_hand.py --side {s}" in fake and "rh56f1_driver.py" not in fake
        assert f"rh56f1_state_node.py --side {s}" in real and f"rh56f1_state_node.py --side {s}" in fake
        assert "rh56f1_ports" in REAL.stages[[x.id for x in REAL.stages].index(f"hand_{s}")].artifacts


def test_the_probe_moves_one_axis_at_a_time_only_with_execute_and_waits_for_the_eye():
    for s in ("right", "left"):
        cmds = _cmds(REAL, REAL_BOOK, f"probe_{s}")
        plan = commands_for(REAL_BOOK, REAL, f"probe_{s}", repo=REPO, execute=False)
        probes = [i for i, c in enumerate(cmds) if any("rh56f1_axis_probe.py" in a for a in c.argv)]
        axes = [cmds[i].argv[cmds[i].argv.index("--axis") + 1] for i in probes]
        assert sorted(axes) == sorted(["index_1", "middle_1", "ring_1", "pinky_1", "thumb_2", "thumb_1"])
        assert all("--execute" in cmds[i].argv and "--execute" not in plan[i].argv and "--back" in cmds[i].argv
                   for i in probes)
        assert all(cmds[i + 1].manual for i in probes)                                      # 눈으로 확인
        assert cmds[0].manual                                                               # 손 주변 확인이 먼저


def test_pd_uses_the_rh56f1_configs_and_fake_launches_refuse_domain_zero():
    for s in ("right", "left"):
        load = " ".join(_cmds(REAL, REAL_BOOK, f"pd_load_{s}")[0].argv)
        arm = " ".join(_cmds(REAL, REAL_BOOK, f"pd_arm_{s}")[1].argv)
        assert "pd_rh56f1.yaml" in load and "execute:=false" in load and f"rh56f1_{s}_real.yaml" in load
        assert "pd_rh56f1_exec.yaml" in arm and "execute:=true" in arm
        for sid in (f"pd_load_{s}", f"pd_arm_{s}"):
            for c in _cmds(FAKE, FAKE_BOOK, sid):
                if c.argv and c.argv[0] == "ros2":
                    assert "fake:=true" in c.argv and any("_fake.yaml" in a for a in c.argv)


def test_every_stop_names_a_background_command():
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        bg = {f"{s.id}#{i}" for s in m.stages for i, c in enumerate(_cmds(m, book, s.id)) if c.background}
        stops = {k for s in m.stages for c in _cmds(m, book, s.id) for k in c.stop}
        assert stops <= bg, sorted(stops - bg)


def test_every_execute_flag_is_one_the_tool_takes():
    for m, book in ((REAL, REAL_BOOK),):
        for s in m.stages:
            for spec in book.commands.get(s.id, ()):
                if not spec.execute_args:
                    continue
                tool = next(a for a in spec.argv if a.endswith(".py"))
                path = Path(tool.replace("{repo}", str(REPO)))
                text = path.read_text(encoding="utf-8")
                assert all(f'"{f}"' in text for f in spec.execute_args if f.startswith("--")), (s.id, tool)


def test_the_driver_launcher_reads_the_port_file_and_refuses_one_port_for_two_hands():
    """10.02: 손은 EtherCAT(손 하나 = NIC 하나). rs485 · canfd 는 벤더 드라이버 경로로 남는다."""
    import rh56f1_driver as D
    cfg = yaml.safe_load((PC / "config" / "rh56f1_ports.yaml").read_text())
    assert cfg["right"]["transport"] == cfg["left"]["transport"] == "ethercat"
    assert (cfg["right"]["ifname"], cfg["left"]["ifname"]) == ("enx00e04c6806e1", "enp6s0")
    argv = D.argv_for(cfg, "right", "P.yaml")
    assert argv[1].endswith("policy_control/rh56f1_ecat_node.py") and argv[2:] == ["--side", "right", "--ports", "P.yaml"]
    assert D.argv_for(cfg, "left", "P.yaml", no_op=True)[-1] == "--no-op"
    with pytest.raises(ValueError, match="같은 NIC"):
        D.argv_for({"right": cfg["right"], "left": dict(cfg["left"], ifname=cfg["right"]["ifname"])}, "right")
    with pytest.raises(ValueError, match="transport"):
        D.argv_for({"right": dict(cfg["right"], transport="can"), "left": cfg["left"]}, "right")
    # 비상용 RS485 블록 · OP 실험 인자
    fb = D.argv_for(cfg, "right", transport="rs485")
    assert fb[:4] == ["ros2", "launch", "rh56f1_driver", "rh56f1_right_driver.launch.py"]
    assert "port:=/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG0327KL-if00-port0" in fb
    assert D.argv_for(cfg, "right", "P.yaml", extra=["--op-enable"])[-1] == "--op-enable"
    with pytest.raises(ValueError, match="transport"):
        D.argv_for(cfg, "right", transport="can")
    rs = {"right": {"transport": "rs485", "port": "/dev/a", "baud": 115200, "hand_id": 1},
          "left": {"transport": "rs485", "port": "/dev/b", "baud": 115200, "hand_id": 1}}
    assert D.argv_for(rs, "right")[:4] == ["ros2", "launch", "rh56f1_driver", "rh56f1_right_driver.launch.py"]
    with pytest.raises(ValueError, match="같은 포트"):
        D.argv_for({"right": rs["right"], "left": dict(rs["left"], port="/dev/a")}, "right")


def test_the_console_opens_the_rh56f1_fake_profile_with_its_robot_module(tmp_path, monkeypatch):
    import mission_run
    from s2r_console.console import Console
    monkeypatch.setattr(mission_run, "MISSION_LOG_DIR", tmp_path / "logs")
    c = Console(bridge=False)
    try:
        snap = c.snapshot()
        (rh,) = [r for r in snap["robots"] if r["id"] == "openarm_rh56f1"]
        assert [p["id"] for p in rh["profiles"]] == ["rh56f1_real", "rh56f1_fake"]
        s = c.open("rh56f1_fake", operator="pytest")
        assert s.robot is not None and s.robot.id == "openarm_rh56f1"
        assert s.joint_limits.get("r_hj_thumb_1") == pytest.approx((0.0, 2.0944))          # 로봇 모듈의 프로필
        view = c.snapshot()["session"]
        assert view["robot_module"]["id"] == "openarm_rh56f1"
        assert {g["title"] for g in view["robot"]["groups"]} >= {"오른손", "왼손"} or view["robot"]["groups"] == []
    finally:
        c.shutdown()


def test_only_the_two_cup_pour_is_blocked_on_the_real_robot():
    """10.01: 인지는 arm4090 안(FP++ 컵 하나)으로 풀었다 — 두 컵이 필요한 양팔 pour_fj 만 이유와 함께 막는다."""
    by = {s.id: s for s in REAL.stages}
    assert {s.id for s in REAL.stages if s.blocked} == {"policy_pourfj"} and "두 컵" in by["policy_pourfj"].blocked
    assert not any(s.blocked for s in FAKE.stages)


def test_real_cups_run_fpp_on_this_pc_after_the_head_home_and_shutdown_takes_it_down():
    """10.01 사용자: FP++ 를 arm4090 에. 런처 --host local · UDP 수신 · base 변환 · start, 머리 기준자세가 먼저."""
    by = {s.id: s for s in REAL.stages}
    assert by["cups"].needs == ("head_home",) and by["head_home"].touches_real and by["cups"].touches_real
    head = _cmds(REAL, REAL_BOOK, "head_home")
    assert any("head_home.py" in " ".join(c.argv) and "head_home_rh56f1.yaml" in " ".join(c.argv) for c in head)
    cups = [" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, "cups")]
    for want in ("perception_launcher_node.py --host local", "fpp_pose_rx.py", "object_pose_node.py --objects cyl60",
                 "--camera-extrinsics", "global_camera_extrinsics_arm4090.yaml",
                 "perception_ctl.py start cyl60 --wait 150"):
        assert any(want in a for a in cups), want
    bg = [c for c in _cmds(REAL, REAL_BOOK, "cups") if c.background]
    assert len(bg) == 3
    down = [" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, "shutdown")]
    assert any("perception_ctl.py stop --camera --host local" in a for a in down)
    stops = {x for c in _cmds(REAL, REAL_BOOK, "shutdown") for x in c.stop}
    assert {"cups#1", "cups#2", "cups#3"} <= stops


@pytest.mark.parametrize("side", ["right", "left"])
def test_home_is_one_approval_then_runs_through(side):
    """10.01 사용자: "home 자세 진행하면 팔-손 한번에". 사람 확인은 맨 앞 하나 → engage → 주먹 → 시작점 검사 → 재생 → 정착 → 손 홈."""
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        cmds = _cmds(m, book, f"home_{side}")
        text = [" ".join(c.argv) for c in cmds]
        assert cmds[0].manual and not any(c.manual for c in cmds[1:])
        order = ["--only pd_engage --hold-s 10", "--only pd_hand_path", "check_path_start.py", "replay_to_pd.py",
                 "--only pd_goto_home", "--only pd_hand_home"]
        assert len(text) == 1 + len(order) and all(want in t for want, t in zip(order, text[1:]))
        assert f"home_rh56f1_{side}.npz" in text[3] and "--contract" in text[3] and "--at" not in text[3]
        assert f"home_rh56f1_{side}.npz" in text[4] and "--reverse" not in text[4] and "--execute" in cmds[4].argv
        assert f"path_{side}" in m.stages[[x.id for x in m.stages].index(f"home_{side}")].artifacts


@pytest.mark.parametrize("side", ["right", "left"])
def test_return_goes_home_to_rest_on_the_same_path_reversed_then_releases(side):
    """10.01 4090:s2r 실기 순서: 홈 정착 → 주먹 → 경로 끝 검사 → 역재생 → pd 해제. 한 번 승인, home 을 되돌린다."""
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        by = {x.id: x for x in m.stages}
        st = by[f"return_{side}"]
        assert st.needs == (f"home_{side}",) and f"home_{side}" in st.undoes
        cmds = _cmds(m, book, f"return_{side}")
        text = [" ".join(c.argv) for c in cmds]
        assert cmds[0].manual and not any(c.manual for c in cmds[1:])
        order = ["--only pd_goto_home", "--only pd_hand_path", "check_path_start.py", "replay_to_pd.py", "--only pd_release"]
        assert len(text) == 1 + len(order) and all(want in t for want, t in zip(order, text[1:]))
        assert "--at end" in text[3] and "--reverse" in text[4] and f"home_rh56f1_{side}.npz" in text[4]
    assert by[f"return_{side}"].touches_real is False                      # fake 미션
    assert {x.id: x for x in REAL.stages}[f"return_{side}"].touches_real


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_saved_path_goes_from_zero_to_the_aglt_home_of_the_current_contract(side):
    import hashlib
    import numpy as np
    d = np.load(REPO / REAL.artifacts[f"path_{side}"])
    contract = REPO / REAL.artifacts["contract"]
    assert str(d["meta_contract_sha1"]) == hashlib.sha1(contract.read_bytes()).hexdigest()   # 계약이 바뀌면 다시 계획
    homes = yaml.safe_load((PC / "config/homes/rh56f1_aglt.yaml").read_text())
    assert np.allclose(d["meta_start"], 0.0) and np.allclose(d["meta_goal"], homes[side])
    a = d["arm_target"]
    assert np.allclose(a[0], 0.0) and np.allclose(a[-1], homes[side])
    # 10.01 사용자: 차렷 손 = 주먹(pd hand_path_pose)으로 계획(--hand-start pd → meta 는 measured + 그 손 값),
    #   경로 전용 속도 0.3 rad/s(--max-speed, 각 약 15 s). pd ramp_speed 0.2 는 그대로.
    assert float(d["meta_min_clearance_non_escape"]) >= 0.02 - 1e-4 and str(d["meta_hand_start"]) == "measured"
    pose = yaml.safe_load((PC / "config/pd_rh56f1.yaml").read_text())["hand_path_pose"][side]
    hand_q = dict(kv.split("=") for kv in str(d["meta_hand_q"]).split(","))
    assert {k: float(v) for k, v in hand_q.items()} == pytest.approx({k: float(v) for k, v in pose.items()})
    assert float(d["meta_max_joint_speed"]) <= 0.3 + 1e-6 and str(d["meta_other_arm"]) == "both"
    assert list(d["meta_joints"]) == [f"{side[0]}_aj_{i}" for i in range(1, 8)]


def test_the_pour_fj_policy_stage_runs_the_picked_contract_with_the_venv_python():
    real = _cmds(REAL, REAL_BOOK, "policy_pourfj")
    node = next(c for c in real if any("pour_fj_node.py" in a for a in c.argv))
    assert node.background and node.argv[0].endswith(".venv/bin/python")
    assert any("contract:=" in a and "pour_fj_contract.json" in a for a in node.argv)
    assert any(a == "max_episode_s:=15.0" for a in node.argv)                      # 정책 카드 deploy
    by = {s.id: s for s in REAL.stages}
    assert {"home_right", "home_left", "cups"} <= set(by["policy_pourfj"].needs) and by["policy_pourfj"].touches_real
    assert [c for c in real if "episode/start" in " ".join(c.argv)][0].argv[-1] == "--execute"   # execute 일 때만
    assert {la.id: la.focus for la in REAL.lanes}["both"] == "policy_pourfj"


def test_the_rh56f1_robot_offers_the_pour_fj_policy_for_both_arms():
    from s2r_console import robots as RB
    (rh,) = [r for r in RB.scan(REPO / "deploy/s2r_console/robots")[0] if r.id == "openarm_rh56f1"]
    assert rh.slots["both"] == "pourfj_both" and RB.slot_contracts("pourfj_both") == ("pour_fj_contract.json",)
    assert REAL.artifacts["pourfj_both"].endswith("both_rh_pourfj_f01/pour_fj_contract.json")


@pytest.mark.parametrize("side,cup", [("right", "src"), ("left", "rcv")])
def test_each_arm_runs_its_rh_aglt_policy_after_home(side, cup):
    """09.30 사용자: RH56F1 정책을 deploy 에 연결 — 한 팔 rh_aglt 는 그 팔의 창에서, 홈 뒤 정책 카드."""
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        cmds = _cmds(m, book, f"policy_aglt_{side}")
        node = next(c for c in cmds if any("rh_aglt_node.py" in a for a in c.argv))
        assert node.background and node.argv[0].endswith(".venv/bin/python")
        assert any(a.startswith("contract:=") and a.endswith("rh_aglt_contract.json") for a in node.argv)
        topic = "/objects/cyl60/pose" if m is REAL else f"/objects/cup_{cup}/pose"   # 실기 컵 = cyl60(10.04) FP++ 물체 하나
        assert f"cup_topic:={topic}" in node.argv and "max_episode_s:=15.0" in node.argv
        by = {s.id: s for s in m.stages}
        assert {f"home_{side}", "cups", f"hand_check_{side}"} <= set(by[f"policy_aglt_{side}"].needs)
        assert {la.id: la.focus for la in m.lanes}[f"arm_{side}"] == f"policy_aglt_{side}"
    assert REAL.stages[[s.id for s in REAL.stages].index(f"policy_aglt_{side}")].touches_real


def test_the_rh56f1_robot_offers_rh_aglt_for_each_arm():
    from s2r_console import robots as RB
    (rh,) = [r for r in RB.scan(REPO / "deploy/s2r_console/robots")[0] if r.id == "openarm_rh56f1"]
    assert rh.slots["right"] == "aglt_right" and rh.slots["left"] == "aglt_left"
    assert RB.slot_contracts("aglt_right") == ("rh_aglt_contract.json",)


def test_real_cup_holders_run_the_marker_node_after_the_head_home_and_shutdown_stops_it():
    """10.02 사용자: 컵홀더 마커 자동 추정. 카메라만 켜고(로봇 무동작) 노드를 배경으로, 머리 기준자세가 먼저."""
    by = {s.id: s for s in REAL.stages}
    assert by["cup_holders"].needs == ("head_home",)
    cmds = _cmds(REAL, REAL_BOOK, "cup_holders")
    argv = [" ".join(c.argv) for c in cmds]
    assert any("scripts/vision/camera_up.sh" in a for a in argv)
    bg = [c for c in cmds if c.background]
    assert len(bg) == 1 and "cup_holder_pose_node.py --write" in " ".join(bg[0].argv)
    assert cmds.index(bg[0]) == 2                                   # shutdown 의 cup_holders#2
    assert cmds[0].manual and cmds[-1].manual and "/cup_holders/status" in " ".join(cmds[-1].argv)
    stops = {x for c in _cmds(REAL, REAL_BOOK, "shutdown") for x in c.stop}
    assert "cup_holders#2" in stops
    fake = {s.id for s in FAKE.stages}
    assert "cup_holders" in fake and not any(c.background for c in _cmds(FAKE, FAKE_BOOK, "cup_holders"))


def test_the_real_preflight_checks_the_cpu_on_this_pc_and_fake_does_not():
    # 10.03 사용자: CPU 최적화는 PC 가 바뀌어도 자동 — 실기 세션마다 실시간 한도 · 코어 배치를 본다. fake 는 무관
    cpu = lambda book, m: [list(c.argv) for c in _cmds(m, book, "preflight") if "--only" in c.argv]  # noqa: E731
    assert cpu(REAL_BOOK, REAL) == [["python3", f"{REPO}/scripts/setup/check_host.py", "--robot", "rh56f1", "--only", "cpu"]]
    assert cpu(FAKE_BOOK, FAKE) == []


@pytest.mark.parametrize("side", ["right", "left"])
def test_each_arm_places_the_cup_after_its_aglt_hand_off(side):
    """10.04 사용자: 성공한 정책부터 실기에 — 놓기(rh_place)는 aglt 뒤, 홀더 자세가 있을 때. fake 는 latched 홀더 · 쥠 검사 끔."""
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        cmds = _cmds(m, book, f"policy_place_{side}")
        node = next(c for c in cmds if any("rh_place_node.py" in a for a in c.argv))
        assert node.background and node.argv[0].endswith(".venv/bin/python")
        assert any(a.startswith("contract:=") and a.endswith("rh_place_contract.json") for a in node.argv)
        assert "holder:=1" in node.argv and f"ns:={side}" in node.argv
        assert ("require_grasp:=false" in node.argv) is (m is FAKE)
        by = {s.id: s for s in m.stages}
        assert {f"policy_aglt_{side}", "cup_holders"} <= set(by[f"policy_place_{side}"].needs)
        resets = [c for c in cmds if "episode/reset" in c.argv]
        assert resets and all(f"{side}" in c.argv for c in resets)
    holder = next(c for c in _cmds(FAKE, FAKE_BOOK, f"policy_place_{side}") if any("fake_cup_pose_pub" in a for a in c.argv))
    assert "--latched" in holder.argv and "/objects/cup_holder_1/pose" in holder.argv
    assert REAL.stages[[s.id for s in REAL.stages].index(f"policy_place_{side}")].touches_real

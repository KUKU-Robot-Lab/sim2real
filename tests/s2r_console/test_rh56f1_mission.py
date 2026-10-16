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
#: 10.09 실기 물체는 config/fpp_active.yaml(fpp_object.py activate)이 정한다 — 테스트도 그것을 읽는다(물체를 바꿔도 안 깨진다)
_ACTIVE = yaml.safe_load((REPO / "config" / "fpp_active.yaml").read_text(encoding="utf-8"))
ACTIVE_SIDES = _ACTIVE["sides"]
FAKE_RAW, FAKE, FAKE_BOOK = _load("mission_rh56f1_fake.yaml")


def _generator():
    """미션 생성기 — ★10.13 로봇 PC(STATIONS) 에 따라 인지 호스트 · 외부 파라미터 · 머리 도구가 바뀐다."""
    import importlib.util
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("make_rh56f1_missions", REPO / "scripts" / "ops" / "make_rh56f1_missions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _generator()


# 10.12 사용자(A안): aglt 정책은 쥘 때까지 FP++ live 를 받고(끊기면 마지막 값), 쥔 뒤는 프레임 시각 FK(가리면 0.7 s 대체)
AGLT_LIVE_ARGS = ("cup_static:=false", "cup_latch:=false", "cup_hold_last:=true", "attach_fallback_s:=0.7")


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
    assert (cfg["right"]["ifname"], cfg["left"]["ifname"]) == ("enx705dccf5a065", "enx00e04c6806e1")   # 10.09 둘 다 USB 3.0 랜
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


def test_real_cups_run_fpp_on_the_perception_pc_after_the_head_home_and_shutdown_takes_it_down():
    """10.01 사용자: FP++ 를 arm4090 에(--host local). 10.13 사용자: 로봇 PC 5090 · 인지 PC vision-3090(--host vision-3090).
    런처 --host <인지 PC> · UDP 수신 · base 변환 · start, 머리 기준자세가 먼저."""
    host, ext = GEN.PERCEPTION_HOST, GEN.CAMERA_EXTRINSICS
    by = {s.id: s for s in REAL.stages}
    assert by["cups"].needs == ("head_home",) and by["head_home"].touches_real and by["cups"].touches_real
    head = _cmds(REAL, REAL_BOOK, "head_home")
    assert any(Path(GEN.HEAD_CONFIG).name in " ".join(c.argv) for c in head)
    cups = [" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, "cups")]
    sys.path.insert(0, str(REPO / "scripts"))
    from object_registry import group_members, load_registry
    side = [ACTIVE_SIDES[s] for s in ("right", "left") if s in ACTIVE_SIDES]
    reg = load_registry()
    groups = list(dict.fromkeys(reg.get(n).fpp["group"] for n in side))          # 팔마다 다른 묶음이어도 된다(10.09)
    objs = " ".join(dict.fromkeys([*side, *(n for g in groups for n in group_members(reg, g))]))
    for want in (f"perception_launcher_node.py --host {host}", "fpp_pose_rx.py", "object_pose_node.py --camera-extrinsics",
                 "--camera-extrinsics", ext,
                 f"perception_ctl.py start {objs} --viewer --wait 150",
                 f"fpp_rescan.py {' '.join(side)} --wait 120"):        # 10.08 다시 실행 = 지금 카메라로 다시 찍기
        assert any(want in a for a in cups), want
    rescan = next(c.argv for c in _cmds(REAL, REAL_BOOK, "cups") if any("fpp_rescan.py" in str(x) for x in c.argv))
    if GEN.REMOTE_PERCEPTION:                                    # 10.13 FP++ 상태는 인지 PC 에서만 보인다
        assert list(rescan[-2:]) == ["--status-host", host]
    else:
        assert "--status-host" not in rescan
    bg = [c for c in _cmds(REAL, REAL_BOOK, "cups") if c.background]
    assert len(bg) == 3
    down = [" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, "shutdown")]
    assert any(f"perception_ctl.py stop --camera --host {host}" in a for a in down)
    if GEN.REMOTE_PERCEPTION:                                            # GPU 여유는 카메라 · FP++ 가 도는 PC 에서 본다
        assert any(a.startswith(f"ssh -o BatchMode=yes {host} nvidia-smi") for a in cups)
    dest = GEN.STATION["pose_dest"]                                      # 10.13 자세 UDP 는 이더넷 허브 주소로(ssh 는 wifi)
    launcher = next(a for a in cups if "perception_launcher_node.py" in a)
    assert (f"--pose-dest {dest}" in launcher) if dest else ("--pose-dest" not in launcher)
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
    assert REAL.artifacts["pourfj_both"].endswith("rh56f1/pour_fj/both_f01/pour_fj_contract.json")


@pytest.mark.parametrize("side,cup", [("right", "src"), ("left", "rcv")])
def test_each_arm_runs_its_rh_aglt_policy_after_home(side, cup):
    """09.30 사용자: RH56F1 정책을 deploy 에 연결 — 한 팔 rh_aglt 는 그 팔의 창에서, 홈 뒤 정책 카드."""
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        cmds = _cmds(m, book, f"policy_aglt_{side}")
        node = next(c for c in cmds if any("rh_aglt_node.py" in a for a in c.argv))
        assert node.background and node.argv[0].endswith(".venv/bin/python")
        assert any(a.startswith("contract:=") and a.endswith("rh_aglt_contract.json") for a in node.argv)
        real_cup = ACTIVE_SIDES.get(side, ACTIVE_SIDES["right"])      # config/fpp_active.yaml
        topic = f"/objects/{real_cup}/live_pose" if m is REAL else f"/objects/cup_{cup}/live_pose"   # 10.12 A안: live 추적
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
    """10.02 사용자: 컵홀더 마커 자동 추정. 카메라만 켜고(로봇 무동작) 노드를 배경으로, 머리 기준자세가 먼저.
    ★10.13 원격 인지(5090 + vision-3090): 노드는 카메라 PC 에서(영상을 그 PC 안에서 읽는다) 그 PC 의 venv 로 돌고,
    고정 자세 파일(STATION holder_poses, 사용자 "5090holder_v1")만 이 PC 로 가져온다."""
    by = {s.id: s for s in REAL.stages}
    assert by["cup_holders"].needs == ("head_home",)
    cmds = _cmds(REAL, REAL_BOOK, "cup_holders")
    stops = {x for c in _cmds(REAL, REAL_BOOK, "shutdown") for x in c.stop}
    argv = [" ".join(c.argv) for c in cmds]
    assert any("scripts/vision/camera_up.sh" in a for a in argv)
    bg = [c for c in cmds if c.background]
    node = " ".join(bg[0].argv)
    assert len(bg) == 1 and "cup_holder_pose_node.py --write" in node
    assert f"--out {GEN.HOLDER_POSES}" in node.replace("{repo}/", "")
    assert f"--extrinsics {GEN.CAMERA_EXTRINSICS}" in node.replace("{repo}/", "")
    assert cmds.index(bg[0]) == 3                                   # shutdown 의 cup_holders#3(앞에 머리 확인 · 맞춤)
    assert "cup_holders#3" in stops
    status = next(c for c in cmds if "/cup_holders/status" in " ".join(c.argv))
    assert cmds[0].manual and status.manual
    if GEN.REMOTE_PERCEPTION:
        host = GEN.PERCEPTION_HOST
        for c in (cmds[2], bg[0], status):                          # 카메라 · 노드 · 확인은 카메라 PC 에서
            assert list(c.argv[:4]) == ["ssh", "-tt", "-o", "BatchMode=yes"] and c.argv[4] == host
        assert f"{GEN.REMOTE_PY} scripts/nodes/cup_holder_pose_node.py" in node
        fetch = cmds[-1]                                            # 마지막 = 파일을 이 PC 로
        assert fetch.argv[0] == "scp" and fetch.argv[-2] == f"{host}:{GEN.REMOTE_SIM2REAL}/{GEN.HOLDER_POSES}"
        assert fetch.argv[-1].endswith("/" + GEN.HOLDER_POSES) and not fetch.manual
    else:
        assert cmds[-1] is status
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


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_episode_stage_brings_up_both_policies_and_the_runner(side):
    """10.04 에피소드 실행기: 한 팔에 aglt(ns <팔>) · place(ns <팔>_place, 이벤트는 팔 토픽) · 실행기 — 진행은 상황판."""
    sid = f"episode_pick_place_{side}"
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        cmds = _cmds(m, book, sid)
        argv = {k: " ".join(c.argv) for k in ("rh_aglt_node", "rh_place_node", "episode_runner_node")
                for c in cmds if any(k + ".py" in a for a in c.argv)}
        assert f"ns:={side} " in argv["rh_aglt_node"] + " " and "stop_on_target:=true" in argv["rh_aglt_node"]
        cup = next(a.split(":=", 1)[1] for a in argv["rh_aglt_node"].split() if a.startswith("cup_topic:="))
        assert cup.startswith("/objects/") and cup.endswith("/live_pose")    # 10.12 A안: 쥘 때까지 FP++ live
        assert all(a in argv["rh_aglt_node"] for a in AGLT_LIVE_ARGS)
        if m is FAKE:
            assert any("fake_cup_pose_pub.py" in a for c in cmds if cup in c.argv for a in c.argv)
        else:
            assert any(cup in " ".join(c.argv) for c in cmds if "rh56f1_record" in " ".join(c.argv))
        assert f"ns:={side}_place" in argv["rh_place_node"]
        assert f"episode_topic:=/policy_control/{side}/episode" in argv["rh_place_node"]
        assert f"episode:=config/episodes/pick_place_{side}.yaml" in argv["episode_runner_node"]
        assert "auto_approve" not in argv["episode_runner_node"]                                 # 승인은 상황판에서
        by = {s.id: s for s in m.stages}
        assert {f"home_{side}", "cups", "cup_holders"} <= set(by[sid].needs)
        assert any(c.manual and "상황판" in c.note for c in cmds)
    assert REAL.stages[[s.id for s in REAL.stages].index(sid)].touches_real


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_episode_stage_checks_before_it_starts_anything(side):
    """실기는 고정 홀더 자세 파일까지 검사하고(fake 는 fake 홀더), 같은 이름 노드가 남아 있으면 띄우기 전에 멈춘다."""
    sid = f"episode_pick_place_{side}"
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        cmds = _cmds(m, book, sid)
        plan = next(i for i, c in enumerate(cmds) if any("episode_run.py" in a for a in c.argv))
        first_node = next(i for i, c in enumerate(cmds) if c.background and any("_node.py" in a for a in c.argv))
        assert plan < first_node and not cmds[plan].background
        assert ("--require-holders" in cmds[plan].argv) is (m is REAL)
        leftover = next(c for c in cmds if list(c.argv[:2]) == ["bash", "-lc"] and "ros2 node list" in c.argv[2])
        assert f"rh_aglt_node|rh_place_node)_{side}" in leftover.argv[2] and "episode_runner" in leftover.argv[2]


def test_the_real_episode_stage_records_what_the_release_threshold_needs():
    """첫 실기에서 놓음 문턱(손끝 · 관절 힘 · 손 목표)을 정하려면 손 bag 과 정책 · 실행기 상태가 같이 있어야 한다."""
    cmds = _cmds(REAL, REAL_BOOK, "episode_pick_place_right")
    start = next(c for c in cmds if "rh56f1_record.sh" in " ".join(c.argv) and "start" in " ".join(c.argv))
    line = " ".join(start.argv)
    for topic in ("/policy_control/status/rh_place_node_right", "/policy_control/status/episode_runner",
                  "/policy_control/joint_target", f"/objects/{ACTIVE_SIDES['right']}/pose", "/objects/cup_holder_1/pose"):
        assert topic in line
    # 10.09 이름마다 따로 — 이 에피소드 기록만 끝낸다(양팔 동시 단계가 서로 끄지 않게)
    assert any(list(c.argv[-2:]) == ["stop", "episode_pick_place_right"] and "rh56f1_record.sh" in " ".join(c.argv) for c in cmds)
    assert not any("rh56f1_record.sh" in " ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, "episode_pick_place_right"))


def test_head_pose_is_checked_and_aligned_before_fpp_and_holder_poses():
    """10.04 사용자 "fpp 진행 전에 자동으로 각도 확인하고 세팅을 제대로 맞춘 다음에 진행": 실기 head_home · cups · cup_holders 가
    카메라 외부 파라미터를 잰 머리 자세(head_pose)와 비교 · 맞춤(실기 실행 = --execute)을 FP++ · 홀더 노드보다 먼저 한다."""
    def idx(cmds, needle):
        return next(i for i, c in enumerate(cmds) if needle in " ".join(c.argv))
    tool = 'head_move.py'
    pairs = [("head_home", None), ("cups", "perception_ctl.py start")]
    if not GEN.REMOTE_PERCEPTION:
        pairs.append(("cup_holders", "cup_holder_pose_node.py"))
    for sid, then in pairs:
        cmds = _cmds(REAL, REAL_BOOK, sid)
        h = idx(cmds, tool)
        assert '--execute' in cmds[h].argv
        if then:
            assert h < idx(cmds, then), sid
        assert not any(tool in " ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, sid))
    home = _cmds(REAL, REAL_BOOK, 'head_home')
    assert len(home) == 2
    assert 'head_joint_publisher.py' in ' '.join(home[0].argv)
    assert str(REPO / GEN.HEAD_CONFIG) in home[0].argv
    assert home[0].background
    assert not any('head_home.py' in ' '.join(c.argv) for c in home)


@pytest.mark.parametrize("side", ["right", "left"])
def test_rehome_brings_the_arm_back_from_where_a_policy_stopped(side):
    """10.04 사용자: "tesollo 에는 rehome 이 있는데 여기는 없어서 오른손 정책이 컵을 못 잡고 끝났는데 초기 자세로 못 간다".
    pd 가 IDLE 이면 그 자리에서 engage(붙들고 있으면 건너뜀) → 손 펴기 → 실측에서 경로 계획(RRT) → 시작점 검사 → 재생 → 정착."""
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        by = {s.id: s for s in m.stages}
        assert f"rehome_{side}" in by and by[f"rehome_{side}"].needs == (f"pd_arm_{side}",)
        cmds = [" ".join(c.argv) for c in _cmds(m, book, f"rehome_{side}")]
        order = ["--only pd_engage --skip-engaged", "--only pd_hand_release", f"plan_rehome.py --side {side} --robot rh56f1",
                 "check_path_start.py", "replay_to_pd.py", "--only pd_goto_home"]
        idx = [next(i for i, a in enumerate(cmds) if o in a) for o in order]
        assert idx == sorted(idx), cmds
        assert all("--execute" in cmds[i] for i in (idx[0], idx[1], idx[4], idx[5]))
        assert f"rehome_{side}.npz" in cmds[idx[3]] and f"rehome_{side}.npz" in cmds[idx[4]]
    assert by[f"rehome_{side}"].touches_real is False                     # fake
    assert {s.id: s for s in REAL.stages}[f"rehome_{side}"].touches_real


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_lone_aglt_stage_follows_the_live_cup_until_the_grasp(side):
    """10.12 사용자(A안): 실기를 sim 에 맞춘다 — 쥘 때까지 FP++ live(끊기면 마지막 값, 10.04 의 '컵 자세가 없다' 멈춤 대신),
    쥔 뒤는 파지 + 100 ms 뒤 찍힌 프레임 시각의 손바닥 FK(가려서 안 오면 0.7 s 에 파지 시작 FK)."""
    for m, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        cmds = _cmds(m, book, f"policy_aglt_{side}")
        node = next(" ".join(c.argv) for c in cmds if "rh_aglt_node.py" in " ".join(c.argv))
        assert all(a in node for a in AGLT_LIVE_ARGS), node
        assert "cup_latch:=true" not in node and "cup_static:=true" not in node
        live = next(a.split(":=", 1)[1] for a in node.split() if a.startswith("cup_topic:="))
        if m is FAKE:                                                      # fake 도 그 토픽을 낸다
            assert any("fake_cup_pose_pub.py" in " ".join(c.argv) and live in c.argv for c in _cmds(m, book, "cups"))
        else:
            assert any(live in " ".join(c.argv) for c in cmds if "rh56f1_record" in " ".join(c.argv))   # bag 에 싣는다


@pytest.mark.parametrize("side", ["right", "left"])
def test_each_arms_policy_stage_records_under_its_own_name_and_stops_only_that(side):
    """★10.09 실기: 양팔 정책 동시 — 왼팔 기록 시작이 오른팔 기록을 끄고, 오른팔 끝이 왼팔 기록을 껐다."""
    cmds = [" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, f"policy_aglt_{side}")]
    assert any(f"rh56f1_record.sh start aglt_{side}" in c and f"HAND_SIDES='{side}'" in c for c in cmds)
    assert any(c.endswith(f"rh56f1_record.sh stop aglt_{side}") for c in cmds)


@pytest.mark.parametrize("side", [s for s in ("right", "left") if s in ACTIVE_SIDES])
def test_each_cup_can_be_recognized_again_on_its_own(side):
    """★10.13 사용자 실기: "다음부턴 개별로 하고, 둘중 하나라도 인지되게 해야함. 개별로 다시 할수 있게 해야해".
    팔마다 그 팔의 컵 하나만 다시 찍는 선택 단계 — 차례(선택 묶음)에 끼지 않고 cups 뒤에 언제든 누른다."""
    cup = ACTIVE_SIDES[side]
    st = next(s for s in REAL.stages if s.id == f"cups_{side}")
    assert st.group == "recognize" and st.needs == ("cups",) and st.skippable and st.touches_real
    assert next(g for g in REAL.groups if g.id == "recognize").optional
    argv = [c.argv for c in _cmds(REAL, REAL_BOOK, f"cups_{side}")]
    rescan = [a for a in argv if any("fpp_rescan.py" in x for x in a)]
    assert len(rescan) == 1 and cup in rescan[0]
    assert ("--status-host" in rescan[0]) == GEN.REMOTE_PERCEPTION
    assert not [o for o in ACTIVE_SIDES.values() if o != cup and o in rescan[0]]     # 이 컵 하나만
    start = [a for a in argv if any("perception_ctl.py" in x for x in a)]
    assert start and set(ACTIVE_SIDES.values()) <= set(start[0])   # start 는 넘긴 물체 밖을 내리므로 컵 전부(떠 있으면 그대로)
    assert not any(s.id == f"cups_{side}" for s in FAKE.stages if s.touches_real)


def test_the_both_cup_stage_passes_with_one_cup_unless_asked_for_all():
    """★10.13 cups 단계의 start · rescan 은 기본(하나라도) — --require-all 을 붙이지 않는다."""
    flat = [x for c in _cmds(REAL, REAL_BOOK, "cups") for x in c.argv]
    assert "--require-all" not in flat
    assert any("fpp_rescan.py" in x for x in flat) and any("perception_ctl.py" in x for x in flat)


def test_episodes_with_holders_read_the_holder_file_this_station_writes():
    """10.13: 놓기 에피소드의 holder_poses = 미션 cup_holders 단계가 쓰는(가져오는) 파일. 다른 자리 값을 쓰지 않는다."""
    import yaml as _y
    used = {}
    for f in sorted((REPO / "config/episodes").glob("*.yaml")):
        raw = _y.safe_load(f.read_text(encoding="utf-8")) or {}
        if raw.get("holders"):
            used[f.name] = raw.get("holder_poses")
    assert used and set(used.values()) == {GEN.HOLDER_POSES}, used


def test_the_remote_vision_recorder_runs_on_the_perception_pc_venv():
    """10.13 vision-3090 시스템 python3 는 scipy 가 깨져(~/.local numpy 2.2) 기록기가 _base_from_camera 에서 죽는다."""
    if not GEN.REMOTE_PERCEPTION:
        return
    recs = [" ".join(c.argv) for st in REAL.stages for c in _cmds(REAL, REAL_BOOK, st.id) if "vision_recorder.py" in " ".join(c.argv)]
    assert recs and all(f"{GEN.REMOTE_PY} scripts/nodes/vision_recorder.py" in r for r in recs)


def test_shutdown_handles_partial_connections_and_orders_policy_perception_pd_hands_arms():
    for mission, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        assert not next(s for s in mission.stages if s.id == 'shutdown').needs
        cmds = _cmds(mission, book, 'shutdown')
        positions = {key: i for i, c in enumerate(cmds) for key in c.stop}
        policy = positions[f'policy_aglt_right#{GEN.AGLT_NODE_IDX[mission is REAL]}']
        perception = positions['cups#1' if mission is REAL else 'cups#0']
        pd = positions['pd_arm_right#1']
        hand = positions['hand_right#1']
        arm = positions['drivers#2' if mission is REAL else 'drivers#0']
        assert policy < perception < pd < hand < arm
        for stage in mission.stages:
            for i, command in enumerate(_cmds(mission, book, stage.id)):
                if command.background:
                    assert f'{stage.id}#{i}' in positions, f'unmanaged at shutdown: {stage.id}#{i}'


def test_head_control_is_a_separate_approved_motion_using_the_head_settle_tool():
    for mission, book in ((REAL, REAL_BOOK), (FAKE, FAKE_BOOK)):
        st = next(s for s in mission.stages if s.id == 'head_control')
        assert st.group == 'motion' and st.skippable
        cmds = _cmds(mission, book, 'head_control')
        if mission is REAL:
            assert st.touches_real
            assert any('head_move.py' in ' '.join(c.argv) and '--execute' in c.argv for c in cmds)
        else:
            assert not st.touches_real
            assert not any('head_home.py' in ' '.join(c.argv) for c in cmds)


def test_shutdown_releases_neck_torque_after_perception_and_before_pd():
    cmds = _cmds(REAL, REAL_BOOK, 'shutdown')
    head = next(i for i, c in enumerate(cmds) if 'head_home.py' in ' '.join(c.argv) and '--release' in c.argv)
    perception = next(i for i, c in enumerate(cmds) if 'cups#1' in c.stop)
    pd = next(i for i, c in enumerate(cmds) if 'pd_arm_right#1' in c.stop)
    assert perception < head < pd
    assert '--execute' in cmds[head].argv
    assert not any('--release' in c.argv for c in _cmds(FAKE, FAKE_BOOK, 'shutdown'))


def test_head_has_single_serial_owner_and_dynamic_camera_transform():
    home = _cmds(REAL, REAL_BOOK, 'head_home')
    assert any(c.background and 'head_joint_publisher.py' in ' '.join(c.argv) for c in home)
    assert any('head_move.py' in ' '.join(c.argv) and '--execute' in c.argv for c in home)
    control = _cmds(REAL, REAL_BOOK, 'head_control')
    assert any('head_move.py' in ' '.join(c.argv) for c in control)
    cups = _cmds(REAL, REAL_BOOK, 'cups')
    converter = next(c for c in cups if 'object_pose_node.py' in ' '.join(c.argv))
    assert '--head-joint-topic' in converter.argv and '/head/joint_states' in converter.argv
    shutdown = _cmds(REAL, REAL_BOOK, 'shutdown')
    stop = next(i for i, c in enumerate(shutdown) if 'head_home#0' in c.stop)
    release = next(i for i, c in enumerate(shutdown) if '--release' in c.argv)
    assert stop < release

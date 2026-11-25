"""RH56F1 로봇(arm4090) 미션 — 실기 · fake 가 한 정의에서 나오고, 손 명령은 승인 단계에서만 나간다. 09.29 사용자.

"sim2real 과 robot_control 쪽에서 rh56f1 제어 part 연결" · 손마다 개별 포트 · USB RS485 / CANFD 를 상황에 따라.
"""
from __future__ import annotations

import shlex
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
    try:
        return commands_for(book, m, sid, repo=REPO, execute=True)
    except KeyError as exc:
        if 'policy.yaml' in str(exc) and 'No such file' in str(exc):
            pytest.skip(f'삭제된 선택 정책의 명령 검사: {exc}')
        raise


def test_the_committed_missions_come_from_the_generator():
    r = subprocess.run([sys.executable, str(REPO / "scripts/ops/make_rh56f1_missions.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("side", ["right", "left"])
def test_student_approach_does_not_abort_on_fpp_dropout(side):
    cmds = _cmds(REAL, REAL_BOOK, f"episode_grasp_s_{side}")
    node = next(c.argv for c in cmds if any("rh_aglt_student_node.py" in a for a in c.argv))
    assert "device:=cuda" in node
    assert [a for a in node if a.startswith("cup_dropout_stop_s:=")] == ["cup_dropout_stop_s:=0.0"]
    assert "cup_hold_last:=true" in node
    teacher = _cmds(REAL, REAL_BOOK, "episode_cocktail_e2e")
    teacher_nodes = [c.argv for c in teacher if any(a.endswith("rh_aglt_node.py") for a in c.argv)]
    assert teacher_nodes
    assert all("cup_dropout_stop_s:=1.0" in a for a in teacher_nodes)


@pytest.mark.parametrize("book", ["real", "fake"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_mask_student_episode_starts_the_mask_node_before_the_policy_node(side, book):
    raw, b = (REAL, REAL_BOOK) if book == "real" else (FAKE, FAKE_BOOK)
    cmds = _cmds(raw, b, f"episode_grasp_m_{side}")
    # ★11.11 카메라가 다른 PC 면 깊이 · 마스크 노드는 ssh 한 줄 안에 있다(상대 경로) — 이름 · 묶음은 문자열로 찾는다
    has = lambda c, k: any(k in a for a in c.argv)  # noqa: E731
    at = {k: next(i for i, c in enumerate(cmds) if has(c, k))
          for k in ("check_student_mask_A.py", "head_depth_node.py", "head_mask_node.py", "rh_aglt_student_node.py")}
    assert at["check_student_mask_A.py"] < at["head_mask_node.py"] < at["rh_aglt_student_node.py"]
    assert at["head_depth_node.py"] < at["rh_aglt_student_node.py"]
    assert has(cmds[at["head_mask_node.py"]], f"deploy/policies/rh56f1/aglt/{side}_student1105mA")
    assert sum(has(c, "head_mask_node.py") for c in cmds) == 1
    # ★11.10 grasp_s 도 1105MA(마스크 학생)로 바꿨다(사용자 결정) — 마스크 노드를 한 번, 같은 묶음으로 띄운다
    s_cmds = _cmds(raw, b, f"episode_grasp_s_{side}")
    s_mask = [c for c in s_cmds if has(c, "head_mask_node.py")]
    assert len(s_mask) == 1 and has(s_mask[0], f"deploy/policies/rh56f1/aglt/{side}_student1105mA")


@pytest.mark.parametrize('partial', [False, True])
def test_generator_keeps_missing_policy_unavailable_without_blocking_other_stages(monkeypatch, partial):
    from types import SimpleNamespace
    from policy_control import policy_registry as R
    policy = GEN._episode('grasp_rg3_right').policies['rh_aglt_r'].policy
    entries = [SimpleNamespace(id=policy, contract='')] if partial else []
    monkeypatch.setattr(R, 'scan', lambda *a, **kw: entries)
    commands = GEN._episode_run('grasp_rg3_right', True)
    argv = [arg for command in commands for arg in command.get('argv', [])]
    assert f'contract:={{repo}}/deploy/policies/{policy}/rh_aglt_contract.json' in argv
    assert '--plan' in argv  # Runtime availability still rejects the missing bundle before any node starts.


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
                # ★10.30 arm_v2: ssh 로 감싼 명령은 도구 경로가 한 원소 안의 낱말(저 PC 의 저장소 상대 경로)이다 — 같은 저장소라 여기서 읽는다
                # ★11.20 act_tilt: 셸 안에 기록기(ssh) · 브리지 · head_move 가 같이 있다 — 브리지가 --execute 를 받는다
                tool = next((t.strip('"') for a in spec.argv for t in a.split() if t.strip('"').endswith("act_tilt_bridge.py")), None) \
                    or next(t.strip('"') for a in spec.argv for t in a.split() if t.strip('"').endswith(".py"))
                if tool.startswith("$HOME/"):   # ★11.19 act_tilt: 다른 저장소(motion_acq) 도구 — 이 PC 에 있을 때만 읽는다
                    path = Path.home() / tool[len("$HOME/"):]
                    if not path.is_file():
                        continue
                else:
                    path = Path(tool.replace("{repo}", str(REPO)))
                    if not path.is_absolute():
                        path = REPO / path
                text = path.read_text(encoding="utf-8")
                assert all(f'"{f}"' in text for f in spec.execute_args if f.startswith("--")), (s.id, tool)


def test_the_driver_launcher_reads_the_port_file_and_refuses_one_port_for_two_hands():
    """10.02: 손은 EtherCAT(손 하나 = NIC 하나). rs485 · canfd 는 벤더 드라이버 경로로 남는다."""
    import rh56f1_driver as D
    cfg = yaml.safe_load((PC / "config" / "rh56f1_ports.yaml").read_text())
    assert cfg["right"]["transport"] == cfg["left"]["transport"] == "ethercat"
    assert (cfg["right"]["ifname"], cfg["left"]["ifname"]) == ("eno1.10", "eno1.20")   # 10.22 CBS220 VLAN 10 · 20
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
        assert [p["id"] for p in rh["profiles"]] == ["rh56f1_real", "rh56f1_real_cam5090", "rh56f1_fake"]
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
    host, ext = GEN.PERCEPTION_HOST, GEN.FPP_EXTRINSICS     # ★11.17 FP++ 쪽은 head_view_check 가 쓴 live
    by = {s.id: s for s in REAL.stages}
    assert by["cups"].needs == ("head_view_check",) and by["head_home"].touches_real and by["cups"].touches_real
    head = _cmds(REAL, REAL_BOOK, "head_home")
    assert any(Path(GEN.HEAD_CONFIG).name in " ".join(c.argv) for c in head)
    cups = [" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, "cups")]
    sys.path.insert(0, str(REPO / "scripts"))
    side = [ACTIVE_SIDES[s] for s in ("right", "left") if s in ACTIVE_SIDES]
    # ★10.28 칵테일 분홍 — 에피소드가 찍는 실물도 띄운다(팔 물체 묶음 밖은 분홍만 든 묶음 source240_pink 하나)
    # ★11.03 사용자 "shaker는 현재 orange" — 묶음의 나머지 색은 넣지 않는다(GPU 24 GB)
    objs = " ".join(dict.fromkeys([*side, "source240_pink"]))
    assert not {"shaker_c_blue", "shaker_c_pink"} & set(objs.split())
    from perception_launcher_core import FIRST_ROUND_S
    assert GEN._start_wait() >= len(GEN._groups()) * FIRST_ROUND_S          # 11.03 묶음을 하나씩 — 묶음마다 첫 회차 상한
    for want in (f"perception_launcher_node.py --host {host}", "fpp_pose_rx.py", "object_pose_node.py --camera-extrinsics",
                 "--camera-extrinsics", ext):
        assert any(want in a for a in cups), want
    # ★11.05 사용자: cups = FP++ 드라이버만 — 물체는 에피소드 첫 명령(덧붙이기)과 setting 탭 "인지 적용" 이 띄운다
    assert not any("perception_ctl.py start" in a or "fpp_rescan.py" in a for a in cups)
    ep = _cmds(REAL, REAL_BOOK, "episode_cocktail_e2e")
    at = next(i for i, c in enumerate(ep) if "perception_ctl.py start --add " in " ".join(c.argv))
    first = " ".join(ep[at].argv)
    assert first.endswith("--require-all")
    # ★11.05 확보는 읽기 전용 시작 검사 뒤, GPU · 수동 확인 사이 — 시작 검사가 막히면 컨테이너를 띄우지 않는다
    assert [c.note.startswith("시작 검사") for c in ep[:at - 1]] == [True, True] and ep[at - 1].manual and "nvidia-smi" in " ".join(ep[at - 1].argv)
    assert ep[at + 1].manual and "기울기 < 3°" in ep[at + 1].note
    assert set(first.split("--add ")[1].split(" --wait")[0].split()) == set(objs.split())   # 화면 선택과 상관없이 물체 셋
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
                 "--only pd_goto_home", "--only pd_hand_home", f"trigger.py hand/tip_zero --side {side}"]   # 11.04 손끝 영점
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


def test_view_check_re_solves_the_extrinsics_into_live_between_head_home_and_cups():
    """11.03 사용자 "sim 과 real 의 카메라 시야가 같아야 의미가 있다" · ★11.17 "안되면 2번으로 자동화해서 FP++ 가 문제없게":
    head_home 뒤 · cups 앞, 카메라를 켠 뒤 카메라 PC 에서 --auto-extrinsics(실행이면 --execute 로 목이 camset 으로)로
    base(m1117)를 읽기만 하고 live 를 쓴다. 원격이면 이 PC 의 live 를 먼저 지우고 성공한 뒤에만 가져온다."""
    for m in (REAL, FAKE):
        by = {s.id: s for s in m.stages}
        ids = [s.id for s in m.stages]
        assert by["head_view_check"].needs == ("head_home",) and by["cups"].needs == ("head_view_check",)
        assert ids.index("head_home") < ids.index("head_view_check") < ids.index("cups")
        assert by["head_view_check"].skippable
    assert {s.id: s for s in REAL.stages}["head_view_check"].touches_real
    cmds = _cmds(REAL, REAL_BOOK, "head_view_check")
    flat = [" ".join(c.argv) for c in cmds]
    assert not any(c.background for c in cmds) and len(cmds) == 2
    assert "camera_up.sh" in flat[0]
    check = flat[1]
    ext = GEN.CAMERA_EXTRINSICS if GEN.REMOTE_PERCEPTION else f"{REPO}/{GEN.CAMERA_EXTRINSICS}"   # 로컬 argv 는 {repo} 가 풀린다
    assert f"head_view_feedback.py --auto-extrinsics --extrinsics {ext}" in check and cmds[1].argv[-1] == "--execute"
    assert GEN.FPP_EXTRINSICS == "config/global_camera_extrinsics_arm4090_live.yaml" != GEN.CAMERA_EXTRINSICS
    if GEN.REMOTE_PERCEPTION:
        script = cmds[1].argv[2]
        assert list(cmds[1].argv[:2]) == ["bash", "-lc"] and script.startswith(f"L='{REPO}/{GEN.FPP_EXTRINSICS}'; ")
        assert script.index('rm -f "$L"') < script.index("--auto-extrinsics") < script.index("live_cli") < script.index('mv "$L.tmp" "$L"')
        assert '$*"' in script                                  # 콘솔의 --execute 가 원격 명령 끝으로 간다
    assert not any("head_view_feedback" in " ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, "head_view_check"))
    sys.path.insert(0, str(REPO / "scripts"))
    import head_view_feedback as hv
    assert hv.DEFAULT_EXTRINSICS == REPO / GEN.CAMERA_EXTRINSICS


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
    rel = node.replace("{repo}/", "").replace(f"{REPO}/", "")      # 로컬 인지(10.27 5090)는 {repo} 가 풀린 절대 경로
    assert f"--out {GEN.HOLDER_POSES}" in rel
    assert f"--extrinsics {GEN.CAMERA_EXTRINSICS}" in rel
    assert cmds.index(bg[0]) == 4                                   # shutdown 의 cup_holders#4(앞에 안내 · 확인 · 머리 맞춤 · 카메라)
    assert "cup_holders#4" in stops
    assert "컵을 놓기 전" in cmds[0].note and cmds[0].manual and cmds[1].manual   # ★11.11 컵 없이 잰다는 안내가 맨 앞
    status = next(c for c in cmds if "/cup_holders/status" in " ".join(c.argv))
    assert cmds[0].manual and status.manual
    if GEN.REMOTE_PERCEPTION:
        host = GEN.PERCEPTION_HOST
        for c in (cmds[3], bg[0], status):                          # 카메라 · 노드 · 확인은 카메라 PC 에서
            assert list(c.argv[:4]) == ["ssh", "-tt", "-o", "BatchMode=yes"] and c.argv[4] == host
        assert f"{GEN.REMOTE_PY} scripts/nodes/cup_holder_pose_node.py" in node
    # ★11.11 사용자: 마지막 = 홀더 확정(한 번 재기 — 원격이면 파일 · 기록을 이 PC 로 가져온다)
    assert "cup_holder_pose_node.py --write --timeout-s" in " ".join(cmds[-1].argv) and cmds[-2] is status
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
        assert f"policy_aglt_{side}" in by[f"policy_place_{side}"].needs and "cup_holders" not in by[f"policy_place_{side}"].needs  # ★11.17 홀더 고정
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
        assert {f"home_{side}", "cups"} <= set(by[sid].needs) and "cup_holders" not in by[sid].needs   # ★11.03 홀더 고정 — 저장 파일
        if m is REAL:
            plan = next(c.argv for c in cmds if "episode_run.py" in " ".join(c.argv) and "--plan" in c.argv)
            assert "--require-holders" in plan                     # 파일이 없거나 홀더가 빠지면 시작 검사가 이유를 찍고 막는다
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
    # ★11.22 끝은 bag_gate 의 회 끝 + 30 s 마무리 · 앞 자르기를 기다린 뒤 stop
    assert any("rh56f1_record.sh stop episode_pick_place_right" in " ".join(c.argv)
               and "bag_gate.py wait --name episode_pick_place_right" in " ".join(c.argv) for c in cmds)
    # ★11.22 녹화량 줄이기: 에피소드 기록은 SLIM, 회 구간 문지기가 같은 시작 명령으로 다시 열 수 있다
    gate = next(c for c in cmds if "bag_gate.py" in " ".join(c.argv) and "watch" in c.argv)
    assert gate.background and "RH56F1_RECORD_SLIM=1" in line
    assert gate.argv[gate.argv.index("--gate") + 1] == "/policy_control/status/episode_runner_right"
    assert gate.argv[gate.argv.index("--start-cmd") + 1] == start.argv[-1]
    assert not any("rh56f1_record.sh" in " ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, "episode_pick_place_right"))


def test_head_pose_is_checked_and_aligned_before_fpp_and_holder_poses():
    """10.04 사용자 "fpp 진행 전에 자동으로 각도 확인하고 세팅을 제대로 맞춘 다음에 진행": 실기 head_home · cups · cup_holders 가
    카메라 외부 파라미터를 잰 머리 자세(head_pose)와 비교 · 맞춤(실기 실행 = --execute)을 FP++ · 홀더 노드보다 먼저 한다."""
    def idx(cmds, needle):
        return next(i for i, c in enumerate(cmds) if needle in " ".join(c.argv))
    tool = 'head_move.py'
    pairs = [("head_home", None), ("cups", None)]   # ★11.05 cups = 드라이버만 — 물체 촬영은 그 뒤 단계(에피소드 · 인지 적용)
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
    assert len(home) == (3 if GEN.REMOTE_HEAD else 2)      # ★10.30 arm_v2: 드라이버 · 랜 토픽 확인 · 맞춤
    assert 'head_joint_publisher.py' in ' '.join(home[0].argv)
    if GEN.REMOTE_HEAD:
        assert list(home[0].argv[:4]) == ["ssh", "-tt", "-o", "BatchMode=yes"] and home[0].argv[4] == GEN.HEAD_HOST
        assert GEN.HEAD_CONFIG in ' '.join(home[0].argv) and "ROS_LOCALHOST_ONLY=0" in ' '.join(home[0].argv)
    else:
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
                 "check_path_start.py", "replay_to_pd.py", "--only pd_goto_home", f"trigger.py hand/tip_zero --side {side}"]
        idx = [next(i for i, a in enumerate(cmds) if o in a) for o in order]
        assert idx == sorted(idx), cmds
        assert all("--execute" in cmds[i] for i in (idx[0], idx[1], idx[4], idx[5]))
        assert f"rehome_{side}.npz" in cmds[idx[3]] and f"rehome_{side}.npz" in cmds[idx[4]]
        assert cmds[-1] == cmds[idx[6]] and "--execute" in cmds[-1]   # 11.04 끝(빈손 · 홈 정착 뒤)에 손끝 영점
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
    # ★11.05 덧붙이기 — 다른 물체 컨테이너는 내리지 않으니 이 팔 컵 하나만 넘긴다
    assert len(start) == 1 and list(start[0][2:4]) == ["start", "--add"] and [o for o in ACTIVE_SIDES.values() if o in start[0]] == [cup]
    assert not any(s.id == f"cups_{side}" for s in FAKE.stages if s.touches_real)


def test_stage_object_starts_add_and_never_take_down_other_objects():
    """★11.05 사용자: 에피소드 · cups_<팔> 이 띄우는 물체는 덧붙이기(--add) — 목록 밖 컨테이너를 내리는 맞추기는 setting 탭 "인지 적용" 만 한다."""
    starts = [list(c.argv) for st in REAL.stages if st.id.startswith(("episode_", "cups"))
              for c in _cmds(REAL, REAL_BOOK, st.id) if "perception_ctl.py" in " ".join(c.argv) and "start" in c.argv]
    assert starts and all(a[a.index("start") + 1] == "--add" and "--require-all" in a for a in starts)


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
    rec = GEN._vision_record("episode_x", ["source240_blue"], 15.0)
    assert f"{GEN.REMOTE_PY} scripts/nodes/vision_recorder.py" in " ".join(map(str, rec["argv"]))


def test_video_recording_and_report_media_are_off_by_default():
    """★10.20 사용자 "영상은 필요 없음. 일단 기본 off" — 보고서 영상 정리가 팔 작업을 6 분 넘게 붙잡아 rehome · tilt_now 승인이 막혔다."""
    assert GEN.VISION_RECORD is False
    argvs = [" ".join(c.argv) for st in REAL.stages for c in _cmds(REAL, REAL_BOOK, st.id)]
    assert not [a for a in argvs if "vision_recorder.py" in a or "episode_media.py" in a]


def test_chain_episodes_bag_the_head_camera_on_the_camera_pc_and_stop_it():
    """★11.16 사용자 "체인 영상 bag -> 4090에서 저장. 최적화 필요 fp++ 부담 안되게" — 체인 · 놓기 에피소드만, 카메라 PC 에서
    기록기가 압축(드라이버 압축 안 켬) · 낮은 우선순위 · ssh HUP 에서 떼어 냄, 에피소드 정지와 shutdown 이 끈다."""
    if not GEN.REMOTE_PERCEPTION:
        pytest.skip("카메라가 로봇 PC 인 사이트 — 체인 영상은 rh56f1_record.sh CAMERA 묶음")
    shut = {k for c in REAL_RAW["run"]["shutdown"] for k in c.get("stop") or []}
    for sid, cmds in REAL_RAW["run"].items():
        recs = [i for i, c in enumerate(cmds) if "camera_bag_recorder.py" in " ".join(c.get("argv") or [])]
        if sid in ("act_tilt", "act_tilt_dry"):   # ★11.20 브리지와 같은 셸에서 띄우고 끝나면 kill — 정지 단계 없음
            assert len(recs) == 1 and "kill $B; " in cmds[recs[0]]["argv"][2], sid
            continue
        if sid.removeprefix("episode_") not in GEN.CHAIN_VIDEO_EPISODES or sid.removeprefix("episode_") in GEN.CHAIN_CAMERA_BAG_OFF:
            assert not recs, sid   # ★11.25 사용자 "일단 없어도 되게" — cocktail_e2e 는 카메라 bag 을 띄우지 않는다
            continue
        assert len(recs) == 1, sid
        argv = cmds[recs[0]]["argv"]
        # ★11.25 사용자 "영상이 안 남더라도 앞으론 문제없이": 로컬 셸이 ssh 를 감싸 끝나도(rc 2 · 255) 정지 신호까지 남는다
        assert argv[:2] == ["bash", "-c"] and argv[2].endswith("trap 'exit 0' HUP TERM INT; while :; do sleep 1; done"), sid
        toks = shlex.split(argv[2])
        argv = toks[:toks.index(";")]
        assert argv[:2] == ["ssh", "-tt"] and GEN.PERCEPTION_HOST in argv and cmds[recs[0]]["background"]
        sh = argv[-1]
        assert "--encode-raw --hz 5 " in sh and " ".join(GEN.CHAIN_CAMERA_BAG_TOPICS) in sh   # 원본 구독 · 기록기 압축
        # ★11.17 s2r:vision "깊이도" — 컬러 + 정렬 깊이 + 두 camera_info, 다른 토픽(/head_depth 등)은 싣지 않는다
        assert sh.count("/compressedDepth") == 1 and sh.count("camera_info") == 2 and "/head_depth" not in sh
        assert "setsid nice -n 19 ionice -c2 -n7 python3" in sh and "{" not in sh and "set +e; trap 'kill -TERM $P; wait $P'" in sh   # set +e: 끊긴 wait 가 변환 전에 셸을 끝내지 않게(11.18)
        key = f"{sid}#{recs[0]}"
        assert any(key in (c.get("stop") or []) for c in cmds) and key in shut, key
        # ★11.18 사용자: 경로 F 만 상자 영상을 카메라 PC 바탕화면에 <컵>_<날짜>_<시작 시:분>.mp4 로 복사
        cup = GEN.FPP_VIDEO_CUP.get(sid.removeprefix("episode_"))
        assert ("foundation pose camera" in sh) == (cup is not None), sid
        if cup:
            assert f'cp $O/camera_bbox.mp4 "$F.mp4"' in sh and f'/{cup}_$T"' in sh and "T=$(date +%y%m%d_%H:%M)" in sh
        assert sh.index("set +e") < sh.index("wait $P; ") < sh.index("camera_bag_to_mp4.py")   # 끊긴 wait 뒤에도 변환 · 복사까지
    assert GEN.FPP_VIDEO_CUP["grasp_f_right"] == "src240_blue" and GEN.FPP_VIDEO_CUP["grasp_f_left"] == "shaker_orange"
    assert set(GEN.FPP_VIDEO_CUP) == {s.removeprefix("episode_") for s in REAL_RAW["run"] if s.startswith("episode_grasp_f_")}


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


@pytest.mark.skipif(_generator().ROBOT_PC != "5090", reason="★11.11 지금 스테이션은 5090_arm4090(아래 arm_v2 테스트가 본다)")
def test_the_5090_station_uses_the_camset_pose_where_all_three_objects_are_in_view():
    """10.28 사용자 "미션 연결 진행" — shaker · 파랑 · 분홍이 한 화면에 드는 목 자세(camset)와 그 자세에서 잰 외부 파라미터.
    ★10.30 arm_v2(5090 + 4090): 머리 · 카메라 조립체는 그대로(USB 만 4090 으로)라 같은 자세 · 같은 파일을 쓴다.
    ★11.03 사용자: 카메라 · 목 · FP++ 를 다시 5090 에(사이트 arm_v2 유지) — 같은 자세 · 같은 파일, 목 포트는 사이트 head_port."""
    import yaml as _y
    site = _y.safe_load((REPO / "deploy/policy_control/config/sites/arm_v2.yaml").read_text(encoding="utf-8"))
    assert GEN.ROBOT_PC == "5090"
    assert not GEN.REMOTE_HEAD and not GEN.REMOTE_PERCEPTION and GEN.STATION["pose_dest"] == ""
    assert GEN._head_config["port"] == site["head_port"]
    assert site["hosts"]["arm4090"]["ssh"] not in (REPO / GEN.MISSION_REL).read_text(encoding="utf-8")   # ★11.11 5090 = _cam5090 파일
    assert GEN.CAMERA_EXTRINSICS == "config/global_camera_extrinsics_arm4090_m1117.yaml"    # ★11.17 테이블 마커 판(목 설정은 h2048 그대로)
    assert GEN.HEAD_CONFIG == "config/head_camset_rh56f1_h2048.yaml"


def test_arm_v2_runs_the_head_driver_and_release_on_the_vision_pc_and_poses_come_over_the_hub_lan():
    """★10.30 사용자 "4090 : 카메라+다이나셀 동작 및 fpp 관련 / 5090 : 로봇 제어 및 정책 실행 관련" — 목 U2D2 · 카메라 · FP++ 는
    4090, 미션은 5090 에서 ssh 로 띄운다. 자세 UDP 는 허브 랜(사이트 arm_v2 hosts.control.ip)으로, 목 포트는 사이트 hosts.vision 과 같다."""
    if not GEN.REMOTE_HEAD:
        pytest.skip("목이 로봇 PC 에 있는 스테이션")
    import yaml as _y
    site = _y.safe_load((REPO / "deploy/policy_control/config/sites/arm_v2.yaml").read_text(encoding="utf-8"))
    assert GEN.HEAD_HOST == GEN.PERCEPTION_HOST == site["hosts"]["arm4090"]["ssh"]
    assert GEN.STATION["pose_dest"] == site["hosts"]["control"]["ip"]
    assert GEN._head_config["port"] == site["head_port"]           # 같은 U2D2(by-id)
    home = _cmds(REAL, REAL_BOOK, "head_home")
    assert "/head/joint_states" in " ".join(home[1].argv) and not home[1].background     # 랜 토픽 확인이 맞춤보다 먼저
    assert "head_move.py" in " ".join(home[2].argv) and "--execute" in home[2].argv
    shutdown = _cmds(REAL, REAL_BOOK, "shutdown")
    release = next(c for c in shutdown if "--release" in c.argv)
    assert release.argv[4] == GEN.HEAD_HOST and "head_home.py" in release.argv[5]
    assert release.argv.index("--release") < release.argv.index("--execute")      # 콘솔이 붙인 execute_args 가 ssh 명령 끝에 간다
    cups = [" ".join(c.argv) for c in _cmds(REAL, REAL_BOOK, "cups")]
    assert any(f"perception_launcher_node.py --host {GEN.PERCEPTION_HOST} --pose-dest {GEN.STATION['pose_dest']}" in a for a in cups)
    assert not any("ssh" in a for a in (" ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, "head_home")))


def test_arm_v2_reaches_the_vision_pc_over_the_hub_lan_only_and_checks_it_in_preflight():
    """★10.30 사용자 "5090과 4090은 ssh가 아니라 lan포트 연결로 통신해야함" · 도메인은 motion_acq 와 분리(config/ros_domain.env) ·
    "4 번 수정": 저 PC 에 손 마스터가 없어야 한다. ssh 별칭은 사이트 hosts.vision.ssh(11.03 부터 hosts.arm4090.ssh, 랜 전용 arm4090-lan, 5090 ~/.ssh/config 가
    192.168.0.40 · 키 s2r_lan 로 간다)이고 tailscale 별칭 arm4090 이 아니다. preflight 가 랜 ssh · 같은 커밋 · 손 마스터 없음을 먼저 본다."""
    if not GEN.REMOTE_PERCEPTION:
        pytest.skip("인지가 로봇 PC 에 있는 스테이션")
    assert GEN.PERCEPTION_HOST == GEN.HEAD_HOST == "arm4090-lan"
    pre = _cmds(REAL, REAL_BOOK, "preflight")
    lan, master = pre[0], pre[1]
    assert list(lan.argv[:2]) == ["bash", "-lc"]
    assert f"ssh -o BatchMode=yes -o ConnectTimeout=5 {GEN.PERCEPTION_HOST} 'cd {GEN.REMOTE_SIM2REAL} && git rev-parse HEAD'" in lan.argv[2]
    assert f"git -C {REPO} rev-parse HEAD" in lan.argv[2]                       # {repo} 가 이 PC 의 저장소로 치환됐다
    assert "|| echo '커밋 다름" in lan.argv[2]                                 # ★11.16 커밋 차이는 막지 않는다(4090 = FP++ 전용)
    assert list(master.argv[:5]) == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"] and master.argv[5] == GEN.PERCEPTION_HOST
    assert "pgrep" in master.argv[6] and "exit 1" in master.argv[6] and ".py" not in master.argv[6]   # 그림이 노드 상자로 읽지 않게
    home = _cmds(REAL, REAL_BOOK, "head_home")
    assert f"도메인 {GEN.REAL_DOMAIN}" in home[1].note and "126" not in home[1].note
    for c in pre + home:
        assert "arm4090 " not in " ".join(c.argv) + " "                           # tailscale 별칭이 argv 에 없다
    assert not any("ssh" in " ".join(c.argv) for c in _cmds(FAKE, FAKE_BOOK, "preflight"))


def test_the_station_head_goes_to_the_pose_its_camera_file_was_measured_at():
    """목 드라이버의 기준 자세(HEAD_CONFIG 목표) = 카메라 외부 파라미터를 잰 자세(head_pose) — 다르면 FP++ 좌표가 틀린다."""
    from head_home import load_head_home
    from head_pose_check import load_head_pose
    from head_position_hold_node import deg_to_tick
    hp = load_head_pose(REPO / GEN.CAMERA_EXTRINSICS)
    if hp is None:
        pytest.skip("hand-eye 스테이션 — 정적 자세가 없다")
    cfg = load_head_home(REPO / GEN.HEAD_CONFIG)
    assert {name: deg_to_tick(cfg.targets_deg[i]) for i, name in cfg.names.items()} == hp.ticks


def test_arm_free_stages_never_start_arm_hand_or_pd_programs():
    """★11.05 건너뛰기 · 팔 상태 무관 단계(arm_free)는 pd · 팔 드라이버 · 손 프로그램을 띄우지 않는다 — 불변식."""
    import re
    arm = re.compile(r"pd_|replay_to_pd|episode_ctl|trigger\.py|openarm|can_up|\bcan[01]\b|rh56f1_(driver|state)|ecat|hand/")
    for raw in (REAL_RAW, FAKE_RAW):
        free = [s["id"] for s in raw["stages"] if s.get("arm_free")]
        assert "cups" in free
        for sid in free:
            for c in raw["run"].get(sid, []):
                line = " ".join(map(str, c.get("argv") or []))
                assert not arm.search(line), (sid, line)
        # 같은 패턴이 실제 팔 · 손 · pd 단계는 잡는다(패턴이 헛돌지 않게)
        hits = {s["id"] for s in raw["stages"] if not s.get("arm_free")
                and any(arm.search(" ".join(map(str, c.get("argv") or []))) for c in raw["run"].get(s["id"], []))}
        if raw is REAL_RAW:
            assert {"drivers", "hand_right", "pd_arm_right", "home_right"} <= hits



def test_episode_without_objects_has_no_perception_ensure():
    """★11.05 확보할 물체가 없으면(FP++ 안 쓰는 에피소드 · fake) 확보 명령을 넣지 않는다."""
    def ensures(raw, sid):
        return [c for c in raw["run"].get(sid, []) if "perception_ctl.py" in " ".join(map(str, c.get("argv") or []))]
    assert not any(ensures(FAKE_RAW, s["id"]) for s in FAKE_RAW["stages"] if s["id"].startswith("episode_"))
    assert not ensures(REAL_RAW, "episode_tilt_base_b")              # FP++ 안 쓰는 tilt — 물체가 있어도 확보하지 않는다
    assert ensures(REAL_RAW, "episode_tilt_base_c")


def test_preflight_builds_contract_before_contract_dependent_tests():
    for name in ('mission_rh56f1_control.yaml', 'mission_rh56f1_fake.yaml'):
        raw, _, _ = _load(name)
        steps = raw['run']['preflight']
        build = next(i for i, s in enumerate(steps) if any('build_deploy_contract.py' in a for a in s.get('argv', [])))
        tests = next(i for i, s in enumerate(steps) if 'pytest' in s.get('argv', []))
        assert build < tests


def _drivers(raw):
    """미션 run 의 목 드라이버 argv — {단계#번호: argv}."""
    return {f"{sid}#{i}": c["argv"] for sid, cmds in raw["run"].items() for i, c in enumerate(cmds)
            if c.get("background") and any("head_joint_publisher.py" in str(a) for a in c.get("argv") or [])}   # ★11.11 원격 목은 ssh 한 줄 안에 있다
    # ★11.18 background: cap 82 드라이버 확인(foreground)은 드라이버가 아니다


def test_only_the_place_student_episode_restarts_the_head_driver_with_tilt_82():
    """★11.08 사용자: 목 tilt 상한 82 는 놓기 미션에서만. 드라이버는 포트를 혼자 쥐므로 에피소드가 head_home 것을 내리고
    같은 인자 + --max-tilt 82 로 다시 띄운다. 시작 검사가 먼저 · 마스크 · 학생 노드는 그 뒤."""
    run = REAL_RAW["run"]["episode_place_m_right"]
    drv = _drivers(REAL_RAW)
    # 이름으로 찾는다 — 단계 번호는 에피소드 앞부분(경로 F 접근 등)이 바뀌면 움직인다
    tilt_idx = next(i for i, c in enumerate(run) if "--max-tilt" in (c.get("argv") or []))
    assert drv[f"episode_place_m_right#{tilt_idx}"] == [*drv["head_home#0"], "--max-tilt", "82"]
    at = {k: next(i for i, c in enumerate(run) if any(k in str(a) for a in c.get("argv") or []))   # ★11.11 원격은 ssh 한 줄
          for k in ("check_place_student.py", "head_joint_publisher.py", "head_mask_node.py", "rh_place_student_node.py")}
    stop = next(i for i, c in enumerate(run) if (c.get("stop") or [None])[0] == "head_home#0")   # 앞 단계(head_sweep) 드라이버도 같이 내린다
    assert at["check_place_student.py"] < stop < at["head_joint_publisher.py"] < at["head_mask_node.py"] < at["rh_place_student_node.py"]
    check = run[at["check_place_student.py"]]["argv"]
    assert check[check.index("--episode") + 1] == "place_m_right" and check[check.index("--side") + 1] == "right"
    node = run[at["rh_place_student_node.py"]]["argv"]
    assert "__node:=rh_place_node_right" in node and "ns:=right_place" in node and "device:=cuda" in node
    assert "student_dir:={repo}/deploy/policies/rh56f1/place/right_student_place" in node
    assert not [k for k in _drivers(FAKE_RAW) if k.startswith("episode_place_m_")]    # fake 는 목 드라이버가 없다


def test_cap_student_stage_runs_its_nodes_then_reset_check_start_and_next_calls_stage_next(tmp_path):
    """★11.18 cap 깊이 학생(오른팔) — 에피소드 실행기 없이 단계로. 시작 검사 → (실기) 82 드라이버 → 깊이 → 학생 노드 → reset → 확인 → start.
    서비스는 trigger.py(rc 1 = 거부) · fake 만 리시버 고정 자리 허락 · 정지는 shutdown 정책 정지가 노드를 내린다."""
    import subprocess
    for raw, real in ((REAL_RAW, True), (FAKE_RAW, False)):
        run = raw["run"]["cap_student_right"]
        node = next(c["argv"] for c in run if any("rh_cap_student_node.py" in str(a) for a in c.get("argv") or []))
        assert "__node:=rh_cap_node_right" in node and "ns:=right_cap" in node
        assert "episode_topic:=/policy_control/right/episode" in node and "student_dir:={repo}/deploy/policies/" + GEN.CAP_STUDENT in node
        assert "rcv_pose_topic:=/objects/shaker_c_orange/live_pose" in node and ("rcv_allow_fixed:=true" in node) is not real
        if not real:   # fake 만 CPU · 넓은 하강 허용(run_cap_student_fake.sh 와 같음)
            assert "device:=cpu" in node and "descend_xy_max_m:=0.30" in node and "descend_len_max_m:=0.50" in node
        calls = [(c["argv"][2], c["argv"][4]) for c in run if "trigger.py" in " ".join(c.get("argv") or [])]
        assert calls == [("episode/reset", "right_cap"), ("episode/start", "right_cap")]
        kinds = ["manual" if c.get("manual") else next((k for k in ("seq 80", "head_joint_publisher", "head_depth_node", "rh_cap_student",
                                                                     "fake_head_stream_driver", "fake_head_depth_pub",
                                                                     "/head_depth/image", "episode/reset", "episode/start", "-c")
                                                            if k in " ".join(c.get("argv") or [])), "stop")
                 for c in run]
        # ★11.18 리뷰 H2: 82 드라이버 바로 뒤 확인 · M5: reset 앞에 깊이 한 프레임 · fake 는 도메인 97 의 fake 목(스트림 추종) · 깊이
        assert kinds == ["-c", *(["stop", "head_joint_publisher", "seq 80", "head_depth_node"] if real else
                                 ["fake_head_stream_driver", "fake_head_depth_pub"]), "rh_cap_student",
                         "/head_depth/image", "episode/reset", "manual", "episode/start"]
        wait = next(c["argv"] for c in run if "/head_depth/image" in " ".join(c.get("argv") or []))
        assert wait[:2] == ["bash", "-lc"] and "timeout 60 ros2 topic echo --once" in wait[2] and not any(
            c.get("background") or c.get("execute_args") for c in run if c.get("argv") == wait)
        if real:
            chk = next(c for c in run if "seq 80" in " ".join(c.get("argv") or []))
            assert chk["argv"] == GEN.CAP_DRIVER_CHECK_ARGV and not chk.get("background")
            assert ("ssh" in chk["argv"]) is GEN.REMOTE_HEAD
        nxt = raw["run"]["cap_next_right"]
        assert len(nxt) == 1 and nxt[0]["argv"][2:5] == ["stage/next", "--episode-ns", "right_cap"] and nxt[0]["execute_args"] == ["--execute"]
        assert "cap_now.py" in " ".join(raw["run"]["cap_now"][-1]["argv"]) and raw["run"]["cap_now"][0].get("manual")
        shut = next(c for c in raw["run"]["shutdown"] if "policy_pourfj#1" in (c.get("stop") or []))
        bg = [f"cap_student_right#{i}" for i, c in enumerate(run) if c.get("background")]
        assert bg and set(bg) <= set(shut["stop"])
    # 시작 검사 한 줄 — md5 · 계약
    d = tmp_path / "b"
    (d / "student").mkdir(parents=True)
    (d / "student" / "x.pt").write_bytes(b"w")
    (d / "student.yaml").write_text("ckpt: {path: student/x.pt, md5: " + __import__("hashlib").md5(b"w").hexdigest() + "}\n")
    check = lambda: subprocess.run(["python3", "-c", GEN.CAP_CHECK_PY, str(d)], capture_output=True, text=True)  # noqa: E731
    assert check().returncode == 1 and "rh_cap_contract.json" in check().stdout
    (d / "rh_cap_contract.json").write_text("{}")
    assert check().returncode == 0
    (d / "student" / "x.pt").write_bytes(b"v")
    assert check().returncode == 1 and "ckpt md5" in check().stdout
    (d / "student" / "x.pt").unlink()                  # 없으면 traceback 대신 한 줄 사유
    r = check()
    assert r.returncode == 1 and "없음" in r.stdout and "Traceback" not in r.stderr


def test_cap_end_stops_the_episode_then_its_nodes_then_restores_the_default_driver_and_head_home():
    """★11.18 리뷰: cap 은 대화식이라 꼬리를 cap_end_right 에 둔다 — episode/stop → cap 단계 배경(노드 · 깊이 · 82 드라이버) 정지 →
    (실기만) 놓기 꼬리와 같은 기본 드라이버 복귀 + head_home. fake 는 목이 없어 드라이버 단계가 없다."""
    stage = next(st for st in REAL_RAW["stages"] if st["id"] == "cap_end_right")
    assert stage["group"] == "policy" and stage["needs"] == ["cap_student_right"]
    for raw, real in ((REAL_RAW, True), (FAKE_RAW, False)):
        end, cap = raw["run"]["cap_end_right"], raw["run"]["cap_student_right"]
        assert end[0]["argv"][2:5] == ["episode/stop", "--episode-ns", "right_cap"] and end[0]["execute_args"] == ["--execute"]
        bg = [f"cap_student_right#{i}" for i, c in enumerate(cap) if c.get("background")]
        assert sorted(end[1]["stop"]) == sorted(bg)
        drv = [k for k in bg if "head_joint_publisher" in " ".join(map(str, cap[int(k.split("#")[1])]["argv"]))]
        assert len(drv) == (1 if real else 0)
        if real:
            assert end[2:] == GEN._place_student_tail()           # 복사 아닌 같은 꼬리
            assert end[2]["argv"] == list(GEN.HEAD_DRIVER_RESTORE_ARGV) and end[2]["background"]
            assert end[3]["argv"] == list(GEN.HEAD_HOME_ARGV)
            # 뒤 단계(놓기 에피소드 · 셧다운)의 head_home#0 정지가 cap_end 의 복귀 드라이버도 내린다
            for sid in ("episode_place_m_right", "shutdown"):
                stop = next(c["stop"] for c in raw["run"][sid] if "head_home#0" in (c.get("stop") or []))
                assert "cap_end_right#2" in stop and drv[0] in stop, sid
            # cap 앞 단계 head_sweep 이 남긴 드라이버는 cap 의 head_home#0 정지가 내린다
            assert {"head_sweep#1", "head_sweep#4"} <= set(cap[1]["stop"])
        else:
            assert len(end) == 2


def test_cap_depth_node_runs_free_head_and_the_cap_driver_takes_over_a_left_driver(tmp_path):
    """★11.18 cap 학생은 머리를 움직인다 — 깊이 노드는 --free-head(묶음 외부 파라미터 · 목 자세 게이트 없음) 실기 두 미션.
    fake 는 실기 카메라 PC 깊이가 도메인 97 에 안 와서 이 PC 의 fake 발행기(ssh 아님).
    82 드라이버는 정지 목록이 못 닿는 뒤 단계(cap_end_right · 놓기 꼬리) 드라이버를 PID 확정 뒤 SIGTERM 하고 띄운다."""
    import time

    import yaml as _y
    for f in ("mission_rh56f1_control.yaml", "mission_rh56f1_control_cam5090.yaml"):
        run = _y.safe_load((REPO / "config" / f).read_text(encoding="utf-8"))["run"]["cap_student_right"]
        depth = " ".join(next(c["argv"] for c in run if "head_depth_node.py" in " ".join(c.get("argv") or [])))
        assert "--free-head" in depth and "--policy-dir" not in depth and "fake_" not in depth, f
        node = " ".join(next(c["argv"] for c in run if "rh_cap_student_node.py" in " ".join(c.get("argv") or [])))
        assert "device:=cuda" in node and "descend_" not in node and "rcv_allow_fixed" not in node, f
    fake = [c["argv"] for c in FAKE_RAW["run"]["cap_student_right"]
            if any("head_depth_node.py" in str(a) or "fake_head_depth_pub.py" in str(a) for a in c.get("argv") or [])]
    assert fake == [["python3", "{repo}/scripts/fakes/fake_head_depth_pub.py", "--depth", "0.6"]]
    drv = next(c for c in REAL_RAW["run"]["cap_student_right"]
               if c.get("background") and "head_joint_publisher.py" in " ".join(c.get("argv") or []))
    assert drv["argv"] == GEN.CAP_HEAD_DRIVER_ARGV and drv["background"]
    for s in (GEN._HEAD_DRIVER_TAKEOVER, GEN._CAP_DRIVER_CHECK):
        assert "pkill" not in s and "killall" not in s
    # ★11.18 리뷰 M2: 원격 판은 common.sh(set -eo pipefail) 뒤에 돈다 — 같은 조건에서 시험한다
    script = "set -eo pipefail; " + GEN._HEAD_DRIVER_TAKEOVER + "echo LAUNCH"
    # 실기 PC 에서 돌릴 수 있다 — 진짜 목 드라이버가 떠 있으면 가드가 그것을 잡으니 건너뛴다
    if subprocess.run(["bash", "-c", GEN._CAP_DRIVER_PIDS + "[ -z \"$P\" ]"]).returncode:
        pytest.skip("이 PC 에 목 드라이버가 떠 있다")
    assert subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=20).stdout.strip() == "LAUNCH"   # (d) 0 개
    (tmp_path / "nodes").mkdir()
    fake = tmp_path / "nodes" / "head_joint_publisher.py"
    body = "import signal, sys, time\nsignal.signal(signal.SIGTERM, lambda *a: sys.exit(0))\ntime.sleep(60)\n"
    fake.write_text(body)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_head_joint_publisher.py").write_text(body)
    (tmp_path / "pytest.py").write_text(body)            # python3 -m pytest … 꼴(cwd 가 sys.path[0])
    py = tmp_path / "python3"
    py.symlink_to(sys.executable)
    procs = []

    def spawn(*argv):
        procs.append(subprocess.Popen([str(py), *argv], cwd=tmp_path))
        return procs[-1]
    try:
        # (b) 스크립트 자리가 아닌 것 — pytest 모듈 · -c · 이름만 비슷한 시험 파일은 살아남는다
        bystanders = [spawn("-m", "pytest", "tests/test_head_joint_publisher.py"),
                      spawn("-c", body, str(fake)), spawn("tests/test_head_joint_publisher.py")]
        time.sleep(0.3)
        for extra in ([], ["--max-tilt", "82"]):      # (a) 기본(cap_end · 놓기 꼬리)이든 82 든 내리고 띄운다
            p = spawn("-u", str(fake), *extra)        # 옵션 뒤 스크립트 자리 · reaper 없음 → SIGTERM 뒤 좀비(L1)
            time.sleep(0.3)
            out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=20)
            assert out.returncode == 0 and f"PID {p.pid}" in out.stdout, out.stdout + out.stderr
            assert out.stdout.strip().endswith("LAUNCH") and p.wait(timeout=5) == 0
        assert all(b.poll() is None for b in bystanders)
        # (c) 둘이면 아무것도 내리지 않고 rc 1 · PID 목록
        two = [spawn(str(fake)), spawn(str(fake), "--max-tilt", "82")]
        time.sleep(0.3)
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=20)
        assert out.returncode == 1 and "LAUNCH" not in out.stdout and all(str(x.pid) in out.stdout for x in two), out.stdout
        assert all(x.poll() is None for x in [*two, *bystanders])
        # H2 확인: 둘이면 실패, 82 하나면 통과, 기본 하나면 사유를 찍고 실패
        chk = ["bash", "-c", "set -eo pipefail; " + GEN._CAP_DRIVER_CHECK]
        assert subprocess.run(chk, capture_output=True, text=True, timeout=20).returncode == 1
        two[0].terminate()
        two[0].wait(timeout=5)
        out = subprocess.run(chk, capture_output=True, text=True, timeout=20)
        assert out.returncode == 0 and str(two[1].pid) in out.stdout, out.stdout + out.stderr
        two[1].terminate()
        two[1].wait(timeout=5)
        spawn(str(fake))
        out = subprocess.run(chk, capture_output=True, text=True, timeout=30)
        assert out.returncode == 1 and "--max-tilt 82 이 없다" in out.stdout, out.stdout
    finally:
        for x in procs:                                # 시험이 띄운 PID 만
            if x.poll() is None:
                x.kill()
                x.wait(timeout=5)


def test_the_place_student_episode_ends_with_the_default_driver_and_head_home():
    """★11.08 리뷰: 82 드라이버가 노드 정지로 내려가면 목 드라이버가 없는데 head_home 완료가 남는다 — 에피소드 끝 꼬리가
    기본 드라이버를 다시 띄우고 head_home(콘솔 확인 --execute)으로 camset 에 되돌린다. 셧다운은 토크 해제 전에 그 드라이버를 내린다."""
    run = REAL_RAW["run"]["episode_place_m_right"]
    halt = next(i for i, c in enumerate(run) if c.get("note") == "에피소드 노드들 정지")
    tail = halt + 1
    assert run[tail]["argv"] == GEN.HEAD_DRIVER_RESTORE_ARGV and run[tail]["background"]   # ★11.11 이미 떠 있으면 다시 안 띄운다
    assert f"episode_place_m_right#{tail}" not in run[halt]["stop"]
    assert run[tail + 1]["argv"] == GEN.HEAD_HOME_ARGV and run[tail + 1]["execute_args"] == ["--execute"]
    shut = next(c for c in REAL_RAW["run"]["shutdown"] if "policy_pourfj#1" in (c.get("stop") or []))
    release = next(i for i, c in enumerate(REAL_RAW["run"]["shutdown"]) if "--release" in (c.get("argv") or []))
    assert f"episode_place_m_right#{tail}" in shut["stop"] and REAL_RAW["run"]["shutdown"].index(shut) < release
    assert "head_move.py" not in " ".join(str(a) for c in FAKE_RAW["run"]["episode_place_m_right"] for a in c.get("argv") or [])


def test_the_console_warns_when_a_failed_stage_left_no_head_driver():
    from s2r_console.console import head_driver_gone
    drv = {"argv": ["python3", "{repo}/scripts/nodes/head_joint_publisher.py", "--max-tilt", "82"], "alive": False}
    remote = {"argv": ["ssh", "-tt", "h", "cd x && exec python3 scripts/nodes/head_joint_publisher.py --config c"], "alive": True}
    assert head_driver_gone([drv, {"argv": ["x"], "alive": True}])
    assert not head_driver_gone([drv, {**drv, "alive": True}])
    assert not head_driver_gone([drv, remote])
    assert not head_driver_gone([{"argv": ["bash", "-lc", "true"], "alive": False}])      # fake — 드라이버가 처음부터 없다


def test_every_other_head_driver_keeps_the_default_tilt_limit():
    """★11.08 다른 단계 · 미션의 목 드라이버 인자는 그대로(기본 −10, head_motion.MAX_TILT_DEG · 사용자 10.13)."""
    for raw in (REAL_RAW, FAKE_RAW):
        drv = _drivers(raw)
        tilt82 = [k for k, v in drv.items() if "--max-tilt" in " ".join(v).split("exec ")[-1]]   # ★11.18 cap 82 는 bash 가드 한 줄
        # ★11.09 82 는 놓기 학생 에피소드 둘(오른손 · 왼손)에만, 에피소드마다 하나씩 + ★11.10 목 스윕(camset 둘레 tilt +75.5 까지) 동안만
        got = sorted(k.rsplit("#", 1)[0] for k in tilt82)
        student = ["episode_place_m_left", "episode_place_m_right"] if "episode_place_m_right" in raw["run"] else []
        student += [f"episode_{n}" for n in GEN.PLACE_M_VARIANTS] if student else []   # ★11.11 놓기 학생 색 · 홀더 판(오른손)
        # ★11.18 cap 학생 단계(머리 tilt −2~82)도 놓기 학생처럼
        assert got == (sorted([*student, "head_sweep", "cap_student_right"]) if raw is REAL_RAW else []), tilt82
        others = {k: v for k, v in drv.items() if k not in tilt82}     # 놓기 에피소드 끝 꼬리 드라이버 포함
        assert all(v in (GEN.HEAD_DRIVER_ARGV, GEN.HEAD_DRIVER_RESTORE_ARGV) for v in others.values()), others
        assert all("--max-tilt" not in " ".join(v).split("exec ")[-1] for v in others.values()), others
    assert "head_home#0" in _drivers(REAL_RAW)


def test_place_student_episode_records_the_mask_and_the_head_stream():
    rec = next(c["argv"][-1] for c in REAL_RAW["run"]["episode_place_m_right"] if "rh56f1_record.sh start" in " ".join(c.get("argv") or []))
    for t in ("/head_mask/image", "/head_mask/config", "/head/target_stream", "/head/stream_log"):
        assert t in rec, t
    assert "compressed" not in rec                                            # 학생 입력 지연(압축 구독) 없음


def test_head_sweep_swaps_to_the_82_driver_and_restores_the_default_driver_and_camset():
    """★11.10 사용자 "swap 천장을 바라보네": 스윕은 camset 둘레(tilt 최대 +75.5)라 82 드라이버로 바꿔 돌고, 끝나면 기본 드라이버 + camset.
    뒤 단계(놓기 에피소드 · shutdown)의 head_home 정지는 스윕이 띄운 드라이버도 같이 내린다(포트 단독 소유)."""
    run = REAL_RAW["run"]["head_sweep"]
    drv = _drivers(REAL_RAW)
    assert run[0]["stop"] == ["head_home#0"]
    assert drv["head_sweep#1"] == [*drv["head_home#0"], "--max-tilt", "82"]
    sweep = run[2]["argv"]
    assert sweep[-3:] == ["{repo}/scripts/head_sweep.py", "--max-tilt", "82"] and run[2]["execute_args"] == ["--execute"]
    assert "--calib-profile" not in sweep and "--return-pan" not in sweep
    assert run[3]["stop"] == ["head_sweep#1"] and drv["head_sweep#4"] == GEN.HEAD_DRIVER_ARGV
    assert run[5]["argv"] == GEN.HEAD_HOME_ARGV
    for stage in [st for st in ("episode_place_m_right", "shutdown") if st in REAL_RAW["run"]]:   # ★11.11 학생 에피소드는 카메라가 같은 PC 일 때만
        stop = next(c["stop"] for c in REAL_RAW["run"][stage] if "head_home#0" in (c.get("stop") or []))
        assert {"head_sweep#1", "head_sweep#4"} <= set(stop), stage


def test_recalib_hint_never_overwrites_the_charuco_extrinsics():
    """★11.17 m1117 외부 파라미터 = 테이블 마커 값(11.15 h2048 은 ChArUco)(학생 묶음 · head_depth_node 가 md5 고정) — 안내가 테이블 CAD 값으로 덮게 하지 않는다."""
    h, ex = GEN.RECALIB_HINT, GEN.CAMERA_EXTRINSICS
    assert f"--out {ex} " not in h and f"--init {ex} --write" not in h
    assert f"--out {ex.replace('.yaml', '_tablecheck.yaml')} --write" in h


def test_both_stations_have_a_mission_and_the_student_head_nodes_follow_the_camera():
    """★11.11 사용자 "5090이든 4090이 둘다 사용 가능하도록": 실기 미션은 스테이션마다 한 벌(카메라 · 목 4090 = 기본,
    5090 = _cam5090). 학생 깊이 · 마스크 노드는 카메라 PC 에서(4090 이면 ssh), 학생 정책 노드는 늘 이 PC 에서."""
    assert set(GEN.REAL_MISSIONS) == {"5090_arm4090_h2048", "5090"}
    raws = {st: yaml.safe_load((REPO / rel).read_text(encoding="utf-8")) for st, rel in GEN.REAL_MISSIONS.items()}
    for st, raw in raws.items():
        run = raw["run"]
        assert "episode_grasp_m_right" in run and "episode_place_m_right" in run, st
        cmds = run["episode_grasp_m_right"]
        depth = next(c["argv"] for c in cmds if any("head_depth_node.py" in a for a in c.get("argv") or []))
        node = next(c["argv"] for c in cmds if any("rh_aglt_student_node.py" in a for a in c.get("argv") or []))
        remote = GEN.STATIONS[st]["perception_host"] != "local"
        assert (depth[0] == "ssh") == remote, (st, depth)
        assert node[0] == "{repo}/.venv-student/bin/python", (st, node)
        rec = next(c["argv"][-1] for c in cmds if "rh56f1_record.sh start" in " ".join(c.get("argv") or []))
        assert "/head_depth/image" in rec and "/camera/camera/color/image_raw/compressed" not in rec, (st, rec)


def test_mask_student_target_class_follows_the_episode_cup_color():
    """★11.10 마스크 학생 target_class = 에피소드 컵(aglt target_object · place source_object) 토픽 이름 끝 색, 못 뽑으면 생성 실패."""
    from policy_control.student_mask import CLASSES
    assert GEN.MASK_COLORS == CLASSES
    for name, node, want in (("grasp_m_right", "rh_aglt_student_node.py", "blue"),
                             ("grasp_m_left", "rh_aglt_student_node.py", "orange"),
                             ("place_m_right", "rh_place_student_node.py", "blue"),
                             ("place_m_pink_h2_right", "rh_place_student_node.py", "pink")):
        argv = next(c["argv"] for c in REAL_RAW["run"][f"episode_{name}"] if any(node in str(a) for a in c.get("argv") or []))
        assert f"target_class:={want}" in argv, (name, argv)
    assert GEN._mask_target("/objects/source240b_blue/pose") == "blue"
    for bad in ("/objects/cup_src/pose", "/objects/source240_red/pose", "/episode/objects/x_blue/pose"):
        with pytest.raises(ValueError):
            GEN._mask_target(bad)


def test_cup_holders_end_with_one_holder_confirm_and_cups_never_touches_the_holders():
    """★11.10 홀더 한 번 재기 — 실기는 --timeout-s · --max-jump-m 로 돌고 '; true' 로 단계를 막지 않는다(결과는 last_check → 시작 검사).

    ★11.11 deploy 리뷰 H2: 실패(카메라 · ssh · scp)는 삼키지 않고 rc 를 받아 ⚠ + last_check.json = failed, yaw 상한도 건다.
    ★11.11 사용자 "컵홀더 측정 완료(고정) → 컵 배치": 확정은 cup_holders 끝에서만 — cups 는 홀더 파일 · 기록을 건드리지 않는다
    (cups_5: 셰이커가 홀더 0 을 가려 cups 끝 재기가 cup_holders 의 written 을 failed 로 덮었다). ★11.17 cups 는 cup_holders 를 기다리지 않는다(홀더 고정 · 저장 파일 · 에피소드 시작 검사).
    """
    real = " ".join(REAL_RAW["run"]["cup_holders"][-1]["argv"])
    assert "cup_holder_pose_node.py --write --timeout-s" in real and "--max-jump-m 0.03" in real and real.endswith("true"), real
    assert f"--max-yaw-deg {GEN.HOLDER_MAX_YAW_DEG:g}" in real and "rc=$?" in real and "⚠ 홀더 다시 재기 실패" in real
    assert "camera_up.sh; " not in real and "; true; " not in real   # 중간 실패를 삼키는 '; true' 가 없다
    if GEN.REMOTE_PERCEPTION:
        assert real.count("rc=$?") == 2 and ".last_check.json config/" in real    # ssh 와 scp 각각
        assert f"timeout {GEN.HOLDER_ONCE_S + GEN.HOLDER_SSH_SLACK_S:g} ssh -o BatchMode=yes" in real   # ★11.11 M3
    for raw in (REAL_RAW, FAKE_RAW):
        cups = " ".join(" ".join(c.get("argv") or []) for c in raw["run"]["cups"])
        assert "cup_holder_pose_node" not in cups and "last_check" not in cups
    st = {x["id"]: x for x in REAL_RAW["stages"]}
    assert "cup_holders" not in st["cups"]["needs"] and st["cup_holders"]["skippable"]   # ★11.17 홀더 고정 — cups 는 홀더를 기다리지 않는다

def test_every_command_resolves_without_a_stray_placeholder():
    """★11.11 cups 홀더 재기 실패 기록(printf JSON)의 `{"time": …}` 가 자리표시자로 읽혀 단계가 죽었다 —
    모든 단계 · 실기와 가짜 모두, 명령 치환이 ValueError 없이 끝나야 한다(사라진 정책 파일만 건너뛴다)."""
    from mission_stages import resolve

    for raw, m in ((REAL_RAW, REAL), (FAKE_RAW, FAKE)):
        for sid, cmds in raw["run"].items():
            for c in cmds or []:
                for a in [*c.get("argv", ()), *(c.get("execute_args") or ())]:
                    try:
                        resolve(str(a), m, repo=REPO)
                    except KeyError:
                        continue  # 정책 · 산출물 파일이 이 체크아웃에 없음 — 자리표시자 자체는 알아본 것
                    except ValueError as exc:
                        raise AssertionError(f"{sid}: {exc} — {str(a)[:120]}") from exc


def test_the_holder_failure_record_is_json_and_cups_goes_on(tmp_path):
    """★11.11 H2: 카메라 · 노드가 실패하면 ⚠ 줄 + last_check.json = failed(시작 검사가 읽는다), 단계 rc 는 0."""
    import json
    from mission_stages import resolve

    sh = GEN._holder_fail_sh("config/x.last_check.json", "카메라")
    (tmp_path / "config").mkdir()
    out = subprocess.run(["bash", "-c", "rc=7; " + resolve(sh, REAL, repo=REPO) + "; true"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0 and "⚠" in out.stdout
    rec = json.loads((tmp_path / "config" / "x.last_check.json").read_text(encoding="utf-8"))
    assert rec["result"] == "failed" and rec["rc"] == 7 and rec["time"]
    import time
    assert abs(rec["epoch"] - time.time()) < 60     # ★11.11 M1: 시작 검사가 읽는 epoch


def test_the_place_tail_driver_reuses_a_default_driver_and_terms_a_tilt_82_one(tmp_path):
    """★11.11 deploy 3 (실기 224018 pink_h2 cleanup rc≠0 "PID 966259 가 단독으로 열고 있다"): 꼬리 드라이버는 포트를 쥔 드라이버가
    있으면 인자를 본다 — 기본이면 다시 띄우지 않고 그 PID 를 지켜보고, --max-tilt 가 남았으면 그 PID 를 SIGTERM 한 뒤 띄운다."""
    import os, signal, subprocess, sys, time
    script = GEN._HEAD_DRIVER_GUARD + "echo LAUNCH"
    fake = tmp_path / "head_joint_publisher.py"
    fake.write_text("import signal, sys, time\nsignal.signal(signal.SIGTERM, lambda *a: sys.exit(0))\ntime.sleep(60)\n")
    py = tmp_path / "python3"
    py.symlink_to(sys.executable)
    for extra, want in (([], "기본 목 드라이버 PID"), (["--max-tilt", "82"], "LAUNCH")):
        drv = subprocess.Popen([str(py), str(fake), *extra])
        import threading
        reaper = threading.Thread(target=drv.wait, daemon=True)     # 실기는 sshd 가 거둔다 — 좀비가 kill -0 에 살아 보이지 않게
        reaper.start()
        try:
            time.sleep(0.3)
            if not extra:       # ★11.11 deploy LOW: 지켜보던 기본 드라이버가 끝나면 rc 1 — 0.8 s 뒤 드라이버를 죽인다
                threading.Timer(0.8, lambda: os.kill(drv.pid, signal.SIGKILL)).start()
            out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=20)
            assert want in out.stdout and f"PID {drv.pid}" in out.stdout, out.stdout + out.stderr
            if extra:
                reaper.join(timeout=5)
                assert drv.returncode == 0 and out.returncode == 0 and out.stdout.strip().endswith("LAUNCH")  # SIGTERM 으로 끝났다
            else:
                assert out.returncode == 1 and "❌ 지켜보던 목 드라이버" in out.stdout and "LAUNCH" not in out.stdout
        finally:
            if drv.poll() is None:
                os.kill(drv.pid, signal.SIGKILL)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=20)
    assert out.stdout.strip() == "LAUNCH"                           # 드라이버가 없으면 그냥 띄운다
    # ★11.11 deploy MEDIUM: pgrep 직후 PID 가 끝나 cmdline 을 못 읽으면 지켜보지 않고 새로 띄운다 · pgrep 은 내 uid 만
    assert 'pgrep -u "$(id -u)"' in GEN._HEAD_DRIVER_GUARD
    drv = subprocess.Popen([str(py), str(fake)])
    try:
        time.sleep(0.3)
        gone = script.replace("</proc/$p/cmdline", "</proc/$p/no_such_cmdline")
        out = subprocess.run(["bash", "-c", gone], capture_output=True, text=True, timeout=20)
        assert out.returncode == 0 and "새로 띄운다" in out.stdout and out.stdout.strip().endswith("LAUNCH"), out.stdout + out.stderr
        assert out.stderr == ""
    finally:
        os.kill(drv.pid, signal.SIGKILL)
        drv.wait()


def test_shutdown_still_stops_the_place_tail_driver_in_every_mission():
    """★11.11 꼬리 드라이버 argv 가 bash 가드로 바뀌어도 shutdown 이 그것을 내린다(로컬 목 cam5090 포함)."""
    import yaml as _y
    for f in ("mission_rh56f1_control.yaml", "mission_rh56f1_control_cam5090.yaml"):
        m = _y.safe_load((REPO / "config" / f).read_text(encoding="utf-8"))
        stops = {s for c in m["run"]["shutdown"] for s in (c.get("stop") or [])}
        for name in GEN.PLACE_M_VARIANTS:
            tail = next(i for i, c in enumerate(m["run"][f"episode_{name}"])
                        if c.get("background") and GEN._HEAD_DRIVER_GUARD in " ".join(c.get("argv") or []))
            assert f"episode_{name}#{tail}" in stops, (f, name)


def test_grasp_s_color_variants_follow_the_student_one_obj_and_only_change_the_cup():
    """★11.11 잡기 학생 색 버튼 — 팔마다 학생 ckpt one_obj 가 허용하는 색만(OBJ_TARGETS, 첫 값 = 기본 = 원본 파일).
    변형은 원본에서 컵 토픽 · 이름만 바꾸고(policy_object 그대로), 미션 단계 · 학생 bag · 마스크 목표 색 · shutdown 이 원본과 같다."""
    from policy_control.student_mask import OBJ_TARGETS
    assert GEN.GRASP_S_VARIANTS == ("grasp_s_pink_right",)
    eps = REPO / "config" / "episodes"
    for side, colors in GEN.GRASP_S_SIDE_COLORS.items():
        raw = yaml.safe_load((eps / f"grasp_s_{side}.yaml").read_text(encoding="utf-8"))
        pol = next(iter(raw["policies"].values()))
        one_obj = yaml.safe_load((REPO / "deploy" / "policies" / pol["student"] / "student.yaml").read_text())["one_obj"]
        assert OBJ_TARGETS[one_obj] == colors and colors[0] == GEN.GRASP_S_DEFAULT[side], (side, one_obj)
        assert raw["objects"]["CUP"]["topic"] == f"/objects/{GEN.GRASP_S_COLORS[GEN.GRASP_S_DEFAULT[side]]}/pose"
        for c in colors[1:]:
            name = GEN.grasp_s_name(c, side)
            var = yaml.safe_load((eps / f"{name}.yaml").read_text(encoding="utf-8"))
            assert var["episode"]["name"] == name and var["objects"]["CUP"]["topic"] == f"/objects/{GEN.GRASP_S_COLORS[c]}/pose"
            assert {**var, "episode": raw["episode"], "objects": raw["objects"]} == raw     # 나머지(policy_object 포함)는 원본 그대로
            assert name in GEN.STUDENT_EPISODES and f"episode_{name}" in {st["id"] for st in REAL_RAW["stages"]}
            # 마스크 노드는 클래스 비트 전부를 내고(목표 인자 없음) 목표 색은 학생 노드 target_class 가 고른다
            argv = {node: " ".join(map(str, next(c2["argv"] for c2 in REAL_RAW["run"][f"episode_{name}"]
                                                 if any(node in str(a) for a in c2.get("argv") or []))))
                    for node in ("head_mask_node.py", "rh_aglt_student_node.py")}
            assert f"target_class:={c}" in argv["rh_aglt_student_node.py"] and "target" not in argv["head_mask_node.py"], argv
            stops = lambda n: sorted(s.split("#")[1] for c2 in REAL_RAW["run"]["shutdown"] for s in c2.get("stop") or []  # noqa: E731
                                     if s.startswith(f"episode_{n}#"))
            assert stops(name) == stops(f"grasp_s_{side}") != []


def test_console_grasp_s_colors_match_the_generator():
    """★11.11 콘솔 policy 탭 S 색 버튼 = 생성기 GRASP_S_* (place_m 처럼 시험이 맞춘다)."""
    sys.path.insert(0, str(REPO / "deploy" / "s2r_console"))
    from s2r_console import experiments as XV
    assert {k: fpp for k, fpp, _ in XV.GRASP_S_COLORS} == GEN.GRASP_S_COLORS
    assert XV.GRASP_S_SIDE_COLORS == GEN.GRASP_S_SIDE_COLORS and XV.GRASP_S_DEFAULT == GEN.GRASP_S_DEFAULT
    assert {f"{XV.grasp_s_head(c, s)}_{s}" for s, cs in XV.GRASP_S_SIDE_COLORS.items() for c in cs
            if c != XV.GRASP_S_DEFAULT[s]} == set(GEN.GRASP_S_VARIANTS)
    v = XV.view(REPO, [st["id"] for st in REAL_RAW["stages"]])
    gs = {i["id"]: i for i in v["items"]}["grasp_s"]["sides"]
    right = gs["right"]["objects"]
    assert [(o["object"], o["action"], o["default"]) for o in right] == [
        ("blue", "ctl_grasp_s_right", True), ("pink", "ctl_grasp_s_pink_right", False)]
    assert "holder" not in right[0] and "objects" not in gs["left"]          # 홀더 버튼 없음 · 왼손은 주황 하나


# ---------- ★11.17 리뷰: 원격 시야 검사 명령을 가짜 ssh(PATH 앞 스크립트)로 실제로 돌린다 ----------

FAKE_SSH = """#!/bin/bash
while [ "${1#-}" != "$1" ]; do [ "$1" = -o ] && shift; shift; done
shift   # 호스트
cd "$FAKE_REMOTE_HOME" && eval "$*"
"""


@pytest.mark.skipif(not GEN.REMOTE_PERCEPTION, reason="원격 인지 스테이션에서만 ssh 로 가져온다")
@pytest.mark.parametrize("case", ["ok", "remote_fail", "empty", "other_base"])
def test_remote_view_check_moves_the_live_only_after_checking_it(tmp_path, case):
    import os
    import time
    sys.path.insert(0, str(REPO / "scripts"))
    import head_view_feedback as hv
    repo, home = tmp_path / "repo", tmp_path / "home"
    (repo / "config").mkdir(parents=True)
    (repo / "scripts").symlink_to(REPO / "scripts")
    base = repo / GEN.CAMERA_EXTRINSICS
    base.write_bytes((REPO / GEN.CAMERA_EXTRINSICS).read_bytes())
    remote = home / GEN.REMOTE_SIM2REAL
    (remote / "scripts" / "vision").mkdir(parents=True)
    (remote / "config").mkdir()
    (remote / "scripts" / "vision" / "common.sh").write_text("", encoding="utf-8")
    (remote / "scripts" / "head_view_feedback.py").write_text(
        "import sys; sys.exit(0 if sys.argv[-1] == '--execute' and %r != 'remote_fail' else 1)\n" % case, encoding="utf-8")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    md5 = "0" * 32 if case == "other_base" else hv.file_md5(base)
    text = "" if case == "empty" else hv.live_text(base, None, (1825, 2833), {"base": base.name, "base_md5": md5, "time": now})
    (remote / GEN.FPP_EXTRINSICS).write_text(text, encoding="utf-8")
    import hashlib
    (remote / GEN.FPP_LIVE_REF).write_text(yaml.safe_dump({"extrinsics": {"md5": hashlib.md5(text.encode()).hexdigest()},
                                                           "verified_at": now}), encoding="utf-8")
    live, live_ref = repo / GEN.FPP_EXTRINSICS, repo / GEN.FPP_LIVE_REF
    live.write_text("old", encoding="utf-8")                  # 지난 live — 실패면 남지 않아야 한다
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "ssh").write_text(FAKE_SSH, encoding="utf-8")
    (bin_ / "ssh").chmod(0o755)
    (bin_ / "python3").symlink_to(sys.executable)
    argv = commands_for(REAL_BOOK, REAL, "head_view_check", repo=repo, execute=True)[1].argv
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "HOME": str(tmp_path), "FAKE_REMOTE_HOME": str(home)}
    r = subprocess.run(list(argv), env=env, capture_output=True, text=True, timeout=60)
    assert not list((repo / "config").glob("*.tmp"))
    if case == "ok":
        assert r.returncode == 0, r.stdout + r.stderr
        assert live.read_text(encoding="utf-8") == text and live_ref.exists()
        hv.require_fresh_live(live)                           # 이 PC 의 FP++ 쪽 시작 검사도 통과
    else:
        assert r.returncode != 0 and not live.exists() and not live_ref.exists(), r.stdout + r.stderr

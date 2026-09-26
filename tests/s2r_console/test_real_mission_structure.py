"""실기 미션(config/mission_dg5f_m_control.yaml)의 순서 — 09.22 실기에서 막힌 순서가 다시 들어오지 않게 잠근다."""
from __future__ import annotations

from pathlib import Path

import yaml
from mission_core import load_mission
from mission_stages import load_runbook

PATH = Path(__file__).resolve().parents[2] / "config" / "mission_dg5f_m_control.yaml"
RAW = yaml.safe_load(PATH.read_text(encoding="utf-8"))
MISSION = load_mission(RAW)
BOOK = load_runbook(RAW.get("run"), MISSION)          # 정지 대상 · 순서 검사가 여기서 돈다
IDS = [s.id for s in MISSION.stages]
RAW_STAGES = {st["id"]: st for st in RAW["stages"]}


def _cmds(stage):
    return BOOK.commands[stage]


def _launches(stage, needle):
    return [i for i, c in enumerate(_cmds(stage)) if any(needle in a for a in c.argv)]


def test_every_stage_sits_in_one_of_the_six_groups_in_order():
    assert [g.title for g in MISSION.groups] == ["점검", "연결", "준비", "자세 이동", "정책 동작", "정리"]
    assert all(s.group for s in MISSION.stages)
    assert {g.id for g in MISSION.groups if g.motion} == {"motion", "policy"}


def test_each_arm_is_readied_engaged_and_homed_before_the_selftest():
    for side in ("right", "left"):
        order = [f"pd_load_{side}", f"pd_arm_{side}", f"home_{side}", f"selftest_{side}", f"release_{side}"]
        assert [IDS.index(s) for s in order] == sorted(IDS.index(s) for s in order), side
        assert any(a == "execute:=false" for c in _cmds(f"pd_load_{side}") for a in c.argv)
        arm = _cmds(f"pd_arm_{side}")
        assert arm[0].stop and any(a == "execute:=true" for a in arm[1].argv)          # 무발행을 내린 뒤 발행을 띄운다
        cmds = _cmds(f"home_{side}")
        home = [" ".join(c.argv) for c in cmds]
        # engage·제자리 → 확인 → **손 기준 자세** → 시작점 정렬 → 검사 → 미리보기 → 확인 → 재생 → 정착 → 확인 → 손
        assert "--only pd_engage --hold-s 10" in home[0] and cmds[1].manual
        # 09.23 사용자 "어떤 hand 자세든 trajectory 시작 자세로 맞춘다" — 손이 먼저, 팔이 움직이기 전에
        assert "hand_to_path_pose.py" in home[2] and f"--side {side}" in home[2]
        assert home.index([h for h in home if "hand_to_path_pose.py" in h][0]) < \
               home.index([h for h in home if "plan_approach_to_start.py" in h][0])
        # 손이 구를 넘으면 실측에서 접어 본다 — 그 다음이 팔이다(09.23 사용자: "현재 joint state 기반해서")
        assert "plan_hand_fold.py" in home[3] and f"--side {side}" in home[3]
        # 저장 경로는 다시 계획하지 않는다 — 시작점까지 정렬 → 확인 → 미리보기 → 확인 → 재생(09.22 · 09.23 사용자)
        assert "plan_approach_to_start.py" in home[4] and f"{{artifact:path_{side}}}" in home[4]
        assert "replay_to_pd.py" in home[5] and f"approach_{side}.npz" in home[5]
        assert "check_path_start.py" in home[6] and f"{{artifact:path_{side}}}" in home[6]
        assert "preview_path_in_viewer.py" in home[7] and "--with-fixed" not in home[7] and cmds[8].manual
        assert "replay_to_pd.py" in home[9] and f"{{artifact:path_{side}}}" in home[9] and "--reverse" not in home[9]
        assert not any("plan_home_from_robot" in a for c in cmds for a in c.argv)
        assert "--only pd_goto_home" in home[10] and "--service-timeout 45" in home[10]    # 정착만 — 도착은 재생이 했다
        assert cmds[11].manual and "--only pd_hand_home" in home[12]
        ret = _cmds(f"return_{side}")
        assert "--only pd_goto_home" in " ".join(ret[0].argv)                               # fabric 뒤 HOLD 를 풀고
        assert "--only pd_hand_rest" in " ".join(ret[1].argv)                               # 손을 출발 자세로 먼저
        back = " ".join(ret[2].argv)
        assert "--reverse" in back and f"{{artifact:path_{side}}}" in back                    # 같은 경로를 되짚는다
        assert f"path_{side}" in MISSION.stages[IDS.index(f"home_{side}")].artifacts          # 승인 근거 해시에 들어간다
        assert IDS.index(f"return_{side}") < IDS.index(f"release_{side}")
        assert "--joints" not in " ".join(_cmds(f"selftest_{side}")[0].argv)          # 초기 자세는 팔꿈치가 굽어 7관절 모두


def test_the_head_publisher_starts_only_after_head_home_released_the_port():
    cmds = _cmds("head_home")
    assert "head_home.py" in cmds[0].argv[-1] and not cmds[0].background
    assert "head_joint_publisher.py" in cmds[1].argv[-1] and cmds[1].background
    assert not any("head_joint_publisher" in a for s in IDS if s != "head_home" for c in _cmds(s) for a in c.argv)


def test_each_arm_runs_its_own_pd_and_no_longer_kills_the_other():
    """09.23 사용자: "pd 를 구분하는 게 맞을 것 같음" — 왼팔 pd 를 띄워도 오른팔 pd 를 내리지 않는다.

    그 전에는 노드 이름이 `/pd_node` 하나뿐이라 왼팔을 띄우기 전에 오른팔을 내려야 했다.
    이제 이름이 `pd_node_<side>` 이고 서비스·status 도 팔마다 따로라 둘이 같이 떠 있어도 된다.
    """
    left = _cmds("pd_load_left")
    assert not any("right" in k for c in left for k in c.stop), [c.stop for c in left]
    assert set(_cmds("release_right")[-1].stop) == {"fabric_direct_right#0", "pd_arm_right#1"}
    #: 각 팔의 pd 는 제 쪽만 띄운다 — sides 인자가 그 팔 하나여야 이름이 갈린다.
    for side in ("right", "left"):
        for stage in (f"pd_load_{side}", f"pd_arm_{side}"):
            launch = [c for c in _cmds(stage) if any("pd_controller.launch.py" in a for a in c.argv)]
            assert launch, stage
            assert f"sides:={side}" in launch[0].argv, (stage, launch[0].argv)


def test_every_pd_service_call_names_the_arm():
    """pd 서비스는 팔마다 따로다 — 미션의 모든 episode_ctl 호출이 `--side` 를 들고 있어야 한다."""
    for sid in IDS:
        for c in _cmds(sid):
            if not any("episode_ctl.py" in a for a in c.argv):
                continue
            only = c.argv[c.argv.index("--only") + 1] if "--only" in c.argv else ""
            if not any(x.startswith("pd_") for x in only.split(",")):
                continue
            assert "--side" in c.argv, (sid, c.argv)
            side = c.argv[c.argv.index("--side") + 1]
            assert sid.endswith(f"_{side}"), (sid, side)     # 그 단계의 팔과 같아야 한다


def test_shutdown_is_last_confirms_support_first_and_drops_the_arm_last():
    assert IDS[-1] == "shutdown"
    cmds = _cmds("shutdown")
    assert cmds[0].manual and "받침" in cmds[0].note
    stops = [c.stop for c in cmds if c.stop]
    arm = _launches("drivers", "openarm.bimanual.launch.py")[0]
    hands = {f"hand_{side}#{_launches(f'hand_{side}', 'dg5f_driver')[0]}" for side in ("right", "left")}
    assert stops[-1] == (f"drivers#{arm}",) and set(stops[-2]) == hands


def test_each_hand_driver_is_its_own_stage_in_its_arms_window():
    """09.25 사용자: "팔은 하나인거 알고 있고, 손만 분리 필요" — 팔 브링업은 양팔이 한 프로세스라 drivers 에 남고,
    손 드라이버는 팔마다 따로 켜고 끄고 다시 띄운다(단계 다시 실행 = 그 단계의 유닛만 내렸다 다시 띄움)."""
    assert not _launches("drivers", "dg5f_driver")                                  # 드라이버 창에는 팔만
    assert len(_launches("drivers", "openarm.bimanual.launch.py")) == 1
    for side, ip in (("right", "169.254.186.72"), ("left", "169.254.186.73")):
        st = MISSION.stages[IDS.index(f"hand_{side}")]
        assert RAW_STAGES[st.id]["lane"] == f"arm_{side}" and st.touches_real and st.skippable
        assert set(st.needs) == {"preflight"}                                        # 팔 브링업과 무관하게 켠다
        (i,) = _launches(st.id, "dg5f_driver")
        cmd = _cmds(st.id)[i]
        assert cmd.background and f"dg5f_{side}_driver.launch.py" in cmd.argv and f"delto_ip:={ip}" in cmd.argv
        assert any(c.manual and "hand_net_dual.sh" in " ".join(c.argv) for c in _cmds(st.id)[:i])   # 네트워크 먼저
        other = "left" if side == "right" else "right"
        assert not _launches(st.id, f"dg5f_{other}_driver")
        load = MISSION.stages[IDS.index(f"pd_load_{side}")]
        assert f"hand_{side}" in load.needs and "drivers" in load.needs              # pd 는 팔과 그 손이 다 떠야
        assert IDS.index(f"hand_{side}") < IDS.index(f"pd_load_{side}")


def test_only_side_stages_and_optional_sensors_can_be_skipped():
    fixed = {s.id for s in MISSION.stages if not s.skippable}
    assert fixed == {"preflight", "drivers", "shutdown"}


def test_only_the_arm_stage_launches_pd_with_the_exec_config():
    # 발행 = launch execute:=true AND yaml execute. 무발행 단계는 원본(false), 발행 전환 단계만 사본(true).
    for side in ("right", "left"):
        load = " ".join(a for c in _cmds(f"pd_load_{side}") for a in c.argv)
        arm = " ".join(a for c in _cmds(f"pd_arm_{side}") for a in c.argv)
        assert "pd_config:={artifact:pd}" in load and "execute:=false" in load
        assert "pd_config:={artifact:pd_exec}" in arm and "execute:=true" in arm
        assert "pd_exec" in MISSION.stages[IDS.index(f"pd_arm_{side}")].artifacts      # 승인 근거 해시에 들어간다
    others = [s for s in IDS if not s.startswith("pd_arm_")]
    assert not any("{artifact:pd_exec}" in a for s in others for c in _cmds(s) for a in c.argv)


def test_the_home_is_the_right_policy_initial_state_and_rviz_stays_off():
    build = " ".join(_cmds("preflight")[1].argv)
    assert "--home run:deploy/policies/right_aglt" in build
    # 09.23 사용자 "오른팔 왼팔 대칭": preflight 가 계약을 다시 만드므로 여기에 없으면 왼팔 홈이 벽 앞(1.6 cm)으로 되돌아간다
    assert "--mirror-other-arm" in build
    arm = _cmds("drivers")[_launches("drivers", "openarm.bimanual.launch.py")[0]]
    assert "use_rviz:=false" in arm.argv
    assert not any(s.startswith("preset") for s in IDS)              # 차렷 기준 궤적 — 초기 자세에서는 쓸 수 없다


def test_the_isaac_viewer_comes_before_anything_that_uses_the_gpu():
    # run_viewer.sh 는 GPU 에 python 계산 프로세스가 있으면 뜨지 않는다(학습 보호) — fabric(cuda) 보다 먼저 켠다.
    assert IDS.index("viewer") < IDS.index("fabric_direct_right")
    (cmd,) = _cmds("viewer")
    assert cmd.background and cmd.argv[-1].endswith("robot/isaacsim_bridge/viewer/run_viewer.sh")
    assert MISSION.stages[IDS.index("viewer")].skippable and not MISSION.stages[IDS.index("viewer")].touches_real


def test_the_fake_mission_is_generated_from_the_real_one_and_walks_the_same_stages():
    # 오른팔 · 오른손 · 왼팔 · 왼손을 fake 로 먼저 확인한다(09.22). fake 가 실기와 어긋나면 확인한 것이 실기가 아니다.
    import subprocess
    import sys

    root = PATH.parents[1]
    rc = subprocess.run([sys.executable, str(root / "scripts/ops/make_fake_mission.py"), "--check"],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    fake_raw = yaml.safe_load((root / "config/mission_dg5f_m_fake.yaml").read_text(encoding="utf-8"))
    fake = load_mission(fake_raw)
    book = load_runbook(fake_raw["run"], fake)
    assert [(s.id, s.group, s.skippable) for s in fake.stages] == [(s.id, s.group, s.skippable) for s in MISSION.stages]
    assert not any(s.touches_real for s in fake.stages)                          # fake 는 승인 없이 — 실기가 아니다
    assert all("fake" in v for k, v in fake_raw["artifacts"].items() if k.startswith(("robot_", "pd")))
    def is_plant(c):
        return any(a.endswith("fake_plant.launch.py") for a in c.argv)

    launches = [c for cmds in book.commands.values() for c in cmds if c.argv[:2] == ("ros2", "launch") and not is_plant(c)]
    assert launches and all("fake:=true" in c.argv for c in launches)     # fake_plant 는 스스로 도메인 0 을 거부한다
    plant = book.commands["drivers"][0]
    assert is_plant(plant) and "hands:=none" in plant.argv                 # 팔만 — 손은 손 단계가 띄운다(실기와 같게)
    for side in ("right", "left"):
        (hand,) = [c for c in book.commands[f"hand_{side}"] if c.background]
        # 손은 pd 의 JTC 를 따르고, 09.23 부터 **경로 기준 자세에서 어긋난 채** 시작한다(실기가 전원 재투입 뒤 그랬다).
        # 0 자세(완전히 편 손)로 시작하면 손을 마는 중간 자세가 몸통을 스쳐 검사가 막는다 — 실기에 없는 상황이다.
        assert is_plant(hand) and f"side:={side}" in hand.argv and "arm:=false" in hand.argv
        assert "hand_follow:=jtc" in hand.argv and "hand_start:=path" in hand.argv
    stops = {k for c in book.commands["shutdown"] if c.stop for k in c.stop}
    assert {"hand_right#1", "hand_left#1", "drivers#0"} <= stops           # fake 에서도 손을 팔마다 내린다


def test_preflight_runs_only_the_real_deployment_tests_not_the_whole_suite():
    # 09.22 사용자: "preflight 는 쓸데없이 오래 걸리게 만든 것 같다" — 전체 785 개(5 분 반)를 실기 켤 때마다 돌렸다.
    argv = _cmds("preflight")[0].argv
    files = [a for a in argv if a.endswith(".py")]
    assert files and all("/tests/" in f for f in files)
    assert not any(a.rstrip("/").endswith("tests/policy_control") for a in argv)      # 디렉터리 통째가 아니다
    for f in files:
        assert Path(f.replace("{repo}", str(PATH.parents[1]))).is_file(), f


def test_the_drivers_stage_ends_by_checking_that_the_motors_answer_on_can():
    # 09.22: can0 · can1 RX 0 · ERROR-PASSIVE 인데 브링업이 "activated" 를 찍고 관절 상태를 전부 0.0 으로 냈다.
    last = " ".join(_cmds("drivers")[-1].argv)
    assert "check_can_rx.py" in last and "can0" in last and "can1" in last
    (sensors,) = [" ".join(c.argv) for c in _cmds("sensors") if "perception_ctl.py" in " ".join(c.argv)]
    assert "--wait" in sensors                                                        # 인지 기동 실패를 기다려 본다


def test_only_the_object_pose_crosses_from_the_vision_pc():
    """09.26 사용자: "vision3090 에서 fpp 를 통해서 받은 것만 노드로 5090 에 넘겨주면 되는거 아닌가" —
    영상 · FP++ 는 저 PC 안에서만(localhost) 돌고, 자세는 UDP 로 와서 이 PC 의 수신기가 다시 낸다."""
    cmds = _cmds("sensors")
    rx = [i for i, c in enumerate(cmds) if any(a.endswith("fpp_pose_rx.py") for a in c.argv)]
    ctl = [i for i, c in enumerate(cmds) if any(a.endswith("perception_ctl.py") for a in c.argv)]
    assert len(rx) == 1 and cmds[rx[0]].background and rx[0] < ctl[0]              # 받을 준비가 먼저
    key = f"sensors#{rx[0]}"
    for stage in ("sensors_off", "shutdown"):
        assert any(key in c.stop for c in _cmds(stage) if c.stop), stage              # 끌 때 같이 내린다
    common = (PATH.parents[1] / "scripts/vision/common.sh").read_text(encoding="utf-8")
    fpp = (PATH.parents[1] / "scripts/vision/fpp_up.sh").read_text(encoding="utf-8")
    assert "export ROS_LOCALHOST_ONLY=1" in common and "-e ROS_LOCALHOST_ONLY=1" in fpp


def test_every_execute_flag_is_one_the_tool_actually_takes():
    """실행 때 덧붙는 `execute_args` 의 플래그를 그 도구가 받는가 — 09.25 fake: 읽기 전용 plan_hand_fold 에
    `--execute` 를 붙여 두어 argparse 가 rc 2 로 거부했고, 실기 미션도 같은 자리에서 멈췄을 것이다.
    도구를 띄우지 않고 소스의 add_argument 로 본다(몇몇은 rclpy 를 import 한다)."""
    repo = PATH.parents[1]
    bad = []
    for stage, cmds in BOOK.commands.items():
        for c in cmds:
            if not c.execute_args:
                continue
            tool = next((a for a in c.argv if a.endswith(".py")), None)
            if tool is None:
                continue
            src = Path(tool.replace("{repo}", str(repo))).read_text(encoding="utf-8")
            bad += [(stage, Path(tool).name, f) for f in c.execute_args
                    if f.startswith("--") and f'"{f}"' not in src and f"'{f}'" not in src]
    assert not bad, bad

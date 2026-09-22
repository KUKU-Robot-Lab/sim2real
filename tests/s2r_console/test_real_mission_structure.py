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
        # engage·제자리 → 확인 → 실측 재계획 → Isaac 미리보기 → 확인 → 경로 재생 → 정착 → 도착 확인 → 손(09.22)
        assert "--only pd_engage --hold-s 10" in home[0] and cmds[1].manual
        # 저장 경로만 쓴다 — 실기에서 다시 계산하지 않는다(09.22 사용자). 시작점 확인 → 미리보기 → 확인 → 재생
        assert "check_path_start.py" in home[2] and f"{{artifact:path_{side}}}" in home[2]
        assert "preview_path_in_viewer.py" in home[3] and "--with-fixed" not in home[3] and cmds[4].manual
        assert "replay_to_pd.py" in home[5] and f"{{artifact:path_{side}}}" in home[5] and "--reverse" not in home[5]
        assert not any("plan_home" in a for c in cmds for a in c.argv)
        assert "--only pd_goto_home" in home[6] and "--service-timeout 45" in home[6]      # 정착만 — 도착은 재생이 했다
        assert cmds[7].manual and "--only pd_hand_home" in home[8]
        ret = _cmds(f"return_{side}")
        assert "--only pd_hand_rest" in " ".join(ret[0].argv)                               # 손을 출발 자세로 먼저
        back = " ".join(ret[1].argv)
        assert "--reverse" in back and f"{{artifact:path_{side}}}" in back                    # 같은 경로를 되짚는다
        assert f"path_{side}" in MISSION.stages[IDS.index(f"home_{side}")].artifacts          # 승인 근거 해시에 들어간다
        assert IDS.index(f"return_{side}") < IDS.index(f"release_{side}")
        assert "--joints" not in " ".join(_cmds(f"selftest_{side}")[0].argv)          # 초기 자세는 팔꿈치가 굽어 7관절 모두


def test_the_head_publisher_starts_only_after_head_home_released_the_port():
    cmds = _cmds("head_home")
    assert "head_home.py" in cmds[0].argv[-1] and not cmds[0].background
    assert "head_joint_publisher.py" in cmds[1].argv[-1] and cmds[1].background
    assert not any("head_joint_publisher" in a for s in IDS if s != "head_home" for c in _cmds(s) for a in c.argv)


def test_the_left_arm_never_starts_a_second_pd_node_beside_the_right_one():
    assert "pd_arm_right#1" in _cmds("pd_load_left")[0].stop
    assert set(_cmds("release_right")[-1].stop) == {"fabric_direct_right#0", "pd_arm_right#1"}


def test_shutdown_is_last_confirms_support_first_and_drops_the_arm_last():
    assert IDS[-1] == "shutdown"
    cmds = _cmds("shutdown")
    assert cmds[0].manual and "받침" in cmds[0].note
    stops = [c.stop for c in cmds if c.stop]
    arm = _launches("drivers", "openarm.bimanual.launch.py")[0]
    hands = _launches("drivers", "dg5f_driver")
    assert stops[-1] == (f"drivers#{arm}",) and set(stops[-2]) == {f"drivers#{i}" for i in hands}


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
    launches = [c for st, cmds in book.commands.items() if st != "drivers" for c in cmds if c.argv[:2] == ("ros2", "launch")]
    assert launches and all("fake:=true" in c.argv for c in launches)     # fake_plant 는 스스로 도메인 0 을 거부한다
    plant = book.commands["drivers"][0]
    assert "hand_follow:=jtc" in plant.argv and "hand_start:=zero" in plant.argv   # 손은 pd 의 JTC 를 따른다


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
    sensors = " ".join(_cmds("sensors")[1].argv)
    assert "--wait" in sensors                                                        # 인지 기동 실패를 기다려 본다

"""컵 좌표 다시 찍기 도구(fpp_rescan) — 언제 명령을 보내고 언제 끝났다고 보는지, 출력 판정(순수).

10.08 사용자: 정책을 돌리면 컵 위치가 바뀐다 — 도커를 껐다 켜지 않고 상황판에서 지금 카메라로 좌표를 다시 뽑는다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))

import fpp_rescan as R  # noqa: E402


def test_the_first_snapshot_still_running_is_waited_for_not_restarted():
    """FP++ 를 막 켰으면 컨테이너가 이미 찍고 있다 — 또 명령하면 18 s 를 버린다."""
    assert R.first_action(None) == "wait_status"
    assert R.first_action({"generation": 0, "pending": ["cyl60"]}) == "wait_round"
    assert R.first_action({"generation": 0, "pending": []}) == "wait_round"
    assert R.first_action({"generation": 2, "pending": []}) == "command"
    assert R.first_action({"generation": 2, "pending": ["cyl60"]}) == "command"


def test_a_round_is_done_when_a_newer_pass_found_every_asked_cup():
    """회차(generation) = 찍기 한 바퀴. 물어본 컵만 본다 — 묶음의 다른 컵(핑크)이 테이블에 없어도 끝난다."""
    found = {"cyl60": {"found": True}, "cyl60_blue": {"found": True}, "cyl60_pink": {"found": False}}
    assert not R.round_done({"generation": 2, "objects": found}, after=2, names=["cyl60"])
    assert R.round_done({"generation": 3, "objects": found}, after=2, names=["cyl60", "cyl60_blue"])
    assert not R.round_done({"generation": 3, "objects": found}, after=2, names=["cyl60", "cyl60_pink"])


def test_each_cup_row_is_checked_against_the_table_and_its_side():
    ok = R.check_cup("cyl60", (0.247, -0.139, 0.290), 0.7)
    assert ok == []
    assert any("z" in w for w in R.check_cup("cyl60_blue", (0.249, 0.128, 0.302), 0.5))   # 1.2 cm 높다
    assert any("기울" in w for w in R.check_cup("cyl60", (0.25, -0.14, 0.29), 5.0))
    assert any("x" in w for w in R.check_cup("cyl60", (0.55, -0.14, 0.29), 0.5))


@pytest.mark.parametrize("status, want", [({"objects": {"cyl60": {"found": True}, "cyl60_pink": {"found": False,
                                                                                                "why": "그 색 후보가 없다"}}},
                                           ["cyl60_pink: 그 색 후보가 없다"])])
def test_missing_cups_are_named_with_the_reason(status, want):
    assert R.missing(status, ["cyl60", "cyl60_pink"]) == want


def test_the_height_check_uses_each_objects_origin():
    """10.09 쉐이커 원점 = 높이 가운데(바닥 위 65 mm) — cyl60 값(85 mm)으로 보면 2 cm 낮다고 잘못 경고한다."""
    assert R.check_cup("shaker_c_orange", (0.25, 0.08, 0.270), 0.5, origin_above_bottom=0.065) == []
    assert any("z" in w for w in R.check_cup("shaker_c_orange", (0.25, 0.08, 0.270), 0.5))


def test_statuses_from_several_group_containers_are_tracked_per_group():
    """10.09 오른손 병(fpp_source200) · 왼손 쉐이커(fpp_shaker_c) — 두 컨테이너가 같은 상태 토픽에 낸다.
    묶음마다 회차를 따로 보고, 물어본 물체가 든 묶음이 모두 새 회차에서 찾았을 때 끝."""
    st = {}
    R.merge_status(st, {"group": "source200", "generation": 2, "objects": {"source200_pink": {"found": True}}})
    R.merge_status(st, {"group": "shaker_c", "generation": 1, "objects": {"shaker_c_orange": {"found": True}}})
    names = ["source200_pink", "shaker_c_orange"]
    assert R.groups_for(st, names) == {"source200": ["source200_pink"], "shaker_c": ["shaker_c_orange"]}
    after = {"source200": 2, "shaker_c": 1}
    assert not R.all_done(st, after, names)
    R.merge_status(st, {"group": "source200", "generation": 3, "objects": {"source200_pink": {"found": True}}})
    assert not R.all_done(st, after, names)
    R.merge_status(st, {"group": "shaker_c", "generation": 2, "objects": {"shaker_c_orange": {"found": True}}})
    assert R.all_done(st, after, names)
    assert R.missing_names(st, ["source200_pink", "cyl60"]) == ["cyl60"]      # 어느 컨테이너에도 없다


def _two_groups(blue_found: bool, orange_gen: int = 2, orange_found: bool = False) -> dict:
    st = {}
    R.merge_status(st, {"group": "source240b", "generation": 3,
                        "objects": {"source240b_blue": {"found": blue_found, "why": "그 색 후보가 없다"}}})
    R.merge_status(st, {"group": "shaker_c", "generation": orange_gen,
                        "objects": {"shaker_c_orange": {"found": orange_found, "why": "그 색 후보가 없다"}}})
    return st


def test_one_found_cup_is_enough_by_default_and_the_other_is_named_with_why():
    """★10.13 사용자 실기: 오렌지 · 블루 중 하나가 테이블에 없어 단계가 실패했다 — "둘중 하나라도 인지되게"."""
    names = ["source240b_blue", "shaker_c_orange"]
    after = {"source240b": 2, "shaker_c": 1}
    st = _two_groups(blue_found=True)
    assert not R.all_done(st, after, names)
    assert R.found_names(st, after, names) == ["source240b_blue"]
    assert R.enough_done(st, after, names)
    assert not R.enough_done(st, after, names, require_all=True)          # --require-all 은 예전처럼 둘 다
    assert R.lost_reasons(st, names) == ["shaker_c_orange: 그 색 후보가 없다"]


def test_one_found_cup_waits_until_every_asked_group_has_a_new_round():
    """다른 묶음이 아직 새 회차를 안 끝냈으면(찍는 중) 그 물체가 곧 찾아질 수 있다 — 기다린다."""
    names = ["source240b_blue", "shaker_c_orange"]
    st = _two_groups(blue_found=True, orange_gen=1)
    assert not R.enough_done(st, {"source240b": 2, "shaker_c": 1}, names)


def test_nothing_found_is_not_enough():
    names = ["source240b_blue", "shaker_c_orange"]
    st = _two_groups(blue_found=False)
    assert R.found_names(st, {"source240b": 2, "shaker_c": 1}, names) == []
    assert not R.enough_done(st, {"source240b": 2, "shaker_c": 1}, names)


def test_a_cup_whose_container_never_came_up_is_reported_as_absent():
    """묶음 컨테이너가 안 떠 상태 자체가 없는 물체도 이유와 함께 찍는다."""
    st = {}
    R.merge_status(st, {"group": "source240b", "generation": 3, "objects": {"source240b_blue": {"found": True}}})
    names = ["source240b_blue", "shaker_c_orange"]
    lost = R.lost_reasons(st, names)
    assert len(lost) == 1 and lost[0].startswith("shaker_c_orange: 상태 없음")
    assert not R.enough_done(st, {"source240b": 2}, names)               # 상태 없는 물체가 있으면 grace 로 따로 본다


def test_the_status_phase_runs_on_the_perception_pc_when_fpp_lives_there():
    """10.13 실기(5090 + vision-3090): FP++ 상태 토픽은 인지 PC 의 localhost DDS 에만 있다 — 5090 에서 기다리면 120 s 뒤 실패했다."""
    remote = R.status_argv(["source240b_blue", "shaker_c_orange"], 120.0, False, "vision-3090")
    assert remote[0] == "ssh" and "BatchMode=yes" in remote and remote[-2] == "vision-3090"
    cmd = remote[-1]
    assert cmd.startswith(f"cd {R.REMOTE_SIM2REAL} && source scripts/vision/common.sh && ")
    assert "fpp_rescan.py source240b_blue shaker_c_orange --wait 120 --phase status" in cmd
    assert "--status-host" not in cmd and "--require-all" not in cmd
    assert "--require-all" in R.status_argv(["a"], 5, True, "h")[-1]
    local = R.status_argv(["cyl60"], 120.0, True)
    assert local[0] == sys.executable and local[-4:] == ["120", "--phase", "status", "--require-all"]


def _proc(tmp_path, *cmdlines):
    for i, argv in enumerate(cmdlines, start=100):
        d = tmp_path / str(i)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (tmp_path / "self").mkdir()
    return tmp_path


def test_without_a_status_host_the_launchers_perception_pc_is_used(tmp_path):
    """10.13 콘솔은 run 시작 때 명령을 고정한다 — 옛 명령(--status-host 없음)으로 cups 를 다시 실행해도 인지 PC 에서 상태를 본다."""
    launcher = ["python3", "/x/scripts/nodes/perception_launcher_node.py"]
    other = ["python3", "/x/scripts/nodes/fpp_pose_rx.py", "--host", "nope"]
    assert R.launcher_host(_proc(tmp_path / "a", other, [*launcher, "--host", "vision-3090", "--pose-dest", "1.2.3.4"])) == "vision-3090"
    assert R.launcher_host(_proc(tmp_path / "b", [*launcher, "--host=box"])) == "box"
    assert R.launcher_host(_proc(tmp_path / "c", launcher)) == R.LAUNCHER_DEFAULT_HOST
    assert R.launcher_host(_proc(tmp_path / "d", [*launcher, "--host", "local"])) == ""       # arm4090: 같은 PC
    assert R.launcher_host(_proc(tmp_path / "e", other)) == ""                                 # 런처 없음 — 이 PC 의 DDS


def test_the_default_host_matches_the_launchers_default():
    import re
    src = (Path(__file__).resolve().parents[1] / "scripts/nodes/perception_launcher_node.py").read_text(encoding="utf-8")
    assert re.search(r'add_argument\("--host", default="([^"]+)"', src).group(1) == R.LAUNCHER_DEFAULT_HOST

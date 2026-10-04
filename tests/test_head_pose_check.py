"""head_pose_check — FP++ 전에 머리 자세를 카메라 캘리브 자세와 맞춘다(10.04 사용자). 모터 없이 순수부만."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "calib"))

from head_pose_check import MAX_GOAL_OFFSET, align, load_head_pose, next_goal  # noqa: E402

ARM4090 = REPO / "config" / "global_camera_extrinsics_arm4090.yaml"


class _Motor:
    """목표보다 offset 틱 앞에서 멈추는 머리 모터(10.04 실측: pan 목표 2015 → 2033 = +18). stuck 이면 움직이지 않는다."""

    def __init__(self, goal: int, offset: int, stuck: bool = False):
        self.goal, self.offset, self.stuck = goal, offset, stuck
        self.present = goal + offset

    def write(self, g: int) -> None:
        self.goal = g
        if not self.stuck:
            self.present = g + self.offset


def _run(motors: dict, targets: dict, tol=4, max_iter=6):
    return align(lambda i: motors[i].present, lambda i: motors[i].goal, lambda i, g: motors[i].write(g),
                 targets, tol, settle_s=0.0, max_iter=max_iter, sleep=lambda s: None, log=lambda s: None)


def test_next_goal_moves_by_the_error_and_stays_near_the_target():
    assert next_goal(2015, 2051, 2033) == 1997                    # 18 틱 앞에서 멈춤 → 목표를 18 틱 당긴다
    assert next_goal(2015, 2015, 2033) == 2033
    assert next_goal(1900, 2300, 2033) == 2033 - MAX_GOAL_OFFSET  # 캘리브 자세 ± MAX_GOAL_OFFSET 밖으로는 안 간다


def test_align_lands_a_motor_that_stops_short_of_its_goal_on_the_calibration_pose():
    """head_home 목표(pan 2015)로는 2033 에 멈추던 머리 — 캘리브 자세가 2051 이어도(카메라를 그 자리에서 쟀으면) 닿는다."""
    m = {1: _Motor(2015, 18), 2: _Motor(2864, 1)}
    assert _run(m, {1: 2051, 2: 2865})
    assert abs(m[1].present - 2051) <= 4 and abs(m[2].present - 2865) <= 4
    assert m[1].goal == 2033                                       # 목표는 캘리브 자세 − 멈춤 오차


def test_align_already_in_place_writes_nothing():
    m = {1: _Motor(2015, 18)}
    m[1].write = lambda g: pytest.fail("이미 맞는 자세에 목표를 보냈다")
    assert _run(m, {1: 2033})


def test_align_gives_up_on_a_stuck_head():
    m = {1: _Motor(2015, 18, stuck=True)}
    m[1].present = 2100
    assert not _run(m, {1: 2033}, max_iter=3)


def test_the_arm4090_calibration_records_its_head_pose(tmp_path):
    hp = load_head_pose(ARM4090)
    assert hp is not None and hp.ticks == {"pan": 2033, "tilt": 2865} and hp.tol_tick == 4
    bare = tmp_path / "cam.yaml"
    bare.write_text("camera:\n  position: [0, 0, 0]\n")
    assert load_head_pose(bare) is None                            # 머리 자세를 적지 않은 캘리브 → 확인 도구가 rc 2


def test_calibration_write_updates_or_adds_the_head_pose_block():
    from table_cad_extrinsics import update_head_pose_yaml
    text = ARM4090.read_text()
    out = update_head_pose_yaml(text, {"pan": 2040, "tilt": 2866})
    hp = out[out.index("head_pose:"):]
    assert "pan_tick: 2040       # −1.27°" in hp and "tilt_tick: 2866" in hp and "tol_tick: 4" in hp
    assert out.replace("2040", "2033").replace("tilt_tick: 2866", "tilt_tick: 2865") == text   # 다른 줄 그대로
    plain = "camera:\n  frame: x\n  position: [0, 0, 0]\n\ndepth_bias:\n  offset_m: 0.0\n"
    added = update_head_pose_yaml(plain, {"pan": 2033, "tilt": 2865})
    assert added.index("head_pose:") < added.index("depth_bias:") and "  pan_tick: 2033\n" in added
    with pytest.raises(ValueError):
        update_head_pose_yaml("head_pose:\n  pan_tick: 1\n", {"pan": 2, "tilt": 3})

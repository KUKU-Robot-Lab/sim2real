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

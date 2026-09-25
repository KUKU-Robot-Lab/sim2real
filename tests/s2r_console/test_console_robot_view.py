"""로봇 상태 표 — 09.23 사용자 "joint state 들에 대한 정보가 표가 있으면 좋겠는데".

그날 손가락이 계약 홈(굽힘 관절 하한 0.0)으로 밀려 꺾였는데 화면에는 그 값이 없었다.
표는 목표와의 차이와 **한계 끝점**을 같이 보여준다.
"""
from __future__ import annotations

import pytest

import s2r_console._paths  # noqa: F401
from s2r_console import robot_view as R

pytestmark = pytest.mark.unit

SIDES = {"right": {"arm_joints": [f"r_aj_{i}" for i in range(1, 8)],
                   "home_arm": [-1.1974, 0.6707, 0.1866, 1.731, 0.692, 0.0416, 0.946],
                   "hand_joints": ["r_hj_index_2", "r_hj_thumb_2"],
                   "home_hand": {"r_hj_index_2": 0.0, "r_hj_thumb_2": -1.57}}}
LIMITS = {"r_hj_index_2": (0.0, 2.007129), "r_hj_thumb_2": (-2.70526, 0.0), "r_aj_1": (-3.0, 3.0)}


def test_a_finger_pushed_to_its_lower_limit_is_marked():
    j = {"r_hj_index_2": [0.005, 0.0, 1.2]}                       # 하한 0.0 바로 위
    v = R.view(j, SIDES, LIMITS, age_s=0.2)
    hand = [g for g in v["groups"] if g["title"] == "오른손"][0]
    row = [r for r in hand["rows"] if r["joint"] == "r_hj_index_2"][0]
    assert row["state"] == "limit" and row["effort"] == pytest.approx(1.2)


def test_a_joint_far_from_its_target_is_marked_off():
    j = {"r_hj_index_2": [0.47, 0.0, 2.0]}                        # 09.23 손이 멈춘 그 값
    hand = [g for g in R.view(j, SIDES, LIMITS, age_s=0.1)["groups"] if g["title"] == "오른손"][0]
    row = [r for r in hand["rows"] if r["joint"] == "r_hj_index_2"][0]
    assert row["state"] == "off" and row["err"] == pytest.approx(0.47)
    assert hand["worst"] == "r_hj_index_2"


def test_joints_we_never_heard_about_are_missing_not_zero():
    v = R.view({}, SIDES, LIMITS, age_s=None)
    arm = [g for g in v["groups"] if g["title"] == "오른팔"][0]
    assert arm["seen"] == 0 and arm["total"] == 7
    assert all(r["state"] == "missing" and r["pos"] is None for r in arm["rows"])
    assert v["stale"] is True                                     # 값이 오래됐으면 화면이 그렇게 말한다


def test_an_arm_inside_the_settle_tolerance_is_ok():
    j = {f"r_aj_{i}": [v, 0.0, 0.0] for i, v in enumerate(SIDES["right"]["home_arm"], 1)}
    arm = [g for g in R.view(j, SIDES, LIMITS, age_s=0.1)["groups"] if g["title"] == "오른팔"][0]
    assert arm["seen"] == 7 and all(r["state"] == "ok" for r in arm["rows"])
    assert abs(arm["worst_err"]) < 1e-9


def test_the_left_side_is_skipped_when_the_contract_has_no_left():
    titles = [g["title"] for g in R.view({}, SIDES, LIMITS, age_s=0.1)["groups"]]
    assert titles == ["오른팔", "오른손"]


def test_every_channel_is_carried_so_the_screen_can_choose():
    # 09.23 사용자: "디폴트는 joint state 고, vel 이나 effort 들도" · "관절값하고 온도값들도 보이면 좋겠는데"
    j = {"r_aj_1": [-1.19, 0.02, -2.65, 35.0, 29.0]}
    arm = [g for g in R.view(j, SIDES, LIMITS, age_s=0.1)["groups"] if g["title"] == "오른팔"][0]
    row = arm["rows"][0]
    assert row["vals"]["pos"] == pytest.approx(-1.19) and row["vals"]["eff"] == pytest.approx(-2.65)
    assert row["vals"]["temp"] == pytest.approx(35.0) and row["vals"]["mos"] == pytest.approx(29.0)
    assert [c["key"] for c in R.view(j, SIDES, LIMITS, age_s=0.1)["channels"]] == \
           ["pos", "vel", "eff", "temp", "mos"]
    assert [c["unit"] for c in R.view(j, SIDES, LIMITS, age_s=0.1)["channels"]][3:] == ["°C", "°C"]


def test_a_channel_the_driver_does_not_send_stays_empty_not_zero():
    # 손끝 촉각을 붙이기 전이다 — 없는 값을 0 으로 그리면 "닿지 않았다" 로 읽힌다
    j = {"r_aj_1": [-1.19]}
    row = [g for g in R.view(j, SIDES, LIMITS, age_s=0.1)["groups"] if g["title"] == "오른팔"][0]["rows"][0]
    assert row["vals"]["pos"] == pytest.approx(-1.19)
    assert row["vals"]["vel"] is None and row["vals"]["eff"] is None


def test_adding_a_channel_needs_only_the_table_at_the_top():
    # 새 채널(촉각)은 CHANNELS 에 한 줄 더하고 브리지가 그 값을 보내면 끝이다
    names = [c[0] for c in R.CHANNELS]
    assert names[0] == "pos"                                   # 기본은 관절 위치
    assert len(R.CHANNELS[0]) == 3                             # (키, 이름, 단위)


def test_source_joint_names_are_mapped_to_canonical():
    # 09.23 실기: /joint_states 는 원본 이름(openarm_right_joint1)으로 온다 — 표가 값을 못 찾아 전부 '—' 였다.
    alias = {"openarm_right_joint1": "r_aj_1", "rj_dg_2_2": "r_hj_index_2"}
    raw = {"openarm_right_joint1": [-1.19, 0.0, -2.65], "rj_dg_2_2": [0.005, 0.0, 1.2]}
    v = R.view(raw, SIDES, LIMITS, age_s=0.1, alias=alias)
    arm = [g for g in v["groups"] if g["title"] == "오른팔"][0]
    assert arm["rows"][0]["pos"] == pytest.approx(-1.19) and arm["seen"] == 1
    hand = [g for g in v["groups"] if g["title"] == "오른손"][0]
    assert [r for r in hand["rows"] if r["joint"] == "r_hj_index_2"][0]["state"] == "limit"


def test_canonical_names_still_work_without_an_alias():
    v = R.view({"r_aj_1": [-1.19]}, SIDES, LIMITS, age_s=0.1)
    assert [g for g in v["groups"] if g["title"] == "오른팔"][0]["rows"][0]["pos"] == pytest.approx(-1.19)

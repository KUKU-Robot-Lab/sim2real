"""★10.19 사용자 "1번 빈손(mimic 비율) · 2번 src240 · 3번 shaker 를 잡았을 때 관절값 · 힘" — tools/rh56f1_finger_meas.py 순수 부분."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy" / "policy_control" / "tools"))
import rh56f1_finger_meas as M  # noqa: E402
from policy_control import rh56f1_map  # noqa: E402


@pytest.fixture(scope="module")
def hmap():
    return rh56f1_map.load()


def test_pd_route_targets_are_clipped_like_pd(hmap):
    assert M.pd_safe(hmap, "thumb_2", 0.0) == pytest.approx(0.05)                 # pd hand_safe_target 한계 여유
    assert M.pd_safe(hmap, "index_1", 1.45) == pytest.approx(1.45)                # 스윕 끝 · 쥐기 목표는 여유 안
    assert M.pd_safe(hmap, "thumb_2", max(M.THUMB2_LEVELS)) == pytest.approx(max(M.THUMB2_LEVELS))
    from policy_control import pd_backends
    assert (M.HAND_LIMIT_MARGIN, M.HAND_MAX_LEAD) == (pd_backends.HAND_LIMIT_MARGIN, pd_backends.HAND_MAX_LEAD)


def test_route_choice_never_fights_pd():
    assert M.choose_route("auto", True, 1) == ("pd", "")                          # home 뒤 pd 가 손을 붙든다 → pd 경로
    assert M.choose_route("auto", False, 0) == ("direct", "")                     # 손 드라이버만
    assert M.choose_route("auto", False, 1)[0] is None                            # 누군가 angle_set 을 쓰는데 pd 상태가 없다
    assert M.choose_route("direct", True, 1)[0] is None                           # pd 가 붙든 손에 직접 쓰지 않는다
    assert M.choose_route("pd", False, 0)[0] is None


def test_pd_fault_matches_hand_grasp():
    assert M.pd_fault({}, 10.0) == "pd status 없음"
    assert M.pd_fault({"phase": "TRACKING", "_t": 9.5}, 10.0) is None
    assert "HOLD" in M.pd_fault({"phase": "HOLD", "_t": 9.9}, 10.0)
    assert M.pd_fault({"phase": "TRACKING", "_t": 8.0}, 10.0) == "pd status 없음"   # 오래된 상태


def test_levels_outside_the_map_range_are_refused(hmap):
    M.check_levels(hmap, "left", "thumb_2", M.THUMB2_LEVELS)
    with pytest.raises(M.MeasError):
        M.check_levels(hmap, "left", "thumb_2", [0.6])                       # 엄지 굽힘 끝 0.4746
    with pytest.raises(M.MeasError):
        M.parse_levels("0,a", M.FINGER_LEVELS)
    assert M.parse_levels(None, M.FINGER_LEVELS) == M.FINGER_LEVELS
    assert {0.3, 0.9} <= set(M.FINGER_LEVELS)                                # 제작기: 0.3~0.9 rad 구간 점


def test_grip_target_leads_the_measurement_like_pd():
    assert M.lead_clamped(1.45, 0.70, 0.2) == pytest.approx(0.90)            # 막힌 손가락 — 미는 간극은 pd 앞섬 상한만큼
    assert M.lead_clamped(0.30, 0.20, 0.2) == pytest.approx(0.30)            # 아직 램프가 실측 근처


def test_still_needs_a_full_window():
    hist = [(0.0, 1300), (0.2, 1301), (0.4, 1302), (0.6, 1302)]
    assert M.still(hist, 0.6, 0.5)
    assert not M.still(hist, 0.3, 0.5)                                      # 이력이 창보다 짧다
    assert not M.still([(0.0, 1300), (0.2, 1295), (0.4, 1290), (0.6, 1285)], 0.6, 0.5)   # 아직 닫히는 중


def test_plan_only_and_domain_guard(monkeypatch, capsys):
    monkeypatch.setenv("ROS_DOMAIN_ID", "0")
    assert M.main(["sweep", "--side", "right"]) == 1
    monkeypatch.setenv("ROS_DOMAIN_ID", "97")
    assert M.main(["sweep", "--side", "right"]) == 0                         # --execute 없으면 ROS 를 열지 않는다
    assert "계획만" in capsys.readouterr().out
    assert M.main(["grip", "--side", "left", "--close-to", "2.0"]) == 1      # 변환표 범위 밖
    assert M.main(["sweep", "--side", "left", "--axes", "index_2"]) == 1    # 종속 마디는 구동 축이 아니다


def test_axis_view_uses_the_side_calibration(hmap):
    reg = [1740, 1740, 1740, 1340, 1350, 865]
    v = M.axis_view(hmap, "right", reg, [1, 2, 3, 4, 5, 6], None)
    a = M.axis_of(hmap, "right", "index_1")
    assert v["index_1"]["rad"] == pytest.approx(a.to_rad(1340), abs=1e-4) and v["index_1"]["force_g"] == 4
    assert v["thumb_1"]["current"] is None

"""head_view_feedback 순수부 — 중력 각 · 외부 파라미터 각 · 다음 tilt 계산."""
import math

import numpy as np
import pytest

import head_view_feedback as hv


def test_gravity_angles_level_and_down():
    assert hv.gravity_angles(0, -9.81, 0) == pytest.approx((0.0, 0.0), abs=1e-9)   # 정면 수평
    assert hv.gravity_angles(0, 0, -9.81)[0] == pytest.approx(90.0)                # 바로 아래
    p, _ = hv.gravity_angles(0, -9.81 * math.cos(math.radians(30)), -9.81 * math.sin(math.radians(30)))
    assert p == pytest.approx(30.0)


def test_gravity_rejects_non_gravity():
    with pytest.raises(ValueError):
        hv.gravity_angles(0.1, 0.1, 0.1)


def test_extrinsics_matches_imu_definition():
    # 광축이 base 앞(+x)을 보고 30° 숙인 카메라: 광학 z = (cos30, 0, -sin30), x = (0,-1,0), y = z × x
    z = np.array([math.cos(math.radians(30)), 0, -math.sin(math.radians(30))])
    x = np.array([0.0, -1.0, 0.0])
    R = np.column_stack([x, np.cross(z, x), z])
    w = math.sqrt(1 + np.trace(R)) / 2
    q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    pitch, roll = hv.extrinsics_angles(q)
    assert pitch == pytest.approx(30.0, abs=1e-6) and roll == pytest.approx(0.0, abs=1e-6)


def test_arm4090_extrinsics_pitch_is_plausible():
    pitch, roll = hv.load_extrinsics_pitch()
    assert 60 < pitch < 80 and abs(roll) < 5


def test_next_tilt_clamps_step_and_limits():
    lim = hv.Limits(min_tilt=-8, max_tilt=74.8, max_step=10)
    assert hv.next_tilt(61.3, 60.0, 1.0, lim) == pytest.approx(51.3)       # 큰 오차는 10° 씩
    assert hv.next_tilt(5.0, 1.0, 1.0, lim) == pytest.approx(4.1)          # 가까우면 90 %
    assert hv.next_tilt(-7.0, 5.0, 1.0, lim) == -8                          # 한계에서 멈춤
    assert hv.next_tilt(70.0, -3.0, 1.0, lim) == pytest.approx(73.0)        # 덜 숙였으면 숙인다


def test_ratio_from_uses_fallback_for_tiny_moves():
    assert hv.ratio_from((71.2, 72.13), (61.32, 62.47), 1.0) == pytest.approx(0.978, abs=1e-3)
    assert hv.ratio_from((61.3, 62.0), (61.4, 62.5), 0.98) == 0.98


def test_sweep_targets_reach_limit():
    lim = hv.Limits(min_tilt=-8, max_tilt=74.8)
    t = hv.sweep_targets(51.2, -5, lim)
    assert t[0] == pytest.approx(46.2) and t[-1] == -8 and len(t) == 12
    assert hv.sweep_targets(-8, -5, lim) == []


def test_grid_plan_up_then_down_per_pan():
    plan = hv.grid_plan([-10, 0], [70, 40, 10])
    assert plan[:5] == [(-10, 70, "up"), (-10, 40, "up"), (-10, 10, "up"), (-10, 40, "down"), (-10, 70, "down")]
    assert len(plan) == 10 and plan[5] == (0, 70, "up")


def test_view_error_sign_and_tolerance():
    assert hv.view_error(68.0, -0.54, 68.25, 1.0) == (pytest.approx(0.29), True)    # 0점 보정 뒤 기준보다 0.29° 더 숙임
    err, ok = hv.view_error(66.5, -0.54, 68.25, 1.0)
    assert err == pytest.approx(-1.21) and not ok                                   # 1° 넘게 들림 → 실패
    assert hv.view_error(69.25, 0.0, 68.25, 1.0)[1]                                 # 경계는 통과


def test_defaults_follow_the_camset_mission():
    """기준 외부 파라미터 = 미션이 FP++ 에 쓰는 것, 검사 허용치는 설정에 있다."""
    import yaml
    import make_rh56f1_missions as gen
    assert hv.DEFAULT_EXTRINSICS == hv.REPO / gen.CAMERA_EXTRINSICS
    cfg = yaml.safe_load(hv.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert 0 < cfg["view_check_tol_deg"] <= 2 and abs(cfg["imu_pitch_bias"]) < 3
    assert abs(cfg["sim_pitch_deg"] - hv.load_extrinsics_pitch()[0]) < cfg["view_check_tol_deg"]   # sim 과 기준이 허용치 안

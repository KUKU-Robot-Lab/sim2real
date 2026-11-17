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


# ---------- ★11.17 기준 파일 비교(--check) ----------

def test_angles_to_up_round_trip():
    for p, r in [(0, 0), (69.07, -0.45), (30, 5), (-10, -20), (89, 3)]:
        up = hv.angles_to_up(p, r)
        assert np.linalg.norm(up) == pytest.approx(1.0)
        assert hv.up_to_angles(up) == pytest.approx((p, r), abs=1e-9)


def test_compare_to_ref_pass_and_each_failure():
    ref, ticks = hv.angles_to_up(69.07, -0.45), (1825, 2833)
    tol = (0.3, 0.3, 4)
    same = hv.compare_to_ref(ref, ref, ticks, ticks, *tol)
    assert same["ok"] and same["angle_deg"] == pytest.approx(0, abs=1e-6)
    roll = hv.compare_to_ref(hv.angles_to_up(69.07, -0.10), ref, ticks, ticks, *tol)   # 11.17 실측: roll 만 0.35° 차
    assert not roll["ok"] and not roll["ok_roll"] and roll["ok_pitch"] and roll["d_roll"] == pytest.approx(0.35)
    tick = hv.compare_to_ref(ref, ref, (1825, 2838), ticks, *tol)
    assert not tick["ok"] and tick["d_tick"] == (0, 5)
    none = hv.compare_to_ref(ref, ref, None, ticks, *tol)
    assert not none["ok"] and not none["ok_tick"] and none["ok_pitch"]
    assert "❌" in hv.ref_report(roll, "ref.yaml").splitlines()[0] and "못 읽음" in hv.ref_report(none, "ref.yaml")


def test_ref_file_matches_the_mission_extrinsics_and_migration():
    import yaml
    ref = yaml.safe_load(hv.DEFAULT_REF.read_text(encoding="utf-8"))
    assert ref["extrinsics"] == {"file": hv.DEFAULT_EXTRINSICS.name, "md5": hv.file_md5(hv.DEFAULT_EXTRINSICS)}
    assert hv.up_to_angles(ref["imu_up_cam"]) == pytest.approx((ref["pitch_raw_deg"], ref["roll_raw_deg"]), abs=0.01)
    cfg = yaml.safe_load(hv.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert {"ref_tol_pitch_deg", "ref_tol_roll_deg", "ref_tol_tick"} <= cfg.keys()
    assert ref["measured"] in (True, False) and ref["approach"] in hv.APPROACH_SIGN
    if not ref["measured"]:   # --record-ref 로 다시 재기 전: 이주 값 = 외부 파라미터 각 + 설정 bias
        ep, er = hv.load_extrinsics_pitch()
        assert hv.up_to_angles(ref["imu_up_cam"]) == pytest.approx((ep + cfg["imu_pitch_bias"], er + cfg["imu_roll_bias"]), abs=0.01)


# ---------- 순수 판정(run_check · record_ref_guard · build_ref · approach_cmd) ----------

TOL = (0.3, 0.3, 4)
UP = hv.angles_to_up(69.07, -0.45)
TICKS = (1825, 2833)


def _ref(**kw):
    return {**hv.build_ref(UP, TICKS, 400, 0.01, "ext.yaml", "abc", "down", "t", "test"), **kw}


def test_run_check_passes_on_a_measured_ref_and_prints_the_approach():
    rc, text = hv.run_check(_ref(), UP, TICKS, 400, 0.01, "abc", TOL)
    assert rc == 0 and "✓" in text.splitlines()[0] and "기준 접근 방향 down" in text


def test_run_check_failures():
    assert hv.run_check(_ref(), UP, TICKS, 400, 0.01, "other", TOL)[0] == 1             # md5 불일치
    assert hv.run_check(None, UP, TICKS, 400, 0.01, "abc", TOL)[0] == 1                 # 기준 없음
    assert hv.run_check({"extrinsics": "x"}, UP, TICKS, 400, 0.01, "abc", TOL)[0] == 1  # 깨진 기준
    rc, text = hv.run_check(_ref(measured=False), UP, TICKS, 400, 0.01, "abc", TOL)      # 실측 아님: 판정 줄은 찍고 실패
    assert rc == 1 and "✓" in text.splitlines()[0] and "기준이 실측이 아님" in text
    assert "흔들림" in hv.run_check(_ref(), UP, TICKS, 400, hv.REF_MAX_SD + 0.01, "abc", TOL)[1]   # σ 초과
    assert hv.run_check(_ref(), UP, TICKS, hv.N_SAMPLES // 2 - 1, 0.01, "abc", TOL)[0] == 1      # 샘플 부족
    assert hv.run_check(_ref(), UP, TICKS, hv.N_SAMPLES // 2, 0.01, "abc", TOL)[0] == 0
    assert hv.run_check(_ref(), UP, (1825, 2838), 400, 0.01, "abc", TOL)[0] == 1                 # 틱 5 차


HP = {"pan_tick": 1825, "tilt_tick": 2833, "tol_tick": 4}


def test_record_ref_guard():
    ok, text = hv.record_ref_guard(UP, TICKS, 400, 0.01, HP, (68.52, 0.27), (0.55, -0.72), False)
    assert ok and "숙임 +0.00°" in text
    assert not hv.record_ref_guard(UP, (1825, 2838), 400, 0.01, HP, (68.52, 0.27), (0.55, -0.72), True)[0]   # 틱 거부는 force 로도
    assert not hv.record_ref_guard(UP, TICKS, 100, 0.01, HP, (68.52, 0.27), (0.55, -0.72), True)[0]          # 샘플 부족
    assert not hv.record_ref_guard(UP, TICKS, 400, 0.1, HP, (68.52, 0.27), (0.55, -0.72), True)[0]           # σ 초과
    ok, text = hv.record_ref_guard(UP, TICKS, 400, 0.01, HP, (68.52, 0.27), (-0.55, -0.72), False)           # 숙임 1.1° 차
    assert not ok and "--force" in text
    assert hv.record_ref_guard(UP, TICKS, 400, 0.01, HP, (68.52, 0.27), (-0.55, -0.72), True)[0]
    assert not hv.record_ref_guard(UP, TICKS, 400, 0.01, HP, (68.52, 1.5), (0.55, -0.72), False)[0]          # 옆 1.23° 차


def test_build_ref_round_trips_through_the_file(tmp_path):
    import yaml
    ref = hv.build_ref(UP * 9.8, TICKS, 397, 0.0123, "ext.yaml", "abc", "up", "2026-11-17 10:00", "test")
    path = tmp_path / "ref.yaml"
    path.write_text("old", encoding="utf-8")
    hv.write_ref(path, ref)
    back = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert back == ref and back["measured"] is True and back["approach"] == "up"
    assert hv.up_to_angles(back["imu_up_cam"]) == pytest.approx((back["pitch_raw_deg"], back["roll_raw_deg"]), abs=1e-3)
    assert hv.ref_problem(back, "abc") is None and len(list(tmp_path.glob("ref.yaml.bak-*"))) == 1


def test_approach_cmd_only_enters_from_the_approach_side():
    lim = hv.Limits(min_tilt=60, max_tilt=74.8, max_step=10)
    assert hv.approach_cmd(70, 72, "down", lim) == (72, False)                          # 숙이며 들어오면 그대로
    assert hv.approach_cmd(70, 68, "down", lim) == (pytest.approx(66.5), True)          # 넘어섰으면 1.5° 더 들었다가
    assert hv.approach_cmd(70, 70, "down", lim, arrived_wrong=True) == (pytest.approx(68.5), True)
    assert hv.approach_cmd(70, 60.5, "down", lim) == (60, True)                         # max_step · 한계 안
    assert hv.approach_cmd(70, 72, "up", lim) == (pytest.approx(73.5), True)
    assert hv.approach_cmd(74, 74.5, "up", lim) == (74.8, True)


def test_execute_extrinsics_refuses_an_unusable_ref_before_ros(tmp_path, capsys):
    """기준이 없거나 measured false 면 bias 로 조용히 내려가지 않고 ROS 를 띄우기 전에 1."""
    assert hv.main(["--execute", "--ref", str(tmp_path / "none.yaml")]) == 1
    assert "기준 파일이 없다" in capsys.readouterr().out
    unmeasured = tmp_path / "ref.yaml"
    hv.write_ref(unmeasured, {**hv.build_ref(UP, TICKS, 400, 0.01, hv.DEFAULT_EXTRINSICS.name,
                                             hv.file_md5(hv.DEFAULT_EXTRINSICS), "down", "t", "test"), "measured": False})
    assert hv.main(["--execute", "--ref", str(unmeasured)]) == 1
    assert "measured false" in capsys.readouterr().out
    assert hv.main(["--check", "--ref", str(tmp_path / "none.yaml")]) == 1


# ---------- ★11.17 고정 마커(ArUco 70 · 71) ----------

K720 = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
DIST0 = np.zeros(5)
ANCHOR_POSES = {70: ([2.5, 0.05, 0.02], [-0.30, 0.12, 0.85]), 71: ([2.5, -0.05, -0.02], [0.30, 0.12, 0.85])}   # 화면을 보고 약 37° 누운 마커


def _scene(poses=ANCHOR_POSES, size=0.08, ss=4):
    """마커를 알려진 T_cam_marker 로 투영한 1280x720 회색 영상(흰 바탕, 마커 둘레 흰 여백). ss 배로 그려 INTER_AREA 로 줄인다
    (한 번에 그리면 가장자리가 0.2 px 넘게 치우쳐 깊이가 5 mm 틀린다)."""
    import cv2
    ar = cv2.aruco
    d = ar.getPredefinedDictionary(ar.DICT_6X6_250)
    Kss = K720.copy()
    Kss[:2] *= ss
    Kss[:2, 2] += (ss - 1) / 2                                   # 픽셀 가운데 규약: 큰 영상 x = ss·x + (ss−1)/2
    img = np.full((720 * ss, 1280 * ss), 255, np.uint8)
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    for i, (rvec, tvec) in poses.items():
        S, pad = 400, 100
        m = cv2.copyMakeBorder(ar.generateImageMarker(d, i, S), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        px, _ = cv2.projectPoints(obj, np.array(rvec), np.array(tvec), Kss, DIST0)
        e0, e1 = pad - 0.5, pad + S - 0.5                        # 검은 부분 가장자리
        src = np.float32([[e0, e0], [e1, e0], [e1, e1], [e0, e1]])
        H = cv2.getPerspectiveTransform(src, px.reshape(4, 2).astype(np.float32))
        img = np.minimum(img, cv2.warpPerspective(m, H, img.shape[::-1], borderValue=255))
    return cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)


def test_detect_anchors_recovers_tvec_to_mm():
    found = hv.detect_anchors(_scene(), K720, DIST0, {70, 71}, 0.08)
    assert set(found) == {70, 71}
    for i, (_, tvec) in ANCHOR_POSES.items():
        assert np.abs(found[i]["tvec"] - tvec).max() < 0.004     # mm 수준(82 px 마커 · 약 0.85 m, 깊이가 가장 크게 틀린다)
    assert hv.detect_anchors(_scene(), K720, DIST0, {70}, 0.08).keys() == {70}   # 설정 id 만


def test_compare_anchors_pass_shift_and_missing():
    img = _scene()
    ref, sd = hv.anchors_from_frames([img] * 3, K720, DIST0, {70, 71}, 0.08)
    assert sd == pytest.approx(0.0)
    tol = (2.0, 3.0, 0.3)
    same = hv.compare_anchors(ref, ref, *tol)
    assert same["ok"] and same["pan_shift_px"] == pytest.approx(0.0)
    shifted = hv.detect_anchors(np.roll(img, 4, axis=1), K720, DIST0, {70, 71}, 0.08)   # 영상이 오른쪽으로 4 px
    a = hv.compare_anchors(shifted, ref, *tol)
    assert not a["ok"] and a["pan_shift_px"] == pytest.approx(4.0, abs=0.2)
    assert all(m["px"] == pytest.approx(4.0, abs=0.2) for m in a["markers"].values())
    gone = img.copy()
    gone[:, 640:] = 255                                                                  # 71(오른쪽) 지움
    b = hv.compare_anchors(hv.detect_anchors(gone, K720, DIST0, {70, 71}, 0.08), ref, *tol)
    assert not b["ok"] and b["markers"][71] == {"seen": False, "ok": False} and b["markers"][70]["ok"]
    assert "71 안 보임" in "\n".join(hv.anchor_report(b))


def test_pan_px_per_deg_matches_a_real_pan_and_sign():
    """숙임 68° 카메라가 세로축으로 왼쪽(+, pan 틱 +) 0.5° 돌면 장면이 오른쪽(+u)으로 pan_px_per_deg 만큼 간다."""
    import cv2
    p = 68.0
    up = hv.angles_to_up(p, 0.0)
    P = np.array([[-0.3, 0.1, 0.9], [0.3, 0.1, 0.9], [-0.25, 0.25, 0.8], [0.0, 0.0, 1.0]])
    uv0 = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K720, DIST0)[0].reshape(-1, 2)
    R = cv2.Rodrigues(-math.radians(0.5) * up)[0]        # 카메라가 +δ 돌면 점은 카메라 프레임에서 −δ
    uv1 = cv2.projectPoints(P @ R.T, np.zeros(3), np.zeros(3), K720, DIST0)[0].reshape(-1, 2)
    measured = (uv1[:, 0] - uv0[:, 0]).mean()
    assert measured > 0
    assert hv.pan_px_per_deg(uv0, K720, p) * 0.5 == pytest.approx(measured, rel=0.03)
    assert hv.pan_px_per_deg([[640, 360]], K720, p) == pytest.approx(900 * math.pi / 180 * math.cos(math.radians(p)))


def test_approach_axis_for_pan():
    assert hv.approach_axis(-19.5, -19.0, 1, -22.5, -16.5, 10) == (-19.0, False)                # 틱 커지며 들어옴
    assert hv.approach_axis(-19.5, -20.0, 1, -22.5, -16.5, 10) == (pytest.approx(-21.5), True)  # 넘어섬 → 더 물림
    assert hv.approach_axis(-19.5, -21.0, 1, -22.5, -16.5, 10) == (-22.5, True)                 # 창 안


def test_run_check_with_anchors(tmp_path):
    img = _scene()
    m, sd = hv.anchors_from_frames([img] * 2, K720, DIST0, {70, 71}, 0.08)
    ref = hv.build_ref(UP, TICKS, 400, 0.01, "e.yaml", "abc", "down", "t", "test", "left", hv.anchors_block(m, 2, sd, 0.08, [1280, 720], 900.0, None))
    path = tmp_path / "ref.yaml"
    hv.write_ref(path, ref)
    import yaml
    back = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert hv.ref_problem(back, "abc") is None and set(back["anchors"]["markers"]) == {70, 71}
    tol, atol = (0.3, 0.3, 4), (2.0, 3.0, 0.3)
    rc, text = hv.run_check(back, UP, TICKS, 400, 0.01, "abc", tol, "r", m, atol)
    assert rc == 0 and "마커 70" in text and "마커 71" in text
    moved = hv.detect_anchors(np.roll(img, 5, axis=1), K720, DIST0, {70, 71}, 0.08)
    assert hv.run_check(back, UP, TICKS, 400, 0.01, "abc", tol, "r", moved, atol)[0] == 1
    assert hv.run_check(back, UP, TICKS, 400, 0.01, "abc", tol, "r")[0] == 1               # 기준에 마커 있는데 영상 없음
    plain = hv.build_ref(UP, TICKS, 400, 0.01, "e.yaml", "abc", "down", "t", "test")
    rc, text = hv.run_check(plain, UP, TICKS, 400, 0.01, "abc", tol, "r")
    assert rc == 0 and "마커 기준 없음" in text                                              # 마커 붙이기 전 호환
    assert hv.ref_problem({**back, "anchors": {"markers": {}}}, "abc") == "기준 anchors 가 깨졌다(markers 없음)"
    assert "approach_pan" in hv.ref_problem({**back, "approach_pan": "up"}, "abc")


def test_ref_problem_checks_anchors_against_the_config():
    m, sd = hv.anchors_from_frames([_scene()] * 2, K720, DIST0, {70, 71}, 0.08)
    ref = hv.build_ref(UP, TICKS, 400, 0.01, "e.yaml", "abc", "down", "t", "test", "left",
                       hv.anchors_block(m, 2, sd, 0.08, [1280, 720], 900.0, None))
    cfg = {"anchor_ids": [70, 71], "anchor_size_m": 0.08}
    assert hv.ref_problem(ref, "abc", cfg) is None
    assert "anchor_ids" in hv.ref_problem(ref, "abc", {**cfg, "anchor_ids": [70, 71, 72]})       # 설정 id 가 기준에 없음
    assert "크기" in hv.ref_problem(ref, "abc", {**cfg, "anchor_size_m": 0.1})
    assert "사전" in hv.ref_problem({**ref, "anchors": {**ref["anchors"], "dict": "DICT_4X4_50"}}, "abc", cfg)
    assert hv.anchor_meta_problem(ref["anchors"], K720, [1280, 720]) is None
    assert "영상 크기" in hv.anchor_meta_problem(ref["anchors"], K720, [640, 480])
    assert "fx" in hv.anchor_meta_problem(ref["anchors"], K720 * [[1.01], [1], [1]], [1280, 720])


class _Frames:
    """grab 이 정해진 영상을 돌려주는 가짜 카메라."""
    def __init__(self, frames):
        self.frames = frames

    def grab(self, n=hv.ANCHOR_FRAMES, timeout_s=5.0):
        assert len(self.frames) == n
        return self.frames, K720, DIST0


def test_record_anchors_stores_self_noise_and_refuses_a_shaky_view(tmp_path):
    import yaml
    img, cfg = _scene(), {"anchor_ids": [70, 71], "anchor_size_m": 0.08}
    anc, why = hv._record_anchors(_Frames([img] * 10), cfg, False)
    assert why is None and anc["n_frames"] == 10 and anc["corner_sd_px"] == 0.0
    assert anc["image_wh"] == [1280, 720] and anc["fx"] == 900.0 and anc["size_m"] == 0.08 and anc["dict"] == hv.ANCHOR_DICT
    assert set(anc["self_noise"]["markers"]) == {70, 71} and anc["self_noise"]["markers"][70]["px"] == pytest.approx(0, abs=1e-6)
    path = tmp_path / "ref.yaml"
    hv.write_ref(path, hv.build_ref(UP, TICKS, 400, 0.01, "e.yaml", "abc", "down", "t", "test", "left", anc))
    assert hv.ref_problem(yaml.safe_load(path.read_text(encoding="utf-8")), "abc", cfg) is None
    shaky = [img] * 5 + [np.roll(img, 2, axis=1)] * 5                    # 뒤 묶음이 2 px 옆으로: σ ≈ 1.05 px
    anc, why = hv._record_anchors(_Frames(shaky), cfg, False)
    assert anc is None and "--force" in why
    anc, why = hv._record_anchors(_Frames(shaky), cfg, True)
    assert why is None and anc["corner_sd_px"] == pytest.approx(math.sqrt(10 / 9), abs=0.02)        # ddof 1 (ddof 0 은 1.0)
    assert anc["self_noise"]["pan_shift_px"] == pytest.approx(2.0, abs=0.2)
    gone = img.copy()
    gone[:, 640:] = 255
    assert "[71]" in hv._record_anchors(_Frames([gone] * 10), cfg, False)[1]


# ---------- _fit_pan: 선형 이득 가짜 목 · 카메라 ----------

REF_M = {70: {"corners": [[280.0, 480.0], [320.0, 480.0], [320.0, 520.0], [280.0, 520.0]],
              "rvec": [2.5, 0.05, 0.02], "tvec": [-0.3, 0.12, 0.85]},
         71: {"corners": [[960.0, 480.0], [1000.0, 480.0], [1000.0, 520.0], [960.0, 520.0]],
              "rvec": [2.5, -0.05, -0.02], "tvec": [0.3, 0.12, 0.85]}}
PAN_CFG = {"anchor_ids": [70, 71], "anchor_size_m": 0.08,
           "ref_tol_anchor_px": 1.0, "ref_tol_anchor_mm": 3.0, "ref_tol_anchor_deg": 0.3}
PAN_REF = {"approach_pan": "left", "anchors": {"markers": REF_M}}
PAN_LIM = hv.Limits(min_tilt=60, max_tilt=74.8, max_step=10, max_iters=6)


class _Rig:
    """sens · cam · head_move 를 겸한다. 마커 가로 이동 = gain · (pan − p0) px. 기하값 pan_px_per_deg ≈ 4.5 px/°."""
    def __init__(self, pan, p0, gain=4.0, hide=(), move_rc=0, stuck=False):
        self.pan, self.p0, self.gain, self.hide, self.move_rc, self.stuck = pan, p0, gain, set(hide), move_rc, stuck
        self.moves = []

    def request_move(self, pan, tilt, max_tilt):
        self.moves.append((pan, tilt))
        if self.move_rc == 0 and not self.stuck:
            self.pan = pan
        return self.move_rc, None

    def read(self):
        return 68.0, 0.0, 0.01, self.pan, 70.0

    def markers(self):
        s = self.gain * (self.pan - self.p0)
        return {i: {**m, "corners": np.asarray(m["corners"]) + [s, 0.0]} for i, m in REF_M.items() if i not in self.hide}


@pytest.fixture
def fake_cam(monkeypatch):
    monkeypatch.setattr(hv, "_read_anchors", lambda cam, cfg: (cam.markers(), 0.0, K720, [1280, 720]))


def _fit(rig):
    return hv._fit_pan(rig, rig, rig, PAN_CFG, PAN_REF, PAN_LIM, -19.0, 70.0, 68.0, 3.0)


@pytest.mark.parametrize("p0", [-17.0, -20.0])
def test_fit_pan_converges_from_the_approach_side(fake_cam, p0):
    rig = _Rig(-19.0, p0)
    rc, pan = _fit(rig)
    assert rc == 0 and pan == pytest.approx(p0, abs=1.0 / rig.gain)
    assert abs(rig.moves[0][0] + 19.0) <= hv.PAN_PROBE_DEG + 1e-9                 # 첫 이동은 1° 이하
    assert all(t == 70.0 for _, t in rig.moves)                                   # tilt 는 마지막 명령 그대로
    assert rig.moves[-1][0] > (rig.moves[-2][0] if len(rig.moves) > 1 else -19.0)  # left = pan 이 커지며 도착


def test_fit_pan_wrong_gain_sign_stops_after_the_probe_and_returns(fake_cam):
    rig = _Rig(-19.0, -21.0, gain=-4.0)
    rc, pan = _fit(rig)
    assert rc == 1 and rig.moves[0][0] == pytest.approx(-18.0)                    # 1° 시험 이동 뒤 부호를 보고 멈춤
    assert [m[0] for m in rig.moves[1:]] == pytest.approx([-20.5, -19.0])          # 넘어선 쪽이라 물렸다가 시작 pan 으로
    assert pan == pytest.approx(-19.0)


def test_fit_pan_refuses_a_goal_outside_the_window(fake_cam):
    rig = _Rig(-19.0, -14.0)
    assert _fit(rig)[0] == 1 and rig.moves == []


def test_fit_pan_needs_every_reference_marker(fake_cam):
    rig = _Rig(-19.0, -17.0, hide={71})
    assert _fit(rig) == (1, -19.0) and rig.moves == []


def test_fit_pan_stops_on_a_failed_move_and_at_max_iters(fake_cam):
    rig = _Rig(-19.0, -17.0, move_rc=1)
    assert _fit(rig)[0] == 1 and len(rig.moves) == 1
    rig = _Rig(-19.0, -17.0, stuck=True)                                           # 요청은 되는데 안 돈다: 이득을 못 잰다
    assert _fit(rig)[0] == 1 and len(rig.moves) == PAN_LIM.max_iters
    assert all(abs(p + 19.0) <= hv.PAN_PROBE_DEG + 1e-9 for p, _ in rig.moves)   # 확인 전이라 끝까지 1° 이하


def test_pan_window_must_be_small_and_positive(capsys):
    for w in ("0", "-1", "5.5"):
        with pytest.raises(SystemExit):
            hv.main(["--pan-window", w])
    assert "--pan-window" in capsys.readouterr().err


def test_execute_does_not_move_when_the_camera_is_silent(tmp_path, monkeypatch, capsys):
    """영상이 안 오면(grab 시간 초과) 머리를 움직이지 않고 1, 안내에 common.sh. 전 grab 의 _bad 가 남아도 시간 초과로 본다."""
    import sys
    import types
    m = {i: {k: np.asarray(v) for k, v in REF_M[i].items()} for i in REF_M}
    ref = tmp_path / "ref.yaml"
    hv.write_ref(ref, hv.build_ref(UP, TICKS, 400, 0.01, hv.DEFAULT_EXTRINSICS.name, hv.file_md5(hv.DEFAULT_EXTRINSICS),
                                   "down", "t", "test", "left", hv.anchors_block(m, 10, 0.1, 0.08, [1280, 720], 900.0, None)))
    import yaml
    cfg = tmp_path / "cfg.yaml"   # REF_M 은 70 · 71 · 80 mm 판
    cfg.write_text(yaml.safe_dump({**yaml.safe_load(hv.DEFAULT_CONFIG.read_text(encoding="utf-8")),
                                   "anchor_ids": [70, 71], "anchor_size_m": 0.08}), encoding="utf-8")
    moves = []
    rclpy = types.SimpleNamespace(init=lambda: None, ok=lambda: False, spin_once=lambda node, timeout_sec: None)
    stubs = {"rclpy": rclpy, "rclpy.node": types.SimpleNamespace(Node=lambda name: types.SimpleNamespace(destroy_node=lambda: None)),
             "head_move": types.SimpleNamespace(request_move=lambda *a, **k: moves.append(a) or (0, None))}
    for k, v in stubs.items():
        monkeypatch.setitem(sys.modules, k, v)

    class Sens:
        _acc = [0] * 400

        def __init__(self, node, rclpy):
            pass

        def read(self):
            return 66.0, -0.45, 0.01, -19.0, 70.0   # 목표(69.07°)와 3° 차: 확인 없이 가면 tilt 부터 움직인다

    def cam(node, rclpy):
        c = object.__new__(cls)
        c._node, c._rclpy, c.K, c.dist, c._grays, c._want, c._bad = node, rclpy, None, None, [], 0, "mono8"
        return c

    cls = hv.AnchorCam
    monkeypatch.setattr(cls.grab, "__defaults__", (hv.ANCHOR_FRAMES, 0.05))
    monkeypatch.setattr(hv, "Sensors", Sens)
    monkeypatch.setattr(hv, "AnchorCam", cam)
    assert hv.main(["--execute", "--ref", str(ref), "--config", str(cfg)]) == 1
    out = capsys.readouterr().out
    assert moves == [] and "source scripts/vision/common.sh" in out and "인코딩" not in out


# ---------- ★11.17 --align: 합성 시야(마커 정답을 투영) · 가짜 목 ----------

ALIGN_CFG = __import__("yaml").safe_load(hv.DEFAULT_CONFIG.read_text(encoding="utf-8"))
K908 = np.array([[908.0, 0, 640], [0, 908.0, 360], [0, 0, 1]])
CAMSET = (-19.60, 69.055)
Y73 = 0.075   # 73 의 실제 y(--align 은 모른다)


def _rot(axis: int, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


class _AlignRig:
    """sens · cam · head_move 를 겸한다. 카메라 = 저장 외부 파라미터(R0, t0)를 엔코더 (p0, t0) 에서 본 자세로 두고 pan 은 base z,
    tilt 는 광학 x 축으로 돌린다(tilt + = 숙임). 마커 코너 = 정답(anchor_truth, 73 은 y = Y73) 투영 px."""
    def __init__(self, enc0=(CAMSET[0] + 0.4, CAMSET[1] + 0.5), start=(0.0, 60.0), hide=(), flip_tilt=False, imu_off=0.0):
        self.R0, self.t0 = hv.load_extrinsics(hv.DEFAULT_EXTRINSICS)
        self.enc0, self.pan, self.tilt, self.hide, self.flip, self.imu_off = enc0, *start, set(hide), flip_tilt, imu_off
        self.dist, self.moves = np.zeros(5), []

    def request_move(self, pan, tilt, max_tilt):
        self.moves.append((pan, tilt))
        self.pan, self.tilt = pan, tilt
        return 0, None

    def R(self, pan=None, tilt=None):
        dp = (self.pan if pan is None else pan) - self.enc0[0]
        dt = ((self.tilt if tilt is None else tilt) - self.enc0[1]) * (-1 if self.flip else 1)
        return _rot(2, dp) @ self.R0 @ _rot(0, -dt)

    def read(self):
        p, r = hv.up_to_angles(self.R()[2, :])
        return p + self.imu_off, r, 0.01, self.pan, self.tilt

    def markers(self, pan=None, tilt=None):
        import cv2
        R, h, z = self.R(pan, tilt), ALIGN_CFG["anchor_size_m"] / 2, ALIGN_CFG["anchor_truth"]["table_z_m"]
        out = {}
        for i, (x, y) in ALIGN_CFG["anchor_truth"]["markers"].items():
            y = Y73 if y is None else y
            P = np.array([[x - h, y + h, z], [x + h, y + h, z], [x + h, y - h, z], [x - h, y - h, z]])
            Pc = (P - self.t0) @ R
            uv = Pc[:, :2] / Pc[:, 2:] * 908.0 + [640, 360]
            if i in self.hide or np.any(Pc[:, 2] <= 0) or not np.all((uv >= 0) & (uv < [1280, 720])):
                continue
            rvec, tvec = hv._pnp(uv, K908, self.dist, 2 * h)
            out[int(i)] = {"corners": uv, "rvec": rvec, "tvec": tvec}
        return out


def _align_ref(rig, **kw):
    hide, rig.hide = rig.hide, set()   # 기준은 마커가 다 보일 때 잡았다
    m = rig.markers(*rig.enc0)
    rig.hide = hide
    return {"approach": "down", "approach_pan": "left", "imu_up_cam": list(rig.R(*rig.enc0)[2, :]),
            "anchors": {"markers": m}, **kw}


@pytest.fixture
def align_cam(monkeypatch):
    monkeypatch.setattr(hv, "_read_anchors", lambda cam, cfg, n=hv.ANCHOR_FRAMES: (cam.markers(), 0.05, K908, [1280, 720]))
    monkeypatch.setattr(hv, "ALIGN_TIMEOUT_S", 60.0)


def _ext_copy(tmp_path, dpos=(0.0, 0.0, 0.0)):
    """저장 외부 파라미터 사본(camera position 을 dpos m 만큼 옮김)."""
    import yaml
    p = tmp_path / hv.DEFAULT_EXTRINSICS.name
    text = hv.DEFAULT_EXTRINSICS.read_text(encoding="utf-8")
    pos = yaml.safe_load(text)["camera"]["position"]
    new = [round(a + b, 6) for a, b in zip(pos, dpos)]
    p.write_text(text.replace(f"position: [{', '.join(str(v) for v in pos)}]", f"position: [{', '.join(str(v) for v in new)}]", 1),
                 encoding="utf-8")
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["camera"]["position"] == new
    return p


def _run_align(rig, tmp_path, ext=None, write=False):
    return hv._align(rig, rig, rig, ALIGN_CFG, _align_ref(rig), hv.Limits(), CAMSET, 3.0,
                     ext or _ext_copy(tmp_path), write)


def test_align_converges_from_the_approach_side_and_the_extrinsics_pass(align_cam, tmp_path):
    """(가) home 에서 camset → 백래시만큼 어긋난 시야를 tilt · pan 으로 기준 px 1 px 안에, 마지막 이동은 접근 방향(down · left)."""
    rig = _AlignRig()
    rc, rec = _run_align(rig, tmp_path)
    assert rc == 0 and rec["ok"] and rec["px_ok"]
    assert all(v is not None and v <= 1.0 for v in rec["marker_px"].values()) and rec["missing"] == []
    assert rec["final_deg"] == pytest.approx(list(rig.enc0), abs=0.1)
    pans, tilts = [p for p, _ in rig.moves], [t for _, t in rig.moves]
    last_t = max(i for i in range(1, len(tilts)) if tilts[i] != tilts[i - 1])
    last_p = max(i for i in range(1, len(pans)) if pans[i] != pans[i - 1])
    assert tilts[last_t] > tilts[last_t - 1] and pans[last_p] > pans[last_p - 1]       # down = 틱 +, left = 틱 +
    e = rec["extrinsics"]
    assert e["ok"] and e["d_pos_mm"] < 0.5 and e["d_rot_deg"] < 0.02 and e["written"] is None
    assert e["free_y_mm"]["73"] == pytest.approx(Y73 * 1000, abs=0.5)                # 73 의 y 는 같이 풀린다
    assert not rec["imu"]["warn"]


def test_align_stops_when_tilt_moves_the_wrong_way(align_cam, tmp_path):
    """(나) tilt 를 숙이면 장면이 아래로 간다(기구 · 부호가 반대) → 시험 이동 뒤 멈추고 시작 각으로 돌아가 1, 외부 파라미터는 안 푼다."""
    rig = _AlignRig(flip_tilt=True, enc0=(CAMSET[0], CAMSET[1] - 0.5))
    rc, rec = _run_align(rig, tmp_path)
    assert rc == 1 and not rec["ok"] and "extrinsics" not in rec
    tilts = [t for _, t in rig.moves]
    assert abs(tilts[-1] - CAMSET[1]) < 1e-9 and len(rig.moves) <= 6                 # camset tilt 로 되돌림


def test_align_extrinsics_out_of_tolerance_fails_and_writes_only_on_request(align_cam, tmp_path):
    """(다) 저장 외부 파라미터가 10 mm 틀리면 1 · 파일 그대로, --write-extrinsics 면 .bak 남기고 camera 만 새 값."""
    import yaml
    ext = _ext_copy(tmp_path, (0.010, 0.0, 0.0))
    before = ext.read_text(encoding="utf-8")
    rc, rec = _run_align(_AlignRig(), tmp_path, ext)
    assert rc == 1 and not rec["extrinsics"]["ok"] and rec["extrinsics"]["d_pos_mm"] == pytest.approx(10.0, abs=0.5)
    assert ext.read_text(encoding="utf-8") == before and not list(tmp_path.glob("*.bak-*"))
    rig = _AlignRig()
    rc, rec = _run_align(rig, tmp_path, ext, write=True)
    assert rc == 0 and rec["extrinsics"]["written"] and len(list(tmp_path.glob("*.bak-*"))) == 1
    a, b = yaml.safe_load(before), yaml.safe_load(ext.read_text(encoding="utf-8"))
    assert {k: v for k, v in a.items() if k != "camera"} == {k: v for k, v in b.items() if k != "camera"}
    assert b["camera"]["position"] == pytest.approx(rig.t0, abs=5e-4) and b["camera"]["frame"] == a["camera"]["frame"]
    assert "--write-extrinsics" in ext.read_text(encoding="utf-8")                   # 출처 주석


def test_align_without_marker_73_warns_and_without_71_stops(align_cam, tmp_path, capsys):
    """(라) x 만 정답인 73 이 가려지면 경고하고 70 · 71 로 맞추고 푼다. 위치를 다 아는 71 이 가려지면 움직이다 멈추고 1."""
    rc, rec = _run_align(_AlignRig(hide={73}), tmp_path)
    assert rc == 0 and rec["missing"] == [73] and rec["extrinsics"]["ids"] == [70, 71] and rec["extrinsics"]["ok"]
    assert "마커 [73] 가 안 보여" in capsys.readouterr().out
    rc, rec = _run_align(_AlignRig(hide={71}), tmp_path)
    assert rc == 1 and "extrinsics" not in rec


def test_align_imu_difference_is_a_warning_only(align_cam, tmp_path, capsys):
    """(마) IMU 가 기준과 2° 달라도 마커 · 외부 파라미터가 맞으면 0, 경고만."""
    rc, rec = _run_align(_AlignRig(imu_off=2.0), tmp_path)
    assert rc == 0 and rec["imu"]["warn"] and rec["imu"]["d_pitch_deg"] == pytest.approx(2.0, abs=0.05)
    assert "경고만" in capsys.readouterr().out


def test_align_cli_needs_execute_and_a_measured_ref_with_anchors(tmp_path, capsys):
    for argv in (["--align"], ["--align", "--execute", "--check"], ["--write-extrinsics", "--execute"]):
        with pytest.raises(SystemExit):
            hv.main(argv)
    assert hv.main(["--align", "--execute", "--ref", str(tmp_path / "none.yaml")]) == 1   # ROS 전에 끝난다
    assert "기준 파일이 없다" in capsys.readouterr().out


def test_tilt_px_per_deg_matches_the_rig():
    rig = _AlignRig(start=CAMSET)
    a, b = rig.markers(*CAMSET), rig.markers(CAMSET[0], CAMSET[1] + 0.2)
    real = np.mean([(b[i]["corners"] - a[i]["corners"])[:, 1].mean() for i in a]) / 0.2
    geo = hv.tilt_px_per_deg(np.concatenate([a[i]["corners"] for i in a]), K908)
    assert real < 0 and 0.8 < real / geo < 1.25


def test_align_main_writes_one_json_line(align_cam, tmp_path, monkeypatch):
    """main 배선: 기준 파일(anchors) · 설정 · camset 을 읽고 --align 을 돌려 --out 에 JSON 한 줄을 덧붙인다."""
    import json
    import sys
    import types
    rig = _AlignRig()
    rig._acc = [0] * 400
    ext = _ext_copy(tmp_path)
    r = _align_ref(rig)
    ref = tmp_path / "ref.yaml"
    hv.write_ref(ref, hv.build_ref(r["imu_up_cam"], (1830, 2839), 400, 0.01, ext.name, hv.file_md5(ext), "down", "t", "test",
                                   "left", hv.anchors_block(r["anchors"]["markers"], 10, 0.05, ALIGN_CFG["anchor_size_m"],
                                                            [1280, 720], 908.0, None)))
    stubs = {"rclpy": types.SimpleNamespace(init=lambda: None, ok=lambda: False),
             "rclpy.node": types.SimpleNamespace(Node=lambda name: types.SimpleNamespace(destroy_node=lambda: None)),
             "head_move": rig}
    for k, v in stubs.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(hv, "Sensors", lambda node, rclpy: rig)
    monkeypatch.setattr(hv, "AnchorCam", lambda node, rclpy: rig)
    out = tmp_path / "align.jsonl"
    argv = ["--align", "--execute", "--ref", str(ref), "--extrinsics", str(ext), "--out", str(out)]
    assert hv.main(argv) == 0 and hv.main(argv) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[-1])["ok"] and json.loads(lines[-1])["final_ticks"]


# ---------- ★11.17 --auto-extrinsics: camset 명령 한 번 → 고정 마커로 외부 파라미터를 다시 풀어 live 파일 ----------

def _auto(rig, tmp_path, ext=None, ref=None, **cfg_kw):
    rig._acc = [0] * 400
    cfg = {**ALIGN_CFG, "auto_occlusion_wait_s": 0.0, "auto_settle_s": 0.0, **cfg_kw}
    base = ext or _ext_copy(tmp_path)
    live, live_ref = tmp_path / "live.yaml", tmp_path / "live_ref.yaml"
    rc, rec = hv._auto_extrinsics(rig, rig, rig, cfg, ref or _align_ref(rig), hv.Limits(), CAMSET, base, live, live_ref)
    return rc, rec, base, live, live_ref


def _mounted(deg):
    """카메라 마운트가 광학 y 축으로 deg 돌았다(11.17 USB 재연결 뒤처럼 화면이 옆으로 밀림). 기준 px 는 돌기 전."""
    rig = _AlignRig(enc0=CAMSET)
    ref = _align_ref(rig)
    rig.R0 = rig.R0 @ _rot(1, deg)
    return rig, ref


def test_auto_extrinsics_writes_live_from_the_markers_and_keeps_it_when_unchanged(align_cam, tmp_path):
    import yaml
    rig, ref = _mounted(0.5)
    rc, rec, base, live, live_ref = _auto(rig, tmp_path, ref=ref)
    base_md5 = hv.file_md5(base)
    assert rc == 0 and rec["live"] == "written" and rec["camera"] == "solved"
    assert rig.moves and rig.moves[-1] == pytest.approx(CAMSET)   # camset 명령으로 끝났다(맞춤 이동 없음)
    d = yaml.safe_load(live.read_text(encoding="utf-8"))
    R, t = hv.load_extrinsics(live)
    dmm, ddeg = hv.extrinsics_diff(R, t, rig.R(), rig.t0)
    assert dmm < 1.0 and ddeg < 0.05                                # 마커로 실제 카메라 자세를 되찾았다
    assert d["head_pose"] == {"pan_tick": hv.deg_to_tick(CAMSET[0]), "tilt_tick": hv.deg_to_tick(CAMSET[1]),
                              "tol_tick": yaml.safe_load(base.read_text(encoding="utf-8"))["head_pose"]["tol_tick"]}
    p = d["provenance"]
    assert p["base_md5"] == base_md5 and p["marker_ids"] == [70, 71, 73] and 0.4 < p["d_rot_deg"] < 0.6
    assert hv.file_md5(base) == base_md5 and not list(tmp_path.glob("*.tmp"))   # base 는 읽기만, 임시 파일 안 남김
    md5 = hv.file_md5(live)
    rc, rec, *_ = _auto(rig, tmp_path, ext=base, ref=ref)
    assert rc == 0 and rec["live"] == "kept" and hv.file_md5(live) == md5         # 같으면 다시 안 쓴다(md5 안정)


def test_auto_extrinsics_inside_extr_tol_keeps_the_base_camera(align_cam, tmp_path):
    import yaml
    rc, rec, base, live, _ = _auto(_AlignRig(enc0=CAMSET), tmp_path)
    assert rc == 0 and rec["camera"] == "base"
    assert yaml.safe_load(live.read_text(encoding="utf-8"))["camera"] == yaml.safe_load(base.read_text(encoding="utf-8"))["camera"]


def test_auto_extrinsics_refuses_without_writing(align_cam, tmp_path, capsys):
    """잘림(목을 더 움직이지 않음) · 가림 · base 와 auto_max 넘게 다름 → 1, live 를 안 쓴다."""
    rig, ref = _mounted(-1.0)                                 # 71 이 오른쪽 가장자리 밖
    rc, rec, _, live, _ = _auto(rig, tmp_path, ref=ref)
    assert rc == 1 and rec["cut"] == [71] and not live.exists() and "화면 밖" in capsys.readouterr().out
    assert len(rig.moves) <= 4                                # camset 접근만(물렸다 들어오기), 찾는 이동 없음
    rc, rec, _, live, _ = _auto(_AlignRig(enc0=CAMSET, hide={71}), tmp_path)
    assert rc == 1 and rec["hidden"] == [71] and not live.exists()
    rc, rec, _, live, _ = _auto(_AlignRig(enc0=CAMSET), tmp_path, ext=_ext_copy(tmp_path, (0.03, 0.0, 0.0)))
    assert rc == 1 and rec["extrinsics"]["d_pos_mm"] > 20 and not live.exists()


def test_auto_extrinsics_main_deletes_live_on_failure(align_cam, tmp_path, monkeypatch):
    import sys
    import types
    live, live_ref = tmp_path / "live.yaml", tmp_path / "live_ref.yaml"
    monkeypatch.setattr(hv, "DEFAULT_LIVE", live)
    monkeypatch.setattr(hv, "DEFAULT_LIVE_REF", live_ref)
    for argv in (["--auto-extrinsics", "--align", "--execute"], ["--auto-extrinsics", "--check"],
                 ["--auto-extrinsics", "--extrinsics", str(live)]):
        live.write_text("old", encoding="utf-8")
        live_ref.write_text("old", encoding="utf-8")
        with pytest.raises(SystemExit):
            hv.main(argv)
        assert not live.exists() and not live_ref.exists()
    with pytest.raises(SystemExit):
        hv.main(["--auto-extr", "--execute"])                 # 줄임은 거부(allow_abbrev=False) — 자동 갱신으로 돌지 않는다
    rig = _AlignRig(enc0=CAMSET)
    rig._acc = [0] * 400
    for k, v in {"rclpy": types.SimpleNamespace(init=lambda: None, ok=lambda: False),
                 "rclpy.node": types.SimpleNamespace(Node=lambda name: types.SimpleNamespace(destroy_node=lambda: None)),
                 "head_move": rig}.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(hv, "Sensors", lambda node, rclpy: rig)
    monkeypatch.setattr(hv, "AnchorCam", lambda node, rclpy: rig)
    assert hv.main(["--auto-extrinsics", "--execute"]) == 0 and live.exists() and live_ref.exists()
    rig.hide = {70}
    monkeypatch.setitem(ALIGN_CFG, "auto_occlusion_wait_s", 0.0)
    assert hv.main(["--auto-extrinsics"]) == 1 and not live.exists() and not live_ref.exists()


def _main_stubs(monkeypatch, rig):
    import sys
    import types
    for k, v in {"rclpy": types.SimpleNamespace(init=lambda: None, ok=lambda: False),
                 "rclpy.node": types.SimpleNamespace(Node=lambda name: types.SimpleNamespace(destroy_node=lambda: None)),
                 "head_move": rig}.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(hv, "Sensors", lambda node, rclpy: rig)
    monkeypatch.setattr(hv, "AnchorCam", lambda node, rclpy: rig)


def test_auto_extrinsics_kept_rewrites_the_same_bytes_and_refreshes_verified_at(align_cam, tmp_path, monkeypatch):
    """★11.17 리뷰: 시작하자마자 지우고(죽어도 옛 값이 안 남음) kept 면 같은 바이트(md5 불변) + 재기준 verified_at 만 갱신."""
    import yaml
    live, live_ref = tmp_path / "live.yaml", tmp_path / "live_ref.yaml"
    monkeypatch.setattr(hv, "DEFAULT_LIVE", live)
    monkeypatch.setattr(hv, "DEFAULT_LIVE_REF", live_ref)
    rig = _AlignRig(enc0=CAMSET)
    rig._acc = [0] * 400
    _main_stubs(monkeypatch, rig)
    assert hv.main(["--auto-extrinsics", "--execute"]) == 0
    md5 = hv.file_md5(live)
    lr = yaml.safe_load(live_ref.read_text(encoding="utf-8"))
    lr["verified_at"] = "2026-01-01 00:00:00"
    live_ref.write_text(hv.ref_text(lr), encoding="utf-8")
    seen = {}
    real_auto = hv._auto_extrinsics

    def spy(*a, **kw):
        seen["gone"] = not live.exists() and not live_ref.exists()     # _auto_extrinsics 에 들어올 때 이미 지워져 있다
        return real_auto(*a, **kw)
    monkeypatch.setattr(hv, "_auto_extrinsics", spy)
    assert hv.main(["--auto-extrinsics", "--execute"]) == 0 and seen["gone"]
    assert hv.file_md5(live) == md5
    assert yaml.safe_load(live_ref.read_text(encoding="utf-8"))["verified_at"] != "2026-01-01 00:00:00"
    assert hv.live_problem(live, hv.file_md5(hv.DEFAULT_EXTRINSICS), live_ref) is None


def test_auto_extrinsics_main_deletes_live_when_the_second_write_fails(align_cam, tmp_path, monkeypatch):
    live, live_ref = tmp_path / "live.yaml", tmp_path / "live_ref.yaml"
    monkeypatch.setattr(hv, "DEFAULT_LIVE", live)
    monkeypatch.setattr(hv, "DEFAULT_LIVE_REF", live_ref)
    rig = _AlignRig(enc0=CAMSET)
    rig._acc = [0] * 400
    _main_stubs(monkeypatch, rig)
    real, calls = hv._atomic_write, []

    def flaky(path, data):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("디스크 가득")
        real(path, data)
    monkeypatch.setattr(hv, "_atomic_write", flaky)
    with pytest.raises(OSError):
        hv.main(["--auto-extrinsics", "--execute"])
    assert calls == [live, live_ref] and not live.exists() and not live_ref.exists()


@pytest.mark.parametrize("case", ["base_md5", "moved", "turned", "ticks", "ref_md5", "not_measured"])
def test_live_kept_says_why_it_must_rewrite(align_cam, tmp_path, case):
    import yaml
    rig, ref = _mounted(0.5)
    rc, rec, base, live, live_ref = _auto(rig, tmp_path, ref=ref)
    assert rc == 0
    prev = {"live": live.read_bytes(), "ref": live_ref.read_bytes()}
    R, t = hv.load_extrinsics(live)
    s = {"R": R, "t": t}
    ticks = tuple(rec["ticks"])
    cfg = {**ALIGN_CFG}
    assert hv._live_kept(prev, base, cfg, s, ticks) is None
    if case == "base_md5":
        base = _ext_copy(tmp_path, (0.001, 0.0, 0.0))
    elif case == "moved":
        s = {"R": R, "t": t + [0.004, 0.0, 0.0]}               # 3 mm 넘게
    elif case == "turned":
        s = {"R": R @ _rot(0, 0.4), "t": t}                   # 0.3° 넘게
    elif case == "ticks":
        ticks = (ticks[0] + int(cfg.get("ref_tol_tick", 4)) + 1, ticks[1])
    elif case == "ref_md5":
        prev["live"] = prev["live"] + b"# edited\n"
    else:
        lr = yaml.safe_load(prev["ref"].decode("utf-8"))
        prev["ref"] = hv.ref_text({**lr, "measured": False}).encode("utf-8")
    assert hv._live_kept(prev, base, cfg, s, ticks)


def test_live_problem_refuses_old_or_foreign_live(align_cam, tmp_path):
    """★11.17 리뷰: FP++ 쪽 시작 검사 — 오래됐거나(provenance · verified_at 둘 다 live_max_age_h 밖) 다른 base 면 거부."""
    import time
    rc, rec, base, live, live_ref = _auto(_AlignRig(enc0=CAMSET), tmp_path)
    assert rc == 0 and hv.live_problem(live, live_ref=live_ref, max_age_h=12) is None
    later = time.time() + 13 * 3600
    assert "h 전" in hv.live_problem(live, live_ref=live_ref, max_age_h=12, now=later)
    assert "다른 base" in hv.live_problem(live, base_md5="0" * 32, live_ref=live_ref, max_age_h=12)
    assert "없거나 비었다" in hv.live_problem(tmp_path / "none_live.yaml")
    import yaml
    d = yaml.safe_load(live.read_text(encoding="utf-8"))
    del d["provenance"]["time"]
    stale = tmp_path / "x_live.yaml"
    stale.write_text(yaml.safe_dump(d), encoding="utf-8")
    assert hv.live_problem(stale, live_ref=tmp_path / "none.yaml", max_age_h=12)   # 확인 시각 없음


def _ref_markers():
    # 11.17 기준 사진 코너(config/head_view_ref_arm4090_h2048.yaml), 1280x720
    return {70: {"corners": [[940.5, 67.9], [1028.1, 42.7], [1073.7, 118.5], [983.2, 145.9]]},
            71: {"corners": [[1273.8, 464.0], [1170.8, 500.1], [1114.9, 394.6], [1213.7, 361.7]]},
            73: {"corners": [[474.5, 200.4], [500.4, 288.3], [396.2, 317.9], [373.8, 227.5]]}}


def test_view_nudge_pulls_cut_marker_in_and_keeps_others():
    """★11.18: 71 이 오른쪽 끝에 잘리면 pan 을 조금 돌려 화면 안으로 들이고 다른 마커는 안에 남는다."""
    K = np.array([[910.0, 0, 640.0], [0, 910.0, 360.0], [0, 0, 1]])
    ref = _ref_markers()
    m = {i: {"corners": (np.asarray(v["corners"]) + [7.0, 0.0]).tolist()} for i, v in ref.items()}   # 0.5° 덜 숙임 ≈ 7 px
    cut, _ = hv.view_problem(m, [70, 71, 73], ref, [1280, 720], 5.0)
    assert cut == [71]
    d = hv.view_nudge(m, [70, 71, 73], ref, [1280, 720], 5.0, K, 69.0, 8.0, 3.0)
    assert d is not None and abs(d[1]) < 1e-9
    moved = {i: np.asarray(v["corners"]) + [d[0] * hv.pan_px_per_deg(v["corners"], K, 69.0), 0.0] for i, v in m.items()}
    assert hv.view_problem({i: {"corners": c.tolist()} for i, c in moved.items()}, [70, 71, 73], ref, [1280, 720], 5.0)[0] == []


def test_view_nudge_refuses_when_both_sides_cut():
    K = np.array([[910.0, 0, 640.0], [0, 910.0, 360.0], [0, 0, 1]])
    ref = _ref_markers()
    m = {70: {"corners": [[2.0, 100.0], [40.0, 100.0], [40.0, 140.0], [2.0, 140.0]]},
         71: {"corners": [[1240.0, 400.0], [1278.0, 400.0], [1278.0, 440.0], [1240.0, 440.0]]},
         73: ref[73]}
    assert hv.view_nudge(m, [70, 71, 73], ref, [1280, 720], 5.0, K, 69.0, 8.0, 3.0) is None

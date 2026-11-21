"""object_pose_node 순수부 — 레지스트리 → Extrinsics → base_link 변환."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from object_pose_node import PoseConverter, correct_depth  # noqa: E402
from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402

REG = load_registry(DEFAULT_REGISTRY)


def test_shaker_camera_pose_maps_to_measured_base_pose():
    """09.03 실측 카메라 프레임 (0.0363,-0.1111,0.5877) → base.

    v1 extrinsics(09-01 hand-eye)로는 (0.3627,-0.0032,0.3224) 였다. 09.05 head_v1 CAD 사전
    모델 hand-eye v2(RGB 개구부 정정·tilt 영점 +90.51°)로 같은 입력이 (0.3616,-0.0038,0.2995).
    """
    conv = PoseConverter(REG, ["shaker_closed"])
    pos, quat = conv.convert("shaker_closed", np.array([0.0363, -0.1111, 0.5877]),
                             np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(pos, [0.3616, -0.0038, 0.2995], atol=0.01)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_cup_applies_yup_to_zup_but_same_camera():
    conv = PoseConverter(REG, ["shaker_closed", "cup_big_s100"])
    p_s, q_s = conv.convert("shaker_closed", np.zeros(3), np.array([1.0, 0, 0, 0]))
    p_c, q_c = conv.convert("cup_big_s100", np.zeros(3), np.array([1.0, 0, 0, 0]))
    # shaker 는 cad_to_body 위치 +4.6 mm(CAD 중심 → sim body 원점, 09.07), cup 은 0 — 차이는 그 길이뿐
    assert np.linalg.norm(p_s - p_c) == pytest.approx(0.0046, abs=1e-6)
    assert not np.allclose(q_c, q_s)      # cup 은 Y-up→Z-up 회전이 붙는다


def test_spin_about_symmetry_axis_does_not_change_output():
    """shaker 를 CAD z 둘레로 아무리 돌려도(추적기 yaw 드리프트) 출력 자세는 같다."""
    conv = PoseConverter(REG, ["shaker_closed"])
    pos = np.array([0.0363, -0.1111, 0.5877])
    ref_p, ref_q = conv.convert("shaker_closed", pos, np.array([1.0, 0.0, 0.0, 0.0]))
    for deg in (37.0, 120.0, -95.0):
        h = np.radians(deg) / 2
        p, q = conv.convert("shaker_closed", pos, np.array([np.cos(h), 0.0, 0.0, np.sin(h)]))
        assert np.allclose(p, ref_p)
        assert np.allclose(q, ref_q, atol=1e-9) or np.allclose(q, -ref_q, atol=1e-9)
    # 출력 자세엔 base z 둘레 twist 가 없다(z 성분 0) → 정립 물체는 항등에 가깝다
    assert abs(ref_q[3]) < 1e-9


def test_cup_spin_about_cad_y_is_removed_in_body_frame():
    """cup 은 CAD Y-up: CAD y 둘레 회전이 body z 둘레 회전이 되고, 출력에서 사라져야 한다."""
    conv = PoseConverter(REG, ["cup_big_s100"])
    pos = np.array([0.0, -0.1, 0.6])
    ref_p, ref_q = conv.convert("cup_big_s100", pos, np.array([1.0, 0.0, 0.0, 0.0]))
    h = np.radians(70.0) / 2
    p, q = conv.convert("cup_big_s100", pos, np.array([np.cos(h), 0.0, np.sin(h), 0.0]))
    assert np.allclose(p, ref_p)
    assert np.allclose(q, ref_q, atol=1e-9) or np.allclose(q, -ref_q, atol=1e-9)


def test_unknown_name_rejected_and_names_resolved():
    conv = PoseConverter(REG, ["cup_big_s080"])
    assert conv.names == ["cup_big_s100"]
    try:
        PoseConverter(REG, ["teapot"])
    except ValueError as err:
        assert "unknown object" in str(err)
    else:
        raise AssertionError("expected ValueError")


ARM4090_CAMERA = Path(__file__).resolve().parents[1] / "config" / "global_camera_extrinsics_arm4090.yaml"


def test_camera_yaml_override_and_depth_ray_correction():
    """10.01 arm4090 전용 camera 파일 · 10.04 z 만 보정(−8 mm)을 깊이 광선 보정으로 바꿨다."""
    shared = PoseConverter(REG, ["cup_big_s100"])
    assert shared.z_bias == 0.0 and shared.depth_bias is None
    own = PoseConverter(REG, ["cup_big_s100"], ARM4090_CAMERA)
    assert own.z_bias == 0.0 and own.depth_bias == pytest.approx((0.0286, -0.059, 0.45, 0.90))
    cam_p, cam_q = np.array([0.02, 0.05, 0.62]), np.array([1.0, 0.0, 0.0, 0.0])
    p1, q1 = own.convert("cup_big_s100", cam_p, cam_q)
    off = PoseConverter(REG, ["cup_big_s100"], ARM4090_CAMERA)
    off.depth_bias = None                                          # 같은 변환에서 깊이 보정만 끈다
    p0, q0 = off.convert("cup_big_s100", cam_p, cam_q)
    assert np.allclose(q1, q0)                                     # 위치만 바뀐다
    e = 0.0286 - 0.059 * 0.62                                     # −8.0 mm (짧게 봄)
    assert np.isclose(np.linalg.norm(p1 - p0), -e * np.linalg.norm(cam_p) / 0.62, atol=1e-6)
    ps, _ = shared.convert("cup_big_s100", cam_p, cam_q)
    assert np.linalg.norm(ps - p0) > 0.005                 # 5090 공유 값과 실제로 다르다


def test_depth_correction_stretches_along_the_ray_and_clamps_outside_the_fit():
    bias = (0.0286, -0.059, 0.45, 0.90)
    p = np.array([0.10, -0.05, 0.70])
    q = correct_depth(p, bias)
    assert np.allclose(np.cross(p, q), 0.0, atol=1e-12)            # 같은 광선 위
    assert np.isclose(q[2] - p[2], -(0.0286 - 0.059 * 0.70))       # 12.7 mm 더 멀리
    far = correct_depth(np.array([0.0, 0.0, 1.5]), bias)
    assert np.isclose(far[2] - 1.5, -(0.0286 - 0.059 * 0.90))      # 맞춘 범위 끝값으로 자른다
    assert np.allclose(correct_depth(p, None), p)


#: 아래 홀더 2 실측(10.03 프레임)이 쓴 외부 파라미터 — 10.01 캘리브. 파일은 10.04 에 다시 캘리브했다(카메라가 움직였다).
CAM_10_01 = (np.array([0.050504, 0.037916, 0.820846]), np.array([0.130161, -0.69891, 0.696068, -0.100366]))


def test_holder2_fpp_lands_near_the_marker_truth_after_the_ray_correction():
    """10.04 arm4090 실측: FP++(홀더 2, z 보정 0) base (0.3748, −0.1572, 0.2454) · 마커 정답 (0.3820, −0.1613, 0.2350).
    그때의 외부 파라미터(10.01)로 카메라 프레임으로 되돌린 뒤 파일의 깊이 광선 보정을 건다."""
    own = PoseConverter(REG, ["cup_big_s100"], ARM4090_CAMERA)
    from table_cad_extrinsics import T_from                      # base ← camera (같은 쿼터니언 규약 wxyz)
    T = T_from(*CAM_10_01)
    raw_base = np.array([0.3748, -0.1572, 0.2454])
    cam = (np.linalg.inv(T) @ np.r_[raw_base, 1.0])[:3]
    fixed = (T @ np.r_[correct_depth(cam, own.depth_bias), 1.0])[:3]
    truth = np.array([0.3820, -0.1613, 0.2350])
    assert np.linalg.norm(raw_base - truth) > 0.013
    assert np.linalg.norm(fixed - truth) < 0.004


def test_z_bias_out_of_range_rejected(tmp_path):
    bad = tmp_path / "cam.yaml"
    bad.write_text(ARM4090_CAMERA.read_text().replace("base_z_bias_m: 0.0", "base_z_bias_m: 0.2"))
    with pytest.raises(ValueError):
        PoseConverter(REG, ["cup_big_s100"], bad)


def test_depth_bias_and_z_bias_together_are_refused(tmp_path):
    both = tmp_path / "cam.yaml"
    both.write_text(ARM4090_CAMERA.read_text().replace("base_z_bias_m: 0.0", "base_z_bias_m: -0.008"))
    with pytest.raises(ValueError):
        PoseConverter(REG, ["cup_big_s100"], both)


def test_depth_bias_out_of_range_rejected(tmp_path):
    bad = tmp_path / "cam.yaml"
    bad.write_text(ARM4090_CAMERA.read_text().replace("offset_m: 0.0286", "offset_m: 0.2"))
    with pytest.raises(ValueError):
        PoseConverter(REG, ["cup_big_s100"], bad)


def test_cyl60_upside_down_fpp_pose_comes_out_standing_with_the_same_position():
    """10.04 arm4090 첫 확인: FP++ 가 노란 원통을 뒤집어 잡았다(기울기 179°) — 레지스트리 symmetry_flip 으로 축을 위로.
    위치(원점 = 중심)는 그대로. symmetry_flip 은 원통에만(홀더는 위아래가 다르다)."""
    from pose_symmetry import quat_axis_direction
    z = np.array([0.0, 0.0, 1.0])
    conv = PoseConverter(REG, ["cyl60"], ARM4090_CAMERA)
    cam = np.array([0.02, -0.05, 0.62])
    b_a, q_a = conv.convert("cyl60", cam, np.array([1.0, 0.0, 0.0, 0.0]))
    b_b, q_b = conv.convert("cyl60", cam, np.array([0.0, 1.0, 0.0, 0.0]))     # 카메라 x 둘레 180° — 반대로 잡은 원통
    assert np.allclose(b_a, b_b, atol=1e-9)
    a, b = quat_axis_direction(q_a, z), quat_axis_direction(q_b, z)
    assert a[2] > 0 and b[2] > 0 and np.dot(a, b) > 0.999
    assert REG.get("cyl60").symmetry_flip and not REG.get("cup_holder").symmetry_flip


# ---- 보정 프로필(hand-eye + 목 각도별 보정) -------------------------------------------------
def _profile(tmp_path, bias=(0.002, -0.001), strict=True, matrix=None):
    import yaml

    from head_camera_pose import load_neck_to_camera
    T = load_neck_to_camera() if matrix is None else matrix
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({
        "profile": {"name": "p"}, "hand_eye": {"neck_to_camera": {"matrix": np.asarray(T).tolist()}},
        "angle_tolerance_deg": 1.0, "strict": strict,
        "angles": [{"pan_deg": 15.0, "tilt_deg": -20.0, "bias_xy_m": list(bias)}]}), encoding="utf-8")
    return path


def test_profile_with_same_hand_eye_only_adds_the_angle_bias(tmp_path):
    base = PoseConverter(REG, ["shaker_closed"])
    prof = PoseConverter(REG, ["shaker_closed"], calib_profile=_profile(tmp_path))
    args = ("shaker_closed", np.array([0.0363, -0.1111, 0.5877]), np.array([1.0, 0.0, 0.0, 0.0]), (15.2, -19.8))
    p0, _ = base.convert(*args)
    p1, _ = prof.convert(*args)
    assert np.allclose(p1 - p0, [0.002, -0.001, 0.0], atol=1e-9)


def test_profile_hand_eye_replaces_the_default_one(tmp_path):
    from head_camera_pose import load_neck_to_camera
    shifted = load_neck_to_camera().copy()
    shifted[0, 3] += 0.010   # 카메라가 목 기준 10 mm 옮겨진 hand-eye
    base = PoseConverter(REG, ["shaker_closed"])
    prof = PoseConverter(REG, ["shaker_closed"], calib_profile=_profile(tmp_path, bias=(0.0, 0.0), matrix=shifted))
    args = ("shaker_closed", np.array([0.0363, -0.1111, 0.5877]), np.array([1.0, 0.0, 0.0, 0.0]), (15.0, -20.0))
    assert np.linalg.norm(prof.convert(*args)[0] - base.convert(*args)[0]) == pytest.approx(0.010, abs=2e-3)


def test_strict_profile_raises_outside_calibrated_angles_and_needs_a_head_angle(tmp_path):
    from calib_profile import UncalibratedAngle
    prof = PoseConverter(REG, ["shaker_closed"], calib_profile=_profile(tmp_path))
    with pytest.raises(UncalibratedAngle):
        prof.convert("shaker_closed", np.zeros(3), np.array([1.0, 0, 0, 0]), (-45.0, -20.0))
    with pytest.raises(ValueError, match="목 각도"):
        prof.convert("shaker_closed", np.zeros(3), np.array([1.0, 0, 0, 0]))


def test_without_a_profile_nothing_changes():
    conv = PoseConverter(REG, ["shaker_closed"])
    a, _ = conv.convert("shaker_closed", np.array([0.0363, -0.1111, 0.5877]), np.array([1.0, 0, 0, 0]), (10.0, -20.0))
    b, _ = PoseConverter(REG, ["shaker_closed"], calib_profile=None).convert(
        "shaker_closed", np.array([0.0363, -0.1111, 0.5877]), np.array([1.0, 0, 0, 0]), (10.0, -20.0))
    assert np.allclose(a, b)


# ---- 정적 스냅샷 camera 파일(head_pose 가 있는 파일, 10.28 RH56F1) -----------------------------------------
def _joint_deg_at(pan_tick, tilt_tick):
    """/head/joint_states 가 싣는 각(URDF 규약, 도) — 그 틱에 서 있는 머리."""
    from head_compliant_hold import tick_to_deg
    from head_fk_chain import urdf_from_encoder
    return urdf_from_encoder(tick_to_deg(pan_tick), tick_to_deg(tilt_tick))


def test_a_static_snapshot_camera_file_is_not_replaced_by_the_neck_fk():
    """10.28: head_pose 가 있는 camera 파일(RH56F1)은 그 머리 자세 하나에서 잰 값이다. 목 각도가 와도 그 camera 블록을 쓴다 —
    목 FK 의 hand-eye(config/head_extrinsics.yaml)는 다른 머리(head_v1) 것이라 이 머리에 쓰면 광축이 약 90° 틀린다."""
    conv = PoseConverter(REG, ["cyl60"], ARM4090_CAMERA)
    cam, q = np.array([0.02, -0.05, 0.62]), np.array([1.0, 0.0, 0.0, 0.0])
    p0, q0 = conv.convert("cyl60", cam, q)
    p1, q1 = conv.convert("cyl60", cam, q, _joint_deg_at(2033, 2864))
    assert np.allclose(p0, p1) and np.allclose(q0, q1)


def test_a_static_snapshot_holds_poses_back_when_the_head_is_off_its_pose():
    """틀린 좌표는 없느니만 못하다 — 목이 head_pose ± tol_tick(4) 밖이면 변환하지 않는다(노드는 발행을 보류한다)."""
    from object_pose_node import HeadOffCalibPose
    conv = PoseConverter(REG, ["cyl60"], ARM4090_CAMERA)
    args = ("cyl60", np.array([0.02, -0.05, 0.62]), np.array([1.0, 0.0, 0.0, 0.0]))
    conv.convert(*args, _joint_deg_at(2033 + 4, 2864 - 4))
    with pytest.raises(HeadOffCalibPose, match="pan"):
        conv.convert(*args, _joint_deg_at(2033 + 5, 2864))
    with pytest.raises(HeadOffCalibPose, match="tilt"):
        conv.convert(*args, _joint_deg_at(2033, 2864 + 49))


def test_known_z_applies_to_static_pose_but_not_to_live_pose(tmp_path):
    import yaml
    base = PoseConverter(REG, ["shaker_closed"], calib_profile=_profile(tmp_path, bias=(0.0, 0.0)))
    args = ("shaker_closed", np.array([0.0363, -0.1111, 0.5877]), np.array([1.0, 0.0, 0.0, 0.0]), (15.0, -20.0))
    z0 = float(base.convert(*args)[0][2])
    origin = REG.get("shaker_closed").origin_above_bottom_m
    table = z0 + 0.004 - origin          # FP++ 값이 아는 값보다 4 mm 낮게 나오는 상황
    path = _profile(tmp_path, bias=(0.0, 0.0))
    raw = yaml.safe_load(path.read_text())
    raw["known_z"] = {"table_top_m": table, "tolerance_m": 0.015}
    path.write_text(yaml.safe_dump(raw))
    conv = PoseConverter(REG, ["shaker_closed"], calib_profile=path)
    assert conv.convert(*args)[0][2] == pytest.approx(table + origin, abs=1e-9)          # 정지 pose: 아는 값
    assert conv.convert(*args, live=True)[0][2] == pytest.approx(z0, abs=1e-9)            # live 추적: FP++ 값 그대로(쥔 뒤 높이가 변한다)
    assert np.allclose(conv.convert(*args)[0][:2], base.convert(*args)[0][:2])           # xy 는 그대로


def test_live_extrinsics_provenance_line_and_stale_refusal(tmp_path):
    """★11.17 리뷰: 시작 로그 한 줄(provenance) · 오래된 live 는 FP++ 런처 · 물체 자세 노드가 시작을 거부한다."""
    import time

    import head_view_feedback as hv
    from object_pose_node import provenance_line
    base = tmp_path / hv.DEFAULT_EXTRINSICS.name
    base.write_bytes(hv.DEFAULT_EXTRINSICS.read_bytes())
    assert provenance_line(base) is None                                   # base(m1117)에는 provenance 가 없다
    hv.require_fresh_live(base)                                            # *_live.yaml 이 아니면 검사하지 않는다
    prov = {"base": base.name, "base_md5": hv.file_md5(base), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "marker_ids": [70, 71, 73], "rms_px": 0.4, "camera": "solved", "d_pos_mm": 4.2, "d_xyz_mm": [4.0, 1.0, 0.5],
            "d_rot_deg": 0.6, "ticks": [1825, 2833]}
    live = tmp_path / "global_camera_extrinsics_arm4090_live.yaml"
    live.write_text(hv.live_text(base, None, (1825, 2833), prov), encoding="utf-8")
    line = provenance_line(live)
    assert "[70, 71, 73]" in line and "rms 0.4" in line and "4.2 mm" in line and "[1825, 2833]" in line
    hv.require_fresh_live(live)
    old = {**prov, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 13 * 3600))}
    live.write_text(hv.live_text(base, None, (1825, 2833), old), encoding="utf-8")
    with pytest.raises(SystemExit, match="head_view_check"):
        hv.require_fresh_live(live)


# ---- 목 FK 실시간(--head-fk, 11.18): 움직이는 머리에서도 head_pose 파일 + HeadCamPose 로 T_base_cam -------------------------
M1117_CAMERA = Path(__file__).resolve().parents[1] / "config" / "global_camera_extrinsics_arm4090_m1117.yaml"


def _T(ext) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "policy_control"))
    from policy_control.place_head_pose import quat_wxyz_to_mat
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_wxyz_to_mat(ext.cam_quat), ext.cam_pos
    return T


def test_head_fk_matches_static_block_at_calib_pose_and_headcampose_elsewhere():
    """q0(head_pose 자세)에서는 처짐 차 0 이라 정적 camera 블록과 같고, 다른 q 에서는 HeadCamPose 와 같다."""
    from cup_pose_relay import load_extrinsics
    from head_fk_chain import urdf_from_encoder
    from object_pose_node import extrinsics_from_head_fk, load_head_cam
    ext, hc = load_extrinsics(M1117_CAMERA), load_head_cam(M1117_CAMERA)
    assert hc.sag is not None
    q0 = urdf_from_encoder(*hc.q0_enc_deg)
    assert np.allclose(_T(extrinsics_from_head_fk(ext, hc, q0)), _T(ext), atol=1e-9)
    q = (q0[0] + 12.0, q0[1] - 6.0)
    got = _T(extrinsics_from_head_fk(ext, hc, q))
    assert np.allclose(got, hc.cam_pose_from_joint_state(*np.radians(q)), atol=1e-9)
    assert not np.allclose(got, _T(ext), atol=1e-3)


def test_head_fk_converter_skips_calib_pose_check_and_default_still_blocks():
    from head_fk_chain import urdf_from_encoder
    from object_pose_node import HeadOffCalibPose, load_head_cam
    off = PoseConverter(REG, ["cyl60"], M1117_CAMERA)
    fk = PoseConverter(REG, ["cyl60"], M1117_CAMERA, head_fk=True)
    hc = load_head_cam(M1117_CAMERA)
    q0 = urdf_from_encoder(*hc.q0_enc_deg)
    pos_cam, quat_cam = np.array([0.03, -0.1, 0.6]), np.array([1.0, 0.0, 0.0, 0.0])
    a, b = off.convert("cyl60", pos_cam, quat_cam, q0), fk.convert("cyl60", pos_cam, quat_cam, q0)
    assert np.allclose(a[0], b[0], atol=1e-9)
    moved = (q0[0] + 10.0, q0[1])
    with pytest.raises(HeadOffCalibPose):
        off.convert("cyl60", pos_cam, quat_cam, moved)
    assert not np.allclose(fk.convert("cyl60", pos_cam, quat_cam, moved)[0], a[0], atol=1e-3)
    with pytest.raises(ValueError, match="head-fk"):
        fk.convert("cyl60", pos_cam, quat_cam, None)


def test_grasp_axis_box_keeps_yaw_but_still_stands_up():
    """★11.19 커피박스(grasp_axis): 뒤집힘은 고치되 축 둘레 회전(yaw)은 남긴다 — plan_reach 가 면 방향으로 들어간다."""
    import dataclasses
    import math
    from pose_symmetry import quat_axis_direction, quat_mul
    spec = REG.get("cyl60")
    class Box:   # 레지스트리 그대로, cyl60 에만 grasp_axis
        def __getattr__(self, k):
            return getattr(REG, k)

        def get(self, n):
            return dataclasses.replace(REG.get(n), grasp_axis=(1.0, 0.0, 0.0))
    box = Box()
    a = np.asarray(spec.symmetry_axis, float) / np.linalg.norm(spec.symmetry_axis)
    twist = np.array([math.cos(math.radians(20)), *(math.sin(math.radians(20)) * a)])   # CAD 축 둘레 40°
    cam = np.array([0.02, -0.05, 0.62])
    for reg, same in ((REG, True), (box, False)):
        conv = PoseConverter(reg, ["cyl60"], ARM4090_CAMERA)
        _, q1 = conv.convert("cyl60", cam, np.array([1.0, 0.0, 0.0, 0.0]))
        _, q2 = conv.convert("cyl60", cam, quat_mul(np.array([1.0, 0.0, 0.0, 0.0]), twist))
        assert (abs(float(np.dot(q1, q2))) > 0.9999) == same
    _, q_up = PoseConverter(box, ["cyl60"], ARM4090_CAMERA).convert("cyl60", cam, np.array([0.0, 1.0, 0.0, 0.0]))
    z = np.array([0.0, 0.0, 1.0])   # 뒤집어 잡은 박스도 바로 선 박스와 같은 축(cyl60 뒤집힘 시험과 같은 기준)
    assert quat_axis_direction(q_up, z)[2] > 0 and np.dot(quat_axis_direction(q_up, z), quat_axis_direction(q1, z)) > 0.999

"""object_pose_node 순수부 — 레지스트리 → Extrinsics → base_link 변환."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from object_pose_node import PoseConverter  # noqa: E402
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


def test_camera_yaml_override_and_z_bias():
    """10.01 arm4090: 로봇 전용 camera 파일 + 그 파일의 base_z_bias_m 이 출력 z 에 더해진다."""
    shared = PoseConverter(REG, ["cup_big_s100"])
    assert shared.z_bias == 0.0
    own = PoseConverter(REG, ["cup_big_s100"], ARM4090_CAMERA)
    assert np.isclose(own.z_bias, -0.008)
    nobias = PoseConverter(REG, ["cup_big_s100"], ARM4090_CAMERA, z_bias=0.0)
    cam_p, cam_q = np.array([0.02, 0.05, 0.62]), np.array([1.0, 0.0, 0.0, 0.0])
    p1, q1 = own.convert("cup_big_s100", cam_p, cam_q)
    p0, q0 = nobias.convert("cup_big_s100", cam_p, cam_q)
    assert np.allclose(p1 - p0, [0.0, 0.0, -0.008]) and np.allclose(q1, q0)
    ps, _ = shared.convert("cup_big_s100", cam_p, cam_q)
    assert np.linalg.norm(ps - p0) > 0.005                 # 5090 공유 값과 실제로 다르다


def test_z_bias_out_of_range_rejected(tmp_path):
    bad = tmp_path / "cam.yaml"
    bad.write_text(ARM4090_CAMERA.read_text().replace("base_z_bias_m: -0.008", "base_z_bias_m: 0.2"))
    with pytest.raises(ValueError):
        PoseConverter(REG, ["cup_big_s100"], bad)

"""cup_pose_relay 순수 로직 검증. numpy+yaml만, ROS 불필요."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cup_pose_relay import (  # noqa: E402
    DEFAULT_EXTRINSICS,
    Extrinsics,
    cad_pose_to_base_body,
    load_extrinsics,
    select_best_detection,
)
from pour_obs_geometry import quat_apply  # noqa: E402

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
ROT_Z_90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])


def _ident_ext(**overrides) -> Extrinsics:
    fields = dict(
        cam_pos=np.zeros(3),
        cam_quat=IDENT.copy(),
        cad_to_body_pos=np.zeros(3),
        cad_to_body_quat=IDENT.copy(),
        base_frame="base_link",
    )
    fields.update(overrides)
    return Extrinsics(**fields)


# ── extrinsics yaml ─────────────────────────────────────────────────────────

def test_default_yaml_loads_and_validates():
    ext = load_extrinsics(DEFAULT_EXTRINSICS)
    assert ext.cam_quat.shape == (4,)
    assert np.linalg.norm(ext.cam_quat) == pytest.approx(1.0)
    assert ext.base_frame


def test_missing_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("camera:\n  position: [0,0,0]\n  orientation_wxyz: [1,0,0,0]\n")
    with pytest.raises(ValueError, match="cad_to_body"):
        load_extrinsics(bad)


def test_unnormalized_quat_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "camera:\n  position: [0,0,0]\n  orientation_wxyz: [1,1,0,0]\n"
        "cad_to_body:\n  position: [0,0,0]\n  orientation_wxyz: [1,0,0,0]\n"
        "base_frame: base_link\n"
    )
    with pytest.raises(ValueError, match="not normalized"):
        load_extrinsics(bad)


# ── 변환 체인 ────────────────────────────────────────────────────────────────

def test_identity_extrinsics_is_passthrough():
    pos, quat = cad_pose_to_base_body(
        _ident_ext(), np.array([0.1, 0.2, 0.3]), ROT_Z_90
    )
    assert np.allclose(pos, [0.1, 0.2, 0.3])
    assert np.allclose(quat, ROT_Z_90)


def test_camera_rotation_and_translation_applied():
    # 카메라가 base에서 (1,0,0.5), z축 90° 회전
    ext = _ident_ext(cam_pos=np.array([1.0, 0.0, 0.5]), cam_quat=ROT_Z_90)
    cad_cam = np.array([0.2, 0.0, 0.0])  # 카메라 x 방향 0.2m
    pos, quat = cad_pose_to_base_body(ext, cad_cam, IDENT)
    # base에서: cam원점 + Rz90·(0.2,0,0) = (1, 0.2, 0.5)
    assert np.allclose(pos, [1.0, 0.2, 0.5], atol=1e-12)
    assert np.allclose(quat_apply(quat, [1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_cad_correction_applied_after_camera_transform():
    # CAD 원점이 body보다 z로 -0.05 아래(즉 body = cad ∘ (0,0,0.05))인 컵이
    # 카메라 프레임에서 z축 90° 회전 상태 → 보정은 CAD 프레임에서 적용돼야 한다
    ext = _ident_ext(cad_to_body_pos=np.array([0.0, 0.0, 0.05]))
    pos, _ = cad_pose_to_base_body(ext, np.array([0.3, 0.0, 0.2]), ROT_Z_90)
    # CAD z축은 회전 후에도 base z와 정렬 → +0.05는 그대로 z로
    assert np.allclose(pos, [0.3, 0.0, 0.25], atol=1e-12)
    # 회전 보정도 합성 순서 확인: x축 90° 보정이면 최종 회전 = Rz90 ∘ Rx90
    rot_x_90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    ext2 = _ident_ext(cad_to_body_quat=rot_x_90)
    _, quat2 = cad_pose_to_base_body(ext2, np.zeros(3), ROT_Z_90)
    # cad y축(보정 후 body z가 되는 축)이 base에서 어디로 가는지:
    # Rz90 ∘ Rx90 을 (0,0,1)에 적용 = Rz90·(0,-1,0) = (1,0,0)
    assert np.allclose(quat_apply(quat2, [0, 0, 1]), [1, 0, 0], atol=1e-12)


# ── 후보 선택 ────────────────────────────────────────────────────────────────

def test_select_best_picks_highest_score():
    a = (0.4, np.zeros(3), IDENT)
    b = (0.9, np.ones(3), IDENT)
    assert select_best_detection([a, b], min_score=0.0) is b


def test_select_best_filters_min_score_and_handles_empty():
    a = (0.4, np.zeros(3), IDENT)
    assert select_best_detection([a], min_score=0.5) is None
    assert select_best_detection([], min_score=0.0) is None


# ── camera-frame PoseStamped 입력 ────────────────────────────────────────────

def test_posestamped_to_candidate_maps_fields():
    from cup_pose_relay import posestamped_to_candidate
    # camera-frame pose (pos, quat wxyz), score 기본 1.0
    score, pos, quat = posestamped_to_candidate(0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0)
    assert score == 1.0
    assert np.allclose(pos, [0.1, 0.2, 0.3])
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_posestamped_candidate_feeds_same_transform_as_detection():
    # 같은 camera-frame pose면 Detection3DArray 경로와 동일한 base 결과여야 한다
    from cup_pose_relay import posestamped_to_candidate, cad_pose_to_base_body
    ext = _ident_ext(cam_pos=np.array([1.0, 0.0, 0.5]), cam_quat=ROT_Z_90)
    _, pos_c, quat_c = posestamped_to_candidate(0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    pos, quat = cad_pose_to_base_body(ext, pos_c, quat_c)
    assert np.allclose(pos, [1.0, 0.2, 0.5], atol=1e-12)

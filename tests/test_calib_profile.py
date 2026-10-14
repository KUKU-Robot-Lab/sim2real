"""보정 프로필(hand-eye + 목 각도별 보정) — 순수부 테스트. ROS·하드웨어 무관."""
import numpy as np
import pytest
import yaml

import calib_profile as cp

MATRIX = np.eye(4)
MATRIX[:3, 3] = [0.03, 0.04, -0.01]


def _write(tmp_path, angles=None, **extra):
    body = {
        "profile": {"name": "t", "notes": "test"},
        "hand_eye": {"neck_to_camera": {"matrix": MATRIX.tolist()}},
        "angle_tolerance_deg": 1.0,
        "strict": True,
        "angles": angles if angles is not None else [
            {"pan_deg": 0.13, "tilt_deg": -20.0, "bias_xy_m": [0.001, -0.002], "n_frames": 3, "n_bolts": 12, "bolt_resid_mm": 1.1},
            {"pan_deg": 30.2, "tilt_deg": -25.0, "bias_xy_m": [0.0, 0.003], "n_frames": 1, "n_bolts": 8, "bolt_resid_mm": 1.6},
        ],
    }
    body.update(extra)
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_load_reads_hand_eye_and_angle_table(tmp_path):
    profile = cp.load_profile(_write(tmp_path))
    assert profile.name == "t" and profile.strict
    assert np.allclose(profile.neck_to_camera, MATRIX)
    assert len(profile.entries) == 2


def test_bias_inside_tolerance_is_returned_as_xyz_with_zero_z(tmp_path):
    profile = cp.load_profile(_write(tmp_path))
    assert np.allclose(cp.bias_for(profile, 0.5, -19.6), [0.001, -0.002, 0.0])
    assert np.allclose(cp.bias_for(profile, 30.0, -25.3), [0.0, 0.003, 0.0])


def test_nearest_entry_wins_when_two_are_within_tolerance(tmp_path):
    angles = [{"pan_deg": 0.0, "tilt_deg": -20.0, "bias_xy_m": [0.001, 0.0]}, {"pan_deg": 1.5, "tilt_deg": -20.0, "bias_xy_m": [0.002, 0.0]}]
    profile = cp.load_profile(_write(tmp_path, angles=angles, angle_tolerance_deg=2.0))
    assert cp.bias_for(profile, 1.2, -20.0)[0] == pytest.approx(0.002)


def test_strict_profile_refuses_an_uncalibrated_angle(tmp_path):
    profile = cp.load_profile(_write(tmp_path))
    with pytest.raises(cp.UncalibratedAngle):
        cp.bias_for(profile, -45.0, -20.0)


def test_lenient_profile_returns_zero_bias_outside_the_table(tmp_path):
    profile = cp.load_profile(_write(tmp_path, strict=False))
    assert np.allclose(cp.bias_for(profile, -45.0, -20.0), 0.0)


def test_pan_and_tilt_are_both_checked_against_the_tolerance(tmp_path):
    profile = cp.load_profile(_write(tmp_path))
    with pytest.raises(cp.UncalibratedAngle):
        cp.bias_for(profile, 0.1, -22.5)  # pan 은 맞지만 tilt 가 2.5° 어긋난다


def test_bad_files_are_rejected_with_a_clear_error(tmp_path):
    bad_matrix = _write(tmp_path)
    raw = yaml.safe_load(bad_matrix.read_text())
    raw["hand_eye"]["neck_to_camera"]["matrix"] = [[1, 0], [0, 1]]
    bad_matrix.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="4x4"):
        cp.load_profile(bad_matrix)
    huge = _write(tmp_path, angles=[{"pan_deg": 0, "tilt_deg": -20, "bias_xy_m": [0.2, 0.0]}])
    with pytest.raises(ValueError, match="bias"):
        cp.load_profile(huge)
    with pytest.raises(ValueError, match="angles"):
        cp.load_profile(_write(tmp_path, angles=[]))


def test_profile_name_resolves_to_the_config_directory(tmp_path, monkeypatch):
    path = _write(tmp_path)
    monkeypatch.setattr(cp, "PROFILE_DIR", tmp_path)
    assert cp.load_profile("t").name == "t"
    assert path.exists()
    with pytest.raises(FileNotFoundError):
        cp.load_profile("does_not_exist")


def test_dump_roundtrips(tmp_path):
    profile = cp.load_profile(_write(tmp_path))
    out = tmp_path / "out.yaml"
    cp.save_profile(profile, out)
    again = cp.load_profile(out)
    assert np.allclose(again.neck_to_camera, profile.neck_to_camera)
    assert [e.bias_xy_m for e in again.entries] == [e.bias_xy_m for e in profile.entries]


def test_frame_anchor_rotates_and_shifts_xy_and_adds_dz(tmp_path):
    path = _write(tmp_path, frame_anchor={"rot_deg": 2.0, "trans_xy_m": [0.01, -0.02], "dz_m": -0.003, "source": "t"})
    profile = cp.load_profile(path)
    out = cp.anchor_position(profile, np.array([0.3, 0.1, 0.29]))
    c, s = np.cos(np.radians(2.0)), np.sin(np.radians(2.0))
    assert np.allclose(out, [c * 0.3 - s * 0.1 + 0.01, s * 0.3 + c * 0.1 - 0.02, 0.287])


def test_profile_without_anchor_leaves_positions_unchanged(tmp_path):
    profile = cp.load_profile(_write(tmp_path))
    assert np.allclose(cp.anchor_position(profile, np.array([0.3, 0.1, 0.29])), [0.3, 0.1, 0.29])


def test_anchor_roundtrips_and_is_sanity_checked(tmp_path):
    profile = cp.load_profile(_write(tmp_path, frame_anchor={"rot_deg": 0.4, "trans_xy_m": [0.012, -0.008], "dz_m": 0.003, "source": "t"}))
    out = tmp_path / "o.yaml"
    cp.save_profile(profile, out)
    again = cp.load_profile(out)
    assert again.anchor is not None and again.anchor.trans_xy_m == (0.012, -0.008)
    with pytest.raises(ValueError, match="anchor"):
        cp.load_profile(_write(tmp_path, frame_anchor={"rot_deg": 20.0, "trans_xy_m": [0.0, 0.0], "dz_m": 0.0}))


def test_known_z_snaps_a_standing_object_to_the_table_plus_origin_height(tmp_path):
    profile = cp.load_profile(_write(tmp_path, known_z={"table_top_m": 0.205, "tolerance_m": 0.015}))
    assert cp.snap_z(profile, 0.2863, 0.085) == pytest.approx(0.290)      # FP++ 깊이 편향으로 4 mm 낮게 나온 값
    assert cp.snap_z(profile, 0.34, 0.085) == pytest.approx(0.34)         # 들렸거나 기운 물체는 그대로 둔다
    assert cp.snap_z(cp.load_profile(_write(tmp_path)), 0.2863, 0.085) == pytest.approx(0.2863)  # known_z 없으면 그대로


def test_known_z_roundtrips_and_is_sanity_checked(tmp_path):
    profile = cp.load_profile(_write(tmp_path, known_z={"table_top_m": 0.205, "tolerance_m": 0.015}))
    out = tmp_path / "z.yaml"
    cp.save_profile(profile, out)
    assert cp.load_profile(out).known_z.table_top_m == 0.205
    with pytest.raises(ValueError, match="known_z"):
        cp.load_profile(_write(tmp_path, known_z={"table_top_m": 0.5, "tolerance_m": 0.015}))

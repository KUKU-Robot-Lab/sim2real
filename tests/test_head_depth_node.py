"""head_depth_node 가 hdgp real_depth.py(고정 커밋 사본)의 블록 중앙값 · K 축소를 그대로 쓰는지(rclpy 없이).

hdgp git 저장소(HDGP_DIR, 기본 ~/rl_ws/hdgp)가 없으면 건너뛴다. 있는데 커밋 · md5 핀이 어긋나면 실패한다.
"""
from __future__ import annotations

import numpy as np
import pytest
from policy_control import hdgp_pin
from policy_control import student_policy as SP

if not (hdgp_pin.hdgp_tree() / ".git").exists():
    pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}", allow_module_level=True)

import head_depth_node as hdn  # noqa: E402

from pathlib import Path  # noqa: E402

STUDENT = Path(__file__).resolve().parents[1] / "deploy/policies/rh56f1/aglt/right_student1104e"
EXTRINSICS = STUDENT / "camera/extrinsics_r20u5_40e6496.yaml"      # 학습 때 값을 얼린 사본(추적 파일)


def test_to_small_matches_real_depth():
    rng = np.random.default_rng(0)
    d = rng.integers(200, 1500, size=(720, 1280), dtype=np.uint16)
    d[rng.random(d.shape) < 0.2] = 0          # 흩어진 구멍
    d[:8, :8] = 0                             # 블록 전체 구멍 → 0
    d[8:16, :8] = 1000
    d[8:13, :8] = 0                           # 유효 24/64 < 0.5 → 0
    out = hdn.to_small(d, hdn.real_depth.BLOCK, hdn.real_depth.MIN_VALID)
    ref = hdn.real_depth.block_median(d.astype(np.float32) / 1000.0)
    assert out.shape == (90, 160) and out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6)
    assert out[0, 0] == 0.0 and out[1, 0] == 0.0
    assert (out > 0).mean() > 0.9


def test_scale_k_matches_real_depth():
    k = np.array(hdn.real_depth.D435I_K_1103, dtype=np.float64).reshape(3, 3)
    got = hdn.real_depth.scale_k(k, 8)
    np.testing.assert_allclose(got[[0, 1], [0, 1]], k[[0, 1], [0, 1]] / 8)
    np.testing.assert_allclose(got[:2, 2], (k[:2, 2] + 0.5) / 8 - 0.5)


def test_floor_cut_matches_real_depth():
    rd = hdn.real_depth
    extr, _ = hdn.load_extrinsics(EXTRINSICS)
    # 합성 장면: 앞쪽 절반은 탁자(0.6 m), 뒤쪽 절반은 탁자보다 훨씬 먼 바닥(1.5 m)
    d = np.full((720, 1280), 600, dtype=np.uint16)
    d[360:] = 1500
    pos, quat = extr
    k = np.array(rd.D435I_K_1103)
    u, v = np.meshgrid(np.arange(1280.0), np.arange(720.0))
    r2 = rd.quat_wxyz_to_mat(quat)[2]
    base_z = pos[2] + d * 1e-3 * ((u - k[0, 2]) / k[0, 0] * r2[0] + (v - k[1, 2]) / k[1, 1] * r2[1] + r2[2])
    below = base_z < hdn.TABLE_Z - rd.FLOOR_MARGIN_M
    assert below.any() and (~below).any()
    cut = rd.floor_cut_frame(d, rd.D435I_K_1103, extr, hdn.TABLE_Z, rd.FLOOR_MARGIN_M)
    assert cut.dtype == np.uint16 and (cut[below] == 0).all() and (cut[~below] == d[~below]).all()
    out = hdn.to_small(d, rd.BLOCK, rd.MIN_VALID, hdn.make_cut(extr))
    np.testing.assert_array_equal(out, rd.block_median(cut.astype(np.float32) * 1e-3, rd.BLOCK, rd.MIN_VALID))
    assert not np.array_equal(out, hdn.to_small(d, rd.BLOCK, rd.MIN_VALID))


def test_precomputed_rz_equals_floor_cut_frame():
    rd = hdn.real_depth
    extr, _ = hdn.load_extrinsics(EXTRINSICS)
    d = np.random.default_rng(1).integers(0, 2500, size=(720, 1280), dtype=np.uint16)   # 경계 근처 화소가 많게
    ref = rd.floor_cut_frame(d, rd.D435I_K_1103, extr, hdn.TABLE_Z, rd.FLOOR_MARGIN_M)
    got = rd.floor_cut(d.astype(np.float32) * 1e-3, *hdn.make_cut(extr))
    np.testing.assert_array_equal(got, ref.astype(np.float32) * 1e-3)
    assert 0 < ((ref == 0) & (d > 0)).mean() < 1


def test_check_info():
    k = np.array(hdn.real_depth.D435I_K_1103)
    assert hdn.check_info(1280, 720, k.ravel()) == (True, "")
    assert hdn.check_info(640, 480, k.ravel())[0] is False
    k2 = k.copy()
    k2[1, 2] += 0.6
    ok, note = hdn.check_info(1280, 720, k2.ravel())
    assert ok and note


def test_real_depth_comes_from_the_pinned_snapshot():
    cfg = hdgp_pin.load_snapshot_cfg(SP.SNAPSHOT_CFG)
    src = hdn.REAL_DEPTH_SRC
    assert src["commit"] == cfg["commit"] and src["md5"] == cfg["md5"][SP.REAL_DEPTH]
    assert "/s2r_hdgp/" in src["path"] and not src["path"].startswith(str(hdgp_pin.hdgp_tree()))


def _frame(d):
    return dict(height=d.shape[0], width=d.shape[1], encoding="16UC1", step=d.shape[1] * 2, data=d.tobytes())


def test_frame_to_small_checks_the_frame():
    rd = hdn.real_depth
    d = np.random.default_rng(2).integers(200, 1500, size=(720, 1280), dtype=np.uint16)
    np.testing.assert_array_equal(hdn.frame_to_small(**_frame(d), block=8, min_valid=0.5),
                                  hdn.to_small(d, rd.BLOCK, rd.MIN_VALID))
    for bad, match in [({"encoding": "32FC1"}, "encoding"), ({"step": 1280}, "바이트"),
                       ({"data": d.tobytes()[:-2]}, "바이트")]:
        with pytest.raises(hdn.FrameError, match=match):
            hdn.frame_to_small(**{**_frame(d), **bad}, block=8, min_valid=0.5)
    with pytest.raises(hdn.FrameError, match="1280x720"):
        hdn.frame_to_small(**_frame(d[:480, :640].copy()), block=8, min_valid=0.5)


def test_node_config_matches_a_training_depth_pre():
    rd = hdn.real_depth
    _, md5 = hdn.load_extrinsics(EXTRINSICS)
    cfg = hdn.node_config(rd.BLOCK, rd.MIN_VALID, hdn.TABLE_Z, rd.FLOOR_MARGIN_M, str(EXTRINSICS), md5)
    k = rd.scale_k(np.array(rd.D435I_K_1103), 8)
    pre = {"block": 8, "min_valid": 0.5, "render_wh": [1280, 720], "out_wh": [160, 90], "K": k.tolist(),
           "extrinsics": "/train/host/r20u5.yaml", "extrinsics_md5": hdn.EXTRINSICS_MD5,
           "floor_cut_z_m": hdn.TABLE_Z - rd.FLOOR_MARGIN_M}
    SP.check_depth_config(pre, cfg)
    with pytest.raises(SP.RhAgltError, match="floor_cut_z_m"):
        SP.check_depth_config(pre, {**cfg, "floor_cut_z_m": cfg["floor_cut_z_m"] + 0.01})


def test_rgb_to_small_is_block_mean_in_rgb_order():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    f = {"height": 720, "width": 1280, "step": 1280 * 3, "data": c.tobytes()}
    out = hdn.rgb_to_small(encoding="rgb8", block=8, **f)
    ref = (c.astype(np.float64) / 255.0).reshape(90, 8, 160, 8, 3).mean((1, 3))
    assert out.shape == (90, 160, 3) and out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6)
    bgr = hdn.rgb_to_small(encoding="bgr8", block=8, **{**f, "data": c[..., ::-1].copy().tobytes()})
    np.testing.assert_allclose(bgr, out, rtol=0, atol=1e-6)
    for bad, match in [({"encoding": "mono8"}, "encoding"), ({"step": 1280}, "바이트"), ({"width": 640, "step": 1920}, "1280x720")]:
        with pytest.raises(hdn.FrameError, match=match):
            hdn.rgb_to_small(**{**f, "encoding": "rgb8", "block": 8, **bad})


def test_nearest_color_pairs_within_slop_only():
    buf = [(1_000_000_000, "a"), (1_033_000_000, "b")]
    assert hdn.nearest_color(buf, 1_030_000_000) == "b"
    assert hdn.nearest_color(buf, 1_000_000_000 - 10_000_000) == "a"
    assert hdn.nearest_color(buf, 1_100_000_000) is None and hdn.nearest_color([], 0) is None


def test_extrinsics_must_be_frozen_and_match(tmp_path):
    if not (STUDENT / "student").is_dir():
        pytest.skip("학생 ckpt 폴더 없음")
    path, md5, rgb = hdn.resolve_extrinsics(str(STUDENT), None, False)
    assert Path(path) == EXTRINSICS and md5 == hdn.EXTRINSICS_MD5 and rgb is True
    assert hdn.resolve_extrinsics(None, str(EXTRINSICS), False) == (str(EXTRINSICS), hdn.EXTRINSICS_MD5, False)
    moved = tmp_path / "r20u5.yaml"
    moved.write_text(EXTRINSICS.read_text() + "# 다시 잼\n")
    with pytest.raises(hdn.PinError, match="시작하지 않는다"):
        hdn.resolve_extrinsics(None, str(moved), False)


def test_head_block_only_at_the_frozen_head_pose():
    """★11.05 깊이는 얼린 외부 파라미터의 head_pose(± tol_tick) 에서만 낸다 — 낡음 · 없음 · 자세 밖은 보류."""
    from head_fk_chain import urdf_from_encoder
    from head_pose_check import load_head_pose
    from head_position_hold_node import tick_to_deg

    hp = load_head_pose(EXTRINSICS)
    assert hp is not None and hp.ticks == {"pan": 1825, "tilt": 2815}

    def deg(pan_tick, tilt_tick):     # 엔코더 틱 → /head/joint_states 의 URDF 각(도)
        return urdf_from_encoder(tick_to_deg(pan_tick), tick_to_deg(tilt_tick))

    assert hdn.head_block(hp, deg(1825, 2815), 10.0, 10.5) is None
    assert hdn.head_block(hp, deg(1828, 2812), 10.0, 10.5) is None
    assert "캘리브 자세" in hdn.head_block(hp, deg(1835, 2815), 10.0, 10.5)
    assert "낡았다" in hdn.head_block(hp, deg(1825, 2815), 10.0, 12.0)
    assert "못 받았다" in hdn.head_block(hp, None, None, 10.0)

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
    d[rng.random(d.shape) < 0.3] = 0          # 흩어진 구멍 30 %
    d[:8, :8] = 0                             # 블록 전체 구멍 → 0
    d[8:16, :8] = 1000
    d[8:13, :8] = 0                           # 유효 24/64 < 0.5 → 0
    out = hdn.to_small(d, hdn.real_depth.BLOCK, hdn.real_depth.MIN_VALID)
    ref = hdn.real_depth.real_like(d.astype(np.float32) * 1e-3)    # 학습 경로: mm 반올림 → 블록 중앙값
    assert out.shape == (90, 160) and out.dtype == np.float32
    np.testing.assert_array_equal(out, ref)
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
    np.testing.assert_array_equal(out, rd.block_median(rd.mm_quantize(cut.astype(np.float32) * 1e-3), rd.BLOCK, rd.MIN_VALID))
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
    for bad, match in [({"encoding": "mono8"}, "encoding"), ({"step": 1280}, "바이트"), ({"width": 640, "step": 1920}, "1280x720"),
                       ({"block": 7}, "정수배")]:
        with pytest.raises(hdn.FrameError, match=match):
            hdn.rgb_to_small(**{**f, "encoding": "rgb8", "block": 8, **bad})


def test_nearest_color_pairs_within_slop_only():
    buf = [(1_000_000_000, "a"), (1_033_000_000, "b")]
    assert hdn.nearest_color(buf, 1_030_000_000) == "b"
    assert hdn.nearest_color(buf, 1_000_000_000 - 10_000_000) == "a"
    assert hdn.nearest_color(buf, 1_100_000_000) is None and hdn.nearest_color([], 0) is None


def _img(t_ns, enc, nbytes):
    from types import SimpleNamespace as NS
    return NS(header=NS(stamp=NS(sec=t_ns // 10**9, nanosec=t_ns % 10**9)), height=720, width=1280,
              encoding=enc, step=nbytes // 720, data=bytes(nbytes))


def _depth_msg(t_ns):
    return _img(t_ns, "16UC1", 1280 * 720 * 2)


def _color_msg(t_ns):
    return _img(t_ns, "rgb8", 1280 * 720 * 3)


def _node(monkeypatch):
    """rclpy 없이 HeadDepth 콜백만 돌린다. 발행 msg 와 경고를 모은다."""
    from collections import deque
    from types import SimpleNamespace as NS
    out, warns = [], []
    pub = NS(publish=out.append)
    info = NS(distortion_model="plumb_bob", d=[], k=np.eye(3).ravel().tolist())
    h = object.__new__(hdn.HeadDepth)
    h.__dict__.update(node=NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=0))), args=NS(block=8, min_valid=0.5),
                      cut=None, log=NS(warning=lambda m, **k: warns.append(m), error=lambda m, **k: warns.append(m)),
                      Image=lambda: NS(), CameraInfo=lambda: NS(), info=info, info_ok=True, lat_ms=[],
                      pending=None, pending_t=0.0, last_ns=None, head_pose=None, head_deg=None, head_t=None,
                      rgb=True, color=deque(maxlen=hdn.COLOR_BUF), pub=pub, pub_rgb=pub, pub_info=NS(publish=lambda m: None))
    monkeypatch.setattr(hdn, "head_block", lambda *a: None)
    return h, out, warns


T0 = 5 * 10**9
DT = 33_333_333


def test_depth_before_its_color_waits_then_publishes(monkeypatch):
    """짝 color 보다 먼저 온 깊이는 버리지 않고 color 가 오면 낸다. color 가 이미 지나갔으면 버린다."""
    import array
    h, out, _ = _node(monkeypatch)
    h._color(_color_msg(T0 - DT))                 # 앞 프레임 color
    h._depth(_depth_msg(T0))                      # 짝 color 아직 없음 → 대기
    assert out == [] and h.pending is not None
    h._color(_color_msg(T0))
    assert [m.encoding for m in out] == ["32FC3", "32FC1"] and h.pending is None
    assert all(isinstance(m.data, array.array) for m in out)    # bytes 대입은 rclpy 가 원소별 검사해 느리다
    h._color(_color_msg(T0 + 3 * DT))
    h._depth(_depth_msg(T0 + DT))                 # 짝 color 는 이미 지나감 → 버림
    assert len(out) == 2 and h.pending is None


def test_color_never_comes_warns_and_drops(monkeypatch):
    h, out, warns = _node(monkeypatch)
    for i in range(3):
        h._depth(_depth_msg(T0 + i * DT))         # color 가 한 번도 안 옴
    assert out == [] and _ns_of(h.pending) == T0 + 2 * DT
    assert sum("짝 없이 깊이를 버린다" in w for w in warns) == 2


def test_next_depth_replaces_pending_and_late_pending_expires(monkeypatch):
    h, out, warns = _node(monkeypatch)
    h._color(_color_msg(T0 - DT))
    h._depth(_depth_msg(T0))
    h._depth(_depth_msg(T0 + DT))                 # 앞 것은 사유와 함께 버린다
    assert _ns_of(h.pending) == T0 + DT and any("다음 깊이가 왔다" in w for w in warns)
    h._depth(_depth_msg(T0))                      # 보류 중인 것보다 이른 깊이 → 버림, 보류는 그대로
    assert _ns_of(h.pending) == T0 + DT and out == []
    h.pending_t -= 1.0                            # 보류가 PEND_MAX_S 를 넘김
    h._color(_color_msg(T0 - DT // 2))            # 짝이 아닌 color → 보류 깊이를 버린다
    assert h.pending is None and any("넘게 보류" in w for w in warns) and out == []


def test_published_stamps_strictly_increase(monkeypatch):
    h, out, warns = _node(monkeypatch)
    for t in [T0, T0 + DT, T0 + 2 * DT]:
        h._color(_color_msg(t))
    for t in [T0 + DT, T0, T0 + DT, T0 + 2 * DT]:   # 역행 · 중복 stamp 섞음
        h._depth(_depth_msg(t))
    got = [_ns_of(m) for m in out if m.encoding == "32FC1"]
    assert got == [T0 + DT, T0 + 2 * DT]
    assert sum("이르거나 같아" in w for w in warns) == 2


def _ns_of(m):
    return m.header.stamp.sec * 10**9 + m.header.stamp.nanosec


def test_torch_block_median_equals_numpy_on_bad_values():
    """노드는 hdgp block_median 의 torch 경로를 쓴다. 학습 쪽 numpy 경로와 비트까지 같아야 한다."""
    import torch
    rd = hdn.real_depth
    rng = np.random.default_rng(3)
    m = rng.uniform(0.2, 1.5, size=(720, 1280)).astype(np.float32)
    m[rng.random(m.shape) < 0.3] = 0.0            # 구멍 30 %
    for v in (np.nan, np.inf, -np.inf, -0.5):
        m[rng.random(m.shape) < 0.02] = v
    m[:8, :8] = np.nan                            # 블록 전체가 비유효
    m[8:16, :8] = -1.0
    got = rd.block_median(torch.from_numpy(m), rd.BLOCK, rd.MIN_VALID).numpy()
    np.testing.assert_array_equal(got, rd.block_median(m, rd.BLOCK, rd.MIN_VALID))
    assert np.isfinite(got).all() and (got >= 0).all() and got[0, 0] == 0 and got[1, 0] == 0


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

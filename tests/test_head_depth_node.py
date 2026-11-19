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
EXTRINSICS = STUDENT / "camera/extrinsics_m1117.yaml"      # 11.17 테이블 마커 판 사본(ckpt 는 r20u5 로 배움 → ckpt_md5)


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
    cfg = hdn.node_config(rd.BLOCK, rd.MIN_VALID, hdn.TABLE_Z, rd.FLOOR_MARGIN_M, str(EXTRINSICS), md5,
                          ckpt_md5=hdn.CKPT_EXTRINSICS_MD5)
    k = rd.scale_k(np.array(rd.D435I_K_1103), 8)
    pre = {"block": 8, "min_valid": 0.5, "render_wh": [1280, 720], "out_wh": [160, 90], "K": k.tolist(),
           "extrinsics": "/train/host/r20u5.yaml", "extrinsics_md5": hdn.CKPT_EXTRINSICS_MD5,
           "floor_cut_z_m": hdn.TABLE_Z - rd.FLOOR_MARGIN_M}
    SP.check_depth_config(pre, cfg)
    with pytest.raises(SP.RhAgltError, match="extrinsics_md5"):   # 새 판 파일인데 ckpt 대조 값이 없으면 거부
        SP.check_depth_config(pre, {k: v for k, v in cfg.items() if k != "extrinsics_ckpt_md5"})
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
                      pending=None, pending_t=0.0, last_ns=None, head_pose=None, head_deg=None, head_t=None, gate=True, clip=None,
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
    path, md5, ckpt_md5, rgb = hdn.resolve_extrinsics(str(STUDENT), None, False)
    assert Path(path) == EXTRINSICS and md5 == hdn.EXTRINSICS_MD5 and ckpt_md5 == hdn.CKPT_EXTRINSICS_MD5 and rgb is True
    assert hdn.resolve_extrinsics(None, str(EXTRINSICS), False) == (str(EXTRINSICS), hdn.EXTRINSICS_MD5,
                                                                   hdn.CKPT_EXTRINSICS_MD5, False)
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
    assert hp is not None and hp.ticks == {"pan": 1825, "tilt": 2833}

    def deg(pan_tick, tilt_tick):     # 엔코더 틱 → /head/joint_states 의 URDF 각(도)
        return urdf_from_encoder(tick_to_deg(pan_tick), tick_to_deg(tilt_tick))

    assert hdn.head_block(hp, deg(1825, 2833), 10.0, 10.5) is None
    assert hdn.head_block(hp, deg(1828, 2830), 10.0, 10.5) is None
    assert "캘리브 자세" in hdn.head_block(hp, deg(1835, 2833), 10.0, 10.5)
    assert "낡았다" in hdn.head_block(hp, deg(1825, 2833), 10.0, 12.0)
    assert "못 받았다" in hdn.head_block(hp, None, None, 10.0)


# --- 11.18 --free-head(머리를 정책이 움직이는 rh_cap 깊이 학생) ---

def test_free_head_parses_without_extrinsics_and_sets_no_cut_no_gate(tmp_path, monkeypatch):
    """(a) 외부 파라미터 파일 없이 시작한다 — 컷 · head_pose 없음, config 에 모드 키."""
    monkeypatch.chdir(tmp_path)                   # 상대 경로로 기본 yaml 을 찾는 일이 없게
    cut, cfg, head_pose = hdn.setup(hdn.parse_args(["--free-head"]))
    assert cut is None and head_pose is None
    assert cfg["floor_cut"] is False and cfg["head_gate"] is False
    assert cfg["extrinsics"] is None and cfg["extrinsics_md5"] is None and cfg["extrinsics_ckpt_md5"] is None
    assert cfg["floor_cut_z_m"] is None and cfg["rgb"] is False
    assert cfg["block"] == 8 and cfg["out_wh"] == [160, 90] and cfg["render_wh"] == [1280, 720]
    import json
    json.dumps(cfg)                               # latched String 으로 낼 수 있어야 한다


def test_free_head_with_policy_dir_or_extrinsics_is_an_argparse_error():
    """(e) --free-head 와 --policy-dir · --extrinsics 동시 지정 거부. 셋 다 없어도 거부(기본 그대로)."""
    for argv in (["--free-head", "--policy-dir", str(STUDENT)], ["--free-head", "--extrinsics", str(EXTRINSICS)], []):
        with pytest.raises(SystemExit):
            hdn.parse_args(argv)


def test_default_argv_still_needs_frozen_extrinsics():
    """(d) 인자 없는 기본 경로: 컷 · head_pose · 기존 config 키(새 키 없음) 그대로."""
    cut, cfg, head_pose = hdn.setup(hdn.parse_args(["--extrinsics", str(EXTRINSICS)]))
    assert cut is not None and head_pose is not None
    assert "floor_cut" not in cfg and "head_gate" not in cfg
    assert cfg["extrinsics_md5"] == hdn.EXTRINSICS_MD5 and cfg["floor_cut_z_m"] == pytest.approx(hdn.TABLE_Z - hdn.real_depth.FLOOR_MARGIN_M)


def test_free_head_output_equals_cut_path_with_all_false_mask():
    """(b) 같은 깊이 → free-head 출력 = 컷 마스크가 전부 False 인 기존 경로. 탁자 아래 화소가 지워지지 않는다."""
    rd = hdn.real_depth
    extr, _ = hdn.load_extrinsics(EXTRINSICS)
    d = np.full((720, 1280), 600, dtype=np.uint16)
    d[360:] = 1500                                 # 뒤쪽 절반 = 탁자 아래 바닥(기존 경로는 지운다)
    cut, _, _ = hdn.setup(hdn.parse_args(["--free-head"]))
    free = hdn.frame_to_small(**_frame(d), block=8, min_valid=0.5, cut=cut)
    cam_z, rz, _ = hdn.make_cut(extr)
    never = (cam_z, rz, -1e9)                      # z_min 을 아주 낮게 → 지워지는 화소 없음
    assert not (rd.floor_cut(d.astype(np.float32) * 1e-3, *never) == 0).any()
    np.testing.assert_array_equal(free, hdn.frame_to_small(**_frame(d), block=8, min_valid=0.5, cut=never))
    fixed = hdn.frame_to_small(**_frame(d), block=8, min_valid=0.5, cut=hdn.make_cut(extr))
    assert (free > 0).all() and (fixed == 0).any()   # 기존 경로는 바닥을 0 으로, free-head 는 그대로


def test_free_head_publishes_when_head_is_off_the_calib_pose(monkeypatch):
    """(c) 머리 관절이 yaml 자세에서 벗어나도(· 낡아도 · 없어도) 프레임이 나온다. 머리 고정 경로는 보류."""
    from head_pose_check import load_head_pose
    h, out, warns = _node(monkeypatch)
    monkeypatch.undo()                             # 진짜 head_block 을 쓴다
    h.rgb = False
    import time
    h.gate, h.head_pose, h.head_deg, h.head_t = False, None, (40.0, -30.0), time.monotonic()   # free-head: config head_gate false
    h._depth(_depth_msg(T0))
    assert [m.encoding for m in out] == ["32FC1"], warns
    h.gate, h.head_pose = True, load_head_pose(EXTRINSICS)   # 같은 각을 머리 고정 경로로 → 보류
    h._depth(_depth_msg(T0 + DT))
    assert len(out) == 1 and any("보류" in w and "캘리브 자세" in w for w in warns), warns


def test_fixed_head_consumer_rejects_free_head_config():
    """머리 고정 학생(check_depth_config)은 free-head 깊이를 받지 않는다."""
    rd = hdn.real_depth
    pre = {"block": 8, "min_valid": 0.5, "render_wh": [1280, 720], "out_wh": [160, 90],
           "K": rd.scale_k(np.array(rd.D435I_K_1103), 8).tolist(), "extrinsics_md5": hdn.CKPT_EXTRINSICS_MD5,
           "floor_cut_z_m": hdn.TABLE_Z - rd.FLOOR_MARGIN_M}
    with pytest.raises(SP.RhAgltError, match="extrinsics_md5.*floor_cut_z_m"):
        SP.check_depth_config(pre, hdn.free_head_config(rd.BLOCK, rd.MIN_VALID))


# --- 11.18 리뷰 반영: free-head 깊이 클립 · 명시 게이트 · 안 쓰는 인자 거부 ---

def _frame_with_far_and_near(block: int = 8) -> np.ndarray:
    """블록 (0,0): 12 m · 0.005 m · 0.5 m 섞임. 블록 (0,1): 전부 12 m. 나머지 1 m."""
    d = np.full((720, 1280), 1000, dtype=np.uint16)
    d[:block, :block] = 500
    d[0, :4] = 12000
    d[1, :2] = 5
    d[:block, block:2 * block] = 12000
    return d


def test_free_head_clip_zeroes_out_of_range_before_block_median():
    """(a) 12 m · 0.005 m 화소는 0(구멍) → 중앙값은 가까운 0.5 m 만으로. 전부 12 m 인 블록은 0."""
    d = _frame_with_far_and_near()
    small = hdn.to_small(d, 8, 0.5, None, hdn.FREE_HEAD_CLIP_M)
    assert small[0, 0] == np.float32(0.5) and small[0, 1] == 0.0 and small[0, 2] == np.float32(1.0)
    assert small.max() <= hdn.FREE_HEAD_CLIP_M[1]
    # 유효 비율도 0 을 무효로 센다: 0.5 m 화소 58/64 → min_valid 0.95 면 블록이 0
    assert hdn.to_small(d, 8, 0.95, None, hdn.FREE_HEAD_CLIP_M)[0, 0] == 0.0
    via_frame = hdn.frame_to_small(720, 1280, "16UC1", 1280 * 2, d.tobytes(), 8, 0.5, None, hdn.FREE_HEAD_CLIP_M)
    assert np.array_equal(via_frame, small)


def test_default_path_has_no_clip_and_keeps_far_pixels():
    """(b) clip None(기본 경로)는 같은 프레임에서 12 m 를 그대로 낸다 — 회귀 0."""
    d = _frame_with_far_and_near()
    small = hdn.to_small(d, 8, 0.5)
    assert small[0, 1] == np.float32(12.0)
    _, cfg, _ = hdn.setup(hdn.parse_args(["--extrinsics", str(EXTRINSICS)]))
    assert "clip_m" not in cfg


def test_free_head_clip_matches_consumer_and_config():
    """(c) 클립 상한 == cap_student.DEPTH_MAX_M, config 에 clip_m."""
    from policy_control import cap_student
    assert hdn.FREE_HEAD_CLIP_M[1] == cap_student.DEPTH_MAX_M
    _, cfg, _ = hdn.setup(hdn.parse_args(["--free-head"]))
    assert cfg["clip_m"] == [0.01, 10.0]


def _fake_ros(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    for name in ("rclpy", "rclpy.qos", "sensor_msgs", "sensor_msgs.msg", "std_msgs", "std_msgs.msg"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    return MagicMock()


def test_gate_on_without_head_pose_is_refused(monkeypatch):
    node = _fake_ros(monkeypatch)
    _, cfg, _ = hdn.setup(hdn.parse_args(["--extrinsics", str(EXTRINSICS)]))
    with pytest.raises(ValueError, match="head_pose"):
        hdn.HeadDepth(node, hdn.parse_args(["--extrinsics", str(EXTRINSICS)]), None, cfg, None)
    with pytest.raises(ValueError, match="head_pose"):
        hdn.HeadDepth(node, None, None, {"rgb": False}, None)     # head_gate 키 없음 = 켬


def test_setup_output_wires_gate_and_clip_into_node(monkeypatch):
    """setup 반환값을 그대로 HeadDepth 에 → free-head 는 게이트 끔 · 클립 켬, 고정 경로는 게이트 켬 · 클립 없음."""
    node = _fake_ros(monkeypatch)
    args = hdn.parse_args(["--free-head"])
    h = hdn.HeadDepth(node, args, *hdn.setup(args))
    assert h.gate is False and h.clip == hdn.FREE_HEAD_CLIP_M
    args = hdn.parse_args(["--extrinsics", str(EXTRINSICS)])
    h = hdn.HeadDepth(node, args, *hdn.setup(args))
    assert h.gate is True and h.clip is None and h.head_pose is not None


def test_free_head_rejects_unused_args():
    for extra in (["--rgb"], ["--table-z", "0.3"], ["--floor-margin", "0.05"]):
        with pytest.raises(SystemExit):
            hdn.parse_args(["--free-head", *extra])
    hdn.parse_args(["--free-head", "--table-z", str(hdn.TABLE_Z)])   # 기본값과 같으면 통과

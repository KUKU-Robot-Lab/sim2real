"""컵홀더 마커 자동 추정(scripts/calib/cup_holder_pose.py) — STL 마커 면 · 합성 영상으로 자세 복원 · 공유 축."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "calib"))
sys.path.insert(0, str(ROOT / "scripts"))

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
import cup_holder_pose as H  # noqa: E402

K = np.array([[908.35, 0.0, 639.17], [0.0, 907.88, 363.88], [0.0, 0.0, 1.0]])
W, HH = 1280, 720
CFG = H.load_cfg()
#: arm4090 10.01 머리 카메라(head_home_rh56f1)
ARM4090 = H.T_from([0.050504, 0.037916, 0.820846], [0.130161, -0.69891, 0.696068, -0.100366])


def _look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """camera_optical(x 오른쪽 · y 아래 · z 앞) → base."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = eye
    return T


#: 마커가 잘 보이는 낮은 시점(로봇 쪽 −x 에서 비스듬히) — 수식 검증용
LOW = _look_at(np.array([-0.05, 0.0, 0.40]), np.array([0.30, 0.0, 0.21]))


def _render(T_bc, poses: dict[int, tuple], k: int = 0, noise: float = 0.0, seed: int = 0,
            pad: int = 32, bg: int = 90, white: int = 255) -> np.ndarray:
    """회색 배경에 마커를 투영해 붙인 RGB. k = 마커가 법선 둘레로 돈 횟수(×90°)."""
    ar = cv2.aruco
    d = (ar.getPredefinedDictionary if hasattr(ar, "getPredefinedDictionary") else ar.Dictionary_get)(
        ar.DICT_6X6_250)
    draw = ar.generateImageMarker if hasattr(ar, "generateImageMarker") else ar.drawMarker
    img = np.full((HH, W), bg, np.uint8)
    for i, p in poses.items():
        T = H.T_planar(*p)
        # 흰 여백 포함 정사각형(마커 30 mm + 양쪽 4 mm)
        m = np.where(draw(d, i, 240) > 127, white, 30).astype(np.uint8)
        tile = np.full((240 + 2 * pad, 240 + 2 * pad), bg, np.uint8)
        tile[pad:pad + 240, pad:pad + 240] = m
        c = CFG.corners_stl
        ctr = c.mean(0)
        big = ctr + (c - ctr) * (240 + 2 * pad) / 240
        obj = H.rolled(big, k)         # 타일 TL,TR,BR,BL 이 놓이는 STL 점
        uv, _ = H.project(T_bc, K, obj @ T[:3, :3].T + T[:3, 3])
        src = np.float32([[0, 0], [tile.shape[1], 0], [tile.shape[1], tile.shape[0]], [0, tile.shape[0]]])
        Hm = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
        warped = cv2.warpPerspective(tile, Hm, (W, HH), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(tile, 255), Hm, (W, HH), flags=cv2.INTER_NEAREST)
        img[mask > 0] = warped[mask > 0]
    if noise:
        img = np.clip(img + np.random.default_rng(seed).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def test_stl_marker_face_is_the_30mm_pad_on_minus_x():
    """10.03 새 홀더(hdgp cup_holder): −x 돌출 패드(x=−41, 30×30) 가운데에 22 mm 마커 — 흰 여백 4 mm."""
    tris = H.load_stl(CFG.stl) * CFG.stl_scale
    P = tris.reshape(-1, 3)
    on = P[np.isclose(P[:, 0], -0.041, atol=1e-6)]
    assert on.size, "x=−41 mm 면이 없다"
    lo, hi = on.min(0), on.max(0)
    assert np.allclose([lo[1], hi[1], lo[2], hi[2]], [-0.015, 0.015, -0.030, 0.0], atol=1e-6)
    c = CFG.corners_stl
    assert np.allclose(c[:, 0], -0.041)
    assert np.allclose(sorted(c[:, 1]), [-0.011, -0.011, 0.011, 0.011])
    assert np.allclose(sorted(c[:, 2]), [-0.026, -0.026, -0.004, -0.004])
    assert CFG.base_bottom_m == pytest.approx(-0.030)              # 바닥은 마커 아래 변(−26)이 아니라 받침 바닥
    # 마커 z(=TL→TR × TL→BL 의 반대) 가 면 바깥(−x)
    n = np.cross(c[3] - c[0], c[1] - c[0])
    assert n[0] < 0


def test_marker_frame_is_right_handed_and_upright():
    c = H.marker_corners_stl([0, 0, 0], [-1, 0, 0], [0, 0, 1], 0.03)
    assert c[0][2] > 0 and c[3][2] < 0            # TL 위 · BL 아래
    assert c[0][1] > c[1][1]                      # −x 에서 보면 오른쪽이 −y
    with pytest.raises(ValueError):
        H.marker_corners_stl([0, 0, 0], [-1, 0, 0], [-1, 0, 1], 0.03)


TRUE = {0: (0.33, 0.12, 0.235, math.radians(4.0)),
        1: (0.30, 0.0, 0.235, math.radians(-3.0)),
        2: (0.36, -0.13, 0.235, math.radians(8.0))}


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_synthetic_scene_recovers_each_holder_and_marker_rotation(k):
    rgb = _render(LOW, TRUE, k=k, noise=2.0)
    det = H.detect_markers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    assert set(det) == {0, 1, 2}
    for i, p in TRUE.items():
        f = H.fit_one(det[i], K, LOW, CFG)
        assert f.k == k
        assert np.allclose(f.pose[:3], p[:3], atol=0.004), (i, f.pose, p)
        assert abs(math.degrees(H.wrap(f.pose[3] - p[3]))) < 2.0
        assert f.rms_px < 1.0


def test_foreshortened_marker_is_found_at_every_nearby_pose_and_rotation():
    """기본 minMarkerDistanceRate(0.05)에서는 이 60장 중 23장을 놓쳤다."""
    missed = []
    for yaw in (-6, -3, 0, 3, 6):
        for x in (0.28, 0.30, 0.32):
            for k in range(4):
                rgb = _render(LOW, {1: (x, 0.0, 0.235, math.radians(yaw))}, k=k)
                if 1 not in H.detect_markers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)):
                    missed.append((yaw, x, k))
    assert missed == []


def test_shared_axis_takes_one_value_and_keeps_fit():
    same_y = {i: (p[0], 0.05, p[2], p[3]) for i, p in TRUE.items()}
    rgb = _render(LOW, same_y, noise=2.0, seed=3)
    det = H.detect_markers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    free = {i: H.fit_one(det[i], K, LOW, CFG) for i in det}
    out = H.fit_shared(det, free, K, LOW, CFG, ("y",))
    ys = [p[1] for p in out.values()]
    assert np.ptp(ys) < 1e-9
    assert abs(ys[0] - 0.05) < 0.003
    for i, p in out.items():
        assert H.rms_of(p, det[i], K, LOW, CFG, free[i].k) < 1.0


def test_shared_yaw_is_averaged_on_the_circle():
    """yaw 공유 초기값이 ±180° 근처에서 0 으로 무너지지 않는다."""
    flip = {0: (0.30, 0.10, 0.235, math.radians(179.0)), 1: (0.30, -0.10, 0.235, math.radians(-179.0))}
    T_bc = _look_at(np.array([0.65, 0.0, 0.40]), np.array([0.30, 0.0, 0.21]))   # +x 쪽에서 본다
    det = H.detect_markers(cv2.cvtColor(_render(T_bc, flip), cv2.COLOR_RGB2GRAY))
    free = {i: H.fit_one(det[i], K, T_bc, CFG) for i in det}
    out = H.fit_shared(det, free, K, T_bc, CFG, ("yaw",))
    for p in out.values():
        assert abs(abs(math.degrees(p[3])) - 180.0) < 2.0


def test_feature_edges_outline_is_small_and_on_the_part():
    tris = H.load_stl(CFG.stl)
    e = H.feature_edges(tris)
    assert 50 < len(e) < len(tris) * 3
    P = tris.reshape(-1, 3)
    assert (e.reshape(-1, 3) >= P.min(0) - 1e-6).all() and (e.reshape(-1, 3) <= P.max(0) + 1e-6).all()


def test_arm4090_head_view_sees_the_marker_strongly_foreshortened():
    """기록: head_home 머리(아래로 ~71°)에서는 −x 수직면 마커가 세로로 크게 눌려 보인다 — 검출 한계 점검용."""
    T = H.T_planar(0.30, 0.0, 0.235, 0.0)
    uv, _ = H.project(ARM4090, K, CFG.corners_stl @ T[:3, :3].T + T[:3, 3])
    w = np.linalg.norm(uv[0] - uv[1])
    h = np.linalg.norm(uv[0] - uv[3])
    assert w > 30 and h < w * 0.6


# ── 대체 검출(marker_template_fit) — 10.02 arm4090 실물 조건 ──────────────────────
REAL = {0: (0.392, 0.083, 0.235, math.radians(2.6)),
        1: (0.392, -0.034, 0.235, math.radians(1.2)),
        2: (0.394, -0.153, 0.235, math.radians(-0.8))}
#: 붙은 회전도 실물과 같게(0: 180° · 1: 90° · 2: 0°) — id 마다 따로 그린다
REAL_K = {0: 2, 1: 1, 2: 0}


def _real_like(seed: int = 0) -> np.ndarray:
    """머리 시점 · 마커 테두리가 검은 몸체(30)와 붙고 흰 여백 없음 · 흰 비트 170 · 잡음 — ArUco 가 못 찾는 장면."""
    g = None
    for i, p in REAL.items():
        one = cv2.cvtColor(_render(ARM4090, {i: p}, k=REAL_K[i], pad=0, bg=30, white=170), cv2.COLOR_RGB2GRAY)
        g = one if g is None else np.maximum(g, one)
    g = cv2.GaussianBlur(g, (0, 0), 0.7)
    g = np.clip(g + np.random.default_rng(seed).normal(0, 3.0, g.shape), 0, 255).astype(np.uint8)
    return g


def test_real_like_scene_defeats_aruco():
    assert H.detect_markers(_real_like()) == {}


def test_template_fallback_finds_every_holder_id_rotation_and_position():
    g = _real_like()
    hits = H.template_fallback(g, K, ARM4090, CFG, [0, 1, 2], 0.205)
    assert set(hits) == {0, 1, 2}
    for i, p in REAL.items():
        h = hits[i]
        assert h.k == REAL_K[i]
        # 22 mm 마커(10.03)는 30 mm 보다 화면에서 작아 점수가 0.78~0.81 — 위치 · 각도 정확도는 아래에서 그대로 본다
        assert h.ncc > 0.75 and h.runner_up < h.ncc - 0.15
        assert np.allclose(h.pose[:2], p[:2], atol=0.003), (i, h.pose, p)
        assert abs(math.degrees(H.wrap(h.pose[3] - p[3]))) < 3.0
        assert h.pose[2] == pytest.approx(0.235, abs=1e-9)        # 바닥이 상판(0.205) 위 · 원점은 +30 mm


def test_template_fallback_ignores_ids_that_are_not_in_the_scene():
    g = cv2.cvtColor(_render(ARM4090, {1: REAL[1]}, k=1, pad=0, bg=30, white=170), cv2.COLOR_RGB2GRAY)
    hits = H.template_fallback(g, K, ARM4090, CFG, [0, 1, 2], 0.205)
    assert set(hits) == {1}


def test_blank_table_finds_nothing():
    g = np.clip(30 + np.random.default_rng(1).normal(0, 3.0, (HH, W)), 0, 255).astype(np.uint8)
    assert H.template_fallback(g, K, ARM4090, CFG, [0, 1, 2], 0.205) == {}


# ── 연속 추정(cup_holder_tracker) · 노드 순수부 ─────────────────────────────────────
import cup_holder_tracker as TR  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "nodes"))
import cup_holder_pose_node as NODE  # noqa: E402


def _scene(poses: dict, seed: int = 0) -> np.ndarray:
    g = None
    for i, p in poses.items():
        one = cv2.cvtColor(_render(ARM4090, {i: p}, k=REAL_K[i], pad=0, bg=30, white=170), cv2.COLOR_RGB2GRAY)
        g = one if g is None else np.maximum(g, one)
    if g is None:
        g = np.full((HH, W), 30, np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 0.7)
    return np.clip(g + np.random.default_rng(seed).normal(0, 3.0, g.shape), 0, 255).astype(np.uint8)


def _same_x(poses: dict) -> dict:
    return {i: (0.392, p[1], p[2], p[3]) for i, p in poses.items()}


def _tracker(**kw) -> TR.HolderTracker:
    return TR.HolderTracker(CFG, ARM4090, table_z=0.205, **kw)


def test_tracker_searches_once_then_tracks_and_becomes_stable():
    tr, truth = _tracker(), _same_x(REAL)
    for n in range(5):
        tr.update(_scene(truth, seed=n), K)
        srcs = {h["src"] for h in tr.status()["holders"].values()}
        assert srcs == ({"template"} if n == 0 else {"track"})
    st = tr.status()
    assert st["ok"] and st["error"] == ""
    poses = tr.poses()
    assert set(poses) == {"cup_holder_0", "cup_holder_1", "cup_holder_2"}
    for (name, p), (i, q) in zip(sorted(poses.items()), sorted(truth.items())):
        assert np.allclose(p[:2], q[:2], atol=0.003), (name, p, q)
    assert np.ptp([p[0] for p in poses.values()]) < 1e-9          # 공유 축 x


def test_tracker_follows_a_holder_that_moves_a_little():
    tr, truth = _tracker(), _same_x(REAL)
    tr.update(_scene(truth), K)
    moved = dict(truth)
    moved[1] = (0.392, truth[1][1] + 0.006, truth[1][2], truth[1][3] + math.radians(4))
    for n in range(5):
        tr.update(_scene(moved, seed=10 + n), K)
    assert tr.states[1].src == "track"
    assert abs(tr.poses()["cup_holder_1"][1] - moved[1][1]) < 0.003


def test_tracker_drops_a_holder_that_disappears_and_reports_not_ok():
    tr, truth = _tracker(lost_after=3), _same_x(REAL)
    for n in range(5):
        tr.update(_scene(truth, seed=n), K)
    gone = {i: p for i, p in truth.items() if i != 2}
    for n in range(3):
        tr.update(_scene(gone, seed=20 + n), K)
    assert "cup_holder_2" not in tr.poses()
    st = tr.status()
    assert not st["ok"] and st["holders"]["cup_holder_2"]["seen"] is False
    assert st["holders"]["cup_holder_0"]["stable"]


def test_tracker_discards_a_frame_that_breaks_the_shared_axis():
    tr = _tracker()
    bad = _same_x(REAL)
    bad[2] = (0.45, bad[2][1], bad[2][2], bad[2][3])               # 한 홀더만 6 cm 앞 — x 공유가 깨진다
    tr.update(_scene(bad), K)
    st = tr.status()
    assert "공유 제약" in st["error"] and tr.poses() == {}
    assert all(h["misses"] == 1 for h in st["holders"].values())


def test_node_converts_ros_image_encodings_to_gray():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 200                                              # 빨강
    g_rgb = NODE.image_to_gray("rgb8", 2, 3, rgb.tobytes())
    g_bgr = NODE.image_to_gray("bgr8", 2, 3, rgb.tobytes())
    # 같은 바이트(첫 채널 200)를 rgb8 은 빨강(가중 0.299), bgr8 은 파랑(0.114)으로 읽는다
    assert g_rgb.shape == (2, 3) and g_rgb[0, 0] == 60 and g_bgr[0, 0] == 23
    assert NODE.image_to_gray("mono8", 2, 3, bytes(6)).shape == (2, 3)
    with pytest.raises(ValueError):
        NODE.image_to_gray("16UC1", 2, 3, bytes(12))


def test_node_rewrites_only_when_the_holders_moved():
    a = {"cup_holder_0": np.array([0.39, 0.08, 0.235, 0.0])}
    assert NODE.should_rewrite(None, a)
    assert not NODE.should_rewrite(a, {"cup_holder_0": a["cup_holder_0"] + [0.001, 0.001, 0, math.radians(1)]})
    assert NODE.should_rewrite(a, {"cup_holder_0": a["cup_holder_0"] + [0.004, 0, 0, 0]})
    assert NODE.should_rewrite(a, {"cup_holder_0": a["cup_holder_0"] + [0, 0, 0, math.radians(3)]})
    assert NODE.should_rewrite(a, {**a, "cup_holder_1": a["cup_holder_0"]})

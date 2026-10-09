#!/usr/bin/env python3
"""같은 모양 · 다른 색 물체 여럿을 FP++ 컨테이너 하나에서 — 색 판정 · 검출 배정 · 한 번 찍기 대표 자세(순수).

10.08 사용자: "컨테이너 하나에, 색깔이 다른 동일 모델을 추출" — 파랑 · 핑크 · 노랑 cyl60. YOLO 는 셋 다 cup(41)으로
잡을 뿐 구분하지 못한다 → 마스크 안 화소의 색상(HSV hue) 비율로 어느 물체인지 가른다. fpp_snapshot_node 가 쓴다.

색상 범위는 OpenCV 눈금(hue 0~180). 10.08~10.09 arm4090 머리 카메라 실측: 파랑 99~105 · 노랑 23~28 · 핑크 155~160 · 주황 11~16 ·
테이블(검정) 채도 21. 채도 · 명도가 낮은 화소(테이블 · 그림자 · 컵 안 어두운 곳)는 어느 색에도 넣지 않는다.
"""
from __future__ import annotations

import numpy as np

#: 색 이름 → hue 구간들(OpenCV 0~180, 양 끝 포함)
HUE_RANGES: dict[str, tuple[tuple[float, float], ...]] = {
    "orange": ((5.0, 19.0),),          # 10.09 source240 주황 병 11~16
    "yellow": ((20.0, 40.0),),         # cyl60 노랑 23~28
    "blue": ((85.0, 130.0),),
    "pink": ((148.0, 175.0),),         # cyl60 핑크 158 · 병 155~160 — 누운 보라 병 144 는 뺀다
    "red": ((0.0, 4.0), (176.0, 180.0)),
}
COLORS = tuple(HUE_RANGES)
S_MIN = 80.0       # 채도(0~255) — 테이블 21
V_MIN = 50.0       # 명도(0~255)
MIN_SCORE = 0.25   # 마스크의 이 비율 넘게 그 색이어야 그 물체로 본다


def hue_ranges(color) -> tuple[tuple[float, float], ...]:
    """색 이름(HUE_RANGES) 또는 잰 구간 [lo, hi](fpp_object calib) → hue 구간들. lo > hi 면 0/180 을 넘는 구간(빨강 부근)."""
    if isinstance(color, str):
        if color not in HUE_RANGES:
            raise ValueError(f"색 {color!r} 을 모른다 — {COLORS} 또는 [lo, hi]")
        return HUE_RANGES[color]
    vals = [float(v) for v in color]
    if len(vals) != 2 or not all(0.0 <= v <= 180.0 for v in vals):
        raise ValueError(f"hue 구간 {color!r} — [lo, hi] (0~180)")
    lo, hi = vals
    return ((lo, hi),) if lo <= hi else ((lo, 180.0), (0.0, hi))


def unique_colors(colors) -> list:
    """색(이름 또는 [lo, hi]) 목록에서 순서를 지키며 중복을 뺀다 — list 는 set 에 못 넣는다."""
    out, seen = [], set()
    for c in colors:
        key = c if isinstance(c, str) else tuple(float(v) for v in c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _hsv(px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N×3 RGB(uint8) → hue(0~180) · 채도(0~255) · 명도(0~255). OpenCV COLOR_RGB2HSV 와 같은 정의."""
    p = px.astype(np.float64)
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    v = p.max(axis=1)
    mn = p.min(axis=1)
    d = v - mn
    s = np.where(v > 0, 255.0 * d / np.maximum(v, 1e-9), 0.0)
    dd = np.maximum(d, 1e-9)
    h = np.where(v == r, (g - b) / dd, np.where(v == g, 2.0 + (b - r) / dd, 4.0 + (r - g) / dd))
    h = np.where(d > 0, (h * 30.0) % 180.0, 0.0)
    return h, s, v


def color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    """영상 전체에서 그 색인 화소(H×W bool)."""
    ranges = hue_ranges(color)
    img = np.asarray(rgb)
    h, s, v = _hsv(img.reshape(-1, 3))
    out = np.zeros(h.shape, bool)
    for lo, hi in ranges:
        out |= (h >= lo) & (h <= hi)
    return (out & (s >= S_MIN) & (v >= V_MIN)).reshape(img.shape[:2])


BLOB_CLOSE_PX = 3   # 닫기 반경 — PLA 결 · 그늘로 끊긴 몸통을 잇는다(10.09 핑크 병)


def color_blobs(rgb: np.ndarray, color: str, min_area: int = 1500, close_px: int = BLOB_CLOSE_PX) -> list[np.ndarray]:
    """그 색의 이어진 덩어리 마스크들(넓이 min_area 화소 이상, 큰 것부터) — YOLO 가 못 잡은 물체의 후보.
    닫기(팽창 → 침식)와 구멍 메우기로 끊긴 결을 이은 뒤 센다."""
    from scipy import ndimage
    m = color_mask(rgb, color)
    if close_px > 0:
        st = np.ones((2 * close_px + 1, 2 * close_px + 1), bool)
        m = ndimage.binary_closing(np.pad(m, close_px), structure=st)[close_px:-close_px, close_px:-close_px]
        m = ndimage.binary_fill_holes(m)
    labels, n = ndimage.label(m)
    if n == 0:
        return []
    areas = ndimage.sum(np.ones(labels.shape), labels, index=np.arange(1, n + 1))
    keep = [i + 1 for i in np.argsort(-areas) if areas[i] >= min_area]
    return [labels == k for k in keep]


def mask_point(mask: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """마스크 화소의 깊이 중앙값 · 화소 중심을 카메라 좌표 점으로. 유효 깊이가 없으면 None."""
    m = np.asarray(mask, bool) & (np.asarray(depth) > 0.05) & np.isfinite(depth)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    z = float(np.median(np.asarray(depth)[m]))
    u, v = float(np.median(xs)), float(np.median(ys))
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def mask_points_base(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                     step: int = 3) -> np.ndarray:
    """마스크의 유효 깊이 화소(step 칸마다)를 base 점들(N×3)로."""
    m = np.asarray(mask, bool) & (np.asarray(depth) > 0.05) & np.isfinite(depth)
    ys, xs = np.nonzero(m)
    ys, xs = ys[::step], xs[::step]
    z = np.asarray(depth)[ys, xs]
    pc = np.column_stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z])
    T = np.asarray(T_base_cam, float)
    return pc @ T[:3, :3].T + T[:3, 3]


def split_by_depth(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                   voxel: float = 0.01, min_px: int = 1500) -> list[np.ndarray]:
    """영상에서 붙어 보이는 물체들을 3D 로 나눈다 — 마스크 화소를 base 점으로 옮겨 voxel 칸에 넣고, 칸들의 이어진
    조각(26 이웃)마다 화소 마스크 하나(min_px 이상, 큰 것부터). 깊이 없는 화소는 버린다."""
    from scipy import ndimage
    m = np.asarray(mask, bool) & (np.asarray(depth) > 0.05) & np.isfinite(depth)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return []
    z = np.asarray(depth)[ys, xs]
    pc = np.column_stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z])
    T = np.asarray(T_base_cam, float)
    P = pc @ T[:3, :3].T + T[:3, 3]
    idx = np.floor((P - P.min(axis=0)) / voxel).astype(int)
    grid = np.zeros(idx.max(axis=0) + 1, bool)
    grid[tuple(idx.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    px_lab = lab[tuple(idx.T)]
    out = []
    for k in range(1, n + 1):
        sel = px_lab == k
        if sel.sum() >= min_px:
            part = np.zeros(m.shape, bool)
            part[ys[sel], xs[sel]] = True
            out.append(part)
    return sorted(out, key=lambda q: -int(q.sum()))


FOOTPRINT_SCALE, FOOTPRINT_PAD = 1.5, 0.02


def fits_footprint(points_base: np.ndarray, aabb, scale: float = FOOTPRINT_SCALE, pad: float = FOOTPRINT_PAD) -> bool:
    """보이는 면의 수평 퍼짐(p5~p95)이 서 있는 물체의 바닥 크기 안인가 — 누운 · 쓰러진 같은 색 물체를 버린다(10.09 핑크 병).
    한계 = 물체 aabb 의 수평 최대 폭 × scale + pad. 점이 20 개 미만이면 False."""
    P = np.asarray(points_base, float)
    if len(P) < 20:
        return False
    lo, hi = np.asarray(aabb[0], float), np.asarray(aabb[1], float)
    limit = float(np.max(hi[:2] - lo[:2])) * scale + pad
    c = np.median(P[:, :2], axis=0)
    r = np.linalg.norm(P[:, :2] - c, axis=1)
    spread = 2.0 * float(np.percentile(r, 95))
    return spread <= limit


TABLE_TOP_Z = 0.205      # arm4090 상판(scripts/calib/table_cad_extrinsics.py 와 같은 값)
TOP_TOL = 0.04           # 서 있는 물체 꼭대기 높이 허용 — 10.09 실측: 서 있음 +0.002 · +0.006, 누움 −0.055 · −0.070


def looks_standing(points_base: np.ndarray, aabb, table_z: float = TABLE_TOP_Z, top_tol: float = TOP_TOL) -> bool:
    """서 있는 물체 같은가 — 수평 퍼짐이 바닥 안(fits_footprint)이고, 보이는 점 위쪽(p95)이 상판 + 물체 높이에 닿는다.
    화면 끝에 잘린 누운 같은 색 물체는 폭으로는 못 거르고 꼭대기 높이로 거른다(10.09)."""
    if not fits_footprint(points_base, aabb):
        return False
    lo, hi = np.asarray(aabb[0], float), np.asarray(aabb[1], float)
    top = float(np.percentile(np.asarray(points_base, float)[:, 2], 95))
    return abs(top - (table_z + float(hi[2] - lo[2]))) <= top_tol


def in_workspace(T_base_cam: np.ndarray, p_cam: np.ndarray | None, ws: dict) -> bool:
    """카메라 점을 base 로 바꿔 작업 영역 상자({x,y,z: [lo, hi]}) 안인가. 점이 없으면 False."""
    if p_cam is None:
        return False
    p = np.asarray(T_base_cam, float)[:3, :3] @ np.asarray(p_cam, float) + np.asarray(T_base_cam, float)[:3, 3]
    return all(ws[k][0] <= p[i] <= ws[k][1] for i, k in enumerate("xyz"))


def color_fraction(rgb: np.ndarray, mask: np.ndarray, color: str) -> float:
    """마스크 화소 중 그 색(채도 · 명도 문턱을 넘고 hue 가 구간 안)인 비율. 빈 마스크는 0."""
    ranges = hue_ranges(color)
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if n == 0:
        return 0.0
    h, s, v = _hsv(np.asarray(rgb)[m].reshape(-1, 3))
    vivid = (s >= S_MIN) & (v >= V_MIN)
    inside = np.zeros(h.shape, bool)
    for lo, hi in ranges:
        inside |= (h >= lo) & (h <= hi)
    return float((vivid & inside).sum()) / n


def assign(rgb: np.ndarray, masks: list[np.ndarray], wanted: dict, min_score: float = MIN_SCORE,
           allowed: dict[str, set[int]] | None = None) -> dict[str, int]:
    """물체 이름 → 검출 번호. 색 비율이 높은 짝부터 하나씩 — 한 검출은 한 물체에만, 문턱 못 넘는 물체는 뺀다.
    allowed[이름] 이 있으면 그 후보들만(서 있는 모양이 맞는 것 — fits_footprint)."""
    pairs = sorted(((color_fraction(rgb, m, color), name, i)
                    for name, color in wanted.items() for i, m in enumerate(masks)
                    if allowed is None or name not in allowed or i in allowed[name]), reverse=True)
    out: dict[str, int] = {}
    used: set[int] = set()
    for score, name, i in pairs:
        if score < min_score or name in out or i in used:
            continue
        out[name] = i
        used.add(i)
    return out


def scores(rgb: np.ndarray, masks: list[np.ndarray]) -> list[dict[str, float]]:
    """검출마다 색별 비율(상태 보고용)."""
    return [{c: round(color_fraction(rgb, m, c), 3) for c in COLORS} for m in masks]


def representative(poses: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """여러 장의 4×4 자세 → (위치 중앙값에 가장 가까운 한 장, 그 중앙값에서 가장 먼 장까지 mm)."""
    if not poses:
        raise ValueError("자세가 없다")
    P = np.stack([np.asarray(T, float) for T in poses])
    t = P[:, :3, 3]
    med = np.median(t, axis=0)
    d = np.linalg.norm(t - med, axis=1)
    return P[int(np.argmin(d))].copy(), float(d.max() * 1e3)

"""ArUco 대체 검출 — 마커 무늬를 영상에 직접 맞춘다 (scripts/calib/cup_holder_pose.py 보조).

10.02 arm4090 실물: 마커 검은 테두리(밝기 ~30)가 검은 홀더 몸체(~30)와 붙고, 흰 여백 1 mm 는
영상에서 1 px 미만이라 사각 윤곽이 생기지 않는다 → OpenCV 검출 0개. 머리 시점에서 마커가 세로로
눌려 칸 하나가 가로 ~5 px · 세로 ~2.3 px 다.

방법: 윤곽 대신 무늬 자체를 쓴다.
  ① 흰 비트 덩어리(top-hat 밝은 점)를 찾고, 그 중심 광선을 마커 중심 높이 평면과 만나게 해 초기 위치를 잡는다.
  ② 홀더는 서 있고(upright) 바닥이 테이블 위라 z 는 고정한다 — 이 시점에서 z 와 앞뒤(x) 는 영상에서
     거의 구분되지 않는다.
  ③ id · 붙은 회전 k(×90°) · 마커 중심 x,y · yaw 를 격자로 훑어, 8×8 칸(테두리 포함) 기대 밝기와
     영상 표본의 정규 상관(NCC)이 가장 큰 것을 고르고 Nelder-Mead 로 다듬는다.
  ④ 그 자세에서 마커 네 모서리를 투영해 ArUco 결과와 같은 꼴(TL,TR,BR,BL px)로 돌려준다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CELLS = 8                    # 6×6 비트 + 테두리 1칸씩
SUB = 4                      # 칸당 표본 SUB×SUB (다듬기)
SUB_COARSE = 2               # 격자 단계 — 속도 4배
NCC_MIN = 0.55               # 이보다 낮으면 그 id 를 못 찾은 것으로 본다
COARSE_MIN = 0.35            # 격자 단계 최고 NCC 가 이보다 낮은 덩어리는 다듬지 않는다(잡광·컵 가장자리)
REFINE_TOP = 3               # 덩어리마다 다듬을 (id, k) 후보 수
TOPHAT_PX = 21               # 흰 비트보다 크고 마커보다 작은 창
TOPHAT_MIN = 50.0            # 흰 비트가 주변보다 이만큼은 밝아야 한다(8bit)
BLOB_AREA = (12, 6000)       # 덩어리 넓이(px) 범위
GRID_MM = (-9.0, -4.5, 0.0, 4.5, 9.0)
GRID_YAW_DEG = 10.0
TRACK_GRID_MM = (-6.0, -3.0, 0.0, 3.0, 6.0)      # 추적: 직전 마커 중심 둘레
TRACK_GRID_DEG = (-8.0, -4.0, 0.0, 4.0, 8.0)


@dataclass(frozen=True)
class TemplateHit:
    corners_px: np.ndarray   # (4,2) 마커 TL,TR,BR,BL — ArUco 와 같은 순서
    pose: np.ndarray         # (4,) 홀더 원점 x,y,z,yaw (z 는 고정값)
    k: int                   # 마커가 법선 둘레로 k·90° 돌아 붙은 것
    ncc: float
    runner_up: float         # 같은 덩어리에서 다른 id/k 의 최고 NCC (모호성). 추적 결과는 -1(따지지 않음)
    center_xy: np.ndarray    # (2,) 마커 중심 base x,y — 추적의 시작점


def marker_cells(dictionary: str, marker_id: int) -> np.ndarray:
    """(8,8) 0/1 — 행 0 이 인쇄 위쪽, 열 0 이 왼쪽."""
    import cv2
    ar = cv2.aruco
    get = ar.getPredefinedDictionary if hasattr(ar, "getPredefinedDictionary") else ar.Dictionary_get
    d = get(getattr(ar, dictionary))
    draw = ar.generateImageMarker if hasattr(ar, "generateImageMarker") else ar.drawMarker
    return (np.asarray(draw(d, int(marker_id), CELLS)) > 127).astype(float)


def _unit_samples(sub: int = SUB) -> tuple[np.ndarray, np.ndarray]:
    """마커 단위 좌표 (u 오른쪽, v 아래) 표본과 그 칸 (행, 열)."""
    f = (np.arange(CELLS * sub) + 0.5) / (CELLS * sub)
    u, v = np.meshgrid(f, f)
    rc = np.c_[(v.ravel() * CELLS).astype(int), (u.ravel() * CELLS).astype(int)]
    return np.c_[u.ravel(), v.ravel()], rc


UV, RC = _unit_samples()
UV_C, RC_C = _unit_samples(SUB_COARSE)


def face_points(corners: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """마커 TL,TR,BR,BL(STL, m) 위의 단위 좌표 → STL 점."""
    tl, tr, _, bl = corners
    return tl + uv[:, :1] * (tr - tl) + uv[:, 1:] * (bl - tl)


def _rz(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _origin_from_center(cxy: np.ndarray, yaw: np.ndarray, ctr_stl: np.ndarray) -> np.ndarray:
    """마커 중심 base x,y → 홀더 원점 x,y."""
    R = _rz(yaw)[..., :2, :2]
    return cxy - (R @ ctr_stl[:2])


def _project(T_cam_base: np.ndarray, K: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    z = Xc[..., 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    uv = np.stack([K[0, 0] * Xc[..., 0] / zs + K[0, 2], K[1, 1] * Xc[..., 1] / zs + K[1, 2]], -1)
    return uv, z


def _sample(img: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """쌍선형 표본 (…,) 과 영상 안 여부."""
    h, w = img.shape
    x, y = uv[..., 0], uv[..., 1]
    ok = (x >= 0) & (y >= 0) & (x <= w - 1.001) & (y <= h - 1.001)
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return v, ok


class Scorer:
    """자세 묶음 → NCC. 홀더 z 고정 · upright."""

    def __init__(self, img: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, corners_stl: np.ndarray,
                 z_origin: float):
        self.img = np.asarray(img, float)
        self.K = K
        self.T_bc = T_base_cam
        self.T_cb = np.linalg.inv(T_base_cam)
        self.ctr = corners_stl.mean(0)
        self.corners = corners_stl
        self.z = float(z_origin)
        self.cam = T_base_cam[:3, 3]

    def points(self, k: int, uv: np.ndarray = UV) -> np.ndarray:
        from cup_holder_pose import rolled
        return face_points(rolled(self.corners, k), uv)

    def world(self, cxy: np.ndarray, yaw: np.ndarray, P: np.ndarray) -> np.ndarray:
        """(M,2),(M,) → (M,N,3) base 점."""
        oxy = _origin_from_center(cxy, yaw, self.ctr)
        R = _rz(yaw)
        X = np.einsum("mij,nj->mni", R, P)
        X[..., :2] += oxy[:, None, :]
        X[..., 2] += self.z
        return X

    def facing(self, cxy: np.ndarray, yaw: np.ndarray, normal_stl: np.ndarray) -> np.ndarray:
        n = np.einsum("mij,j->mi", _rz(yaw), normal_stl)
        c = np.c_[cxy, np.full(len(cxy), self.z + self.ctr[2])]
        return np.einsum("mi,mi->m", n, self.cam - c) > 0

    def ncc(self, cxy: np.ndarray, yaw: np.ndarray, P: np.ndarray, expect: np.ndarray) -> np.ndarray:
        uv, z = _project(self.T_cb, self.K, self.world(cxy, yaw, P))
        v, ok = _sample(self.img, uv)
        ok &= z > 0
        e = expect - expect.mean()
        vc = v - v.mean(1, keepdims=True)
        num = (vc * e).sum(1)
        den = np.sqrt((vc ** 2).sum(1) * (e ** 2).sum()) + 1e-9
        return np.where(ok.all(1), num / den, -1.0)

    def corners_px(self, cxy: np.ndarray, yaw: float, k: int) -> np.ndarray:
        from cup_holder_pose import rolled
        X = self.world(np.asarray(cxy, float)[None], np.array([yaw]), rolled(self.corners, k))[0]
        return _project(self.T_cb, self.K, X)[0]


def bright_blobs(gray: np.ndarray) -> list[np.ndarray]:
    """흰 비트 무리 중심(px) 들 — 작은 밝은 점이 모인 곳."""
    import cv2
    g = np.asarray(gray, np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_PX, TOPHAT_PX))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ker)
    m = (th > TOPHAT_MIN).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        if BLOB_AREA[0] <= stats[i, cv2.CC_STAT_AREA] <= BLOB_AREA[1]:
            out.append(cent[i])
    return out


def ray_plane(px: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, z: float) -> np.ndarray | None:
    d_cam = np.linalg.solve(K, np.r_[px, 1.0])
    d = T_base_cam[:3, :3] @ d_cam
    o = T_base_cam[:3, 3]
    if abs(d[2]) < 1e-9:
        return None
    t = (z - o[2]) / d[2]
    return o + t * d if t > 0 else None


class _Ctx:
    """한 장의 영상 · 카메라 · 마커 형상에 대한 NCC 계산 준비물."""

    def __init__(self, gray, K, T_base_cam, corners_stl, dictionary: str, ids, z_origin: float):
        import cv2
        g = np.asarray(gray, float)
        self.coarse = Scorer(cv2.GaussianBlur(g, (0, 0), 1.6), K, T_base_cam, corners_stl, z_origin)
        self.fine = Scorer(cv2.GaussianBlur(g, (0, 0), 0.8), K, T_base_cam, corners_stl, z_origin)
        self.z_origin = float(z_origin)
        cells = {i: marker_cells(dictionary, i) for i in ids}
        self.expect = {i: c[RC[:, 0], RC[:, 1]] for i, c in cells.items()}
        self.expect_c = {i: c[RC_C[:, 0], RC_C[:, 1]] for i, c in cells.items()}
        self.P = {k: self.coarse.points(k) for k in range(4)}
        self.P_c = {k: self.coarse.points(k, UV_C) for k in range(4)}

    def refine(self, i: int, k: int, cxy: np.ndarray, yaw: float, step_mm: float = 3.0,
               step_deg: float = 5.0) -> tuple[float, np.ndarray, float]:
        """Nelder-Mead 두 단계(흐린 → 덜 흐린). 반환 (NCC, 마커 중심 xy, yaw)."""
        from scipy.optimize import minimize

        def f(p, sc):
            return -float(sc.ncc(p[None, :2] * 1e-3, np.radians(p[2:3]), self.P[k], self.expect[i])[0])

        p0 = np.r_[np.asarray(cxy, float) * 1e3, math.degrees(yaw)]
        r = minimize(f, p0, args=(self.coarse,), method="Nelder-Mead",
                     options={"initial_simplex": p0 + np.vstack([np.zeros(3), np.diag([step_mm, step_mm, step_deg])]),
                              "xatol": 0.05, "fatol": 1e-5, "maxiter": 400})
        r = minimize(f, r.x, args=(self.fine,), method="Nelder-Mead",
                     options={"initial_simplex": r.x + np.vstack([np.zeros(3), np.diag([1.0, 1.0, 2.0])]),
                              "xatol": 0.02, "fatol": 1e-6, "maxiter": 400})
        return -float(r.fun), r.x[:2] * 1e-3, math.radians(r.x[2])

    def hit(self, i: int, k: int, s: float, cxy: np.ndarray, yaw: float, runner_up: float) -> TemplateHit:
        o = _origin_from_center(np.asarray(cxy)[None], np.array([yaw]), self.fine.ctr)[0]
        return TemplateHit(corners_px=self.fine.corners_px(cxy, yaw, k), pose=np.r_[o, self.z_origin, yaw],
                           k=k, ncc=float(s), runner_up=float(runner_up), center_xy=np.asarray(cxy, float).copy())


def fit_templates(gray: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, corners_stl: np.ndarray,
                  normal_stl: np.ndarray, dictionary: str, ids: list[int], z_origin: float,
                  roi_xy: tuple[tuple[float, float], tuple[float, float]] | None = None,
                  ) -> dict[int, TemplateHit]:
    """전체 탐색: id → TemplateHit. 못 찾은 id 는 빠진다. 덩어리 하나에는 id 하나만."""
    ctx = _Ctx(gray, K, T_base_cam, corners_stl, dictionary, ids, z_origin)
    zc = z_origin + float(ctx.coarse.ctr[2])
    yaws = np.radians(np.arange(-180.0, 180.0, GRID_YAW_DEG))
    off = np.array([(a, b) for a in GRID_MM for b in GRID_MM]) * 1e-3

    by_blob: dict[int, list] = {}      # 덩어리 → [(ncc, id, k, cxy, yaw)]
    for b, px in enumerate(bright_blobs(gray)):
        c0 = ray_plane(np.asarray(px, float), K, T_base_cam, zc)
        if c0 is None:
            continue
        if roi_xy is not None and not (roi_xy[0][0] <= c0[0] <= roi_xy[0][1] and roi_xy[1][0] <= c0[1] <= roi_xy[1][1]):
            continue
        cxy = np.repeat(c0[None, :2] + off, len(yaws), 0)
        yw = np.tile(yaws, len(off))
        face = ctx.coarse.facing(cxy, yw, normal_stl)
        cxy, yw = cxy[face], yw[face]
        if not len(yw):
            continue
        for i in ids:
            for k in range(4):
                sc = ctx.coarse.ncc(cxy, yw, ctx.P_c[k], ctx.expect_c[i])
                j = int(np.argmax(sc))
                by_blob.setdefault(b, []).append((float(sc[j]), i, k, cxy[j].copy(), float(yw[j])))

    refined = []        # (ncc, blob, id, k, cxy, yaw, runner_up)
    for b, cs in by_blob.items():
        cs.sort(key=lambda c: -c[0])
        if cs[0][0] < COARSE_MIN:
            continue
        top = sorted(((*ctx.refine(i, k, cxy, yw), i, k) for _, i, k, cxy, yw in cs[:REFINE_TOP]),
                     key=lambda t: -t[0])
        s, cxy, yw, i, k = top[0]
        others = [t[0] for t in top[1:] if (t[3], t[4]) != (i, k)]
        refined.append((s, b, i, k, cxy, yw, max(others) if others else -1.0))

    out: dict[int, TemplateHit] = {}
    used = set()
    for s, b, i, k, cxy, yw, ru in sorted(refined, key=lambda t: -t[0]):
        if s < NCC_MIN or i in out or b in used:
            continue
        used.add(b)
        out[i] = ctx.hit(i, k, s, cxy, yw, ru)
    return out


def track_templates(gray: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, corners_stl: np.ndarray,
                    dictionary: str, z_origin: float, prev: dict[int, TemplateHit]) -> dict[int, TemplateHit]:
    """추적: 직전 자세(id · 회전 k 고정)에서만 다듬는다 — 전체 탐색보다 수십 배 빠르다.
    NCC 가 NCC_MIN 밑으로 떨어진 id 는 빠진다(호출자가 전체 탐색으로 되돌아간다)."""
    if not prev:
        return {}
    ctx = _Ctx(gray, K, T_base_cam, corners_stl, dictionary, list(prev), z_origin)
    off = np.array([(a, b) for a in TRACK_GRID_MM for b in TRACK_GRID_MM]) * 1e-3
    dyaw = np.radians(TRACK_GRID_DEG)
    out: dict[int, TemplateHit] = {}
    for i, h in prev.items():
        # 직전 자세 둘레 작은 격자에서 시작점을 고른다 — NM 하나만으로는 4 mm · 3° 움직임에도 빠졌다
        cxy = np.repeat(h.center_xy[None] + off, len(dyaw), 0)
        yw = float(h.pose[3]) + np.tile(dyaw, len(off))
        sc = ctx.coarse.ncc(cxy, yw, ctx.P_c[h.k], ctx.expect_c[i])
        j = int(np.argmax(sc))
        s, cxy, yw = ctx.refine(i, h.k, cxy[j], float(yw[j]), step_mm=2.0, step_deg=3.0)
        if s >= NCC_MIN:
            out[i] = ctx.hit(i, h.k, s, cxy, yw, -1.0)
    return out

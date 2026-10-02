#!/usr/bin/env python3
"""컵홀더 자세 자동 추정 — 각 홀더 -x 면의 ArUco 마커(30 mm)로 STL 원점의 base 자세를 낸다.

  ① RGB + K 한 장(라이브: scripts/vision/grab_rgbd.py, 또는 --npz). 로봇은 움직이지 않는다.
  ② ArUco(DICT_6X6_250) 검출 → 마커 네 모서리(서브픽셀).
     못 찾은 id 는 무늬 맞춤 대체 검출(marker_template_fit)로 찾는다 — 10.02 실물은 마커 테두리가 검은 몸체와 붙어
     ArUco 가 0개였다. 대체 검출은 홀더 바닥이 상판(--table-z)에 놓였다고 보고 z 를 고정한다(--no-template 로 끈다).
  ③ 마커 → STL 고정 변환(config/cup_holders.yaml marker 블록)으로 모서리를 STL 점에 대응시킨다.
     마커가 몇 도 돌아 붙었는지(0/90/180/270°)는 재투영 잔차로 고른다.
  ④ T_base_cam(머리 카메라 외부 파라미터)은 고정, 홀더는 세워져 있다(upright: x · y · z · yaw 4 자유도)고 보고
     네 모서리 재투영 오차를 최소화한다. 먼저 홀더마다 따로(free), 다음에 shared 축을 세 홀더가 공유하게 한 번에.
  ⑤ 출력: 홀더별 STL 원점 (x, y, z) m · yaw° · 잔차 px, free 값과 공유 값의 차이, 겹친 영상(--png).

  python3 scripts/calib/cup_holder_pose.py                     # 라이브 1장 · 보고만
  python3 scripts/calib/cup_holder_pose.py --npz /tmp/rgbd.npz --png /tmp/holders.png
  python3 scripts/calib/cup_holder_pose.py --write             # config/cup_holder_poses_arm4090.yaml 에 기록

⚠ 외부 파라미터는 head_home_rh56f1 자세 한 장에서 맞춘 값이다. 머리가 그 자세가 아니면 결과도 틀린다.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
SIM2REAL = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from table_cad_extrinsics import TABLE_TOP_Z, T_from, grab_live, project  # noqa: E402

DEFAULT_CFG = SIM2REAL / "config/cup_holders.yaml"
DEFAULT_OUT = SIM2REAL / "config/cup_holder_poses_arm4090.yaml"
AXES = ("x", "y", "z", "yaw")
#: 공유 제약을 걸었을 때 이 이상 잔차가 커지면 가정(같은 값)이 실물과 다르다고 경고한다
SHARED_WARN_PX = 2.0
SHARED_FAIL_PX = 5.0     # 공유 제약 뒤 rms 가 이보다 크면 가정이 틀린 것 — 기록하지 않는다
MIN_MARKER_DISTANCE_RATE = 0.02   # OpenCV 기본 0.05 — detect_markers 주석 참고
SHARED_WARN_SPREAD = {"x": 0.01, "y": 0.01, "z": 0.01, "yaw": math.radians(5.0)}


# ── 설정 · 형상 ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HolderCfg:
    stl: Path
    stl_scale: float
    extrinsics: Path
    dictionary: str
    marker_size: float
    corners_stl: np.ndarray          # (4,3) m — ArUco 모서리 순서(TL, TR, BR, BL), 회전 0° 가정
    names: tuple[str, ...]
    ids: tuple[int, ...]
    upright: bool
    on_table: bool               # 바닥이 상판 위 — 최종 자세의 z 를 상판 높이로 고정
    shared: tuple[str, ...]


def marker_corners_stl(center, normal, up, size: float) -> np.ndarray:
    """마커 프레임(x 오른쪽 · y 위 · z 바깥)의 TL·TR·BR·BL 모서리를 STL 좌표로."""
    z = np.asarray(normal, float)
    y = np.asarray(up, float)
    z, y = z / np.linalg.norm(z), y / np.linalg.norm(y)
    if abs(float(z @ y)) > 1e-6:
        raise ValueError(f"marker normal {z} 과 up {y} 이 직교하지 않는다")
    x = np.cross(y, z)
    h = size / 2
    uv = [(-h, h), (h, h), (h, -h), (-h, -h)]
    return np.array([np.asarray(center, float) + u * x + v * y for u, v in uv])


def load_cfg(path: Path = DEFAULT_CFG) -> HolderCfg:
    raw = yaml.safe_load(Path(path).read_text())
    s = float(raw.get("stl_units_m", 1.0))
    m = raw["marker"]
    corners = marker_corners_stl(np.asarray(m["center_stl_mm"], float) * s, m["normal_stl"], m["up_stl"],
                                 float(m["size_m"]))
    shared = tuple(raw.get("constraints", {}).get("shared", []) or [])
    bad = [a for a in shared if a not in AXES]
    if bad:
        raise ValueError(f"{path}: constraints.shared {bad} — 가능한 축 {AXES}")
    hs = raw["holders"]
    ids = tuple(int(h["marker_id"]) for h in hs)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: marker_id 중복 {ids}")
    return HolderCfg(stl=SIM2REAL / raw["stl"], stl_scale=s, extrinsics=SIM2REAL / raw["camera_extrinsics"],
                     dictionary=str(m["dictionary"]), marker_size=float(m["size_m"]), corners_stl=corners,
                     names=tuple(str(h["name"]) for h in hs), ids=ids,
                     upright=bool(raw.get("constraints", {}).get("upright", True)),
                     on_table=bool(raw.get("constraints", {}).get("on_table", True)), shared=shared)


def load_stl(path: Path) -> np.ndarray:
    """(N,3,3) 삼각형. 이진 · ASCII 둘 다."""
    d = Path(path).read_bytes()
    if d[:5] == b"solid" and b"facet" in d[:400]:
        import re
        v = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", d)
        return np.array(v, float).reshape(-1, 3, 3)
    n = int(np.frombuffer(d[80:84], "<u4")[0])
    dt = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
    return np.frombuffer(d[84:84 + n * 50], dtype=dt)["v"].reshape(-1, 3, 3).astype(float)


def feature_edges(tris: np.ndarray, angle_deg: float = 30.0) -> np.ndarray:
    """두 면 법선이 angle_deg 이상 꺾이는 모서리(+ 경계) — 겹친 영상용 (M,2,3)."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    key = np.round(tris.reshape(-1, 3), 4)
    _, idx = np.unique(key, axis=0, return_inverse=True)
    idx = idx.reshape(-1, 3)
    faces: dict[tuple[int, int], list[int]] = {}
    for f, (a, b, c) in enumerate(idx):
        for e in ((a, b), (b, c), (c, a)):
            faces.setdefault(tuple(sorted(e)), []).append(f)
    cos_t = math.cos(math.radians(angle_deg))
    out = []
    verts = np.zeros((idx.max() + 1, 3))
    verts[idx.reshape(-1)] = tris.reshape(-1, 3)
    for (a, b), fs in faces.items():
        if len(fs) != 2 or float(n[fs[0]] @ n[fs[1]]) < cos_t:
            out.append((verts[a], verts[b]))
    return np.array(out)


# ── 자세 수식 ──────────────────────────────────────────────────────────────
def T_planar(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = (x, y, z)
    return T


def rolled(corners: np.ndarray, k: int) -> np.ndarray:
    """마커가 법선 둘레로 k·90° 돌아 붙었을 때, 검출 모서리 j 에 대응하는 STL 점."""
    return np.roll(corners, -k, axis=0)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── 검출 ──────────────────────────────────────────────────────────────────
def detect_markers(gray: np.ndarray, dictionary: str = "DICT_6X6_250") -> dict[int, np.ndarray]:
    """{id: (4,2) 모서리 px(TL,TR,BR,BL)}. OpenCV 4.5(구 API) · 4.7+(ArucoDetector) 둘 다."""
    import cv2
    ar = cv2.aruco
    d = (ar.getPredefinedDictionary if hasattr(ar, "getPredefinedDictionary") else ar.Dictionary_get)(
        getattr(ar, dictionary))
    new_api = hasattr(ar, "ArucoDetector")
    p = ar.DetectorParameters() if new_api else ar.DetectorParameters_create()
    p.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
    # 기본 0.05 는 머리 시점(-x 면이 세로로 크게 눌림)에서 마커 바깥·안쪽 윤곽을
    # "너무 가까운 두 후보"로 보고 둘 다 버린다 (합성 60장 중 23장 미검출 → 0.02 에서 0).
    p.minMarkerDistanceRate = MIN_MARKER_DISTANCE_RATE
    if new_api:
        corners, ids, _ = ar.ArucoDetector(d, p).detectMarkers(gray)
    else:
        corners, ids, _ = ar.detectMarkers(gray, d, parameters=p)
    if ids is None:
        return {}
    out: dict[int, np.ndarray] = {}
    dup: set[int] = set()
    for c, i in zip(corners, ids.reshape(-1)):
        if int(i) in out:      # 같은 id 가 두 번 — 어느 쪽인지 모르니 버린다
            dup.add(int(i))
        out[int(i)] = np.asarray(c, float).reshape(4, 2)
    return {k: v for k, v in out.items() if k not in dup}


# ── 풀이 ──────────────────────────────────────────────────────────────────
@dataclass
class Fit:
    pose: np.ndarray            # (4,) x, y, z, yaw
    k: int                      # 마커 회전(×90°)
    rms_px: float
    k_margin: float             # 2등 회전의 rms / 1등 rms (클수록 확실)
    tilt_deg: float             # 6D PnP 로 본 STL z 와 base z 사이 각 (upright 가정 점검)


def _resid(pose, T_base_cam, K, obj, img):
    T = T_planar(*pose)
    uv, z = project(T_base_cam, K, obj @ T[:3, :3].T + T[:3, 3])
    r = (uv - img).reshape(-1)
    return np.where(np.repeat(z, 2) > 0, r, 1e3)


def _pnp_init(corners_px, K, size, T_base_cam, corners_stl, k):
    """IPPE_SQUARE 6D → T_base_stl. 반환 (평면 초기값 4, 기울기°)."""
    import cv2
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
    ok, rvec, tvec = cv2.solvePnP(obj, corners_px.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T_cam_m = np.eye(4)
    T_cam_m[:3, :3], T_cam_m[:3, 3] = R, tvec.reshape(3)
    # 마커 프레임에서 본 STL 점 = 모서리 대응으로 정한 강체변환(Kabsch)
    src = rolled(corners_stl, k)                     # STL 점, 검출 모서리 순서
    T_m_stl = _kabsch(src, obj)                      # obj = T_m_stl · src
    T = T_base_cam @ T_cam_m @ T_m_stl
    zax = T[:3, 2]
    tilt = math.degrees(math.acos(float(np.clip(zax[2], -1.0, 1.0))))
    yaw = math.atan2(T[1, 0], T[0, 0])
    return np.array([T[0, 3], T[1, 3], T[2, 3], yaw]), tilt


def _kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """B ≈ R·A + t 의 4×4."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def fit_one(corners_px: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, cfg: HolderCfg) -> Fit:
    """홀더 하나 — 네 회전 × yaw 초기값 여럿 중 잔차 최소."""
    from scipy.optimize import least_squares
    best: list[tuple[float, int, np.ndarray, float]] = []
    for k in range(4):
        init = _pnp_init(corners_px, K, cfg.marker_size, T_base_cam, cfg.corners_stl, k)
        if init is None:
            continue
        p0, tilt = init
        obj = rolled(cfg.corners_stl, k)
        cand = None
        for dyaw in np.radians([0, 45, -45, 90, -90, 135, -135, 180]):
            r = least_squares(_resid, p0 + [0, 0, 0, dyaw], args=(T_base_cam, K, obj, corners_px), method="lm")
            rms = float(np.sqrt(np.mean(r.fun ** 2)))
            if cand is None or rms < cand[0]:
                cand = (rms, r.x.copy())
        if cand is not None:
            best.append((cand[0], k, cand[1], tilt))
    if not best:
        raise RuntimeError("solvePnP 실패")
    best.sort(key=lambda t: t[0])
    rms, k, pose, tilt = best[0]
    pose[3] = wrap(pose[3])
    margin = best[1][0] / max(rms, 1e-6) if len(best) > 1 else float("inf")
    return Fit(pose=pose, k=k, rms_px=rms, k_margin=margin, tilt_deg=tilt)


def fit_shared(det: dict[int, np.ndarray], free: dict[int, Fit], K, T_base_cam, cfg: HolderCfg,
               shared: tuple[str, ...], fixed: dict[str, float] | None = None) -> dict[int, np.ndarray]:
    """shared 축은 한 값, fixed 축은 주어진 값, 나머지는 홀더별 — 모든 모서리를 한 번에.

    ★10.02: 머리 시점에서 x(앞뒤)와 z 는 영상에서 거의 구분되지 않는다. z 를 풀어 두고 x 를 공유하면 x 오차가 z 로
    빠져(6 cm 앞에 놓인 홀더가 z 0.29 · 나머지가 0.19) 잔차로 드러나지 않는다 → 상판 위 홀더는 z 를 fixed 로 준다."""
    from scipy.optimize import least_squares
    fixed = dict(fixed or {})
    ids = [i for i in cfg.ids if i in free]
    if not ids:
        return {}
    fx = {AXES.index(a): v for a, v in fixed.items()}
    sh = [AXES.index(a) for a in shared if AXES.index(a) not in fx] if len(ids) > 1 else []
    own = [j for j in range(4) if j not in sh and j not in fx]
    P = np.array([free[i].pose for i in ids])
    y0 = math.atan2(np.sin(P[:, 3]).mean(), np.cos(P[:, 3]).mean())   # yaw 평균은 원 위에서
    x0 = [y0 if j == 3 else P[:, j].mean() for j in sh] + list(P[:, own].reshape(-1))

    def unpack(x):
        out = {}
        for n, i in enumerate(ids):
            p = np.zeros(4)
            for j, v in fx.items():
                p[j] = v
            p[sh] = x[:len(sh)]
            p[own] = x[len(sh) + n * len(own): len(sh) + (n + 1) * len(own)]
            out[i] = p
        return out

    def res(x):
        ps = unpack(x)
        return np.concatenate([_resid(ps[i], T_base_cam, K, rolled(cfg.corners_stl, free[i].k), det[i]) for i in ids])

    if not x0:
        return unpack(np.zeros(0))
    r = least_squares(res, np.array(x0), method="lm" if len(x0) <= 2 * 4 * len(ids) else "trf")
    out = unpack(r.x)
    for p in out.values():
        p[3] = wrap(p[3])
    return out


def rms_of(pose, corners_px, K, T_base_cam, cfg, k) -> float:
    return float(np.sqrt(np.mean(_resid(pose, T_base_cam, K, rolled(cfg.corners_stl, k), corners_px) ** 2)))


def depth_check(depth: np.ndarray | None, corners_px: np.ndarray, pose, T_base_cam, cfg, k) -> float | None:
    """마커 안쪽 depth 중앙값 − 추정 마커 중심의 카메라 z (m). 참고용(D435i 는 ~8 mm 가깝게 본다)."""
    if depth is None:
        return None
    import cv2
    mask = np.zeros(depth.shape, np.uint8)
    c = corners_px.mean(0)
    cv2.fillConvexPoly(mask, np.round(c + 0.6 * (corners_px - c)).astype(np.int32), 1)
    d = depth[(mask > 0) & (depth > 0)]
    if d.size < 5:
        return None
    T = T_planar(*pose)
    ctr = cfg.corners_stl.mean(0) @ T[:3, :3].T + T[:3, 3]
    _, z = project(T_base_cam, np.eye(3), ctr[None])
    return float(np.median(d) - z[0])


# ── 출력 ──────────────────────────────────────────────────────────────────
def overlay(rgb: np.ndarray, K, T_base_cam, cfg: HolderCfg, det, poses: dict[int, np.ndarray], path: Path) -> None:
    import cv2
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    edges = feature_edges(load_stl(cfg.stl)) * cfg.stl_scale
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]
    for n, (i, p) in enumerate(poses.items()):
        T = T_planar(*p)
        E = edges.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
        uv, z = project(T_base_cam, K, E)
        uv = uv.reshape(-1, 2, 2)
        col = colors[n % len(colors)]
        for (a, b), za in zip(uv, z.reshape(-1, 2)):
            if (za > 0).all():
                cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, 1, cv2.LINE_AA)
        o, _ = project(T_base_cam, K, T[:3, 3][None])
        cv2.drawMarker(img, tuple(np.round(o[0]).astype(int)), col, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, f"{i}: x{p[0]:+.3f} y{p[1]:+.3f} z{p[2]:.3f} {math.degrees(p[3]):+.1f}deg",
                    (10, 30 + 26 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    for i, c in det.items():
        for j, q in enumerate(c):
            cv2.circle(img, tuple(np.round(q).astype(int)), 3, (0, 0, 255) if j == 0 else (0, 255, 0), -1)
    cv2.imwrite(str(path), img)


def write_poses(path: Path, cfg: HolderCfg, poses: dict[int, np.ndarray], meta: str) -> None:
    lines = ["# 컵홀더 STL 원점의 base_link 자세 — scripts/calib/cup_holder_pose.py 가 쓴다(손으로 고치지 않는다).",
             "# 원점 = 컵 축 · 받침 윗면(z=0 of cup_holder.stl). yaw = base z 둘레(rad), roll·pitch 0(upright).",
             f"# {meta}", "holders:"]
    for name, i in zip(cfg.names, cfg.ids):
        if i not in poses:
            continue
        x, y, z, yaw = poses[i]
        lines.append(f"  {name}: {{marker_id: {i}, position: [{x:.4f}, {y:.4f}, {z:.4f}], yaw_rad: {yaw:.4f}, "
                     f"yaw_deg: {math.degrees(yaw):.2f}}}")
    path.write_text("\n".join(lines) + "\n")


def _template_geometry(cfg: HolderCfg, table_z: float) -> tuple[np.ndarray, float]:
    """마커 법선(STL) · 홀더 원점 z — 바닥(= 마커 아래 변 높이)이 table_z 에 놓인다."""
    c = cfg.corners_stl
    normal = np.cross(c[3] - c[0], c[1] - c[0])
    return normal / np.linalg.norm(normal), table_z - float(c[:, 2].min())


def template_fallback(gray: np.ndarray, K: np.ndarray, T_bc: np.ndarray, cfg: HolderCfg, ids: list[int],
                      table_z: float) -> dict:
    """ArUco 가 못 찾은 id → marker_template_fit.TemplateHit (전체 탐색)."""
    from marker_template_fit import fit_templates
    from table_cad_extrinsics import TABLE_X, TABLE_Y

    normal, z_origin = _template_geometry(cfg, table_z)
    return fit_templates(gray, K, T_bc, cfg.corners_stl, normal, cfg.dictionary, list(ids), z_origin,
                         roi_xy=(TABLE_X, TABLE_Y))


@dataclass
class Estimate:
    """한 장의 추정 결과. poses 는 shared 제약까지 건 최종값(id → x,y,z,yaw)."""
    det: dict[int, np.ndarray]           # id → 마커 모서리 px
    src: dict[int, str]                  # id → aruco | template | track
    free: dict[int, Fit]
    poses: dict[int, np.ndarray]
    rms: dict[int, float]                # 최종 자세의 재투영 rms(px)
    hits: dict                           # id → TemplateHit (다음 장 추적용)
    missing: list[int]
    shared: tuple[str, ...]

    @property
    def worst_rms(self) -> float:
        return max(self.rms.values(), default=0.0)

    @property
    def ok(self) -> bool:
        return bool(self.poses) and not self.missing and self.worst_rms <= SHARED_FAIL_PX


def estimate(gray: np.ndarray, K: np.ndarray, T_bc: np.ndarray, cfg: HolderCfg, *,
             shared: tuple[str, ...] | None = None, table_z: float = TABLE_TOP_Z, use_template: bool = True,
             prev_hits: dict | None = None) -> Estimate:
    """ArUco → (못 찾은 id) 직전 자세 추적 → (그래도 없으면) 전체 무늬 탐색 → 홀더별 4자유도 → shared 공유."""
    shared = cfg.shared if shared is None else shared
    det_all = detect_markers(gray, cfg.dictionary)
    det = {i: det_all[i] for i in cfg.ids if i in det_all}
    src = {i: "aruco" for i in det}
    hits: dict = {}
    todo = [i for i in cfg.ids if i not in det]
    if todo and use_template:
        if prev_hits:
            from marker_template_fit import track_templates
            _, z_origin = _template_geometry(cfg, table_z)
            tracked = track_templates(gray, K, T_bc, cfg.corners_stl, cfg.dictionary, z_origin,
                                      {i: h for i, h in prev_hits.items() if i in todo})
            for i, h in tracked.items():
                det[i], src[i], hits[i] = h.corners_px, "track", h
            todo = [i for i in todo if i not in tracked]
        if todo:
            for i, h in template_fallback(gray, K, T_bc, cfg, todo, table_z).items():
                det[i], src[i], hits[i] = h.corners_px, "template", h
            todo = [i for i in todo if i not in hits]
    free = {i: fit_one(det[i], K, T_bc, cfg) for i in det}
    _, z_origin = _template_geometry(cfg, table_z)
    fixed = {"z": z_origin} if cfg.on_table else {}
    poses = fit_shared(det, free, K, T_bc, cfg, shared, fixed) if free else {}
    rms = {i: rms_of(p, det[i], K, T_bc, cfg, free[i].k) for i, p in poses.items()}
    return Estimate(det=det, src=src, free=free, poses=poses, rms=rms, hits=hits, missing=todo, shared=shared)


def yaw_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def main(argv=None) -> int:
    from cup_pose_relay import load_extrinsics

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", type=Path, help="rgb · (depth) · K npz (없으면 라이브 1장)")
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--extrinsics", type=Path, default=None, help="기본: cfg 의 camera_extrinsics")
    ap.add_argument("--shared", default=None, help="공유 축 덮어쓰기(쉼표, 예: x 또는 x,yaw · 빈 문자열이면 없음)")
    ap.add_argument("--png", type=Path, default=Path("/tmp/cup_holder_overlay.png"))
    ap.add_argument("--write", action="store_true", help=f"결과를 {DEFAULT_OUT.relative_to(SIM2REAL)} 에 쓴다")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--domain", type=int, default=126, help="카메라 노드 도메인(실기 126 · localhost 전용)")
    ap.add_argument("--no-template", action="store_true",
                    help="ArUco 가 못 찾은 마커에 무늬 맞춤 대체 검출(marker_template_fit)을 쓰지 않는다")
    ap.add_argument("--table-z", type=float, default=TABLE_TOP_Z,
                    help="대체 검출의 테이블 상판 높이(base, m) — 홀더 바닥이 여기 놓였다고 보고 z 를 고정한다")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.cfg)
    shared = cfg.shared if args.shared is None else tuple(a for a in args.shared.split(",") if a)
    ext = load_extrinsics(args.extrinsics or cfg.extrinsics)
    T_bc = T_from(ext.cam_pos, ext.cam_quat)

    npz = args.npz or grab_live(args.domain)
    d = np.load(npz)
    rgb, K = d["rgb"], d["K"].astype(float)
    depth = None
    if "depth" in d.files:
        depth = d["depth"].astype(float)
        depth = depth / 1000.0 if depth.max() > 50 else depth
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    est = estimate(gray, K, T_bc, cfg, shared=shared, table_z=args.table_z, use_template=not args.no_template)
    print("검출 " + " · ".join(f"id{i} {est.src[i]}" for i in sorted(est.det))
          + (f" · ✗ 못 찾음 {est.missing}" if est.missing else ""))
    for i, h in sorted(est.hits.items()):
        print(f"  [무늬] id{i} NCC {h.ncc:.3f} (다른 id/회전 최고 {h.runner_up:.3f})"
              + ("  ⚠모호" if h.runner_up > h.ncc - 0.15 else ""))
    if not est.det:
        print("✗ 마커가 하나도 없다 — 영상/조명/머리 자세 확인")
        return 1

    print("\n[free] 홀더별 4자유도(x · y · z · yaw), T_base_cam 고정")
    for name, i in zip(cfg.names, cfg.ids):
        if i not in est.free:
            continue
        f, sr = est.free[i], est.src[i]
        side = float(max(np.linalg.norm(est.det[i] - np.roll(est.det[i], 1, 0), axis=1)))
        # 무늬 맞춤 모서리는 맞춘 자세의 투영이라 rms≈0 — 회전 판별 근거는 무늬 NCC 쪽이다
        kq = f"2등/1등 {f.k_margin:.1f}" if sr == "aruco" else "무늬 NCC 로 판별"
        warn = "  ⚠회전 판별 약함" if sr == "aruco" and f.k_margin < 2.0 else ""
        warn += f"  ⚠6D 기울기 {f.tilt_deg:.0f}° — 세워져 있지 않거나 PnP 불안정" if f.tilt_deg > 15 else ""
        print(f"  {name}(id{i}·{sr}) x {f.pose[0]:+.4f}  y {f.pose[1]:+.4f}  z {f.pose[2]:.4f}  "
              f"yaw {math.degrees(f.pose[3]):+6.1f}°  | rms {f.rms_px:.2f} px · 마커 {side:.0f} px · "
              f"부착 {f.k * 90}°({kq}) · 6D 기울기 {f.tilt_deg:.1f}°{warn}")

    label = ",".join(est.shared) if est.shared and len(est.poses) > 1 else "없음"
    print(f"\n[최종] 공유 축: {label}")
    for a in est.shared:
        j = AXES.index(a)
        vals = np.array([est.free[i].pose[j] for i in est.poses])
        spread = float(np.ptp(np.unwrap(vals)) if a == "yaw" else np.ptp(vals))
        if spread > SHARED_WARN_SPREAD[a]:
            unit = f"{math.degrees(spread):.1f}°" if a == "yaw" else f"{spread * 1000:.1f} mm"
            print(f"  ⚠ free 값의 {a} 폭이 {unit} — 같은 값이라는 가정이 실물과 맞는지 확인")
    for name, i in zip(cfg.names, cfg.ids):
        if i not in est.poses:
            continue
        p, r = est.poses[i], est.rms[i]
        dz = depth_check(depth, est.det[i], p, T_bc, cfg, est.free[i].k)
        dtxt = f" · depth−추정 {dz * 1000:+.0f} mm" if dz is not None else ""
        w = "  ⚠공유 제약으로 잔차 증가" if r > max(SHARED_WARN_PX, 2 * est.free[i].rms_px) else ""
        print(f"  {name}(id{i}·{est.src[i]}) origin x {p[0]:+.4f}  y {p[1]:+.4f}  z {p[2]:.4f} m  "
              f"yaw {math.degrees(p[3]):+6.1f}°  | 바닥 z {p[2] - 0.030:.4f}(상판 {TABLE_TOP_Z}) · rms {r:.2f} px{dtxt}{w}")

    overlay(rgb, K, T_bc, cfg, est.det, est.poses, args.png)
    print(f"\n겹친 영상: {args.png}")
    if est.worst_rms > SHARED_FAIL_PX:
        print(f"✗ 공유 제약 뒤 rms 최대 {est.worst_rms:.1f} px > {SHARED_FAIL_PX} — 공유 축({label}) 가정이 실물과 맞지 않는다."
              f" 위 [free] 값을 보고 --shared 로 축을 바꾸거나 \"\" 로 끈다" + (" · 기록 안 함" if args.write else ""))
        return 2
    if args.write:
        how = ",".join(f"{i}:{est.src[i]}" for i in sorted(est.poses))
        meta = f"npz={npz} · extrinsics={(args.extrinsics or cfg.extrinsics).name} · shared={label} · 검출={how}"
        write_poses(args.out, cfg, est.poses, meta)
        print(f"기록: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

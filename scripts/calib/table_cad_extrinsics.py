#!/usr/bin/env python3
"""머리 카메라 외부 파라미터(T_base_cam)를 **테이블 CAD 와 실제 사진의 픽셀 비교**로 잰다 — 이 로봇의 기본 캘리브.

10.01 사용자(arm4090): "sim 환경의 부분하고, 실제 환경 사진을 픽셀로 분석하면 될텐데" · "앞으로 이게 디폴트 칼리브레이션 방법임".
5090 의 global_camera_extrinsics.yaml 을 arm4090 에 그대로 쓰면 env_v1 상판 외곽 · 구멍이 사진에서 40~60 px 어긋났다
(컵 xy ≈ 3 cm). 머리 조립 · 카메라 장착이 로봇마다 달라 남의 값을 빌려 쓸 수 없다.

방법(머리는 기준 자세 — head_home 을 먼저 한다. 이 값은 그 자세 하나에서만 맞는 정적 스냅샷이다):
  ① RGB + 정렬 depth + K 한 장(라이브: scripts/vision/grab_rgbd.py, 또는 --npz)
  ② CAD 기준점: hdgp env_v1 상판(PaintedMetal, z 0.205)의 구멍 원 — 꼭짓점을 군집해 **원 맞춤**(최소제곱)으로 중심을 낸다
     (점 평균은 꼭짓점이 원에 고르게 없어 ~3 mm 치우친다). 은색 와셔가 끼워진 구멍과 빈 검은 구멍 모두 쓴다.
  ③ 초기 외부 파라미터(--init)로 CAD 점을 투영 → 그 주변 창에서 밝기 대비 부호로 와셔(밝음)/구멍(어두움)을 가려 중심을 다듬는다
  ④ RANSAC PnP → 재투영 → 다시 다듬기(반복) → T_base_cam
  ⑤ 확인: 상판 외곽 + 구멍 원을 초기/새 값으로 겹친 PNG · depth RANSAC 테이블 평면(z 0.205 · 수평이어야 한다)
  ⑥ --write 일 때만 --out yaml 의 camera 블록(position · orientation_wxyz)을 바꾼다(다른 줄 보존)

    python3 scripts/calib/table_cad_extrinsics.py                                   # 라이브 1장 · 보고만
    python3 scripts/calib/table_cad_extrinsics.py --npz /tmp/rgbd.npz --png /tmp/overlay.png
    python3 scripts/calib/table_cad_extrinsics.py --write --out config/global_camera_extrinsics_arm4090.yaml

라이브 캡처는 카메라 노드(scripts/vision/camera_up.sh, 도메인 126 · localhost)가 떠 있어야 한다. 로봇은 움직이지 않는다.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
SIM2REAL = _HERE.parents[1]
RL_WS = SIM2REAL.parent
sys.path.insert(0, str(_HERE.parent))

ENV_USDA = RL_WS / "hdgp/assets/simulation_setting/env_v1/usd/env_v1.usda"
DEFAULT_INIT = SIM2REAL / "config/global_camera_extrinsics.yaml"
TABLE_TOP_Z = 0.205
TABLE_X = (0.07, 0.47)
TABLE_Y = (-0.45, 0.45)
#: 상판 가장자리 근처 꼭짓점(외곽선)은 구멍이 아니다
EDGE_MARGIN = 0.005
CLUSTER_R = 0.012
MIN_FEATURES = 8


@dataclass(frozen=True)
class Hole:
    xyz: np.ndarray       # base [m], 상판 위
    radius: float         # CAD 구멍 반지름 [m]


# ---------------------------------------------------------------- CAD
def _mesh_points(usda_text: str, mesh: str) -> np.ndarray:
    start = usda_text.index(f'def Mesh "{mesh}"')
    nxt = usda_text.find('def Mesh "', start + 10)
    block = usda_text[start:nxt if nxt > 0 else len(usda_text)]
    m = re.search(r"point3f\[\] points = \[(.*?)\]", block, re.S)
    if not m:
        raise ValueError(f"{mesh}: points 없음")
    nums = re.findall(r"\(([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)\)", m.group(1))
    return np.array([[float(a), float(b), float(c)] for a, b, c in nums])


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """대수적 원 맞춤(Kasa): x² + y² + Dx + Ey + F = 0."""
    A = np.c_[xy[:, 0], xy[:, 1], np.ones(len(xy))]
    b = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.array([-D / 2, -E / 2])
    return c, float(np.sqrt(max(c @ c - F, 0.0)))


def cad_holes(usda: Path = ENV_USDA, mesh: str = "PaintedMetal_000000") -> list[Hole]:
    """상판 윗면(z 0.205)의 구멍 원들 — 군집 → 원 맞춤."""
    pts = _mesh_points(usda.read_text(), mesh)
    top = pts[np.isclose(pts[:, 2], TABLE_TOP_Z, atol=1e-3)][:, :2]
    inner = top[(top[:, 0] > TABLE_X[0] + EDGE_MARGIN) & (top[:, 0] < TABLE_X[1] - EDGE_MARGIN)
                & (top[:, 1] > TABLE_Y[0] + EDGE_MARGIN) & (top[:, 1] < TABLE_Y[1] - EDGE_MARGIN)]
    clusters: list[list[np.ndarray]] = []
    for p in inner:
        for c in clusters:
            if np.linalg.norm(np.mean(c, axis=0) - p) < CLUSTER_R:
                c.append(p)
                break
        else:
            clusters.append([p])
    holes: list[Hole] = []
    for c in clusters:
        if len(c) < 5:
            continue
        center, r = fit_circle(np.array(c))
        # 같은 구멍의 위 · 아래 테두리(동심원)가 따로 군집된다 — 하나로 친다
        if any(np.linalg.norm(h.xyz[:2] - center) < 0.003 for h in holes):
            continue
        holes.append(Hole(np.array([center[0], center[1], TABLE_TOP_Z]), r))
    return sorted(holes, key=lambda h: (round(h.xyz[0], 3), h.xyz[1]))


def table_outline(n: int = 400) -> np.ndarray:
    xs = np.linspace(*TABLE_X, n // 2)
    ys = np.linspace(*TABLE_Y, n)
    e = [np.c_[xs, np.full_like(xs, y0), np.full_like(xs, TABLE_TOP_Z)] for y0 in TABLE_Y]
    e += [np.c_[np.full_like(ys, x0), ys, np.full_like(ys, TABLE_TOP_Z)] for x0 in TABLE_X]
    return np.vstack(e)


# ---------------------------------------------------------------- 기하
def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def T_from(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = pos
    return T


def project(T_base_cam: np.ndarray, K: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_cam_base = np.linalg.inv(T_base_cam)
    Xc = X @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    uv = np.c_[K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]]
    return uv, Xc[:, 2]


# ---------------------------------------------------------------- 영상
def refine_center(gray: np.ndarray, u: float, v: float, r_px: float) -> tuple[float, float] | None:
    """(u,v) 둘레 창에서 와셔(밝음) 또는 빈 구멍(어두움)의 중심 — 창 가장자리 대비 부호로 가린다."""
    h, w = gray.shape
    half = int(max(10, 2.2 * r_px))
    u0, v0 = int(round(u)) - half, int(round(v)) - half
    if u0 < 0 or v0 < 0 or u0 + 2 * half >= w or v0 + 2 * half >= h:
        return None
    win = gray[v0:v0 + 2 * half, u0:u0 + 2 * half].astype(float)
    yy, xx = np.mgrid[0:2 * half, 0:2 * half]
    rr = np.hypot(xx - half, yy - half)
    inner, ring = win[rr < 0.9 * r_px], win[(rr > 1.6 * r_px) & (rr < 2.1 * r_px)]
    if inner.size == 0 or ring.size == 0:
        return None
    contrast = float(np.median(inner) - np.median(ring))
    if abs(contrast) < 12:
        return None
    lo, hi = np.percentile(win, 20), np.percentile(win, 80)
    mask = (win > (lo + hi) / 2) if contrast > 0 else (win < (lo + hi) / 2)
    mask &= rr < 1.5 * r_px            # 이웃 구멍 · 테이블 무늬 배제
    if mask.sum() < 6:
        return None
    ys, xs = np.nonzero(mask)
    return u0 + float(xs.mean()), v0 + float(ys.mean())


def contrast_map(gray: np.ndarray, r_px: float) -> np.ndarray:
    """원판(반지름 r) 평균 − 고리(1.6r~2.1r) 평균 — 와셔는 +, 빈 구멍은 −."""
    import cv2

    R = int(np.ceil(2.1 * r_px))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    disk = (rr < 0.9 * r_px).astype(np.float32)
    ring = ((rr > 1.6 * r_px) & (rr < 2.1 * r_px)).astype(np.float32)
    k = disk / disk.sum() - ring / ring.sum()
    return cv2.filter2D(gray.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)


def coarse_shift(gray: np.ndarray, uv: np.ndarray, r_px: float, search: int = 120, step: int = 2) -> np.ndarray:
    """초기 투영이 수십 px 어긋나도 잡도록 — 투영점 전체를 평행이동해 |대비| 합이 최대인 이동을 찾는다."""
    C = np.abs(contrast_map(gray, r_px))
    h, w = C.shape
    best, best_s = (0, 0), -1.0
    for du in range(-search, search + 1, step):
        for dv in range(-search, search + 1, step):
            u = np.round(uv[:, 0] + du).astype(int)
            v = np.round(uv[:, 1] + dv).astype(int)
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if ok.sum() < MIN_FEATURES:
                continue
            s = float(C[v[ok], u[ok]].sum()) / ok.sum() * min(ok.sum(), len(uv))
            if s > best_s:
                best, best_s = (du, dv), s
    return np.array(best, float)


def solve(holes: list[Hole], gray: np.ndarray, K: np.ndarray, T_init: np.ndarray, iters: int = 4):
    import cv2

    obj_all = np.array([h.xyz for h in holes])
    T = T_init.copy()
    used, img, res = None, None, None
    shift = None
    for it in range(iters):
        uv, z = project(T, K, obj_all)
        if it == 0:
            r_med = float(np.median(K[0, 0] * np.array([h.radius for h in holes]) / np.maximum(z, 1e-3)))
            shift = coarse_shift(gray, uv, r_med)
            uv = uv + shift
        # 구멍 반지름의 화면 크기(px) ≈ f·r/z
        r_px = K[0, 0] * np.array([h.radius for h in holes]) / np.maximum(z, 1e-3)
        found = [refine_center(gray, u, v, r) if zz > 0 else None for (u, v), r, zz in zip(uv, r_px, z)]
        keep = [i for i, f in enumerate(found) if f is not None]
        if len(keep) < MIN_FEATURES:
            raise RuntimeError(f"검출된 기준점 {len(keep)} 개 < {MIN_FEATURES} — 머리 자세 · 초기 외부 파라미터 · 조명을 볼 것")
        obj = obj_all[keep]
        img = np.array([found[i] for i in keep])
        T_cam_base = np.linalg.inv(T)
        rvec = cv2.Rodrigues(T_cam_base[:3, :3])[0]
        tvec = T_cam_base[:3, 3].reshape(3, 1).copy()
        inl = np.arange(len(obj))
        # 반복 PnP + 잔차 이상치 제거(중앙값의 3배 · 최소 6 px) — 기준점이 20 개 남짓이라 RANSAC 보다 안정적이다
        for _ in range(3):
            ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], K, None, rvec, tvec, useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise RuntimeError("PnP 실패")
            pr, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            e = np.linalg.norm(pr[:, 0] - img, axis=1)
            inl = np.nonzero(e < max(6.0, 3 * float(np.median(e))))[0]
            if len(inl) < MIN_FEATURES:
                raise RuntimeError(f"이상치 제거 뒤 기준점 {len(inl)} 개 < {MIN_FEATURES}")
        Rcb = cv2.Rodrigues(rvec)[0]
        T = np.eye(4)
        T[:3, :3] = Rcb.T
        T[:3, 3] = -Rcb.T @ tvec[:, 0]
        used = np.array(keep)[inl]
        img = img[inl]
        proj, _ = project(T, K, obj[inl])
        res = np.linalg.norm(proj - img, axis=1)
    return T, used, img, res, shift


def plane_normal_cam(depth_m: np.ndarray, K: np.ndarray, T_guess: np.ndarray, seed: int = 0) -> np.ndarray | None:
    """테이블 상판의 법선을 **카메라 프레임**에서 잰다(깊이의 z 치우침과 무관 — 방향만 쓴다).

    상판 후보 점은 대략적인 T_guess 로 고르고, 맞춤은 카메라 좌표 그대로 한다. 법선은 base 의 +z 쪽을 향하게 둔다.
    """
    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h:4, 0:w:4]
    zz = depth_m[::4, ::4]
    P = np.stack([(uu - K[0, 2]) * zz / K[0, 0], (vv - K[1, 2]) * zz / K[1, 1], zz], -1).reshape(-1, 3)
    P = P[(P[:, 2] > 0.15) & (P[:, 2] < 2.0)]
    B = P @ T_guess[:3, :3].T + T_guess[:3, 3]
    sel = P[(B[:, 0] > TABLE_X[0] + 0.02) & (B[:, 0] < TABLE_X[1] - 0.02) & (np.abs(B[:, 1]) < TABLE_Y[1] - 0.03)
            & (np.abs(B[:, 2] - TABLE_TOP_Z) < 0.03)]
    if len(sel) < 500:
        return None
    rng = np.random.default_rng(seed)
    best = (0, None, None)
    for _ in range(300):
        a = sel[rng.choice(len(sel), 3, replace=False)]
        n = np.cross(a[1] - a[0], a[2] - a[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        c = int((np.abs((sel - a[0]) @ n) < 0.003).sum())
        if c > best[0]:
            best = (c, n, a[0])
    inl = sel[np.abs((sel - best[2]) @ best[1]) < 0.003]
    c = inl.mean(axis=0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
    if (T_guess[:3, :3] @ n)[2] < 0:
        n = -n
    return n


def joint_refine(T: np.ndarray, obj: np.ndarray, img: np.ndarray, K: np.ndarray, n_cam: np.ndarray,
                 sigma_px: float = 2.0, sigma_deg: float = 0.3) -> np.ndarray:
    """재투영(px) + 상판 법선(깊이)이 base +z 와 맞을 것 — 평면 기준점만으로는 앞뒤 기울기와 카메라 x 가 엉킨다."""
    import cv2
    from scipy.optimize import least_squares

    T_cb = np.linalg.inv(T)
    x0 = np.r_[cv2.Rodrigues(T_cb[:3, :3])[0][:, 0], T_cb[:3, 3]]

    def resid(x):
        R_cb = cv2.Rodrigues(x[:3])[0]
        pr, _ = cv2.projectPoints(obj, x[:3], x[3:], K, None)
        r_img = (pr[:, 0] - img).ravel() / sigma_px
        nb = R_cb.T @ n_cam                       # 카메라 법선 → base
        r_n = nb[:2] / np.radians(sigma_deg)
        return np.r_[r_img, r_n]

    x = least_squares(resid, x0, loss="huber", f_scale=1.0).x
    R_cb = cv2.Rodrigues(x[:3])[0]
    out = np.eye(4)
    out[:3, :3] = R_cb.T
    out[:3, 3] = -R_cb.T @ x[3:]
    return out


def table_plane(depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, seed: int = 0) -> dict:
    """depth → base 점 → 상판 근처 RANSAC 평면 → z = a·x + b·y + c (최소제곱)."""
    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h:4, 0:w:4]
    zz = depth_m[::4, ::4]
    P = np.stack([(uu - K[0, 2]) * zz / K[0, 0], (vv - K[1, 2]) * zz / K[1, 1], zz], -1).reshape(-1, 3)
    P = P[(P[:, 2] > 0.15) & (P[:, 2] < 2.0)]
    B = P @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    sel = B[(B[:, 0] > TABLE_X[0] + 0.01) & (B[:, 0] < TABLE_X[1] - 0.01) & (np.abs(B[:, 1]) < TABLE_Y[1] - 0.02)
            & (np.abs(B[:, 2] - TABLE_TOP_Z) < 0.04)]
    if len(sel) < 500:
        return {"ok": False}
    rng = np.random.default_rng(seed)
    best = (0, None, None)
    for _ in range(300):
        a = sel[rng.choice(len(sel), 3, replace=False)]
        n = np.cross(a[1] - a[0], a[2] - a[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        c = int((np.abs((sel - a[0]) @ n) < 0.004).sum())
        if c > best[0]:
            best = (c, n, a[0])
    inl = sel[np.abs((sel - best[2]) @ best[1]) < 0.004]
    A = np.c_[inl[:, 0], inl[:, 1], np.ones(len(inl))]
    a, b, c = np.linalg.lstsq(A, inl[:, 2], rcond=None)[0]
    xm = float(np.mean(TABLE_X))
    return {"ok": True, "z_mid": float(a * xm + c), "pitch_deg": float(np.degrees(np.arctan(a))),
            "roll_deg": float(np.degrees(np.arctan(b))), "inliers": int(len(inl))}


def overlay(rgb: np.ndarray, K: np.ndarray, holes: list[Hole], cands: dict, path: Path) -> None:
    import cv2

    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    colors = [(0, 0, 255), (0, 255, 0)]
    ring = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    circles = np.vstack([h.xyz + h.radius * np.c_[np.cos(ring), np.sin(ring), np.zeros_like(ring)] for h in holes])
    for (name, T), col in zip(cands.items(), colors):
        for X in (table_outline(), circles):
            uv, z = project(T, K, X)
            for u, v in uv[z > 0]:
                if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                    cv2.circle(vis, (int(u), int(v)), 1, col, -1)
    label = "  ".join(f"{'red' if i == 0 else 'green'}={n}" for i, n in enumerate(cands))
    cv2.putText(vis, label, (20, 40), 0, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(path), vis)


# ---------------------------------------------------------------- 실행
def grab_live(domain: int, timeout: float = 15.0) -> Path:
    out = Path(tempfile.mkdtemp(prefix="table_cad_")) / "rgbd.npz"
    env_cmd = (f"source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID={domain} ROS_LOCALHOST_ONLY=1 && "
               f"/usr/bin/python3 {SIM2REAL}/scripts/vision/grab_rgbd.py --out {out} --timeout {timeout}")
    r = subprocess.run(["bash", "-lc", env_cmd], capture_output=True, text=True, timeout=timeout + 30)
    if r.returncode != 0:
        raise SystemExit(f"✗ 라이브 캡처 실패(카메라 노드가 떠 있는가):\n{(r.stdout + r.stderr)[-800:]}")
    return out


def main(argv=None) -> int:
    from cup_pose_relay import load_extrinsics

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", type=Path, help="rgb · depth · K npz (없으면 라이브 1장)")
    ap.add_argument("--init", type=Path, default=DEFAULT_INIT, help="초기 외부 파라미터 yaml(투영 · 대응용)")
    ap.add_argument("--out", type=Path, default=None, help="--write 대상 yaml (기본: --init)")
    ap.add_argument("--write", action="store_true", help="camera 블록을 실제로 바꾼다(없으면 보고만)")
    ap.add_argument("--png", type=Path, default=Path("/tmp/table_cad_overlay.png"), help="겹친 영상")
    ap.add_argument("--usda", type=Path, default=ENV_USDA)
    ap.add_argument("--no-depth-tilt", action="store_true", help="깊이 상판 법선 제약을 빼고 PnP 만")
    ap.add_argument("--domain", type=int, default=126, help="카메라 노드 도메인(실기 126 · localhost 전용) — 셸의 ROS_DOMAIN_ID 를 믿지 않는다")
    args = ap.parse_args(argv)

    npz = args.npz or grab_live(args.domain)
    d = np.load(npz)
    rgb, K = d["rgb"], d["K"].astype(float)
    depth = d["depth"].astype(float)
    if depth.max() > 50:
        depth = depth / 1000.0
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    init = load_extrinsics(args.init)
    T0 = T_from(init.cam_pos, init.cam_quat)
    holes = cad_holes(args.usda)
    T, used, img, res, shift = solve(holes, gray, K, T0)
    print(f"[coarse] 초기 투영 평행이동 {shift.tolist()} px")
    print(f"[pnp] 재투영 평균 {res.mean():.2f} px (깊이 제약 전)")
    obj_used = np.array([holes[i].xyz for i in used])
    n_cam = plane_normal_cam(depth, K, T)
    if n_cam is not None and not args.no_depth_tilt:
        T = joint_refine(T, obj_used, img, K, n_cam)
        proj, _ = project(T, K, obj_used)
        res = np.linalg.norm(proj - img, axis=1)
    q = R_to_quat_wxyz(T[:3, :3])
    dR = T[:3, :3] @ T0[:3, :3].T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    print(f"[cad] env_v1 상판 구멍 {len(holes)} 개 · 사용 {len(used)} 개 · 재투영 평균 {res.mean():.2f} px · 최대 {res.max():.2f} px")
    print(f"[new] position {np.round(T[:3, 3], 6).tolist()}  orientation_wxyz {np.round(q, 6).tolist()}")
    print(f"[init→new] 위치 {np.round((T[:3, 3] - T0[:3, 3]) * 1000, 1).tolist()} mm · 회전 {ang:.2f}°")
    for name, TT in (("init", T0), ("new", T)):
        pl = table_plane(depth, K, TT)
        if pl["ok"]:
            print(f"[depth {name}] 상판 z {pl['z_mid']:.4f} (CAD {TABLE_TOP_Z}) · 앞뒤 기울기 {pl['pitch_deg']:+.2f}° · "
                  f"좌우 기울기 {pl['roll_deg']:+.2f}°")
    overlay(rgb, K, holes, {"init": T0, "new": T}, args.png)
    print(f"[png] {args.png}")
    if args.write:
        from calibrate_camera_extrinsics import update_camera_extrinsics_yaml

        out = args.out or args.init
        out.write_text(update_camera_extrinsics_yaml(str(out), T[:3, 3], q))
        print(f"[write] {out} camera 블록 갱신")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

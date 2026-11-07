"""11.08 놓기 학생 실기: 홀더 CAD 메쉬를 머리 카메라에 투영해 홀더 마스크 · 투영 깊이를 만든다(YOLO 대신, 결측 · 지연 없음).

학습 정의 = hdgp scripts/tools/student_adapters/holder_proj.py(43074945a) + mask_place.PlaceMask._holder_blocks 의 numpy 판:
  원해상도(1280x720, 원해상도 K)로 래스터 → 투영 깊이 > 측정 깊이 + occ_m(0.01) 화소를 가림으로 뺌 → 블록 b(8) 의 절반 이상이면 1(160x90).
  학습은 목표 홀더 하나만 그린다. 메쉬 assets/meshes/cup_holder.obj = 물체 좌표 m, 원점 = 컵 축 위 받침 윗면, z 위.
  홀더 자세 = base 위치 + z 축 yaw(config/cup_holder_poses_*.yaml 의 position · yaw_rad).
광학 프레임 = OpenCV(x 오른쪽 · y 아래 · z 앞), 화소 중심 = 정수 (u, v).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import _paths

HOLDER_MESH = _paths.SIM2REAL / "assets" / "meshes" / "cup_holder.obj"
HOLDER_POSES = _paths.SIM2REAL / "config" / "cup_holder_poses_5090_v1.yaml"
MESH_MAX_EXTENT_M = 0.5         # mm 단위 obj 를 m 로 잘못 읽는 사고를 막는다(hdgp 와 같다)
HOLDER_OCC_M = 0.01             # hdgp mask_place.HOLDER_OCC_M
BLOCK = 8                       # 1280x720 → 160x90(깊이 경로 block 과 같다)


def load_obj(path: str | Path = HOLDER_MESH) -> tuple[np.ndarray, np.ndarray]:
    """obj → (정점 (V,3) float64, 삼각형 (F,3) int64). 다각형 면은 부채꼴로 나눈다."""
    vs, fs = [], []
    for ln in Path(path).read_text().splitlines():
        p = ln.split()
        if p and p[0] == "v":
            vs.append([float(x) for x in p[1:4]])
        elif p and p[0] == "f":
            ix = [int(x.split("/")[0]) for x in p[1:]]
            ix = [i - 1 if i > 0 else len(vs) + i for i in ix]
            fs += [[ix[0], ix[j], ix[j + 1]] for j in range(1, len(ix) - 1)]
    if not vs or not fs:
        raise ValueError(f"메쉬가 비었다: {path}")
    v = np.asarray(vs, np.float64)
    ext = v.max(0) - v.min(0)
    if (ext > MESH_MAX_EXTENT_M).any():
        raise ValueError(f"메쉬 폭 {ext.round(3).tolist()} m > {MESH_MAX_EXTENT_M} m(단위가 mm 인가?): {path}")
    return v, np.asarray(fs, np.int64)


def load_holder_poses(path: str | Path = HOLDER_POSES) -> dict[str, tuple[np.ndarray, float]]:
    """{이름: (base 위치 (3,), yaw rad)}."""
    hs = yaml.safe_load(Path(path).read_text())["holders"]
    return {k: (np.asarray(v["position"], np.float64), float(v["yaw_rad"])) for k, v in hs.items()}


def holder_T(pos, yaw: float) -> np.ndarray:
    """base ← 홀더 4x4(서 있는 홀더, z 축 yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = pos
    return T


def raster_depth(vc: np.ndarray, faces: np.ndarray, k, w: int, h: int) -> np.ndarray:
    """vc (V,3) 광학 좌표 m → (h,w) 가장 가까운 면 깊이 m, 면 없으면 inf. 원근 보정 깊이(hdgp raster_depth 와 같은 식).
    shortcut: 꼭짓점 하나라도 z ≤ 1 mm 인 면은 버린다(근평면 자르기 없음), 홀더가 카메라 바로 앞에 올 일이 생기면 자르기를 넣는다."""
    k = np.asarray(k, np.float64)
    z = vc[:, 2]
    good_v = z > 1e-3
    zs = np.where(good_v, z, 1.0)
    u = k[0, 0] * vc[:, 0] / zs + k[0, 2]
    v = k[1, 1] * vc[:, 1] / zs + k[1, 2]
    U, V, Z = u[faces], v[faces], zs[faces]
    den = (U[:, 1] - U[:, 0]) * (V[:, 2] - V[:, 0]) - (U[:, 2] - U[:, 0]) * (V[:, 1] - V[:, 0])
    good = good_v[faces].all(1) & (np.abs(den) > 1e-9)
    x0 = np.clip(np.ceil(U.min(1)), 0, w).astype(np.int64)
    x1 = np.clip(np.floor(U.max(1)), -1, w - 1).astype(np.int64)
    y0 = np.clip(np.ceil(V.min(1)), 0, h).astype(np.int64)
    y1 = np.clip(np.floor(V.max(1)), -1, h - 1).astype(np.int64)
    bw = np.maximum(x1 - x0 + 1, 0)
    A = np.where(good, bw * np.maximum(y1 - y0 + 1, 0), 0)
    zbuf = np.full(h * w, np.inf)
    if A.sum() == 0:
        return zbuf.reshape(h, w)
    sel = np.flatnonzero(A)
    U, V, Z, den, A, bw, x0, y0 = U[sel], V[sel], Z[sel], den[sel], A[sel], bw[sel], x0[sel], y0[sel]
    # 면마다 w0 · w1 · 1/z 를 화면 (px, py) 의 1차식 계수로 한 번 만들고 짝마다 곱셈만 한다(1/z 는 화면에서 선형 = 원근 보정)
    a0, b0 = (V[:, 1] - V[:, 2]) / den, (U[:, 2] - U[:, 1]) / den
    c0 = (U[:, 1] * V[:, 2] - U[:, 2] * V[:, 1]) / den
    a1, b1 = (V[:, 2] - V[:, 0]) / den, (U[:, 0] - U[:, 2]) / den
    c1 = (U[:, 2] * V[:, 0] - U[:, 0] * V[:, 2]) / den
    iz = 1.0 / Z
    ai = a0 * (iz[:, 0] - iz[:, 2]) + a1 * (iz[:, 1] - iz[:, 2])
    bi = b0 * (iz[:, 0] - iz[:, 2]) + b1 * (iz[:, 1] - iz[:, 2])
    ci = c0 * (iz[:, 0] - iz[:, 2]) + c1 * (iz[:, 1] - iz[:, 2]) + iz[:, 2]
    ids = np.repeat(np.arange(len(A)), A)
    off = np.arange(ids.size) - np.repeat(np.cumsum(A) - A, A)
    b = bw[ids]
    px, py = x0[ids] + off % b, y0[ids] + off // b
    w0 = a0[ids] * px + b0[ids] * py + c0[ids]
    w1 = a1[ids] * px + b1[ids] * py + c1[ids]
    ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w0 + w1 <= 1.0 + 1e-6)
    ids, px, py = ids[ins], px[ins], py[ins]
    np.minimum.at(zbuf, py * w + px, 1.0 / (ai[ids] * px + bi[ids] * py + ci[ids]))
    return zbuf.reshape(h, w)


def block_reduce(m: np.ndarray, b: int = BLOCK) -> np.ndarray:
    """(H,W) bool → (H/b,W/b) bool: 블록 절반 이상이 켜지면 True(hdgp holder_proj.block_reduce)."""
    H, W = m.shape
    return m.reshape(H // b, b, W // b, b).mean((1, 3)) >= 0.5


def holder_mask(verts, faces, T_base_holder, T_base_cam, k_full, w: int = 1280, h: int = 720,
                depth_m: np.ndarray | None = None, occ_m: float = HOLDER_OCC_M, block: int = BLOCK):
    """→ (마스크 (h/b, w/b) bool, 원해상도 투영 깊이 (h,w) m · 없는 곳 0).
    depth_m = 원해상도 측정 깊이 m((h,w), 0 · nan = 없음)를 주면 투영 > 측정 + occ_m 화소를 뺀다(hdgp occlusion_keep)."""
    T = np.linalg.inv(T_base_cam) @ T_base_holder
    zb = raster_depth(verts @ T[:3, :3].T + T[:3, 3], faces, k_full, w, h)
    keep = np.isfinite(zb)
    if depth_m is not None:
        if depth_m.shape != (h, w):
            raise ValueError(f"측정 깊이 모양 {depth_m.shape} ≠ 원해상도 {(h, w)}(학습은 원해상도에서 가림을 뺀다)")
        ok = np.isfinite(depth_m) & (depth_m > 0)
        keep &= ~ok | (zb <= np.where(ok, depth_m, 0.0) + occ_m)
    return block_reduce(keep, block), np.where(np.isfinite(zb), zb, 0.0)

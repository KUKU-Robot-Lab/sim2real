#!/usr/bin/env python3
"""양팔 영점 확인 — 목 D435i 포인트 클라우드로 팔 부위별 (실물 − FK) 자세 차이와 좌우 거울 비대칭을 잰다.

손: RH56F1(openarm_rh56f1_bi_rl). 실기를 움직이지 않는다 — sym_check.py 스냅샷(meta.json · <snap>_depth.npy)만 읽는다.

방법
  1. depth → 카메라 점 → base_link 점(목 자세에 맞는 외부 파라미터 파일 --extrinsics. 스냅샷 목 틱이 그 파일 자세와 다르면 [head] ★다름 — 절대값은 믿지 않고 좌우 차만 본다).
  2. 팔마다 FK 메쉬(읽은 관절값)를 부위로 묶는다: forearm(al_5·al_6) · flange(al_7) · hand(손 몸체, 손가락 뺌).
  3. 부위마다 클라우드 점 → 메쉬 최근접으로 강체 ICP(잘라낸 최소제곱). 보이는 면만 쓰므로 가림 치우침이 없다.
     결과 = 그 부위의 (실물 − FK) 이동 · 회전(base 프레임).
  4. 명령 · 읽기가 좌우 거울이면 오른팔 결과를 y 로 뒤집어 왼팔과 뺀 값이 실물 비대칭이다.
     양팔 공통 오차(외부 파라미터 · 장착)는 이 차에서 지워지고, 팔별 절대값에는 남는다.

사용
  python3 scripts/ops/sym_cloud.py logs/runs/261112/sym_check/135830 --snap table
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy/policy_control"))
from policy_control.fk_numpy import UrdfTree  # noqa: E402

URDF = REPO.parent / "urdf/generated/rl/openarm_rh56f1_bi_rl.urdf"
EXTR = REPO / "config/global_camera_extrinsics_arm4090.yaml"
GROUPS = {
    "forearm": ("al_5", "al_6"),
    "flange": ("al_7",),
    "hand": ("hl_base", "hl_palm_1", "hl_palm_2", "hl_palm_3", "hl_palm_sensor"),
}
FINGER_KEYS = ("thumb", "index", "middle", "ring", "pinky")
MIRROR = np.diag([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class SegResult:
    n: int
    rms_before_mm: float
    rms_after_mm: float
    shift_mm: tuple  # 부위 중심에서 본 (실물 − FK) 이동, base 프레임
    rotvec_deg: tuple  # (실물 − FK) 회전, base 프레임 회전 벡터

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def quat_wxyz(q: list) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def load_extrinsics(path: Path) -> tuple[np.ndarray, dict]:
    e = yaml.safe_load(path.read_text())
    T = np.eye(4)
    T[:3, :3] = quat_wxyz(e["camera"]["orientation_wxyz"])
    T[:3, 3] = e["camera"]["position"]
    return T, e.get("head_pose", {})


def base_cloud(depth: np.ndarray, scale: float, intr: dict, T_bc: np.ndarray, short_m: float) -> np.ndarray:
    # sym_check 는 float32 미터로 저장한다(uint16 원시면 depth_scale 을 곱한다)
    z = depth.astype(float) * (1.0 if np.issubdtype(depth.dtype, np.floating) else scale)
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W]
    ok = (z > 0.25) & (z < 1.5)
    x = (u - intr["ppx"]) / intr["fx"]
    y = (v - intr["ppy"]) / intr["fy"]
    ray = np.stack([x, y, np.ones_like(x)], -1)
    # D435i 원시 depth 가 상판에서 4.1 mm 짧았다(10.27, 외부 파라미터 파일 주석) → 광선 방향으로 늘린다.
    zc = z + short_m / np.linalg.norm(ray, axis=-1)
    P = (ray * zc[..., None])[ok]
    return P @ T_bc[:3, :3].T + T_bc[:3, 3]


def link_meshes(urdf: Path, side: str, n_per_link: int = 6000) -> dict:
    """부위 이름 → [(link, 링크 프레임 점)]. 손가락은 'finger' 로 따로 둔다(맞춤에서 뺀다)."""
    root = ET.parse(urdf).getroot()
    out: dict = {}
    for L in root.findall("link"):
        name = L.get("name")
        if not name.startswith(side + "_"):
            continue
        short = name[2:]
        grp = next((g for g, ls in GROUPS.items() if short in ls), None)
        if grp is None and short.startswith("hl_") and any(k in short for k in FINGER_KEYS):
            grp = "finger"
        if grp is None:
            continue
        for vis in L.findall("visual"):
            g = vis.find("geometry/mesh")
            if g is None:
                continue
            mesh = trimesh.load(g.get("filename").replace("file://", ""), force="mesh")
            sc = np.array([float(s) for s in (g.get("scale") or "1 1 1").split()])
            o = vis.find("origin")
            xyz = [float(s) for s in (o.get("xyz") if o is not None else "0 0 0").split()]
            rpy = [float(s) for s in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
            M = trimesh.transformations.euler_matrix(*rpy, "sxyz")
            M[:3, 3] = xyz
            pts = mesh.sample(n_per_link) * sc
            out.setdefault(grp, []).append((name, pts @ M[:3, :3].T + M[:3, 3]))
    return out


def fk_points(tree: UrdfTree, meshes: dict, q: dict) -> dict:
    links = sorted({n for parts in meshes.values() for n, _ in parts})
    T = tree.transforms(links, q)
    return {g: np.vstack([p @ T[n][:3, :3].T + T[n][:3, 3] for n, p in parts]) for g, parts in meshes.items()}


def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B ≈ R A + t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def fit_segment(cloud: np.ndarray, model: np.ndarray, iters: int = 60, keep: float = 0.8) -> tuple:
    """클라우드 점 → 모델(FK 메쉬) 최근접. 모델을 (R, t)로 옮겨 실물에 맞춘다(실물 ≈ R·FK + t)."""
    R, t = np.eye(3), np.zeros(3)
    d0 = None
    for _ in range(iters):
        M = model @ R.T + t
        d, i = cKDTree(M).query(cloud)
        if d0 is None:
            d0 = d
        k = d <= np.quantile(d, keep)
        # cloud ≈ R_inc (M) + t_inc  →  누적
        Ri, ti = kabsch(M[i[k]], cloud[k])
        R, t = Ri @ R, Ri @ t + ti
        if np.linalg.norm(ti) < 1e-5 and np.arccos(np.clip((np.trace(Ri) - 1) / 2, -1, 1)) < 1e-5:
            break
    d, _ = cKDTree(model @ R.T + t).query(cloud)
    k = d <= np.quantile(d, keep)
    return R, t, float(np.sqrt((d0[d0 <= np.quantile(d0, keep)] ** 2).mean())), float(np.sqrt((d[k] ** 2).mean()))


def analyse_side(cloud: np.ndarray, fkp: dict, crop_m: float) -> dict:
    all_fk = np.vstack(list(fkp.values()))
    labels = np.concatenate([np.full(len(p), i) for i, p in enumerate(fkp.values())])
    names = list(fkp.keys())
    d, i = cKDTree(all_fk).query(cloud, distance_upper_bound=crop_m)
    hit = np.isfinite(d)
    lab = np.full(len(cloud), -1)
    lab[hit] = labels[i[hit]]
    out = {}
    for gi, g in enumerate(names):
        if g == "finger":
            continue
        pts = cloud[lab == gi]
        if len(pts) < 200:
            out[g] = None
            continue
        R, t, rb, ra = fit_segment(pts, fkp[g])
        c = fkp[g].mean(0)
        shift = R @ c + t - c
        out[g] = SegResult(len(pts), round(rb * 1e3, 1), round(ra * 1e3, 1),
                           tuple(float(v) for v in np.round(shift * 1e3, 1)),
                           tuple(float(v) for v in np.round(np.degrees(Rotation.from_matrix(R).as_rotvec()), 2)))
    return out


def mirror_diff(left: SegResult, right: SegResult) -> dict:
    sl, sr = np.array(left.shift_mm), MIRROR @ np.array(right.shift_mm)
    Rl = Rotation.from_rotvec(np.radians(left.rotvec_deg)).as_matrix()
    Rr = MIRROR @ Rotation.from_rotvec(np.radians(right.rotvec_deg)).as_matrix() @ MIRROR
    dR = Rotation.from_matrix(Rl @ Rr.T).as_rotvec()
    return {"shift_mm": tuple(float(v) for v in np.round(sl - sr, 1)), "shift_norm_mm": round(float(np.linalg.norm(sl - sr)), 1),
            "rot_deg": round(float(np.degrees(np.linalg.norm(dR))), 2),
            "rotvec_deg": tuple(float(v) for v in np.round(np.degrees(dR), 2))}


def overlay(color_png: Path, fk_all: dict, T_bc: np.ndarray, intr: dict, out: Path) -> None:
    """FK 메쉬 점(왼 초록 · 오른 빨강)을 컬러 사진에 투영 — 눈으로 맞춤을 확인한다."""
    from PIL import Image
    img = np.array(Image.open(color_png).convert("RGB"))
    Tcb = np.linalg.inv(T_bc)
    for side, col in (("left", (0, 255, 0)), ("right", (255, 0, 0))):
        P = np.vstack(list(fk_all[side].values()))[::4]
        C = P @ Tcb[:3, :3].T + Tcb[:3, 3]
        C = C[C[:, 2] > 0.1]
        u = (C[:, 0] / C[:, 2] * intr["fx"] + intr["ppx"]).astype(int)
        v = (C[:, 1] / C[:, 2] * intr["fy"] + intr["ppy"]).astype(int)
        ok = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
        img[v[ok], u[ok]] = (0.4 * img[v[ok], u[ok]] + 0.6 * np.array(col)).astype(np.uint8)
    Image.fromarray(img).save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--snap", default="table")
    ap.add_argument("--extrinsics", type=Path, default=EXTR)
    ap.add_argument("--depth-short-m", type=float, default=0.0041)
    ap.add_argument("--crop-m", type=float, default=0.03, help="FK 메쉬에서 이만큼 안의 점만 그 팔 부위로 쓴다")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    meta = json.loads((a.run_dir / "meta.json").read_text())
    S = meta["snaps"][a.snap]
    cam = S["camera"]
    T_bc, hp = load_extrinsics(a.extrinsics)
    ht = S.get("head_tick", {})
    tol = hp.get("tol_tick", 4)
    head_ok = bool(hp) and abs(ht.get("pan", -9e9) - hp["pan_tick"]) <= tol and abs(ht.get("tilt", -9e9) - hp["tilt_tick"]) <= tol
    print(f"[head] 스냅샷 pan/tilt {ht.get('pan')}/{ht.get('tilt')} · 외부 파라미터 자세 {hp.get('pan_tick')}/{hp.get('tilt_tick')} ±{tol} → {'OK' if head_ok else '★다름 — 결과의 절대값은 믿지 않는다'}")

    depth = np.load(a.run_dir / f"{a.snap}_depth.npy")
    cloud = base_cloud(depth, cam["depth_scale"], cam["intrinsics"], T_bc, a.depth_short_m)

    tree = UrdfTree(URDF)
    q = {}
    for side, s in (("left", "l"), ("right", "r")):
        for k, v in enumerate(S["q_mean_rad"][side]):
            q[f"{s}_aj_{k + 1}"] = v
    # 손 관절은 기록이 없어 0(펼침) — 손가락 부위는 맞춤에서 뺀다(손 몸체 · 팔 링크는 손 관절과 무관)
    for j in ET.parse(URDF).getroot().findall("joint"):
        if j.get("type") in ("revolute", "prismatic", "continuous"):
            q.setdefault(j.get("name"), 0.0)

    fk_all: dict = {}
    res: dict = {"snap": a.snap, "head_ok": head_ok, "head_tick": ht, "extrinsics": str(a.extrinsics), "sides": {}}
    for side, s in (("left", "l"), ("right", "r")):
        fkp = fk_points(tree, link_meshes(URDF, s), q)
        fk_all[side] = fkp
        sel = cloud[(cloud[:, 1] > 0) if s == "l" else (cloud[:, 1] < 0)]
        res["sides"][side] = analyse_side(sel, fkp, a.crop_m)
    res["mirror"] = {g: (mirror_diff(res["sides"]["left"][g], res["sides"]["right"][g])
                         if res["sides"]["left"].get(g) and res["sides"]["right"].get(g) else None) for g in GROUPS}

    q_asym = np.degrees(np.array(S["q_mean_rad"]["left"]) - np.array(S["q_mean_rad"]["right"]) * np.array([-1, -1, -1, 1, -1, -1, -1]))
    res["q_mirror_residual_deg"] = np.round(q_asym, 2).tolist()

    print(f"[q] 읽기 거울 잔차(왼 − 오른 거울) deg: {res['q_mirror_residual_deg']}")
    print(f"{'부위':8s} {'팔':5s} {'n':>6s} {'rms 전→후 mm':>14s}  {'이동(실물−FK) mm':>24s}  {'회전 deg':>22s}")
    for side in ("left", "right"):
        for g, r in res["sides"][side].items():
            if r is None:
                print(f"{g:8s} {side:5s}  점 부족")
                continue
            print(f"{g:8s} {side:5s} {r.n:6d} {r.rms_before_mm:6.1f}→{r.rms_after_mm:5.1f}   {str(r.shift_mm):>24s}  {str(r.rotvec_deg):>22s}")
    print("[거울 비대칭] 왼 − 거울(오른)")
    for g, m in res["mirror"].items():
        print(f"  {g:8s} {m}")

    out = a.out or (a.run_dir / "analysis" / f"sym_cloud_{a.snap}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    ser = {**res, "sides": {s: {g: (r.as_dict() if r else None) for g, r in d.items()} for s, d in res["sides"].items()}}
    out.write_text(json.dumps(ser, ensure_ascii=False, indent=1))
    color = Path(cam["color"]) if Path(cam["color"]).exists() else a.run_dir / f"{a.snap}_color.png"
    if color.exists():
        ov = out.with_name(f"sym_cloud_{a.snap}_overlay.png")
        overlay(color, fk_all, T_bc, cam["intrinsics"], ov)
        print(f"[overlay] {ov}")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ── 관절 오프셋 직접 맞춤: 팔 링크 + 손 몸체 메쉬를 q+Δq 로 움직여 클라우드 거리 최소 ──
def fit_joint_offsets(cloud: np.ndarray, tree: UrdfTree, q: dict, side: str, crop_m: float = 0.04,
                      seed: int = 0) -> dict:
    from scipy.optimize import least_squares
    root = ET.parse(URDF).getroot()
    rng = np.random.default_rng(seed)
    parts = []
    for L in root.findall("link"):
        n = L.get("name")
        short = n[2:]
        if not (n.startswith(side + "_") and (short in ("al_2", "al_3", "al_4", "al_5", "al_6", "al_7") or short in GROUPS["hand"])):
            continue
        for vis in L.findall("visual"):
            g = vis.find("geometry/mesh")
            mesh = trimesh.load(g.get("filename").replace("file://", ""), force="mesh")
            o = vis.find("origin")
            xyz = [float(s) for s in (o.get("xyz") if o is not None else "0 0 0").split()]
            rpy = [float(s) for s in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
            M = trimesh.transformations.euler_matrix(*rpy, "sxyz"); M[:3, 3] = xyz
            pts = trimesh.sample.sample_surface(mesh, 3000, seed=int(rng.integers(1 << 30)))[0]
            parts.append((n, pts @ M[:3, :3].T + M[:3, 3]))
    names = [f"{side}_aj_{k}" for k in range(1, 8)]
    links = sorted({n for n, _ in parts})

    def model(dq):
        qq = dict(q); [qq.__setitem__(j, q[j] + d) for j, d in zip(names, dq)]
        T = tree.transforms(links, qq)
        return np.vstack([p @ T[n][:3, :3].T + T[n][:3, 3] for n, p in parts])

    m0 = model(np.zeros(7))
    d, _ = cKDTree(m0).query(cloud, distance_upper_bound=crop_m)
    pts = cloud[np.isfinite(d)]
    pts = pts[rng.choice(len(pts), min(6000, len(pts)), replace=False)]

    def resid(dq):
        return cKDTree(model(dq)).query(pts)[0]

    r0 = resid(np.zeros(7))
    sol = least_squares(resid, np.zeros(7), loss="huber", f_scale=0.005, diff_step=1e-3,
                        bounds=(-np.radians(15), np.radians(15)))
    J = sol.jac
    sv = np.linalg.svd(J, compute_uv=False)
    return {"dq_deg": [round(float(v), 2) for v in np.degrees(sol.x)], "n": int(len(pts)),
            "rms_before_mm": round(float(np.sqrt((r0[r0 < np.quantile(r0, 0.8)] ** 2).mean()) * 1e3), 1),
            "rms_after_mm": round(float(np.sqrt((sol.fun[sol.fun < np.quantile(sol.fun, 0.8)] ** 2).mean()) * 1e3), 1),
            "cond": round(float(sv[0] / max(sv[-1], 1e-12)), 1)}

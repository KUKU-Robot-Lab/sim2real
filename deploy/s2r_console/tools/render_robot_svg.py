#!/usr/bin/env python3
"""자산 URDF 의 메쉬에서 **링크마다 하나의 실루엣**을 가진 SVG 를 만든다(ROS·GPU 없음, 오프라인).

    python3 deploy/s2r_console/tools/render_robot_svg.py --view arms  --out .../web/robot_arms.svg
    python3 deploy/s2r_console/tools/render_robot_svg.py --view hand --side right --out .../web/hand_right.svg

09.23 사용자: "urdf 에서 usd 등을 추출해서 하면 안 되는 건지" — 사진 대신 자산에서 뽑는다. 그림이 자산과
같은 것이 되고, **링크마다 `id` 가 있으므로 화면이 관절 상태에 따라 색을 칠할 수 있다**(정지 사진과 다른 점).

각 링크의 collision 메쉬를 FK 로 놓고 직교 투영한 뒤, 투영점의 볼록 껍질을 폴리곤 하나로 쓴다.
정확한 외형이 아니라 **어느 링크가 어디 있는지** 알아보게 하는 그림이다.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM2REAL = HERE.parents[2]
TOOLS = SIM2REAL / "deploy" / "policy_control" / "tools"
#: 보는 방향 → (가로축, 세로축) 로봇 좌표계 성분. 로봇 +x 앞 · +y 왼쪽 · +z 위.
VIEWS = {"front": ((1, -1.0), (2, 1.0)), "side": ((0, 1.0), (2, 1.0)), "top": ((0, 1.0), (1, -1.0))}
#: 투영에 쓰지 않는 축 = 깊이. 값이 클수록 카메라에 가깝다(그 순서로 그린다).
DEPTH = {"front": (0, 1.0), "side": (1, -1.0), "top": (2, 1.0)}
PAD = 0.02          # [m] 그림 테두리 여백
MIN_POINTS = 3


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def link_transforms(links, joints, q: dict[str, float]) -> dict[str, np.ndarray]:
    """링크 이름 → 4x4 월드 변환. 고정 관절은 각도 0, 회전 관절은 `q` (없으면 0)."""
    from scipy.spatial.transform import Rotation

    out: dict[str, np.ndarray] = {}
    children: dict[str, list] = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j)
    roots = [n for n in links if all(j.child != n for j in joints)]
    stack = [(r, np.eye(4)) for r in roots]
    while stack:
        name, T = stack.pop()
        out[name] = T
        for j in children.get(name, ()):
            Tj = np.array(j.T, dtype=float)
            if j.jtype in ("revolute", "continuous"):
                ang = float(q.get(j.name, 0.0))
                R = np.eye(4)
                R[:3, :3] = Rotation.from_rotvec(np.asarray(j.axis, float) * ang).as_matrix()
                Tj = Tj @ R
            stack.append((j.child, T @ Tj))
    return out


def link_polygons(links, T: dict[str, np.ndarray], keep, view: str) -> list[tuple[str, np.ndarray]]:
    """(링크 이름, 투영 볼록껍질 2xN). 메쉬가 없거나 점이 모자란 링크는 건너뛴다."""
    import trimesh
    from scipy.spatial import ConvexHull

    (ax_u, su), (ax_v, sv) = VIEWS[view]
    out = []
    for name, lk in links.items():
        if not keep(name) or name not in T:
            continue
        pts = []
        for Tc, kind, data in lk.collisions:
            if kind != "mesh":
                continue
            path, scale = data                                   # parse_urdf 의 mesh 항목 = (경로, scale 3)
            try:
                m = trimesh.load(str(path), force="mesh")
            except Exception:                                    # noqa: BLE001 — 그림이다, 막지 않는다
                continue
            v = np.asarray(m.vertices, float) * np.asarray(scale, float)
            v = v @ np.asarray(Tc, float)[:3, :3].T + np.asarray(Tc, float)[:3, 3]
            v = v @ T[name][:3, :3].T + T[name][:3, 3]
            pts.append(np.stack([v[:, ax_u] * su, v[:, ax_v] * sv], axis=1))
        if not pts:
            continue
        p = np.vstack(pts)
        if len(p) < MIN_POINTS:
            continue
        try:
            p = p[ConvexHull(p).vertices]
        except Exception:                                        # noqa: BLE001 — 퇴화 껍질은 그대로 둔다
            pass
        out.append((name, p))
    return out


def link_faces(links, T: dict[str, np.ndarray], keep, view: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(삼각형 3점의 2D 투영 Nx3x2, 깊이 N). 음영 렌더가 쓴다 — SVG 와 **같은 투영**이라 겹쳐진다."""
    import trimesh

    (ax_u, su), (ax_v, sv) = VIEWS[view]
    ax_d, sd = DEPTH[view]
    tris, depth, normal = [], [], []
    for name, lk in links.items():
        if not keep(name) or name not in T:
            continue
        for Tc, kind, data in lk.collisions:
            if kind != "mesh":
                continue
            path, scale = data
            try:
                m = trimesh.load(str(path), force="mesh")
            except Exception:                                    # noqa: BLE001
                continue
            v = np.asarray(m.vertices, float) * np.asarray(scale, float)
            for M in (np.asarray(Tc, float), T[name]):
                v = v @ M[:3, :3].T + M[:3, 3]
            f = np.asarray(m.faces, int)
            if not len(f):
                continue
            tri = v[f]                                           # N×3×3
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            tris.append(np.stack([tri[:, :, ax_u] * su, tri[:, :, ax_v] * sv], axis=2))
            depth.append(tri[:, :, ax_d].mean(axis=1) * sd)
            normal.append(n)
    if not tris:
        return np.zeros((0, 3, 2)), np.zeros(0), np.zeros((0, 3))
    return np.vstack(tris), np.concatenate(depth), np.vstack(normal)


def shade_png(links, T, keep, view: str, bbox, out: Path, width: int) -> int:
    """SVG 와 같은 창(bbox)에 메쉬를 음영으로 그린다. GPU 없음 — Isaac 이 GPU 를 쓸 수 없을 때의 길."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    tris, depth, normal = link_faces(links, T, keep, view)
    if not len(tris):
        return 0
    order = np.argsort(depth)                                    # 먼 것부터 그린다
    (ax_d, sd) = DEPTH[view]
    light = np.zeros(3)
    light[ax_d] = sd                                             # 카메라 쪽 조명
    light[2] += 0.45                                             # 위에서 조금
    light /= np.linalg.norm(light)
    lam = np.clip(np.abs(normal @ light), 0.0, 1.0)              # 법선 방향은 신경 쓰지 않는다(양면)
    grey = 0.13 + 0.42 * lam ** 1.3                              # 콘솔이 어두운 바탕이다 — 밝게 그리면 표가 안 보인다
    lo, hi = bbox
    span = np.maximum(hi - lo, 1e-6)
    height = int(round(width * span[1] / span[0]))
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1])
    ax.set_axis_off(), ax.set_aspect("equal")
    c = np.stack([grey * 0.94, grey, grey * 1.16, np.ones_like(grey)], axis=1)[order]   # 살짝 푸른 금속
    ax.add_collection(PolyCollection(tris[order], facecolors=np.clip(c, 0, 1), edgecolors="none"))
    fig.savefig(out, transparent=True, dpi=100)
    plt.close(fig)
    return len(tris)


def bbox_of(polys) -> tuple[np.ndarray, np.ndarray]:
    allp = np.vstack([p for _, p in polys])
    return allp.min(axis=0) - PAD, allp.max(axis=0) + PAD


def svg(polys, *, title: str, width: int = 340) -> str:
    """폴리곤들을 하나의 SVG 로. 링크 이름이 곧 `id` 라 화면이 색을 칠할 수 있다."""
    lo, hi = bbox_of(polys)
    span = np.maximum(hi - lo, 1e-6)
    height = int(width * span[1] / span[0])
    body = []
    for name, p in polys:
        xy = (p - lo) / span * np.array([width, height])
        xy[:, 1] = height - xy[:, 1]                             # SVG 는 y 가 아래로 간다
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        body.append(f'<polygon id="{name}" points="{pts}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{title}">\n  ' + "\n  ".join(body) + "\n</svg>\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--view", choices=("arms", "hand"), default="arms")
    ap.add_argument("--side", choices=("right", "left", "both"), default="both")
    ap.add_argument("--projection", choices=tuple(VIEWS), default="front")
    ap.add_argument("--frame", default="", help="이 링크 기준으로 본다(손은 손바닥 링크 — 월드 정면은 손이 옆으로 선다)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--png", type=Path, default=None,
                    help="같은 창에 메쉬를 음영으로 렌더한 PNG — SVG 를 그 위에 겹친다(09.23 사용자)")
    ap.add_argument("--png-width", type=int, default=680, help="PNG 가로 [px] — SVG 창과 같은 비율")
    args = ap.parse_args()

    W = _load("home_path_world")
    links, joints, _ = W.parse_urdf(W.URDF_DEFAULT)
    q = {}                                                       # 차렷(전부 0) — 화면은 실제 각도를 숫자로 따로 보여준다
    T = link_transforms(links, joints, q)
    if args.frame:
        if args.frame not in T:
            raise SystemExit(f"✗ 그런 링크가 없다: {args.frame}")
        inv = np.linalg.inv(T[args.frame])
        T = {k: inv @ v for k, v in T.items()}
    sides = ("r", "l") if args.side == "both" else (args.side[0],)

    def keep(name: str) -> bool:
        if not name.startswith(tuple(f"{s}_" for s in sides)) and not name.startswith(("base", "torso", "body")):
            return False
        is_hand = "_hl_" in name
        return is_hand if args.view == "hand" else not is_hand

    polys = link_polygons(links, T, keep, args.projection)
    if not polys:
        raise SystemExit(f"✗ 그릴 링크가 없다 (view {args.view} · side {args.side})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg(polys, title=f"{args.view} {args.side}"))
    print(f"[svg] {args.out} · 링크 {len(polys)} 개 · 투영 {args.projection}")
    if args.png:
        n = shade_png(links, T, keep, args.projection, bbox_of(polys), args.png, args.png_width)
        print(f"[png] {args.png} · 삼각형 {n} 개 (SVG 와 같은 창 — 그대로 겹친다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

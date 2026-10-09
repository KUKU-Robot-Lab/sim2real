#!/usr/bin/env python3
"""CAD 하나로 FP++ 물체를 등록하고, 색을 카메라로 맞추고, 정책이 볼 물체를 고른다(10.09 사용자: "cad 만 올리면 자동으로").

    # 1) 등록 — 메쉬(m · 원점) + config/objects.d/<이름>.yaml(색마다 물체 하나, 묶음 = 이름). 어느 PC 에서나
    python3 scripts/ops/fpp_object.py add <cad.stl|obj|ply> --name source240 --colors orange,pink \\
        [--origin-above-bottom 0.085] [--units auto|mm|m] [--class 39] [--no-symmetric] [--flip]
    # 2) 색 맞춤 — 로봇 PC(카메라 켠 상태). 물체를 테이블에 왼쪽부터 --colors 순서로 놓고. hue 를 재서 파일에 적는다
    python3 scripts/ops/fpp_object.py calib source240
    # 3) 정책이 볼 물체 — config/fpp_active.yaml + 미션 다시 생성(컵 단계 · 정책 cup_topic 이 이것을 따른다)
    python3 scripts/ops/fpp_object.py activate source240 --right source240_pink --left source240_orange

그 뒤 커밋 → push → 로봇 PC pull(손 복사 금지). 상황판 컵 단계가 묶음 컨테이너(fpp_<이름>) 하나로 한 번 찍는다.

원점: 정책은 FP++ 자세(물체 원점)를 기준으로 손을 보낸다 — 학습 sim 물체의 body 원점과 같아야 한다. hdgp OBJECTS 의
bottom_z 가 −0.085 면 --origin-above-bottom 0.085. 주지 않으면 높이 가운데(sim 원통 자산 대부분이 그렇다).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import fpp_color_pick as C  # noqa: E402

ACTIVE = "config/fpp_active.yaml"
DROPIN = "config/objects.d"
MESH_DIR = "assets/meshes"
MESH_IN_CONTAINER = "assets/s2r_meshes"     # fpp_group_up.sh 가 sim2real assets/meshes 를 붙인 자리
DEFAULT_CLASS = 41                           # YOLO cup — 묶음 노드는 bottle · cup · vase 를 다 받으니 기록용
HUE_HALF_MIN = 7.0                           # 잰 hue 중앙 ± max(이것, 3σ)


def convert_mesh(src: Path, dst: Path, origin_above_bottom: float | None, units: str = "auto") -> Path:
    """CAD → FP++ 메쉬(m, z 위). 원점 = 바닥 + origin_above_bottom(없으면 높이 가운데), x · y 는 바닥 둘레 중심."""
    import trimesh
    m = trimesh.load(str(src), force="mesh")
    ext = float(np.max(m.extents))
    scale = {"mm": 0.001, "m": 1.0}.get(units, 0.001 if ext > 2.0 else 1.0)    # auto: 2 m 넘으면 mm 로 본다
    m.apply_scale(scale)
    lo, hi = m.bounds
    z0 = float(lo[2])
    oz = (float(hi[2]) - z0) / 2.0 if origin_above_bottom is None else float(origin_above_bottom)
    cx, cy = (float(lo[0]) + float(hi[0])) / 2.0, (float(lo[1]) + float(hi[1])) / 2.0
    m.apply_translation([-cx, -cy, -(z0 + oz)])
    dst.parent.mkdir(parents=True, exist_ok=True)
    body = m.export(file_type="obj")
    lines = [ln for ln in str(body).splitlines() if not ln.startswith("#")]
    head = (f"# {src.name} → FP++ mesh (m, ×{scale:g}) — 원점 = 바닥 위 {oz:.4f} m · x y = 둘레 중심 "
            f"(scripts/ops/fpp_object.py, 손으로 고치지 않는다)")
    dst.write_text("\n".join([head, *lines]) + "\n", encoding="utf-8")
    return dst


def _objects_yaml_names(repo: Path) -> set[str]:
    raw = yaml.safe_load((repo / "config" / "objects.yaml").read_text(encoding="utf-8")) or {}
    return set((raw.get("objects") or {}).keys())


def add_object(repo: Path, cad: Path, *, name: str, colors: list[str], origin_above_bottom: float | None,
               units: str = "auto", yolo_class: int = DEFAULT_CLASS, symmetric: bool = True, flip: bool = False,
               dry_run: bool = False) -> list[str]:
    """메쉬 + objects.d/<name>.yaml(색마다 <name>_<색>, 묶음 <name>). 같은 이름을 다시 하면 파일을 갈아 끼운다."""
    repo = Path(repo)
    for c in colors:
        C.hue_ranges(c)                                           # 모르는 색 이름이면 여기서 ValueError
    names = [f"{name}_{c}" for c in colors]
    clash = sorted(({name, *names}) & _objects_yaml_names(repo))
    if clash:
        raise ValueError(f"{', '.join(clash)} 는 config/objects.yaml 에 손으로 적힌 물체다 — 다른 --name")
    if dry_run:
        return names
    mesh = convert_mesh(Path(cad), repo / MESH_DIR / f"{name}.obj", origin_above_bottom, units)
    import trimesh
    lo, hi = trimesh.load(str(mesh), force="mesh").bounds
    oz = -float(lo[2])
    objs = {}
    for n, c in zip(names, colors):
        entry = {
            "real": f"{c} {name} — {Path(cad).name} 에서 등록(fpp_object.py add)",
            "fpp": {"mesh_path": f"{MESH_IN_CONTAINER}/{name}.obj", "mesh_scale_to_meters": 1.0,
                    "cup_class_id": int(yolo_class), "detection_pick": "confidence", "yolo_confidence": 0.05,
                    "group": name, "color": c},
            "cad_to_body": {"position": [0.0, 0.0, 0.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "sim": {"usd": "", "origin_above_bottom_m": round(oz, 6)},
            "aabb": [[round(float(v), 4) for v in lo], [round(float(v), 4) for v in hi]],
        }
        if symmetric:
            entry["symmetry_axis"] = [0.0, 0.0, 1.0]
            entry["symmetry_flip"] = bool(flip)
        objs[n] = entry
    _write_dropin(repo, name, {"source_cad": Path(cad).name, "origin_above_bottom_m": round(oz, 6), "colors": colors}, objs)
    return names


def _write_dropin(repo: Path, name: str, meta: dict, objs: dict) -> Path:
    f = repo / DROPIN / f"{name}.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    head = (f"# 생성됨 — scripts/ops/fpp_object.py (묶음 '{name}'). add 로 다시 만들고 calib 로 색을 잰다. 손으로 고치지 않는다.\n")
    f.write_text(head + yaml.safe_dump({"meta": meta, "objects": objs}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return f


def hue_stats(h: np.ndarray) -> tuple[float, float]:
    """hue(0~180) 원형 중앙 · 표준편차 — 빨강(0/180 경계)도."""
    ang = np.asarray(h, float) * (2.0 * math.pi / 180.0)
    s, c = float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang)))
    mean = (math.atan2(s, c) % (2.0 * math.pi)) * 180.0 / (2.0 * math.pi)
    r = min(1.0, math.hypot(s, c))
    sd = math.sqrt(max(0.0, -2.0 * math.log(max(r, 1e-12)))) * 180.0 / (2.0 * math.pi)
    return mean, sd


def pair_left_to_right(ys: list[float], hues: list[np.ndarray], names: list[str]) -> dict[str, tuple[float, float]]:
    """base y 큰 쪽(왼쪽)부터 덩어리를 names 순서와 짝짓고 (hue 중앙, σ)."""
    if len(ys) != len(names):
        raise ValueError(f"테이블 위 색 덩어리 {len(ys)} 개 — 물체 {len(names)} 개를 왼쪽부터 놓아야 한다")
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    return {n: hue_stats(hues[i]) for n, i in zip(names, order)}


def set_hues(repo: Path, name: str, measured: dict[str, tuple[float, float]]) -> Path:
    """잰 (중앙, σ) → color = [중앙 − w, 중앙 + w], w = max(HUE_HALF_MIN, 3σ). 0/180 을 넘으면 lo > hi."""
    f = Path(repo) / DROPIN / f"{name}.yaml"
    doc = yaml.safe_load(f.read_text(encoding="utf-8"))
    for n, (mid, sd) in measured.items():
        w = max(HUE_HALF_MIN, 3.0 * sd)
        lo, hi = (mid - w) % 180.0, (mid + w) % 180.0
        doc["objects"][n]["fpp"]["color"] = [round(lo, 1), round(hi, 1)]
        doc["objects"][n]["fpp"]["hue_measured"] = [round(mid, 1), round(sd, 2)]
    return _write_dropin(Path(repo), name, doc.get("meta", {}), doc["objects"])


def write_active(repo: Path, *, group: str, right: str, left: str | None) -> Path:
    f = Path(repo) / ACTIVE
    sides = {"right": right, **({"left": left} if left else {})}
    head = "# 정책이 볼 FP++ 물체(묶음 · 팔별) — scripts/ops/fpp_object.py activate 가 쓴다. 미션 생성기가 읽는다.\n"
    f.write_text(head + yaml.safe_dump({"group": group, "sides": sides}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return f


# ── 로봇 PC: 카메라 한 장으로 색 재기 ────────────────────────────────────────────
def _grab(timeout: float = 10.0):
    import rclpy
    from cv_bridge import CvBridge
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init()
    node = rclpy.create_node("fpp_object_calib")
    box: dict = {}
    br = CvBridge()

    def cb(rgb, depth, info):
        box["v"] = (np.asarray(br.imgmsg_to_cv2(rgb, "rgb8")), np.asarray(br.imgmsg_to_cv2(depth, "passthrough")),
                    depth.encoding, np.asarray(info.k).reshape(3, 3))
    subs = [Subscriber(node, Image, "/camera/camera/color/image_raw", qos_profile=qos_profile_sensor_data),
            Subscriber(node, Image, "/camera/camera/aligned_depth_to_color/image_raw", qos_profile=qos_profile_sensor_data),
            Subscriber(node, CameraInfo, "/camera/camera/color/camera_info", qos_profile=qos_profile_sensor_data)]
    sync = ApproximateTimeSynchronizer(subs, 10, 0.04)
    sync.registerCallback(cb)
    import time
    t0 = time.monotonic()
    while "v" not in box and time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if "v" not in box:
        raise SystemExit("카메라 영상이 안 온다 — camera_up.sh(또는 상황판 컵 단계)로 RealSense 를 켤 것")
    rgb, depth, enc, K = box["v"]
    depth_m = depth.astype(float) * (0.001 if enc in ("16UC1", "mono16") else 1.0)
    return rgb, depth_m, K


def calib(repo: Path, name: str, camera_yaml: Path) -> dict[str, tuple[float, float]]:
    """테이블 위 선명한 색 덩어리(작업 영역 안)를 왼쪽부터 물체와 짝지어 hue 를 잰다."""
    from object_registry import GROUP_WORKSPACE, _base_from_camera
    doc = yaml.safe_load((Path(repo) / DROPIN / f"{name}.yaml").read_text(encoding="utf-8"))
    names = list(doc["objects"])
    aabb = doc["objects"][names[0]]["aabb"]
    rgb, depth, K = _grab()
    T = np.asarray(_base_from_camera(camera_yaml))
    h, s, v = C._hsv(rgb.reshape(-1, 3))
    vivid = ((s >= C.S_MIN) & (v >= C.V_MIN)).reshape(rgb.shape[:2])
    from scipy import ndimage
    st = np.ones((7, 7), bool)
    vivid = ndimage.binary_fill_holes(ndimage.binary_closing(vivid, structure=st))
    labels, n = ndimage.label(vivid)
    ys, hues = [], []
    hh = h.reshape(rgb.shape[:2])
    pieces = []
    for k in range(1, n + 1):
        if (labels == k).sum() >= 1500:
            pieces += C.split_by_depth(labels == k, depth, K, T, min_px=1500)    # 영상에서 붙은 물체를 3D 로 나눈다
    for m in pieces:
        p = C.mask_point(m, depth, K)
        if not C.in_workspace(T, p, GROUP_WORKSPACE):
            continue
        pb = T[:3, :3] @ p + T[:3, 3]
        if not C.looks_standing(C.mask_points_base(m, depth, K, T), aabb):
            print(f"  (버림) y {pb[1]:+.3f} x {pb[0]:.3f} · 서 있는 {name} 모양이 아니다(넓게 퍼짐 · 꼭대기 높이 다름 — 누운 물체?)")
            continue
        core = m & ndimage.binary_erosion(m, iterations=3) & ((s >= C.S_MIN) & (v >= C.V_MIN)).reshape(m.shape)
        ys.append(float(pb[1]))
        hues.append(hh[core] if core.any() else hh[m])
        print(f"  덩어리 y {pb[1]:+.3f} x {pb[0]:.3f} z {pb[2]:.3f} · {int(m.sum())} px · hue {hue_stats(hues[-1])[0]:.1f}")
    measured = pair_left_to_right(ys, hues, names)
    set_hues(Path(repo), name, measured)
    return measured


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="CAD → 메쉬 + objects.d/<이름>.yaml")
    a.add_argument("cad", type=Path)
    a.add_argument("--name", required=True)
    a.add_argument("--colors", required=True, help=f"쉼표로 — {', '.join(C.COLORS)}")
    a.add_argument("--origin-above-bottom", type=float, default=None, help="학습 sim body 원점의 바닥 위 높이 [m]")
    a.add_argument("--units", choices=("auto", "mm", "m"), default="auto")
    a.add_argument("--class", dest="yolo_class", type=int, default=DEFAULT_CLASS)
    a.add_argument("--no-symmetric", action="store_true", help="축대칭이 아닌 물체(손잡이 · 클램프)")
    a.add_argument("--flip", action="store_true", help="위아래도 같은 원통(FP++ 가 뒤집어 잡는다)")
    c = sub.add_parser("calib", help="카메라로 색(hue) 재기 — 로봇 PC, 물체를 왼쪽부터 --colors 순서로")
    c.add_argument("name")
    c.add_argument("--camera-extrinsics", type=Path, default=_ROOT / "config" / "global_camera_extrinsics_arm4090.yaml")
    t = sub.add_parser("activate", help="정책이 볼 물체 + 미션 다시 생성")
    t.add_argument("group")
    t.add_argument("--right", required=True)
    t.add_argument("--left", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "add":
        names = add_object(_ROOT, args.cad, name=args.name, colors=[x.strip() for x in args.colors.split(",") if x.strip()],
                           origin_above_bottom=args.origin_above_bottom, units=args.units, yolo_class=args.yolo_class,
                           symmetric=not args.no_symmetric, flip=args.flip)
        print(f"[fpp_object] 등록: {', '.join(names)} · 묶음 {args.name} · {MESH_DIR}/{args.name}.obj · {DROPIN}/{args.name}.yaml")
        print("  다음: (로봇 PC, 물체를 왼쪽부터 놓고) calib → activate → 커밋 · push · pull")
    elif args.cmd == "calib":
        got = calib(_ROOT, args.name, args.camera_extrinsics)
        for n, (mid, sd) in got.items():
            print(f"  {n}: hue {mid:.1f} ± {sd:.1f}")
    else:
        from object_registry import load_registry
        reg = load_registry()
        for n in filter(None, (args.right, args.left)):
            if reg.get(n).fpp.get("group") != args.group:
                raise SystemExit(f"{n} 는 묶음 {args.group} 이 아니다")
        write_active(_ROOT, group=args.group, right=args.right, left=args.left)
        subprocess.run([sys.executable, str(_ROOT / "scripts" / "ops" / "make_rh56f1_missions.py")], check=True)
        print(f"[fpp_object] 활성: {args.group} · 오른팔 {args.right} · 왼팔 {args.left or '-'} — 미션을 다시 만들었다")
    return 0


if __name__ == "__main__":
    sys.exit(main())

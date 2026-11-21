#!/usr/bin/env python3
"""글로벌 카메라(고정 D435i, ns /global_camera) 외부 파라미터를 테이블 고정 마커(70 · 71 · 73 · 75 · 76)로 푼다(★11.18). 로봇은 움직이지 않는다.

머리 m1117 을 만든 풀이(head_view_feedback.solve_extrinsics · anchors_from_frames, 마커 정답 = config/head_view_feedback_arm4090.yaml
anchor_truth)를 그대로 쓰고, 영상 토픽만 바꾼다. solve_extrinsics 는 코너가 마커의 어느 base 모서리인지 정하려고 대략의 초기 외부
파라미터가 필요한데 글로벌 카메라는 아직 값이 없다. 그래서:
  1. 머리 카메라(/camera, camset 자세)로 같은 마커를 보고 m1117 로 코너 모서리 부호를 정한다. 같은 영상으로 머리 외부 파라미터를 다시
     풀어 m1117 과 extr_tol 안인지 본다 — 머리가 camset 이 아니거나 m1117 이 틀리면 여기서 멈춘다(목은 움직이지 않는다).
  2. 그 부호로 글로벌 영상의 y-정답 마커(둘 이상) 코너 base 좌표를 만들어 PnP → 초기값 → solve_extrinsics 로 다듬는다. 최종 풀이는
     초기값으로 글로벌에 보인 정답 마커 전부의 모서리를 정하므로 머리에 안 보인 마커(75 · 76)도 들어간다. 머리 부호 마커가 글로벌에
     둘 미만이면 --init(글로벌 외부 파라미터 파일)이 있을 때만 그 자세를 초기값으로 쓴다.
  3. 73 의 자유 y 와 비교 마커(--compare-id, 기본 72 = 정답에서 뺀 마커)의 base 중심을 두 카메라로 비교해 출력한다.
결과는 새 파일(--out)로만 쓴다(있으면 거부). 확인 절차: docs/experiments/2026-11-18_global_camera_arm4090.md.

  source scripts/vision/common.sh
  python3 scripts/calib/global_cam_extrinsics.py --out config/global_camera_extrinsics_arm4090_global_<MMDD>.yaml
  python3 scripts/calib/global_cam_extrinsics.py --selftest
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import head_view_feedback as hv  # noqa: E402

GLOBAL_NS = "global_camera"


def corner_signs(head_m: dict, K, dist, size_m: float, R0: np.ndarray, t0: np.ndarray) -> dict[int, np.ndarray]:
    """머리 영상 마커 → {id: 4x2 base 모서리 부호}(aruco 코너 순서). 마커가 테이블에 고정이라 어느 카메라에서 봐도 같다."""
    return {i: hv._corner_offsets(v["corners"], K, dist, size_m, R0, t0)[0] / (size_m / 2) for i, v in head_m.items()}


def init_pose(glob_m: dict, signs: dict, truth: dict, z: float, size_m: float, K, dist) -> tuple[np.ndarray, np.ndarray]:
    """위치를 다 아는 마커(y 있음)의 코너 base 좌표 ↔ 글로벌 영상 코너 PnP → (R base←광학, t). 그런 마커가 둘 미만이면 ValueError."""
    import cv2
    ids = [i for i in sorted(glob_m) if i in signs and i in truth and truth[i][1] is not None]
    if len(ids) < 2:
        raise ValueError(f"글로벌 영상에서 위치를 다 아는 마커가 둘 이상 보여야 한다(보인 {sorted(glob_m)}, 머리에서 부호를 정한 {sorted(signs)})")
    h = size_m / 2
    obj = np.concatenate([np.c_[truth[i][0] + signs[i][:, 0] * h, truth[i][1] + signs[i][:, 1] * h, np.full(4, z)] for i in ids])
    img = np.concatenate([np.asarray(glob_m[i]["corners"], float) for i in ids])
    ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, float), np.asarray(dist, float), flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("글로벌 초기 PnP 실패")
    Rcb = cv2.Rodrigues(rvec)[0]
    return Rcb.T, -Rcb.T @ tvec.ravel()


def solve_global(head_m: dict, Kh, dh, glob_m: dict, Kg, dg, cfg: dict, R_head: np.ndarray, t_head: np.ndarray,
                 init: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """머리 · 글로벌 마커 → {"head": 머리 재풀이, "global": 글로벌 풀이, "head_d": (mm, deg) m1117 대비, "init": "head"|"file"}.
    초기 자세는 머리 부호가 있는 y-정답 마커 둘 이상의 PnP, 모자라면 init(글로벌 외부 파라미터 (R, t)), 그것도 없으면 ValueError.
    최종 풀이(solve_extrinsics)는 글로벌에 보인 정답 마커 전부의 모서리 부호를 초기 자세로 정하므로 머리에 안 보인 마커(75 · 76)도 들어간다.
    global["sign_src"] = {id: "head"(머리 부호로 초기 PnP 에 쓰임) | "global-init"(초기 자세로만 부호를 정함)}. 순수부(ROS 무관)."""
    size, z = float(cfg["anchor_size_m"]), float(cfg["anchor_truth"]["table_z_m"])
    truth = {int(i): (float(v[0]), None if v[1] is None else float(v[1])) for i, v in (cfg["anchor_truth"]["markers"] | (cfg.get("global_anchor_markers") or {})).items()}
    head_c = {i: v for i, v in head_m.items() if i in truth}
    sh = hv.solve_extrinsics({i: v["corners"] for i, v in head_c.items()}, Kh, dh, size, truth, z, R_head, t_head)
    signs = corner_signs(head_c, Kh, dh, size, R_head, t_head)
    try:
        R0, t0 = init_pose(glob_m, signs, truth, z, size, Kg, dg)
        src = "head"
    except ValueError:
        if init is None:
            raise
        (R0, t0), src = init, "file"
    sg = hv.solve_extrinsics({i: v["corners"] for i, v in glob_m.items() if i in truth}, Kg, dg, size, truth, z, R0, t0)
    sg["sign_src"] = {i: "head" if src == "head" and i in signs and truth[i][1] is not None else "global-init" for i in sg["ids"]}
    return {"head": sh, "global": sg, "head_d": hv.extrinsics_diff(sh["R"], sh["t"], R_head, t_head), "init": src}


def marker_base(m: dict, R: np.ndarray, t: np.ndarray) -> dict[int, np.ndarray]:
    """마커 중심(카메라 PnP tvec) → base xyz."""
    return {i: R @ np.asarray(v["tvec"], float) + t for i, v in m.items()}


def _yaml_text(res: dict, args, sd: float, cmp: dict) -> str:
    pos, quat = hv._camera_values(res["global"]["R"], res["global"]["t"])
    g = res["global"]
    prov = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": "scripts/calib/global_cam_extrinsics.py", "ns": f"/{args.ns}",
            "head_extrinsics": f"{Path(args.head_extrinsics).name} md5 {hv.file_md5(Path(args.head_extrinsics))}",
            "markers": g["ids"], "rms_px": round(g["rms_px"], 3), "max_px": round(g["max_px"], 3), "corner_sd_px": round(sd, 3),
            "free_y_mm": {int(i): round(v * 1000, 1) for i, v in g["free_y_m"].items()}, "compare_mm": cmp,
            "init": "head" if res["init"] == "head" else f"{Path(args.init).name} md5 {hv.file_md5(Path(args.init))}",
            "sign_src": {int(i): v for i, v in g["sign_src"].items()}}
    return ("# arm4090 글로벌 D435i(고정, ns /global_camera) 외부 파라미터 — scripts/calib/global_cam_extrinsics.py 가 썼다.\n"
            "# 테이블 마커(config/head_view_feedback_arm4090.yaml anchor_truth), RGB 만. 머리 카메라와 달리 head_pose 가 없다(고정).\n"
            "# ── T_base_cam: robot base ← camera_color_optical_frame ──\n"
            + yaml.safe_dump({"camera": {"frame": "camera_color_optical_frame", "position": pos, "orientation_wxyz": quat}},
                             sort_keys=False, default_flow_style=None, allow_unicode=True)
            + yaml.safe_dump({"provenance": prov}, sort_keys=False, default_flow_style=None, allow_unicode=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="새 외부 파라미터 파일(있으면 거부). 없으면 출력만")
    ap.add_argument("--ns", default=GLOBAL_NS, help="글로벌 카메라 ns(camera_up.sh global)")
    ap.add_argument("--head-ns", default="camera", help="머리 카메라 ns(코너 부호 · 머리 확인용)")
    ap.add_argument("--head-extrinsics", default=str(hv.DEFAULT_EXTRINSICS), help="머리 외부 파라미터(camset 자세)")
    ap.add_argument("--config", default=str(hv.DEFAULT_CONFIG), help="마커 정답 · 허용치")
    ap.add_argument("--frames", type=int, default=2 * hv.ANCHOR_FRAMES)
    ap.add_argument("--compare-id", type=int, nargs="*", default=[72], help="두 카메라로 base 중심을 비교할 마커 id(정답에 없는 것)")
    ap.add_argument("--compare-size", type=float, default=None, help="비교 마커 한 변 m(기본 anchor_size_m)")
    ap.add_argument("--init", type=Path, default=None,
                    help="글로벌 외부 파라미터 파일(초기 자세). 머리 부호 마커가 글로벌에 둘 미만 보일 때만 쓴다")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.out is not None and args.out.exists():
        print(f"❌ {args.out} 가 이미 있다 — 새 이름으로(덮어쓰지 않는다)")
        return 1
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    size = float(cfg["anchor_size_m"])
    ids = {int(i) for i in cfg["anchor_truth"]["markers"] | (cfg.get("global_anchor_markers") or {})}
    cids, csize = set(args.compare_id) - ids, args.compare_size or size
    R_head, t_head = hv.load_extrinsics(Path(args.head_extrinsics))

    import rclpy
    from rclpy.node import Node
    rclpy.init()
    node = Node("global_cam_extrinsics")
    try:
        cams = {k: hv.AnchorCam(node, rclpy, f"/{ns}/camera/color/image_raw", f"/{ns}/camera/color/camera_info")
                for k, ns in (("head", args.head_ns), ("global", args.ns))}
        got = {}
        for k, cam in cams.items():
            grays, K, dist = cam.grab(args.frames, timeout_s=10.0)
            m, sd = hv.anchors_from_frames(grays, K, dist, ids, size)
            c, _ = hv.anchors_from_frames(grays, K, dist, cids, csize) if cids else ({}, 0.0)
            got[k] = (m, sd, c, K, dist, list(grays[0].shape[::-1]))
            print(f"  {k}: {cam.color_topic} {got[k][5]} · 마커 {sorted(m)} · 비교 {sorted(c)} · 코너 σ {sd:.3f} px")
    except RuntimeError as err:
        print(f"❌ {err}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    (hm, hsd, hc, Kh, dh, _), (gm, gsd, gc, Kg, dg, _) = got["head"], got["global"]
    if max(hsd, gsd) > hv.ANCHOR_MAX_SD_PX:
        print(f"❌ 코너 σ 머리 {hsd:.3f} · 글로벌 {gsd:.3f} px > {hv.ANCHOR_MAX_SD_PX} — 카메라가 흔들리거나 가림")
        return 1
    try:
        res = solve_global(hm, Kh, dh, gm, Kg, dg, cfg, R_head, t_head, hv.load_extrinsics(args.init) if args.init else None)
    except ValueError as err:
        print(f"❌ {err}")
        return 1
    dmm, ddeg = res["head_d"]
    tol_mm, tol_deg = float(cfg.get("extr_tol_mm", 3.0)), float(cfg.get("extr_tol_deg", 0.3))
    print(f"  머리 재풀이: {Path(args.head_extrinsics).name} 대비 {dmm:.1f} mm · {ddeg:.3f}°(허용 {tol_mm:g} mm · {tol_deg:g}°), "
          f"rms {res['head']['rms_px']:.2f} px")
    g = res["global"]
    print(f"  글로벌: 마커 {g['ids']} rms {g['rms_px']:.2f} px(max {g['max_px']:.2f}) · 위치 "
          f"{[round(float(v), 4) for v in g['t']]} · 73 y 머리 {res['head']['free_y_m']} / 글로벌 {g['free_y_m']}")
    print(f"  초기 자세 {res['init']} · 모서리 부호 출처 {g['sign_src']}")
    hb, gb = marker_base(hc, R_head, t_head), marker_base(gc, g["R"], g["t"])
    cmp = {int(i): [round(float(v) * 1000, 1) for v in gb[i] - hb[i]] for i in sorted(set(hb) & set(gb))}
    for i in cmp:
        print(f"  비교 마커 {i}: 머리 {np.round(hb[i], 4).tolist()} · 글로벌 {np.round(gb[i], 4).tolist()} · 글로벌−머리 {cmp[i]} mm")
    if dmm > tol_mm or ddeg > tol_deg:
        print("❌ 머리 영상이 m1117 과 맞지 않는다 — 머리가 camset 자세(m1117 head_pose 틱)인지 보고, 목은 사용자 승인 뒤에만 옮긴다")
        return 1
    if g["rms_px"] > hv.ALIGN_MAX_RMS_PX:
        print(f"❌ 글로벌 재투영 rms {g['rms_px']:.2f} px > {hv.ALIGN_MAX_RMS_PX}")
        return 1
    if args.out is None:
        print("  --out 없음 — 쓰지 않는다")
        return 0
    args.out.write_text(_yaml_text(res, args, gsd, cmp), encoding="utf-8")
    print(f"✓ {args.out} md5 {hv.file_md5(args.out)}")
    return 0


def _synth() -> tuple:
    """합성 장면: 머리 = m1117 자세, 글로벌 = 테이블 옆 높은 곳에서 내려다봄 → (cfg, K, dist, R_h, t_h, R_g, eye, hm, gm, true_y)."""
    import cv2
    from scipy.spatial.transform import Rotation
    cfg = yaml.safe_load(hv.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    size, z = float(cfg["anchor_size_m"]), float(cfg["anchor_truth"]["table_z_m"])
    true_y = {70: -0.398, 71: -0.398, 73: -0.10, 72: 0.05, 75: 0.398, 76: 0.398}
    xy = {i: (float(v[0]), true_y[int(i)]) for i, v in (cfg["anchor_truth"]["markers"] | (cfg.get("global_anchor_markers") or {})).items()} | {72: (0.25, 0.05)}
    K, dist = np.array([[640.0, 0, 640], [0, 640.0, 360], [0, 0, 1]]), np.zeros(5)
    h = size / 2
    rot = {i: Rotation.from_euler("z", a, degrees=True).as_matrix() for i, a in ((70, 0), (71, 90), (73, 180), (72, -90), (75, 90), (76, 0))}

    def view(R, t):   # R, t = base←광학
        out = {}
        for i, (x, y) in xy.items():
            obj = (rot[i] @ np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]]).T).T + [x, y, z]
            c = cv2.projectPoints(obj, cv2.Rodrigues(R.T)[0], -R.T @ t, K, dist)[0].reshape(4, 2)
            rv, tv = hv._pnp(c, K, dist, size)
            out[int(i)] = {"corners": c, "rvec": rv, "tvec": tv}
        return out
    R_h, t_h = hv.load_extrinsics(hv.DEFAULT_EXTRINSICS)
    # 글로벌: (0.30, 0.45, 1.10) 에서 테이블 가운데(0.30, -0.15, 0.205)를 본다
    eye, tgt = np.array([0.30, 0.45, 1.10]), np.array([0.30, -0.15, z])
    zc = (tgt - eye) / np.linalg.norm(tgt - eye)
    xc = np.cross(zc, [0, 0, 1.0])
    xc /= np.linalg.norm(xc)
    R_g = np.c_[xc, np.cross(zc, xc), zc]
    return cfg, K, dist, R_h, t_h, R_g, eye, view(R_h, t_h), view(R_g, eye), true_y


def selftest() -> None:
    """합성 장면(_synth)에서 코너를 투영해 풀면 글로벌 자세가 1 mm · 0.05° 안에 돌아온다."""
    cfg, K, dist, R_h, t_h, R_g, eye, hm, gm, true_y = _synth()
    z = float(cfg["anchor_truth"]["table_z_m"])
    res = solve_global({i: v for i, v in hm.items() if i != 72}, K, dist, {i: v for i, v in gm.items() if i != 72}, K, dist,
                       cfg, R_h, t_h)
    dmm, ddeg = hv.extrinsics_diff(res["global"]["R"], res["global"]["t"], R_g, eye)
    assert dmm < 1.0 and ddeg < 0.05 and res["global"]["rms_px"] < 0.01 and res["head_d"][0] < 1.0, (dmm, ddeg, res["head_d"])
    assert abs(res["global"]["free_y_m"][73] - true_y[73]) < 1e-3
    b = marker_base({72: gm[72]}, res["global"]["R"], res["global"]["t"])[72]
    assert np.linalg.norm(b - [0.25, 0.05, z]) < 1e-3, b
    try:   # 위치를 다 아는 마커가 하나뿐이면 풀지 않는다
        solve_global(hm, K, dist, {73: gm[73], 70: gm[70]}, K, dist, cfg, R_h, t_h)
        raise AssertionError("마커 하나로 풀렸다")
    except ValueError:
        pass
    print(f"selftest ok: 글로벌 {dmm:.3f} mm · {ddeg:.4f}° · 73 y {res['global']['free_y_m'][73]:.4f} · 72 base {np.round(b, 4).tolist()}")


if __name__ == "__main__":
    sys.exit(main())

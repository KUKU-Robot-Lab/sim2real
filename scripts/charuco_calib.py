#!/usr/bin/env python3
"""ChArUco 보드 검출 → T_cam_board(optical) JSON 출력 (vision-3090, cv2 5.x).

NPZ(rgb,depth,K) 1프레임 → CharucoDetector → matchImagePoints → solvePnP.
출력 JSON(stdout): T_cam_board(4x4), board_objp, reproj_px, charuco_corners.
recompute_from_cad.py 가 이 JSON 을 받아 T_base_cam 산출.

보드: 7x5, DICT_6X6, square 0.030 / marker 0.022 (사용자 인쇄물).
cv2 신 API(CharucoDetector, ≥4.7/5.0) 필요 — vision-3090 .venv(5.0.0) 사용.
컵 검출(--cup)은 선택: ultralytics 있으면 카메라프레임 centroid 도 함께 출력.
"""
import argparse
import json
import sys

import numpy as np
import cv2

SX, SY, SQUARE, MARKER = 7, 5, 0.030, 0.022
DICT = cv2.aruco.DICT_6X6_250
CUP_CLASS_ID = 41   # COCO cup


def build_board():
    adict = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard((SX, SY), SQUARE, MARKER, adict)


def detect_charuco(gray, K):
    """(T_cam_board 4x4, objp Nx3, imgp Nx2, reproj_px) 또는 실패 시 None."""
    board = build_board()
    det = cv2.aruco.CharucoDetector(board)
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None or len(ci) < 6:
        return None
    objp, imgp = board.matchImagePoints(cc, ci)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, None)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
    reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)))
    return T, objp.reshape(-1, 3), imgp.reshape(-1, 2), reproj


def _cup_centroid_cam(rgb, depth, K, weights):
    """YOLO bbox(cup) + near-depth 밴드 → 카메라프레임 centroid. 없으면 None."""
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    best = None
    for r in YOLO(weights)(rgb, verbose=False):
        if r.boxes is None:
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        box = r.boxes.xyxy.cpu().numpy()
        for i, (c, p) in enumerate(zip(cls, conf)):
            if c == CUP_CLASS_ID and (best is None or p > best[0]):
                best = (float(p), box[i])
    if best is None:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in best[1]]
    h, w = depth.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    sub = depth[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    valid = (sub > 0.1) & (sub < 3.0)
    if not valid.any():
        return None
    centre = np.percentile(sub[valid], 35)
    m = valid & (sub > centre - 0.08) & (sub < centre + 0.08)
    zc = sub[m]
    pts = np.stack([(xs[m] - cx) / fx * zc, (ys[m] - cy) / fy * zc, zc], 1)
    return round(best[0], 3), pts.mean(0).round(4).tolist()


def main():
    ap = argparse.ArgumentParser(description="ChArUco 검출 → T_cam_board JSON")
    ap.add_argument("npz", help="capture_frame.py 출력(rgb,depth,K)")
    ap.add_argument("--cup", action="store_true", help="컵 centroid 도 검출(ultralytics)")
    ap.add_argument("--weights", default="models/yolo/yolo11n.pt")
    a = ap.parse_args()

    d = np.load(a.npz)
    rgb, depth, K = d["rgb"], d["depth"], d["K"].astype(np.float64)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    out = {}
    res = detect_charuco(gray, K)
    if res is None:
        out["charuco_corners"] = 0
        print(json.dumps(out))
        sys.exit("ChArUco 검출 실패 — 보드 가시/조명/포커스 확인")
    T, objp, _, reproj = res
    out["charuco_corners"] = len(objp)
    out["reproj_px"] = round(reproj, 3)
    out["T_cam_board"] = T.round(6).tolist()
    out["board_objp"] = objp.round(5).tolist()

    if a.cup:
        cup = _cup_centroid_cam(rgb, depth, K, a.weights)
        if cup is not None:
            out["cup_conf"], out["cup_cam"] = cup

    print(json.dumps(out))


if __name__ == "__main__":
    main()

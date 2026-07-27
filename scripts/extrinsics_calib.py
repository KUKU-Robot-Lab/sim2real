"""QR 정적 extrinsics 캘리브 순수 수학 (ROS 비의존).

T_base_cam = T_base_qr ∘ inv(T_cam_qr),  T_cam_qr = solvePnP(QR 코너).
쿼터니언은 wxyz. 코너 순서는 TL,TR,BR,BL (OpenCV QRCodeDetector 순서, 이미지 y하).
객체 좌표계는 cv2.SOLVEPNP_IPPE_SQUARE가 요구하는 y상(canonical) 규약을 따른다
(2D 코너 TL,TR,BR,BL ↔ 이미지 y하, 객체 y상).
"""
from __future__ import annotations

import numpy as np


def qr_object_points(qr_size: float) -> np.ndarray:
    """QR 중심 원점, 평면 z=0, cv2.SOLVEPNP_IPPE_SQUARE canonical 순서 (y상).

    2D 코너 TL,TR,BR,BL (이미지 y하)에 대응하는 객체점은 y상 규약이므로
    TL→(-h,h), TR→(h,h), BR→(h,-h), BL→(-h,-h) 순서가 된다.
    """
    h = qr_size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def solve_qr_pose(corners_2d: np.ndarray, qr_size: float, K: np.ndarray,
                  dist: np.ndarray | None) -> np.ndarray:
    import cv2
    obj = qr_object_points(qr_size).astype(np.float64)
    img = np.asarray(corners_2d, np.float64).reshape(-1, 1, 2)
    d = np.zeros(5) if dist is None else np.asarray(dist, np.float64).ravel()
    ok, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float64), d,
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("solvePnP failed for QR corners")
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def compose_base_cam(T_base_qr: np.ndarray, T_cam_qr: np.ndarray) -> np.ndarray:
    return T_base_qr @ np.linalg.inv(T_cam_qr)


def mat_to_pos_quat_wxyz(T: np.ndarray) -> tuple[list[float], list[float]]:
    m = T[:3, :3]
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return [float(v) for v in T[:3, 3]], [float(v) for v in q]


def reprojection_residual_px(corners_2d: np.ndarray, T_cam_qr: np.ndarray,
                             qr_size: float, K: np.ndarray,
                             dist: np.ndarray | None) -> float:
    import cv2
    obj = qr_object_points(qr_size).astype(np.float64)
    R = T_cam_qr[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = T_cam_qr[:3, 3].reshape(3, 1)
    d = np.zeros(5) if dist is None else np.asarray(dist, np.float64).ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), d)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - np.asarray(corners_2d, float), axis=1)))

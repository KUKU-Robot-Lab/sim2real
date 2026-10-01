"""테이블 CAD 캘리브(scripts/calib/table_cad_extrinsics.py) — CAD 구멍 · 합성 영상으로 자세 복원 · 깊이 법선 제약."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "calib"))
sys.path.insert(0, str(ROOT / "scripts"))

import table_cad_extrinsics as T  # noqa: E402

cv2 = pytest.importorskip("cv2")

K = np.array([[908.35, 0.0, 639.17], [0.0, 907.88, 363.88], [0.0, 0.0, 1.0]])
#: arm4090 10.01 캘리브 값(head_home_rh56f1) — 합성 장면의 정답
TRUE = T.T_from([0.050504, 0.037916, 0.820846], [0.130161, -0.69891, 0.696068, -0.100366])
#: 5090 값 — 사진에서 40~60 px 어긋났던 초기값
INIT = T.T_from([0.0608598821, 0.03255, 0.816465107], [0.119659999, -0.69690852, 0.69690852, -0.119659999])


def _holes():
    if not T.ENV_USDA.is_file():
        pytest.skip(f"hdgp 자산 없음: {T.ENV_USDA}")
    return T.cad_holes()


def test_fit_circle_recovers_center_from_uneven_arc():
    """점이 원의 한쪽에 몰려도 중심이 치우치지 않는다(점 평균이면 수 mm 치우친다)."""
    ang = np.linspace(0.0, 1.2 * np.pi, 15)
    xy = np.c_[0.12 + 0.009 * np.cos(ang), -0.015 + 0.009 * np.sin(ang)]
    c, r = T.fit_circle(xy)
    assert np.allclose(c, [0.12, -0.015], atol=1e-6) and np.isclose(r, 0.009, atol=1e-6)
    assert np.linalg.norm(xy.mean(axis=0) - [0.12, -0.015]) > 0.002


def test_cad_holes_are_the_twenty_table_holes():
    holes = _holes()
    assert len(holes) == 20
    xy = {(round(h.xyz[0], 3), round(h.xyz[1], 3)) for h in holes}
    assert {(0.12, 0.015), (0.12, -0.015), (0.15, 0.015), (0.15, -0.015)} <= xy       # 가운데 은색 볼트 4 개
    assert {(0.255, 0.285), (0.285, 0.315), (0.255, -0.315), (0.285, -0.285)} <= xy   # 빈 검은 구멍
    assert all(np.isclose(h.xyz[2], T.TABLE_TOP_Z) for h in holes)


def _synthetic(holes, Ttrue):
    """회색 상판 위에 와셔(밝음) · 빈 구멍(어두움) 원판을 그린 영상."""
    img = np.full((720, 1280), 40, np.uint8)
    for h in holes:
        uv, z = T.project(Ttrue, K, h.xyz[None])
        r = K[0, 0] * h.radius / z[0]
        bright = h.xyz[0] < 0.2          # 로봇 가까운 줄 = 은색 와셔
        cv2.circle(img, (int(round(uv[0, 0])), int(round(uv[0, 1]))), int(round(r)), 230 if bright else 5, -1,
                   lineType=cv2.LINE_AA)
    return img


def test_solve_recovers_pose_from_offset_initial_guess():
    holes = _holes()
    img = _synthetic(holes, TRUE)
    Test, used, pts, res, shift = T.solve(holes, img, K, INIT)
    assert len(used) >= 18
    assert np.linalg.norm(shift) > 20                      # 초기 투영이 수십 px 어긋났다 — 거친 평행이동이 잡았다
    assert res.mean() < 1.5
    assert np.linalg.norm(Test[:3, 3] - TRUE[:3, 3]) < 0.006
    dR = Test[:3, :3] @ TRUE[:3, :3].T
    assert np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 0.5


def test_joint_refine_keeps_truth_when_normal_agrees():
    holes = _holes()
    obj = np.array([h.xyz for h in holes])
    img, _ = T.project(TRUE, K, obj)
    n_cam = TRUE[:3, :3].T @ np.array([0.0, 0.0, 1.0])     # base +z 를 카메라 프레임으로
    out = T.joint_refine(INIT, obj, img, K, n_cam)
    assert np.linalg.norm(out[:3, 3] - TRUE[:3, 3]) < 1e-3
    assert np.allclose(out[:3, :3] @ n_cam, [0, 0, 1], atol=1e-3)


def test_quat_roundtrip():
    q = np.array([0.130161, -0.69891, 0.696068, -0.100366])
    q /= np.linalg.norm(q)
    assert np.allclose(T.R_to_quat_wxyz(T.quat_wxyz_to_R(q)), q, atol=1e-9)

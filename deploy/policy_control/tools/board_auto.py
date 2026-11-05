"""board_jog 자동 접근의 순수 계산(numpy): 보드 자세 강체 맞춤(Kabsch) · 바깥 4 모서리 검사 · 방문 순서.

11.05 사용자: "최외각 직사각형 4점만 내가 맞춰주고, 나머진 점들은 자동화". 맞춘 자세는 표시 · 다음 접근 위치 예측용이다.
FK 파라미터(관절 영점 · 손끝 오프셋) 풀이는 fit_board.py 몫이고 여기서는 하지 않는다.
"""
from __future__ import annotations

import numpy as np

DIAG_WARN_MM, DIAG_REJECT_MM = 4.0, 10.0   # 4 모서리 대각선 · 네 변 길이(재기 − 인쇄) 차: 경고 · 자동 단계 거부(잘못 댄 점)
TILT_MAX_DEG = 5.0                         # 맞춘 보드 z(인쇄면 → 보드 속)가 base −z 에서 이보다 기울면 거부(거울 순서 · 잘못 댄 점, 11.05 15 → 5)
RES_MAX_MM = 3.0                           # 맞춤 잔차(점마다 |R a + t − b|) 최대가 이보다 크면 거부(4 점 · 전체 모두) — z 이상점은 변 길이로 안 보인다
YAW_MAX_DIFF_DEG = 90.0                    # 맞춘 yaw 와 표시(카메라 · 입력 추정) yaw 차 상한 — 180° 뒤집힌 순서는 강체 맞춤이 그대로 받는다
REFIT_MIN = 8                              # 보드의 사람 기록이 이만큼이면 4 모서리 대신 전체로 맞춘다
PAIRS = ((0, 3), (1, 2), (0, 1), (2, 3), (0, 2), (1, 3))   # 4 모서리 [원점, +x, +y, +x+y]: 대각선 2 · x 변 2 · y 변 2


def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """min Σ|R a + t − b|², det R = +1(반사 금지). 평면 점(보드 z = 0)이어도 det 보정이 한 해를 고른다."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def yaw_deg(T: np.ndarray) -> float:
    """보드 x 축의 base xy 방위 °(board_jog.board_xyyaw 와 같은 식)."""
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def fit_board(P_board: np.ndarray, P_base: np.ndarray, corners: bool = False, yaw_ref: float | None = None) -> dict:
    """보드 좌표 점 ↔ FK 손끝 점(base) → {T, n, res_mm, tilt_deg, diag_mm, edge_mm, ok, why, warn}.
    corners = 점이 바깥 4 모서리 [원점, +x, +y, +x+y] 차례(대각선 0-3 · 1-2, 변 0-1 · 2-3 · 0-2 · 1-3 검사).
    yaw_ref = 표시 자세 yaw °(있으면 맞춘 yaw 와 YAW_MAX_DIFF_DEG 넘게 다르면 거부)."""
    P_board, P_base = np.asarray(P_board, float), np.asarray(P_base, float)
    R, t = kabsch(P_board, P_base)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    res = np.linalg.norm(P_board @ R.T + t - P_base, axis=1) * 1000
    tilt = float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))
    f = {"T": T, "n": len(P_base), "res_mm": res, "tilt_deg": tilt, "diag_mm": None, "edge_mm": None, "ok": True, "why": "", "warn": ""}
    bad: list = []
    if corners:
        dd = np.array([abs(np.linalg.norm(P_base[i] - P_base[j]) - np.linalg.norm(P_board[i] - P_board[j])) for i, j in PAIRS]) * 1000
        f["diag_mm"], f["edge_mm"] = dd[:2], dd[2:]
        what = "대각선" if dd[:2].max() >= dd[2:].max() else "변"
        if dd.max() > DIAG_REJECT_MM:
            bad.append(f"{what} 길이 차 {dd.max():.1f} mm > {DIAG_REJECT_MM:g} — 잘못 댄 모서리를 지우고 다시")
        elif dd.max() > DIAG_WARN_MM:
            f["warn"] = f"{what} 길이 차 {dd.max():.1f} mm > {DIAG_WARN_MM:g}"
    if res.max() > RES_MAX_MM:
        bad.append(f"맞춤 잔차 최대 {res.max():.1f} mm > {RES_MAX_MM:g}(점 {int(res.argmax())}) — 높이 · 위치가 어긋난 점")
    if tilt > TILT_MAX_DEG:
        bad.append(f"보드 면이 {tilt:.0f}° 기울어짐 > {TILT_MAX_DEG:g} — 모서리 순서가 뒤집혔거나 잘못 댄 점")
    if yaw_ref is not None and abs(dy := (yaw_deg(T) - yaw_ref + 180.0) % 360.0 - 180.0) > YAW_MAX_DIFF_DEG:
        bad.append(f"맞춘 yaw 가 표시 자세와 {dy:+.0f}° 다르다 > {YAW_MAX_DIFF_DEG:g} — 모서리 순서가 180° 뒤집혔다")
    if bad:
        f.update(ok=False, why=" · ".join(bad))
    return f


def fit_pose(f: dict) -> dict:
    """맞춤 → 기록 JSON 의 pose_display(boards.*.pose_display · 창 srcs 와 같은 꼴)."""
    s = {"kind": f["kind"], "n": int(f["n"]), "res_mm": np.round(f["res_mm"], 2).tolist(), "tilt_deg": round(f["tilt_deg"], 2),
         "diag_mm": None if f["diag_mm"] is None else np.round(f["diag_mm"], 2).tolist(),
         "edge_mm": None if f.get("edge_mm") is None else np.round(f["edge_mm"], 2).tolist(), "warn": f["warn"]}
    return {"source": f"fit {f['kind']}", "editable": False, "T_base_board": np.round(f["T"], 6).tolist(), "fit": s}


def visit_order(start, P) -> list[int]:
    """start 에서 가장 가까운 점, 그 점에서 가장 가까운 남은 점 … (탐욕 최근접 사슬). 첫 원소 = 지금 갈 점."""
    P = np.asarray(P, float).reshape(-1, 3)
    left, cur, out = list(range(len(P))), np.asarray(start, float), []
    while left:
        k = min(left, key=lambda i: float(np.linalg.norm(P[i] - cur)))
        out.append(k)
        left.remove(k)
        cur = P[k]
    return out

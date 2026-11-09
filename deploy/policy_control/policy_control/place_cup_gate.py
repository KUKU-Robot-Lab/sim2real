"""11.08 놓기 학생 실기: 손 안 컵 YOLO 마스크 고르기(s2r:vision 11.08 확정 규칙의 실시간판).

모델 seg_s640_A(YOLO11s-seg), conf ≥ 0.25, 클래스마다 conf 가장 높은 인스턴스 하나(top1_per_class, head_mask_node.keep_instances).
그 목표 클래스 인스턴스의 마스크 화소 평균이 손바닥 기대 위치 X = p + R·c([lr]_hl_palm_sensor 위치 · 회전 base, 실기 UrdfChainFK(contract palm_body) 와 같은 링크, c = 손바닥 좌표 m)의
투영에서 GATE_PX 보다 멀면 컵 없음. 거리는 원해상도 1280x720 화소(vision FX 908 = 원해상도 K)라 마스크 · K 도 원해상도.
근거: logs/cup_seg_1107/yolo11s_A(gate_new.py · results/offset3d_new_selffit.json). 옛 gate/offset3d.json 의 shaker 값은 bag 이 뒤바뀌어 쓰지 않는다.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

CLASSES = ("blue", "pink", "orange")   # student_mask.CLASSES(seg_s640_A: 0 파랑 source240_blue · 1 분홍 · 2 shaker_c)
CONF_MIN = 0.25
GATE_PX = 80.0
# 손바닥([lr]_hl_palm_sensor) 좌표 고정점 c(m) = 두 bag 자기 적합 평균(offset3d_new_selffit.json) = hdgp 20795489 place_head.CUP_C.
# 키 = 컵 형상 이름(ckpt meta head.cup_c_m 키와 대조용). 분홍 source240_pink 는 실측이 없어 같은 형상 파랑 값을 쓴다.
C_DEFAULT = {"source240": (0.0551, 0.0619, 0.0633), "shaker": (-0.0209, 0.0526, -0.0017)}
CLASS_CUP = {"blue": "source240", "pink": "source240", "orange": "shaker"}


def expected_px(p, R, c, T_base_cam, k) -> np.ndarray | None:
    """손바닥 고정점의 화소 (u, v). 카메라 뒤(z ≤ 1 mm)면 None."""
    X = np.asarray(p, np.float64) + np.asarray(R, np.float64) @ np.asarray(c, np.float64)
    T = np.asarray(T_base_cam, np.float64)
    xc = T[:3, :3].T @ (X - T[:3, 3])
    if xc[2] <= 1e-3:
        return None
    uv = np.asarray(k, np.float64) @ xc
    return uv[:2] / uv[2]


def pick_cup(dets: Sequence[tuple[int, float, np.ndarray]], target: str, p, R, T_base_cam, k, c=None,
             conf_min: float = CONF_MIN, gate_px: float = GATE_PX) -> tuple[np.ndarray | None, float, int]:
    """dets = [(클래스 번호, conf, 마스크 (H,W) bool)](노드 keep_instances 결과 또는 원 인스턴스) · target = CLASSES 이름 →
    (고른 마스크 또는 None, 기대 화소까지 거리 px(후보 없으면 inf), 고른 번호 또는 -1). c 를 안 주면 C_DEFAULT[CLASS_CUP[target]]."""
    ci = CLASSES.index(target)
    c = C_DEFAULT[CLASS_CUP[target]] if c is None else c
    cand = [j for j, (cl, conf, m) in enumerate(dets) if cl == ci and conf >= conf_min and m.any()]
    if not cand:
        return None, float("inf"), -1
    j = max(cand, key=lambda q: dets[q][1])                     # top1_per_class
    e = expected_px(p, R, c, T_base_cam, k)
    if e is None:
        return None, float("inf"), -1
    ys, xs = np.nonzero(dets[j][2])
    d = float(np.hypot(xs.mean() - e[0], ys.mean() - e[1]))
    return (dets[j][2], d, j) if d <= gate_px else (None, d, -1)

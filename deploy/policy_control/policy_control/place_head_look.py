"""11.08 놓기 학생 실기: 목표점을 화면 가운데 두는 목 각(hdgp distillation/student/head.py HeadModel.look_at, 20795489 의 numpy 판).

hdgp 와 같게 둔 것: 처짐 경로(HeadCamPose.cam_pose_from_encoder = HeadModel.cam_pose(ideal=True)), 한계(pan −45~30 ·
tilt −2~82), 가우스-뉴턴 10 회, 유한 차분 eps 0.05°, 감쇠 1e-6, 한 번 이동 ±20° 묶음, 한쪽 축이 한계에 걸려 그 바깥으로
밀리면 그 열을 0 으로 두고 남은 축만 푸는 규칙, 광학 z 하한 1e-3. 시작점 q 는 hdgp 처럼 묶지 않고 그대로 쓴다(첫 걸음 뒤 묶인다).

각 규약 = 드라이버 인코더 deg(tick·360/4095 − 180, head_position_hold_node.tick_to_deg). 시작점은 매 틱 /head/joint_states
(URDF rad)를 head_fk_chain.encoder_from_urdf 로 바꾼 값이다. 출력 [pan, tilt] 도 같은 규약이라 /head/target_stream
{"seq", "pan_deg", "tilt_deg"} 에 그대로 싣는다. 그 값은 모터가 서는 자리(읽힌 Present 각)다: 드라이버가 레지스터에 쓸 때
goal_bias_tick 을 더해(head_home.goal_tick) 펌웨어 궤적 끝 어긋남을 메우므로 여기서 bias 를 빼거나 더하지 않는다.
드라이버는 받은 값을 stream_limits(학생 한계 ∩ 처짐 하한 −10° ∩ --max-tilt, 양수 bias 만큼 상한 축소)로 묶고
60 °/s · 300 °/s² 로 따라간다(head_motion.stream_step).
hdgp 각(tick−2048)·360/4096 과의 차는 camset 자세에서 0.04° 남짓이다. 외부 파라미터 기준 자세 q0 를 같은 드라이버 규약으로
잡았으므로 카메라 자세에는 그 차가 들어가지 않고, 한계 −45/30/−2/82 만 그만큼 어긋난다.
"""
from __future__ import annotations

import numpy as np

from .place_head_pose import HeadCamPose

PAN_LIM = (-45.0, 30.0)     # hdgp HeadCfg.pan_lim = head_motion.STREAM_PAN_RANGE_DEG
TILT_LIM = (-2.0, 82.0)     # hdgp HeadCfg.tilt_lim = head_motion.STREAM_TILT_RANGE_DEG
ITERS = 10
EPS_DEG = 0.05
LAM = 1e-6
STEP_MAX_DEG = 20.0
STUCK_TOL_DEG = 1e-4
Z_MIN = 1e-3


def _proj(hc: HeadCamPose, q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """광학 프레임 정규 좌표 (x/z, y/z). 0 이면 광학축 위."""
    T = hc.cam_pose_from_encoder(float(q[0]), float(q[1]))
    p = T[:3, :3].T @ (target - T[:3, 3])
    return p[:2] / max(p[2], Z_MIN)


def look_at(hc: HeadCamPose, target_base, q_start_enc_deg, iters: int = ITERS,
            lo=(PAN_LIM[0], TILT_LIM[0]), hi=(PAN_LIM[1], TILT_LIM[1])) -> np.ndarray:
    """base 좌표 점 → 그 점을 화면 가운데 두는 인코더 deg [pan, tilt](드라이버 규약, 한계 안).

    q_start_enc_deg: 지금 목 각(인코더 deg). hdgp 는 HeadModel.q(지연 · 속도 제한을 거친 실제 각)에서 시작한다.
    """
    target = np.asarray(target_base, np.float64)
    q = np.asarray(q_start_enc_deg, np.float64).copy()
    if target.shape != (3,) or q.shape != (2,) or not (np.isfinite(target).all() and np.isfinite(q).all()):
        raise ValueError(f"look_at 입력이 잘못됐다: target {target_base} q {q_start_enc_deg}")
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    e = np.eye(2) * EPS_DEG
    for _ in range(iters):
        r = _proj(hc, q, target)
        J = np.stack([(_proj(hc, q + e[0], target) - r) / EPS_DEG, (_proj(hc, q + e[1], target) - r) / EPS_DEG], -1)
        g = J.T @ -r                                                   # 내리막 방향(각 공간)
        stuck = ((q <= lo + STUCK_TOL_DEG) & (g < 0)) | ((q >= hi - STUCK_TOL_DEG) & (g > 0))
        J = J * ~stuck
        step = np.linalg.solve(J.T @ J + LAM * np.eye(2), J.T @ -r)
        q = np.minimum(np.maximum(q + np.clip(step, -STEP_MAX_DEG, STEP_MAX_DEG), lo), hi)
    return q

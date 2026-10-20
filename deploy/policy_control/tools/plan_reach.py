#!/usr/bin/env python3
"""1-B 접근 — aglt_new 사전 파지까지 ready_carry 로 옮긴 뒤, 편 손 그대로 tilt 기준 손 안 컵 자리까지 짧게 들어간다.
테이블 · 자기 충돌 · 반대팔 · 컵 · 관절 한계를 지키는 후보만 쓴다.

10.19 사용자 "1-B: READY CARRY 로 컵으로 접근 + 스크립트 파지/리프트(파지는 사용자 승인으로) · 왼손/오른손 독립 실행".
사용자 결정(10.19): 쥐는 자리 = tilt 기준 자리(single734 행 0 의 손 안 컵 자리) · 승인 = 닫기 전만.
10.20 사용자 "컵 접근은 ready-carry 세팅을 aglt_new 세션에 물어봐 여기는 금방하던데" → 사전 파지 · 이송을 aglt_new 와 같게.
10.20 사용자 "1-b 세팅은 ready-carry 에 좀더 조건이 많아야함. 테이블, 자기 충돌, 관절 제약 등을 고려해야함" →
  aglt_new 격자에서 컵이 중심선 가까우면(|y| < 약 0.08) Mover 이동 중 손목 j6 · j2 가 한계에 붙었다(clamp). 그래서 후보를 돌며
  조건을 다 지키는 것을 고른다. 움직이지 않는다(관절 실측 구독만). 재생은 에피소드 실행기가 replay_to_pd 로(그 팔 7 관절).

  컵: FP++ 원점(몸통 가운데). 서 있어야 한다(기울기 ≤ 10°).
  사전 파지(hdgp tasks/rh_aglt_r/ready_approach.pregrasp_pose 그대로, 값 = config/ready_approach/rh56f1.yaml): 손바닥 = 컵 축에서
    2.5 R · 컵 원점 높이 · 손 방향 컵 틀 고정 · 방위 = 회귀식 + 후보 0, ±10, ±20, ±30°(작은 것부터).
  후보마다:
    1) 사전 파지 6D IK(ready_carry posture, 관절 여유 ≥ 10°).
    2) 들어가기: 사전 파지 → tilt 기준 쥔 자세(컵 축 둘레 방위 = 사전 파지 방향에 가장 가까운 것) — 위치 직선 · 방향 구면 보간 ·
       최소 저크(손바닥 2 cm/s · 회전 15°/s · 관절 0.38 rad/s 평균 중 느린 쪽). 컵과는 관통만 금지 + 손 점 3 mm.
    3) 이송: ready_carry Mover(aglt_new 와 같은 경유점 바깥 4 · 위 4 cm · 24 cm/s · 0.72 rad/s) → 걸리면 지금 → 경유점 관절 공간
       직선 → 그래도 걸리면 RRT(홈 경로 계획기 그대로, 관절 여유 6° 안쪽 표본), 둘 다 경유점에서 사전 파지까지는 곧게(4 cm/s).
       컵 원기둥과 1 cm.
  검사(reach_world · Judge — 모든 토막 같은 기준): 충돌 = 홈 경로 세계(URDF 볼록 조각 · 테이블 USD 상자 · 뒤 박스 · 옆 벽 ·
    몸통 · 머리 · 반대팔 실측 자세) 여유 2 cm · 같은 팔 사슬 이웃 관절은 관통만 금지 · 관절 여유 ≥ 5°(URDF · 프로필 중 좁은 쪽) ·
    60 Hz 관절 속도 ≤ 2 rad/s · 실측 탁자 높이 위 손 점 1.5 cm · 팔 관절 3 cm · 손 점 ↔ 컵 겉면.
  하나도 없으면 종료 코드 2(재생 금지) — check.json tries 에 후보마다 걸린 이유.

    python3 deploy/policy_control/tools/plan_reach.py --side right --object source240b_blue --cup 0.244,-0.116,0.288 \\
        --out logs/x/reach
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

# 실기 계획은 로봇 PC(로컬 5090)에서 돈다(10.20 사용자). torch 스레드는 그대로 두고 우선순위만 낮춰
# 팔 · 손 노드가 먼저 돌게 한다(빈 코어가 있으면 계획 속도는 같다).
try:
    __import__("os").nice(5)
except OSError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_carry import cup_in_palm  # noqa: E402
from plan_tilt import ARMS, ASSET, H, SIM2REAL, current_q, table_z  # noqa: E402

RATE_HZ = 60.0
CUP_CENTER_M = 0.085            # tilt 자산 컵 원점 = 바닥 위 85 mm(plan_carry.CUP_CENTER_M — D4 README, 두 컵 같음)
DEEP_SPEED_M_S = 0.02          # 사전 파지 → 쥔 자리 평균 속도(최소 저크 최대 1.875 배)
VIA_SPEED_M_S = 0.04           # 경유점 → 사전 파지(관절 공간 · RRT 이송일 때) 평균 속도
TRANSPORT_QD_RAD_S = 0.5       # 관절 공간 · RRT 이송 관절 최고 속도(pd ramp_speed 상한 · 홈 경로와 같은 시간 매개)
TRANSPORT_RAMP_S = 0.5
PHI_OFFSETS_DEG = (0, 10, -10, 20, -20, 30, -30)   # 사전 파지 방위 후보(aglt_new 회귀식 기준, 작은 것부터)
#: 사전 파지 높이 후보(컵 원점 위, aglt_new 기본 0 = 10.19 사용자 "손 중심 높이 = 병 원점"). 0 에서 되는 방위가 없을 때만 올린다 —
#: 10.20 왼손 shaker(원점 6.5 cm): 높이 0 이면 편 손 새끼 손끝 충돌 형상이 상판 13.5 mm 위(여유 2 cm 미달).
PRE_H_M = (0.0, 0.01, 0.02, 0.03)
PRE_MARGIN_DEG = 10.0          # 사전 파지 · 경유점 IK 의 관절 여유 하한(경로 하한 5° 보다 넉넉히 — 들어가기 · 이송이 거기서 시작 · 끝난다)
APPROACH_CUP_MARGIN_M = 0.01   # 이송 중 팔 · 손 충돌 형상 ↔ 컵 원기둥(FP++ 오차 1 cm)
DEEP_CUP_MARGIN_M = 0.0        # 들어가기: 손가락이 컵 옆으로 들어온다 — 관통만 금지(손 점 3 mm 는 따로)
DEEP_ROT_DEG_S = 15.0          # 그 사이 손바닥 방향 바꿈 평균 속도
DEEP_QD_RAD_S = 0.38           # 그 사이 관절 평균 속도(최대 0.72 = Mover qd_max 와 같다)
LINE_STEP_M = 0.0005           # 직선 IK 칸(손바닥 이동)
LINE_STEP_DEG = 0.25           # 직선 IK 칸(방향)
LINE_ANG_DEV_DEG = 2.0         # 60 Hz 직선 경로의 손바닥 방향 ↔ 그 진행 위치의 목표 방향
DEEP_ROT_MAX_DEG = 30.0        # 사전 파지 방향과 tilt 기준 쥔 방향이 이보다 다르면 거부(손목을 크게 돌리며 컵 옆)
CUP_TILT_MAX_DEG = 10.0
POS_TOL_M = 0.002
ANG_TOL_DEG = 1.0
MARGIN_MIN_DEG = 5.0
VMAX_RAD_S = 2.0
LINE_DEV_M = 0.003
HAND_TABLE_M = 0.015
ARM_TABLE_M = 0.03
APPROACH_CUP_CLEAR_M = 0.01   # 이송 중 손 점 ↔ 컵 겉면(FP++ 오차 1 cm). 사전 파지 끝은 손가락이 컵 옆이라 2 cm 는 못 둔다
IN_CUP_CLEAR_M = 0.003
#: ★10.20 실기 1-B: 쥔 자리 깊이 — 손바닥(센서점) ↔ 컵 겉면 수평 거리 상한. tilt 기준 쥔 자리는 손바닥을 컵 축에서
#: 늘 52 mm 에 둬서 반지름이 작은 src240(30 mm)은 겉면까지 22 mm 가 남아 손끝으로만 닿았고 닫을 때 컵이 밀렸다.
#: shaker(35.4 mm)는 15.7 mm 에서 깊숙이 쥐고 들었다(오른손 src240 · 왼손 shaker, 10.20 03:25). 그 깊이를 모든 컵에 쓴다 —
#: 더 멀면 손바닥 → 컵 축 수평 방향으로 그만큼 더 들어간다(가까우면 그대로).
GRASP_PALM_GAP_M = 0.016
LINE_ITERS = 12
HAND_CONTRACT = "config/hand/rh56f1_{side}_hand.json"   # ★10.20 1-B 손 계약 고정본(옛 접근 정책 rg3wc 계약 그대로)
READY_CFG = "deploy/policy_control/config/ready_approach/rh56f1.yaml"
READY_APPROACH_REL = "source/openarm/openarm/agnostic/tasks/rh_aglt_r/ready_approach.py"


class ReachError(RuntimeError):
    pass


# ============================================================================ 순수
def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_tilt_deg(q_wxyz) -> float:
    """wxyz → 물체 +z 축과 세계 +z 사이 각."""
    w, x, y, z = (float(v) for v in np.asarray(q_wxyz, float) / np.linalg.norm(q_wxyz))
    uz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, uz))))


def cup_origin_d4(p_fpp, origin_above_bottom_m: float) -> np.ndarray:
    """FP++ 원점(바닥 위 origin_above_bottom_m) → tilt 자산 컵 원점(바닥 위 CUP_CENTER_M) — 바닥을 맞춘다(컵은 서 있다)."""
    p = np.asarray(p_fpp, float).copy()
    p[2] += CUP_CENTER_M - float(origin_above_bottom_m)
    return p


def palm_targets(T_to: np.ndarray, cup: np.ndarray, azimuth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """컵 원점 cup(세계, 축 = +z)에서 손가락 방향(수평)이 azimuth 인 쥔 손바닥 자세 → (R, p, 손가락 수평 단위벡터)."""
    Pc = np.linalg.inv(T_to)                         # 컵 좌표계의 손바닥
    f = Pc[:3, :3] @ np.array([0.0, 1.0, 0.0])        # 손가락 방향(손바닥 +y)
    a0 = math.atan2(f[1], f[0])
    Rz = rot_z(azimuth - a0)
    R = Rz @ Pc[:3, :3]
    p = cup + Rz @ Pc[:3, 3]
    a = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    return R, p, a


def minjerk(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def cup_surface_dist(points: np.ndarray, cup_xy, z_lo: float, z_hi: float, radius: float) -> np.ndarray:
    """점 (…,3) → 선 원기둥 컵(축 +z, 바닥 z_lo · 입구 z_hi) 겉면까지 거리(안이면 음수)."""
    pts = np.asarray(points, float)
    r = np.linalg.norm(pts[..., :2] - np.asarray(cup_xy, float)[:2], axis=-1) - radius
    z = pts[..., 2]
    dz = np.maximum(z_lo - z, z - z_hi)
    out = np.where(dz <= 0, r, np.sqrt(np.maximum(r, 0.0) ** 2 + dz ** 2))
    inside = (dz <= 0) & (r < 0)
    return np.where(inside, np.maximum(r, dz), out)


def line_deviation(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """점들이 a → b 선분에서 벗어난 최대 거리."""
    d = b - a
    t = np.clip(((points - a) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(points - (a + t[:, None] * d), axis=1).max())


# ============================================================================ IK
def ik_pose(P, ch, q0: torch.Tensor, p: np.ndarray, R: np.ndarray, iters: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """손바닥 자세 IK(위치 + 방향) — (q (B,7), 위치 오차 m (B,), 방향 오차 deg (B,))."""
    B = q0.shape[0]
    eye = torch.eye(4, dtype=q0.dtype)[None].expand(B, 4, 4)
    pt = torch.as_tensor(np.broadcast_to(p, (B, 3)).copy(), dtype=q0.dtype)
    Rt = torch.as_tensor(np.broadcast_to(R, (B, 3, 3)).copy(), dtype=q0.dtype)
    q, ep, ea = P.solve(ch, q0, eye, None, pt, Rt, P.IkCfg(iters=iters), mode="pose", iters=iters)
    T = ch.fk(q)
    ep = (T[:, :3, 3] - pt).norm(dim=-1)
    Re = Rt @ T[:, :3, :3].transpose(1, 2)
    ang = torch.acos(((Re.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1))
    return q, ep.double().numpy(), np.degrees(ang.double().numpy())


def margin_deg(ch, q: torch.Tensor) -> np.ndarray:
    return np.degrees(ch.margin(q).amin(-1).double().numpy())


# ============================================================================ 손 · 팔 점 검사
class HandPoints:
    """편 손 손바닥 + 손끝 5 개(세계) — 배포 계약 FK(fk_numpy)."""

    def __init__(self, contract: Path):
        from policy_control import rh_aglt as A
        from policy_control.contract_assets import ASSETS
        from policy_control.fk_numpy import UrdfChainFK
        c = A.load_contract(str(contract))
        s = c.sides[next(iter(c.sides))]
        self.fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
        self.hand_open = np.asarray(s.hand_open, float)
        self.names = ["palm", *s.tip_bodies]

    def __call__(self, path: np.ndarray) -> np.ndarray:
        out = []
        for q in path:
            P = self.fk.palm_pose(np.asarray(q, float), self.hand_open)
            out.append(np.vstack([P.palm_pos[None], P.tips]))
        return np.array(out)                                  # (N, 6, 3)


def arm_points(ch, path: np.ndarray) -> np.ndarray:
    _T, _ax, org = ch.fk(torch.tensor(path, dtype=torch.float32), frames=True)
    return org[:, 3:7].double().numpy()                       # 관절 4~7 원점


# ============================================================================ ReadyApproach(hdgp, aglt_new)
def load_ready_approach(hdgp: Path, RC):
    """hdgp ready_approach.py 를 md5 확인 뒤 읽는다 → (모듈, 배포 설정 dict).

    그 파일은 `openarm.agnostic.modules.ready_carry` 이름으로 Mover · posture 를 import 한다. openarm 패키지 초기화
    (isaaclab 과제 설정)를 거치지 않게, 이미 핀 확인으로 읽은 ready_carry(H.READY_CARRY_PKG)를 그 이름으로도 올려 둔다."""
    cfg = yaml.safe_load((SIM2REAL / READY_CFG).read_text(encoding="utf-8"))
    H.verify(hdgp, cfg["hdgp_md5"])
    alias = "openarm.agnostic.modules.ready_carry"
    for sub in ("", ".chain", ".posture", ".mover"):
        sys.modules.setdefault(alias + sub, sys.modules[H.READY_CARRY_PKG + sub])
    name = "ready_approach_hdgp"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, hdgp / READY_APPROACH_REL)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name], cfg


def ready_params(RA, cfg: dict, side: str):
    return RA.ReadyParams.from_cfg(SimpleNamespace(**cfg[side]))


def _t(v) -> torch.Tensor:
    return torch.tensor(np.asarray(v, float)[None], dtype=torch.float32)


def pregrasp_target(RA, p, cup_fpp: np.ndarray, radius: float, phi: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """aglt_new 사전 파지 손바닥 자세(ready_approach.pregrasp_pose 그대로, 높이 h) → (위치, 회전, 바깥 방향 e_r)."""
    pos, R = RA.pregrasp_pose(_t(cup_fpp), _t(radius), _t(p.r_mult[0]), _t(h), _t(phi), p.n_cup, p.f_cup)
    return pos[0].double().numpy(), R[0].double().numpy(), np.array([math.cos(phi), math.sin(phi), 0.0])


def solve_pose(RC, ch, q_now: np.ndarray, pos: np.ndarray, R: np.ndarray, n_seeds: int, gen) -> tuple[np.ndarray, bool, float]:
    """손바닥 6D 자세 IK(ready_carry posture.plan_posture — 시드 = 지금 + 무작위, 관절 여유 큰 해 · 지금에 가까운 해)."""
    AP = RC.posture
    q0 = _t(q_now)
    rnd = ch.lo + (ch.hi - ch.lo) * (0.1 + 0.8 * torch.rand(1, max(1, n_seeds) - 1, ch.n, generator=gen))
    seeds = torch.cat([q0[:, None], rnd], dim=1)
    z1 = torch.zeros(1, 3)
    pl = AP.plan_posture(ch, seeds, torch.eye(4)[None], AP.Region(_t(pos), _t(R), z1, z1, mode="pose"), q_ref=q0)
    return pl.q[0].double().numpy(), bool(pl.ok[0]), math.degrees(float(pl.margin[0]))


def roll_mover(RC, p, ch, q_now: np.ndarray, q_goal: np.ndarray, pos: np.ndarray, R: np.ndarray,
               via: np.ndarray) -> tuple[np.ndarray, dict]:
    """ready_carry Mover(aglt_new ReadyApproach 와 같은 설정 · 경유점)로 q_goal 까지 굴린다(명령 = 실측, 지연 없음)."""
    M = RC.mover
    mv = M.Mover(ch, M.MoverCfg(v_max=p.v_max, qd_max=p.qd_max, max_steps=p.max_steps, seeds=p.plan_seeds), 1, "cpu")
    q = _t(q_now)
    mv.start(torch.tensor([0]), q, torch.eye(4)[None], None, _t(pos), _t(R)[:, :, 2], None,
             via=_t(via)[:, None], q_goal=_t(q_goal))
    rows, arrived = [q[0].clone()], False
    for _ in range(p.max_steps + 60):
        on = bool(mv.on[0])
        dq, _on, done, arrive = mv.step(q)
        if on:
            q = torch.minimum(torch.maximum(q + dq.to(q.dtype), ch.lo), ch.hi)
            rows.append(q[0].clone())
        if bool(done[0]) or not on:
            arrived = bool(arrive[0]) or arrived
            break
    path = torch.stack(rows).double().numpy()
    return path, {"plan_ok": bool(mv.ok[0]), "arrived": arrived, "end_err_mm": round(float(mv.last_err[0]) * 1000, 1)}


def line_path(P, ch, q_a: np.ndarray, p_b: np.ndarray, R_b: np.ndarray, v_m_s: float, w_deg_s: float,
              qd_rad_s: float) -> tuple[np.ndarray, dict]:
    """지금 손바닥 자세(q_a 의 FK) → (p_b, R_b): 위치 직선 · 방향 구면 보간을 0.5 mm 칸으로 IK(앞 칸 시드).
    시간: 칸마다 손바닥 이동 / v · 회전 / w · 최대 관절 변화 / qd 중 큰 것을 쌓아 총 시간으로 최소 저크(각 평균, 최대 1.875 배).
    10.20: 왼손 shaker 칸은 사전 파지에서 팔꿈치가 거의 펴져(j4 0.22 rad) 손바닥만으로 시간을 정하면 4.3 rad/s 였다."""
    T_a = ch.fk(_t(q_a))[0].double().numpy()
    R_a, p_a = T_a[:3, :3], T_a[:3, 3]
    rv = Rotation.from_matrix(R_a.T @ R_b).as_rotvec()
    ang, dist = float(np.linalg.norm(rv)), float(np.linalg.norm(p_b - p_a))
    m = max(2, int(math.ceil(max(dist / LINE_STEP_M, math.degrees(ang) / LINE_STEP_DEG)))) + 1
    s = np.linspace(0.0, 1.0, m)
    q = _t(q_a)
    nodes, e_p, e_a = [q[0].clone()], 0.0, 0.0
    for k in range(1, m):
        q, ep, ea = ik_pose(P, ch, q, p_a + s[k] * (p_b - p_a), R_a @ Rotation.from_rotvec(s[k] * rv).as_matrix(), LINE_ITERS)
        e_p, e_a = max(e_p, float(ep[0])), max(e_a, float(ea[0]))
        nodes.append(q[0].clone())
    Q = torch.stack(nodes).double().numpy()
    dq = np.abs(np.diff(Q, axis=0)).max(1)
    dsig = np.maximum.reduce([np.full(m - 1, dist / (m - 1) / v_m_s), np.full(m - 1, math.degrees(ang) / (m - 1) / w_deg_s),
                              dq / qd_rad_s])
    sig = np.concatenate([[0.0], np.cumsum(dsig)])
    n = max(2, int(math.ceil(float(sig[-1]) * RATE_HZ)) + 1)
    u = minjerk(n) * sig[-1]
    path = np.stack([np.interp(u, sig, Q[:, j]) for j in range(Q.shape[1])], axis=1)
    # 60 Hz 경로 그대로 FK 해 목표(같은 진행 위치)와 대조 — 칸 사이 관절 보간이 손을 선에서 떼지 않는지
    # (10.20 왼손 shaker: 첫 칸에서 j4 0.221 → 0.297 — 손바닥은 0.45 mm 안에 그대로, 7 축 여유 자세로 고쳐 잡는 것)
    Tp = ch.fk(torch.tensor(path, dtype=torch.float32)).double().numpy()
    sk = np.interp(u, sig, s)
    tgt_p = p_a[None] + sk[:, None] * (p_b - p_a)[None]
    tgt_R = R_a[None] @ Rotation.from_rotvec(sk[:, None] * rv[None]).as_matrix()
    ang_dev = np.linalg.norm(Rotation.from_matrix(np.transpose(tgt_R, (0, 2, 1)) @ Tp[:, :3, :3]).as_rotvec(), axis=1)
    info = {"n": n, "T_s": round((n - 1) / RATE_HZ, 2), "dist_mm": round(dist * 1000, 1), "rot_deg": round(math.degrees(ang), 1),
            "ik_nodes": m, "max_node_jump_rad": round(float(dq.max()), 4),
            "max_pos_err_mm": round(e_p * 1000, 2), "max_ang_err_deg": round(e_a, 2),
            "path_pos_dev_mm": round(float(np.linalg.norm(Tp[:, :3, 3] - tgt_p, axis=1).max()) * 1000, 2),
            "path_ang_dev_deg": round(math.degrees(float(ang_dev.max())), 2),
            "line_dev_mm": round(line_deviation(Tp[:, :3, 3], p_a, p_b) * 1000, 2),
            "vmax_rad_s": round(float(np.abs(np.diff(path, axis=0)).max()) * RATE_HZ, 3)}
    info["line_ok"] = bool(info["max_pos_err_mm"] <= POS_TOL_M * 1000 and info["max_ang_err_deg"] <= ANG_TOL_DEG
                           and info["path_pos_dev_mm"] <= LINE_DEV_M * 1000 and info["path_ang_dev_deg"] <= LINE_ANG_DEV_DEG)
    return path, info


def grasp_target(ch, q_pre: np.ndarray, T_to: np.ndarray, cup_d4: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """tilt 기준 쥔 손바닥 자세 — 컵 축 둘레 방위는 사전 파지 손 방향에 가장 가까운 것 → (R, p, 방위 deg, 방향 차 deg)."""
    T_pre = ch.fk(_t(q_pre))[0].double().numpy()
    best = None
    for az in np.radians(np.arange(-180.0, 180.0, 1.0)):
        R_g, p_g, _ = palm_targets(T_to, cup_d4, float(az))
        ang = float(np.linalg.norm(Rotation.from_matrix(T_pre[:3, :3].T @ R_g).as_rotvec()))
        if best is None or ang < best[0]:
            best = (ang, R_g, p_g, float(az))
    return best[1], best[2], math.degrees(best[3]), math.degrees(best[0])


def deepen(p_g: np.ndarray, cup: np.ndarray, radius: float, gap: float = GRASP_PALM_GAP_M) -> tuple[np.ndarray, float]:
    """쥔 손바닥 위치 → 손바닥 ↔ 컵 겉면 수평 거리가 gap 을 넘으면 그만큼 컵 축 쪽(수평)으로 옮긴 위치, 옮긴 거리. 순수."""
    v = np.asarray(cup, float)[:2] - np.asarray(p_g, float)[:2]
    d = float(np.linalg.norm(v))
    extra = d - radius - gap
    if d < 1e-9 or extra <= 0.0:
        return np.asarray(p_g, float).copy(), 0.0
    out = np.asarray(p_g, float).copy()
    out[:2] += v / d * extra
    return out, extra


def time_joint(path: np.ndarray) -> np.ndarray:
    """관절 공간 꼭짓점 경로 → 60 Hz(홈 경로와 같은 코사인 가감속, L-inf 호 길이)."""
    from plan_home_path import time_parametrize
    return time_parametrize(np.asarray(path, float), TRANSPORT_QD_RAD_S, dt=1.0 / RATE_HZ, ramp=TRANSPORT_RAMP_S)


# ============================================================================ 검사
class Judge:
    """경로 한 토막의 검사(충돌 · 관절 여유 · 속도 · 실측 탁자 높이) — 같은 기준을 이송 · 들어가기에 쓴다."""

    def __init__(self, ch, rw, hand, tz: float, cup: np.ndarray, z_lo: float, z_hi: float, radius: float):
        self.ch, self.rw, self.hand, self.tz = ch, rw, hand, tz
        self.cup, self.z_lo, self.z_hi, self.radius = cup, z_lo, z_hi, radius

    def __call__(self, path: np.ndarray, cup_margin_m: float, pt_cup_clear_m: float) -> dict:
        path = np.asarray(path, float)
        mc = np.degrees(self.ch.margin(torch.tensor(path, dtype=torch.float32)).double().numpy())
        mj = np.minimum(mc, self.rw.profile_margin_deg(path)).min(0)
        pts = self.hand(path[::2])
        c = {"collision": self.rw.check(path, cup_margin_m),
             "margin_min_deg": round(float(mj.min()), 2), "margin_min_joint": int(mj.argmin()) + 1,
             "hand_table_mm": round(float(pts[..., 2].min() - self.tz) * 1000, 1),
             "arm_table_mm": round(float(arm_points(self.ch, path[::2])[..., 2].min() - self.tz) * 1000, 1),
             "cup_clear_mm": round(float(cup_surface_dist(pts, self.cup, self.z_lo, self.z_hi, self.radius).min()) * 1000, 1),
             "vmax_rad_s": round(float(np.abs(np.diff(path, axis=0)).max()) * RATE_HZ, 3) if len(path) > 1 else 0.0}
        c["fails"] = [x for x in (
            None if c["collision"]["ok"] else "충돌 " + json.dumps(c["collision"].get("worst"), ensure_ascii=False),
            None if c["margin_min_deg"] >= MARGIN_MIN_DEG else f"관절 j{c['margin_min_joint']} 여유 {c['margin_min_deg']}° < {MARGIN_MIN_DEG:.0f}°",
            None if c["hand_table_mm"] >= HAND_TABLE_M * 1000 else f"손 점 탁자 위 {c['hand_table_mm']} mm",
            None if c["arm_table_mm"] >= ARM_TABLE_M * 1000 else f"팔 관절 탁자 위 {c['arm_table_mm']} mm",
            None if c["cup_clear_mm"] >= pt_cup_clear_m * 1000 else f"손 점 ↔ 컵 {c['cup_clear_mm']} mm < {pt_cup_clear_m * 1000:.0f}",
            None if c["vmax_rad_s"] <= VMAX_RAD_S else f"관절 속도 {c['vmax_rad_s']} rad/s") if x]
        c["ok"] = not c["fails"]
        return c


# ============================================================================ 계획
def plan(side: str, cup_fpp, cup_quat, obj: str, out: Path, q: dict, hdgp: Path, **kw) -> tuple[bool, dict]:
    """계획 · 검사 → (ok, meta). 거부여도 out/check.json 을 남긴다(무엇이 걸렸는지). reach_cmd.npz 는 ok 일 때만."""
    t0 = time.monotonic()
    ok, meta, full = _plan(side, cup_fpp, cup_quat, obj, q, hdgp, **kw)
    meta["plan_wall_s"] = round(time.monotonic() - t0, 2)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reach_cmd.npz").unlink(missing_ok=True)
    if ok:
        np.savez_compressed(out / "reach_cmd.npz", arm_target=full, joint_names=np.array(ARMS[side]),
                            meta_step_dt=np.array(1.0 / RATE_HZ), n_carry=np.array(meta["approach"]["steps"] + 1))
    (out / "check.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return ok, meta


def _plan(side: str, cup_fpp, cup_quat, obj: str, q: dict, hdgp: Path, *, asset: Path = ASSET,
          contract: Path | None = None):
    tm, tk = {}, [time.monotonic()]

    def lap(k):
        now = time.monotonic()
        tm[k] = round(tm.get(k, 0.0) + now - tk[0], 3)
        tk[0] = now

    sys.path.insert(0, str(SIM2REAL / "scripts"))
    from object_registry import load_registry
    from reach_world import ReachWorld
    cfg = H.load_policy(asset)
    H.verify(hdgp, cfg["hdgp"]["md5"])
    H.verify_refs(asset, cfg.get("ref") or {})
    RC = H.load_ready_carry(hdgp)
    RA, rcfg = load_ready_approach(hdgp, RC)
    P = RC.posture
    spec = load_registry().get(obj)
    radius = float(max(abs(v) for v in (*spec.aabb[0][:2], *spec.aabb[1][:2])))
    oab, top = float(spec.origin_above_bottom_m), float(spec.aabb[1][2])
    tz = table_z(cfg)
    ref = dict(np.load(asset / "ref" / "single734_t120.npz", allow_pickle=False))
    ch = RC.chain.Chain.from_urdf(RC.chain.default_urdf(), f"{side[0]}_hl_palm_sensor", ARMS[side])
    T_to = cup_in_palm(RC, ch, ref, side)[0].double().numpy()
    hand = HandPoints(contract or SIM2REAL / HAND_CONTRACT.format(side=side))
    q_now = np.asarray(q[side], float)
    other = "left" if side == "right" else "right"

    cup_fpp = np.asarray(cup_fpp, float)
    tilt = quat_tilt_deg(cup_quat) if cup_quat is not None else None
    cup = cup_origin_d4(cup_fpp, oab)
    z_lo, z_hi = cup_fpp[2] - oab, cup_fpp[2] + top
    meta = {"side": side, "object": obj, "cup_fpp": [round(float(v), 4) for v in cup_fpp],
            "cup_tilt_deg": None if tilt is None else round(tilt, 1), "cup_origin_d4": [round(float(v), 4) for v in cup],
            "cup_radius_m": radius, "cup_z_lo": round(z_lo, 4), "cup_z_hi": round(z_hi, 4), "table_z": tz,
            "q_start": [round(float(v), 4) for v in q_now], "hdgp": str(hdgp), "hdgp_commit_pin": cfg["hdgp"].get("commit"),
            "ready_cfg": {"file": READY_CFG, "source": rcfg.get("source")}, "problems": [], "timing_s": tm}
    lap("load")
    if tilt is not None and tilt > CUP_TILT_MAX_DEG:
        meta["problems"].append(f"컵이 {tilt:.1f}° 기울었다(한계 {CUP_TILT_MAX_DEG:.0f}°) — 세워 둘 것")
        meta["ok"] = False
        return False, meta, None
    if abs(z_lo - tz) > 0.03:
        meta["problems"].append(f"컵 바닥 z {z_lo:.3f} 이 탁자 {tz:.3f} 에서 {abs(z_lo - tz) * 100:.1f} cm — FP++ 자세 확인")
    rw = ReachWorld(side, hdgp, q_now, q[other], {"x": float(cup_fpp[0]), "y": float(cup_fpp[1]), "z_lo": float(z_lo),
                                                  "z_hi": float(z_hi), "radius": radius})
    start = rw.chk.check(q_now)
    meta["start_collision"] = {"ok": bool(start.ok), "worst": None if start.ok else
                               {"pair": rw_pair(start.worst_pair), "dist_mm": round(float(start.worst_dist) * 1000, 1)}}
    judge = Judge(ch, rw, hand, tz, cup_fpp, z_lo, z_hi, radius)
    lap("world")

    # 사전 파지 방위 후보: aglt_new 회귀식 그대로부터 ±10 · 20 · 30°. 후보마다
    #   쥔 자리 · 사전 파지 IK(관절 여유 ≥ PRE_MARGIN_DEG) → 들어가기(사전 파지 → 쥔 자리) 검사 →
    #   이송 ready_carry Mover(aglt_new 와 같은 경유점 · 속도) → 걸리면 관절 공간 직선 → RRT(둘 다 경유점까지, 거기서 곧게 사전 파지로).
    p = ready_params(RA, rcfg, side)
    phi0 = math.radians(p.phi_coef[0] + p.phi_coef[1] * cup_fpp[0] + p.phi_coef[2] * cup_fpp[1])
    gen = torch.Generator().manual_seed(0)
    tries, chosen = [], None
    for h, off in ((h, off) for h in PRE_H_M for off in PHI_OFFSETS_DEG):
        if chosen is not None:
            break
        tr = {"h_m": h, "phi_offset_deg": off, "phi_deg": round(math.degrees(phi0) + off, 1)}
        tries.append(tr)
        pos, R, er = pregrasp_target(RA, p, cup_fpp, radius, phi0 + math.radians(off), h)
        q_pre, ok_pre, m_pre = solve_pose(RC, ch, q_now, pos, R, p.plan_seeds, gen)
        m_pre = min(m_pre, float(rw.profile_margin_deg(q_pre[None]).min()))
        tr["pregrasp"] = {"ik_ok": ok_pre, "margin_deg": round(m_pre, 1)}
        lap("pregrasp_ik")
        if not ok_pre or m_pre < PRE_MARGIN_DEG:
            tr["why"] = f"사전 파지 IK {'실패' if not ok_pre else f'관절 여유 {m_pre:.1f}° < {PRE_MARGIN_DEG:.0f}°'}"
            continue
        via = pos + p.via_out_m * er + np.array([0.0, 0.0, p.via_up_m])
        methods = []
        appr, ainfo = roll_mover(RC, p, ch, q_now, q_pre, pos, R, via)
        lap("mover")
        aj = judge(appr, APPROACH_CUP_MARGIN_M, APPROACH_CUP_CLEAR_M)
        a_fails = aj["fails"] + ([] if ainfo["arrived"] else ["Mover 가 사전 파지에 못 닿았다"])
        methods.append({"method": "ready_carry", "ok": not a_fails, "fails": a_fails, "T_s": round((len(appr) - 1) / RATE_HZ, 2)})
        lap("check")
        if a_fails:
            q_via, ok_v, m_v = solve_pose(RC, ch, q_now, via, R, p.plan_seeds, gen)
            m_v = min(m_v, float(rw.profile_margin_deg(q_via[None]).min()))
            lap("pregrasp_ik")
            seg2, s2 = line_path(P, ch, q_via, pos, R, VIA_SPEED_M_S, DEEP_ROT_DEG_S, DEEP_QD_RAD_S)
            s2_ok = ok_v and m_v >= PRE_MARGIN_DEG and s2["line_ok"]
            lap("via_line")
            for name in ("joint_line", "rrt"):
                if not s2_ok:
                    methods.append({"method": name, "ok": False, "fails": [f"경유점 IK {'실패' if not ok_v else f'여유 {m_v:.1f}°'}"
                                                                          if not (ok_v and m_v >= PRE_MARGIN_DEG) else "경유점 → 사전 파지 직선 실패"]})
                    break
                if name == "joint_line":
                    verts = np.array([q_now, q_via])
                else:
                    ins = math.radians(MARGIN_MIN_DEG + 1.0)
                    lo_s = np.maximum(rw.lo, ch.lo.double().numpy()) + ins
                    hi_s = np.minimum(rw.hi, ch.hi.double().numpy()) - ins
                    verts = rw.rrt(q_now, q_via, lo_s, hi_s, APPROACH_CUP_MARGIN_M)
                    lap("rrt")
                    if verts is None:
                        methods.append({"method": name, "ok": False, "fails": ["RRT 가 경로를 못 찾았다"]})
                        continue
                seg1 = time_joint(verts)
                cand = np.vstack([seg1, seg2[1:]])
                cj = judge(cand, APPROACH_CUP_MARGIN_M, APPROACH_CUP_CLEAR_M)
                methods.append({"method": name, "ok": cj["ok"], "fails": cj["fails"], "T_s": round((len(cand) - 1) / RATE_HZ, 2)})
                lap("check")
                if cj["ok"]:
                    appr, aj, ainfo = cand, cj, {"plan_ok": True, "arrived": True, "end_err_mm": 0.0}
                    break
        tr["transport"] = methods
        if not methods[-1]["ok"]:
            tr["why"] = "이송 — " + " | ".join(f"{m['method']}: {'; '.join(m['fails'])}" for m in methods)
            continue
        # 들어가기는 이송이 실제로 끝난 자세에서 푼다 — 관절 공간 이송은 경유점 → 사전 파지 직선 IK 끝이 q_pre 와 다른
        # 팔 자세(같은 손바닥)일 수 있고, Mover 도 q* 에 0.05 rad 안까지만 맞춘다(10.20 왼손 중심선: 이음매 8.6 rad/s).
        q_pre = appr[-1]
        R_g, p_g, az_g, rot_g = grasp_target(ch, q_pre, T_to, cup)
        p_g, deeper = deepen(p_g, cup, radius)
        path_in, dinfo = line_path(P, ch, q_pre, p_g, R_g, DEEP_SPEED_M_S, DEEP_ROT_DEG_S, DEEP_QD_RAD_S)
        dinfo.update(grasp_azimuth_deg=round(az_g, 1), rot_deg=round(rot_g, 1), deeper_mm=round(deeper * 1000, 1),
                     palm_gap_mm=round((float(np.linalg.norm(cup[:2] - p_g[:2])) - radius) * 1000, 1))
        dj = judge(path_in, DEEP_CUP_MARGIN_M, IN_CUP_CLEAR_M)
        dinfo.update({k: v for k, v in dj.items() if k not in ("ok", "fails")}, in_cup_clear_mm=dj["cup_clear_mm"],
                     in_hand_table_mm=dj["hand_table_mm"])
        dfails = dj["fails"] + ([] if dinfo["line_ok"] else ["들어가기 직선 · 방향을 못 따라감"]) + \
            ([] if rot_g <= DEEP_ROT_MAX_DEG else [f"방향 바꿈 {rot_g:.1f}° > {DEEP_ROT_MAX_DEG:.0f}°"])
        dinfo["ok"], dinfo["fails"] = not dfails, dfails
        tr["deep_in"] = {k: dinfo[k] for k in ("ok", "fails", "dist_mm", "rot_deg", "T_s", "margin_min_deg", "deeper_mm", "palm_gap_mm")}
        lap("deep_in")
        if dfails:
            tr["why"] = "들어가기 — " + "; ".join(dfails)
            continue
        chosen = (h, off, appr, aj, ainfo, methods[-1]["method"], path_in, dinfo)

    meta["tries"] = tries
    if chosen is None:
        meta["problems"].append("조건을 지키는 접근이 없다 — " + " / ".join(
            f"h{t['h_m'] * 100:+.0f}cm φ{t['phi_offset_deg']:+d}° {t.get('why', '')}" for t in tries))
        meta["ok"] = False
        return False, meta, None
    h, off, appr, aj, ainfo, method, path_in, dinfo = chosen
    ainfo.update(method=method, phi_offset_deg=off, phi_deg=round(math.degrees(phi0) + off, 1),
                 phi_regress_deg=round(math.degrees(phi0), 1), r_mult=float(p.r_mult[0]), h_m=float(h),
                 steps=len(appr) - 1, T_s=round((len(appr) - 1) / RATE_HZ, 2),
                 **{k: v for k, v in aj.items() if k not in ("fails",)})
    meta["approach"], meta["deep_in"] = ainfo, dinfo
    full = np.vstack([appr, path_in[1:]])
    meta["vmax_rad_s"] = round(float(np.abs(np.diff(full, axis=0)).max()) * RATE_HZ, 3)
    meta["n_cmd"] = int(len(full))
    meta["T_end_s"] = round((len(full) - 1) / RATE_HZ, 2)
    meta["q_grasp"] = [round(float(v), 4) for v in path_in[-1]]
    hard = [x for x in meta["problems"] if "FP++ 자세 확인" not in x]
    meta["ok"] = bool(not hard and meta["vmax_rad_s"] <= VMAX_RAD_S)
    return meta["ok"], meta, full


def rw_pair(k) -> str:
    from plan_home_path import fmt_pair
    return fmt_pair(k)


def arm_points_all(ch, q: np.ndarray) -> np.ndarray:
    _T, _ax, org = ch.fk(torch.tensor([q], dtype=torch.float32), frames=True)
    return org[0].double().numpy()


def parse_vec(text: str, n: int, what: str) -> np.ndarray:
    v = [float(x) for x in str(text).split(",") if x.strip()]
    if len(v) != n or not all(math.isfinite(x) for x in v):
        raise ReachError(f"{what} {text!r} — 숫자 {n} 개(쉼표)")
    return np.array(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=tuple(ARMS), required=True)
    ap.add_argument("--object", required=True, help="실물 이름(config/objects — 원점 높이 · 반지름)")
    ap.add_argument("--cup", required=True, help="FP++ 컵 원점 x,y,z(로봇 base, m)")
    ap.add_argument("--quat", default="", help="FP++ 컵 자세 w,x,y,z — 있으면 서 있는지 본다")
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--contract", type=Path, default=None, help="편 손 · 손끝 FK 계약(기본 그 팔 접근 정책 계약)")
    ap.add_argument("--q-json", default="", help='{"right": [7], "left": [7]} — 없으면 실측')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        cup = parse_vec(a.cup, 3, "--cup")
        quat = parse_vec(a.quat, 4, "--quat") if a.quat else None
        ok, meta = plan(a.side, cup, quat, a.object, a.out, current_q(a.q_json), H.hdgp_dir(), asset=a.asset,
                        contract=a.contract)
    except H.PinError as exc:
        print(f"✗ {exc}")
        return 3
    except (ReachError, KeyError) as exc:
        print(f"✗ {exc}")
        return 1
    print(json.dumps({"ok": ok, "T_end_s": meta.get("T_end_s"), "plan_wall_s": meta.get("plan_wall_s"),
                      "approach": meta.get("approach"), "deep_in": meta.get("deep_in"),
                      "problems": meta["problems"], "npz": str(a.out / "reach_cmd.npz")}, ensure_ascii=False))
    if not ok:
        print(f"✗ 계획 거부 — {'; '.join(meta['problems']) or '검사 실패(check.json)'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

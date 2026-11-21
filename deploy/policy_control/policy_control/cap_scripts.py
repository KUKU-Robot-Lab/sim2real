"""rh_cap 깊이 학생 노드의 스크립트 구간(하강 · 이송)과 캡 · 리시버 자세 공급(11.18) — rclpy 없이 도는 판단 논리.

학습 정본 = hdgp db1160bc rh_cap_env.py(_descend_actions 1536 · _pinch_ref 1599 · _carry_anchor 1604 · _carry_region 1611 ·
_carry_actions 1691), 값 = 학습 런 env.yaml(server wt_cap/runs/cap_split/depth_h01/s31/run/params, 아래 줄 번호).
  하강  손바닥 방향을 시작 순간에 고정하고 pinch_ref 를 캡 원점 + h·캡 축으로 보내는 곧은 선(속도 grasp_descend_speed)을
        DLS 한 걸음씩 따라간다. 손은 편 채. 끝 = 도달(선 끝 · 오차 ≤ tol) · 닿음 · max_steps.
  이송  hdgp ready_carry Mover(배포 핀 사본) — 앵커 = 리시버 입구 위 carry_h_clr 에 캡 입, 경유 = 위 → 수평 → 축 따라.
        팔 행동 = Mover 증분 / (k_arm · arm_ema), Mover 가 끝나면 0(유지).
실기에만 있는 보호(학습과 다르다): 하강 시작 검사(집기점 xy · 선 길이 · 이미 닿음), 이송 명령 − 실측 추종 가드(0.1 rad × 6 틱,
  place_ready_carry 값) — 둘 다 ScriptError 를 reset 까지 매 틱 낸다(노드 gap → abort). 이송 미도달(Mover 가 도착 없이 끝남)은
  sim 은 그대로 넘기지만 여기는 목표를 유지하고 자동 끼우기를 끈다(사람 stage/next).
실기 근사(sim 은 참값을 쓴다):
  · 캡 자세 = start 순간 공급 자세(FP++ 신선하면 그것, 아니면 고정 자리) 한 번. 하강은 그 자세로 목표를 잡는다.
  · 손 안 캡 자세 = 하강이 끝나 잡기로 넘어가는 순간의 캡 자세를 그때 손바닥 FK 에 붙인 것(rh_cap.CapChain.cup_from_palm 방식).
    잡기 동안 손가락이 캡을 옮기는 것은 반영하지 않는다(sim 은 이송 시작 순간 참 캡 자세).
  · 닿음 = 손끝 촉각 5 개 최대 > touch_n. sim 은 손가락 링크(근위 · 손끝) 접촉력과 손바닥 센서도 본다 — 실기 계약에는 없다.
"""
from __future__ import annotations

import logging
from typing import Callable, Mapping, Sequence

import numpy as np

from policy_control import rh_aglt as A
from policy_control import rh_cap as C
from policy_control.place_ready_carry import TRACK_ABORT_RAD, TRACK_ABORT_TICKS

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------- 학습 런 값(env.yaml 줄)
PINCH_OPEN_OFFSET = (0.033, 0.074, 0.034)   # 1471 pinch_open_offset — 손바닥 틀 편 손 집기점
DESCEND_SPEED = 0.06                        # 1519 grasp_descend_speed [m/s]
DESCEND_TOL_M = 0.006                       # 1520 grasp_descend_tol_m
DESCEND_MAX_STEPS = 150                     # 1521 grasp_descend_max_steps
DESCEND_TOUCH_N = 1.0                       # 1534 grasp_descend_touch_n
# 하강 멈춤 h: 1518 grasp_descend_h 0.0 + 커리큘럼 U(lo, hi). level0 4 = steps 4(1528 · 1532) → 끝 범위 h_end (0.03, 0.05)(1525).
#   런은 에피소드마다 이 범위에서 뽑는다. 배포 기본은 가운데 — 노드 파라미터 descend_h_m 으로 덮는다.
DESCEND_H_RANGE = (0.03, 0.05)
DESCEND_H = 0.04
IK_DAMPING = 0.05                           # 1508 grasp_ik_damping
CARRY_H_CLR = 0.03                          # 1536 carry_h_clr
CARRY_ABOVE = 0.06                          # 1537 carry_above
CARRY_VIA_X_MIN = 0.10                      # 1538 carry_via_x_min
CARRY_V_MAX = 0.0015                        # 1539 carry_v_max [m/스텝]
CARRY_QD_MAX = 0.0045                       # 1540 carry_qd_max [rad/스텝]
CARRY_MAX_STEPS = 900                       # 1541 carry_max_steps(cfg 기본과 다르다)
CARRY_SETTLE_STEPS = 60                     # 1542 carry_settle_steps
CARRY_PLAN_DOWN = 0.05                      # 1543 carry_plan_down
CARRY_PLAN_XY = 0.02                        # 1544 carry_plan_xy
CARRY_PLAN_TILT_DEG = 5.0                   # 1545 carry_plan_tilt_deg
CARRY_PLAN_SEEDS = 8                        # 1546 carry_plan_seeds(carry_ref_bank '' — 기준 뱅크 없음)
CARRY_N_DOWN = 4                            # env _carry_region n_down
# 하강 시작 검사(실기 전용, sim 에 없다). 근거 수치 없는 첫 판 — 접근 성공 조건(손바닥-캡 오차 2 cm · 높이 4~10 cm)에서 잡았다.
#   노드 파라미터 descend_xy_max_m · descend_len_max_m 로 덮는다.
DESCEND_XY_MAX_M = 0.04                     # 시작 순간 pinch_ref 와 캡의 수평 거리 상한
DESCEND_LEN_MAX_M = 0.12                    # 하강 선 길이 L 상한
DESCEND_NAN_WARN_TICKS = 10                 # 촉각이 전부 NaN 인 틱이 이만큼 이어지면 경고 한 번(멈추지는 않는다)
# 캡 기하 = cap_asset_dir assets/cyl_cap/shaker_pair_donut12 · cap_clearance_mm 6.0(1393 · 1396) → cap_c6.json z_bottom
#   (같은 폴더 c1.1 · c3 도 −0.015). 리시버 원점 → 입구 = cap_geom.RCV_HALF_H(rh_cap.RCV_HALF_H 0.085).
CAP_Z_BOTTOM = -0.015

# ---------------------------------------------------------------- 자세 공급 기본값 · 학습 범위
# 캡 소환 = spawn_center (0.215, −0.185) ± spawn_half (0.085, 0.165)(425-430). 원점 높이 = table_surface_z 0.205(395)
#   + cup_origin_offset_z 0.014(403) — 캡은 입(z_bottom)을 아래로 상판에 선다. 로봇 루트 틀.
CAP_SPAWN_CENTER = (0.215, -0.185)
CAP_SPAWN_HALF = (0.085, 0.165)
CAP_FIXED_POS = (0.215, -0.185, 0.219)
CAP_FIXED_QUAT = (1.0, 0.0, 0.0, 0.0)
# 리시버 = 왼손이 든 쉐이커 몸통. 뱅크 receiver_bank_path rcv_bank_rcg04_base_ik.npz(1421, 511 행) rcv_root 의 중앙값 · 범위 · 기울기.
RCV_FIXED_POS = (0.2519, 0.1241, 0.4084)
RCV_FIXED_QUAT = (0.9617, 0.0487, -0.0012, -0.2637)
RCV_BANK_LO = (0.2383, 0.0711, 0.3783)
RCV_BANK_HI = (0.2677, 0.1276, 0.4323)
RCV_TILT_MAX_DEG = 38.0                     # 뱅크 축 기울기 1.8~37.8°
# FP++ 쉐이커(shaker_c_*) 메쉬 원점 → env 리시버 원점: 같은 몸통에서 축 방향 −0.020 m
#   (hdgp db1160bc scripts/tools/make_cap_receiver_bank.py: dz = 쉐이커 bottom_z −0.065 − json rcv_z_bottom −0.045).
RCV_FROM_SHAKER_DZ = -0.020
# 범위 거부 폭. 근거 수치 없음(첫 판) — 캡: 소환 경계 밖 2 cm · 원점 높이 ± 2 cm(상판에 선 캡), 리시버: 뱅크 경계 밖 3 cm
#   (뱅크 관측 오차 rcv_obs_dp 최대 2.3~4.0 cm 와 같은 크기. 이 런은 receiver_obs_err false). 실기 첫 시험 뒤 좁힌다.
CAP_RANGE_TOL_M = 0.02
RCV_RANGE_TOL_M = 0.03


class ScriptError(ValueError):
    """스크립트 구간 입력이 없다 — 노드는 틱 오류(ValueError 계열)로 받는다."""


# ---------------------------------------------------------------- 하강
def descend_goal(palm_pos, palm_R, cap_pos, cap_quat, h: float, pinch_off=PINCH_OPEN_OFFSET) -> np.ndarray:
    """env _descend_actions 목표 손바닥 = 손바닥 + (캡 원점 + h·캡 축 − pinch_ref), pinch_ref = 손바닥 + R·pinch_open_offset."""
    palm = np.asarray(palm_pos, float)
    pinch = palm + np.asarray(palm_R, float) @ np.asarray(pinch_off, float)
    return palm + (np.asarray(cap_pos, float) + float(h) * C.axis_up(cap_quat) - pinch)


def dls_arm_action(c, fk, arm_q, hand_q, arm_target, pos_t, R_t, damping: float = IK_DAMPING) -> np.ndarray:
    """env _dls_arm_action — 손바닥을 (pos_t, R_t) 로 보내는 DLS 한 걸음 → 팔 증분 행동 (q_des − q*) / k_arm (rh_cap.ik_lift_action 과 같은 식)."""
    s = c.side()
    pose = fk.palm_pose(arm_q, hand_q)
    err = np.concatenate([np.asarray(pos_t, float) - pose.palm_pos, C._rotvec(np.asarray(R_t, float) @ pose.extra["palm_rot"].T)])
    J = C.palm_jacobian(fk, arm_q, hand_q)
    dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(6), err)
    q_des = np.clip(np.asarray(arm_q, float) + dq, np.asarray(s.arm_lo, float), np.asarray(s.arm_hi, float))
    return np.clip((q_des - np.asarray(arm_target, float)) / c.k_arm, -1.0, 1.0)


class Descend:
    """한 에피소드 하강. start(그 순간 손바닥 · 캡) → step 마다 (팔 행동, 끝)."""

    def __init__(self, c, fk, h: float = DESCEND_H, touch_n: float = DESCEND_TOUCH_N,
                 xy_max: float = DESCEND_XY_MAX_M, len_max: float = DESCEND_LEN_MAX_M) -> None:
        self.c, self.fk, self.h, self.touch_n = c, fk, float(h), float(touch_n)
        self.xy_max, self.len_max = float(xy_max), float(len_max)
        self.p0 = self.goal = self.R0 = None
        self.t, self.why, self.err, self.nan_n = 0, "", float("nan"), 0

    def start(self, palm_pos, palm_R, cap_pos, cap_quat, tactile_n=None) -> None:
        """시작 검사(실기 전용): 집기점이 캡 위 xy_max 안 · 선 길이 ≤ len_max · 손끝이 아직 안 닿음. 아니면 ScriptError."""
        p0, R0 = np.asarray(palm_pos, float).copy(), np.asarray(palm_R, float).copy()
        goal = descend_goal(p0, R0, cap_pos, cap_quat, self.h)
        xy = float(np.linalg.norm((p0 + R0 @ np.asarray(PINCH_OPEN_OFFSET))[:2] - np.asarray(cap_pos, float)[:2]))
        L = float(np.linalg.norm(goal - p0))
        bad = [f"집기점-캡 수평 {xy * 1e3:.1f} mm > {self.xy_max * 1e3:.0f} mm"] if not xy <= self.xy_max else []
        bad += [f"하강 길이 L {L * 1e3:.1f} mm > {self.len_max * 1e3:.0f} mm"] if not L <= self.len_max else []
        if tactile_n is not None and np.all(np.isnan(np.asarray(tactile_n, float))):     # 촉각이 끊겼으면 닿음 정지 없이 끝까지 내려간다
            bad.append("하강 시작 전에 촉각이 전부 NaN — 촉각 연결 확인")
        elif tactile_n is not None and np.any(np.asarray(tactile_n, float) > self.touch_n):  # 기준값이 떠 있으면 첫 틱에 grasp 로 간다
            bad.append(f"하강 시작 전에 촉각 {np.round(np.asarray(tactile_n, float), 2).tolist()} N > {self.touch_n} — 촉각 영점 확인")
        if bad:
            raise ScriptError("하강 시작 거부: " + " · ".join(bad))
        self.p0, self.R0, self.goal = p0, R0, goal
        self.t, self.why, self.nan_n = 0, "", 0

    def step(self, m, hand_q, arm_target) -> tuple[np.ndarray, bool]:
        self.t += 1
        self.nan_n = self.nan_n + 1 if np.all(np.isnan(np.asarray(m.tactile_n, float))) else 0
        if self.nan_n == DESCEND_NAN_WARN_TICKS:
            _log.warning("하강: 촉각이 %d 틱째 전부 NaN — 닿음으로 멈추지 못한다(도달 · 시간 초과로만 끝난다)", self.nan_n)
        d = self.goal - self.p0
        L = max(float(np.linalg.norm(d)), 1e-6)
        s = min(self.t * DESCEND_SPEED / float(self.c.policy_hz), L)
        a = dls_arm_action(self.c, self.fk, m.arm_q, hand_q, arm_target, self.p0 + d / L * s, self.R0)
        self.err = float(np.linalg.norm(self.goal - np.asarray(m.palm_pos, float)))
        reached = s >= L - 1e-6 and self.err <= DESCEND_TOL_M
        touch = bool(np.any(np.asarray(m.tactile_n, float) > self.touch_n))     # NaN(끊긴 촉각)은 닿음이 아니다
        timeout = self.t >= DESCEND_MAX_STEPS
        self.why = "reached" if reached else "touch" if touch else "timeout" if timeout else ""
        return a, bool(self.why)

    def as_dict(self) -> dict:
        return {"t": self.t, "h": self.h, "err_mm": round(self.err * 1000, 1) if np.isfinite(self.err) else None, "end": self.why}


# ---------------------------------------------------------------- 이송
def carry_anchor(rcv_pos, rcv_quat, h_clr: float = CARRY_H_CLR, z_bottom: float = CAP_Z_BOTTOM) -> tuple[np.ndarray, np.ndarray]:
    """env _carry_anchor — (앵커 캡 원점, 리시버 축). 캡 입이 리시버 입구 위 h_clr."""
    ru = C.axis_up(rcv_quat)
    ru = ru / max(float(np.linalg.norm(ru)), 1e-9)
    return np.asarray(rcv_pos, float) + ru * (C.RCV_HALF_H + h_clr - z_bottom), ru


def carry_via(p0, c_pre, ru) -> np.ndarray:
    """env _carry_actions 경유점 (2, 3): v1 = p0 를 앵커 위(c_pre + ru·above) 높이까지 올리고 x ≥ via_x_min, v2 = 앵커 위."""
    c_ab = np.asarray(c_pre, float) + np.asarray(ru, float) * CARRY_ABOVE
    v1 = np.asarray(p0, float).copy()
    v1[2] = max(v1[2], c_ab[2])
    if CARRY_VIA_X_MIN > 0.0:
        v1[0] = max(v1[0], CARRY_VIA_X_MIN)
    return np.stack([v1, c_ab])


def carry_region(ru) -> tuple[np.ndarray, np.ndarray]:
    """env _carry_region 표본 (offsets, tilts) (K, 3): 축 따라 내림 4 · 축 수직 xy ± (dz 0 · −down) · 기울기 ± (dz 0 · −down)."""
    u = np.asarray(ru, float)
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(u[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    down, xy, z3 = CARRY_PLAN_DOWN, CARRY_PLAN_XY, np.zeros(3)
    offs = [z3] + [-u * down * i / CARRY_N_DOWN for i in range(1, CARRY_N_DOWN + 1)]
    tils = [z3] * (CARRY_N_DOWN + 1)
    for dz in (0.0, -down):
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                offs.append(u * dz + e1 * sx * xy + e2 * sy * xy)
                tils.append(z3)
    t = np.radians(CARRY_PLAN_TILT_DEG)
    for dz in (0.0, -down):
        for r in (e1 * t, -e1 * t, e2 * t, -e2 * t):
            offs.append(u * dz)
            tils.append(r)
    return np.stack(offs), np.stack(tils)


class Carry:
    """한 에피소드 이송(B = 1, CPU). RC · chain = place_ready_carry.load_chain(배포 핀 hdgp ready_carry 사본 · 손바닥 몸체 사슬)."""

    def __init__(self, RC, chain, *, k_arm: float, arm_ema: float, seed: int = 0) -> None:
        self.RC, self.ch, self.scale, self.seed = RC, chain, float(k_arm) * float(arm_ema), int(seed)
        if CARRY_QD_MAX > self.scale:                  # env 와 같은 검사(계획 시간이 행동 한계에 잘리면 틀어진다)
            raise ScriptError(f"carry_qd_max {CARRY_QD_MAX} > 팔 slew k_arm·arm_ema {self.scale}")
        self.reset()

    def reset(self) -> None:
        self.mv, self.plan, self.t = None, {}, 0
        self.arrived, self.track_n, self.track_dev, self.fail, self.miss_mm = False, 0, 0.0, "", None
        self.T_to = self.anchor = None

    def _fail(self, why: str):
        self.fail = why                                 # 한 번 실패하면 reset 전까지 매 틱 같은 오류(노드 gap → abort)
        raise ScriptError(why)

    def _T(self, pos, quat):
        import torch
        from policy_control.place_ready_carry import quat_matrix
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = quat_matrix(quat), np.asarray(pos, float)
        return torch.tensor(T[None], dtype=torch.float32)

    def cap_in_palm(self, arm_q, cap_pos, cap_quat):
        """손바닥(사슬 끝) → 캡 4×4 = FK(실측 q)⁻¹ ⊗ 캡. 근사: 이 순간 캡 자세를 손바닥에 붙인다(모듈 주석)."""
        import torch
        q = torch.tensor([list(np.asarray(arm_q, float))], dtype=torch.float32)
        return torch.linalg.inv(self.ch.fk(q)) @ self._T(cap_pos, cap_quat)

    def start(self, q_cmd, T_to, rcv_pos, rcv_quat) -> dict:
        """env _carry_actions 시작 — p0 = FK(명령 q)⊗T_to 의 캡 점, Mover.start(명령 q, …, via)."""
        import torch
        if self.fail:
            raise ScriptError(self.fail)
        if not np.all(np.isfinite(np.asarray(T_to, float))):
            self._fail("이송: 손 안 캡 자세(T_to)에 NaN/inf — 이송을 시작하지 않는다")
        RC = self.RC
        try:                                            # Mover 는 계획이 받아들여진 뒤에만 self.mv(시작 안 된 Mover 의 on False = 끝으로 읽힌다)
            mv = RC.mover.Mover(self.ch, RC.mover.MoverCfg(v_max=CARRY_V_MAX, qd_max=CARRY_QD_MAX, max_steps=CARRY_MAX_STEPS,
                                                           settle_steps=CARRY_SETTLE_STEPS, seeds=CARRY_PLAN_SEEDS),
                                1, "cpu", seed=self.seed)
            q = torch.tensor([list(np.asarray(q_cmd, float))], dtype=torch.float32)
            c_pre, ru = carry_anchor(rcv_pos, rcv_quat)
            p0 = (self.ch.fk(q) @ T_to)[0, :3, 3].double().numpy()
            offs, tils = carry_region(ru)
            f = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float32)  # noqa: E731
            reg = RC.posture.Region(f(c_pre[None]), f(ru[None]), f(offs), f(tils))
            plan = mv.start(torch.arange(1), q, T_to, None, f(c_pre[None]), f(ru[None]), reg, via=f(carry_via(p0, c_pre, ru)[None]))
            info = {"ok": bool(plan.ok[0]), "margin_deg": round(float(np.degrees(float(plan.margin[0]))), 2),
                    "T_steps": int(mv.T[0]), "anchor": [round(float(v), 4) for v in c_pre],
                    "q_goal": [round(float(v), 4) for v in plan.q[0]]}
        except Exception as exc:                        # 계획 예외도 틱 오류(→ abort)로 — 노드 틱을 죽이지 않는다
            self._fail(f"이송 계획 예외: {exc!r}")
        if not np.all(np.isfinite(info["q_goal"])):
            self._fail(f"이송 계획 목표 q 에 NaN/inf {info['q_goal']} — 이송을 시작하지 않는다")
        if not info["ok"]:                              # 학습 env 도 ok 로 막지 않는다(뱅크 plan_ok 30 %) — 기록만, 보호는 추종 가드 · 미도달
            _log.warning("이송 계획 ok False(여유 %.2f° · 앵커 %s) — 학습과 같이 그대로 이송한다", info["margin_deg"], info["anchor"])
        self.plan, self.mv, self.t, self.T_to, self.anchor = info, mv, 0, T_to, c_pre
        return self.plan

    @property
    def on(self) -> bool:
        return self.mv is not None and bool(self.mv.on[0])

    @property
    def missed(self) -> bool:
        """Mover 가 도착 없이 끝났다(시간 초과 등) — 목표 유지, 자동 끼우기 없음, 사람 stage/next 만."""
        return self.mv is not None and not self.on and not self.arrived

    def step(self, q_cmd, arm_q) -> tuple[np.ndarray, bool]:
        """(팔 행동, 끝 = 도착해서 끝남). env 처럼 명령 q(디코더 누적 목표)로 Mover 를 굴린다. 끝난 뒤 = 0(유지).
        명령 − 실측 이탈 가드(place_ready_carry 와 같은 값 · NaN 은 위반)는 ScriptError. 도착 없이 끝나면 missed(끝 False 유지)."""
        import torch
        if self.fail:
            raise ScriptError(self.fail)
        if self.mv is None:
            raise ScriptError("이송: start 전")
        dev = float(np.max(np.abs(np.asarray(q_cmd, float) - np.asarray(arm_q, float))))
        self.track_dev = dev
        self.track_n = self.track_n + 1 if not dev <= TRACK_ABORT_RAD else 0
        if self.track_n >= TRACK_ABORT_TICKS:
            self._fail(f"이송: 명령 − 실측 {dev:.3f} rad > {TRACK_ABORT_RAD} 가 {self.track_n} 틱 — 팔이 명령을 못 따라간다")
        a = np.zeros(7)
        if self.on:
            dq, _, done, arrive = self.mv.step(torch.tensor([list(np.asarray(q_cmd, float))], dtype=torch.float32))
            self.t += 1
            if bool(done[0]):
                self.arrived = bool(arrive[0])
            a = np.clip(dq[0].double().numpy() / self.scale, -1.0, 1.0)
        if self.missed:                                 # 남은 오차 = 실측 q 의 캡 점 − 앵커(틱마다 — 사람이 보고 넘긴다)
            q = torch.tensor([list(np.asarray(arm_q, float))], dtype=torch.float32)
            p = (self.ch.fk(q) @ self.T_to)[0, :3, 3].double().numpy()
            first = self.miss_mm is None
            self.miss_mm = round(float(np.linalg.norm(p - self.anchor)) * 1e3, 1)
            if first:
                _log.warning("carry 미도달: Mover 가 %d 틱(상한 %d)에 도착 없이 끝났다 · 앵커까지 %.1f mm — 목표 유지, "
                             "자동 끼우기 없음(stage/next 로 넘긴다)", self.t, CARRY_MAX_STEPS, self.miss_mm)
        return a, self.arrived and not self.on

    def as_dict(self) -> dict:
        return {"t": self.t, "on": self.on, "arrived": self.arrived, "track_dev": round(self.track_dev, 4),
                "carry_missed": self.missed, "miss_err_mm": self.miss_mm, "fail": self.fail or None, **self.plan}


class CapScripts:
    """노드가 한 번 만들고 에피소드마다 reset — 체인이 부른다(테스트는 가짜로 바꾼다)."""

    def __init__(self, c, carry: Carry, h: float = DESCEND_H, xy_max: float = DESCEND_XY_MAX_M,
                 len_max: float = DESCEND_LEN_MAX_M) -> None:
        self.c, self.carry, self.h, self.xy_max, self.len_max = c, carry, float(h), float(xy_max), float(len_max)
        self.fk = None                                  # UrdfChainFK(노드가 reset 때 넣는다)
        self.reset()

    def reset(self) -> None:
        self.dsc: Descend | None = None
        self.T_to, self.fail = None, ""
        self.carry.reset()

    def descend_step(self, m, hand_q, arm_target, cap) -> tuple[np.ndarray, bool]:
        if self.fail:                                   # 시작 거부는 reset 까지 유지(촉각이 잠깐 내려가도 중간에서 시작하지 않는다)
            raise ScriptError(self.fail)
        if self.dsc is None:
            if self.fk is None:
                raise ScriptError("하강: 손바닥 FK 없음")
            dsc = Descend(self.c, self.fk, self.h, xy_max=self.xy_max, len_max=self.len_max)
            try:
                dsc.start(m.palm_pos, m.palm_R, *cap, tactile_n=m.tactile_n)
            except ScriptError as exc:
                self.fail = str(exc)
                raise
            self.dsc = dsc
        a, done = self.dsc.step(m, hand_q, arm_target)
        if done:                                        # 잡기로 넘어가는 순간 — 캡을 손바닥에 붙인다(근사)
            try:                                        # torch 예외(특이 행렬 등)가 타이머 콜백까지 나가지 않게
                T_to = self.carry.cap_in_palm(m.arm_q, *cap)
                ok = bool(np.all(np.isfinite(np.asarray(T_to, float))))
            except Exception as exc:
                self.fail = f"손 안 캡 자세 계산 예외: {exc!r}"
                raise ScriptError(self.fail) from exc
            if not ok:
                self.fail = "손 안 캡 자세에 NaN/inf(실측 arm_q 확인) — 이송하지 않는다"
                raise ScriptError(self.fail)
            self.T_to = T_to
        return a, done

    def carry_step(self, m, arm_target, rcv) -> tuple[np.ndarray, bool]:
        if self.carry.mv is None:
            if self.T_to is None:
                raise ScriptError("이송: 손 안 캡 자세 없음(하강이 끝나지 않았다)")
            self.carry.start(arm_target, self.T_to, *rcv)
        return self.carry.step(arm_target, m.arm_q)

    @property
    def carry_missed(self) -> bool:
        return self.carry.missed

    def as_dict(self) -> dict:
        return {"descend": None if self.dsc is None else self.dsc.as_dict(), "fail": self.fail or None,
                "carry": self.carry.as_dict() if self.carry.mv is not None or self.carry.fail else None}


# ---------------------------------------------------------------- 자세 공급
def rcv_from_shaker(pos, quat) -> tuple[np.ndarray, np.ndarray]:
    """FP++ 쉐이커 메쉬 원점 → env 리시버 원점(같은 축 RCV_FROM_SHAKER_DZ)."""
    q = np.asarray(quat, float)
    return np.asarray(pos, float) + A.quat_apply(q, np.array([0.0, 0.0, RCV_FROM_SHAKER_DZ])), q


STAMP_FUTURE_TOL_S = 0.05                               # 영상 시계와 시스템 시계의 작은 어긋남은 신선한 것으로 받는다


def pick_pose(got, fixed: tuple, now: float, stale_s: float, conv: Callable | None = None, wall: float | None = None) -> dict:
    """got = (받은 monotonic, pos, quat, stamp) | None(pour_fj_node.cups 형식). stale_s 안이면 FP++(conv 로 환산), 아니면 고정 자리.
    age_s = 마지막 FP++ 수신 나이(받은 적 없으면 None). stamp(영상 시각, 시스템 시계 초)가 0 이 아니고 wall(time.time())이
    있으면 stamp 나이도 stale_s 안이어야 FP++ 다(stamp_age_s)."""
    age = None if got is None else float(now - got[0])
    st = None if got is None or len(got) < 4 or not got[3] or wall is None else float(wall - float(got[3]))
    if age is not None and 0.0 <= age <= stale_s and (st is None or -STAMP_FUTURE_TOL_S <= st <= stale_s):
        pos, quat = (got[1], got[2]) if conv is None else conv(got[1], got[2])
        src = "fpp"
    else:
        (pos, quat), src = fixed, "fixed"
    q = np.asarray(quat, float)
    return {"src": src, "pos": np.asarray(pos, float), "quat": q / np.linalg.norm(q), "age_s": age, "stamp_age_s": st}


def _box_refusals(name: str, pos, lo, hi, tol: float) -> list:
    p = np.asarray(pos, float)
    out = np.maximum(np.asarray(lo, float) - tol - p, p - np.asarray(hi, float) - tol)
    return [] if np.all(out <= 0.0) else [f"{name} 위치 {np.round(p, 3).tolist()} 가 학습 범위 {np.round(lo, 3).tolist()}~"
                                         f"{np.round(hi, 3).tolist()} ± {tol} m 밖"]


def _tilt(quat) -> float:
    return float(np.degrees(np.arccos(np.clip(C.axis_up(quat)[2], -1.0, 1.0))))


def cap_refusals(p: Mapping) -> list:
    """캡 자세(공급 결과)가 학습 소환 범위 · 서 있는 캡에서 크게 벗어나면 거부 사유."""
    lo = (CAP_SPAWN_CENTER[0] - CAP_SPAWN_HALF[0], CAP_SPAWN_CENTER[1] - CAP_SPAWN_HALF[1], CAP_FIXED_POS[2])
    hi = (CAP_SPAWN_CENTER[0] + CAP_SPAWN_HALF[0], CAP_SPAWN_CENTER[1] + CAP_SPAWN_HALF[1], CAP_FIXED_POS[2])
    bad = _box_refusals(f"캡({p['src']})", p["pos"], lo, hi, CAP_RANGE_TOL_M)
    t = _tilt(p["quat"])
    return bad + ([f"캡({p['src']}) 기울기 {t:.1f}° > {C.START_TILT_MAX_DEG:.0f}° — 서 있는 캡이어야 한다"]
                  if t > C.START_TILT_MAX_DEG else [])


def rcv_refusals(p: Mapping) -> list:
    """리시버 자세가 뱅크 범위(위치 ± 폭 · 기울기)에서 크게 벗어나면 거부 사유."""
    bad = _box_refusals(f"리시버({p['src']})", p["pos"], RCV_BANK_LO, RCV_BANK_HI, RCV_RANGE_TOL_M)
    t = _tilt(p["quat"])
    return bad + ([f"리시버({p['src']}) 기울기 {t:.1f}° > 뱅크 최대 {RCV_TILT_MAX_DEG:.0f}°"] if t > RCV_TILT_MAX_DEG else [])


def rcv_fixed_refusals(p: Mapping, allow_fixed: bool, stale_s: float) -> list:
    """리시버가 고정 자리로 떨어졌는데(FP++ 낡음 · 없음) 노드가 그걸 허락하지 않으면(rcv_allow_fixed false) 거부 사유."""
    if p["src"] != "fixed" or allow_fixed:
        return []
    v = pose_view(p)
    st = v["stamp_age_s"]
    fut = f" · stamp 가 미래({-st} s, 허용 {STAMP_FUTURE_TOL_S} s)" if st is not None and st < -STAMP_FUTURE_TOL_S else ""
    return [f"리시버 FP++ 낡음(수신 나이 {v['age_s']} s · stamp 나이 {st} s, 상한 {stale_s} s{fut}) — "
            f"고정 자리로 시작하려면 rcv_allow_fixed:=true"]


def pose_view(p: Mapping) -> dict:
    r = lambda v: None if v is None else round(v, 3)  # noqa: E731
    return {"src": p["src"], "age_s": r(p["age_s"]), "stamp_age_s": r(p.get("stamp_age_s")),
            "pos": [round(float(v), 4) for v in p["pos"]]}


def as_pair(p: Mapping) -> tuple[np.ndarray, np.ndarray]:
    return p["pos"], p["quat"]


def vec(v: Sequence[float], n: int, name: str) -> tuple:
    """노드 파라미터 → 길이 n 실수 튜플(아니면 ValueError)."""
    out = tuple(float(x) for x in v)
    if len(out) != n or not np.all(np.isfinite(out)):
        raise ValueError(f"{name}: 실수 {n} 개가 필요하다 — {list(v)}")
    return out

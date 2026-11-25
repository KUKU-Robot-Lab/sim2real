"""rh_place_node: RH56F1 한 팔 컵 홀더 놓기(hdgp open-rh_{r,l}_place) — pour_fj_node 의 rh_place 계열 + 이 노드 고유 부분(10.04).

  in   robot yaml(한 팔, rh56f1_<side>_*): 팔 /joint_states · 손 /hand_<side>/joint_states · 촉각 · 관절 힘
       컵 PoseStamped(base): cup_topic(기본 /objects/cyl60/pose) — reset 순간에만 쓴다(그 뒤는 손바닥에 붙은 컵)
       홀더 PoseStamped(base, latched): holder_topic(기본 /objects/cup_holder_<holder>/pose) — 목표 = 원점 + (0, 0, seat_dz)
       /policy_control/joint_target — pd 가 붙잡고 있는 마지막 목표(aglt 가 낸 것) = 인계 팔 · 손 목표(sim 뱅크 arm_q_target)
  out  /policy_control/joint_target (팔 7 · 손 6) · /policy_control/<ns>/{obs,action,episode} · status/<노드>
  srv  /policy_control/<ns>/episode/{reset,start,stop,abort} — 놓은 뒤 스크립트가 끝나면 스스로 stop

    python3 deploy/policy_control/policy_control/rh_place_node.py --ros-args -r __node:=rh_place_node_right -p ns:=right \\
        -p contract:=deploy/policies/rh56f1/place/right_i09/rh_place_contract.json -p robot:=… [-p holder:=2]

★10.22 newton_05 계약(latch_s2r · attach_source bundle · holder_range):
  · 붙인 자세 = 잡기 노드가 남긴 grasp_state 넘김 묶음(place_handover) — bundle:=auto(그 팔 rh_aglt_node 의 가장 새 묶음,
    bundle_max_age_s 안 · 아직 안 쓴 것 · 손 실측이 묶음과 bundle_hand_tol_rad 안) 또는 파일 경로. 없으면 reset 거부(FP++ 로 안 붙인다).
  · 래치 컵 = 손바닥 FK ⊗ 래치 추정기(정책 폴더 latch/, md5 핀) — place_latch_est.
  · holder = 실물 홀더 번호(로봇 왼쪽 +y 부터 0 · 1 · 2). 학습 목표 홀더는 sim 슬롯 번호라 번호로 거르지 않고, 그 홀더 원점이 학습
    범위(holder_range)에서 holder_tol_m 넘게 밖이면 reset 거부. ★10.29 holder_tol_m 은 reset 마다 다시 읽는다(실행기가 job 마다
    넣는다 · 카드가 그 job 에 넓힌 허용, 없으면 0.02).
  · 학습 episode_length_s 끝: 놓기 전이어도 stop(★11.10 — 판정은 실행기 cup_in_holder), 놓은 뒤(빼기 중)면 stop.
  · ★11.11 정책 손 목표 · 실측 손가락이 둘 다 open_min_rad 넘게 열리면(OPENED_END_TICKS 연속) 그 틱에 stop(outcome "opened_before_latch", 이 규칙은 래치 전에만).
★10.25 준비 자세 이송(계약 ready_carry — newton_07_r · 06_l): reset 때 끝 자세를 풀고(Mover, 움직이지 않음) start 뒤 첫 틱부터
  Mover 행동으로 슬롯 위까지 옮긴 다음 정책. 에피소드 시간은 이송이 끝난 틱부터 센다. 이송 자체 상한 =
  (max_steps + settle_steps + 멈춤 확인 STILL_MAX_TICKS) 틱 + READY_CARRY_SLACK_S — 넘으면 abort(Mover 가 그 전에 끝나야 한다).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

import time  # noqa: E402

READY_CARRY_SLACK_S = 2.0
OPENED_END_TICKS = 3      # ★11.11 정책 손 목표 · 실측 손가락이 둘 다 열린 틱이 이만큼 연속(60 Hz 50 ms)이면 끝
#: ★11.18 래치 전 손 폄 종료의 문턱 하한(rad) — 놓음 게이트 open_min_rad(0.15)와 따로. 학습 정책은 래치 전 손가락을 인계 대비
#  0.23~0.25 rad 편 채 내린다(체크포인트 running_mean_std). 06:28 cocktail_1c_e4000 은 인계 손이 뱅크보다 0.17 rad 더 닫혀
#  0.15 를 14 틱에 넘어 래치(~52 틱) 전에 끊겼다. 0.35 = 학습 연 양 위 · 완전 개방 아래.
OPENED_END_MIN_RAD = 0.35
OPENED_MEAS_FRAC = 1.0    # ★11.11 "다 핀" 쪽 — 실측 평균 ≥ open_min_rad. 0.5(0.075 rad)는 sim 1111b place_blue 에서 컵을 쥔 채
                          #   0.12 rad(안착 높이 +40.5 mm)에 끝냈다. 실기 bag 184024 는 707.1 s 0.067 → 707.2 s 0.548 로 뛰어 종료 시각 거의 같음

from policy_control import codec  # noqa: E402
from policy_control import place_handover as PH  # noqa: E402
from policy_control.pour_fj_node import NS, Family, PourFjNode, PourFjNodeError, main  # noqa: E402
from policy_control.rh_place import (ATTACH_SOURCES, HOLDER_FLOOR_Z, HOLDER_TOL_M, JOINT_FREE_G,  # noqa: E402
                                     OPEN_MIN_RAD, RhPlaceError, _matrix_to_quat, _quat_mul, holder_offset_of,
                                     holder_refusals, holder_tol_of)
from policy_control import tin_live as TL  # noqa: E402
from policy_control.place_ready_carry import STILL_MAX_TICKS as RCY_STILL_MAX_TICKS, ReadyCarryError  # noqa: E402
from policy_control import place_ready_carry as RCY  # noqa: E402

TIN_FPP_TILT_DEG = 15.0     # ★11.07 tin_json — FP++ 를 FK 와 비교해 남기는 조건(컵이 거의 서 있고 탁자 위)
# ★11.20 사용자 왼손 place 이송 끝 FP++ 재측정 · 보정(tin_refresh_at_carry_end · carry_fpp_xy_tol_m). 진단 logs/analysis/place_shaker_1120:
#   tilt 직전 T_in 을 놓기 자세로 옮기면 왼팔 FK 자세 오차로 실제 shaker 가 FK⊗T_in 보다 9~11 mm 벗어난다 · 이송 끝 FP++ 는 영상과 1~3 mm.
#   정지 창 = ReadyCarry.hold(settle + 지연 tin_live.FPP_LATENCY_S + 창) 동안 FP++ 갱신분 · 팔 관절을 모아 tin_live.measure(실행기와 같은 식)로 다시 잰다.
TIN_REFRESH_WINDOW_S = 0.6  # 지연 뒤 표본 창(FP++ 약 6~10 Hz → 갱신 4~6 개)
# ★11.20 사용자 "FP++ 가 추론할 수 있는 시간을 충분히 주고 진행" — 팔 멈춤 확인 뒤 이만큼 hold 만 내고, 그 뒤 찍힌 영상(stamp)만 표본에 넣는다
TIN_REFRESH_SETTLE_S = 1.0      # 노드 파라미터 tin_refresh_settle_s 기본
TIN_REFRESH_SETTLE_MAX_S = 2.0  # 파라미터 상한 — 실행기 place 시간 한도(episode_runner TIN_REFRESH_EXTRA_S)가 이 값으로 잡혀 있다
TIN_REFRESH_MIN_FRESH = 3
TIN_REFRESH_POS_DEV_M = 0.002   # 표본별 T_in 위치 흩어짐 상한(실행기 2 s 창은 tin_live.JIT_POS_M 5 mm)
TIN_REFRESH_MAX_DPOS_M = 0.025  # 옛 T_in 과의 차 상한 — 진단 9~11 mm, 넘으면 다른 물체 · 잘못 잡은 FP++ 로 보고 옛 값
TIN_REFRESH_MAX_DROT_DEG = 10.0  # 옛 rel_q 와의 컵 축(z 열) 기울기 차 상한 — 넘으면 다른 물체 · 잘못 잡은 FP++ 로 보고 옛 값(축 둘레 차는 안 봄)
TIN_REFRESH_QUAT_NORM_TOL = 0.01  # 새 rel_q 노름이 1 에서 이만큼 넘게 벗어나면 계산이 깨진 것
CARRY_FPP_XY_TOL_MAX_M = 0.05
RECARRY_PLAN_WARN_S = 0.15       # 보정 이송 계획(rc.start)은 60 Hz 틱 안에서 동기로 돈다 — pd watchdog 0.25 s 전에 경고
#   CPU 실측(11.20, left_n09_e3800 · torch 1 스레드 = JointPolicy 기본 · 부하 14~18): 60~63 ms. 스레드를 코어 전부로 두면 0.5~3.9 s.


def holder_candidates(text: str, first: int) -> list[int]:
    """★10.28 holders:="1,2" → 구독할 실물 홀더 번호(시작 holder 를 늘 넣는다 · 정렬). 빈 문자열이면 시작 holder 하나."""
    try:
        got = {int(x) for x in str(text).split(",") if x.strip()}
    except ValueError as exc:
        raise PourFjNodeError(f"holders:={text!r} 는 쉼표로 가른 홀더 번호여야 한다") from exc
    return sorted(got | {int(first)})


def opened_both(chain) -> bool:
    """★11.11 이 틱에 정책 손 목표와 실측 손가락이 둘 다 열렸다(네 손가락 평균, 인계 · 정책 시작 손 기준)."""
    lim = max(float(chain.open_min_rad), OPENED_END_MIN_RAD)   # ★11.18 래치 전 종료만 높인 문턱
    return chain.opened >= lim and chain.opened_meas >= OPENED_MEAS_FRAC * lim


def hand_released(chain, release_force_n: float) -> bool | None:
    """★11.11 손이 컵을 놓았다 — 손끝 촉각 < release_force_n 이고 관절 힘 < joint_free_g(rh_place.is_free · episode_ros 8d9b873 문턱).
    sim 1111 place_blue: 0.16~0.18 rad 로 열린 채 손 힘 22~44 N 으로 컵을 쥐고 있었다. 촉각 0·NaN(토픽 없음 → pour_fj_node 가 0 을 넣음)과
    관절 힘 None·NaN 은 신호 없음. 둘 다 없으면 None — 부르는 쪽이 각도만으로 판단한다."""
    tact, jf = chain.tact_max, chain.jf_max
    if jf is not None and not np.isfinite(jf):           # 채널 하나라도 NaN(np.max 전파) = 모름 = 쥔 것(is_free 와 같은 fail-closed)
        return False
    has_t = tact is not None and bool(np.isfinite(tact)) and tact > 0.0
    has_j = jf is not None and bool(np.isfinite(jf))
    if not (has_t or has_j):
        return None
    return not ((has_t and tact >= release_force_n) or (has_j and jf >= chain.joint_free_g))


def tin_minus_offset(rel_p, rel_q, palm_R, holder, side: str) -> tuple[tuple, np.ndarray | None]:
    """★11.16 T_in(카메라 컵 ⊗ FK 손바닥) → FK 틀: 컵 rel_p − R_palmᵀ·t, 홀더 − t(t = 팔별 CUP_OBS_OFFSET_BASE_M, base m).
    홀더도 같이 빼서 컵 − 홀더 관계(둘 다 카메라가 본 값)는 그대로 두고 손 안 자세만 FK 기준으로 옮긴다. R_palm = reset 때 실측."""
    from policy_control.episode_spec import CUP_OBS_OFFSET_BASE_M
    t = np.asarray(CUP_OBS_OFFSET_BASE_M[{"r": "right", "l": "left"}.get(side, side)], float)
    rel = np.asarray(rel_p, float).reshape(3) - np.asarray(palm_R, float).reshape(3, 3).T @ t
    return (rel, rel_q), None if holder is None else np.asarray(holder, float).reshape(3) - t


def tin_refresh(live, arm, fk, t0: float, t1: float, side: str, old_rel, sub_palm_R=None,
                stamp_min: float = -np.inf) -> tuple[dict, tuple | None]:
    """★11.20 이송 끝 정지 창 표본 → (기록, 새 붙인 자세 (rel_p, rel_q) 또는 None = 옛 T_in 그대로).
    live = [(받은 시각, FP++ stamp, (pos3, quat wxyz))] · arm = [(시각, 관절 7)] · fk = tin_live.palm_fk(side). 창 [t0 + 지연, t1]
    (받은 시각) · stamp ≥ stamp_min(settle 대기 끝의 시스템 시계 — 멈춘 뒤 settle 이 지나서 찍힌 영상만).
    틀 · 계산 = 실행기 tilt 직전 T_in(tin_live.measure, z_off = Z_OFF → load_tin 의 to_origin), tin_subtract_offset 면 sub_palm_R 로 뺀다.
    판정 = measure ok(갱신 수 · 비율 · 흔들림 · 손바닥–컵) + 위치 흩어짐 ≤ TIN_REFRESH_POS_DEV_M + 창 안 FP++ 컵이 탁자 위 ·
    창 기울기 중앙값 ≤ TIN_FPP_TILT_DEG(넘는 프레임은 measure 에서 뺌) + 새 rel 유한 · quat 노름 1 + 옛 값(old_rel = (rel_p, rel_q))과 차 ≤ TIN_REFRESH_MAX_DPOS_M ·
    컵 축 기울기 ≤ TIN_REFRESH_MAX_DROT_DEG. 통과하면 tin_live.axis_aligned 로 위치 · 컵 축 = 새 값, 축 둘레 = 옛 값을 채택한다
    (★11.20 shaker 축대칭 — 축 둘레 12.9° 차로 정상 측정을 버린 회차, d_rot 13.44 · 컵 축 3.7° · 축 둘레 12.9°)."""
    X = np.array([f[2] for f in live], float).reshape(-1, 7)
    t, hs = np.array([f[0] for f in live], float), np.array([f[1] for f in live], float)
    # ★11.20 FP++ 축 오류(서 있는 shaker 를 127.8°) 한 프레임이 measure 흔들림 검사를 깨지 않게 선 프레임만 잰다. 기울기 판정은 창 중앙값.
    tilts = np.array([TL.tilt_deg(x[3:]) if np.isfinite(x).all() and abs(np.linalg.norm(x[3:]) - 1.0) <= TIN_REFRESH_QUAT_NORM_TOL
                      else np.nan for x in X])
    up = tilts <= TIN_FPP_TILT_DEG                    # NaN(깨진 quat)은 False
    win = (t >= t0 + TL.FPP_LATENCY_S) & (t <= t1) & (hs >= stamp_min)
    try:
        res = TL.measure(t[up], hs[up], X[up], np.array([a[0] for a in arm], float),
                         np.array([a[1] for a in arm], float).reshape(-1, 7), fk, t0 + TL.FPP_LATENCY_S, t1,
                         z_off=TL.Z_OFF[side], min_fresh=TIN_REFRESH_MIN_FRESH, near_m=TL.NEAR_M_SIDE[side], hs_min=stamp_min)
    except Exception as exc:                          # noqa: BLE001 — 측정 코드가 무엇을 던지든 실패 사유로(옛 T_in 그대로)
        res = {"ok": False, "reason": f"측정 실패({exc!r})"}
    rec = {k: v for k, v in res.items() if k != "T_in"}
    if (win & ~up).any():
        rec["n_tilted_dropped"] = int((win & ~up).sum())
    bad = [res.get("reason", "")] if not res.get("ok") else []
    if win.any():                                     # 과반이 기울면(중앙값) 거부 — measure 가 실패해도 사유에 남긴다
        tilt = float(np.median(np.where(np.isnan(tilts[win]), 180.0, tilts[win])))   # 깨진 quat 은 기운 쪽으로 센다
        if tilt > TIN_FPP_TILT_DEG:
            bad.append(f"FP++ 컵 기울기 중앙값 {tilt:.1f}° > {TIN_FPP_TILT_DEG:.0f}°")
    if res.get("ok"):
        if res["pos_dev_mm_max"] > TIN_REFRESH_POS_DEV_M * 1000:
            bad.append(f"위치 흩어짐 {res['pos_dev_mm_max']} mm > {TIN_REFRESH_POS_DEV_M * 1000:.0f}")
        zmin = X[win, 2].min()
        if zmin <= HOLDER_FLOOR_Z:
            bad.append(f"FP++ 컵 z 최소 {zmin:.3f} m(탁자 위 아님)")
    rel = None
    if not bad:
        T = np.asarray(res["T_in"], float)
        rel = TL.to_origin(T[:3, 3], TL.R_to_quat(T[:3, :3]), TL.Z_OFF[side])
        if sub_palm_R is not None:
            rel, _ = tin_minus_offset(rel[0], rel[1], sub_palm_R, None, side)
            rec["tin_subtract_offset"] = True
        p, q = np.asarray(rel[0], float).reshape(-1), np.asarray(rel[1], float).reshape(-1)
        if p.size != 3 or q.size != 4 or not np.isfinite(np.r_[p, q]).all():
            bad.append(f"새 T_in 이 유한한 위치 3 · quat 4 가 아니다(p {p.tolist()} · q {q.tolist()})")
        elif abs(float(np.linalg.norm(q)) - 1.0) > TIN_REFRESH_QUAT_NORM_TOL:
            bad.append(f"새 T_in quat 노름 {np.linalg.norm(q):.4f} ≠ 1")
        else:
            d = (p - np.asarray(old_rel[0], float).reshape(3)) * 1000
            qo = np.asarray(old_rel[1], float).reshape(4)
            rot = float(np.degrees(2 * np.arccos(np.clip(abs(q @ qo) / (np.linalg.norm(q) * np.linalg.norm(qo)), 0.0, 1.0))))
            Tn, To = np.eye(4), np.eye(4)
            Tn[:3, :3], Tn[:3, 3], To[:3, :3] = TL.quat_to_R(q / np.linalg.norm(q)), p, TL.quat_to_R(qo / np.linalg.norm(qo))
            try:
                Ta, axis = TL.axis_aligned(Tn, To)            # 위치 · 컵 축 = 새 값, 축 둘레 = 옛 값
                spin = TL.ang_deg(Ta[:3, :3], Tn[:3, :3])     # 새 값과 남은 차 = 축 둘레 회전만
            except TL.TinError as exc:                        # 축이 거의 반대
                Ta, axis, spin = None, float("nan"), float("nan")
                bad.append(f"옛 T_in 과 {exc}")
            rec.update(d_mm=np.round(d, 1).tolist(), d_norm_mm=round(float(np.linalg.norm(d)), 1), d_rot_deg=round(rot, 2),
                       d_axis_deg=None if np.isnan(axis) else round(axis, 2),     # NaN 은 JSON 기록에서 None
                       d_spin_deg=None if np.isnan(spin) else round(spin, 2))
            if not np.linalg.norm(d) <= TIN_REFRESH_MAX_DPOS_M * 1000:
                bad.append(f"옛 T_in 과 {np.linalg.norm(d):.1f} mm > {TIN_REFRESH_MAX_DPOS_M * 1000:.0f}")
            if Ta is not None and not axis <= TIN_REFRESH_MAX_DROT_DEG:   # NaN(옛 q 가 깨짐)도 거부
                bad.append(f"옛 T_in 과 컵 축 기울기 {axis:.1f}° > {TIN_REFRESH_MAX_DROT_DEG:.0f}°(축 둘레 {spin:.1f}° 는 안 봄)")
            if not bad:
                qa = TL.R_to_quat(Ta[:3, :3])
                rel = (p, -qa if qa @ qo < 0 else qa)   # 옛 rel_q 와 같은 반구(정책 관측 quat 부호 연속)
        if bad:
            rel = None
    return {**rec, "ok": not bad, **({"reason": " · ".join(b for b in bad if b)} if bad else {})}, rel



def _cup_with(palm_pos, palm_R, rel) -> tuple[np.ndarray, np.ndarray]:
    """PlaceChain.cup_from_palm 과 같은 식 — chain.rel 대신 주어진 (rel_p, rel_q)로(새 T_in 을 넣기 전에 잰다)."""
    R = np.asarray(palm_R, float)
    return np.asarray(palm_pos, float) + R @ np.asarray(rel[0], float), _quat_mul(_matrix_to_quat(R), rel[1])

def start_err_with(rc, arm_q, cup_pos, cup_quat) -> tuple[float, float]:
    """★11.20 새 T_in 컵으로 본 이송 끝 시작 문턱 값(앵커 오차 m · 축 deg) — rc.start 와 같은 T_to 식 · _meas_err 그대로.
    shortcut: place_ready_carry 는 이번 범위 밖이라 rc.T_to 를 잠깐 바꿔 _meas_err 를 빌린다, 그 파일을 고칠 때 공개 메서드로."""
    torch = rc.torch
    q = torch.tensor([list(np.asarray(arm_q, float))], dtype=torch.float32)
    Tc = torch.eye(4).repeat(1, 1, 1)
    Tc[0, :3, :3] = torch.tensor(RCY.quat_matrix(cup_quat), dtype=torch.float32)
    Tc[0, :3, 3] = torch.tensor(np.asarray(cup_pos, float), dtype=torch.float32)
    old, rc.T_to = rc.T_to, torch.linalg.inv(rc.ch.fk(q)) @ Tc
    try:
        return rc._meas_err(q)
    finally:
        rc.T_to = old


class PlaceNode(PourFjNode):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from policy_control.raw_poll import poll_subscription

        for name, default in (("holder", -1), ("holders", ""), ("holder_topic", ""), ("allow_measured_start", False),
                              ("joint_free_g", JOINT_FREE_G), ("release_open_min_rad", OPEN_MIN_RAD), ("require_grasp", True),
                              # ★10.22 newton_05 — 넘김 묶음 · 홀더 범위
                              ("bundle", "auto"), ("bundle_max_age_s", 900.0), ("bundle_hand_tol_rad", 0.05),
                              ("holder_tol_m", HOLDER_TOL_M),
                              # ★11.04 사용자 "place 는 묶은 값만": 계약이 cup_obs live 여도 FP++ 를 넣지 않고 손바닥 FK ⊗ 묶음만.
                              #   실기 131817 — tilt 뒤 FP++ 가 쥔 컵을 놓치고 탁자 아래 자세를 내 놓기가 멈췄다. true 면 live(5 cm 문)
                              ("place_live_cup", False),
                              # ★11.07 실행기가 job 마다 넣는다: attach_source:=tin_json(빈 값 = 계약) · tin_json:=런 폴더 T_in json
                              ("attach_source", ""), ("tin_json", ""),
                              # ★11.16 true 면 tin_json T_in 에서 카메라−FK 오프셋 t 를 뺀다(tin_minus_offset). 기본 false = 기존 그대로
                              ("tin_subtract_offset", False),
                              # ★11.18 사용자 "place는 x를 조금만 덜 이송시키면 될것 같고?" — 이송 · 정책 목표 홀더만 base 로 옮긴다
                              #   (DOUBLE_ARRAY, 기본 0). 홀더 범위 검사 · 실행기 cup_in_holder 판정은 실물 홀더 그대로
                              ("holder_offset_base_m", [0.0, 0.0, 0.0]),
                              # ★11.20 사용자 왼손 place 이송 끝 FP++ 재측정 · 보정(1 · 2번) — 기본 끔 = 기존 그대로. tin_json 일 때만.
                              #   1 tin_refresh_at_carry_end: 이송 끝 정지 창에서 T_in 을 다시 재 붙인 자세를 바꾼다
                              #   2 carry_fpp_xy_tol_m(> 0): 새 T_in 컵 xy ↔ 목표 홀더 xy 가 넘으면 ReadyCarry 한 번 더(최대 1회) 뒤 다시 잰다
                              ("tin_refresh_at_carry_end", False), ("carry_fpp_xy_tol_m", 0.0),
                              # ★11.20 멈춤 확인 뒤 표본 전 settle 대기(s, 0~TIN_REFRESH_SETTLE_MAX_S) — 재-ReadyCarry 뒤 재측정도 같다
                              ("tin_refresh_settle_s", TIN_REFRESH_SETTLE_S)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if not bool(p("require_grasp")):                   # fake 손은 촉각이 0 — 리허설에서만 끈다
            self.fam = Family(**{**self.fam.__dict__, "refusals": lambda _c, _m, _t: []})
        self.holder_id = self._check_holder_id(int(p("holder")))
        self.holder_tol = holder_tol_of(p("holder_tol_m"))    # ★10.29 reset 과 같은 검사(0 < 값 ≤ 0.10 m)
        self.bundle_param = str(p("bundle") or "auto").strip()
        self.bundle_max_age = float(p("bundle_max_age_s"))
        self.bundle_hand_tol = float(p("bundle_hand_tol_rad"))
        self._bundle: PH.Bundle | None = None                # 이번 에피소드의 넘김 묶음(attach_source bundle) 또는 T_in(tin_json)
        self._tin = False                                    # ★11.07 이번 reset 의 붙인 자세가 tin_json 인가
        self._bundles_used: list = []
        self._holder_excess: dict = {}
        self._latch_est = None
        if self.contract.latch_est:                          # ★10.22 래치 추정기 — 정책 폴더 latch/ (md5 핀)
            from policy_control.place_latch_est import LatchEst
            root = Path(str(p("contract"))).resolve().parent
            self._latch_est = LatchEst(root / self.contract.latch_est["file"], self.contract.latch_est["md5"], self.contract.obs_dim)
        self._rc_chain = None
        if self.contract.ready_carry:                        # ★10.25 Mover 사슬(고정 hdgp 사본 URDF) — 한 번만 읽는다
            from policy_control import place_ready_carry as RCY
            s = self.contract.side()
            self._rc_chain = RCY.load_chain(s.palm_body, list(s.arm_joints))
            self.get_logger().info(f"준비 자세 이송 켬 · {self.contract.ready_carry}")
        self._carry_t0: float | None = None
        self._rf: dict | None = None        # ★11.20 열린 정지 창(표본) — None = 없음
        self._rf_want = self._rf_recarried = False
        self._rf_tol = 0.0
        self._rf_settle = TIN_REFRESH_SETTLE_S
        self._rf_fk = None                  # tin_live.palm_fk(팔) — 켠 첫 reset 에 한 번 만든다(URDF 읽기)
        self._opened_n = 0
        self.holder_topic = str(p("holder_topic")) or f"/objects/cup_holder_{self.holder_id}/pose"
        # ★10.28 칵테일(같은 팔이 홀더 1 · 2 에 놓는다) — 그 팔 놓기 job 의 홀더를 모두 구독해 두고 reset 에서 고른다
        #   (holders:=1,2, 미션 생성기가 넣는다 · 없으면 holder 하나). holder_topic 은 시작 holder 에만 쓴다.
        self._holder_topics = {k: (self.holder_topic if k == self.holder_id else f"/objects/cup_holder_{k}/pose")
                               for k in holder_candidates(str(p("holders")), self.holder_id)}
        self._holder_poses: dict[int, np.ndarray] = {}
        self.attach = {}                                   # 붙은 컵은 PlaceChain 이 든다(인계 순간 한 번, sim attached)
        self._holder: np.ndarray | None = None
        self._held_q: dict[str, float] = {}
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for k, topic in self._holder_topics.items():
            self.create_subscription(PoseStamped, topic, lambda msg, k=k: self._on_holder(msg, k), latched)
        self._JointState = JointState
        self._poll_held = poll_subscription(self._poll, JointState, f"{NS}/joint_target",
                                            QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
        c = self.contract
        if c.latch_s2r:                                    # ★10.22 newton_05 — 놓음은 래치 뒤 시간(촉각 · 관절 힘 안 봄)
            self.get_logger().info(
                f"place · 홀더 {self.holder_id} ({self.holder_topic}) · 래치 {c.latch_debounce} 틱(추정기 "
                f"{'켬' if self._latch_est is not None else '없음'}) → 손 펴기 {c.release_timed_steps} 틱 → 놓음 → 빼기 "
                f"{c.withdraw} · 붙인 자세 {c.attach_source}(bundle:={self.bundle_param}, {self.bundle_max_age:.0f} s · "
                f"손 {self.bundle_hand_tol} rad 안) · 홀더 범위 {c.holder_range} ± {self.holder_tol} m")
        else:
            self.get_logger().info(f"place · 홀더 {self.holder_id} ({self.holder_topic}) · 놓음 손끝 < "
                                   f"{c.release_force_n} N · 관절 힘 < {float(p('joint_free_g'))} g · 손 목표가 "
                                   f"{float(p('release_open_min_rad'))} rad 넘게 열린 뒤")

    def _check_holder_id(self, k: int) -> int:
        """홀더 번호 → 쓸 번호. holder_range 계약은 실물 번호(≥ 0)를 받아 자세 범위로 거르고(reset), 옛 계약은 학습 목표 홀더만."""
        c = self.contract
        if c.holder_range:                                 # ★10.22 학습 목표 = sim 슬롯 번호 — 실물 번호는 자세 범위로 거른다
            if k < 0:
                raise PourFjNodeError("holder:=<실물 홀더 번호> 가 필요하다(학습 목표 홀더 번호는 sim 슬롯이라 실물 번호가 아니다) — "
                                      f"학습 범위 x {c.holder_range['x']} · y {c.holder_range['y']}")
            return k
        h = c.target_holders[0] if k < 0 else k
        if h not in c.target_holders:
            raise PourFjNodeError(f"holder {h} 는 학습 목표 홀더 {c.target_holders} 가 아니다")
        return h

    def _select_holder(self) -> None:
        """★10.28 같은 팔이 두 홀더에 놓는 칵테일(파랑 → 1 · 분홍 → 2): reset 마다 holder 파라미터를 다시 읽는다 — 실행기가
        reset 전에 넣는다(episode_ros.job_params). 자세는 시작 때 holders:= 의 홀더를 모두 구독해 두고 고른다(reset 은 콜백을
        기다릴 수 없다). 바꿀 수 없으면 PourFjNodeError — reset 이 움직이기 전에 거부한다.
        ★10.29 holder_tol_m 도 job 마다 다시 읽는다(카드가 분홍 → 홀더 2 에만 넓힌 허용 · 다른 job 은 0.02)."""
        try:
            tol = holder_tol_of(self.get_parameter("holder_tol_m").value)
        except RhPlaceError as exc:
            raise PourFjNodeError(str(exc)) from exc
        k = self._check_holder_id(int(self.get_parameter("holder").value))
        if k not in self._holder_topics:
            raise PourFjNodeError(f"홀더 {k} 는 이 노드가 구독하지 않는다(holders {sorted(self._holder_topics)}) — "
                                  "미션 생성기가 그 팔 놓기 job 의 홀더를 holders:= 로 다 넣는다")
        if k != self.holder_id:
            pose = self._holder_poses.get(k)
            if pose is None:
                raise PourFjNodeError(f"홀더 {k} 자세({self._holder_topics[k]})를 아직 못 받았다 — cup_holders 단계가 내는지 본다")
            self.get_logger().info(f"홀더 바꿈 {self.holder_id} → {k} {np.round(pose, 4).tolist()}")
            self.holder_id, self._holder_excess = k, {}
        self._holder = self._holder_poses.get(k, self._holder)
        self.holder_tol = tol

    # ---------------------------------------------------------------- inputs
    def _on_holder(self, msg, k: int | None = None) -> None:
        k = self.holder_id if k is None else k
        try:
            pos = np.asarray(codec.decode_pose(msg).pos, float)
        except (codec.CodecError, ValueError) as exc:
            self._errors["holder"] = str(exc)
            return
        self._holder_poses[k] = pos
        if k == self.holder_id:
            self._holder = pos

    def _drain(self) -> None:
        super()._drain()
        from rclpy.serialization import deserialize_message
        from policy_control.raw_poll import take_all
        for raw, _t in take_all(self._poll_held):             # 여러 팔 노드가 같은 토픽에 낸다 — 관절 이름으로 합친다
            msg = deserialize_message(raw, self._JointState)
            self._held_q.update(zip(msg.name, (float(v) for v in msg.position)))

    def _held(self, meas: dict):
        s = self.contract.side()
        if not all(j in self._held_q for j in list(s.arm_joints) + list(s.hand_joints)):
            return None
        arm = np.array([self._held_q[j] for j in s.arm_joints])
        err = float(np.abs(arm - meas["arm"].arm_q).max())
        if err > self.reset_tol:
            raise PourFjNodeError(f"pd 가 붙잡은 마지막 목표가 팔 실측에서 {err:.3f} rad(> {self.reset_tol}) — "
                                  "aglt 바로 뒤가 아니다(그 사이 다른 경로가 돌았다)")
        return arm, np.array([self._held_q[j] for j in s.hand_joints])

    # ---------------------------------------------------------------- 고리
    def _cup_for(self, role: str, advance: bool):
        if getattr(self.chain, "rel", None) is not None:
            if self._tin:                                  # ★11.07 tin_json — 제어는 FK ⊗ T_in 만, FP++ 는 차이 기록만
                return lambda palm_pos, palm_R, _tact, _st: self._tin_cup(role, palm_pos, palm_R)
            if self.contract.cup_obs == "live":            # ★11.02 'true' 런 — FP++ 신선분을 사슬에 넣고, 없으면 FK(끊김 · 가림)
                use_live = bool(self.get_parameter("place_live_cup").value)

                def live(palm_pos, palm_R, _tact, _st):
                    got = self._fresh_cup(role) if use_live else None
                    self.chain.live_cup = None if got is None else (np.asarray(got[0], float), np.asarray(got[1], float))
                    return self.chain.cup_obs(np.asarray(palm_pos, float), np.asarray(palm_R, float))
                return live
            return lambda palm_pos, palm_R, _tact, _st: self.chain.cup_from_palm(palm_pos, palm_R)
        if self._bundle is not None:                       # ★10.22 reset 측정 — 컵 = 손바닥 FK ⊗ 넘김 묶음(FP++ 안 봄)
            from policy_control.place_latch_est import _quat_mul
            from policy_control.rh_place import _matrix_to_quat
            b = self._bundle
            return lambda palm_pos, palm_R, _tact, _st: (np.asarray(palm_pos, float) + np.asarray(palm_R, float) @ b.rel_p,
                                                         _quat_mul(_matrix_to_quat(np.asarray(palm_R, float)), b.rel_q))
        if self.contract.attach_source == "bundle":       # 묶음 전 — 상태 표시만(이 계약은 FP++ 컵을 쓰지 않는다)
            return lambda palm_pos, _R, _tact, _st: (np.asarray(palm_pos, float), np.array([1.0, 0.0, 0.0, 0.0]))
        return super()._cup_for(role, advance)

    def _tin_cup(self, role, palm_pos, palm_R):
        """★11.07 손바닥 FK ⊗ T_in(정책 관측 · 래치 같은 값). FP++ 가 그럴듯하면(탁자 위 · 기울기 15° 안) 차이만 로그 — 미끄러짐 관찰용."""
        self.chain.live_cup = None
        fk = self.chain.cup_obs(np.asarray(palm_pos, float), np.asarray(palm_R, float))
        got = self._fresh_cup(role)
        if got is not None:
            p, q = np.asarray(got[0], float), np.asarray(got[1], float)
            tilt = TL.tilt_deg(q)
            if p[2] > HOLDER_FLOOR_Z and tilt <= TIN_FPP_TILT_DEG:
                d = (p - fk[0]) * 1000
                self.get_logger().info(f"tin_json FP++ − FK 컵 {np.round(d, 1).tolist()} mm(|{np.linalg.norm(d):.1f}|) · "
                                       f"기울기 FP++ {tilt:.1f}° / FK {TL.tilt_deg(fk[1]):.1f}°", throttle_duration_sec=1.0)
        return fk

    def _attach_source(self) -> str:
        src = str(self.get_parameter("attach_source").value).strip() or self.contract.attach_source
        if src not in ATTACH_SOURCES:
            raise PourFjNodeError(f"attach_source {src!r} — {ATTACH_SOURCES} 중 하나")
        return src

    def _tin_bundle(self) -> PH.Bundle:
        """★11.07 실행기가 넘긴 tilt 직전 T_in json → 붙인 자세(넘김 묶음과 같은 자리로). 이 값에는 cup_obs_offset 를 더하지 않는다 —
        T_in 이 이미 FK 손바닥 → 카메라 컵이라, 묶음의 카메라−FK 보정을 또 더하면 두 번 옮긴다."""
        path = str(self.get_parameter("tin_json").value).strip()
        if not path:
            raise TL.TinError("attach_source tin_json 인데 tin_json 경로가 비었다(실행기가 tilt 직전 T_in 을 못 냈다)")
        side = self.contract.side().side
        rel_p, rel_q, d = TL.load_tin(path, {"r": "right", "l": "left"}.get(side, side))
        return PH.Bundle(path=Path(path), side=PH.SIDES.get(side, side), object_contract=str(d.get("object", "")), rel_p=rel_p, rel_q=rel_q,
                         hand_q=np.zeros(0), valid=True, n_steps=int((d.get("measured") or {}).get("n_fresh") or 0),
                         created=Path(path).stat().st_mtime,
                         meta={"node": d.get("node"), "episode": d.get("episode"), "source": d.get("source"),
                               "fallback_reason": d.get("fallback_reason", ""), "z_off_applied": d.get("z_off_applied")})

    def _attach_info(self) -> dict:
        """사슬 attach_info — 묶음 as_dict, tin_json 이면 출처(measured | fallback) · fallback 사유까지."""
        b = self._bundle
        if b is None:
            return {}
        if not self._tin:
            return b.as_dict()
        return {**b.as_dict(), "attach_source": "tin_json", "source": b.meta.get("source"),
                "fallback_reason": b.meta.get("fallback_reason", ""), "z_off_applied": b.meta.get("z_off_applied")}

    def _hand_now(self) -> np.ndarray:
        s = self.contract.side()
        st = self.srcs[s.role].snapshot(time.monotonic())
        if st.ee_q is None or "ee" in set(st.stale) | set(st.missing):
            raise PourFjNodeError(f"{s.side}: 손 관절 측정이 없다(stale)")
        hand = dict(zip(st.ee_names, st.ee_q))
        return np.array([float(hand[j]) for j in s.hand_joints])

    def _pick_bundle(self) -> PH.Bundle:
        """★10.22 이번 place 의 넘김 묶음 — auto 면 그 팔 잡기 노드의 가장 새 묶음, 아니면 파일 경로."""
        side = self.contract.side().side
        obj = str(self.get_parameter("object").value).strip() or None
        hand = self._hand_now()
        if self.bundle_param == "auto":
            return PH.pick(side, object_name=obj, hand_q=hand, hand_tol_rad=self.bundle_hand_tol,
                           max_age_s=self.bundle_max_age, used=self._bundles_used)
        b = PH.load(self.bundle_param)
        bad = PH.check(b, side=side, object_name=obj, hand_q=hand, hand_tol_rad=self.bundle_hand_tol, max_age_s=self.bundle_max_age)
        if bad:
            raise PH.HandoverError(f"넘김 묶음 {b.path.name}: " + " · ".join(bad))
        return b

    def _srv_reset(self, req, res):
        try:
            self._select_holder()                          # ★10.28 이번 job 의 홀더(실행기가 reset 전에 holder 를 넣는다)
        except PourFjNodeError as exc:
            if self.book.phase == "armed":
                self.book.phase = "idle"
            return self._reply(res, False, [f"reset: {exc}"])
        if self.chain is not None:
            self.chain.rel = None                          # 새 인계 — 컵은 다시 붙인다(FP++ 한 번 또는 넘김 묶음)
        self._bundle = None
        self._tin = False
        try:
            src = self._attach_source()
        except PourFjNodeError as exc:
            if self.book.phase == "armed":
                self.book.phase = "idle"
            return self._reply(res, False, [f"reset: {exc}"])
        if src in ("bundle", "tin_json"):
            self._drain()
            try:
                self._bundle = self._tin_bundle() if src == "tin_json" else self._pick_bundle()
                self._tin = src == "tin_json"
            except (PH.HandoverError, PourFjNodeError, KeyError, TL.TinError, OSError, ValueError) as exc:
                if self.book.phase == "armed":
                    self.book.phase = "idle"
                return self._reply(res, False, [f"reset: {exc}"])
        out = super()._srv_reset(req, res)
        ok = bool(getattr(out, "success", False))
        if ok and self._bundle is not None:
            self._bundles_used.append(str(self._bundle.path))
            self.get_logger().info(f"place 붙인 자세 = {'tilt 직전 T_in' if self._tin else '넘김 묶음'} {self._bundle.as_dict()}"
                                   + (f" · 출처 {self._bundle.meta.get('source')} · 원점 틀로 되돌린 z_off {self._bundle.meta.get('z_off_applied')}" if self._tin else ""))
            if self._tin and self._bundle.meta.get("source") != "measured":
                self.get_logger().warning(f"place 붙인 자세 = 고정 T_in(실측 아님): {self._bundle.meta.get('fallback_reason', '')}")
        return out

    def _reset_chain(self, meas: dict) -> None:
        self.chain.object_name = str(self.get_parameter("object").value).strip() or None   # ★10.12 다물체 place(잡기와 같은 물체)
        self.chain.joint_free_g = float(self.get_parameter("joint_free_g").value)
        self.chain.open_min_rad = float(self.get_parameter("release_open_min_rad").value)
        held = self._held(meas)
        self.chain.fk = self.fk[self.contract.side().role]     # ★10.13 이송: 손바닥 자코비안(실측 FK)
        if self._holder is not None and self.contract.holder_range:   # ★10.22 학습 범위 밖 홀더는 움직이기 전에 거부
            bad, self._holder_excess = holder_refusals(self.contract, self._holder, self.holder_tol)
            if bad:
                raise RhPlaceError(f"홀더 {self.holder_id}: " + " · ".join(bad))
            if any(v > 0 for v in self._holder_excess.values()):
                self.get_logger().warning(f"홀더 {self.holder_id} {np.round(self._holder, 4).tolist()} 가 학습 범위 밖 "
                                          f"{self._holder_excess} m(허용 {self.holder_tol} m 안이라 진행)")
        self.chain.est = self._latch_est
        self.chain.rc_chain = self._rc_chain
        b = self._bundle
        self.chain.attach_info = self._attach_info()
        attach, holder = None if b is None else (b.rel_p, b.rel_q), self._holder
        if self._tin and bool(self.get_parameter("tin_subtract_offset").value):
            attach, holder = tin_minus_offset(b.rel_p, b.rel_q, meas["arm"].palm_R, holder, self.contract.side().side)
            self.chain.attach_info = {**self.chain.attach_info, "tin_subtract_offset": True,
                                      "rel_p_used": [round(float(v), 5) for v in attach[0]],
                                      "holder_used": None if holder is None else [round(float(v), 5) for v in holder]}
            self.get_logger().info(f"tin_subtract_offset · 컵 rel_p {np.round(b.rel_p, 4).tolist()} → {np.round(attach[0], 4).tolist()} · "
                                   f"홀더 {None if self._holder is None else np.round(self._holder, 4).tolist()} → "
                                   f"{None if holder is None else np.round(holder, 4).tolist()}")
        try:                                               # job 마다 실행기가 넣는다(앞 job 값이 남지 않게)
            off = holder_offset_of(self.get_parameter("holder_offset_base_m").value)
        except RhPlaceError as exc:
            raise PourFjNodeError(str(exc)) from exc
        if holder is not None and np.any(off):
            holder = np.asarray(holder, float) + off
            self.chain.attach_info = {**self.chain.attach_info, "holder_offset_base_m": off.tolist(),
                                      "holder_used": [round(float(v), 5) for v in holder]}
            self.get_logger().info(f"holder_offset_base_m {off.tolist()} · 목표 홀더 → {np.round(holder, 4).tolist()}")
        self._rf_setup()
        self.chain.reset(meas, holder=holder, held=held,
                         allow_measured=bool(self.get_parameter("allow_measured_start").value),
                         attach=attach)
        if self.chain.rcarry is not None:
            self.get_logger().info(f"준비 자세 이송 계획 · {self.chain.rcarry.plan}")
        rel = self.chain.rel[0]
        self.get_logger().info(f"place reset · 자리 {self.chain.as_dict()['seat']} · 시작 목표 "
                               f"{'pd 마지막 joint_target' if held is not None else '실측(allow_measured_start)'} · "
                               f"손바닥 기준 컵 {np.round(rel, 3).tolist()}")

    # ---------------------------------------------------------------- ★11.20 이송 끝 T_in 재측정 · 홀더 xy 보정
    def _rf_setup(self) -> None:
        """reset 마다 두 파라미터를 다시 읽는다(실행기가 job 마다 넣는다). 끔이면 아무것도 바꾸지 않는다."""
        self._rf, self._rf_recarried = None, False
        tol = float(self.get_parameter("carry_fpp_xy_tol_m").value)
        if not (np.isfinite(tol) and tol <= CARRY_FPP_XY_TOL_MAX_M):
            raise PourFjNodeError(f"carry_fpp_xy_tol_m {tol} — 유한한 m, ≤ {CARRY_FPP_XY_TOL_MAX_M}(0 이하 = 끔)")
        self._rf_tol = max(tol, 0.0)
        settle = float(self.get_parameter("tin_refresh_settle_s").value)
        if not 0.0 <= settle <= TIN_REFRESH_SETTLE_MAX_S:     # NaN 도 거부
            raise PourFjNodeError(f"tin_refresh_settle_s {settle} — 0~{TIN_REFRESH_SETTLE_MAX_S} s")
        self._rf_settle = settle
        on = bool(self.get_parameter("tin_refresh_at_carry_end").value)
        self._rf_want = on and self._tin and self.chain.rc_chain is not None and bool(self.contract.ready_carry)
        if (on or self._rf_tol > 0) and not self._rf_want:
            self.get_logger().warning(f"tin_refresh_at_carry_end {on} · carry_fpp_xy_tol_m {self._rf_tol} — attach_source tin_json · "
                                      f"준비 자세 이송 계약에서 재측정을 켤 때만 쓴다 → 이번 reset 은 끔")
        if self._rf_want and self._rf_fk is None:
            self._rf_fk = TL.palm_fk(self.contract.side().side)

    def _measure(self, advance: bool = False) -> dict:
        meas = super()._measure(advance)
        rf = self._rf
        if advance and rf is not None:                     # 정지 창 표본 — 팔 관절(이 틱) · FP++ 새 메시지(받은 시각이 바뀐 것만)
            role = self.contract.side().role
            m, now = meas[role], time.monotonic()
            rf["arm"].append((now, np.asarray(m.arm_q, float).copy()))
            rf["palm"] = (np.asarray(m.palm_pos, float), np.asarray(m.palm_R, float), np.asarray(m.arm_q, float))
            got = self.cups.get(role)
            if got is not None and got[0] != rf["last"] and got[3] is not None:
                rf["last"] = got[0]
                rf["live"].append((float(got[0]), float(got[3]), (*np.asarray(got[1], float), *np.asarray(got[2], float))))
        return meas

    def _rf_record(self, rec: dict) -> None:
        self.chain.attach_info = {**self.chain.attach_info, "tin_refresh": [*self.chain.attach_info.get("tin_refresh", []), rec]}

    def _rf_compute(self, rf: dict, n: int) -> tuple[dict, tuple | None, tuple | None]:
        """정지 창 표본 → (기록, 새 붙인 자세 또는 None, palm). chain 은 바꾸지 않는다(_rf_finish 가 한 번에)."""
        side = self.contract.side().side
        old = (np.asarray(self.chain.rel[0], float).copy(), np.asarray(self.chain.rel[1], float).copy())
        palm = rf.get("palm")
        sub = palm[1] if palm is not None and bool(self.get_parameter("tin_subtract_offset").value) else None
        rec, rel = tin_refresh(rf["live"], rf["arm"], self._rf_fk, rf["t0"], time.monotonic(), side, old, sub, rf["stamp_min"])
        rec = {"round": n, **rec, "rel_p_before": [round(float(v), 5) for v in old[0]]}
        new = None if rel is None else (np.asarray(rel[0], float).reshape(3).copy(), np.asarray(rel[1], float) / np.linalg.norm(rel[1]))
        rc = self.chain.rcarry
        if new is not None:                                # 새 rel 로 본 컵이 sim 시작 분포(START_*) 밖이면 옛 T_in 그대로
            if palm is None or rc is None:
                rec.update(ok=False, reason="새 T_in 의 시작 문턱을 잴 손바닥 · 이송이 없다")
                new = None
            else:
                err, ax = start_err_with(rc, palm[2], *_cup_with(palm[0], palm[1], new))
                rec.update(start_err_mm=round(err * 1000, 2), start_axis_deg=round(ax, 2))
                if not (err <= RCY.START_ERR_MAX_M and ax <= RCY.START_AXIS_MAX_DEG):
                    rec.update(ok=False, reason=f"새 T_in 컵이 시작 문턱 밖(오차 {err * 1000:.1f} mm > {RCY.START_ERR_MAX_M * 1000:.0f} "
                                                f"또는 축 {ax:.1f}° > {RCY.START_AXIS_MAX_DEG})")
                    new = None
        if new is not None:
            rec["rel_p_after"] = [round(float(v), 5) for v in new[0]]
        if palm is not None and rc is not None and rc.target is not None:
            cup, _ = _cup_with(palm[0], palm[1], old if new is None else new)
            rec["xy_err_mm"] = round(float(np.linalg.norm(cup[:2] - rc.target[:2])) * 1000, 2)
        return rec, new, palm

    def _rf_finish(self) -> tuple[dict, tuple | None]:
        """정지 창 끝 — T_in 을 다시 재 붙인 자세(chain.rel)를 바꾼다(실패면 그대로 + 경고). 기록은 attach_info tin_refresh.
        새 값은 다 계산한 뒤 한 번에 넣는다 — 계산 중 무엇이 던지든 옛 T_in 그대로."""
        rf, self._rf = self._rf, None
        n = len(self.chain.attach_info.get("tin_refresh", [])) + 1
        try:
            rec, new, palm = self._rf_compute(rf, n)
        except Exception as exc:                          # noqa: BLE001 — 재측정 실패는 옛 T_in 으로 정책(조용히 넘기지 않고 기록)
            rec, new, palm = {"round": n, "ok": False, "reason": f"재측정 처리 실패({type(exc).__name__}: {exc})"}, None, None
        if new is not None:
            self.chain.rel = new
        self._rf_record(rec)
        if new is not None:
            self.get_logger().info(f"이송 끝 T_in 재측정 {n}차 · rel_p {rec['rel_p_before']} → {rec['rel_p_after']} · "
                                   f"차 {rec.get('d_mm')} mm(|{rec.get('d_norm_mm')}|) · 회전 {rec.get('d_rot_deg')}°(축 {rec.get('d_axis_deg')}° · "
                                   f"축 둘레 {rec.get('d_spin_deg')}° 는 옛 값 유지) · "
                                   f"표본 {rec.get('n_fresh')}/{rec.get('n_msgs')} · 흩어짐 {rec.get('pos_dev_mm_max')} mm · "
                                   f"시작 오차 {rec.get('start_err_mm')} mm · 축 {rec.get('start_axis_deg')}° · 컵−홀더 xy {rec.get('xy_err_mm')} mm")
        else:
            self.get_logger().warning(f"이송 끝 T_in 재측정 {n}차 실패 → 기존 T_in 그대로 · {rec.get('reason', '')} · "
                                      f"표본 {rec.get('n_fresh')}/{rec.get('n_msgs')}")
        return rec, palm

    def _carry_end_refresh(self, rc) -> str:
        """이송 끝(start_ok) → 정책 사이. "wait" = 정지 창 또는 보정 이송을 시작했다(정책은 그 뒤) · "ended" = 끝냈다 · "" = 정책. 순서:
        정지 창 → 재측정 → (tol > 0 · 재측정 성공 · 컵−홀더 xy > tol · 보정 전이면) ReadyCarry 한 번 더 → 정지 창 → 재측정 → 정책.
        두 번째에도 tol 을 넘으면 경고만. 팔이 멈춤 확인(still "ok")을 못 했으면 재측정 없이 정책. 꺼져 있으면 늘 ""(기존 그대로)."""
        if self._rf is not None:
            rec, palm = self._rf_finish()
            xy = rec.get("xy_err_mm")
            if self._rf_tol > 0 and rec["ok"] and xy is not None and xy > self._rf_tol * 1000:
                if self._rf_recarried:
                    self.get_logger().warning(f"보정 이송 뒤에도 컵−홀더 xy {xy} mm > {self._rf_tol * 1000:.1f} → 그대로 정책")
                    return ""
                cup = self.chain.cup_from_palm(palm[0], palm[1])
                # 같은 틱에서 방금 목표를 냈다(_on_tick 발행 → _after_step) — 다음 목표까지 공백 ≈ 계획 시간 + 틱
                self.get_logger().info(f"컵−홀더 xy {xy} mm > {self._rf_tol * 1000:.1f} → 보정 이송 계획 시작(이 동안 틱이 멈춘다)")
                t0 = time.perf_counter()
                try:                                        # 이송 계획 · 안전 장치 = 처음 이송과 같은 ReadyCarry
                    plan = rc.start(palm[2], cup[0], cup[1], rc.target)
                except Exception as exc:                    # noqa: BLE001 — 계획이 무엇을 던지든 정책으로 가지 않는다
                    self._end("abort", f"보정 이송 계획 실패({time.perf_counter() - t0:.3f} s): {type(exc).__name__}: {exc}")
                    return "ended"
                plan = {**plan, "plan_s": round(time.perf_counter() - t0, 3)}
                self._rf_recarried, self._rf_want = True, True
                self.chain.attach_info = {**self.chain.attach_info, "tin_refresh_recarry": plan}
                self.get_logger().info(f"보정 이송(1회) 계획 {plan['plan_s']} s · {plan}")
                if plan["plan_s"] > RECARRY_PLAN_WARN_S:
                    self.get_logger().warning(f"보정 이송 계획 {plan['plan_s']} s > {RECARRY_PLAN_WARN_S} s — pd watchdog(0.25 s) 가까움 · "
                                              f"torch 스레드 · CPU 부하를 본다")
                return "wait"
            return ""
        if not self._rf_want:
            return ""
        self._rf_want = False
        if rc.still != "ok":                               # 멈춤 확인 못 한 팔(timeout)의 창은 흔들림이 섞인다 — 재지 않는다
            reason = f"이송 끝 팔 멈춤 {rc.still!r}(ok 아님) → 재측정 건너뜀 · 기존 T_in 그대로"
            self._rf_record({"round": len(self.chain.attach_info.get("tin_refresh", [])) + 1, "ok": False, "reason": reason})
            self.get_logger().warning(reason)
            return ""
        try:
            rc.hold(int(round((self._rf_settle + TL.FPP_LATENCY_S + TIN_REFRESH_WINDOW_S) * float(self.contract.policy_hz))))
        except ReadyCarryError as exc:
            self.get_logger().warning(f"이송 끝 T_in 재측정 못 함 → 기존 T_in 그대로: {exc}")
            return ""
        # 창 시작 = 멈춤 확인 + settle(수신 시각) · 영상 stamp 도 그 뒤(시스템 시계) — 그 전 표본은 measure 가 버린다(hold 는 그동안 계속)
        self._rf = {"t0": time.monotonic() + self._rf_settle, "stamp_min": time.time() + self._rf_settle,
                    "arm": [], "live": [], "palm": None, "last": None}
        return "wait"

    def _after_step(self) -> bool:
        c = self.contract
        if self._carry_t0 is not None:                     # ★10.25 이송 중 — 에피소드 시계는 이송이 끝난 틱부터
            now = time.monotonic()
            rc = self.chain.rcarry
            if rc is not None and rc.on:
                self._t_start = now
                lim = (rc.cfg.max_steps + rc.mv.cfg.settle_steps + RCY_STILL_MAX_TICKS) / float(c.policy_hz) + READY_CARRY_SLACK_S
                if now - self._carry_t0 > lim:
                    self._end("abort", f"준비 자세 이송 시간 초과 {lim:.1f} s · {rc.as_dict()}")
                    return True
                return False
            self._carry_t0 = None
            self._t_start = now
            if rc is not None and not rc.start_ok:         # ★11.10 컵이 sim 시작 분포 밖(place_ready_carry START_*) — 정책 전에 멈춘다
                self._end("abort", f"준비 자세 이송 끝 문턱 미달 → 놓기 정책 시작 안 함 · {rc.as_dict()}")
                return True
            got = self._carry_end_refresh(rc) if rc is not None else ""   # ★11.20 정지 창 · 보정 이송(켰을 때만)
            if got == "ended":
                return True
            if got == "wait":                              # 이송 시계를 다시 센다 — 정책은 창 · 보정 이송 뒤
                self._carry_t0 = now
                return False
            self.get_logger().info(f"준비 자세 이송 끝 → 정책 · {None if rc is None else rc.as_dict()}")
            if rc is not None and rc.still == "timeout":    # ★11.11 멈춤 확인 상한을 넘겼다 — 그대로 진행
                self.get_logger().warning(f"준비 자세 이송 끝 팔이 {RCY_STILL_MAX_TICKS} 틱 안에 멈추지 않음 → 그대로 정책 · {rc.as_dict()}")
        if self.chain.settle_t == 0:                       # 놓음이 난 스텝 — 첫 실기 문턱 정하기용 값
            self.get_logger().info(f"place 놓음 판정 · {self.chain.as_dict()}")
        if c.latch_s2r and self.chain.latch and self.chain.latch_t == 0 and not self.chain.settling and not getattr(self, "_latch_logged", False):
            self._latch_logged = True
            self.get_logger().info(f"place 래치 · {self.chain.as_dict()}")
        if self.chain.done:
            if c.withdraw:
                self._end("stop", f"placed — 래치 {c.release_timed_steps} 틱 뒤 놓음 → 빼기 {self.chain.settle_t} 틱 "
                                  f"({self.chain.outcome}, 손–컵 {self.chain.clearance})")
            else:
                self._end("stop", f"placed — 놓음 {c.release_steps} 스텝 + 스크립트 {c.settle_steps} 스텝 끝")
            return True
        # ★11.11 사용자 "정책 기준으론 손가락을 다 피게 되어있으니까 손가락 다 피면 에피소드 끝" — 6701ae9 가 지운 규칙을 되살리되
        #   정책 손 목표(opened ≥ open_min_rad)와 실측 손가락(opened_meas ≥ OPENED_MEAS_FRAC · open_min_rad)이 같이 OPENED_END_TICKS 연속.
        #   래치가 켜졌으면 손은 래치가 편다 → 기존 놓기 · 빼기 경로. 학습 시간 끝 stop(episode_end)은 그대로.
        if c.latch_s2r and not self.chain.latch and not self.chain.settling and not self.chain.done:
            rel = hand_released(self.chain, self.contract.release_force_n) if opened_both(self.chain) else False
            if rel is None and not getattr(self, "_rel_unknown_logged", False):
                self._rel_unknown_logged = True
                self.get_logger().warning(f"래치 전 손 폄 — 촉각 · 관절 힘 신호 없음(tact {self.chain.tact_max} · jf {self.chain.jf_max}) "
                                          f"→ 각도만으로 판단")
            self._opened_n = self._opened_n + 1 if rel is not False else 0   # 손 힘이 남았으면(쥔 채 벌어짐) 세지 않는다
            if self._opened_n >= OPENED_END_TICKS:
                self._release_force = rel   # ★11.16 끝난 틱의 힘 근거(True 빈손 · None 신호 없음) — 실행기 continue_on_fail
                self.chain.outcome = "opened_before_latch"   # ★11.11 래치 전 — 놓였는지는 실행기가 cup_in_holder 로 판정
                self.chain.released_cup = self.chain.latch_cup
                self.get_logger().info(f"place 정책이 래치 전에 손 폄 → 끝 · {self.chain.as_dict()}")
                self._end("stop", f"opened before latch — 정책이 래치 전에 손을 폄(opened {self.chain.opened:.2f} · "
                                  f"실측 {self.chain.opened_meas:.2f} rad, 기준 {self.chain.open_min_rad:.2f}, {OPENED_END_TICKS} 틱 · "
                                  f"촉각 {self.chain.tact_max} N · 관절 힘 {self.chain.jf_max} g) · "
                                  f"놓임 판정은 cup_in_holder → 다음 노드(home)가 뺀다")
                return True
        if c.latch_s2r and self.max_s > 0 and time.monotonic() - self._t_start >= self.max_s and not self.chain.settling:
            # ★11.10 사용자 "place 노드의 '열리면 끝' 규칙 제거" — 정책을 학습 episode_length_s 끝까지 돌린다. 끝은 abort 가 아니라
            #   정상 종료(stop), 들어갔는지는 에피소드 실행기가 cup_in_holder(FP++ live · 묶은 컵 FK)로 판정한다. 빼기는 다음 rehome 노드가.
            self.chain.outcome = "episode_end"
            self.chain.released_cup = self.chain.latch_cup
            self.get_logger().info(f"place 학습 시간 끝 · {self.chain.as_dict()}")
            self._end("stop", f"placed — 학습 시간 {self.max_s:.1f} s(episode_length_s) 끝 · 래치 {self.chain.latch} · "
                              f"{self.chain.latch_t} 틱 · 손 폄 {self.chain.opened:.2f} rad → 다음 노드(home)가 뺀다")   # 빼기 중 시간 초과는 부모가 stop
            return True
        return False

    def _srv_start(self, req, res):
        self._latch_logged = False
        self._opened_n = 0
        self._rel_unknown_logged = False
        self._release_force = None
        out = super()._srv_start(req, res)
        rc = getattr(self.chain, "rcarry", None)
        self._carry_t0 = time.monotonic() if rc is not None and rc.on else None
        return out

    def _status_extra(self) -> dict:
        return {"holder": self.holder_id, "holder_pose": None if self._holder is None else [round(float(v), 4) for v in self._holder],
                "release_force_n": self.contract.release_force_n,     # 실행기 opened_before_latch 촉각 판정 문턱
                "released_by_force": getattr(self, "_release_force", None),
                **({"holder_excess_m": self._holder_excess} if self.contract.holder_range else {}),
                **({"place": self.chain.as_dict()} if getattr(self.chain, "as_dict", None) else {})}


if __name__ == "__main__":
    raise SystemExit(main(family="rh_place", node_cls=PlaceNode))

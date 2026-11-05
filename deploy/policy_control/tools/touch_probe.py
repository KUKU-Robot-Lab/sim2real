#!/usr/bin/env python3
"""손끝 접촉 보정 재생 — plan_touch.py 의 계획(plan_<side>.npz)을 teach_place 와 같은 길(pd TRACKING 중 /policy_control/joint_target
팔 7 + 손 6, 60 Hz)로 내며, 하강 구간에서만 검지 접촉을 본다. 접촉 = 명령을 실측으로 바꿔 누름을 풀고 → 0.5 s 정지 기록 →
하강 길을 되짚어 복귀. 하강 끝까지 무접촉이면(구멍에 빠짐 · FK 가 높음) 그것을 기록하고 복귀.

    deploy/policy_control/tools/touch_probe.sh --plan logs/analysis/fk_touch_probe_1105/plan_right.npz \\
        --out logs/analysis/fk_touch_probe_1105/rec/right_1.npz [--step] [--execute]
    python3 deploy/policy_control/tools/touch_probe.py --selftest      # 접촉 판정 · 재생 · 중단 순서(가짜 힘 · 가짜 탁자), ROS 없음

--execute 없이는 계획 · pd 상태 · 힘 토픽 · 시작 자세만 읽고 출력한다(발행 0). ★실기 동작 — 사용자 승인 뒤에만 --execute.
키(teach_place 와 같게): space = 멈춤/다시(멈출 때 명령 = 실측, 하강 중 다시면 기준선 다시) · x = 끝(저장 안 함) · q = 저장하고 끝 ·
Enter = --step 에서 다음 대상.
시작 조건: pd 가 그 팔을 TRACKING 으로 붙듦 · 팔 실측이 계획 홈(계약 home_arm)에서 0.05 rad 안 · 손 힘 영점 토픽이 옴
(미션 home_<side> 의 tip_zero 가 joint_forces_zeroed 를 연다) · 계획의 URDF · 홀더 yaml md5 = 지금 파일 · 계획 시간 ≤ --seconds.
손은 시작에 고정 자세(계획 hand)로 1 rad/s 로 옮기고 그 뒤 바꾸지 않는다.

"명령 = 실측" 은 실측 − e0 이다(중력 처짐, 수~수십 mrad. 처짐을 빼지 않으면 그만큼 더 내려앉아 누른다). e0 는 두 가지:
  홈 e0      시작 때 0.5 s 실측 중앙값 − pd 세트포인트. 멈춰 있고(속도 · 분산) |e0| ≤ 0.06 rad 가 아니면 시작 거부.
  e0_local   대상마다 접근점 정지(dwell)에서 같은 식으로 다시 잰 값. 못 재면(3 s 안에 안 멈춤) 그 대상은 내리지 않는다.
             하강 · 정지 · 상승 · hover 이동(검색 격자점 사이) · 그 사이 중단 해제는 이 값을 쓴다. 기록 json 대상마다 e0_local.
             대상마다 첫 정지 측정(보상 전)에서 |e0_local − 홈 e0| 가 관절별로 0.06 rad(손목 j5~j7 0.12)를 넘으면 어딘가에 기대어
             멈춘 것으로 보고 그 대상 거부(e0_outlier). pd 는 적분이 없어 앞 대상에서 이월된 off 는 첫 e0 에 남지 않는다.
             보상 반복 중에는 정지마찰로 관절이 거의 안 움직여 e0 가 −Δoff 만큼 밀리고 기댄 것과 구별이 안 되므로 이 검사를
             하지 않는다(그 단계는 COMP_MAX · COMP_ITERS · comp_limit 이 막는다).
처짐 보상 off(= 명령 − 계획)는 접근점에서 맞추고 다음 대상까지 이어 쓰되, 홈으로 돌아가는 구간에는 더하지 않는다(다음 실행 시작 검사).
off 를 받기 전에 그 대상 남은 구간(접근점 · 하강 끝 · hover 격자점, 홈 복귀 전까지)의 계획 + off 가 관절 한계(+여유) 안인지 보고 밖이면
거부(comp_limit). 그래도 명령이 한계에 잘리면 그 기록에 clipped = true.
--seconds 는 동작 시간이다 — --step Enter 대기 · space 멈춤 시간은 세지 않는다.
기록: touch.json 최상위와 _raw.npz meta(json)에 e0_home · start_wall_ns(기록 t = 0 의 time.time_ns()) · ep_id.
중단(q · x · --seconds · SIGTERM · SIGHUP · Ctrl-C · 예외 · 추종 오차 · hover 접촉): 명령 = 실측 → 하강 · 정지 · 복귀 · hover 이동 중이었으면
지나온 길(하강 · hover 이동)을 되짚어 접근점까지 올린 뒤 끝(x 도 같고 저장만 안 함). pd 이상 · 측정 끊김 · 두 번째 중단 · 복귀 60 s
초과면 명령 = 실측으로 두고 그 자리에서 끝. 측정이 끊겼으면 마지막 유효 실측을 쓴다. 비유한 명령은 보내지 않는다(codec 이 거부).
끝낼 때: 해제 명령을 pd 세트포인트가 따라올 시간(max(0.5, max|Δ명령|/0.25 + 0.2) s)만큼 먼저 보내고 episode stop 을 낸다 —
pd 는 stop 순간의 세트포인트를 붙들고 그 뒤 이 에피소드 목표를 버린다(pd_arm.on_episode · take_target).
추종 오차 |실측 − 명령 − e0| 문턱: e0_local 쓰는 하강 · 정지 구간 0.04 rad(j7 0.07), 그 밖(이동 · 손 · 접근점 정착 전 · 상승 · 복귀)
0.08 rad(손목 j5~j7 0.12). 복귀 중 추종 오차면 그 자리에서 끝내고 종료 코드 3.
끝낼 때 stop 이벤트를 못 냈거나 STOP_WAIT_S 안에 pd target 이 internal 이 안 되면 stderr 에 ★ 로 알리고 종료 코드를 3 이상으로,
기록 info 에 stop_ok = false.

접촉 신호(하나라도 넘으면, 하강 시작 직전 0.3 s 정지 기준선 대비):
  index_force  검지 관절 힘(joint_forces_zeroed[2]) |Δ| ≥ 20 g 가 0.04 s — approach_contact_1104 의 손가락 접촉 기준(FORCE_G · HOLD_S)과
               같다. 손이 멈춘 0.5 s 창 잡음 ptp 중앙 1~2 g · 최대 12 g(11.03~04 접근 8회). 검지를 편 채 축 방향으로 누르는 신호는
               기록이 없어 약할 수 있다 → 아래 둘을 같이 쓴다.
  tip_touch    검지 손끝 촉각(tip_forces_zeroed[1]) ≥ 0.2 N 가 0.04 s — 쉴 때 정확히 0(같은 8회, 닿은 칸만 0.06~14 N).
  arm_lag      팔 추종: FK 손끝 z(실측) − z(명령) 가 하강 0.5~1.5 s 중앙값보다 3 mm 넘게 0.1 s(그 전에는 하강 시작 정지값보다 6 mm).
               막힌 손끝은 남고 명령만 내려가므로 커진다. 정지 잡음 0.2 mm(20261104_calib_handpose2).
하강 직전 기준선이 이미 닿은 값(촉각 ≥ 0.2 N 또는 |검지 관절 힘| > 15 g, 홈 영점 대비)이거나 비유한값이면 그 대상은 내리지 않는다.
힘 값이 비유한값 · 길이 부족이면 끊김으로 본다(0.25 s 넘으면 force_stale).
hover 이동(접근점보다 낮은 이동)에서도 index_force · tip_touch(직전 하강 기준선 대비) · force_stale 을 본다 → 걸리면 중단 · 되짚기.
종료 코드: 0 = 끝 · 2 = 기록 없음 · 3 = pd 이상 · 추종 오차 · hover 접촉 · stop 확인 실패 · 4 = 측정 없음 · 5 = 정지 신호 · 6 = --seconds 지남 ·
1 = 그 밖(시작 거부 포함).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import place_teach as P  # noqa: E402
from policy_control import _paths  # noqa: E402,F401
from policy_control.fk_numpy import UrdfTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_grasp import HAND_STATE, MEAS_WAIT_S, STALE_S, arm_hold_target  # noqa: E402
from replay_to_pd import EPISODE_TOPIC_SIDE, PD_STATUS, STATE_TOPIC, STOP_WAIT_S, TOPIC, replay_episode_id  # noqa: E402
from teach_place import MOVING, arm_limits, pd_fault  # noqa: E402

JF_TOPIC = "/hand_{side}/joint_forces_zeroed"   # rh56f1_state_node(5dc40cf) — 홈 tip_zero 뒤에만 나온다
TIP_TOPIC = "/hand_{side}/tip_forces_zeroed"
INDEX_JF, INDEX_TIP = 2, 1                       # joint_forces 순 엄지굽힘·엄지회전·검지… / tip_forces 순 엄지→새끼
FORCE_G, TIP_N, HOLD_S = 20.0, 0.2, 0.04
PRE_JF_G = 15.0                                  # 하강 직전 |검지 관절 힘|(홈 영점 대비)이 이보다 크면 이미 닿음
LAG_MM, LAG_EARLY_MM, LAG_HOLD_S = 3.0, 6.0, 0.1
LAG_BASE = (0.5, 1.5)                            # 하강 시작 뒤 이 구간 중앙값 = 속도 추종 지연 기준선
DWELL_S, BASE_S, HOLD_AT_S = 0.5, 0.3, 0.5
ASCEND_RATE = 1.0                                # 복귀 = 하강 길 되짚기, 하강과 같은 속도(LAG_CURSOR 보호)
START_TOL_RAD = 0.05
# 추종 오차 |실측 − 명령 − e0| 문턱(관절별. 하강 문턱의 j7 = 손목 회전 — 손끝 z 에 거의 안 들어가 +0.03). 11.05 재검토 N3: 홈 한 점 e0 를
# 모든 자세에 쓰면 자세마다 다른 중력 처짐이 오차로 잡혀 오탐이 난다 → 접근점에서 다시 잰 e0_local 을 쓰는 구간은 좁게,
# 홈 e0 를 쓰는 이동 구간은 자세 차 처짐을 덮게 넓게.
TRACK_TOL_DESC = np.array([0.04] * 6 + [0.07])
TRACK_TOL_MOVE = np.array([0.08] * 4 + [0.12] * 3)   # 손목 j5~j7 kp 10 — 자세 차 처짐 0.06~0.07(11.05 실기)
E0_WIN_S = 0.5                                   # e0 = 이 창의 실측 중앙값 − 명령
E0_VEL_MAX = 0.01                                # rad/s — 창 앞 · 뒤 절반 중앙값 차로 잰 속도, 넘으면 아직 움직인다
E0_STD_MAX = 0.003                               # rad — 창 안 관절별 표준편차 상한
E0_START_MAX = 0.06                              # 시작 홈 max|e0| 상한 — 넘으면 pd 가 못 붙들었거나 어딘가 닿았다
E0_SETTLE_MAX_S = 3.0                            # 접근점 정착 대기 상한 — 넘으면 그 대상 거부(e0_unsettled)
# 대상 첫 정지 측정의 |e0_local − 홈 e0| 관절별 상한 — 넘으면 팔 몸체가 어딘가에 기대어 멈췄다고 보고 그 대상 거부(e0_outlier)
# 11.05 실기: 손목 j5~j7 은 kp 10 이라 자세가 바뀌면 처짐이 0.06~0.07 rad 달라진다(홈 j6 −0.052 → 접근점 +0.01) — 기댄 것이 아니다 → 0.12
E0_LOCAL_DEV = np.array([0.06] * 4 + [0.12] * 3)
# 처짐 보상(접근점): 손끝(실측 FK)이 계획 접근점에서 COMP_TIP_MM 안에 올 때까지 명령을 COMP_GAIN 만큼 옮겨 다시 잰다. 횟수 · 보상 크기 상한을 넘으면 그 대상 거부
COMP_TIP_MM, COMP_GAIN, COMP_ITERS, COMP_MAX = 3.0, 0.6, 8, 0.10
PD_VEL_RAD_S = 0.25                              # pd TRACKING 세트포인트 전진 상한(config/pd_rh56f1_exec.yaml max_vel.reduced · pd_law._advance_setpoint)
RELEASE_MIN_S, RELEASE_PAD_S = 0.5, 0.2          # 끝낼 때 해제 명령 송신 = max(MIN, max|Δ명령|/PD_VEL + PAD)
RETREAT_MAX_S = 60.0
HAND_TOL = np.array([0.2, 0.2, 0.1, 0.2, 0.2, 0.2])   # 검지는 FK 에 바로 들어간다(1° = 2 mm)
HAND_TIMEOUT_S = 8.0
LAG_CURSOR_RAD = 0.05                            # 속도 제한으로 명령이 계획보다 이만큼 뒤지면 계획 시각을 세운다
FORCE_STALE_S = 0.25
SPIN_MAX = 50
DESC_PHASES = ("dwell", "descend", "hold", "ascend")


class ProbeError(ValueError):
    pass


def clean_force(v, index: int) -> np.ndarray | None:
    """힘 배열이 다 유한하고 index 를 담으면 그 배열, 아니면 None(= 끊김). 순수."""
    if v is None:
        return None
    a = np.asarray(v, float).ravel()
    return a if a.size > index and np.all(np.isfinite(a)) else None


def settled_e0(ts, qs, q_cmd) -> tuple[np.ndarray | None, str]:
    """창(E0_WIN_S) 실측이 멈춰 있으면 (중앙값 − 명령, ""), 아니면 (None, 이유). 순수."""
    ts = np.asarray(ts, float)
    span = float(ts[-1] - ts[0]) if len(ts) else 0.0
    if len(ts) < 5 or span < 0.9 * E0_WIN_S:
        return None, f"창 부족({len(ts)} 개 · {span:.2f} s)"
    Q = np.asarray(qs, float)
    if not np.all(np.isfinite(Q)):
        return None, "비유한 실측"
    h = len(ts) // 2
    vel = float(np.abs(np.median(Q[h:], 0) - np.median(Q[:h], 0)).max() / (span / 2))
    sd = float(Q.std(0).max())
    if vel > E0_VEL_MAX or sd > E0_STD_MAX:
        return None, f"안 멈춤(속도 {vel:.3f} rad/s > {E0_VEL_MAX} 또는 sd {sd:.4f} rad > {E0_STD_MAX})"
    return np.median(Q, 0) - np.asarray(q_cmd, float), ""


def start_e0(ts, qs, q_set) -> tuple[np.ndarray | None, str]:
    """시작 홈 e0: 멈춰 있고 max|e0| ≤ E0_START_MAX 여야 한다. 순수."""
    e0, why = settled_e0(ts, qs, q_set)
    if e0 is None:
        return None, f"시작 e0 못 잼 — {why}"
    if float(np.abs(e0).max()) > E0_START_MAX:
        return None, f"시작 e0 max {float(np.abs(e0).max()):.3f} rad > {E0_START_MAX} — pd 가 못 붙들었거나 어딘가 닿았다"
    return e0, ""


def release_s(cmd: np.ndarray, sp: np.ndarray | None) -> float:
    """끝낼 때 해제 명령을 보낼 시간 — pd 세트포인트(추정 sp)가 PD_VEL_RAD_S 로 cmd 까지 오는 시간 + 여유."""
    d = 0.0 if sp is None else float(np.abs(np.asarray(cmd, float) - sp).max())
    return max(RELEASE_MIN_S, d / PD_VEL_RAD_S + RELEASE_PAD_S)


def release_and_stop(probe: "Probe", send, stop_event, spin, pd_target, clock=time.monotonic, sleep=time.sleep) -> tuple[float, bool]:
    """해제 명령(probe.cmd)을 release_s 동안 보낸 뒤 episode stop, pd 가 internal 이 될 때까지(STOP_WAIT_S) 계속 보낸다.
    순서가 중요하다: pd 는 stop 순간의 q_setpoint 를 붙들고(pd_arm.on_episode) 그 뒤 이 에피소드 목표를 버린다(pd_arm.take_target).
    → (해제 시간, stop_ok). stop_ok = stop 이벤트를 냈고 STOP_WAIT_S 안에 pd target 이 internal — 아니면 stderr 에 크게 알린다."""
    def safe(f) -> bool:   # 송신 · spin 한 번이 실패해도 stop 은 반드시 낸다
        try:
            f()
            return True
        except Exception as error:   # noqa: BLE001
            print(f"해제 중 {getattr(f, '__name__', f)} 실패: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            return False

    T = release_s(probe.cmd, probe.sp_est)
    t1 = clock()
    while clock() - t1 < T:
        safe(send)
        safe(spin)
        sleep(probe.dt)
    sent = safe(stop_event)
    t1 = clock()
    while clock() - t1 < STOP_WAIT_S and pd_target() != "internal":
        safe(send)
        safe(spin)
        sleep(probe.dt)
    target = pd_target()
    if not (sent and target == "internal"):
        print("★★★ episode stop 확인 실패 — " + ("stop 이벤트를 못 냈다" if not sent else
              f"stop 뒤 {STOP_WAIT_S:g} s 안에 pd target 이 internal 이 아니다(지금 {target!r})")
              + " — pd 가 이 에피소드 목표를 계속 따를 수 있다. pd 상태를 확인한다 ★★★", file=sys.stderr, flush=True)
    return T, sent and target == "internal"


# ------------------------------------------------------------------ 접촉 판정(순수)
@dataclass
class Detector:
    jf0: float = 0.0
    tip0: float = 0.0
    lag0: float = 0.0
    t0: float = 0.0
    lags: list = field(default_factory=list)
    since: dict = field(default_factory=dict)
    last_force_t: float = 0.0

    def begin(self, t: float, jf0: float, tip0: float, lag0: float) -> None:
        self.jf0, self.tip0, self.lag0, self.t0 = jf0, tip0, lag0, t
        self.lags, self.since, self.last_force_t = [], {}, t

    def _held(self, name: str, on: bool, t: float, need: float) -> bool:
        if not on:
            self.since.pop(name, None)
            return False
        self.since.setdefault(name, t)
        return t - self.since[name] >= need - 1e-9

    def rearm(self, t: float) -> None:
        """기준선은 그대로 두고 지속 시간 · 끊김 시계만 새로 — hover 이동 시작."""
        self.since, self.last_force_t = {}, t

    def touch(self, t: float, jf: float | None, tip: float | None) -> str | None:
        """힘 신호만(index_force · tip_touch · force_stale) — hover 이동에서도 쓴다."""
        if jf is not None and tip is not None:
            self.last_force_t = t
        elif t - self.last_force_t > FORCE_STALE_S:
            return "force_stale"
        if jf is not None and self._held("index_force", abs(jf - self.jf0) >= FORCE_G, t, HOLD_S):
            return "index_force"
        if tip is not None and self._held("tip_touch", tip - self.tip0 >= TIP_N, t, HOLD_S):
            return "tip_touch"
        return None

    def update(self, t: float, jf: float | None, tip: float | None, lag_mm: float) -> str | None:
        """→ 신호 이름(index_force · tip_touch · arm_lag · force_stale) 또는 None."""
        if (sig := self.touch(t, jf, tip)) is not None:
            return sig
        el = t - self.t0
        if LAG_BASE[0] <= el < LAG_BASE[1]:
            self.lags.append(lag_mm)
        if el < LAG_BASE[1] or not self.lags:
            on = lag_mm - self.lag0 >= LAG_EARLY_MM
        else:
            on = lag_mm - float(np.median(self.lags)) >= LAG_MM
        return "arm_lag" if self._held("arm_lag", on, t, LAG_HOLD_S) else None


# ------------------------------------------------------------------ 재생 상태기계(순수)
class Probe:
    """plan npz 를 틱마다 (팔 명령 7, 손 명령 6) 로. tick 은 측정과 힘을 받고, 하강 결과를 results 에 쌓는다.
    단계: hand → (move · dwell → descend → hold → ascend)… → done. 중단은 stop()(하강 · hover 쪽이면 retreat) · halt()(그 자리)."""

    def __init__(self, plan: dict, fk, lower: np.ndarray, upper: np.ndarray, step: bool, dt: float = 1.0 / P.POLICY_HZ,
                 max_vel: float = P.MAX_VEL_RAD_S):
        self.A, self.fk, self.step, self.dt, self.max_vel = plan, fk, step, dt, max_vel
        self.lo, self.hi = lower + P.LIMIT_MARGIN_RAD, upper - P.LIMIT_MARGIN_RAD
        self.hand_goal = np.asarray(plan["hand"], float)
        self.nseg = len(plan["seg_kind"])
        self.idx = [np.flatnonzero(plan["seg"] == s) for s in range(self.nseg)]
        kind = [str(k) for k in plan["seg_kind"]]
        z_app = max((float(plan["tip"][self.idx[s][0], 2]) for s in range(self.nseg) if kind[s] == "descend"), default=np.inf)
        # hover 이동 = 하강 바로 뒤의 이동 중 접근점(하강 시작 최고 높이)보다 2 mm 넘게 낮게 가는 것(--search 격자점 사이 · 마지막 격자점 → 접근점)
        self.hover = {s for s in range(1, self.nseg) if kind[s] == "move" and kind[s - 1] == "descend"
                      and float(plan["tip"][self.idx[s], 2].min()) < z_app - 0.002}
        # 홈 복귀 = 끝점이 계획 첫 점(홈)인 이동 — 대상마다 하나. 여기에는 처짐 보상을 더하지 않는다(다음 실행 START_TOL_RAD 검사)
        self.home_segs = {s for s in range(self.nseg) if kind[s] == "move" and np.allclose(plan["q"][self.idx[s][-1]], plan["q"][0], atol=1e-9)}
        self.det = Detector()
        self.results: list[dict] = []
        self.phase, self.seg, self.tau, self.mark = "hand", 0, 0.0, 0.0
        self.paused, self.waiting = False, False
        self.cmd: np.ndarray | None = None
        self.hand: np.ndarray | None = None
        self.e0 = np.zeros(7)
        self.e0_loc: np.ndarray | None = None      # 이 대상 접근점에서 다시 잰 처짐 — None 이면 홈 e0
        self.last_q: np.ndarray | None = None      # 마지막 유효 실측
        self.sent: tuple[np.ndarray, np.ndarray] | None = None   # 마지막으로 내보낸 (팔, 손) 명령
        self.sp_est: np.ndarray | None = None      # pd 세트포인트 추정 = 보낸 명령을 PD_VEL_RAD_S 로 따라간 값(끝낼 때 해제 시간)
        self.nonfinite = 0
        self.dwell_q: np.ndarray | None = None
        self.off = np.zeros(7)                     # 처짐 보상 = 명령 − 계획. 접근점에서 실측이 계획에 오게 맞추고 다음 대상까지 이어 쓴다
        self.comp_n = 0                            # 이 대상 접근점에서 보상을 다시 맞춘 횟수
        self.comp_tip: float | None = None         # 마지막으로 잰 접근점 손끝 오차 [mm]
        self.stop_why: str | None = None
        self.retreat_fail: str | None = None       # 복귀 중 추종 오차(그 자리에서 끝) — run 이 rc · why 에 반영
        self.clipped = False                       # 마지막 기록 뒤 명령이 관절 한계에 잘렸나 — 다음 기록에 남긴다
        self.buf: list = []
        self.qbuf: list = []
        self.cur: dict = {}

    # 계획 경로 위 시각 tau 의 관절(구간 안 선형 보간)
    def _at(self, seg: int, tau: float) -> np.ndarray:
        i = self.idx[seg]
        tt = self.A["t"][i] - self.A["t"][i[0]]
        return np.array([np.interp(tau, tt, self.A["q"][i, j]) for j in range(7)])

    def _pq(self, seg: int, tau: float) -> np.ndarray:
        """보낼 명령 = 계획 + 처짐 보상(실측이 계획 위에 오게, 홈 복귀 구간은 계획 그대로). 관절 한계(+여유) 안으로 자른다 — 밖이면
        명령이 한계에 막혀 이동 · 접근점 도착 판정이 영영 안 끝난다(11.05 right_5 t3). 잘렸으면 clipped 로 다음 기록에 남긴다."""
        want = self._at(seg, tau) + (0.0 if seg in self.home_segs else self.off)
        q = np.clip(want, self.lo, self.hi)
        self.clipped = self.clipped or not np.array_equal(q, want)
        return q

    def _len(self, seg: int) -> float:
        i = self.idx[seg]
        return float(self.A["t"][i[-1]] - self.A["t"][i[0]])

    def _kind(self, seg: int) -> str:
        return str(self.A["seg_kind"][seg])

    def _e0(self) -> np.ndarray:
        return self.e0 if self.e0_loc is None else self.e0_loc

    def _valid_q(self, q_meas) -> np.ndarray | None:
        """유한한 7 관절 실측이면 그것(마지막 유효 실측으로 기억), 아니면 마지막 유효 실측."""
        if q_meas is not None:
            a = np.asarray(q_meas, float).ravel()
            if a.shape == (7,) and np.all(np.isfinite(a)):
                self.last_q = a.copy()
                return a
        return self.last_q

    def _hold_cmd(self, q_meas: np.ndarray | None) -> np.ndarray:
        """명령 = 실측(처짐 e0 보정) — pd 가 지금 자리를 그대로 붙들어 누름이 없다. 비유한이면 지금 명령 그대로."""
        if q_meas is None:
            return self.cmd
        c = np.clip(np.asarray(q_meas, float) - self._e0(), self.lo, self.hi)
        return c if c.shape == (7,) and np.all(np.isfinite(c)) else self.cmd

    def outgoing(self) -> tuple[np.ndarray, np.ndarray] | None:
        """송신할 (팔, 손). 비유한값이면 None(보내지 않는다 — codec._vec 가 NaN 목표를 거부한다)이고 명령을 마지막 송신값으로 되돌린다."""
        if self.cmd is not None and self.hand is not None and np.all(np.isfinite(self.cmd)) and np.all(np.isfinite(self.hand)):
            self.sent = (self.cmd.copy(), self.hand.copy())
            self.sp_est = self.cmd.copy() if self.sp_est is None else \
                self.sp_est + np.clip(self.cmd - self.sp_est, -PD_VEL_RAD_S * self.dt, PD_VEL_RAD_S * self.dt)
            return self.sent
        self.nonfinite += 1
        if self.sent is not None:
            self.cmd, self.hand = self.sent[0].copy(), self.sent[1].copy()
        return None

    def _tau_of_cmd(self) -> float:
        """하강 구간에서 지금 명령 손끝 높이에 해당하는 계획 시각(되짚기 시작점). 막힌 손끝은 계획보다 위다."""
        i = self.idx[self.seg]
        z, tt = self.A["tip"][i, 2], self.A["t"][i] - self.A["t"][i[0]]
        return float(np.clip(min(self.tau, np.interp(-self.fk(self.cmd - self.off)[2], -z, tt)), 0.0, self._len(self.seg)))

    def start(self, q_set: np.ndarray, q_meas: np.ndarray, hand_meas: np.ndarray) -> None:
        """q_meas = 정지 창 실측 중앙값(start_e0 로 확인한 것) — e0 = q_meas − 명령."""
        self.cmd = np.clip(np.asarray(q_set, float), self.lo, self.hi)
        self.e0 = np.asarray(q_meas, float) - self.cmd
        self.last_q = np.asarray(q_meas, float).copy()
        self.hand = np.asarray(hand_meas, float).copy()

    def go(self) -> None:
        self.waiting = False

    def pause(self, q_meas: np.ndarray) -> None:
        self.paused, self.cmd = True, self._hold_cmd(self._valid_q(q_meas))
        if self.phase == "descend":   # 다시 할 때 그 자리에서 기준선부터(e0_local 은 접근점 값 그대로)
            self.tau = self._tau_of_cmd()
            self.phase, self.dwell_q = "dwell", self.cmd.copy()

    def resume(self, t: float) -> None:
        self.paused = False
        if self.phase == "dwell":
            self.mark, self.buf, self.qbuf = t, [], []

    def stop(self, q_meas: np.ndarray | None, why: str) -> None:
        """중단: 명령 = 실측(끊겼으면 마지막 유효 실측). 하강 쪽(dwell · descend · hold · ascend) · hover 이동이면 지나온 길을
        되짚어 접근점까지(retreat), 아니면 그 자리."""
        self.stop_why = why
        self.cmd = self._hold_cmd(self._valid_q(q_meas))
        self.paused = self.waiting = False
        if self.phase == "hold":
            self._close()
        if self.phase in DESC_PHASES and self._kind(self.seg) == "descend":
            self.tau, self.phase = self._tau_of_cmd(), "retreat"
        elif self.phase == "move" and self.seg in self.hover:
            self.tau, self.phase = min(self.tau, self._len(self.seg)), "retreat"
        else:
            self.phase = "done"

    def halt(self, q_meas: np.ndarray | None, why: str | None = None) -> None:
        if why and not self.stop_why:
            self.stop_why = why
        if self.cmd is not None:
            self.cmd = self._hold_cmd(self._valid_q(q_meas))
        if self.phase == "hold":
            self._close()
        self.phase = "done"

    def _enter(self, seg: int, t: float) -> str | None:
        self.seg, self.tau = seg, 0.0
        if seg >= self.nseg:
            self.phase = "done"
            return "계획 끝"
        kind, tid = self._kind(seg), int(self.A["seg_target"][seg])
        if seg not in self.hover:   # hover 이동은 직전 접근점 e0_local 을 이어 쓴다(같은 높이 · xy ±4 mm)
            self.e0_loc, self.comp_n, self.comp_tip = None, 0, None
        if kind == "descend":
            self.phase, self.mark, self.buf, self.qbuf, self.dwell_q = "dwell", t, [], [], self._pq(seg, 0.0)
        else:
            self.phase = "move"
            if seg in self.hover:
                self.det.rearm(t)
        name = str(self.A["tgt_name"][tid])
        if bool(self.A["seg_gate"][seg]):
            self.waiting = self.step
            return f"대상 {name}" + (" — Enter 로 시작" if self.step else "")
        return None

    def tick(self, t: float, q_meas: np.ndarray, hand_meas: np.ndarray, jf: np.ndarray | None,
             tip: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, str | None]:
        if self.cmd is None or self.hand is None:
            raise ProbeError("start 먼저")
        q_in = np.asarray(q_meas, float).ravel() if q_meas is not None else None
        if q_in is None or q_in.shape != (7,) or not np.all(np.isfinite(q_in)):
            raise ProbeError(f"팔 실측이 비유한 · 모양 틀림: {q_meas}")
        q_meas = self._valid_q(q_in)
        jf, tip = clean_force(jf, INDEX_JF), clean_force(tip, INDEX_TIP)
        jfi = None if jf is None else float(jf[INDEX_JF])
        tpi = None if tip is None else float(tip[INDEX_TIP])
        lag = (self.fk(q_meas)[2] - self.fk(self.cmd)[2]) * 1000
        loc = self.e0_loc is not None
        # 올라가는 길(ascend · retreat)은 누름이 풀리며 처짐이 접근점 값에서 벗어난다 — 11.05 실기 h7 에서 j4 0.040 으로 오탐 → 넓게
        tol = TRACK_TOL_DESC if loc and self.phase not in ("ascend", "retreat") else TRACK_TOL_MOVE
        dev = np.abs(q_meas - self.cmd - self._e0())
        if self.phase != "done" and np.any(dev > tol):
            j = int(np.argmax(dev - tol))
            why = f"추종 오차 j{j + 1} {dev[j]:.3f} rad > {tol[j]:.2f}({'e0_local' if loc else '홈 e0'} 기준)"
            if self.phase == "retreat":
                self.retreat_fail = why
                self.halt(q_meas, why)
                return self.cmd, self.hand, f"복귀 중 {why} — 명령 = 실측, 그 자리에서 끝"
            self.stop(q_meas, why)
            return self.cmd, self.hand, f"{why} — 명령 = 실측" + (", 지나온 길 되짚어 복귀" if self.phase == "retreat" else ", 그 자리에서 끝")
        note, desired = None, self.cmd.copy()
        if self.phase == "hand":
            self.hand = self.hand + np.clip(self.hand_goal - self.hand, -P.HAND_VEL_RAD_S * self.dt, P.HAND_VEL_RAD_S * self.dt)
            ok = np.all(np.abs(np.asarray(hand_meas) - self.hand_goal) <= HAND_TOL) and np.allclose(self.hand, self.hand_goal)
            self.buf = self.buf + [t] if ok else []
            if ok and t - self.buf[0] >= 1.0:
                note = "손 고정 자세 도착 · " + (self._enter(0, t) or "")
            elif t - self.mark > HAND_TIMEOUT_S and self.mark > 0:
                raise ProbeError(f"손이 {HAND_TIMEOUT_S:g} s 안에 고정 자세에 못 갔다: 실측 {np.round(hand_meas, 3).tolist()}")
            if self.mark == 0:
                self.mark = t
            return self.cmd, self.hand, note
        if self.phase == "done" or self.paused or self.waiting:
            return self.cmd, self.hand, None
        if self.phase == "move":
            if self.seg in self.hover and (sig := self.det.touch(t, jfi, tpi)) is not None:
                self.results.append(self._record(t, q_meas, hand_meas, sig, contact=sig != "force_stale") | {"hover": True})
                self.stop(q_meas, f"hover 이동 중 {sig}")
                return self.cmd, self.hand, f"{self.results[-1]['target']}: hover 이동 중 {sig} — 명령 = 실측, 지나온 길 되짚어 복귀"
            if np.abs(self._pq(self.seg, self.tau) - self.cmd).max() < LAG_CURSOR_RAD:
                self.tau += self.dt
            desired = self._pq(self.seg, min(self.tau, self._len(self.seg)))
            if self.tau >= self._len(self.seg) and np.abs(desired - self.cmd).max() < 1e-3:
                note = self._enter(self.seg + 1, t)
        elif self.phase == "dwell":
            desired = self.dwell_q
            self.buf.append((t, jfi, tpi, lag))
            if np.abs(desired - self.cmd).max() < 1e-3:
                self.qbuf.append((t, q_meas))
                if t - self.mark >= DWELL_S:
                    note = self._dwell_done(t, q_meas, hand_meas)
        elif self.phase == "descend":
            sig = self.det.update(t, jfi, tpi, lag)
            if sig is not None:
                note = self._finish(t, q_meas, hand_meas, sig, contact=sig != "force_stale")
            else:
                self.tau += self.dt
                desired = self._pq(self.seg, min(self.tau, self._len(self.seg)))
                if self.tau >= self._len(self.seg):
                    note = self._finish(t, q_meas, hand_meas, None, contact=False)
        elif self.phase == "hold":
            self.cur["samples"].append((q_meas, np.asarray(hand_meas, float), jfi, tpi, lag))
            if t - self.mark >= HOLD_AT_S:
                self._close()
                self.phase = "ascend"
        elif self.phase in ("ascend", "retreat"):
            if np.abs(self._pq(self.seg, self.tau) - self.cmd).max() < LAG_CURSOR_RAD:
                self.tau -= ASCEND_RATE * self.dt
            desired = self._pq(self.seg, max(self.tau, 0.0))
            if self.tau <= 0 and np.abs(desired - self.cmd).max() < 1e-3:
                if self.phase == "retreat":
                    # 하강 시작점 · hover 이동 시작점에 왔다 — 그 앞이 hover 이동이면(검색 격자) 그것도 되짚는다
                    p = self.seg - (1 if self._kind(self.seg) == "descend" else 2)
                    if p >= 0 and p in self.hover:
                        self.seg, self.tau = p, self._len(p)
                    else:
                        self.phase, note = "done", f"접근점까지 복귀 — 끝({self.stop_why})"
                else:
                    note = self._enter(self.seg + 1, t)
        if self.phase != "hold":   # hold 는 명령 = 실측 그대로(누름 0)
            step = np.clip(desired - self.cmd, -self.max_vel * self.dt, self.max_vel * self.dt)
            self.cmd = np.clip(self.cmd + step, self.lo, self.hi)
        return self.cmd, self.hand, note

    def _dwell_done(self, t: float, q_meas, hand_meas) -> str | None:
        """접근점 정지: e0_local 을 아직 안 쟀으면 0.5 s 정지 창으로 잰다(못 재면 E0_SETTLE_MAX_S 까지 기다린 뒤 거부) → 기준선."""
        if self.e0_loc is None:
            win = [b for b in self.qbuf if b[0] >= t - E0_WIN_S]
            e0, why = settled_e0([b[0] for b in win], [b[1] for b in win], self.cmd)
            if e0 is None:
                if t - self.qbuf[0][0] < E0_SETTLE_MAX_S:
                    return None
                return self._refuse(t, q_meas, hand_meas, "e0_unsettled", f"접근점 e0 못 잼 — {why}")
            # 기댐 검사는 대상 첫 측정에서만: pd 는 적분이 없어 이월 off 가 첫 e0 에 안 남는다(right_5_raw.npz). 보상 반복 중에는
            #   정지마찰로 관절이 안 움직여 e0 가 −Δoff 만큼 밀려 기댄 것과 구별이 안 된다 → COMP_MAX · COMP_ITERS · comp_limit 이 막는다
            dev = np.abs(e0 - self.e0)
            if self.comp_n == 0 and np.any(dev > E0_LOCAL_DEV):
                j = int(np.argmax(dev - E0_LOCAL_DEV))
                return self._refuse(t, q_meas, hand_meas, "e0_outlier",
                                    f"접근점 e0 j{j + 1} 가 홈과 {dev[j]:.3f} rad 다름 > {E0_LOCAL_DEV[j]:.2f} — 어딘가에 기댔을 수 있다")
            # 처짐 보상: 실측(= 명령 + e0)이 계획 접근점에 오도록 명령을 옮기고 다시 잰다. 11.05 실기 h7: 보상 없이
            #   관절 0.01~0.03 rad 처짐이 손끝 xy 22 mm 로 번져 구멍을 빗나갔다
            #   판정은 손끝 위치로 한다 — 관절이 정지 마찰로 0.005~0.01 rad 씩 끊겨 움직여 관절 0.004 rad 판정은 못 닿고,
            #   11.05 right_3 에서 손끝 1.9 mm 까지 맞은 뒤 더 고치다 반대로 6.5 mm 넘어갔다 → 손끝 COMP_TIP_MM · 이득 COMP_GAIN
            q0 = self._at(self.seg, 0.0)
            res = self.cmd + e0 - q0
            self.comp_tip = float(np.linalg.norm(self.fk(self.cmd + e0) - self.fk(q0)) * 1000)
            if self.comp_tip > COMP_TIP_MM:
                if self.comp_n >= COMP_ITERS:
                    return self._refuse(t, q_meas, hand_meas, "comp_fail",
                                        f"처짐 보상 {COMP_ITERS} 회 뒤에도 접근점 손끝이 계획에서 {self.comp_tip:.1f} mm")
                off = self.off - COMP_GAIN * res
                if np.abs(off).max() > COMP_MAX:
                    return self._refuse(t, q_meas, hand_meas, "comp_fail",
                                        f"처짐 보상이 {np.abs(off).max():.3f} rad > {COMP_MAX} — 어딘가에 기댔을 수 있다")
                # 보상한 명령이 관절 한계(+여유) 밖이면 명령이 한계에 막혀 접근점에 영영 못 간다(11.05 right_5 t3: j6 0.765 에 걸려 멈춤).
                #   이 off 를 쓸 남은 구간 전부(접근점 · 하강 끝 · hover 격자점 · 접근점 복귀, 홈 복귀 전까지)를 본다
                end = min((s for s in self.home_segs if s > self.seg), default=self.nseg)
                want = self.A["q"][np.concatenate([self.idx[s] for s in range(self.seg, end)])] + off
                over = np.maximum(self.lo - want, want - self.hi)
                if np.any(over > 0):
                    i, j = np.unravel_index(int(np.argmax(over)), over.shape)
                    return self._refuse(t, q_meas, hand_meas, "comp_limit",
                                        f"처짐 보상 명령 j{j + 1} {want[i, j]:+.3f} 가 관절 한계 [{self.lo[j]:+.3f}, {self.hi[j]:+.3f}] 밖(이 대상 남은 구간)")
                self.off, self.comp_n = off, self.comp_n + 1
                self.dwell_q = self._pq(self.seg, 0.0)
                self.mark, self.buf, self.qbuf = t, [], []
                return f"처짐 보상 {self.comp_n} — 접근점 손끝이 계획에서 {self.comp_tip:.1f} mm, 명령을 옮겨 다시 잰다"
            self.e0_loc = e0
        return self._baseline(t, q_meas, hand_meas)

    def _baseline(self, t: float, q_meas, hand_meas) -> str | None:
        win = [b for b in self.buf if b[0] >= t - BASE_S]
        if any(b[1] is None or b[2] is None for b in win):
            return self._refuse(t, q_meas, hand_meas, "force_stale", "기준선 창에 힘 끊김")
        jf0, tip0, lag0 = (float(np.median([b[k] for b in win])) for k in (1, 2, 3))
        if not np.all(np.isfinite([jf0, tip0, lag0])):
            return self._refuse(t, q_meas, hand_meas, "force_bad", f"기준선 비유한값 {jf0} {tip0} {lag0}")
        if tip0 >= TIP_N or abs(jf0) > PRE_JF_G:
            return self._refuse(t, q_meas, hand_meas, "pre_touch", f"이미 닿음(촉각 {tip0:.2f} N · 검지 {jf0:+.0f} g)")
        self.det.begin(t, jf0, tip0, lag0)
        self.phase = "descend"
        return None

    def _record(self, t: float, q_meas, hand_meas, sig: str | None, contact: bool) -> dict:
        tid = int(self.A["seg_target"][self.seg])
        i = self.idx[self.seg]
        return dict(target=str(self.A["tgt_name"][tid]), kind=str(self.A["tgt_kind"][tid]), hole=int(self.A["tgt_hole"][tid]),
                    psi=float(self.A["tgt_psi"][tid]), offset_mm=self.A["tgt_offset_mm"][tid].tolist(),
                    target_xyz=self.A["tgt_xyz"][tid].tolist(), contact=contact, signal=sig, t=round(t, 3),
                    descent_s=round(self.tau, 3), reached_end=self.tau >= self._len(self.seg),
                    cmd_tip_z_mm=round(float(np.interp(self.tau, self.A["t"][i] - self.A["t"][i[0]], self.A["tip"][i, 2])) * 1000, 2),
                    q_at_signal=np.round(np.asarray(q_meas, float), 5).tolist(),
                    hand_at_signal=np.round(np.asarray(hand_meas, float), 4).tolist(),
                    e0_local=None if self.e0_loc is None else np.round(self.e0_loc, 5).tolist(),
                    comp_off=np.round(self.off, 5).tolist(), comp_n=self.comp_n,
                    comp_tip_mm=None if self.comp_tip is None else round(self.comp_tip, 2), clipped=self._take_clipped())

    def _take_clipped(self) -> bool:
        c, self.clipped = self.clipped, False
        return c

    def _refuse(self, t: float, q_meas, hand_meas, sig: str, why: str) -> str:
        self.results.append(self._record(t, q_meas, hand_meas, sig, contact=False) | {"refused": why})
        self.cmd = self._hold_cmd(q_meas)
        self.tau, self.phase = self._tau_of_cmd(), "ascend"
        return f"{self.results[-1]['target']}: 하강 거부 — {why}"

    def _finish(self, t: float, q_meas, hand_meas, sig: str | None, contact: bool) -> str:
        self.cmd = self._hold_cmd(q_meas)   # 누름 풀기: 명령을 실측으로
        self.cur = self._record(t, q_meas, hand_meas, sig, contact) | {"samples": []}
        self.tau = self._tau_of_cmd()
        self.phase, self.mark = "hold", t
        what = {None: "하강 끝까지 무접촉", "force_stale": "힘 신호 끊김 — 멈추고 복귀"}.get(sig, f"접촉({sig})")
        return f"{self.cur['target']}: {what} · 명령 손끝 z {self.cur['cmd_tip_z_mm']:.1f} mm"

    def _close(self) -> None:
        S = self.cur.pop("samples", [])
        if S:
            q = np.median([s[0] for s in S], 0)
            h = np.median([s[1] for s in S], 0)
            jf = [s[2] for s in S if s[2] is not None]
            tp = [s[3] for s in S if s[3] is not None]
            self.cur.update(q_contact=np.round(q, 5).tolist(), hand_contact=np.round(h, 4).tolist(),
                            fk_tip_mm=np.round(self.fk(q) * 1000, 2).tolist(),
                            fk_tip_hand_meas_mm=np.round(self.fk(q, h) * 1000, 2).tolist(),
                            jf_delta_g=round(jf[-1] - self.det.jf0, 1) if jf else None,
                            tip_delta_n=round(tp[-1] - self.det.tip0, 3) if tp else None,
                            lag_mm=round(float(np.median([s[4] for s in S])), 2))
        if self.cur:
            self.results.append(self.cur)
        self.cur = {}


def make_fk(urdf: str, side: str, hand_default: np.ndarray):
    tree = UrdfTree(Path(urdf))
    arm, hand, tip = list(P.arm_names(side)), list(P.hand_names(side)), f"{side[0]}_hl_index_tip"

    def fk(q: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        d = dict(zip(arm, q)) | dict(zip(hand, hand_default if h is None else h))
        return tree.transforms([tip], tree.expand_mimic(d))[tip][:3, 3]
    return fk


def load_plan(path: Path, side: str) -> dict:
    d = dict(np.load(path))
    meta = json.loads(str(d["meta"]))
    if meta["side"] != side or [str(n) for n in d["arm_names"]] != list(P.arm_names(side)) \
            or [str(n) for n in d["hand_names"]] != list(P.hand_names(side)):
        raise ProbeError(f"{path}: 계획 팔 {meta['side']} · 관절 이름이 --side {side} 와 다르다")
    for k in ("urdf", "holders"):
        if f"{k}_md5" not in meta:
            raise ProbeError(f"{path}: 옛 계획(meta 에 {k}_md5 없음) — plan_touch.py 를 다시 돌린다")
        if hashlib.md5(Path(meta[k]).read_bytes()).hexdigest() != meta[f"{k}_md5"]:
            raise ProbeError(f"{meta[k]} 가 계획 뒤 바뀌었다(md5) — plan_touch.py 를 다시 돌린다")
    return d | {"_meta": meta}


def plan_seconds(plan: dict) -> float:
    """무접촉 최악: 계획 + 하강마다 되짚기 · 접근점 정착 상한(E0_SETTLE_MAX_S) · 정지 · 여유 + 손."""
    ds = plan["t"][plan["desc_end"]] - plan["t"][plan["desc_start"]]
    return float(plan["t"][-1] + ds.sum() / ASCEND_RATE + len(ds) * (max(DWELL_S, E0_SETTLE_MAX_S) + HOLD_AT_S + 0.5) + HAND_TIMEOUT_S)


# ------------------------------------------------------------------ ROS 노드(teach_place.run 과 같은 길)
def run(side: str, plan: dict, out: Path, seconds: float, execute: bool, step: bool) -> tuple[int, dict]:
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String

    from robot_control.keys import KeyReader

    arm_names, hand_names = list(P.arm_names(side)), list(P.hand_names(side))
    arm_src = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    rclpy.init()
    node = rclpy.create_node(f"touch_probe_{side}")
    arm, hand, pd, force, cnt = {}, {}, {}, {}, [0]

    def on_arm(m) -> None:
        cnt[0] += 1
        arm.update({n: (p, time.monotonic()) for n, p in zip(m.name, m.position)})

    def on_hand(m) -> None:
        cnt[0] += 1
        hand.update({n: (p, time.monotonic()) for n, p in zip(m.name, m.position)})

    def on_pd(m) -> None:
        cnt[0] += 1
        try:
            d = json.loads(m.data)
        except (ValueError, TypeError):
            return
        if isinstance(d, dict):
            pd.clear()
            pd.update(d, _t=time.monotonic())

    def on_force(key: str):
        def cb(m) -> None:   # Float64MultiArray 에는 header 가 없다 — 받은 시각으로 끊김을 본다
            cnt[0] += 1
            force[key] = (np.asarray(m.data, float), time.monotonic())
        return cb

    node.create_subscription(JointState, STATE_TOPIC, on_arm, qos_profile_sensor_data)
    node.create_subscription(JointState, HAND_STATE.format(side=side), on_hand, qos_profile_sensor_data)
    node.create_subscription(String, PD_STATUS.format(side=side), on_pd, 10)
    node.create_subscription(Float64MultiArray, JF_TOPIC.format(side=side), on_force("jf"), 10)
    node.create_subscription(Float64MultiArray, TIP_TOPIC.format(side=side), on_force("tip"), 10)

    def spin_all() -> None:
        """대기 메시지가 빌 때까지(상한 SPIN_MAX) — 틱마다 콜백이 밀리지 않게."""
        for _ in range(SPIN_MAX):
            n0 = cnt[0]
            rclpy.spin_once(node, timeout_sec=0.0)
            if cnt[0] == n0:
                break

    def fresh(store: dict, names: list) -> np.ndarray | None:
        now = time.monotonic()
        if not all(n in store and now - store[n][1] < STALE_S for n in names):
            return None
        v = np.array([store[n][0] for n in names], float)
        return v if np.all(np.isfinite(v)) else None

    def fresh_force(key: str, index: int) -> np.ndarray | None:
        v = force.get(key)
        return None if v is None or time.monotonic() - v[1] > FORCE_STALE_S else clean_force(v[0], index)

    stop_sig = {"on": None}
    prev_sig = {sg: signal.getsignal(sg) for sg in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    for sg in (signal.SIGTERM, signal.SIGHUP):   # 터미널이 닫혀도(SIGHUP) SIGTERM 과 같이 되짚어 끝낸다
        signal.signal(sg, lambda n, _f: stop_sig.__setitem__("on", signal.Signals(n).name))
    fk = make_fk(plan["_meta"]["urdf"], side, np.asarray(plan["hand"], float))
    lo, hi = arm_limits(side)
    probe = Probe(plan, fk, lo, hi, step)
    rows: list = []
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < MEAS_WAIT_S:
            if stop_sig["on"]:
                return 5, {"phase": "failed", "why": f"정지 신호({stop_sig['on']}) — 시작 전, 아무것도 보내지 않았다"}
            spin_all()
            if fresh(arm, arm_src) is not None and fresh(hand, hand_names) is not None and pd \
                    and fresh_force("jf", INDEX_JF) is not None and fresh_force("tip", INDEX_TIP) is not None:
                break
            time.sleep(0.02)
        ts, qs, t1 = [], [], time.monotonic()   # 홈 e0 창(E0_WIN_S) — 정지 확인 · 중앙값
        while time.monotonic() - t1 < E0_WIN_S + 0.05:
            if stop_sig["on"]:
                return 5, {"phase": "failed", "why": f"정지 신호({stop_sig['on']}) — 시작 전, 아무것도 보내지 않았다"}
            spin_all()
            if (v := fresh(arm, arm_src)) is not None and (not qs or arm[arm_src[0]][1] != ts[-1]):
                ts.append(arm[arm_src[0]][1])
                qs.append(v)
            time.sleep(0.005)
        q_arm, q_hand = (np.median(qs, 0) if qs else None), fresh(hand, hand_names)
        jf, tip = fresh_force("jf", INDEX_JF), fresh_force("tip", INDEX_TIP)
        if q_arm is None or q_hand is None or fresh(arm, arm_src) is None:
            return 4, {"phase": "failed", "why": f"측정 없음 — {STATE_TOPIC} 또는 {HAND_STATE.format(side=side)}"}
        info = {"side": side, "out": str(out), "q_meas": np.round(q_arm, 4).tolist(), "hand_meas": np.round(q_hand, 3).tolist(),
                "home": np.round(plan["home"], 4).tolist(), "home_err_rad": round(float(np.abs(q_arm - plan["home"]).max()), 4),
                "jf_zeroed": None if jf is None else np.round(jf, 1).tolist(),
                "tip_zeroed": None if tip is None else np.round(tip, 3).tolist(), "pd_phase": pd.get("phase"),
                "fk_dz_mm": plan["_meta"].get("fk_dz_mm")}
        why = pd_fault(pd, time.monotonic())
        q_set, why2 = arm_hold_target(pd, side, q_arm)
        e0, why3 = start_e0(ts, qs, q_set) if q_set is not None else (None, "")
        refuse = [w for w in (why, None if q_set is not None else why2, why3 or None,
                              f"팔이 홈이 아니다(최대 {info['home_err_rad']} rad > {START_TOL_RAD}) — home_{side} 먼저"
                              if info["home_err_rad"] > START_TOL_RAD else None,
                              f"{JF_TOPIC.format(side=side)} · {TIP_TOPIC.format(side=side)} 가 안 온다(또는 비유한값) — 홈 tip_zero(미션 home_{side}) 먼저"
                              if jf is None or tip is None else None) if w]
        if refuse:
            return (3 if why else 1), {"phase": "failed", "why": refuse, "state": info}
        if e0 is not None:
            info["sag_e0_rad"] = np.round(e0, 4).tolist()
        if stop_sig["on"]:
            return 5, {"phase": "failed", "why": f"정지 신호({stop_sig['on']}) — 아무것도 보내지 않았다", "state": info}
        if not execute:
            return 0, {"phase": "dry_run", "state": info, "why": "--execute 없이 발행하지 않았다"}
        if not sys.stdin.isatty():
            return 1, {"phase": "failed", "why": "터미널(tty)에서 실행해야 키를 읽는다 — 아무것도 보내지 않았다"}

        pub = node.create_publisher(JointState, TOPIC, 10)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        ep_pub = node.create_publisher(String, EPISODE_TOPIC_SIDE.format(side=side), latched)
        ep_id = replay_episode_id()

        def event(name: str, **kw) -> None:
            ep_pub.publish(String(data=json.dumps({"episode": ep_id, "event": name, "t_ns": time.time_ns(), **kw})))

        warn_t = [-1e9]

        def send(k: int) -> None:
            """probe 명령을 보낸다 — 비유한이면 보내지 않고 probe 가 마지막 유한 명령으로 되돌린다."""
            out_cmd = probe.outgoing()
            if out_cmd is None:
                if time.monotonic() - warn_t[0] >= 1.0:   # 초당 한 번만
                    warn_t[0] = time.monotonic()
                    print(f"비유한 명령 — 보내지 않음(마지막 유한 명령 유지, 누적 {probe.nonfinite})", file=sys.stderr, flush=True)
                return
            qa, qh = out_cmd
            m = JointState()
            m.header.stamp = node.get_clock().now().to_msg()
            m.header.frame_id = f"{ep_id}:{k}"
            m.name = arm_names + hand_names
            m.position = [float(v) for v in np.concatenate([qa, qh])]
            m.velocity = [0.0] * len(m.name)
            m.effort = [0.0] * len(m.name)
            pub.publish(m)

        signal.signal(signal.SIGINT, lambda *_: stop_sig.__setitem__("on", "SIGINT"))
        event("hand_mode", hand_command="position", reasons=["touch_probe 손 위치 명령"])
        start, start_wall_ns = time.monotonic(), time.time_ns()   # 기록 t = 0 (보드 도구 시각 맞춤용 벽시계)
        idle, prev_now = 0.0, start                                  # --step Enter 대기 · 멈춤 시간 — --seconds 시계에서 뺀다
        probe.start(q_set, q_set + e0, q_hand)
        print(f"접촉 보정 시작({side}): space 멈춤 · q 저장하고 끝 · x 버림" + (" · Enter 다음 대상" if step else ""), flush=True)
        k, rc, why, keep, stopping, stop_t, last = 0, 0, "계획 끝", True, False, 0.0, q_arm
        try:
            with KeyReader() as keys:
                t_next = time.monotonic()
                while probe.phase != "done":
                    spin_all()
                    now = time.monotonic()
                    if probe.paused or probe.waiting:
                        idle += now - prev_now
                    prev_now = now
                    meas, hmeas = fresh(arm, arm_src), fresh(hand, hand_names)
                    if meas is not None:
                        last = meas
                    fault = pd_fault(pd, now, MOVING)
                    key = keys.poll()
                    req = None
                    if stop_sig["on"]:
                        req, stop_sig["on"] = (5, f"정지 신호({stop_sig['on']})", True), None
                    elif meas is None or hmeas is None:
                        req = (4, "팔 · 손 측정이 끊겼다", True)
                    elif fault:
                        req = (3, fault, True)
                    elif key in ("q", "x"):
                        req = (0, "q" if key == "q" else "x(버림)", key == "q")
                    elif not stopping and now - start - idle > seconds:
                        req = (6, f"동작 {seconds:g} s 지남(대기 · 멈춤 {idle:.0f} s 제외)", True)
                    if req is not None:
                        # 두 번째 중단 · pd 이상 · 측정 끊김(정지 신호와 겹쳐도) = 그 자리(명령 = 실측, 끊겼으면 마지막 유효 실측)
                        if stopping or req[0] in (3, 4) or meas is None or hmeas is None:
                            probe.halt(last, req[1])
                            rc, why = (req[0] or rc), (f"{why} / 복귀 중 {req[1]}" if stopping else req[1])
                            keep = keep and req[2]
                            print(f"{now - start:7.2f} s  {req[1]} — 명령 = 실측, 그 자리에서 끝", flush=True)
                            break
                        rc, why, keep = req
                        stopping, stop_t = True, now
                        probe.stop(meas, why)
                        print(f"{now - start:7.2f} s  {why} — 명령 = 실측" + (", 지나온 길 되짚어 복귀" if probe.phase == "retreat" else ""), flush=True)
                    elif stopping and now - stop_t > RETREAT_MAX_S:
                        probe.halt(meas, "복귀 시간 초과")
                        why += f" / 복귀 {RETREAT_MAX_S:g} s 초과"
                        break
                    elif key == " " and not stopping:
                        if probe.paused:
                            probe.resume(now - start)
                        else:
                            probe.pause(meas)
                        print(f"{now - start:7.2f} s  {'멈춤 — 명령 = 실측' if probe.paused else '다시'}", flush=True)
                    elif key in ("\n", "\r") and probe.waiting:
                        probe.go()
                    if probe.phase != "done":
                        try:
                            _, _, note = probe.tick(now - start, meas, hmeas, fresh_force("jf", INDEX_JF), fresh_force("tip", INDEX_TIP))
                        except Exception as error:   # noqa: BLE001 — 어떤 오류든 누름을 풀고 되짚는 길로
                            if stopping:
                                probe.halt(meas)
                                why += f" / 복귀 중 오류 {error}"
                                break
                            rc, why, stopping, stop_t = 1, f"{type(error).__name__}: {error}", True, now
                            probe.stop(meas, why)
                            note = f"{why} — 명령 = 실측" + (", 지나온 길 되짚어 복귀" if probe.phase == "retreat" else "")
                        if probe.stop_why and not stopping:   # tick 안 추종 오차
                            rc, why, stopping, stop_t = 3, probe.stop_why, True, now
                        elif stopping and probe.retreat_fail:   # 복귀 중 추종 오차 — 그 자리에서 끝났다
                            rc, why = 3, f"{why} / 복귀 중 {probe.retreat_fail}"
                        if note:
                            print(f"{now - start:7.2f} s  {note}", flush=True)
                    send(k)
                    rows.append((now - start, meas, probe.cmd.copy(), hmeas, probe.hand.copy(),
                                 force.get("jf", (np.full(6, np.nan),))[0], force.get("tip", (np.full(5, np.nan),))[0], probe.seg, probe.phase))
                    k += 1
                    t_next = max(t_next + probe.dt, time.monotonic())
                    time.sleep(max(0.0, t_next - time.monotonic()))
        except BaseException as error:   # noqa: BLE001 — 루프 밖 오류(rclpy 등): 명령 = 마지막 실측
            rc, why = rc or 1, f"{why} / {type(error).__name__}: {error}"
            probe.halt(last)
        finally:
            stop_ok = False
            try:   # 해제 명령을 세트포인트가 따라올 만큼 보낸 뒤 stop(그 뒤 pd 는 이 에피소드 목표를 버린다)
                T, stop_ok = release_and_stop(probe, lambda: send(k),
                                              lambda: event("stop", object_anchor=None, home_q={}, reasons=[f"touch_probe 끝: {why}"]),
                                              spin_all, lambda: pd.get("target"))
                print(f"해제 명령 {T:.2f} s 보낸 뒤 stop", flush=True)
            except Exception as error:   # noqa: BLE001
                print(f"★★★ 해제 · stop 실패: {type(error).__name__}: {error} — pd 상태를 확인한다 ★★★", file=sys.stderr, flush=True)
            if not stop_ok:
                rc = max(rc, 3)
            saved: dict = {"saved": None}
            if keep and rows:
                try:
                    saved = save(out, side, plan, rows, probe.results, {"e0_home": np.round(probe.e0, 5).tolist(),
                                                                         "start_wall_ns": start_wall_ns, "ep_id": str(ep_id)})
                except Exception as error:   # noqa: BLE001
                    saved = {"saved": None, "save_error": f"{type(error).__name__}: {error}"}
        info = {"phase": "stopped", "why": why, "ticks": k, "idle_s": round(idle, 1), "pd_target": pd.get("target"), "stop_ok": stop_ok,
                "results": probe.results, **saved}
        return ((rc or 2) if keep and not rows else rc), info
    finally:
        restore_signals(prev_sig)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def restore_signals(prev_sig: dict) -> None:
    """시그널 핸들러를 되돌린다. 파이썬 밖에서 건 핸들러면 getsignal 이 None — 기본 동작으로. 실패해도 뒤 정리(destroy_node · rc 출력)는 간다."""
    for sg, prev in prev_sig.items():
        try:
            signal.signal(sg, signal.SIG_DFL if prev is None else prev)
        except (ValueError, TypeError, OSError) as error:
            print(f"시그널 {sg.name} 복원 실패: {error}", file=sys.stderr)


def check_out(out: Path) -> None:
    """기록 파일과 .part(지난 실행이 제 이름으로 옮기지 못한 기록)가 하나라도 있으면 시작 전 거부."""
    for f in out_paths(Path(out)):
        for g in (f, f.with_name(f.name + ".part")):
            if g.exists():
                raise ProbeError(f"{g} 가 이미 있다 — 덮어쓰지 않는다(.part 면 지난 기록을 확인해 옮기거나 지운 뒤)")


def out_paths(out: Path) -> tuple[Path, Path]:
    out = Path(out)
    return out.with_name(out.stem + "_raw.npz"), out.with_name(out.stem + "_touch.json")


def _columns(rows: list) -> dict[str, np.ndarray]:
    """행 → 열 배열. 한 열이 쌓이지 않으면(길이 다른 힘 등) 그 열만 행마다 따로 둔다(pickle 없이)."""
    out: dict[str, np.ndarray] = {}
    for j, n in enumerate(("t", "q_meas", "q_cmd", "hand_meas", "hand_cmd", "jf_zeroed", "tip_zeroed", "seg", "phase")):
        col = [r[j] for r in rows]
        try:
            if n == "phase":
                out[n] = np.array(col, dtype=str)
            elif n in ("t", "seg"):
                out[n] = np.array(col, float if n == "t" else int)
            else:
                out[n] = np.vstack([np.full(7 if "q_" in n else 6, np.nan) if v is None else np.asarray(v, float) for v in col])
        except (ValueError, TypeError):
            out |= {f"{n}_{i:06d}": np.asarray(np.nan if v is None else v, float) for i, v in enumerate(col)}
    return out


def save(out: Path, side: str, plan: dict, rows: list, results: list, extra: dict | None = None) -> dict:
    """<out>_raw.npz(모든 틱 — measure.load_npz 와 같은 이름 + 힘 · 구간) · <out>_touch.json(대상별 결과 · e0_local).
    둘 다 <이름>.part 에 배타적으로 쓰고, 둘 다 성공한 뒤에만 제 이름으로 옮긴다(반쪽 기록이 남지 않게). 기존 파일은 덮지 않는다."""
    raw, js = out_paths(out)
    raw.parent.mkdir(parents=True, exist_ok=True)
    parts, made = [f.with_name(f.name + ".part") for f in (raw, js)], []
    try:
        with open(parts[1], "x") as f:
            made.append(parts[1])
            json.dump({"side": side, "plan_meta": plan["_meta"], **(extra or {}), "results": results}, f, ensure_ascii=False, indent=1)
        with open(parts[0], "xb") as f:
            made.append(parts[0])
            np.savez(f, side=np.array(side), arm_names=np.array(P.arm_names(side), dtype=str), hand_names=np.array(P.hand_names(side), dtype=str),
                     results=np.array(json.dumps(results, ensure_ascii=False)), plan_meta=np.array(json.dumps(plan["_meta"])),
                     meta=np.array(json.dumps(extra or {})), **_columns(rows))
        if raw.exists() or js.exists():
            raise FileExistsError(f"{raw} 또는 {js} 가 이미 있다 — 덮어쓰지 않는다(기록은 {parts[0].name} · {parts[1].name} 에 남김)")
        for f, dst in zip(parts, (raw, js)):
            f.rename(dst)
    except FileExistsError:
        raise
    except BaseException:
        for f in made:
            f.unlink(missing_ok=True)
        raise
    return {"raw": str(raw), "touch": str(js), "frames": len(rows)}


# ------------------------------------------------------------------ 자체 검사(ROS 없음)
def _fake_plan() -> dict:
    """1 관절 가짜 팔(손끝 z = 0.3 − q0): 홈 → 접근(q0 0.02 = z 280) · 하강(→ q0 0.08 = z 220) · 홈."""
    q = np.zeros((1, 7))
    plan = dict(q=np.vstack([q, q + [0.02, 0, 0, 0, 0, 0, 0], q + [0.02, 0, 0, 0, 0, 0, 0], q + [0.08, 0, 0, 0, 0, 0, 0],
                             q + [0.02, 0, 0, 0, 0, 0, 0], q]),
                t=np.array([0, 1, 1.1, 13.1, 13.2, 14.2]), seg=np.array([0, 0, 1, 1, 2, 2]), tip=np.zeros((6, 3)),
                seg_kind=np.array(["move", "descend", "move"]), seg_target=np.array([0, 0, 0]), seg_gate=np.array([True, False, False]),
                tgt_name=np.array(["t0"]), tgt_kind=np.array(["table"]), tgt_hole=np.array([-1]), tgt_psi=np.array([0.0]),
                tgt_offset_mm=np.zeros((1, 2)), tgt_xyz=np.array([[0, 0, 0.25]]), hand=np.zeros(6))
    plan["tip"][:, 2] = 0.3 - plan["q"][:, 0]
    return plan


def _fake_plan_hover() -> dict:
    """검색 격자 한 칸: 접근(q0 .02 = z 280) → 하강 g0 → hover 이동(z 280 → 265, j2 0 → .05) → 하강 g1 → hover 이동(→ 접근점) → 홈."""
    z = np.zeros(7)
    a, h = z + np.r_[0.02, 0, 0, 0, 0, 0, 0], z + np.r_[0.035, 0.05, 0, 0, 0, 0, 0]
    d0, d1 = z + np.r_[0.08, 0, 0, 0, 0, 0, 0], z + np.r_[0.08, 0.05, 0, 0, 0, 0, 0]
    plan = dict(q=np.vstack([z, a, a, d0, a, h, h, d1, h, a, a, z]),
                t=np.array([0, 1, 1.1, 13.1, 13.2, 16.2, 16.3, 25.3, 25.4, 28.4, 28.5, 29.5]),
                seg=np.repeat(np.arange(6), 2), tip=np.zeros((12, 3)),
                seg_kind=np.array(["move", "descend", "move", "descend", "move", "move"]), seg_target=np.array([0, 0, 1, 1, 1, 1]),
                seg_gate=np.array([True, False, False, False, False, False]), tgt_name=np.array(["g0", "g1"]),
                tgt_kind=np.array(["hole", "hole"]), tgt_hole=np.array([12, 12]), tgt_psi=np.zeros(2),
                tgt_offset_mm=np.array([[0, 0], [2, 0]], float), tgt_xyz=np.array([[0, 0, 0.25]] * 2), hand=np.zeros(6))
    plan["tip"][:, 2] = 0.3 - plan["q"][:, 0]
    return plan


E0 = np.array([0.0, 0, 0, 0.01, 0, 0, 0])   # 가짜 처짐(손끝 z 에 안 들어가는 관절)


def _sim(pr: Probe, n: int = 4000, force=None, track=None, on_tick=None) -> list:
    """가짜 탁자(z 0.25, 명령이 탁자 아래로 1 mm 당 0.1 N): 실측 = 명령 + e0, 다만 손끝이 탁자 아래로 못 간다. → [(phase, cmd)…]"""
    dt, log = 1 / 60, []
    for k in range(n):
        t = k * dt
        if pr.waiting and k % 50 == 49:
            pr.go()
        meas = pr.cmd + E0
        meas[0] = min(meas[0], 0.05)
        if track is not None:
            meas = meas + track(pr, k)
        touch = max(0.0, pr.cmd[0] - 0.05) * 100   # 명령이 탁자 아래로 간 1 mm 당 0.1 N
        jf, tip = (np.zeros(6), np.r_[0, touch, 0, 0, 0]) if force is None else force(pr, k, touch)
        if on_tick is not None:
            on_tick(pr, k, meas)
        if pr.phase == "done":
            break
        pr.tick(t, meas, np.zeros(6), jf, tip)
        pr.outgoing()   # run 의 send 와 같이 — 마지막 송신 명령(probe.sent)
        log.append((pr.phase, pr.cmd.copy()))
    return log


def _selftest() -> None:
    dt = 1 / 60
    # 1) 판정: 잡음(±10 g) · 한 틱 튐 무시, 30 g 계단은 0.04 s 뒤, 촉각, 끊김 0.25 s, 지연.
    d = Detector()
    d.begin(0.0, 5.0, 0.0, 1.0)
    rng = np.random.default_rng(0)
    for k in range(120):
        jf = 5 + rng.uniform(-10, 10) + (40 if k == 30 else 0)
        assert d.update(k * dt, jf, 0.0, 1.0 + rng.uniform(-0.3, 0.3)) is None, k
    hit = [d.update(2 + k * dt, 35.0, 0.0, 1.0) for k in range(5)]
    assert hit[:2] == [None, None] and "index_force" in hit, hit
    d.begin(0.0, 0.0, 0.0, 0.0)
    assert [d.update(k * dt, 0.0, 0.3, 0.0) for k in range(4)][-1] == "tip_touch"
    d.begin(0.0, 0.0, 0.0, 0.0)
    assert d.update(0.1, None, None, 0.0) is None and d.update(0.3, None, None, 0.0) == "force_stale"
    d.begin(0.0, 0.0, 0.0, 0.0)
    assert all(d.update(k * dt, 0.0, 0.0, 2.0) is None for k in range(120))
    assert [d.update(2 + k * dt, 0.0, 0.0, 5.5) for k in range(8)][-1] == "arm_lag"
    assert clean_force([0, 1, np.nan], 1) is None and clean_force([0, 1], 2) is None and clean_force([0, 1, 2], 2) is not None
    fk = lambda qq, h=None: np.array([0, 0, 0.3 - qq[0]])   # noqa: E731
    new = lambda: Probe(_fake_plan(), fk, np.full(7, -3.0), np.full(7, 3.0), step=True)   # noqa: E731

    # 2) 정상: tip_touch 로 멈춤 → 명령 = 실측(누름 0) → 되짚어 홈.
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    log = _sim(pr)
    assert pr.phase == "done" and len(pr.results) == 1, (pr.phase, pr.results)
    r = pr.results[0]
    assert r["contact"] and r["signal"] in ("tip_touch", "arm_lag") and abs(r["fk_tip_mm"][2] - 250) < 2, r
    held = [c[0] for p, c in log if p == "hold"]
    assert held and max(held) <= 0.05 + 1e-6, max(held)   # hold 중 명령이 탁자 아래로 안 간다
    # 가짜 처짐(j4)은 이 fk 의 손끝을 안 움직인다 → 보상 없음(손끝 판정)
    assert r["comp_n"] == 0 and r["comp_tip_mm"] == 0.0, r
    assert np.abs(pr.cmd).max() < 1e-6, pr.cmd
    # 2b) 처짐 보상: j4 처짐 0.01 rad 가 손끝 x 10 mm 로 번지는 fk → 접근점에서 이득 0.6 으로 두 번(10 → 4 → 1.6 mm) 맞추고 내려간다
    fk2 = lambda qq, h=None: np.array([qq[3], 0, 0.3 - qq[0]])   # noqa: E731
    pr = Probe(_fake_plan(), fk2, np.full(7, -3.0), np.full(7, 3.0), step=True)
    pr.start(np.zeros(7), E0, np.zeros(6))
    _sim(pr)
    r = pr.results[0]
    assert r["contact"] and r["comp_n"] == 2 and r["comp_tip_mm"] <= 3.0 and abs(r["fk_tip_mm"][0]) < 3.0, r
    assert np.abs(pr.cmd).max() < 1e-6 and abs(pr.off[3] + 0.0084) < 1e-4, (pr.cmd, pr.off)   # 홈 복귀는 off 없이(M2)
    assert pr.home_segs == {2} and not r["clipped"], (pr.home_segs, r)
    # 2c) comp_limit: 접근점 + off 는 한계 안이지만 하강 끝(j4 −0.495) + off(−0.006) 가 하한 −0.497 밖 → 내리지 않고 거부(H3)
    pl = _fake_plan()
    pl["q"][3, 3] = -0.495
    lower = np.full(7, -3.0)
    lower[3] = -0.497 - P.LIMIT_MARGIN_RAD
    pr = Probe(pl, fk2, lower, np.full(7, 3.0), step=True)
    pr.start(np.zeros(7), E0, np.zeros(6))
    log = _sim(pr)
    assert pr.results[0]["signal"] == "comp_limit" and "j4" in pr.results[0]["refused"], pr.results
    assert "descend" not in [ph for ph, _ in log] and pr.phase == "done" and np.abs(pr.cmd).max() < 1e-6
    # 2d) 이월된 off 가 한계에 잘리면 그 기록에 clipped(j2 계획 0 + off 0.04 > 상한 0.03)
    upper = np.full(7, 3.0)
    upper[1] = 0.03 + P.LIMIT_MARGIN_RAD
    pr = Probe(_fake_plan(), fk, np.full(7, -3.0), upper, step=True)
    pr.start(np.zeros(7), E0, np.zeros(6))
    pr.off = np.r_[0, 0.04, 0, 0, 0, 0, 0]
    _sim(pr)
    assert pr.results[0]["contact"] and pr.results[0]["clipped"], pr.results

    # 3) 하강 중 x: 명령 = 실측 − e0 로 풀고 → 하강 길을 되짚어 접근점(q0 0.02)에서 끝.
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    seen = {}

    def press_x(p: Probe, k: int, meas) -> None:
        if p.phase == "descend" and p.tau > 3.0 and "x" not in seen:
            seen["x"] = meas.copy()
            p.stop(meas, "x")
            seen["cmd"] = p.cmd.copy()
    _sim(pr, on_tick=press_x)
    assert np.allclose(seen["cmd"], seen["x"] - E0), (seen["cmd"], seen["x"])
    assert pr.phase == "done" and abs(pr.cmd[0] - 0.02) < 1e-3 and pr.stop_why == "x", (pr.cmd, pr.stop_why)

    # 4) 하강 중 NaN 힘 → force_stale(0.25 s) → 복귀 · 홈.
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    nan = lambda p, k, t: ((np.full(6, np.nan), np.full(5, np.nan)) if p.phase == "descend" and p.tau > 2 else (np.zeros(6), np.zeros(5)))  # noqa: E731
    _sim(pr, force=nan)
    assert pr.phase == "done" and pr.results[0]["signal"] == "force_stale" and not pr.results[0]["contact"], pr.results
    assert np.abs(pr.cmd - pr.off).max() < 1e-6

    # 5) 기준선이 이미 닿음(촉각 0.5 N) → 하강 거부, 내려가지 않고 홈.
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    log = _sim(pr, force=lambda p, k, t: (np.zeros(6), np.r_[0, 0.5, 0, 0, 0]))
    assert pr.results[0]["signal"] == "pre_touch" and "refused" in pr.results[0], pr.results
    assert "descend" not in [p for p, _ in log] and max(c[0] for _, c in log) <= 0.02 + 1e-6
    # 검지 관절 힘 +20 g(홈 영점 대비)도 거부
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    _sim(pr, force=lambda p, k, t: (np.r_[0, 0, 20.0, 0, 0, 0], np.zeros(5)))
    assert pr.results[0]["signal"] == "pre_touch", pr.results

    # 6) 이동 중 추종 오차 0.09 rad(> 홈 e0 기준 0.08) → 중단(이동 구간이라 그 자리, 명령 = 실측 − e0). 0.07 은 통과.
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    _sim(pr, track=lambda p, k: np.r_[0, 0.07, 0, 0, 0, 0, 0] if p.phase == "move" and p.tau > 0.5 else np.zeros(7))
    assert pr.stop_why is None and pr.results and pr.results[0]["contact"], (pr.stop_why, pr.results)
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    bump = lambda p, k: np.r_[0, 0.09, 0, 0, 0, 0, 0] if p.phase == "move" and p.tau > 0.5 else np.zeros(7)  # noqa: E731
    _sim(pr, track=bump)
    assert pr.phase == "done" and pr.stop_why.startswith("추종 오차") and not pr.results, (pr.phase, pr.stop_why)
    assert abs(pr.cmd[1] - 0.09) < 1e-6, pr.cmd
    # 하강 중 0.05 rad(> e0_local 기준 0.04) → 중단 · 하강 되짚어 접근점
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    _sim(pr, track=lambda p, k: np.r_[0, 0.05, 0, 0, 0, 0, 0] if p.phase == "descend" and p.tau > 1 else np.zeros(7))
    assert pr.phase == "done" and "j2" in pr.stop_why and abs(pr.cmd[0] - 0.02) < 1e-3, (pr.stop_why, pr.cmd)
    # 올라가는 길 0.05 rad(누름이 풀리며 처짐 변화 — 11.05 실기 j4 0.040 오탐)는 이동 문턱(0.08)이라 통과
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    _sim(pr, track=lambda p, k: np.r_[0, 0, 0, 0.05, 0, 0, 0] if p.phase == "ascend" else np.zeros(7))
    assert pr.stop_why is None and pr.phase == "done" and pr.results[0]["contact"], (pr.stop_why, pr.results)

    # 7) 하강 중 멈춤 → 명령 = 실측, 다시 → 기준선 다시 잡고 이어서 하강 · 접촉.
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))

    def pause_resume(p: Probe, k: int, meas) -> None:
        if p.phase == "descend" and p.tau > 2.0 and "p" not in seen:
            seen["p"] = k
            p.pause(meas)
        elif seen.get("p") and k == seen["p"] + 60 and p.paused:
            p.resume(k / 60)
    seen.clear()
    _sim(pr, on_tick=pause_resume)
    assert pr.phase == "done" and pr.results and pr.results[0]["contact"], pr.results
    r0 = r

    # (a) 측정 끊김 중 SIGTERM: run 은 halt(마지막 유효 실측) — 명령 유한, 송신 NaN 0. stop(None) · _hold_cmd(NaN) · outgoing(NaN) 도.
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    _sim(pr, on_tick=lambda p, k, m: p.halt(None, "정지 신호(SIGTERM)") if p.phase == "descend" and p.tau > 3 else None)
    assert pr.phase == "done" and np.all(np.isfinite(pr.cmd)) and np.allclose(pr.cmd, pr.last_q - pr.e0_loc), pr.cmd
    tk, sent = [0.0], []
    release_and_stop(pr, lambda: sent.append(pr.outgoing()), lambda: None, lambda: None, lambda: "internal",
                     clock=lambda: tk[0], sleep=lambda d: tk.__setitem__(0, tk[0] + d))
    assert sent and all(o is not None and np.all(np.isfinite(o[0])) for o in sent) and pr.nonfinite == 0
    good = pr.cmd.copy()
    assert np.allclose(pr._hold_cmd(np.full(7, np.nan)), good)
    pr.cmd = np.full(7, np.nan)
    assert pr.outgoing() is None and np.allclose(pr.cmd, good) and pr.nonfinite == 1
    pr2 = new()
    pr2.start(np.zeros(7), E0, np.zeros(6))
    seen = {}

    def lost_stop(p: Probe, k: int, m) -> None:
        if p.phase == "descend" and p.tau > 3 and "s" not in seen:
            seen["s"] = p.last_q.copy()
            p.stop(None, "SIGTERM")
            seen["cmd"] = p.cmd.copy()
    _sim(pr2, on_tick=lost_stop)
    assert np.allclose(seen["cmd"], seen["s"] - pr2.e0_loc) and pr2.phase == "done" and abs(pr2.cmd[0] - 0.02) < 1e-3

    # (b) 하강 중(누르는 중) halt → stop 이벤트 전에 해제 명령을 max(0.5, Δ/0.25 + 0.2) s 동안 보낸다.
    assert abs(release_s(np.r_[0.1, 0, 0, 0, 0, 0, 0], np.zeros(7)) - 0.6) < 1e-9 and release_s(np.zeros(7), None) == RELEASE_MIN_S
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    seen = {}

    def press_halt(p: Probe, k: int, m) -> None:
        if p.phase == "descend" and p.cmd[0] > 0.052 and "h" not in seen:
            seen["h"], seen["sent"] = True, p.sp_est.copy()
            assert np.allclose(p.sp_est, p.sent[0])   # 명령 0.2 rad/s < pd 0.25 — 추정 세트포인트 = 마지막 송신
            p.halt(m, "pd 이상")
    _sim(pr, force=lambda p, k, t: (np.zeros(6), np.zeros(5)), on_tick=press_halt)
    hold = pr.cmd.copy()
    assert hold[0] <= 0.05 + 1e-9 < seen["sent"][0], (hold, seen["sent"])
    tk, sends, ev = [0.0], [], []
    T, ok = release_and_stop(pr, lambda: sends.append((tk[0], pr.outgoing()[0].copy())), lambda: ev.append(tk[0]), lambda: None,
                             lambda: "internal" if ev else "tracking", clock=lambda: tk[0], sleep=lambda d: tk.__setitem__(0, tk[0] + d))
    assert ok
    before = [c for t_, c in sends if t_ < ev[0]]
    assert abs(T - max(RELEASE_MIN_S, np.abs(hold - seen["sent"]).max() / PD_VEL_RAD_S + RELEASE_PAD_S)) < 1e-9
    assert len(before) >= int(T / dt) and all(np.allclose(c, hold) for c in before) and ev[0] >= T - 1e-9, (len(before), T)

    # (c) 접근점 자세 처짐이 홈과 j2 0.03 rad 다르고 하강 중 0.015 더 뒤처져도(홈 e0 기준이면 0.045 > 0.04) 오탐 없음.
    assert TRACK_TOL_DESC[1] < 0.045
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))

    def lean(p: Probe, k: int) -> np.ndarray:
        if p.phase in ("hand", "move", "done") or p._kind(p.seg) != "descend":
            return np.zeros(7)
        return np.r_[0, 0.03 + (0.015 if p.phase == "descend" else 0), 0, 0, 0, 0, 0]
    _sim(pr, track=lean)
    rc = pr.results[0]
    assert pr.stop_why is None and rc["contact"] and abs(rc["e0_local"][1] - 0.03) < 1e-6 and abs(rc["e0_local"][3] - 0.01) < 1e-6, rc
    assert abs(rc["fk_tip_mm"][2] - 250) < 2 and np.abs(pr.cmd - pr.off).max() < 1e-6

    # (d) 시작 e0: 멈춤 · |e0| ≤ 0.06 이어야. 접근점 e0 가 3 s 안에 안 멈추면 그 대상 거부(내리지 않음).
    ts = np.arange(31) / 60
    qs0 = np.tile(E0, (31, 1))
    assert np.allclose(start_e0(ts, qs0, np.zeros(7))[0], E0)
    assert start_e0(ts, qs0 + ts[:, None] * 0.05, np.zeros(7))[0] is None                       # 움직임 0.05 rad/s
    assert start_e0(ts, qs0 + np.random.default_rng(1).normal(0, 0.01, (31, 7)), np.zeros(7))[0] is None   # 떨림
    assert start_e0(ts, qs0 + np.r_[0, 0.07, 0, 0, 0, 0, 0], np.zeros(7))[0] is None              # 처짐 0.07 > 0.06
    assert start_e0(ts[:10], qs0[:10], np.zeros(7))[0] is None                                    # 창 짧음
    pr = new()
    pr.start(np.zeros(7), E0, np.zeros(6))
    rng = np.random.default_rng(2)
    log = _sim(pr, track=lambda p, k: np.r_[0, rng.normal(0, 0.008), 0, 0, 0, 0, 0] if p.phase == "dwell" else np.zeros(7))
    assert pr.results[0]["signal"] == "e0_unsettled" and "refused" in pr.results[0] and pr.stop_why is None, pr.results
    assert "descend" not in [ph for ph, _ in log] and pr.phase == "done"

    # (e) hover 이동(검색 격자) — 접촉 없으면 끝까지 · 접촉이면 명령 = 실측 − e0 후 지나온 길 되짚어 접근점 · 하강 중 중단도 hover 길로.
    newh = lambda: Probe(_fake_plan_hover(), fk, np.full(7, -3.0), np.full(7, 3.0), step=False)   # noqa: E731
    pr = newh()
    assert pr.hover == {2, 4}, pr.hover
    pr.start(np.zeros(7), E0, np.zeros(6))
    _sim(pr)
    assert pr.phase == "done" and pr.stop_why is None and [x["contact"] for x in pr.results] == [True, True], pr.results
    assert not any(x.get("hover") for x in pr.results) and np.abs(pr.cmd - pr.off).max() < 1e-6
    pr = newh()
    pr.start(np.zeros(7), E0, np.zeros(6))
    seen = {}

    def bump_tip(p: Probe, k: int, t: float):
        hit = p.phase == "move" and p.seg == 2 and p.tau > 1.0
        if hit and "q" not in seen:
            seen["q"], seen["e0"] = p.last_q.copy(), p.e0_loc.copy()
        return np.zeros(6), np.r_[0, t + (0.5 if hit else 0.0), 0, 0, 0]
    log = _sim(pr, force=bump_tip)
    assert pr.stop_why.startswith("hover") and pr.results[-1].get("hover") and pr.results[-1]["signal"] == "tip_touch", pr.results
    assert pr.phase == "done" and np.allclose(pr.cmd - pr.off, [0.02, 0, 0, 0, 0, 0, 0], atol=1e-3), pr.cmd
    assert not any(ph == "descend" and c[1] > 0.01 for ph, c in log)                              # g1 은 내리지 않았다
    pr = newh()
    pr.start(np.zeros(7), E0, np.zeros(6))
    _sim(pr, on_tick=lambda p, k, m: p.stop(m, "x") if p.seg == 3 and p.phase == "descend" and p.tau > 1 and not p.stop_why else None)
    assert pr.phase == "done" and np.allclose(pr.cmd - pr.off, [0.02, 0, 0, 0, 0, 0, 0], atol=1e-3), pr.cmd   # 하강 g1 → hover 길 → 접근점

    # H1) e0_outlier 는 대상 첫 정지 측정에서 |e0 − 홈 e0| 만 본다(off 항 없음).
    on_dwell = ("dwell", "descend", "hold", "ascend")
    # a) 이월 off 는 무시: off j1 −0.07(+ 쪽은 가짜 탁자에 닿는다), 기댐 0 → 통과 · 보상 · 하강. 옛 |Δe0 + off| = 0.07 은 거부했다.
    # b) 기댐 검출: off j2 −0.07 에서 정지 중 실측 +0.07 → |Δe0| 0.07 거부, 하강 없음. 옛 식은 0 이라 통과시켰다.
    for off0, lean_q, ok in ((np.r_[-0.07, 0, 0, 0, 0, 0, 0], np.zeros(7), True),
                             (np.r_[0, -0.07, 0, 0, 0, 0, 0], np.r_[0, 0.07, 0, 0, 0, 0, 0], False)):
        pr = new()
        pr.start(np.zeros(7), E0, np.zeros(6))
        pr.off = off0
        log = _sim(pr, track=lambda p, k, v=lean_q: v if p.phase in on_dwell else np.zeros(7))
        got = pr.results[0]
        if ok:
            assert got["contact"] and pr.stop_why is None and "descend" in [ph for ph, _ in log] and got["comp_n"] > 0, got
        else:
            assert got["signal"] == "e0_outlier" and "descend" not in [ph for ph, _ in log], got
    # c) 보상 중 정지마찰은 통과: 첫 측정 j2 −0.02(통과) → 보상 4 회 동안 관절이 그 자리(track = −0.02 − (off − off_m0))라
    #    Δe0 j2 가 −0.068 까지 밀린 뒤 풀려 따라간다. 보상 중에도 검사하면 0.068 > 0.06 으로 거부됐다.
    fk3 = lambda qq, h=None: np.array([0.2 * qq[1], 0, 0.3 - qq[0]])   # noqa: E731
    pr = Probe(_fake_plan(), fk3, np.full(7, -3.0), np.full(7, 3.0), step=True)
    pr.start(np.zeros(7), E0, np.zeros(6))
    stuck = np.r_[0, -0.02, 0, 0, 0, 0, 0]
    log = _sim(pr, track=lambda p, k: (stuck - p.off * (np.arange(7) == 1) if p.comp_n <= 4 else stuck) if p.phase in on_dwell else np.zeros(7))
    got = pr.results[0]
    assert got["contact"] and pr.stop_why is None and "descend" in [ph for ph, _ in log] and got["comp_n"] > 5, got
    assert abs(got["e0_local"][1] + 0.02) < 1e-6, got

    # r3-1) 접근점 e0 가 홈과 j2 0.07 rad(> 0.06) 다르면 그 대상 거부(e0_outlier) · 0.05 는 통과 · j6 0.075 는 통과(손목 상한 0.12,
    #       이동 추종 문턱 0.08 아래) · j6 0.13 은 거부 — 손목은 이동 추종 문턱(0.12)이 e0_outlier 상한과 같아 정지 중 추종 오차로 먼저 멈춘다.
    for lean_q, ok in ((np.r_[0, 0.07, 0, 0, 0, 0, 0], False), (np.r_[0, 0.05, 0, 0, 0, 0, 0], True), (np.r_[0, 0, 0, 0, 0, 0.075, 0], True),
                       (np.r_[0, 0, 0, 0, 0, 0.13, 0], False)):
        pr = new()
        pr.start(np.zeros(7), E0, np.zeros(6))
        log = _sim(pr, track=lambda p, k, v=lean_q: v if p.phase in ("dwell", "descend", "hold", "ascend") else np.zeros(7))
        if lean_q[5] > 0.12:
            assert not pr.results and pr.stop_why.startswith("추종 오차 j6") and "descend" not in [ph for ph, _ in log], pr.stop_why
            continue
        got = pr.results[0]
        if ok:
            assert got["contact"] and pr.stop_why is None, (lean_q, got)
        else:
            assert got["signal"] == "e0_outlier" and "refused" in got and "descend" not in [ph for ph, _ in log], got

    # r3-6) 해제 송신 · spin 이 계속 실패해도 stop 은 한 번 나간다.
    tk, ev = [0.0], []

    def boom() -> None:
        raise RuntimeError("publish 실패")
    _, ok = release_and_stop(pr, boom, lambda: ev.append(tk[0]), boom, lambda: "internal",
                             clock=lambda: tk[0], sleep=lambda d: tk.__setitem__(0, tk[0] + d))
    assert len(ev) == 1 and ev[0] >= RELEASE_MIN_S - 1e-9 and ok, ev
    # H2) stop 이벤트 실패 · STOP_WAIT_S 안에 internal 이 안 되면 stop_ok = False
    tk = [0.0]
    assert not release_and_stop(pr, lambda: None, boom, lambda: None, lambda: "internal",
                                clock=lambda: tk[0], sleep=lambda d: tk.__setitem__(0, tk[0] + d))[1]
    assert not release_and_stop(pr, lambda: None, lambda: None, lambda: None, lambda: "tracking",
                                clock=lambda: tk[0], sleep=lambda d: tk.__setitem__(0, tk[0] + d))[1]

    # r3-3) 시그널 복원: None(파이썬 밖 핸들러) → SIG_DFL, 잘못된 값이어도 예외 없이 지나간다.
    old = signal.getsignal(signal.SIGHUP)
    restore_signals({signal.SIGHUP: None, signal.SIGUSR1: "잘못된 값"})
    assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
    signal.signal(signal.SIGHUP, old)

    # r3-5 · 8) .part 가 있으면 시작 전 거부 · 기록에 start_wall_ns · ep_id(json 최상위 · npz meta).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "r.npz"
        check_out(out)
        raw, js = out_paths(out)
        raw.with_name(raw.name + ".part").write_bytes(b"")
        try:
            check_out(out)
            raise AssertionError(".part 를 못 봤다")
        except ProbeError:
            pass
        raw.with_name(raw.name + ".part").unlink()
        rows = [(0.0, np.zeros(7), np.zeros(7), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(5), 0, "move")]
        extra = {"e0_home": [0.0] * 7, "start_wall_ns": 1793800000123456789, "ep_id": "ep7"}
        save(out, "right", {"_meta": {"side": "right"}}, rows, [], extra)
        assert json.loads(js.read_text())["start_wall_ns"] == extra["start_wall_ns"] and json.loads(js.read_text())["ep_id"] == "ep7"
        assert json.loads(str(np.load(raw)["meta"])) == extra and sorted(p.name for p in Path(td).iterdir()) == [raw.name, js.name]
        try:
            check_out(out)
            raise AssertionError("기존 기록을 못 봤다")
        except ProbeError:
            pass

    print("selftest ok", {k: r0[k] for k in ("signal", "cmd_tip_z_mm", "fk_tip_mm")},
          "· x 되짚기 · NaN stale · 이미 닿음 거부 · 추종 오차 정지 · 멈춤/다시 · (a) 끊김+SIGTERM NaN 0 · (b) 해제 뒤 stop",
          f"{T:.2f} s · (c) e0_local 오탐 없음 · (d) e0 미정착 거부 · (e) hover 접촉 되짚기",
          "· r3: e0_outlier 거부 · 송신 실패에도 stop · 시그널 복원 · .part 거부 · start_wall_ns/ep_id 기록",
          "· r4: e0_outlier 첫 측정만(이월 off 무시 · 보상 중 정지마찰 통과) · comp_limit 남은 구간 · clipped · 홈 복귀 off 없음 · stop_ok")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0], allow_abbrev=False)   # 래퍼(touch_probe_rec.sh)가 같은 인자를 읽는다
    ap.add_argument("--side", choices=("right", "left"))
    ap.add_argument("--plan", help="plan_touch.py 의 plan_<side>.npz")
    ap.add_argument("--out", help="기록 이름(.npz) → <out>_raw.npz · <out>_touch.json")
    ap.add_argument("--step", action="store_true", help="대상마다 Enter 로 시작")
    ap.add_argument("--seconds", type=float, default=None,
                    help="최대 동작 시간(손목 발열, --step 대기 · 멈춤 제외). 기본 = 계획 최악 × 1.3 + 120 s. 계획 최악보다 짧게 주면 시작 거부")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return 0
    try:
        if not (a.plan and a.out):
            raise ProbeError("--plan · --out 이 필요하다")
        plan = load_plan(Path(a.plan), a.side or json.loads(str(np.load(a.plan)["meta"]))["side"])
        side = plan["_meta"]["side"]
        if Path(a.out).suffix != ".npz":
            raise ProbeError(f"--out 은 .npz: {a.out}")
        check_out(Path(a.out))
        need = plan_seconds(plan)
        if a.seconds is not None and need > a.seconds:
            raise ProbeError(f"계획 최악 {need:.0f} s > --seconds {a.seconds:g} — --seconds 를 늘리거나 계획을 나눈다")
        seconds = a.seconds if a.seconds is not None else round(need * 1.3 + 120)
    except (ProbeError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"plan": a.plan, "side": side, "fk_dz_mm": plan["_meta"].get("fk_dz_mm"),
                      "targets": [str(n) for n in plan["tgt_name"]], "descents": len(plan["desc_start"]),
                      "plan_s": round(need, 1), "seconds": seconds, "hand": plan["hand"].tolist(), "step": a.step}, ensure_ascii=False), flush=True)
    rc, info = run(side, plan, Path(a.out), seconds, a.execute, a.step)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

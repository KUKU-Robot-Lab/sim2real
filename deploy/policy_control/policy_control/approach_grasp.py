"""경로 C — 접근 정책으로 다가간 뒤 정책을 끄고 스크립트로 손을 닫아 쥐고 든다(10.17 사용자 승인, docs/PLAN_GRASP_TILT_EVAL_2026-10-17.md).

rh_aglt 노드(AgltChain) 안에서 단계만 바꾼다. 스크립트 단계도 정책과 같은 13 칸 행동을 만들어 같은 디코더(RaDecoder)를 거친다 —
팔 관절 범위 · 손 EMA · 손 속도 상한 · 막힘 쥠 멈춤(hand_block)이 정책 때와 같다.

  approach  접근 정책(10.18 rg3wg ep300 — 편 손으로 컵 옆까지 깊이 들어감. 10.17 은 rg3wc ep800)을 그대로 돌린다.
            끝 = |컵 − 손바닥| ≤ stop_palm_m AND 팔 관절 속도 최대 < still_rad_s 가 still_s 이어짐, 또는 approach_max_s.
  close     팔 목표 고정(팔 행동 0). 엄지 회전(thumb_1)은 접근 끝 목표 그대로. 엄지 굽힘(thumb_delay_s 뒤)과 네 손가락은
            close_rad_s 로 쥔 자세(q_grip) 쪽으로 닫는다. 막힘이 래치된 축은 그 자리에서 더 닫지 않는다(사용자 규칙: 닿을 때까지
            닫기 · 모양에 맞춘 파지 금지 · 엄지 끝 접촉은 요구하지 않음).
  ★10.18 사용자 "접근만 하게 하고 스크립트처럼 파지를 하게 우리가 강제하는 거니까 쥠 판정 따위 필요 없음, 무조건 잡을 수밖에 없을 것"
  → 기본 judge=0(강제 파지): 닫는 축이 모두 막히거나(래치) 쥔 자세에 닿으면 squeeze_s 더 기다린 뒤(또는 close_max_s) 판정 없이 든다.
     들기 · 유지 중에도 풀림 판정으로 멈추지 않는다(들기 높이 미달 = 팔이 못 따라온 것만 실패).
  ★10.18 사용자 "스크립트 기준 접촉 판단이 되면 멈추면 됨 · 실기는 핸드에 과도한 토크(과전류)가 걸리지 않게만 조정"
  → 강제 파지에서 막힘(래치 = 실기 접촉 판단)이 걸린 축은 그 순간 실측 + pin_lead_rad(기본 0)에 고정한다 — 그 뒤 더 밀지 않는다
     (리드 스크루가 전류 없이 쥔 자리를 붙든다, 래치가 풀려도 고정 유지). 아직 안 닿은 축은 지령이 실측을 close_lead_rad 넘게
     앞서지 못한다 — 막힘을 놓쳐도 미는 간극(= 힘 · 전류)이 pd HAND_MAX_LEAD 0.2 rad 대신 이 값에 묶인다.
     펌웨어 보호(전류 한계 800 mA · FORCE_SET)는 그대로다.
     squeeze=1 이면 고정 · 앞섬 상한 없이 쥔 자세까지 계속 민다(pd 손 앞섬 상한 0.2 rad 가 미는 힘을 묶는다).
  judge=1   (옛 동작) 쥔 판정 = HandBlockOpp(네 손가락 중 judge_min_axes 축 이상 래치) AND |컵 − 손바닥| < near_m 가 judge_ticks
            이어짐. close_max_s 안에 못 쥐면 실패 — 들지 않고 멈춘다(abort, pd 가 그 자세로 붙든다. 복구는 운영자 rehome).
  lift      손바닥 방향 고정, 수직 lift_m 를 lift_speed 로(DLS IK, rh_cap.ik_lift_action 과 같은 식 → 증분 팔 행동).
  settle    hold_s 동안 그대로 유지 — (judge=1 만) 네 손가락 래치 수가 judge_min_axes 아래로 loss_ticks 넘게 떨어지면 실패.
            (들린 뒤에는 FP++ 컵이 가려 옛 값으로 남을 수 있어 거리 조건을 보지 않는다.)
  done      성공 → 노드가 stop 으로 끝낸다(pd 가 마지막 팔 · 손 목표로 붙든다 — 다음 단계 ready_carry 가 이어받는다).

10.17 계획과 다른 점: 계획은 손 펌웨어 force_set 으로 손가락을 멈추려 했으나, 배포 rh_aglt 계약은 "실기 펌웨어 힘 멈춤은 증거가
없다(10.12)" 로 막힘 신호(hand_block, 10.12 실기 검출 0.93~1.00)만 쓴다. 같은 규칙을 쓴다 — 힘은 위 고정 · 앞섬 상한으로 묶는다.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Mapping

import numpy as np

from policy_control import hand_block as HB
from policy_control import rh_aglt as A
from policy_control.rh_cap import _rotvec, palm_jacobian

PHASES = ("approach", "close", "lift", "settle", "policy", "done", "failed")   # policy = ★10.20 경로 D 넘김 뒤 다음 정책
#: 10.12 실기 검증 명목값(rh_aglt.HAND_BLOCK_NOMINAL) — 접근 런의 학습 값(0.0128, pos 판정)이 아니라 배포 막힘 명목값을 쓴다
CLOSE_BLOCK = dict(A.HAND_BLOCK_NOMINAL)


class ApproachGraspError(A.RhAgltError):
    pass


#: 값마다 허용 범위(10.17 리뷰) — 값 하나로 판정 게이트 · 들기 상한 · 풀림 감지가 꺼지지 않게 한다.
#: judge_ticks ≥ 1(0 이면 래치 없이 든다) · lift_damping ≥ 0.01(DLS 특이) · lift_m ≤ 0.2 · near_m ≤ 0.12 · loss_ticks ≤ 60.
SAFE_RANGE: Mapping[str, tuple] = {
    "stop_palm_m": (0.01, 0.12), "still_rad_s": (0.005, 0.5), "still_s": (0.1, 3.0), "approach_max_s": (1.0, 30.0),
    "close_rad_s": (0.05, 2.0), "thumb_delay_s": (0.0, 2.0), "close_max_s": (0.5, 10.0),
    "judge_min_axes": (1, len(HB.FINGERS)), "near_m": (0.02, 0.12), "judge_ticks": (1, 120),
    "lift_m": (0.02, 0.20), "lift_speed": (0.01, 0.10), "lift_damping": (0.01, 0.5), "lift_extra_s": (0.0, 5.0),
    "lift_tol_m": (0.002, 0.05), "hold_s": (0.5, 10.0), "loss_ticks": (1, 60),
    "judge": (0, 1), "squeeze": (0, 1), "squeeze_s": (0.0, 3.0),
    "pin_lead_rad": (0.0, 0.1), "close_lead_rad": (0.03, 0.2),
    "handoff": (0, 1), "stop_ready": (0, 1),
}
#: ★10.20 handoff 끝 = 손바닥이 사전 파지 자리(계약 ready_r_mult · ready_h_m) 이 거리 안에서 멈춤 — ap05 학습 접근 성공 반경
#  (approach_succ_pos_m 0.02)과 같다. 다음 정책(rcg) 시작 검사는 3 cm(pour_fj_node.READY_POSE_TOL_M)라 잡음에 여유가 있다.
HANDOFF_TOL_M = 0.02


@dataclass(frozen=True)
class ApproachGraspCfg:
    stop_palm_m: float = 0.06          # 결정 7(10.17): 6 cm — 1-C 시뮬 접근 끝 거리로 조정
    still_rad_s: float = 0.05
    still_s: float = 0.5
    approach_max_s: float = 15.0       # 결정 7: 그 전에 15 s 면 멈춤 → 닫기로 넘어간다(못 쥐면 judge 가 막는다)
    close_rad_s: float = 0.6           # 손가락 목표 닫는 속도 — 컵 밀림을 줄이려 낮게(1-C 에서 조정)
    thumb_delay_s: float = 0.3         # 엄지 굽힘을 늦게 닫는다(먼저 닿은 쪽이 컵을 미는 것 완화)
    close_max_s: float = 4.0
    judge_min_axes: int = 2            # opp k=2(locking capg_j11~ 와 같은 판정)
    near_m: float = 0.08
    judge_ticks: int = 15              # 0.25 s
    lift_m: float = 0.10
    lift_speed: float = 0.05           # m/s
    lift_damping: float = 0.05
    lift_extra_s: float = 1.5          # 램프 끝 뒤 높이 도달을 기다리는 시간
    lift_tol_m: float = 0.01
    hold_s: float = 3.0                # 결정 1(10.17): 들어 올린 뒤 3 s 유지 = 1 회 성공
    loss_ticks: int = 30
    judge: int = 0                     # ★10.18 사용자: 0 = 강제 파지(판정 없이 든다) · 1 = 래치 판정(옛 동작)
    squeeze: int = 0                   # 1 = 래치된 축도 쥔 자세까지 계속 민다(강제 파지 힘)
    squeeze_s: float = 0.5             # 닫기 끝(모든 축 막힘 · 쥔 자세) 뒤 들기 전 대기
    pin_lead_rad: float = 0.0          # ★10.18 강제 파지: 막힘(접촉) 축 고정 = 실측 + 이만큼 닫는 쪽(0 = 실측에 멈춤 — 더 밀지 않음)
    close_lead_rad: float = 0.08       # ★10.18 강제 파지: 닫는 축 지령이 실측을 앞서는 상한(막힘 G 0.0218 보다 커야 래치가 걸린다)
    # ★10.20 사용자 "ap_05로 가까이 가게한 다음에, rcg_03 행하면 잡을 것 같음" — 1 = 접근 정책이 사전 파지 자리에서 멈추면
    #   스크립트 닫기 대신 stop 으로 끝낸다(pd 가 그 자세로 붙들고, 미션 다음 단계의 rcg 정책이 그 자리에서 넘겨받는다).
    #   끝 = 사전 파지 자리(HANDOFF_TOL_M) + 팔 정지 still_s. approach_max_s 안에 못 오면 실패(abort). stop_palm_m 은 안 쓴다.
    handoff: int = 0
    # ★10.20 사용자 "1-c : ap05+grasp(가능성 있음)" — 1 = 접근 정책(ap05)이 사전 파지 자리(HANDOFF_TOL_M)에 still_s 멈추면
    #   스크립트 닫기 · 들기(경로 C)로 넘어간다. ap05 는 컵 축 2.5R(source240 7.5 cm)에서 멈춰 stop_palm_m(6 cm)에 못 닿는다.
    #   approach_max_s 안에 못 오면 실패(abort — 허공에서 닫지 않는다). handoff 와 같이 쓰지 않는다.
    stop_ready: int = 0

    def __post_init__(self) -> None:
        bad = [f"{f.name}={getattr(self, f.name)} (허용 {SAFE_RANGE[f.name][0]:g}~{SAFE_RANGE[f.name][1]:g})"
               for f in fields(self)
               if not (isinstance(getattr(self, f.name), (int, float)) and math.isfinite(getattr(self, f.name))
                       and SAFE_RANGE[f.name][0] <= getattr(self, f.name) <= SAFE_RANGE[f.name][1])]
        if bad:
            raise ApproachGraspError(f"approach_grasp: 안전 범위 밖 — {bad}")
        if self.handoff and self.stop_ready:
            raise ApproachGraspError("approach_grasp: handoff(다음 정책) 와 stop_ready(스크립트 닫기) 는 같이 쓰지 않는다")
        if not self.lift_tol_m < self.lift_m:
            raise ApproachGraspError(f"approach_grasp: lift_tol_m {self.lift_tol_m} 는 lift_m {self.lift_m} 보다 작아야 한다")

    @property
    def worst_s(self) -> float:
        """가장 늦게 끝나는 경로(접근 시간 초과 → 닫기 시간 초과 직전 판정 → 들기 램프 + 대기 → 유지) [s]."""
        if self.handoff:                     # ★10.20 경로 D — 접근 뒤 같은 노드에서 넘겨받은 정책이 aglt 기본 시간을 더 쓴다
            return self.approach_max_s + DEFAULT_EPISODE_S
        return (self.approach_max_s + self.close_max_s + self.squeeze_s + self.lift_m / self.lift_speed + self.lift_extra_s
                + self.hold_s)

    @staticmethod
    def parse(text: str) -> "ApproachGraspCfg | None":
        """노드 파라미터 approach_grasp — "" = 끔, "on" = 기본값, "키=값,키=값" = 덮어쓰기(모르는 키 · 숫자 아닌 값 거부).
        JSON 이 아닌 까닭: ROS 파라미터 값은 YAML 로 읽히고, 미션 명령줄은 {…} 를 자리표시자로 바꾼다."""
        text = str(text).strip()
        if not text:
            return None
        if text == "on":
            return ApproachGraspCfg()
        names = {f.name: f for f in fields(ApproachGraspCfg)}
        kw: dict = {}
        for part in text.split(","):
            k, sep, v = part.partition("=")
            k = k.strip()
            if not sep or k not in names:
                raise ApproachGraspError(f"approach_grasp {part!r} — 'on' 또는 키=값(키 {sorted(names)})")
            try:
                kw[k] = int(v) if isinstance(getattr(ApproachGraspCfg, k), int) else float(v)
            except ValueError as exc:
                raise ApproachGraspError(f"approach_grasp {k}={v!r} — 숫자가 아니다") from exc
        return ApproachGraspCfg(**kw)

    @staticmethod
    def to_param(over: Mapping) -> str:
        """에피소드 YAML 의 {키: 값} → 노드 파라미터 문자열(빈 dict = "on"). parse 로 한 번 검사한다."""
        text = ",".join(f"{k}={over[k]}" for k in sorted(over)) or "on"
        ApproachGraspCfg.parse(text)
        return text


#: 노드 max_episode_s = 최악 경로 + 이만큼(LSTM 예열 hold_steps · 상태 지연). 실행기는 여기에 POLICY_MARGIN_S 를 더 기다린다.
EPISODE_PAD_S = 3.0
DEFAULT_EPISODE_S = 15.0          # 경로 C 가 아닌 aglt 노드(미션 max_episode_s 15.0 과 같다)


def episode_s(over: Mapping | None) -> float:
    """에피소드 YAML 의 approach_grasp(None = 끔) → 노드 max_episode_s [s]. 시간 초과는 성공이 아니라 abort 다(노드)."""
    if over is None:
        return DEFAULT_EPISODE_S
    cfg = ApproachGraspCfg.parse(ApproachGraspCfg.to_param(over))
    return float(math.ceil(cfg.worst_s + EPISODE_PAD_S))


def close_contract(c: A.RaContract) -> A.RaContract:
    """닫기 · 판정용 계약 — 막힘 명목값 + opp 판정(네 손가락, k = 호출자가 cfg 로) + 거리 조건. 관측 · 행동 해석은 그대로."""
    if not c.hand_block:
        raise ApproachGraspError("경로 C 는 hand_block 계약이 필요하다(막힘 신호로 손가락을 멈춘다)")
    block = {**{k: v for k, v in c.hand_block.items() if k not in ("g_axis_rad",)}, **CLOSE_BLOCK,
             "axes": list(A.HAND_BLOCK_AXES), "release_on_open": False}
    return replace(c, hand_block=block, grasp_rule=HB.GRASP_RULES[1], grasp_judge_axes=list(HB.FINGERS),
                   grasp_hold_rule="", grasp_release_open_rad=[], grasp_release_steps=0)


def hand_action_for(law, q_des: np.ndarray) -> np.ndarray:
    """손 법칙 raw(a) 의 역 — 원하는 목표 → 행동([-1, 1]). 범위 밖은 끝으로 자른다."""
    lo, hi = law.bounds()
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    return np.clip(2.0 * (np.asarray(q_des, float) - lo) / span - 1.0, -1.0, 1.0)


def ik_lift_arm_action(fk, arm_q, hand_q, arm_target, p0, R0, dz: float, damping: float, k_arm, lo, hi) -> np.ndarray:
    """rh_cap.ik_lift_action 과 같은 식(손바닥 p0 + dz · 방향 R0 고정, DLS) — CapContract 없이."""
    pose = fk.palm_pose(np.asarray(arm_q, float), hand_q)
    err = np.concatenate([np.asarray(p0, float) + np.array([0.0, 0.0, dz]) - pose.palm_pos,
                          _rotvec(np.asarray(R0, float) @ pose.extra["palm_rot"].T)])
    J = palm_jacobian(fk, np.asarray(arm_q, float), hand_q)
    dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(6), err)
    q_des = np.clip(np.asarray(arm_q, float) + dq, lo, hi)
    return np.clip((q_des - np.asarray(arm_target, float)) / np.asarray(k_arm, float), -1.0, 1.0)


class ApproachGrasp:
    """AgltChain 에 붙는 단계 기계. 체인은 approach 동안 정책을 돌리고 after_policy 를 부른다 — 그 뒤는 step 이 맡는다."""

    def __init__(self, cfg: ApproachGraspCfg, hz: float) -> None:
        self.cfg, self.hz = cfg, float(hz)
        self.next: tuple | None = None   # ★10.20 경로 D — (계약, 정책): 노드가 넣는다. None 이면 넘김 = stop(미션 다음 단계가 이어받음)
        self.reset()

    def reset(self) -> None:
        self.phase = "approach"
        self.t = 0                       # 이 단계 틱
        self.still = 0
        self.g_streak = 0
        self.lost = 0
        self.closed_t = -1               # 강제 파지: 닫기가 끝난(모든 축 막힘 · 쥔 자세) 틱
        self.pinned = np.zeros(6, bool)   # 강제 파지: 막힘(접촉)이 걸려 실측에 고정한 축
        self.why = ""
        self.end: tuple[str, str] | None = None
        self.q_cmd: np.ndarray | None = None
        self.p0 = self.R0 = None
        self.z0 = float("nan")
        self.last: dict = {}
        self.grip: dict | None = None    # ★10.19 쥔 순간(닫기 → 들기) 관절 상태 — 물체마다 기록해 학습 세션에 넘긴다(사용자)

    @property
    def scripted(self) -> bool:
        return self.phase not in ("approach", "policy")

    def _go(self, phase: str, why: str = "") -> None:
        self.phase, self.t, self.why = phase, 0, why or self.why
        if self.grip is not None:
            if self.last.get("rise_m") is not None:          # 들기 끝(→ settle · 실패)의 높이 — settle 에서는 last 가 바뀐다
                self.grip["rise_m"] = self.last["rise_m"]
            if phase in ("done", "failed"):
                self.grip.update(outcome=phase, end_why=why)
        if phase == "done":
            self.end = ("stop", f"approach_grasp: 들어 올려 {self.cfg.hold_s:.1f} s 유지({self.why})")
        elif phase == "failed":
            self.end = ("abort", f"approach_grasp 실패: {why}")

    # ------------------------------------------------------------ approach
    def after_policy(self, chain, m: A.RaMeas) -> None:
        """정책 한 스텝 뒤 — 멈춤 조건을 보고 닫기로 넘긴다. 대기(hold_steps) 중에는 세지 않는다."""
        if self.phase == "policy":                       # 경로 D 넘김 뒤 — 끝은 다음 정책의 목표 · lift_hold(노드 end_reason)
            return
        self.t += 1
        if chain.step_i <= chain.c.hold_steps:
            return
        d = float(np.linalg.norm(np.asarray(m.cup_pos, float) - np.asarray(m.palm_pos, float)))
        v = float(np.max(np.abs(np.asarray(m.arm_qd, float))))
        if self.cfg.handoff or self.cfg.stop_ready:
            self._after_policy_ready(chain, m, d, v)
            return
        self.still = self.still + 1 if (np.isfinite(d) and d <= self.cfg.stop_palm_m and v < self.cfg.still_rad_s) else 0
        self.last = {"palm_cup_m": round(d, 4), "arm_qd_max": round(v, 3), "still": self.still}
        if self.still >= max(1, round(self.cfg.still_s * self.hz)):
            self._start_close(chain, f"접근 끝 거리 {d:.3f} m · 정지 {self.cfg.still_s:.1f} s")
        elif self.t >= round(self.cfg.approach_max_s * self.hz):
            self._start_close(chain, f"접근 시간 {self.cfg.approach_max_s:.1f} s · 거리 {d:.3f} m")

    def _after_policy_ready(self, chain, m: A.RaMeas, d: float, v: float) -> None:
        """★10.20 사전 파지 자리에 멈추면 — handoff: 다음 정책(같은 노드) 또는 stop · stop_ready: 스크립트 닫기. 시간 안에 못 오면 abort."""
        rr, out_h, out_r = A.ready_pose_error(chain.c, m.cup_pos, m.palm_pos)
        at = bool(np.isfinite(rr) and out_r <= HANDOFF_TOL_M and out_h <= HANDOFF_TOL_M)
        self.still = self.still + 1 if (at and v < self.cfg.still_rad_s) else 0
        self.last = {"palm_cup_m": round(d, 4), "r_over_R": round(rr, 2), "off_r_m": round(out_r, 4), "off_h_m": round(out_h, 4),
                     "arm_qd_max": round(v, 3), "still": self.still}
        where = f"r/R {rr:.2f}(학습 {chain.c.ready_r_mult}) · 반경 밖 {out_r * 100:.1f} cm · 높이 밖 {out_h * 100:.1f} cm"
        if self.still >= max(1, round(self.cfg.still_s * self.hz)):
            if self.cfg.stop_ready:                      # 경로 C(1-C): 그 자리에서 닫고 든다
                self._start_close(chain, f"사전 파지 자리 정지 {self.cfg.still_s:.1f} s · {where}")
                return
            if self.next is not None:                    # 같은 노드에서 다음 정책이 이 자세로 넘겨받는다
                chain.handover(self.next[0], self.next[1], m)
                self.last = {**self.last, "handoff_t_s": round(self.t / self.hz, 2)}   # 접근 틱(대기 포함) → 넘긴 시각
                self.phase, self.t, self.why = "policy", 0, f"넘김 {where}"
                return
            self.phase, self.t, self.why = "done", 0, where
            self.end = ("stop", f"approach_grasp handoff: 사전 파지 자리 도착 · 정지 {self.cfg.still_s:.1f} s — 다음 정책에 넘김({where})")
        elif self.t >= round(self.cfg.approach_max_s * self.hz):
            self._go("failed", f"접근 시간 {self.cfg.approach_max_s:.1f} s 안에 사전 파지 자리에 멈추지 못함({where})")

    def _start_close(self, chain, why: str) -> None:
        st = chain.dec.state["arm"]
        arm, hand = st.arm_target.copy(), st.hand_target.copy()
        cc = close_contract(chain.c)
        cc = replace(cc, grasp_min_axes=int(self.cfg.judge_min_axes), grasp_near_m=float(self.cfg.near_m))
        A.validate(cc)
        chain.c = cc
        chain.dec = A.RaDecoder(cc)
        chain.dec.reset(arm_start=arm, hand_start=hand)
        self.q_cmd = hand.copy()
        self.stop_palm = self.last.get("palm_cup_m")
        self._go("close", why)

    # ------------------------------------------------------------ scripted
    def step(self, chain, meas: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
        m = meas["arm"]
        c, dec = chain.c, chain.dec
        s, st = c.side(), dec.state["arm"]
        hand_q = np.array([m.hand_q[j] for j in s.hand_joints], float)
        d = float(np.linalg.norm(np.asarray(m.cup_pos, float) - np.asarray(m.palm_pos, float)))
        self.t += 1
        assert self.q_cmd is not None, "close 전에 step"
        a_arm = np.zeros(7)
        if self.phase == "lift":
            if self.p0 is None:
                pose = chain.fk.palm_pose(np.asarray(m.arm_q, float), hand_q)
                self.p0, self.R0 = np.asarray(pose.palm_pos, float), np.asarray(pose.extra["palm_rot"], float)
                self.z0 = float(self.p0[2])
            dz = min(self.t * self.cfg.lift_speed / self.hz, self.cfg.lift_m)
            a_arm = ik_lift_arm_action(chain.fk, m.arm_q, hand_q, st.arm_target, self.p0, self.R0, dz,
                                       self.cfg.lift_damping, c.k_arm, np.asarray(s.arm_lo, float), np.asarray(s.arm_hi, float))
        if self.phase in ("close", "lift", "settle"):
            self._ramp_hand(dec, c, hand_q)
        a = np.concatenate([a_arm, hand_action_for(dec.law, self.q_cmd)])
        targets = dec.step(a, active=True, tactile_n=m.tactile_n, hand_q=hand_q, joint_force=m.joint_force, obj_palm_m=d)
        self._advance(chain, m, d)
        obs = A.build_obs(c, m, dec, chain.goal, chain.prev)   # 정책은 안 돌지만 관측 기록(bag · 상황판)은 같은 모양으로
        chain.prev = np.clip(a, -1.0, 1.0)
        chain.step_i += 1
        return obs, a, targets

    @property
    def _forced(self) -> bool:
        return not self.cfg.judge and not self.cfg.squeeze

    def _ramp_hand(self, dec, c: A.RaContract, hand_q: np.ndarray) -> None:
        """q_cmd 를 쥔 자세 쪽으로 close_rad_s 만큼 — 엄지 회전은 고정, 엄지 굽힘은 thumb_delay_s 뒤, 래치된 축은 멈춘다.
        강제 파지는 막힘이 걸린 축을 실측(+ pin_lead_rad)에 고정하고, 안 닿은 닫는 축의 앞섬을 close_lead_rad 로 묶는다."""
        assert self.q_cmd is not None
        s = c.side()
        o, g = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
        sgn = np.where(np.sign(g - o) == 0, 1.0, np.sign(g - o))
        closing = np.zeros(6, bool)
        closing[[dec.block.thumb_flex, *dec.block.fingers]] = True
        lo, hi = np.minimum(o, g), np.maximum(o, g)
        if self._forced:
            new = np.asarray(dec.block.latch, bool) & closing & ~self.pinned
            if new.any():
                self.q_cmd = np.where(new, np.clip(hand_q + sgn * self.cfg.pin_lead_rad, lo, hi), self.q_cmd)
                self.pinned = self.pinned | new
        step = np.full(6, self.cfg.close_rad_s / self.hz)
        names = list(s.hand_joints)
        thumb_rot = [i for i, n in enumerate(names) if n.endswith("thumb_1")]
        step[thumb_rot] = 0.0
        flex = dec.block.thumb_flex
        if self.phase == "close" and self.t <= round(self.cfg.thumb_delay_s * self.hz):
            step[flex] = 0.0
        if self._forced:
            step[self.pinned] = 0.0
        elif not self.cfg.squeeze:
            step[dec.block.latch] = 0.0
        nxt = self.q_cmd + sgn * step
        self.q_cmd = np.where(sgn > 0, np.minimum(nxt, np.maximum(g, self.q_cmd)), np.maximum(nxt, np.minimum(g, self.q_cmd)))
        if self._forced:                                # 안 닿은 닫는 축 — 실측 + close_lead_rad 를 넘지 않는다
            cap = hand_q + sgn * self.cfg.close_lead_rad
            capped = np.where(sgn > 0, np.minimum(self.q_cmd, cap), np.maximum(self.q_cmd, cap))
            self.q_cmd = np.where(closing & ~self.pinned, np.clip(capped, lo, hi), self.q_cmd)

    def _latched_fingers(self, dec) -> int:
        return int(dec.block.latch[dec.block.fingers].sum())

    def _close_done(self, dec, c: A.RaContract) -> bool:
        """닫는 축(엄지 굽힘 · 네 손가락)이 모두 막혔거나(래치) 쥔 자세에 닿았는가 — 강제 파지가 들기로 넘어가는 조건.
        squeeze 면 래치를 보지 않는다 — 지령이 쥔 자세에 닿아야 끝(래치된 축도 끝까지 민다)."""
        assert self.q_cmd is not None
        s = c.side()
        g = np.asarray(s.hand_grip, float)
        idx = [dec.block.thumb_flex, *dec.block.fingers]
        at_grip = [abs(float(self.q_cmd[i]) - float(g[i])) < 1e-6 for i in idx]
        if self.cfg.squeeze:
            return all(at_grip)
        return all(bool(self.pinned[i]) or a for i, a in zip(idx, at_grip))

    def _advance(self, chain, m: A.RaMeas, d: float) -> None:
        dec, cfg = chain.dec, self.cfg
        n = self._latched_fingers(dec)
        self.last = {"palm_cup_m": round(d, 4), "latched_fingers": n, "latch": dec.block.latch.astype(int).tolist(),
                     "pinned": self.pinned.astype(int).tolist(), "t_s": round(self.t / self.hz, 2)}
        if self.phase == "close" and not cfg.judge:             # ★10.18 강제 파지 — 판정 없이 든다
            if self.closed_t < 0 and self._close_done(dec, chain.c):
                self.closed_t = self.t
            self.last["closed"] = self.closed_t >= 0
            waited = self.closed_t >= 0 and self.t - self.closed_t >= round(cfg.squeeze_s * self.hz)
            if waited or self.t >= round(cfg.close_max_s * self.hz):
                self.grip = self._grip_now(chain, m, d)
                self._go("lift", f"강제 파지 — 접촉 고정 축 {int(self.pinned.sum())} · 거리 {d:.3f} m"
                                 + ("" if waited else f" · 닫기 {cfg.close_max_s:.1f} s 상한"))
            return
        if self.phase == "close":
            self.g_streak = self.g_streak + 1 if dec.grasped() else 0
            self.last["g_streak"] = self.g_streak
            if self.g_streak >= cfg.judge_ticks:
                self.grip = self._grip_now(chain, m, d)
                self._go("lift", f"쥠 {n} 손가락 · 거리 {d:.3f} m")
            elif self.t >= round(cfg.close_max_s * self.hz):
                self._go("failed", f"{cfg.close_max_s:.1f} s 안에 못 쥠(래치 손가락 {n} < {cfg.judge_min_axes} 또는 "
                                   f"거리 {d:.3f} ≥ {cfg.near_m} m) — 들지 않고 멈춘다")
            return
        self.lost = self.lost + 1 if (cfg.judge and n < cfg.judge_min_axes) else 0
        self.last["lost"] = self.lost
        if self.lost > cfg.loss_ticks:
            self._go("failed", f"{self.phase} 중 쥠이 풀림(래치 손가락 {n} 이 {cfg.loss_ticks} 틱 넘게 {cfg.judge_min_axes} 미만)")
            return
        if self.phase == "lift":
            pose = chain.fk.palm_pose(np.asarray(m.arm_q, float), [m.hand_q[j] for j in chain.c.side().hand_joints])
            rise = float(pose.palm_pos[2]) - self.z0
            self.last["rise_m"] = round(rise, 4)
            ramp = round(cfg.lift_m / cfg.lift_speed * self.hz)
            if rise >= cfg.lift_m - cfg.lift_tol_m:
                self._go("settle", f"{self.why} · 들기 {rise:.3f} m")
            elif self.t >= ramp + round(cfg.lift_extra_s * self.hz):
                self._go("failed", f"들기 높이 미달 {rise:.3f} m < {cfg.lift_m - cfg.lift_tol_m:.3f} m")
        elif self.phase == "settle" and self.t >= round(cfg.hold_s * self.hz):
            self._go("done")

    def _grip_now(self, chain, m: A.RaMeas, d: float) -> dict:
        """쥔 순간(닫기 끝 → 들기 시작)의 손 관절 — 실측 · 스크립트 지령 · 접촉 고정 축 · 래치(계약 hand_joints 순, rad)."""
        names = list(chain.c.side().hand_joints)
        dec = chain.dec
        return {"hand_joints": names, "q_meas_rad": [round(float(m.hand_q[n]), 4) for n in names],
                "q_cmd_rad": None if self.q_cmd is None else [round(float(v), 4) for v in self.q_cmd],
                "pinned": self.pinned.astype(int).tolist(),
                "latch": dec.block.latch.astype(int).tolist() if dec.block is not None else None,
                "palm_cup_m": round(d, 4), "stop_palm_m": self.stop_palm, "close_s": round(self.t / self.hz, 2)}

    def as_dict(self) -> dict:
        return {"phase": self.phase, "why": self.why, **self.last, **({"grip": self.grip} if self.grip is not None else {})}

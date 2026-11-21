#!/usr/bin/env python3
"""cap 깊이 학생(depth_h01) 시작 자세로 — 두 팔을 한 팔씩(오른팔 → 왼팔), 끝에 오른손을 cap 편 손으로. 11.18.

tilt_now 와 같은 두 걸음(plan_carry 계획 · 검사 → replay_to_pd 재생)을 자산 config/cap/cap_start 로 돌린다.
끝 자세 = config/homes/rh56f1_cap_start.yaml(오른팔 = rh_cap_student_node.LIVE_ARM_START_Q · 왼팔 = 리시버 뱅크 받기 자세).
전제: 왼손은 쉐이커를 이미 쥐고 있다(경로 f 왼손 잡기 뒤) — 왼손 안 쉐이커는 가상 컵으로 탁자 · 두 팔 사이 검사에 들어가고,
오른손은 빈손이라 --empty right 로 가상 컵을 뗀다. 왼손은 건드리지 않는다(pd 가 붙든 채).
마지막 단계(오른팔이 --sides 에 있고 --no-hand 가 아닐 때): 오른손 구동 6 → rh_cap.CAPTOP_OPEN, HAND_RAD_S 이하 램프.
학생 노드는 start 때 오른손이 그 값에서 각 HAND_TOL_RAD 안이어야 받는다(홈 손 thumb_1 1.57 이면 거부).
손 단계는 hand_grasp.run 을 그대로 쓴다(램프 시작 = pd applied 손) — 팔 = pd 세트포인트 유지 + 손 목표를 /policy_control/joint_target 에, 끝나면 episode stop
→ pd 가 마지막 팔 · 손 목표를 붙든다.

    python3 deploy/policy_control/tools/cap_now.py               # 계획 · 검사만(손은 지금 값 → 목표 · 예상 시간만 출력)
    python3 deploy/policy_control/tools/cap_now.py --execute     # ★실기: 오른팔 계획 → 재생 → 왼팔 계획 → 재생 → 오른손 펴기
그 밖 옵션(--sides · --rate-scale · --out-root)은 tilt_now 그대로, --no-hand 는 손 단계를 끈다.
기록: logs/runs/<YYMMDD>/cap/<시각>_setting_cap_start/.
종료 코드 = tilt_now 와 같다(0 끝 · 2 검사 실패 · 3 핀 · 기준 파일 불일치 · 입력 오류(--asset · --empty 는 cap_now 가 고정) · 그 밖 계획 · 재생 실패).
손 단계: 2 = 거부(손 계약을 못 읽음 · 목표가 한계 밖 — 둘은 팔이 움직이기 전에 · pd TRACKING 아님 · 손 값 없음) 또는 끝난 뒤 실측이 목표에서 HAND_TOL_RAD 밖 · 5 = SIGTERM.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np

import hand_grasp as H
import tilt_now
from tilt_pour import SIM2REAL, TOOLS, run_paths

from policy_control import rh_aglt as A
from policy_control.rh_cap import CAPTOP_OPEN

ASSET = TOOLS.parent / "config" / "cap" / "cap_start"
CONTRACT = SIM2REAL / "config" / "hand" / "rh56f1_right_hand.json"   # 오른손 관절 이름 · 한계 · policy_hz 만 쓴다
TARGET = CAPTOP_OPEN
HAND_RAD_S = 0.5
HAND_TOL_RAD = 0.10     # < rh_cap_student_node.HAND_OPEN_TOL_RAD 0.15 — 학생 시작 허용 안쪽에서 끝낸다(그 모듈은 학생 torch 를 끌어와 import 하지 않는다)
SETTLE_S = 3.0          # 램프가 목표에 닿은 뒤 실측이 들어오기를 기다리는 시간


class HandRamp:
    """hand_grasp.run 이 부르는 HandGrasp 자리 — q0(pd applied 손) → target 을 rad_s 로, 닿은 뒤 실측이 tol 안이면 끝(밖이면 failed)."""

    def __init__(self, q0, target, names: list, hz: float, rad_s: float = HAND_RAD_S, tol: float = HAND_TOL_RAD) -> None:
        self.q0, self.target, self.names, self.tol = np.asarray(q0, float), np.asarray(target, float), list(names), tol
        self.dq, self.settle_ticks = rad_s / hz, int(SETTLE_S * hz)
        self.q_cmd, self.finished, self.phase, self.why, self.n, self.dev = None, False, "ramp", "", 0, None

    def start(self, q_hand) -> None:      # run 이 주는 실측이 아니라 pd 가 지금 내는 손(applied)에서 — 첫 틱에 손이 뛰지 않게
        self.q_cmd = self.q0.copy()

    def step(self, meas) -> np.ndarray:
        if self.phase == "ramp":
            d = self.target - self.q_cmd
            self.q_cmd = np.where(np.abs(d) <= self.dq, self.target, self.q_cmd + np.sign(d) * self.dq)
            if np.array_equal(self.q_cmd, self.target):
                self.phase = "settle"
            return self.q_cmd
        self.n += 1
        self.dev = np.asarray(meas, float) - self.target
        if np.abs(self.dev).max() <= self.tol:
            self.finished, self.phase, self.why = True, "done", f"실측이 목표에서 {np.abs(self.dev).max():.3f} rad 안"
        elif self.n >= self.settle_ticks:
            bad = " · ".join(f"{j} {e:+.3f}" for j, e in zip(self.names, self.dev) if abs(e) > self.tol)
            self.finished, self.phase, self.why = True, "failed", f"{SETTLE_S:g} s 뒤 실측 − 목표 > {self.tol}: {bad}"
        return self.q_cmd

    def latched_fingers(self) -> int:
        return 0

    def as_dict(self) -> dict:
        return {"phase": self.phase, "why": self.why, "target": self.target.tolist(),
                "dev": None if self.dev is None else np.round(self.dev, 4).tolist()}


def hand_q(names: list) -> list[float]:
    """계획만 모드의 오른손 실측(구독만)."""
    from check_path_start import measure
    q = measure(names)
    return [float(q[j]) for j in names]


def applied_hand(names: list) -> list[float] | None:
    """pd 가 지금 내는 오른손 목표(/policy_control/pd_right/applied — pd 가 손을 붙들 때만 손 이름이 실린다). 구독만."""
    import rclpy
    from sensor_msgs.msg import JointState
    rclpy.init()
    node, got = None, {}
    try:
        node = rclpy.create_node("cap_now_applied")
        node.create_subscription(JointState, "/policy_control/pd_right/applied", lambda m: got.update(zip(m.name, m.position)), 10)
        t0 = time.monotonic()
        while time.monotonic() - t0 < H.MEAS_WAIT_S and not all(n in got for n in names):
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return [float(got[n]) for n in names] if all(n in got for n in names) else None


def hand_contract():
    """움직이기 전 거부 — 오른손 계약 로드 · 오른손인지 · TARGET 이 한계 안인지(상수와 파일만 본다). 거부면 None."""
    try:
        c = A.load_contract(CONTRACT)
        s = c.side()
    except (OSError, ValueError, KeyError, TypeError) as exc:      # read_text · json/RhAgltError(ValueError) · sides 키 · RaSide 필드
        print(f"[cap_now] ✗ 오른손 거부 — 손 계약 {CONTRACT} 를 못 읽었다: {exc} · 아무것도 움직이지 않았다", flush=True)
        return None
    target = np.asarray(TARGET, float)
    lo, hi = np.asarray(s.hand_lim_lo, float), np.asarray(s.hand_lim_hi, float)
    if not all(j.startswith("r_") for j in s.hand_joints) or np.any(target < lo) or np.any(target > hi):
        print(f"[cap_now] ✗ 오른손 거부 — 목표 {list(TARGET)} 가 {s.hand_joints} 한계 [{lo.tolist()}, {hi.tolist()}] 밖"
              " · 아무것도 움직이지 않았다", flush=True)
        return None
    return c


def hand_step(c, execute: bool) -> int:
    s = c.side()
    target = np.asarray(TARGET, float)
    if not execute:
        try:
            q = np.asarray(hand_q(list(s.hand_joints)), float)
        except (SystemExit, KeyError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            print(f"[cap_now] ✗ 오른손 거부 — 손 실측을 못 읽었다: {exc}", flush=True)
            return 2
        print(f"[cap_now] 오른손(계획만) {np.round(q, 3).tolist()} → {list(TARGET)} · "
              f"약 {np.abs(target - q).max() / HAND_RAD_S:.1f} s({HAND_RAD_S} rad/s)", flush=True)
        return 0
    q0 = applied_hand(list(s.hand_joints))
    if q0 is None:
        print("[cap_now] ✗ 오른손 거부 — pd applied 에 오른손 값이 없다(pd 가 손을 붙들고 있지 않다) · 아무것도 보내지 않았다", flush=True)
        return 2
    ramp = HandRamp(q0, target, list(s.hand_joints), float(c.policy_hz))
    rc, res = H.run("right", ramp, c)
    print(f"[cap_now] {'✓' if rc == 0 else '✗'} 오른손 {res.get('phase')} — {res.get('why')} · 실측 − 목표 {res.get('dev')}",
          flush=True)
    return 2 if rc in (3, 4) else rc     # hand_grasp.run 3 = pd 이상 · 4 = 측정 없음 → 거부


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--no-hand", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--sides", default="right,left")
    a, _ = ap.parse_known_args(argv)
    fixed = [x for x in argv if any(f.startswith(x.split("=")[0]) and len(x.split("=")[0]) > 2 for f in ("--asset", "--empty"))]
    if fixed:       # argparse 줄임말(--ass 등)도 잡는다
        print(f"[cap_now] ✗ 입력 오류 — {fixed}: --asset · --empty 는 cap_now 가 고정해서 넘긴다", flush=True)
        return 3
    hand = not a.no_hand and "right" in [x.strip() for x in a.sides.split(",")]
    c = hand_contract() if hand else None
    if hand and c is None:
        return 2
    argv = [x for x in argv if x != "--no-hand"]
    if "--out-root" not in argv:
        argv += ["--out-root", str(run_paths.day_dir("cap"))]
    rc = tilt_now.main(["--asset", str(ASSET), "--empty", "right", *argv])
    if rc != 0 or not hand:
        return rc
    return hand_step(c, a.execute)


if __name__ == "__main__":
    raise SystemExit(main())

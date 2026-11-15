#!/usr/bin/env python3
# openarm-can-zero-position-calibration 의 최소 수정 사본 (원본: robot_control/ros_ws/src/openarm_can/setup, SDK 396cf53,
# 원본 SHA256 c6cdda3b1ee3d44637fc08a6c28724d90ce2b45cb0969c8092f31c7baee80423). 원본은 고치지 않는다.
# 바뀐 것은 아래 4가지뿐이다(영점 쓰는 방식·순서·게인·속도는 원본 그대로):
#   1) stock 그리퍼(ID 8) 초기화·사전 닫기·범프 제거 (이 로봇은 RH56F1 이고 L7 만 남음)
#   2) 무한 탐색 루프에 시간(120 s)·이동량(200°) 상한 — 초과하면 영점을 쓰지 않고 예외 → finally 에서 disable
#   3) --execute 가 없으면 CAN 을 열지 않음
#   4) 실행 전·후 관절값 출력
#   5) (11.15) 끝 판정 최소 이동량: j1 30°, j4 0°(차렷=펴짐 스톱), 나머지 3° 이상 움직인 뒤에만 스톱 인정. 그 전에 지령-실제 차이가
#      5° 를 넘으면(못 움직이고 막힘) 예외. — 11.15 왼팔(손 장착)에서 j1 이 출발 즉시(0.94°) 스톱으로 오판되어
#      영점이 차렷 기준 ~63° 틀어짐(logs/runs/261115/arm_zero_mfr/left_hand_153949.log).
#   6) (11.15) 영점 자리 = 스톱 실제 위치 + 그 뒤 지령 이동량. 쓰기 전 위치 지령을 적분으로 밀어 실제를 0.3° 안으로 맞추고
#      (refine_to_goals, 15 s·±15° 한도), 모터를 끈 직후 다시 1° 안인지 확인해야 쓴다. 손 장착 시 처짐·마찰 보상.
#   7) (11.15) --shift-joint/--shift-deg: 관절 하나만 영점을 옮긴다(스톱 찾기 없음, 다른 관절 영점 그대로). 이동량 상한 ·
#      끈 뒤 안정 확인 · 같은 팔·관절 30 분 안 재실행 거부(--again) · 기록 logs/arm_zero_shift_history.jsonl.
# 사용: 팔 하나씩. 손·ADAPTOR 분리, 지지, 주변 정리, 다른 CAN 프로세스 종료 후:
#   ~/.cache/arm-zero/venv/bin/python scripts/ops/openarm_zero_mfr.py --arm-side right_arm --execute
# -*- coding: utf-8 -*-
#
# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openarm_can as oa
import argparse
import json
import sys
from pathlib import Path
import time
import numpy as np
from enum import IntEnum

# ---------- IDs / Limits / Signs ----------


class JointID(IntEnum):
    J1 = 0
    J2 = 1
    J3 = 2
    J4 = 3
    J5 = 4
    J6 = 5
    J7 = 6
    GRIPPER = 0


MECH_LIM_V1 = {
    JointID.J1:      [np.deg2rad(-80),  np.deg2rad(200)],
    JointID.J2:      [np.deg2rad(-100), np.deg2rad(100)],
    JointID.J3:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.J4:      [np.deg2rad(0),    np.deg2rad(140)],
    JointID.J5:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.J6:      [np.deg2rad(-45),  np.deg2rad(45)],
    JointID.J7:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.GRIPPER: [np.deg2rad(-60),  np.deg2rad(0)],
}

MECH_LIM_V2_RIGHT = {
    JointID.J1:      [np.deg2rad(-80),  np.deg2rad(200)],
    JointID.J2:      [np.deg2rad(-100), np.deg2rad(100)],
    JointID.J3:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.J4:      [np.deg2rad(0),    np.deg2rad(140)],
    JointID.J5:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.J6:      [np.deg2rad(-45),  np.deg2rad(45)],
    JointID.J7:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.GRIPPER: [0.0,              -np.pi / 2.0],
}

MECH_LIM_V2_LEFT = {
    JointID.J1:      [np.deg2rad(-80),  np.deg2rad(200)],
    JointID.J2:      [np.deg2rad(-100), np.deg2rad(100)],
    JointID.J3:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.J4:      [np.deg2rad(0),    np.deg2rad(140)],
    JointID.J5:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.J6:      [np.deg2rad(-45),  np.deg2rad(45)],
    JointID.J7:      [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.GRIPPER: [0.0,               np.pi / 2.0],
}

# Per-joint sign convention used in ideal-zero delta computation
JOINT_SIGN = {
    JointID.J1: -1.0, JointID.J2: -1.0, JointID.J3: +1.0, JointID.J4: +1.0,
    JointID.J5: +1.0, JointID.J6: +1.0, JointID.J7: +1.0, JointID.GRIPPER: +1.0,
}

# ---------- Core motion primitives ----------
SEARCH_TIMEOUT_S = 120.0   # [openarm_zero_mfr] 원본은 무한 루프. 이 시간·이동량 안에 스톱이 없으면 영점을 쓰지 않고 끝낸다.
SEARCH_MAX_RAD = np.deg2rad(200)
MIN_TRAVEL_RAD = {0: np.deg2rad(30.0),   # [openarm_zero_mfr] 관절별 최소 이동량(idx 0 = j1). 없으면 기본값
                  3: 0.0}                # j4: 영점=팔꿈치 펴짐 스톱(범위 0~140°). 차렷에서 이미 스톱이라 이동 0 이 정상(11.15 양팔 확인)
MIN_TRAVEL_DEFAULT_RAD = np.deg2rad(3.0)
EARLY_LAG_MAX_RAD = np.deg2rad(5.0)      # 최소 이동 전 지령-실제 차이 상한 (넘으면 막힌 것으로 보고 중단)
ZERO_TOL_RAD = np.deg2rad(1.0)           # 영점 쓰기 전 지령-실제 차이 상한
LAST_TARGET = {}                         # idx -> 마지막으로 보낸 위치 지령 [rad]
LAST_GAINS = {}                          # idx -> (kp, kd) 마지막 지령 게인
HIT = {}                                 # idx -> 스톱에서 잰 실제 위치 [rad]
NET = {}                                 # idx -> 스톱 이후 interpolate 로 더한 이동량 합 [rad]
# 영점 자리(목표) = HIT + NET. 원본은 interpolate 가 '현재 실제 위치'에서 출발해 처짐·마찰 오차가 누적되고,
# 영점은 실제 위치에 써진다(손 장착 시 11.15 j1 ±6.6°, j4 9°, j3 3.4° 차이). 그래서 쓰기 전에 목표로 보정한다.
REFINE_KI = 2.0                          # [1/s] 지령 += KI·(목표-실제)·dt
REFINE_TOL_RAD = np.deg2rad(0.3)
REFINE_HOLD_S = 0.5
REFINE_TIMEOUT_S = 15.0
REFINE_MAX_PUSH_RAD = np.deg2rad(15.0)   # 지령이 목표에서 벗어날 수 있는 최대량


def interpolate(openarm, comp, joint_id: JointID, delta_rad,
                kp=52.0, kd=1.5, torque_assist=0.0, interp_time=2.0):
    """Linear interp: current -> current+delta with MIT control."""
    idx = int(joint_id)
    m = comp.get_motors()[idx]
    q0 = m.get_position()
    q1 = q0 + float(delta_rad)
    n_steps, dt = 500, (interp_time / 500.0)
    tau = np.copysign(1.0, q1 - q0) * abs(torque_assist)
    if idx in NET:
        NET[idx] += float(delta_rad)
    LAST_GAINS[idx] = (kp, kd)
    for i in range(n_steps + 1):
        alpha = i / n_steps
        q = q0 + (q1 - q0) * alpha
        comp.mit_control_one(idx, oa.MITParam(kp, kd, q, 0.0, tau))
        LAST_TARGET[idx] = q
        openarm.recv_all()
        time.sleep(dt)


def _hit_thresholds(comp, idx):
    """Return (dq_th, tau_th) based on joint/gripper."""
    is_gripper = (len(comp.get_motors()) == 1)
    if is_gripper:
        return 0.3, 0.3
    if idx == 0:
        return 0.0125, 5.0
    return 0.1, 2.0


def bump_to_limit(openarm, comp, joint_id,
                  step_deg=0.2, kp=45.0, kd=1.2, torque_bias=0.0):
    """Step until mechanical stop. Return traveled delta [rad]."""
    idx = int(joint_id)
    step_rad = np.deg2rad(step_deg)
    tau_cmd = np.copysign(1.0, step_deg) * abs(torque_bias)

    motors = comp.get_motors()
    q_start = motors[idx].get_position()
    q_target = q_start

    dq_th, tau_th = _hit_thresholds(comp, idx)
    LAST_GAINS[idx] = (kp, kd)

    t_begin = time.monotonic()
    while True:
        if time.monotonic() - t_begin > SEARCH_TIMEOUT_S or abs(q_target - q_start) > SEARCH_MAX_RAD:
            raise RuntimeError(f'stop not found: joint {joint_id.name} (timeout/travel) — zero NOT written')
        q_target += step_rad
        comp.mit_control_one(idx, oa.MITParam(kp, kd, q_target, 0.0, tau_cmd))
        LAST_TARGET[idx] = q_target
        openarm.recv_all()
        time.sleep(0.005)

        m = comp.get_motors()[idx]
        travel = abs(m.get_position() - q_start)
        if travel < MIN_TRAVEL_RAD.get(idx, MIN_TRAVEL_DEFAULT_RAD):
            lag = abs(q_target - m.get_position())
            if lag > EARLY_LAG_MAX_RAD:
                raise RuntimeError(
                    f'joint {joint_id.name}: 최소 이동량 전에 막힘 (이동 {np.rad2deg(travel):.1f}°, '
                    f'지령 차이 {np.rad2deg(lag):.1f}°) — 출발 자세 확인, zero NOT written')
            continue
        if np.abs(m.get_velocity()) < dq_th and np.abs(m.get_torque()) > tau_th:
            delta_rad = m.get_position() - q_start
            delta_deg = np.rad2deg(delta_rad)
            print(f"[INFO] mechanical stop (Joint {joint_id.name}): "
                  f"{delta_rad:.4f} rad / {delta_deg:.2f}°")
            HIT[idx] = float(m.get_position())
            NET[idx] = 0.0
            return float(delta_rad)


def calc_delta_to_zero_pos_joint(initial_rad: float,
                                 ideal_limit_rad: float,
                                 delta_to_stop_rad: float,
                                 joint_id: JointID) -> float:
    """Relative delta from 'hit position' to 'ideal limit', with per-joint sign."""
    q_hit = initial_rad + delta_to_stop_rad
    delta_to_ideal = ideal_limit_rad - q_hit
    return float(JOINT_SIGN.get(joint_id, 1.0) * delta_to_ideal)


def move_to_precise_home(openarm, arm_goal_abs):
    """TODO: PI torque-only refine to arm_goal_abs (absolute)."""
    pass


def zero_goals():
    """[openarm_zero_mfr] 관절별 영점 자리 = 스톱 실제 위치 + 그 뒤 지령 이동량."""
    missing = [f'j{i + 1}' for i in range(7) if i not in HIT]
    if missing:
        raise RuntimeError(f'스톱을 찾지 않은 관절 {missing} — zero NOT written')
    return [HIT[i] + NET[i] for i in range(7)]


def refine_to_goals(openarm, comp, goals):
    """[openarm_zero_mfr] 실제 위치가 goals 에 들 때까지 위치 지령을 적분으로 밀어 준다(처짐·마찰 보상).
    7관절 모두 REFINE_TOL 안에 REFINE_HOLD_S 동안 머물면 끝. 못 하면 예외(영점 쓰지 않음)."""
    cmd = [LAST_TARGET.get(i, goals[i]) for i in range(7)]
    dt = 0.005
    t0 = time.monotonic()
    t_ok = None
    while True:
        openarm.recv_all()
        q = [m.get_position() for m in comp.get_motors()]
        err = [goals[i] - q[i] for i in range(7)]
        now = time.monotonic()
        if all(abs(e) < REFINE_TOL_RAD for e in err):
            t_ok = t_ok or now
            if now - t_ok >= REFINE_HOLD_S:
                print('[REFINE] 완료 %.1fs, 목표-실제 [deg]:' % (now - t0),
                      {f'j{i + 1}': round(np.rad2deg(e), 2) for i, e in enumerate(err)})
                return
        else:
            t_ok = None
        if now - t0 > REFINE_TIMEOUT_S:
            raise RuntimeError('영점 자리 보정 시간 초과, 목표-실제 [deg]: '
                               + str({f'j{i + 1}': round(np.rad2deg(e), 2) for i, e in enumerate(err)})
                               + ' — zero NOT written')
        for i in range(7):
            c = cmd[i] + REFINE_KI * err[i] * dt
            cmd[i] = float(np.clip(c, goals[i] - REFINE_MAX_PUSH_RAD, goals[i] + REFINE_MAX_PUSH_RAD))
            kp, kd = LAST_GAINS.get(i, (52.0, 1.5))
            comp.mit_control_one(i, oa.MITParam(kp, kd, cmd[i], 0.0, 0.0))
            LAST_TARGET[i] = cmd[i]
        time.sleep(dt)

# ---------- [openarm_zero_mfr] (11.15) 관절 하나 영점 옮기기 — 스톱 찾기 없음 ----------
HOLD_KP = [300, 300, 150, 150, 40, 40, 30]   # main 의 '지금 자세 유지' 게인과 같다
HOLD_KD = [2.5, 2.5, 2.5, 2.5, 0.8, 0.8, 0.8]
SHIFT_MAX_DEG = 15.0
SHIFT_TOL_RAD = np.deg2rad(0.5)              # 모터 끈 뒤(안정) 실제가 목표에서 이만큼 안이어야 쓴다
SHIFT_MOVE_MAX_RAD = np.deg2rad(SHIFT_MAX_DEG + 3.0)   # 출발 → 영점 자리 이동량 상한(출발은 차렷 근처라는 전제)
SETTLE_S, STABLE_RAD = 0.3, np.deg2rad(0.05)            # 끈 뒤 처짐이 멈출 때까지 기다리고 두 번 읽어 같은지 본다
SHIFT_REPEAT_S = 1800.0                      # 같은 팔·관절을 이 안에 다시 옮기려면 --again(11.15 같은 명령 두 번 실행 사고)
SHIFT_HISTORY = Path(__file__).resolve().parents[2] / 'logs' / 'arm_zero_shift_history.jsonl'


def recent_shift(arm_side, joint, now=None, path=SHIFT_HISTORY):
    """같은 팔·관절의 최근 기록(SHIFT_REPEAT_S 안) 또는 None."""
    now = time.time() if now is None else now
    if not path.exists():
        return None
    hits = [r for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())
            if r.get('arm_side') == arm_side and r.get('joint') == joint and now - r.get('t', 0) < SHIFT_REPEAT_S]
    return hits[-1] if hits else None


def record_shift(arm_side, joint, shift_deg, done_deg, path=SHIFT_HISTORY):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps({'t': time.time(), 'at': time.strftime('%F %T'), 'arm_side': arm_side, 'joint': joint,
                             'shift_deg': shift_deg, 'done_deg': round(done_deg, 3)}) + '\n')


def send_set_zero_one(iface, can_id):
    """DaMiao 한 모터: set zero(FF×7 FE) → disable(FF×7 FD). openarm-can-cli set-zero 와 같은 classic CAN 프레임.
    모터는 이미 꺼져 있으므로 앞 disable 은 뺐다(점검한 자리에 바로 쓰게)."""
    import socket
    import struct
    with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as s:
        s.bind((iface,))
        for last in (0xFE, 0xFD):
            s.send(set_zero_frame(can_id, last))
            time.sleep(0.05)


def set_zero_frame(can_id, last=0xFE):
    import struct
    return struct.pack("=IB3x8s", can_id, 8, bytes([0xFF] * 7 + [last]))


def settled_positions(openarm, arm):
    """끈 뒤 처짐이 멈춘 위치 — SETTLE_S 기다린 뒤 refresh 로 두 번 읽어 STABLE_RAD 안이어야."""
    reads = []
    for _ in range(2):
        time.sleep(SETTLE_S)
        openarm.refresh_all()
        time.sleep(0.02)
        openarm.recv_all()
        reads.append(np.array([m.get_position() for m in arm.get_motors()]))
    return reads[-1], float(np.abs(reads[1] - reads[0]).max())


def shift_one(openarm, arm, idx, shift_rad, iface):
    """관절 idx 의 읽기값을 shift 만큼 옮긴다(새 읽기 = 옛 읽기 + shift). 옛 읽기 −shift 자리에 세우고 그 자리를 0 으로 쓴다.
    나머지 관절은 지금 자세를 유지하고 영점도 건드리지 않는다. 반환 = 실제로 옮긴 양 [rad](= −모터 끈 직후 옛 읽기)."""
    openarm.recv_all()
    q0 = [m.get_position() for m in arm.get_motors()]
    target = -float(shift_rad)
    if abs(target - q0[idx]) > SHIFT_MOVE_MAX_RAD:
        raise RuntimeError(f'j{idx + 1} 이동량 {np.rad2deg(abs(target - q0[idx])):.1f}° > {np.rad2deg(SHIFT_MOVE_MAX_RAD):.0f}° — '
                           f'출발 자세(차렷)를 확인. CAN 명령 없음, zero NOT written')
    for i in range(7):
        LAST_TARGET[i], LAST_GAINS[i] = q0[i], (HOLD_KP[i], HOLD_KD[i])
    print(f'[SHIFT] j{idx + 1}: 지금 {np.rad2deg(q0[idx]):+.2f}° → {np.rad2deg(target):+.2f}° 에 세우고 영점')
    interpolate(openarm, arm, JointID(idx), target - q0[idx], kp=HOLD_KP[idx], kd=HOLD_KD[idx], interp_time=2.0)
    goals = list(q0)
    goals[idx] = target
    refine_to_goals(openarm, arm, goals)
    openarm.disable_all()                    # 팔 전체 토크 해제(라이브러리에 한 모터 끄기가 없다) — 받침 필요
    q_off, wobble = settled_positions(openarm, arm)
    err = np.rad2deg(target - q_off[idx])
    print(f'[CHECK] 모터 끈 뒤 {SETTLE_S * 2:.1f} s j{idx + 1} 목표-실제 {err:+.2f}° · 두 읽기 차 {np.rad2deg(wobble):.3f}°')
    if wobble > STABLE_RAD:
        raise RuntimeError(f'끈 뒤 팔이 아직 움직인다({np.rad2deg(wobble):.2f}°, 받침 확인) — zero NOT written')
    if abs(target - q_off[idx]) > SHIFT_TOL_RAD:
        raise RuntimeError(f'모터 끈 뒤 j{idx + 1} 자리 벗어남 {err:+.2f}° (허용 {np.rad2deg(SHIFT_TOL_RAD):.1f}°, 손 받침) — zero NOT written')
    try:
        send_set_zero_one(iface, 0x01 + idx)
    except OSError as error:
        raise RuntimeError(f'영점 프레임 전송 중 오류({error}) — j{idx + 1} 영점 기록 상태 불명, jog 로 확인할 것') from error
    print(f'wrote zero position to j{idx + 1} only (CAN id 0x{0x01 + idx:02x})')
    openarm.refresh_all()
    time.sleep(0.1)
    openarm.recv_all()
    q_new = [m.get_position() for m in arm.get_motors()]
    # 11.15 실측: 모터를 끈 채로는 j6 읽기가 갱신되지 않아 영점 전 값(+6.1°)이 그대로 나왔다 — 이 줄로 성공 여부를 판정하지 않는다
    print('[AFTER] 끈 상태 읽기 [deg](갱신 안 될 수 있음, 판정에 쓰지 말 것 — 다시 실행하면 두 번 옮겨진다):',
          [round(np.rad2deg(v), 2) for v in q_new])
    return -float(q_off[idx])

# ---------- One function per arm-side to keep intentions explicit ----------


def _run_right_sequence(openarm, arm, grip, mech_lim):
    openarm.recv_all()
    d_grip = 0.0  # 그리퍼 없음
    d_j4 = bump_to_limit(openarm, arm,  JointID.J4, step_deg=-0.2)
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J2, np.deg2rad(5),  interp_time=0.4)
    time.sleep(0.5)
    d_j3 = bump_to_limit(openarm, arm,  JointID.J3)
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J3, -
                mech_lim[JointID.J3][1], interp_time=1.0)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J2, -np.deg2rad(5), interp_time=0.4)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J4,  np.pi / 2.0)
    time.sleep(0.5)

    d_j5 = bump_to_limit(openarm, arm,  JointID.J5)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J5, -mech_lim[JointID.J5][1])
    time.sleep(0.5)

    d_j6 = bump_to_limit(openarm, arm,  JointID.J6)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J6, -mech_lim[JointID.J6][1])
    time.sleep(0.5)

    d_j7 = bump_to_limit(openarm, arm,  JointID.J7)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J7, -mech_lim[JointID.J7][1])
    time.sleep(0.5)

    d_j2 = bump_to_limit(openarm, arm,  JointID.J2, step_deg=-0.2)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J2, np.deg2rad(
        10), interp_time=0.9, kp=280, kd=2.0)
    time.sleep(0.5)

    d_j1 = bump_to_limit(openarm, arm,  JointID.J1,
                         step_deg=-0.2, kp=280, kd=2.1)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J1, np.deg2rad(80))
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J4, -np.pi / 2.0)
    time.sleep(2.5)

    ideal = {
        JointID.J7: np.pi / 2.0,
        JointID.J6: np.pi / 4.0,
        JointID.J5: np.pi / 2.0,
        JointID.J4: 0.0,
        JointID.J3: np.pi / 2.0,
        JointID.J2: np.deg2rad(-10),
        JointID.J1: np.deg2rad(-80),
    }
    deltas = [d_j1, d_j2, d_j3, d_j4, d_j5, d_j6, d_j7, d_grip]
    return ideal, deltas


def _run_left_sequence(openarm, arm, grip, mech_lim, robot_version='v1'):
    openarm.recv_all()
    d_grip = 0.0  # 그리퍼 없음
    d_j4 = bump_to_limit(openarm, arm,  JointID.J4, step_deg=-0.2)
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J2, -np.deg2rad(5), interp_time=0.4)
    time.sleep(0.5)
    d_j3 = bump_to_limit(openarm, arm,  JointID.J3)
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J3, -
                mech_lim[JointID.J3][1], interp_time=1.0)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J2,  np.deg2rad(5),  interp_time=0.4)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J4,  np.pi / 2.0)
    time.sleep(0.5)

    d_j5 = bump_to_limit(openarm, arm,  JointID.J5)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J5, -mech_lim[JointID.J5][1])
    time.sleep(0.5)

    d_j6 = bump_to_limit(openarm, arm,  JointID.J6)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J6, -mech_lim[JointID.J6][1])
    time.sleep(0.5)

    d_j7 = bump_to_limit(openarm, arm,  JointID.J7)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J7, -mech_lim[JointID.J7][1])
    time.sleep(0.5)

    d_j2 = bump_to_limit(openarm, arm,  JointID.J2)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J2, -np.deg2rad(10),
                interp_time=0.9, kp=280, kd=2.0)
    time.sleep(0.5)

    d_j1 = bump_to_limit(openarm, arm,  JointID.J1, kp=280, kd=2.1)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J1, -np.deg2rad(80))
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J4, -np.pi / 2.0)
    time.sleep(2.5)

    ideal = {
        JointID.J7: np.pi / 2.0,
        JointID.J6: np.pi / 4.0,
        JointID.J5: np.pi / 2.0,
        JointID.J4: 0.0,
        JointID.J3: np.pi / 2.0,
        JointID.J2: np.deg2rad(10),
        JointID.J1: np.deg2rad(80),
    }
    deltas = [d_j1, d_j2, d_j3, d_j4, d_j5, d_j6, d_j7, d_grip]
    return ideal, deltas

# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(
        description='Zero-pos calibration (compact)')
    parser.add_argument('--canport',       type=str, default=None)

    parser.add_argument('--arm-side',      type=str, default='right_arm',
                        choices=['right_arm', 'left_arm'])
    parser.add_argument('--robot-version', type=str, default='v2',
                        choices=['v1', 'v2'])
    parser.add_argument('--execute', action='store_true', help='없으면 계획만 출력하고 CAN 을 열지 않는다')
    parser.add_argument('--shift-joint', type=int, choices=range(1, 8), default=None,
                        help='(11.15) 이 관절 하나만 영점을 옮긴다(스톱 찾기 없음). --shift-deg 와 같이')
    parser.add_argument('--shift-deg', type=float, default=None, help='새 읽기 = 옛 읽기 + 이 값 [deg]')
    parser.add_argument('--again', action='store_true', help=f'같은 팔·관절을 {SHIFT_REPEAT_S / 60:.0f} 분 안에 다시 옮길 때만(의도한 두 번째)')
    args = parser.parse_args()
    if (args.shift_joint is None) != (args.shift_deg is None):
        parser.error('--shift-joint 와 --shift-deg 는 같이 준다')
    if args.shift_joint is not None and '--arm-side' not in ' '.join(sys.argv):
        parser.error('--shift-joint 에는 --arm-side 를 꼭 준다(기본 right_arm 으로 다른 팔 영점을 쓰지 않게)')
    if args.shift_joint is not None and args.execute and not args.again:
        last = recent_shift(args.arm_side, args.shift_joint)
        if last:
            parser.error(f"{args.arm_side} j{args.shift_joint} 를 {last['at']} 에 이미 {last['done_deg']:+.2f}° 옮겼다 — 그 결과는 "
                         f"끈 상태 읽기로는 안 보인다. jog 로 확인한 뒤 정말 또 옮길 때만 --again")
    if args.shift_deg is not None and not 0.05 <= abs(args.shift_deg) <= SHIFT_MAX_DEG:
        parser.error(f'--shift-deg 는 0.05~{SHIFT_MAX_DEG:.0f}° 크기만 받는다')
    if args.shift_joint is not None and not args.execute:
        print(f'dry: j{args.shift_joint} 를 지금 읽기 {-args.shift_deg:+.2f}° 자리로 옮겨 그 자리를 0 으로 쓴다'
              f'(새 읽기 = 옛 + {args.shift_deg:+.2f}°). 나머지 관절은 지금 자세 유지, 영점 그대로. 되돌리기 = 부호 반대로 같은 명령')
        return
    if not args.execute:
        print('dry: --execute 없음 — CAN 열지 않음. 순서 j4→j3→j5→j6→j7→j2→j1, 그리퍼·목 지령 없음, 마지막에 7관절 영점을 한 번에 씀')
        print(f'     끝 판정 최소 이동량: j1 {np.rad2deg(MIN_TRAVEL_RAD[0]):.0f}°, j4 0°, 나머지 {np.rad2deg(MIN_TRAVEL_DEFAULT_RAD):.0f}° '
              f'(그 전 지령 차이 > {np.rad2deg(EARLY_LAG_MAX_RAD):.0f}° 면 중단)')
        print(f'     영점 쓰기 전: 목표(스톱+이동량)로 {np.rad2deg(REFINE_TOL_RAD):.1f}° 안까지 보정, 모터 끈 직후 {np.rad2deg(ZERO_TOL_RAD):.0f}° 안이어야 씀. 출발 자세: 차렷')
        return
    if args.canport is None:
        args.canport = 'can1' if args.arm_side == 'left_arm' else 'can0'
    print(f"parser arg : {args}")

    # Select mech_lim based on version and side
    if args.robot_version == 'v2':
        mech_lim = MECH_LIM_V2_RIGHT if args.arm_side == 'right_arm' else MECH_LIM_V2_LEFT
    else:
        mech_lim = MECH_LIM_V1

    # Init
    openarm = oa.OpenArm(args.canport, True)
    openarm.init_arm_motors(
        [oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM4340, oa.MotorType.DM4340,
         oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310],
        [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
        [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
    )
    openarm.set_callback_mode_all(oa.CallbackMode.STATE)

    print("Enabling...")
    openarm.enable_all()
    time.sleep(0.1)
    print("Enabled...")

    # Hold current pose (light PD)
    openarm.recv_all()
    arm = openarm.get_arm()
    grip = None
    am = arm.get_motors()

    initial_arm_q = [m.get_position() for m in am]
    print('[BEFORE] initial q [rad]:', [round(v, 4) for v in initial_arm_q])

    arm_params = [oa.MITParam(kp, kd, q, 0.0, 0.0)
                  for kp, kd, q in zip([300, 300, 150, 150, 40, 40, 30],
                                       [2.5, 2.5, 2.5, 2.5, 0.8, 0.8, 0.8],
                                       initial_arm_q)]
    arm.mit_control_all(arm_params)
    openarm.recv_all()

    try:
        if args.shift_joint is not None:
            print(f'[SHIFT] {args.arm_side} · {args.canport} · j{args.shift_joint} {args.shift_deg:+.2f}° — 팔 전체 토크가 잠깐 풀린다(받침)')
            done = shift_one(openarm, arm, args.shift_joint - 1, np.deg2rad(args.shift_deg), args.canport)
            record_shift(args.arm_side, args.shift_joint, args.shift_deg, float(np.rad2deg(done)))
            print(f'[SHIFT] j{args.shift_joint} 실제로 옮긴 양 {np.rad2deg(done):+.2f}° '
                  f'(되돌리기: --shift-joint {args.shift_joint} --shift-deg {-np.rad2deg(done):+.2f})')
            return
        # Run side-specific calibration sequence
        if args.arm_side == 'right_arm':
            ideal, deltas = _run_right_sequence(openarm, arm, grip, mech_lim)
        else:
            ideal, deltas = _run_left_sequence(
                openarm, arm, grip, mech_lim, args.robot_version)

        # Compute ideal deltas per joint
        joint_order = [JointID.J1, JointID.J2, JointID.J3, JointID.J4,
                       JointID.J5, JointID.J6, JointID.J7, JointID.GRIPPER]
        ideal_delta_map = {}
        for j, d in zip(joint_order, deltas):
            if j == JointID.GRIPPER:
                print("ideal[GRIPPER] skipped")
                continue
            val = calc_delta_to_zero_pos_joint(
                initial_rad=initial_arm_q[int(j)],
                ideal_limit_rad=ideal[j],
                delta_to_stop_rad=d,
                joint_id=j
            )
            ideal_delta_map[j] = val

        # Absolute goal = initial + ideal_delta
        arm_goal_abs = [initial_arm_q[i] +
                        ideal_delta_map.get(JointID(i), 0.0) for i in range(7)]
        # TODO: move_to_precise_home(openarm, arm_goal_abs)

        # [openarm_zero_mfr] 영점은 '지금 실제 위치'에 써진다. 목표(스톱+지령 이동량)로 보정한 뒤 확인하고 쓴다.
        goals = zero_goals()
        time.sleep(0.5)
        openarm.recv_all()
        q_now = [m.get_position() for m in arm.get_motors()]
        print('[CHECK] 보정 전 목표-실제 [deg]:', {f'j{i + 1}': round(np.rad2deg(goals[i] - q_now[i]), 2) for i in range(7)})
        refine_to_goals(openarm, arm, goals)

        # Safe disable before writing zeros
        openarm.disable_all()
        openarm.recv_all()
        q_off = [m.get_position() for m in arm.get_motors()]
        err_off = {f'j{i + 1}': round(np.rad2deg(goals[i] - q_off[i]), 2) for i in range(7)}
        print('[CHECK] 모터 끈 직후 목표-실제 [deg]:', err_off)
        bad = {k: v for k, v in err_off.items() if abs(v) > np.rad2deg(ZERO_TOL_RAD)}
        if bad:
            raise RuntimeError(f'모터 끈 뒤 자리 벗어남 {bad} (허용 {np.rad2deg(ZERO_TOL_RAD):.1f}°, 팔 받침 필요) — zero NOT written')
        openarm.set_zero_all()
        openarm.recv_all()
        print("wrote zero position to arm")
        time.sleep(0.2)
        openarm.recv_all()
        print("[AFTER] q after set_zero (disabled readback) [rad]:", [round(m.get_position(), 4) for m in arm.get_motors()])

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed → stopping safely")
    finally:
        openarm.disable_all()
        print("[INFO] Motors disabled, exiting safely.")


if __name__ == "__main__":
    main()

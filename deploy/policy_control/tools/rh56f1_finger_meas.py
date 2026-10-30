#!/usr/bin/env python3
"""RH56F1 손가락 실측 — 빈손 스윕(mimic 비율용) · 쥔 손 기록(src240 · shaker).

10.19 사용자: "1번 빈손 측정(urdf mimic 비율 맞추기) · 2번 src240 을 실제로 적절히 잡았을 때의 관절값과 측정되는 jointstate,
force 등(src200 은 길이만 다르고 잡는 자리 형상은 같다) · 3번 shaker 도 같은 것". 두 번째 마디는 링크로 첫 마디에 묶여
따로 움직이지 않는다 — 사진 없이 구동 축(_1 · 엄지 굽힘)의 실측 각 · 힘 · 전류 · 촉각만 기록한다(_2 는 링크 표로 계산).

  sweep   빈손. 축 하나씩 levels 로 단계 이동(다른 축은 그대로) → 도착 뒤 --sample-s 동안 평균. 축이 끝나면 그 축을 시작
          각으로 되돌린다. 도착 못 하거나 힘이 --sweep-force-g 를 넘으면(막힘 = 무엇에 닿음) 시작 각으로 되돌리고 멈춘다(rc 2).
  grip    사람이 물체를 손에 댄 뒤 닫는다. 엄지 회전은 --thumb1(없으면 그대로). 네 손가락 + 엄지 굽힘(--thumb-delay-s 뒤)을
          --close-rad-s 램프로 닫는 목표 쪽으로 — 목표는 실측보다 --lead(pd HAND_MAX_LEAD 0.2 rad)까지만 앞선다(실기 정책이
          pd 를 거쳐 쥘 때와 같은 미는 간극). 손 펌웨어 힘 멈춤(FORCESET, 드라이버 기본 600 g)도 그대로다. 힘이 --grip-force-g
          를 넘은 축은 그 각에 고정. 모든 축이 멈춘 뒤 --record-s 동안 기록하고 쥔 채 둔다(--release 면 연다 — 물체를 받칠 것).
  open    네 손가락 + 엄지 굽힘을 0 rad(편 손)로. 엄지 회전은 그대로.
  record  움직이지 않는다. 지금 손 상태(콘솔 경로 B · C · 정책으로 쥔 손)를 --record-s 동안 기록.

보내는 길(--via, 기본 auto):
  pd      ★10.19 사용자 "home 자세를 먼저" — 콘솔 home 뒤처럼 pd 가 팔 · 손을 붙들고 있으면 pd 를 거친다. /policy_control/
          joint_target 에 팔(pd 지금 세트포인트 그대로, hand_grasp 경로 B 와 같다) + 손 6 축을 CMD_HZ 로 계속 보내고(워치독),
          끝나면 그 팔 episode stop → pd 가 마지막 팔 · 손 목표를 붙든다(쥔 손은 쥔 채). pd 의 손 한계 여유 · 속도 · 앞섬 상한이 그대로 걸린다.
  direct  pd 가 없을 때(손 드라이버만) /hand_<side>/angle_set 에 바로 — 바꿀 축만, 나머지 -1. angle_set 에 다른 발행자가
          있으면 거부한다(둘이 번갈아 보내면 손가락이 떤다).
기록: logs/finger_meas/<날짜_시각>_<side>_<mode>[_<object>]/ — summary.json · samples.npz · levels.csv(sweep)

    ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/rh56f1_finger_meas.py sweep --side right            # 계획만
    ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/rh56f1_finger_meas.py sweep --side right --execute  # ★실기
    ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/rh56f1_finger_meas.py grip --side right --object src240 --execute
    ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/rh56f1_finger_meas.py record --side right --object shaker
종료 코드: 0 = 끝 · 1 = 입력 · 연결 문제(보내지 않음) · 2 = 막힘 · 3 = pd 이상(HOLD · estop · TRACKING 아님) 또는 다른 발행자.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from policy_control import rh56f1_map  # noqa: E402

SIM2REAL = HERE.parents[2]
OUT_ROOT = SIM2REAL / "logs" / "finger_meas"
FINGERS = ("index_1", "middle_1", "ring_1", "pinky_1")
CLOSING = (*FINGERS, "thumb_2")
#: 제작기(10.19): 정책이 쥐는 동안 _1 은 0.05~0.85 rad — 0 · 25 · 50 · 75 · 100 % 사이에 0.3~0.9 rad 점을 더 넣는다
FINGER_LEVELS = (0.0, 0.15, 0.30, 0.38, 0.45, 0.55, 0.65, 0.76, 0.90, 1.05, 1.15, 1.30, 1.45)
THUMB2_LEVELS = (0.0, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42)
DEFAULT_AXES = CLOSING
REACH_REG = 8                 # 도착 판정 [레지스터 0.1°] — 0.014 rad
STILL_REG = 3                 # 멈춤 판정: --still-s 동안 이 칸 안
HAND_MAX_LEAD = 0.2           # pd_backends.HAND_MAX_LEAD 와 같다(실기 정책 경로의 미는 간극)
HAND_LIMIT_MARGIN = 0.05      # pd_backends.HAND_LIMIT_MARGIN — pd 경로는 목표를 한계 안쪽 이만큼으로 물린다
#: 힘 상한 [g, 센서 위치]. 드라이버 pos_guard(10.14 사용자 1,300 g)가 지금 꺼져 있어 도구가 직접 본다.
SWEEP_FORCE_G = 800.0         # 빈손 끝까지 닫아도 ~300 g(09.30 스윕 sweep_right.csv 최대 314 g) — 넘으면 무엇에 닿은 것(막힘)
GRIP_FORCE_G = 1300.0         # 쥐기 — 넘으면 그 축을 지금 각도에 고정(더 닫지 않는다)
READ_TIMEOUT_S = 3.0
DISCOVERY_S = 1.5             # 다른 발행자(pd)를 찾기 전 기다림
RESEND_S = 0.5                # direct: 같은 목표를 다시 보내는 주기
CMD_HZ = 50.0                 # pd: joint_target 주기(pd 워치독 0.25 s 보다 촘촘히)
SLOW_SEND_S = 0.1             # pd: 이보다 긴 발행 간격은 기록(워치독 0.25 s 원인 가르기)
SAMPLE_MAX_HZ = 100.0
PD_STALE_S = 1.0


class MeasError(ValueError):
    pass


class PdFault(MeasError):
    pass


# ── 순수 ───────────────────────────────────────────────────────────────
def parse_levels(text: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if not text:
        return tuple(default)
    try:
        vals = tuple(float(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise MeasError(f"levels {text!r} — 쉼표로 나눈 rad") from exc
    if not vals:
        raise MeasError("levels 가 비었다")
    return vals


def axis_of(hmap: rh56f1_map.HandMap, side: str, name: str) -> rh56f1_map.Axis:
    if name not in hmap.joint_order:
        raise MeasError(f"축 {name!r} — {list(hmap.joint_order)} 중")
    a = hmap.axes_of(side)[hmap.joint_order.index(name)]
    if not a.ok(side):
        raise MeasError(f"{side} {name}: 변환표에서 방향 확인 안 된 축")
    return a


def axis_range(hmap: rh56f1_map.HandMap, name: str) -> tuple[float, float]:
    lo, hi = sorted(hmap.axes[hmap.joint_order.index(name)].rad)
    return lo, hi


def check_levels(hmap: rh56f1_map.HandMap, side: str, name: str, levels: Sequence[float]) -> None:
    lo, hi = axis_range(hmap, name)
    bad = [v for v in levels if not lo <= v <= hi]
    if bad:
        raise MeasError(f"{name} levels {bad} — 변환표 범위 {lo:.3f}~{hi:.3f} rad 밖")


def pd_safe(hmap: rh56f1_map.HandMap, name: str, q: float, margin: float = HAND_LIMIT_MARGIN) -> float:
    """pd hand_safe_target 와 같은 물림 — 한계 안쪽 margin. 손 한계 = 변환표 범위(URDF 와 같다)."""
    lo, hi = axis_range(hmap, name)
    return float(np.clip(q, lo + margin, hi - margin))


def lead_clamped(q_ramp: float, q_meas: float, lead: float) -> float:
    """닫는 쪽(+) 목표는 실측보다 lead 까지만 앞선다 — pd hand_lead_clamp 와 같은 미는 간극. 여는 쪽은 그대로."""
    return min(q_ramp, q_meas + lead)


def still(history: Sequence[tuple[float, int]], now: float, window_s: float, tol: int = STILL_REG) -> bool:
    """(t, 레지스터) 이력에서 최근 window_s 동안 tol 칸 안에 머물렀나. 이력이 window 보다 짧으면 False."""
    if not history or now - history[0][0] < window_s:
        return False
    recent = [r for t, r in history if now - t <= window_s]
    return bool(recent) and max(recent) - min(recent) <= tol


def stats(rows: np.ndarray) -> dict:
    """(N, k) 표본 → 열마다 평균 · 표준편차 · 최소 · 최대."""
    if rows.size == 0:
        return {}
    return {"mean": np.round(rows.mean(0), 3).tolist(), "std": np.round(rows.std(0), 3).tolist(),
            "min": np.round(rows.min(0), 3).tolist(), "max": np.round(rows.max(0), 3).tolist(), "n": int(rows.shape[0])}


def axis_view(hmap: rh56f1_map.HandMap, side: str, reg_mean: Sequence[float], force: Sequence[float] | None,
              current: Sequence[float] | None) -> dict:
    """슬롯 순 평균 → 축 이름별 {reg, rad, force_g, current}. rad 는 손 · 축별 보정(변환표 calibration)."""
    q = hmap.to_rad(reg_mean, side=side)
    out = {}
    for a, v in zip(hmap.axes_of(side), q):
        out[a.name] = {"reg": round(float(reg_mean[a.slot]), 1), "rad": round(float(v), 4),
                       "force_g": None if force is None else round(float(force[a.slot]), 1),
                       "current": None if current is None else round(float(current[a.slot]), 1)}
    return out


def pd_fault(pd: Mapping, now: float) -> str | None:
    """pd 상태(JSON)로 보낼 수 없는 이유 — hand_grasp(경로 B)와 같은 조건."""
    if not pd or now - float(pd.get("_t", 0.0)) > PD_STALE_S:
        return "pd status 없음"
    if pd.get("estop") or pd.get("phase") != "TRACKING":
        return f"pd {pd.get('phase')} estop={pd.get('estop')} {pd.get('reasons', '')}"
    return None


def choose_route(via: str, pd_alive: bool, other_pubs: int) -> tuple[str | None, str]:
    """(길, 이유). auto = pd 가 살아 있으면 pd, 아니면 다른 발행자가 없을 때만 direct."""
    if via == "pd" or (via == "auto" and pd_alive):
        return ("pd", "") if pd_alive else (None, "pd status 가 안 온다 — pd 를 띄우고 engage · home 뒤에")
    if other_pubs:
        return None, f"angle_set 에 다른 발행자 {other_pubs} 개(pd 가 status 없이?) — 직접 보내지 않는다"
    return "direct", ""


def out_dir(root: Path, side: str, mode: str, obj: str | None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = root / f"{stamp}_{side}_{mode}{'_' + obj if obj else ''}"
    out, k = base, 1
    while out.exists() and k < 100:
        out, k = Path(f"{base}_{k}"), k + 1
    return out


# ── ROS ───────────────────────────────────────────────────────────────
@dataclass
class Buf:
    """한 토픽의 (t, 값) — 기록 중일 때만 쌓는다."""
    t: list = field(default_factory=list)
    v: list = field(default_factory=list)

    def add(self, t: float, v: Sequence[float]) -> None:
        if self.t and t - self.t[-1] < 1.0 / SAMPLE_MAX_HZ:
            return
        self.t.append(t)
        self.v.append(list(v))

    def since(self, t0: float) -> np.ndarray:
        rows = [v for t, v in zip(self.t, self.v) if t >= t0]
        return np.asarray(rows, float) if rows else np.zeros((0, 0))


class Hand:
    """손 드라이버 상태 · pd 상태 구독, 기록 버퍼. 보내기는 route(DirectRoute · PdRoute)가 — spin 마다 route.tick()."""

    def __init__(self, side: str) -> None:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from rh56f1_interfaces.msg import GetAngleAct1, GetCurrentAct1, GetForceAct1, TouchData1
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self.rclpy, self.side = rclpy, side
        rclpy.init()
        self.node = rclpy.create_node(f"rh56f1_finger_meas_{side}")
        self.ns = f"/hand_{side}"
        self.latest: dict = {}
        self.bufs = {k: Buf() for k in ("angle", "force", "current", "touch", "joint_states", "arm")}
        self.recording = False
        self.pd: dict = {}
        self.route = None
        self.js_names: list[str] = []
        self.arm_src = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        sub = self.node.create_subscription

        def keep(key: str, values) -> None:
            now = time.monotonic()
            self.latest[key] = (now, list(values))
            if self.recording:
                self.bufs[key].add(now, values)

        def on_js(m) -> None:
            self.js_names = list(m.name)
            keep("joint_states", m.position)

        def on_arm(m) -> None:
            pos = dict(zip(m.name, m.position))
            if all(n in pos for n in self.arm_src):
                keep("arm", [pos[n] for n in self.arm_src])

        def on_pd(m) -> None:
            try:
                self.pd = {**json.loads(m.data), "_t": time.monotonic()}
            except ValueError:
                pass
        sub(GetAngleAct1, f"{self.ns}/angle_actual", lambda m: keep("angle", m.joint_values), 10)
        sub(GetForceAct1, f"{self.ns}/force_actual", lambda m: keep("force", m.joint_values), 10)
        sub(GetCurrentAct1, f"{self.ns}/current_actual", lambda m: keep("current", m.joint_values), 10)
        # 열 = 법선 5 · 접선 5 · 손바닥 9 · 접선 방향 5(도, 접촉 없으면 65535) · 근접 5 — 앞 19 열 순서는 그대로 둔다
        sub(TouchData1, f"{self.ns}/touch_data",
            lambda m: keep("touch", [*m.finger_forces, *m.finger_tangentials, *m.palm_data,
                                     *m.finger_angles, *m.finger_proximity]), 10)
        sub(JointState, f"{self.ns}/joint_states", on_js, qos_profile_sensor_data)
        sub(JointState, "/joint_states", on_arm, qos_profile_sensor_data)
        sub(String, f"/policy_control/status/pd_{side}", on_pd, 10)

    def spin(self, sec: float, until=lambda: False) -> None:
        t_end = time.monotonic() + sec
        while True:
            if self.route is not None:
                self.route.tick()
            if time.monotonic() >= t_end or until():
                return
            self.rclpy.spin_once(self.node, timeout_sec=0.005)

    def _fresh(self, key: str) -> list | None:
        got = self.latest.get(key)
        return None if got is None or time.monotonic() - got[0] > 0.5 else got[1]

    def reg(self) -> list[int] | None:
        r = self._fresh("angle")
        return None if r is None else [int(v) for v in r]

    def force(self) -> list[float] | None:
        f = self._fresh("force")
        return None if f is None else [float(v) for v in f]

    def arm_q(self) -> np.ndarray | None:
        q = self._fresh("arm")
        return None if q is None else np.asarray(q, float)

    def angle_set_publishers(self) -> int:
        return self.node.count_publishers(f"{self.ns}/angle_set")

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


class DirectRoute:
    """/hand_<side>/angle_set 에 바로 — 정한 축만(나머지 -1), RESEND_S 마다 다시."""
    name = "direct"

    def __init__(self, hand: Hand, hmap: rh56f1_map.HandMap) -> None:
        from rh56f1_interfaces.msg import SetAngle1
        self.hand, self.Msg = hand, SetAngle1
        self.axes = {a.name: a for a in hmap.axes_of(hand.side)}
        self.q: dict[str, float] = {}
        self.last = 0.0
        self.pub = hand.node.create_publisher(SetAngle1, f"{hand.ns}/angle_set", 10)

    def connect(self) -> str:
        self.hand.spin(READ_TIMEOUT_S, lambda: self.pub.get_subscription_count() >= 1)
        return "" if self.pub.get_subscription_count() >= 1 else f"{self.hand.ns}/angle_set 을 구독하는 드라이버가 안 보인다"

    def effective(self, name: str, q: float) -> float:
        return float(q)

    def target_reg(self, name: str) -> int:
        return self.axes[name].to_reg(self.q[name])

    def set(self, q: Mapping[str, float]) -> None:
        self.q.update({k: float(v) for k, v in q.items()})
        self._send()

    def tick(self) -> None:
        if self.q and time.monotonic() - self.last >= RESEND_S:
            self._send()

    def _send(self) -> None:
        cmd = [rh56f1_map.LEAVE] * rh56f1_map.N
        for n, v in self.q.items():
            cmd[self.axes[n].slot] = self.axes[n].to_reg(v)
        m = self.Msg()
        m.hand_id = 0
        m.joint_values = cmd
        self.pub.publish(m)
        self.last = time.monotonic()

    def finish(self) -> None:
        pass


class PdRoute:
    """pd 를 거쳐 — joint_target 에 팔(pd 세트포인트) + 손 6 축, CMD_HZ 로. 끝에 그 팔 episode stop → pd 가 붙든다."""
    name = "pd"

    def __init__(self, hand: Hand, hmap: rh56f1_map.HandMap, q_arm: np.ndarray, q_hand0: Mapping[str, float]) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from replay_to_pd import EPISODE_TOPIC_SIDE, TOPIC, replay_episode_id

        self.hand, self.hmap, self.JS, self.String = hand, hmap, JointState, String
        self.axes = {a.name: a for a in hmap.axes_of(hand.side)}
        p = hand.side[0]
        self.arm_names = [f"{p}_aj_{i}" for i in range(1, 8)]
        self.hand_names = hmap.names(hand.side)
        self.q_arm = np.asarray(q_arm, float)
        self.q = {n: pd_safe(hmap, n, float(v)) for n, v in q_hand0.items()}
        self.ep_id, self.k, self.last = replay_episode_id(), 0, 0.0
        self.t0, self.max_gap, self.slow = time.monotonic(), 0.0, []
        self.pub = hand.node.create_publisher(JointState, TOPIC, 10)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ep_pub = hand.node.create_publisher(String, EPISODE_TOPIC_SIDE.format(side=hand.side), latched)
        self.finished = False

    def connect(self) -> str:
        self.hand.spin(READ_TIMEOUT_S, lambda: self.pub.get_subscription_count() >= 1)
        return "" if self.pub.get_subscription_count() >= 1 else "pd 가 /policy_control/joint_target 을 구독하지 않는다"

    def effective(self, name: str, q: float) -> float:
        return pd_safe(self.hmap, name, q)

    def target_reg(self, name: str) -> int:
        return self.axes[name].to_reg(self.q[name])

    def set(self, q: Mapping[str, float]) -> None:
        self.q.update({k: self.effective(k, float(v)) for k, v in q.items()})
        self._send()

    def tick(self) -> None:
        if self.finished:
            return
        why = pd_fault(self.hand.pd, time.monotonic())
        if why:
            raise PdFault(why)
        if time.monotonic() - self.last >= 1.0 / CMD_HZ:
            self._send()

    def _send(self) -> None:
        m = self.JS()
        m.header.stamp = self.hand.node.get_clock().now().to_msg()
        m.header.frame_id = f"{self.ep_id}:{self.k}"
        m.name = self.arm_names + self.hand_names
        m.position = [float(v) for v in self.q_arm] + [float(self.q[n]) for n in self.hmap.joint_order]
        m.velocity = [0.0] * len(m.name)
        m.effort = [0.0] * len(m.name)
        self.pub.publish(m)
        self.k += 1
        now = time.monotonic()
        if self.last:                                     # ★10.19 실기: pd 워치독(0.25 s)이 0.254 s 두절로 걸렸다 —
            gap = now - self.last                         #   도구가 늦게 보낸 것인지 전달이 늦은 것인지 가르려고 보낸 간격을 남긴다
            self.max_gap = max(self.max_gap, gap)
            if gap > SLOW_SEND_S:
                self.slow = (self.slow + [(round(now - self.t0, 3), round(gap, 3))])[-20:]
        self.last = now

    def send_stats(self) -> dict:
        return {"sends": self.k, "max_gap_s": round(self.max_gap, 4), "slow": self.slow}

    def finish(self) -> None:
        """episode stop → pd 가 마지막 팔 · 손 목표를 내부 목표로 붙들 때까지(STOP_WAIT_S) 같은 목표를 보낸다."""
        from replay_to_pd import STOP_WAIT_S
        if self.finished:
            return
        self.finished = True
        self.ep_pub.publish(self.String(data=json.dumps({"episode": self.ep_id, "event": "stop", "object_anchor": None,
                                                          "home_q": {}, "reasons": ["rh56f1_finger_meas done"],
                                                          "t_ns": time.time_ns()})))
        t1 = time.monotonic()
        while time.monotonic() - t1 < STOP_WAIT_S and self.hand.pd.get("target") != "internal":
            self._send()
            self.hand.rclpy.spin_once(self.hand.node, timeout_sec=1.0 / CMD_HZ)


def open_route(hand: Hand, hmap: rh56f1_map.HandMap, via: str):
    """(route, rc, 이유). 보내기 전 검사 — 실패면 아무것도 보내지 않았다."""
    hand.spin(DISCOVERY_S)
    pd_alive = pd_fault(hand.pd, time.monotonic()) != "pd status 없음"
    kind, why = choose_route(via, pd_alive, hand.angle_set_publishers())
    if kind is None:
        return None, 3, why
    if kind == "pd":
        why = pd_fault(hand.pd, time.monotonic())
        if why:
            return None, 3, f"시작 전 {why}"
        q_arm = hand.arm_q()
        if q_arm is None:
            return None, 1, f"/joint_states 에 {hand.side} 팔이 안 온다"
        sys.path.insert(0, str(HERE))
        from hand_grasp import arm_hold_target
        q_set, why = arm_hold_target(hand.pd, hand.side, q_arm)
        if q_set is None:
            return None, 3, why
        q_hand = dict(zip(hmap.joint_order, hmap.to_rad(hand.reg(), side=hand.side).tolist()))
        route = PdRoute(hand, hmap, q_set, q_hand)
    else:
        route = DirectRoute(hand, hmap)
    why = route.connect()
    if why:
        return None, 1, why
    hand.route = route
    return route, 0, ""


def sample(hand: Hand, sec: float) -> dict:
    """sec 동안 기록 → 토픽마다 stats. 원표본은 hand.bufs 에 남는다."""
    t0 = time.monotonic()
    hand.recording = True
    hand.spin(sec)
    hand.recording = False
    return {k: stats(b.since(t0)) for k, b in hand.bufs.items()}


def go_axis(hand: Hand, route, a: rh56f1_map.Axis, q: float, timeout_s: float,
            max_force_g: float) -> tuple[bool, float, list[int] | None]:
    """한 축을 q 로 — (도착, 걸린 시간, 마지막 레지스터). 도착 못 하거나 힘이 max_force_g 를 넘으면 막힘(그 자리에 세운다)."""
    route.set({a.name: q})
    target = route.target_reg(a.name)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        hand.spin(0.02)
        r, f = hand.reg(), hand.force()
        if r is not None and abs(r[a.slot] - target) <= REACH_REG:
            return True, time.monotonic() - t0, r
        if r is not None and f is not None and f[a.slot] > max_force_g:
            route.set({a.name: a.to_rad(r[a.slot])})
            print(f"[{a.name}] 힘 {f[a.slot]:.0f} g > {max_force_g:.0f} — 그 자리(reg {r[a.slot]})에 세운다", flush=True)
            return False, time.monotonic() - t0, r
    return False, time.monotonic() - t0, hand.reg()


def run_sweep(hand: Hand, route, hmap, axes: Sequence[str], levels: dict, a: argparse.Namespace,
              out: Path) -> tuple[int, dict]:
    side = hand.side
    start = hand.reg()
    q_start = dict(zip(hmap.joint_order, hmap.to_rad(start, side=side).tolist()))
    rows, rc = [], 0
    try:
        for name in axes:
            ax = axis_of(hmap, side, name)
            for q in levels[name]:
                ok, took, _ = go_axis(hand, route, ax, q, a.arrive_s, a.sweep_force_g)
                hand.spin(a.settle_s)
                st = sample(hand, a.sample_s)
                reg_m = (st.get("angle") or {}).get("mean")
                f_m, c_m = (st.get("force") or {}).get("mean"), (st.get("current") or {}).get("mean")
                row = {"axis": name, "level_rad": q, "sent_rad": round(route.effective(name, q), 4),
                       "reg_cmd": route.target_reg(name), "arrived": ok, "t_arrive_s": round(took, 2)}
                if reg_m:
                    v = axis_view(hmap, side, reg_m, f_m, c_m)[name]
                    row.update(reg_meas=v["reg"], reg_std=st["angle"]["std"][ax.slot], rad_meas=v["rad"],
                               force_g=v["force_g"], force_g_max=((st.get("force") or {}).get("max") or [None] * 6)[ax.slot],
                               current=v["current"])
                rows.append(row)
                print(f"[sweep] {side} {name} {q:.2f} rad → reg {row.get('reg_meas')} ({row.get('rad_meas')} rad) · "
                      f"힘 {row.get('force_g')} g · {'도착' if ok else '★막힘'} {took:.1f} s", flush=True)
                if not ok:
                    rc = 2
                    break
            go_axis(hand, route, ax, q_start[name], a.arrive_s, a.sweep_force_g)
            if rc:
                print(f"[sweep] {name} 막힘 — 시작 각 {q_start[name]:.3f} rad 로 되돌리고 멈춘다", flush=True)
                break
    finally:
        if rows:
            with (out / "levels.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
    return rc, {"start_reg": start, "levels": rows}


def run_grip(hand: Hand, route, hmap, a: argparse.Namespace) -> tuple[int, dict]:
    side = hand.side
    start = hand.reg()
    axes = [axis_of(hmap, side, n) for n in CLOSING]
    if a.thumb1 is not None:
        ok, _, _ = go_axis(hand, route, axis_of(hmap, side, "thumb_1"), a.thumb1, a.arrive_s, a.grip_force_g)
        if not ok:
            return 2, {"why": f"엄지 회전 {a.thumb1} rad 에 못 감(막힘)", "start_reg": start}
    goal = {"thumb_2": a.thumb2_to, **{n: a.close_to for n in FINGERS}}
    q0 = dict(zip(hmap.joint_order, hmap.to_rad(hand.reg(), side=side).tolist()))
    q_ramp = {ax.name: q0[ax.name] for ax in axes}
    hand.recording = True                                 # ★10.19 사용자: 닫는 과정(어느 마디가 먼저 닿나)도 데이터 — 닫기부터 기록
    hist: dict[str, list] = {ax.name: [] for ax in axes}
    frozen: dict[str, float] = {}                         # 힘 상한을 넘은 축 → 고정 각
    q_cmd: dict[str, float] = {}
    t0, dt = time.monotonic(), 1.0 / CMD_HZ
    while time.monotonic() - t0 < a.close_max_s:
        now = time.monotonic()
        r = hand.reg()
        if r is None:
            return 1, {"why": "angle_actual 끊김", "start_reg": start}
        q_meas = dict(zip(hmap.joint_order, hmap.to_rad(r, side=side).tolist()))
        f = hand.force() or [0.0] * rh56f1_map.N
        for ax in axes:
            n, qm = ax.name, q_meas[ax.name]
            if n not in frozen and f[ax.slot] > a.grip_force_g:
                frozen[n] = qm
                print(f"[grip] {n} 힘 {f[ax.slot]:.0f} g > {a.grip_force_g:.0f} — {qm:.3f} rad 에 고정", flush=True)
            if n in frozen:
                q_ramp[n], q_cmd[n] = goal[n], frozen[n]
                continue
            if n == "thumb_2" and now - t0 < a.thumb_delay_s:
                q_cmd[n] = q_ramp[n]
                continue
            q_ramp[n] = min(goal[n], q_ramp[n] + a.close_rad_s * dt)
            q_cmd[n] = lead_clamped(q_ramp[n], qm, a.lead)
            hist[n] = [(t, v) for t, v in hist[n] if now - t <= a.still_s + 0.1] + [(now, r[ax.slot])]
        route.set(q_cmd)
        ramp_done = all(q_ramp[n] >= goal[n] - 1e-9 for n in goal)
        if ramp_done and all(still(hist[n], now, a.still_s) for n in hist if hist[n]):
            break
        hand.spin(dt)
    sent = {n: round(route.effective(n, v), 4) for n, v in q_cmd.items()}
    print(f"[grip] {side} 닫기 끝 {time.monotonic() - t0:.1f} s · 명령 {sent} — {a.record_s:g} s 기록", flush=True)
    t_rec = time.monotonic()
    hand.spin(a.record_s)                                 # 같은 목표를 route 가 계속 보낸다
    hand.recording = False
    st = {k: stats(b.since(t_rec)) for k, b in hand.bufs.items()}   # 평균은 쥔 뒤 구간만
    res = {"start_reg": start, "goal_rad": goal, "cmd_rad": sent, "frozen": frozen, "stats": st,
           "close_t0": t0, "record_t0": t_rec}            # samples.npz 의 *_t 와 같은 시계(time.monotonic)
    if a.release:
        route.set({n: 0.0 for n in CLOSING})
        hand.spin(a.arrive_s)
        print("[grip] 열었다(--release)", flush=True)
    return 0, res


def summarize(hmap, side: str, st: dict) -> dict:
    ang, frc, cur = (st.get(k) or {} for k in ("angle", "force", "current"))
    view = axis_view(hmap, side, ang["mean"], frc.get("mean"), cur.get("mean")) if ang else {}
    touch = st.get("touch") or {}
    out = {"axes": view, "mimic_urdf": hmap.with_mimic([v["rad"] for v in view.values()]) if view else {}}
    if touch:
        m = touch["mean"]
        # 접선 방향(19:24)은 65535(무접촉)가 섞이면 평균이 뜻이 없어 요약에서 뺀다 — samples.npz touch_v 에 원본이 있다
        out["touch_vendor_order"] = {"order": "pinky,ring,middle,index,thumb", "normal": m[0:5], "tangential": m[5:10],
                                     "palm": m[10:19], "proximity": m[24:29]}
    arm = st.get("arm") or {}
    if arm:
        out["arm_q"] = arm["mean"]
    return out


def _args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=("sweep", "grip", "open", "record"))
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--via", choices=("auto", "pd", "direct"), default="auto")
    ap.add_argument("--object", default=None, help="grip · record 기록 이름(src240 · shaker …)")
    ap.add_argument("--axes", default=",".join(DEFAULT_AXES), help="sweep 축 순서")
    ap.add_argument("--levels", default=None, help=f"네 손가락 단계 rad (기본 {FINGER_LEVELS})")
    ap.add_argument("--thumb-levels", default=None, help=f"엄지 굽힘 단계 rad (기본 {THUMB2_LEVELS})")
    ap.add_argument("--arrive-s", type=float, default=3.0)
    ap.add_argument("--settle-s", type=float, default=0.4)
    ap.add_argument("--sample-s", type=float, default=1.0)
    ap.add_argument("--thumb1", type=float, default=None, help="grip: 닫기 전 엄지 회전 rad(없으면 그대로)")
    ap.add_argument("--close-to", type=float, default=1.45, help="grip: 네 손가락 닫는 목표 rad")
    ap.add_argument("--thumb2-to", type=float, default=0.42, help="grip: 엄지 굽힘 닫는 목표 rad")
    ap.add_argument("--close-rad-s", type=float, default=0.6, help="grip 램프 — 경로 B · C 와 같다")
    ap.add_argument("--thumb-delay-s", type=float, default=0.3)
    ap.add_argument("--lead", type=float, default=HAND_MAX_LEAD, help="grip: 실측보다 앞서는 상한 rad(pd 와 같다)")
    ap.add_argument("--close-max-s", type=float, default=6.0)
    ap.add_argument("--still-s", type=float, default=0.5)
    ap.add_argument("--record-s", type=float, default=5.0)
    ap.add_argument("--release", action="store_true", help="grip: 기록 뒤 연다(물체를 받칠 것)")
    ap.add_argument("--sweep-force-g", type=float, default=SWEEP_FORCE_G, help="sweep: 넘으면 막힘")
    ap.add_argument("--grip-force-g", type=float, default=GRIP_FORCE_G, help="grip: 넘으면 그 축 고정")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--execute", action="store_true", help="★실기에 보낸다(sweep · grip · open). record 는 필요 없다")
    return ap.parse_args(argv)


def _validate(hmap, a: argparse.Namespace) -> tuple[list[str], dict]:
    axes = [x.strip() for x in a.axes.split(",") if x.strip()]
    levels = {}
    for n in axes:
        axis_of(hmap, a.side, n)
        levels[n] = (parse_levels(a.thumb_levels, THUMB2_LEVELS) if n.startswith("thumb")
                     else parse_levels(a.levels, FINGER_LEVELS))
        check_levels(hmap, a.side, n, levels[n])
    if not 0.0 < a.close_rad_s <= 1.5 or not 0.0 < a.lead <= HAND_MAX_LEAD:
        raise MeasError(f"--close-rad-s 0~1.5 · --lead 0~{HAND_MAX_LEAD}")
    if not 0.0 < a.sweep_force_g <= GRIP_FORCE_G or not 0.0 < a.grip_force_g <= GRIP_FORCE_G:
        raise MeasError(f"힘 상한은 0~{GRIP_FORCE_G:g} g")
    check_levels(hmap, a.side, "index_1", [a.close_to])
    check_levels(hmap, a.side, "thumb_2", [a.thumb2_to])
    if a.thumb1 is not None:
        check_levels(hmap, a.side, "thumb_1", [a.thumb1])
    return axes, levels


def _plan(a: argparse.Namespace, axes: list[str], levels: dict) -> None:
    if a.mode == "sweep":
        n = sum(len(levels[x]) for x in axes)
        print(f"[sweep] {a.side} · 길 {a.via} · 축 {axes} · {n} 단계 · 최악 {n * (a.arrive_s + a.settle_s + a.sample_s):.0f} s")
        for x in axes:
            print(f"  {x}: {levels[x]}")
    elif a.mode == "grip":
        print(f"[grip] {a.side} {a.object} · 길 {a.via} · 엄지 회전 {a.thumb1 if a.thumb1 is not None else '그대로'} · "
              f"손가락 → {a.close_to} · 엄지 굽힘 → {a.thumb2_to}({a.thumb_delay_s} s 뒤) · {a.close_rad_s} rad/s · "
              f"앞섬 {a.lead} · 기록 {a.record_s} s{' · 기록 뒤 연다' if a.release else ' · 쥔 채 둔다'}")
    elif a.mode == "open":
        print(f"[open] {a.side} · 길 {a.via} · 네 손가락 + 엄지 굽힘 → 0 rad(엄지 회전 그대로)")


def main(argv: list[str] | None = None) -> int:
    a = _args(argv)
    hmap = rh56f1_map.load()
    domain = str(os.environ.get("ROS_DOMAIN_ID", "")).strip()
    if domain in ("", "0"):
        print(f"✗ ROS_DOMAIN_ID={domain or 'unset'} — 실기 config/ros_domain.env · fake 97/99 를 정할 것")
        return 1
    try:
        axes, levels = _validate(hmap, a)
    except MeasError as exc:
        print(f"✗ {exc}")
        return 1
    moves = a.mode != "record"
    _plan(a, axes, levels)
    if moves and not a.execute:
        print("  (계획만 — --execute 없이는 보내지 않는다)")
        return 0
    hand = Hand(a.side)
    route = None
    try:
        hand.spin(READ_TIMEOUT_S, lambda: hand.reg() is not None)
        if hand.reg() is None:
            print(f"✗ {hand.ns}/angle_actual 이 {READ_TIMEOUT_S} s 안에 안 온다 — 손 드라이버를 먼저")
            return 1
        if moves:
            route, rc, why = open_route(hand, hmap, a.via)
            if route is None:
                print(f"✗ {why} — 아무것도 보내지 않았다")
                return rc
            print(f"[{a.mode}] 길 = {route.name}", flush=True)
        out = out_dir(a.out_root, a.side, a.mode, a.object)
        out.mkdir(parents=True, exist_ok=True)
        res: dict = {"side": a.side, "mode": a.mode, "object": a.object, "at": time.strftime("%F %T"), "domain": domain,
                     "route": None if route is None else route.name,
                     "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
        rc = 0
        try:
            if a.mode == "sweep":
                rc, body = run_sweep(hand, route, hmap, axes, levels, a, out)
                res.update(body)
            elif a.mode == "open":
                route.set({n: 0.0 for n in CLOSING})
                hand.spin(a.arrive_s)
                res["end_reg"] = hand.reg()
            else:
                if a.mode == "grip":
                    rc, body = run_grip(hand, route, hmap, a)
                    st = body.pop("stats", {})
                    res.update(body)
                else:
                    st = sample(hand, a.record_s)
                res["stats"] = st
                if st.get("angle"):
                    res["summary"] = summarize(hmap, a.side, st)
        except PdFault as exc:
            rc, res["why"] = 3, f"pd 이상 — {exc}"
            print(f"✗ {res['why']} — 멈춘다(pd 가 붙든다)", flush=True)
        finally:
            if route is not None:
                route.finish()
                if hasattr(route, "send_stats"):
                    res["pd_send"] = route.send_stats()
                    if rc == 3:
                        print(f"  도구 발행 간격: {res['pd_send']}", flush=True)
        res["rc"], res["joint_state_names"], res["pd_target_end"] = rc, hand.js_names, hand.pd.get("target")
        np.savez_compressed(out / "samples.npz", **{f"{k}_t": np.asarray(b.t) for k, b in hand.bufs.items()},
                            **{f"{k}_v": np.asarray(b.v, float) for k, b in hand.bufs.items()})
        (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        if res.get("summary"):
            for name, v in res["summary"]["axes"].items():
                print(f"  {name:9s} {v['rad']:+.3f} rad (reg {v['reg']}) · 힘 {v['force_g']} g · 전류 {v['current']}")
        print(f"[{a.mode}] rc {rc} · {out}", flush=True)
        return rc
    finally:
        hand.close()


if __name__ == "__main__":
    raise SystemExit(main())

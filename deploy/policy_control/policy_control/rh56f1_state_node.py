"""RH56F1 손 상태 → rad JointState · 손끝 힘. 구독 · 발행만(명령을 내지 않는다).

09.29 사용자: rh56f1 제어 연결. 벤더 드라이버(robot_control inspire_rh56f1)는 각도 레지스터(0.1°, 슬롯 순 새끼부터)만 낸다.
pd · 정책 · 콘솔은 rad JointState 를 읽는다 — 이 노드가 변환표(config/rh56f1_hand_map.yaml)로 바꾼다.

  구독  /hand_<side>/angle_actual (rh56f1_interfaces/GetAngleAct1)  ·  /hand_<side>/touch_data (TouchData1)
        /hand_<side>/force_actual (GetForceAct1 — 슬롯 순 새끼 · 약지 · 중지 · 검지 · 엄지 굽힘 · 엄지 회전, g)
  발행  /hand_<side>/joint_states (sensor_msgs/JointState, canonical 이름, 구동 6 + 종속 6(mimic), rad · rad/s)
        /hand_<side>/tip_forces   (std_msgs/Float64MultiArray, 5, sim 순 엄지 → 새끼, N — √(법선² + 접선²), 단위 실측 전)
        /hand_<side>/joint_forces (std_msgs/Float64MultiArray, 6, 엄지 굽힘 · 엄지 회전 · 검지 · 중지 · 약지 · 새끼, g — 센서 위치,
                                   force_actual 그대로. 10.04 쥔 판정: 엄지를 첫마디로 감싸 쥐면 손끝 촉각엔 잘 안 잡힌다)
        ★11.04 손끝 소프트웨어 영점(tip_forces 원값은 그대로 — 학습 정책 입력 계약):
        /hand_<side>/tip_forces_zeroed (Float64MultiArray, 5, tip_forces 와 같은 순 · 단위 — 법선 · 접선 원시에서 각각 bias 를 빼고 합친 크기)
        /hand_<side>/tip_bias          (Float64MultiArray, 5, transient_local — 마지막으로 받아들인 bias 의 크기, N)
        /hand_<side>/tip_zero_status   (std_msgs/String JSON, transient_local — 시도마다: 시각 · 받아들임 · 거부 이유 · 원시 bias 법선/접선
                                        · 관절 힘 bias(joint_accepted · joint_reasons · joint_bias_g · joint_jump_g))
        ★11.04 관절 힘 소프트웨어 영점(joint_forces 원값은 그대로. 실기 오른손 엄지 굽힘이 포화 하중 뒤 편 손에서 +205~213 g 로 떠
        쥠 판정 엄지 받침 B(80 g)가 늘 참이었다):
        /hand_<side>/joint_forces_zeroed (Float64MultiArray, 6, joint_forces 와 같은 순 · 단위 g — 원값 − bias, 부호 그대로.
                                          관절 힘 bias 를 처음 받아들이기 전에는 내지 않는다 → 정책 노드는 None = 결손)
  서비스 /hand_<side>/tip_zero (std_srvs/Trigger) — 직전 0.5 s 창의 촉각 · 관절 힘 평균을 bias 로. 손이 안 펴졌거나 · 움직이거나 ·
        값이 크거나 튀면 거부하고 이전 bias 를 둔다(관절 힘: 칸별 sd > 5 g · |평균| > 400 g). 관절 힘이 직전 bias 에서 30 g 넘게
        옮겨 가면 받되 경고(joint_jump_g). 손끝 · 관절 힘은 따로 받아들인다. 시도마다 logs/rh56f1_tip_zero/<날짜>.jsonl 에 한 줄.
        /hand_<side>/tip_zero_force (Trigger) — 같은 일인데 관절 힘 |평균| 상한만 건너뛴다(사용자가 강제로 잡을 때). 편 손 · 정지 · sd 는 그대로.

    python3 deploy/policy_control/policy_control/rh56f1_state_node.py --side right
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):              # 파일 경로로 띄울 때(미션 명령) — colcon 진입점을 새로 만들지 않는다
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import rh56f1_map  # noqa: E402

#: 속도 = 위치 차분의 지수 평균. ★10.03 시정수로 정한다 — 발행 주기(RS485 50 Hz · EtherCAT 200 Hz)가 바뀌어도 같은 필터.
#  옛 값 α 0.3 @ 50 Hz 와 같은 시정수(−dt/ln(1−α) ≈ 0.056 s).
VEL_TAU_S = 0.056


#: force_actual 슬롯(새끼 · 약지 · 중지 · 검지 · 엄지 굽힘 · 엄지 회전) → joint_forces 순(엄지 굽힘 · 엄지 회전 · 검지 · 중지 · 약지 · 새끼)
JOINT_FORCE_ORDER = (4, 5, 3, 2, 1, 0)


def joint_forces_out(slots: Sequence[float]) -> list[float]:
    """벤더 슬롯 순 관절 힘(g) → 엄지 먼저 순. 순수."""
    if len(slots) != 6:
        raise ValueError(f"force_actual 은 6 칸이어야 한다(got {len(slots)})")
    return [float(slots[i]) for i in JOINT_FORCE_ORDER]


#: ★11.04 사용자 "데이터 모을 때 tip 센서도 home 이나 rehome 하면 bias 를 잡아 영점을 맞추고 기록". 하드웨어 영점(--force-calibrate)은 쓰지 않는다.
ZERO_WINDOW_S = 0.5          # 부르기 직전 이 창의 평균
ZERO_MIN_SAMPLES = 10        # 창 안 촉각 샘플 하한(EtherCAT 250 Hz ≈ 125 · RS485 50 Hz ≈ 25)
ZERO_STALE_S = 0.2           # 마지막 촉각 · 각도가 이보다 오래면 거부
ZERO_OPEN_MAX_RAD = 0.25     # 엄지 굽힘 · 네 손가락이 이 안 = 편 손(홈 손 0 rad). 엄지 회전(홈 1.57)은 안 본다
ZERO_STILL_RAD = 0.02        # 창 안 관절 범위 — 넘으면 아직 움직인다
ZERO_MAX_RAW = 300           # 창 평균 원시값(0.01 N 추정 → 3 N) — 넘으면 닿아 있다고 본다(09.30: 쉴 때 정확히 0)
ZERO_MAX_STD_RAW = 20        # 창 안 표준편차 — 넘으면 튄다
ZERO_MOVING = "손이 아직 움직인다"   # tools/trigger.py 가 이 거부만 다시 부른다(hand_home 직후)
#: ★11.04 관절 힘 bias 거부 — 창 안 칸별 표준편차 상한. 11.04 bag 13개(logs/analysis/thumb_support_1104/raw/hand_20261104_*)의
#  편 손 · 정지 0.5 s 창(span ≤ 0.02 rad · 굽힘 ≤ 0.25 rad, 0.25 s 간격 약 2.4만 창)에서 칸 최대 sd 중앙 0.6 g · bag 별 p99 ≤ 1.7 g,
#  튄 창(오른손 114514 · 131813 · 134810 최대 12~18 g)은 거른다
JOINT_ZERO_MAX_STD_G = 5.0
#: 창 평균 절대 상한 — 같은 창 |평균| 최대 217.8 g(오른손 엄지 굽힘 영점 이동). 이보다 크면 무엇을 누르는 채로 본다. tip_zero_force 만 넘는다
JOINT_ZERO_MAX_ABS_G = 400.0
#: 직전 bias 에서 이만큼 넘게 바뀐 칸은 받되 경고(tip_zero_status joint_jump_g) — 10.20 포화 하중 뒤 엄지 +205~213 g 같은 이동
JOINT_ZERO_JUMP_WARN_G = 30.0
JOINT_FORCE_NAMES = ("thumb_bend", "thumb_rot", "index", "middle", "ring", "pinky")   # joint_forces 순


@dataclass
class TipZero:
    """촉각 원시(벤더 순 법선 5 · 접선 5) bias 를 잡고 뺀다. 순수 — 시각은 부르는 쪽이 준다."""

    hmap: rh56f1_map.HandMap
    normal_bias: np.ndarray = field(default_factory=lambda: np.zeros(5))
    tangential_bias: np.ndarray = field(default_factory=lambda: np.zeros(5))
    joint_bias: np.ndarray = field(default_factory=lambda: np.zeros(6))   # g, joint_forces 순
    joint_bias_set: bool = False
    _touch: deque = field(default_factory=deque, init=False)
    _angle: deque = field(default_factory=deque, init=False)
    _jf: deque = field(default_factory=deque, init=False)

    @staticmethod
    def _push(buf: deque, t: float, row: np.ndarray) -> None:
        buf.append((t, row))
        while buf and buf[0][0] < t - 2 * ZERO_WINDOW_S:
            buf.popleft()

    def add_touch(self, t: float, normal: Sequence[float], tangential: Sequence[float]) -> None:
        self._push(self._touch, t, np.concatenate([np.asarray(normal, float), np.asarray(tangential, float)]))

    def add_angle(self, t: float, q6: Sequence[float]) -> None:
        self._push(self._angle, t, np.asarray(q6, float))

    def add_joint_force(self, t: float, jf6: Sequence[float]) -> None:
        self._push(self._jf, t, np.asarray(jf6, float))

    def joint_zeroed(self, jf6: Sequence[float]) -> list[float] | None:
        """원값 − bias(부호 그대로). 관절 힘 bias 를 받아들이기 전에는 None — 내지 않는다(원값을 영점 뺀 값으로 오해하지 않게)."""
        return (np.asarray(jf6, float) - self.joint_bias).tolist() if self.joint_bias_set else None

    def zeroed(self, normal: Sequence[float], tangential: Sequence[float]) -> np.ndarray:
        n = np.clip(np.asarray(normal, float) - self.normal_bias, 0.0, None)
        g = np.clip(np.asarray(tangential, float) - self.tangential_bias, 0.0, None)
        return self.hmap.touch_sim_order(n, g)

    def bias_n(self) -> np.ndarray:
        return self.hmap.touch_sim_order(self.normal_bias, self.tangential_bias)

    def capture(self, now: float, force_joint: bool = False) -> tuple[bool, list[str], dict]:
        """창 [now − 0.5 s, now] 평균을 bias 로. 거부면 이전 bias 그대로. → (손끝 받아들임, 손끝 이유, 기록용 값 — 관절 힘 결과 포함).
        관절 힘은 손끝과 따로 받아들인다(같은 창 · 같은 편 손 · 정지 조건). force_joint = 절대 상한만 건너뛴다."""
        touch = np.array([r for t, r in self._touch if t >= now - ZERO_WINDOW_S]).reshape(-1, 10)
        angle = np.array([r for t, r in self._angle if t >= now - ZERO_WINDOW_S]).reshape(-1, 6)
        why: list[str] = []
        if len(touch) < ZERO_MIN_SAMPLES or now - self._touch[-1][0] > ZERO_STALE_S:
            why.append(f"촉각 샘플이 모자라거나 늦다({len(touch)}개 / {ZERO_WINDOW_S} s)")
        hand = self._hand_reasons(now, angle)
        why += hand
        mean = touch.mean(0) if len(touch) else np.zeros(10)
        std = touch.std(0) if len(touch) else np.zeros(10)
        if mean.max() > ZERO_MAX_RAW:
            why.append(f"원시값이 크다(평균 최대 {mean.max():.0f} > {ZERO_MAX_RAW}) — 손끝이 무엇에 닿아 있는가")
        if std.max() > ZERO_MAX_STD_RAW:
            why.append(f"값이 튄다(표준편차 최대 {std.max():.1f} > {ZERO_MAX_STD_RAW})")
        if not why:
            self.normal_bias, self.tangential_bias = mean[:5].copy(), mean[5:].copy()
        info = {"samples": len(touch), "mean_raw": [round(float(v), 2) for v in mean], "std_raw": [round(float(v), 2) for v in std],
                "normal_bias_raw": self.normal_bias.tolist(), "tangential_bias_raw": self.tangential_bias.tolist(),
                "bias_n": self.bias_n().tolist(), **self._capture_joint(now, hand, force_joint)}
        return not why, why or ["bias 갱신"], info

    def _capture_joint(self, now: float, hand: list[str], force: bool) -> dict:
        """관절 힘 6 bias — 편 손 · 정지(hand) · 칸별 sd ≤ JOINT_ZERO_MAX_STD_G · |평균| ≤ JOINT_ZERO_MAX_ABS_G(force 면 이것만 안 본다).
        받아들일 때 직전 bias 에서 JOINT_ZERO_JUMP_WARN_G 넘게 바뀐 칸은 joint_jump_g 에 남긴다(받기는 한다)."""
        jf = np.array([r for t, r in self._jf if t >= now - ZERO_WINDOW_S]).reshape(-1, 6)
        why = list(hand)
        if len(jf) < ZERO_MIN_SAMPLES or now - self._jf[-1][0] > ZERO_STALE_S:
            why.append(f"관절 힘 샘플이 모자라거나 늦다({len(jf)}개 / {ZERO_WINDOW_S} s)")
        mean = jf.mean(0) if len(jf) else np.zeros(6)
        std = jf.std(0) if len(jf) else np.zeros(6)
        noisy = {n: round(float(v), 1) for n, v in zip(JOINT_FORCE_NAMES, std) if v > JOINT_ZERO_MAX_STD_G}
        if noisy:
            why.append(f"관절 힘이 튄다(표준편차 {noisy} g > {JOINT_ZERO_MAX_STD_G})")
        big = {n: round(float(v), 1) for n, v in zip(JOINT_FORCE_NAMES, mean) if abs(v) > JOINT_ZERO_MAX_ABS_G}
        if big and not force:
            why.append(f"관절 힘이 크다(|평균| {big} g > {JOINT_ZERO_MAX_ABS_G:.0f}) — 무엇을 누르는가(강제로 잡기: tip_zero_force)")
        jump = {}
        if not why:
            if self.joint_bias_set:
                jump = {n: round(float(d), 1) for n, d in zip(JOINT_FORCE_NAMES, mean - self.joint_bias) if abs(d) > JOINT_ZERO_JUMP_WARN_G}
            self.joint_bias, self.joint_bias_set = mean.copy(), True
        return {"joint_accepted": not why, "joint_reasons": why or ["bias 갱신"], "joint_forced": bool(force), "joint_jump_g": jump,
                "joint_samples": len(jf), "joint_mean_g": [round(float(v), 1) for v in mean],
                "joint_std_g": [round(float(v), 2) for v in std],
                "joint_bias_g": [round(float(v), 1) for v in self.joint_bias], "joint_bias_set": self.joint_bias_set}

    def _hand_reasons(self, now: float, angle: np.ndarray) -> list[str]:
        """손끝 · 관절 힘 영점이 같이 보는 손 조건 — 각도가 오고, 멈춰 있고, 펴져 있다."""
        why: list[str] = []
        if len(angle) == 0 or now - self._angle[-1][0] > ZERO_STALE_S:
            why.append("손 각도(angle_actual)가 안 온다")
        else:
            flex = [i for i, n in enumerate(self.hmap.joint_order) if n != "thumb_1"]
            span = float((angle.max(0) - angle.min(0)).max())
            bent = {self.hmap.joint_order[i]: round(float(angle[-1, i]), 3) for i in flex if angle[-1, i] > ZERO_OPEN_MAX_RAD}
            if span > ZERO_STILL_RAD:
                why.append(f"{ZERO_MOVING}(창 안 관절 범위 {span:.3f} rad > {ZERO_STILL_RAD})")
            elif bent:
                why.append(f"손이 펴지지 않았다 {bent} rad > {ZERO_OPEN_MAX_RAD}")
        return why


@dataclass
class HandStateCore:
    """레지스터 → (이름, 위치, 속도). 순수 — ROS 없이 테스트한다."""

    hmap: rh56f1_map.HandMap
    side: str
    tau_s: float = VEL_TAU_S
    _last: tuple[float, np.ndarray] | None = field(default=None, init=False)
    _vel: np.ndarray | None = field(default=None, init=False)

    def names(self) -> list[str]:
        p = self.side[0]
        order = list(self.hmap.joint_order) + [m[0] for m in self.hmap.mimic]
        return [f"{p}_hj_{n}" for n in order]

    def on_angle(self, reg: Sequence[float], t: float) -> tuple[list[str], np.ndarray, np.ndarray]:
        q6 = self.hmap.to_rad(reg, self.side)                     # 이 손의 보정(09.30 스윕)
        full = self.hmap.with_mimic(q6)
        pos = np.array([full[n.split("_hj_", 1)[1]] for n in self.names()])
        if self._last is not None and t > self._last[0]:
            dt = t - self._last[0]
            v = (pos - self._last[1]) / dt
            a = 1.0 - math.exp(-dt / self.tau_s)
            self._vel = v if self._vel is None else a * v + (1.0 - a) * self._vel
        self._last = (t, pos)
        vel = np.zeros_like(pos) if self._vel is None else self._vel
        return self.names(), pos, vel

    def tip_forces(self, finger_forces: Sequence[float], tangentials: Sequence[float] | None = None) -> np.ndarray:
        return self.hmap.touch_sim_order(finger_forces, tangentials)


def main(argv: list[str] | None = None) -> int:
    import rclpy
    from sensor_msgs.msg import JointState
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray, String
    from std_srvs.srv import Trigger
    from rh56f1_interfaces.msg import GetAngleAct1, GetForceAct1, TouchData1

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--map", default=str(rh56f1_map.DEFAULT_PATH), help="변환표 yaml")
    args, _ = ap.parse_known_args(argv)
    core = HandStateCore(rh56f1_map.load(args.map), args.side)
    ns = f"/hand_{args.side}"

    rclpy.init()
    from policy_control.lean_node import make_lean_node
    node = make_lean_node(f"rh56f1_state_{args.side}")  # ★10.03 CPU: QoS 이벤트 · 파라미터 서비스 끔
    from policy_control.cpu_plan import keep_off_rt
    node.get_logger().info(keep_off_rt())               # ★10.03 실시간 코어를 비켜 간다
    js_pub = node.create_publisher(JointState, f"{ns}/joint_states", 10)
    tip_pub = node.create_publisher(Float64MultiArray, f"{ns}/tip_forces", 10)
    jf_pub = node.create_publisher(Float64MultiArray, f"{ns}/joint_forces", 10)
    zero = TipZero(core.hmap)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    zeroed_pub = node.create_publisher(Float64MultiArray, f"{ns}/tip_forces_zeroed", 10)
    jf_zeroed_pub = node.create_publisher(Float64MultiArray, f"{ns}/joint_forces_zeroed", 10)
    bias_pub = node.create_publisher(Float64MultiArray, f"{ns}/tip_bias", latched)
    zstat_pub = node.create_publisher(String, f"{ns}/tip_zero_status", latched)
    zero_log = Path(__file__).resolve().parents[3] / "logs" / "rh56f1_tip_zero"

    def on_angle(msg) -> None:
        # ★10.03 원본 도장을 이어받는다(EtherCAT 노드 = 하드웨어 샘플 시각) — 없으면(0) 받은 시각. bag 정렬 · 속도 계산 모두 이 시각
        src = msg.header.stamp
        stamp_ns = src.sec * 1_000_000_000 + src.nanosec if (src.sec or src.nanosec) else node.get_clock().now().nanoseconds
        names, pos, vel = core.on_angle(list(msg.joint_values), stamp_ns * 1e-9)
        out = JointState()
        out.header.stamp = msg.header.stamp if (src.sec or src.nanosec) else node.get_clock().now().to_msg()
        out.name, out.position, out.velocity = names, pos.tolist(), vel.tolist()
        js_pub.publish(out)
        zero.add_angle(time.monotonic(), pos[:6])          # names 앞 6 = joint_order(구동)

    def on_touch(msg) -> None:
        n, g = list(msg.finger_forces), list(msg.finger_tangentials)
        tip_pub.publish(Float64MultiArray(data=core.tip_forces(n, g).tolist()))
        zero.add_touch(time.monotonic(), n, g)
        zeroed_pub.publish(Float64MultiArray(data=zero.zeroed(n, g).tolist()))

    def on_tip_zero(req, resp, force: bool = False):
        ok, why, info = zero.capture(time.monotonic(), force_joint=force)
        row = {"at": time.strftime("%F %T"), "side": args.side, "domain": os.environ.get("ROS_DOMAIN_ID", ""),
               "accepted": ok, "reasons": why, **info}
        zstat_pub.publish(String(data=json.dumps(row, ensure_ascii=False)))
        if ok:
            bias_pub.publish(Float64MultiArray(data=info["bias_n"]))
        try:                                                # 기록 실패가 영점을 막지 않는다 — 상태 토픽 · bag 에는 남는다
            zero_log.mkdir(parents=True, exist_ok=True)
            with open(zero_log / f"{time.strftime('%Y%m%d')}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            node.get_logger().error(f"tip_zero 기록 실패 {zero_log}: {e}")
        (node.get_logger().info if ok else node.get_logger().warning)(
            f"tip_zero {'받아들임' if ok else '거부(이전 bias 유지)'}: {'; '.join(why)} · bias N {[round(v, 3) for v in info['bias_n']]}")
        jok = info["joint_accepted"]
        (node.get_logger().info if jok else node.get_logger().warning)(
            f"관절 힘 영점{'(강제)' if force else ''} {'받아들임' if jok else '거부(이전 bias 유지)'}: {'; '.join(info['joint_reasons'])} · "
            f"창 평균 g {info['joint_mean_g']} · bias g {info['joint_bias_g']}")
        if info["joint_jump_g"]:
            node.get_logger().warning(f"관절 힘 영점이 직전 bias 에서 {JOINT_ZERO_JUMP_WARN_G:.0f} g 넘게 옮겨 갔다(받아들임) {info['joint_jump_g']} g")
        resp.success = ok
        resp.message = json.dumps({"ok": ok, "reasons": why, "joint_ok": jok, "joint_reasons": info["joint_reasons"]}, ensure_ascii=False)
        return resp

    node.create_service(Trigger, f"{ns}/tip_zero", on_tip_zero)
    node.create_service(Trigger, f"{ns}/tip_zero_force", lambda req, resp: on_tip_zero(req, resp, force=True))

    node.create_subscription(GetAngleAct1, f"{ns}/angle_actual", on_angle, 10)
    node.create_subscription(TouchData1, f"{ns}/touch_data", on_touch, 10)

    def on_force(msg) -> None:
        jf = joint_forces_out(list(msg.joint_values))
        jf_pub.publish(Float64MultiArray(data=jf))
        zero.add_joint_force(time.monotonic(), jf)
        out = zero.joint_zeroed(jf)
        if out is not None:   # 영점 전에는 내지 않는다(정책 노드는 None = 결손으로 본다)
            jf_zeroed_pub.publish(Float64MultiArray(data=out))

    node.create_subscription(GetForceAct1, f"{ns}/force_actual", on_force, 10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

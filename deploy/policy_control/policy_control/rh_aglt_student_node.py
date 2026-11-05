"""rh_aglt_student_node: RH56F1 한 팔 깊이 학생 잡기(1104e, 깊이 + RGB) — pour_fj_node 의 rh_aglt 계열에 체인만 StudentChain(11.05).

  in   rh_aglt_node 와 같은 팔 · 손 · 촉각 · 관절 힘 · 컵(FP++ live — 래치 · lift_hold 판정에만, 관측에는 없음)
       /head_depth/image 32FC1 (90, 160) [m] · /head_depth/rgb 32FC3 0~1(같은 stamp) · /head_depth/config(latched JSON)
       — scripts/nodes/head_depth_node.py --policy-dir <같은 묶음> 이 낸다
  out  rh_aglt_node 와 같다(joint_target · obs · action(실행 행동) · episode · status)

    .venv-student/bin/python deploy/policy_control/policy_control/rh_aglt_student_node.py --ros-args \\
        -r __node:=rh_aglt_node_right -p ns:=right -p robot:=… \\
        -p contract:=deploy/policies/rh56f1/aglt/right_approach_ap05/rh_aglt_contract.json \\
        -p student_dir:=deploy/policies/rh56f1/aglt/right_student1104e

계약 = 홈 출발 계약(디코더 · 물체 · 시작 검사). 학생 ckpt · 외부 파라미터 md5 · hdgp 스냅숏 커밋은 묶음 student.yaml 로 시작 때 검사한다.
reset 은 /head_depth/config 를 ckpt depth_pre 와 대조해 다르면 거부한다. 깊이가 낡거나(> 0.2 s) 없거나 rgb 짝이 없으면 그 틱은
오류(StaleDepth) — max_gap_ticks 넘게 이어지면 abort. 접근 상한(sim ap_max_steps) 안에 못 멈추면 abort.
max_episode_s 기본 = sim 평가 episode_length_s(묶음 spec.json).
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import rh_aglt as A  # noqa: E402
from policy_control.pour_fj_node import Family, PourFjNode, main  # noqa: E402
from policy_control.student_chain import StudentChain, spec_limits  # noqa: E402
from policy_control.student_policy import StaleDepth, StudentPolicy, check_depth_config  # noqa: E402

DEPTH_TOPIC, RGB_TOPIC, CONFIG_TOPIC = "/head_depth/image", "/head_depth/rgb", "/head_depth/config"
RGB_KEEP = 8          # rgb 는 깊이보다 먼저 온다 — 최근 몇 장을 stamp 로 짝짓는다
EMPTY_LOG_S = 2.0


def image_array(msg, shape: tuple, encoding: str) -> np.ndarray:
    """sensor_msgs/Image(32FC1 · 32FC3, 리틀 엔디언) → float32 배열. 모양 · 형식이 다르면 StaleDepth(그 틱만 버린다)."""
    ch = {"32FC1": 1, "32FC3": 3}[encoding]
    h, w = shape
    if msg.encoding != encoding or (msg.height, msg.width) != (h, w) or msg.step != w * 4 * ch or msg.is_bigendian:
        raise StaleDepth(f"이미지 {msg.encoding} {msg.height}x{msg.width} step {msg.step} ≠ {encoding} {h}x{w}")
    a = np.frombuffer(bytes(msg.data), np.float32)
    return (a.reshape(h, w) if ch == 1 else a.reshape(h, w, 3)).copy()


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)


class StudentAgltNode(PourFjNode):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        from policy_control.raw_poll import poll_subscription

        # StaleDepth(RuntimeError)도 틱 오류(gap → abort) · reset 거부 경로를 타게 한다. FAMILIES 는 그대로(family_of)
        self.fam = Family(**{**self.fam.__dict__, "chain": StudentChain, "errors": (*self.fam.errors, StaleDepth)})
        d = str(self.declare_parameter("student_dir", "").value).strip()
        if not d:
            raise A.RhAgltError("student_dir:=<학생 묶음 폴더(student.yaml)> 가 필요하다")
        self._policy = StudentPolicy.from_dir(d, device=self.device)      # JointPolicy 대신(reset 이 그대로 쓴다)
        side = self.contract.side().side
        if str(self._policy.policy_cfg.get("side")) != side:
            raise A.RhAgltError(f"학생 묶음 side {self._policy.policy_cfg.get('side')} ≠ 계약 {side}")
        _, episode_s = spec_limits(self._policy.policy_cfg)
        if float(self.get_parameter("max_episode_s").value) < 0:
            self.max_s = episode_s
        self._Image = Image
        self._sub_depth = poll_subscription(self._poll, Image, DEPTH_TOPIC, qos_profile_sensor_data)
        self._sub_rgb = poll_subscription(self._poll, Image, RGB_TOPIC, qos_profile_sensor_data) if self._policy.rgb_on else None
        self._depth: tuple[np.ndarray, int] | None = None
        self._rgb: deque = deque(maxlen=RGB_KEEP)
        self._depth_cfg: dict | None = None
        self._empty_log = 0.0
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, CONFIG_TOPIC, self._on_depth_cfg, latched)
        self.get_logger().info(f"학생 {d} · rgb {self._policy.rgb_on} · 넘김 {self._policy.handover} · "
                               f"에피소드 상한 {self.max_s:.1f} s")

    def _on_depth_cfg(self, msg) -> None:
        try:
            self._depth_cfg = json.loads(msg.data)
        except ValueError as exc:
            self._depth_cfg = None
            self._errors["head_depth_config"] = f"{CONFIG_TOPIC} JSON 아님: {exc}"

    def _drain(self) -> None:
        super()._drain()
        from rclpy.serialization import deserialize_message
        from policy_control.raw_poll import take_all, take_last

        hw = self._policy.img_hw
        try:
            if self._sub_rgb is not None:
                for data, _ in take_all(self._sub_rgb):
                    msg = deserialize_message(data, self._Image)
                    self._rgb.append((stamp_ns(msg), image_array(msg, hw, "32FC3")))
            got = take_last(self._sub_depth)
            if got is not None:
                msg = deserialize_message(got[0], self._Image)
                self._depth = (image_array(msg, hw, "32FC1"), stamp_ns(msg))
            self._errors.pop("head_depth", None)
        except StaleDepth as exc:
            self._errors["head_depth"] = str(exc)

    def _measure(self, advance: bool = False) -> dict:
        meas = super()._measure(advance)
        return {**meas, "depth": self._frame()} if advance else meas

    def _frame(self) -> tuple:
        """(깊이 m, 같은 stamp rgb | None, 나이 s). 나이 상한은 StudentPolicy.step 이 본다."""
        why = self._errors.get("head_depth")
        why = f" (최근 수신 오류: {why})" if why else ""
        if self._depth is None:
            raise StaleDepth(f"{DEPTH_TOPIC} 를 아직 못 받았다 — head_depth_node 가 도는가{why}")
        depth, st = self._depth
        rgb = None
        if self._policy.rgb_on:
            rgb = next((a for s, a in reversed(self._rgb) if s == st), None)
            if rgb is None:
                raise StaleDepth(f"깊이 stamp {st} 와 같은 {RGB_TOPIC} 가 없다{why}")
        return depth, rgb, (time.time_ns() - st) * 1e-9

    def _reset_chain(self, meas: dict) -> None:
        if self._depth_cfg is None:
            raise A.RhAgltError(f"{CONFIG_TOPIC} 를 못 받았다 — head_depth_node --policy-dir <같은 묶음> 을 먼저 띄운다")
        check_depth_config(self._policy.depth_pre, self._depth_cfg)
        super()._reset_chain(meas)

    def _after_step(self) -> bool:
        if getattr(self._policy, "depth_empty", False) and time.monotonic() - self._empty_log > EMPTY_LOG_S:
            self._empty_log = time.monotonic()
            self.get_logger().warning("깊이에 유효 화소가 없는 프레임이 이어진다 — 카메라 · 바닥 컷을 본다")
        end = getattr(self.chain, "end", None)
        if end is not None:
            self._end(*end)
            return True
        return super()._after_step()

    def _status_extra(self) -> dict:
        extra = super()._status_extra()
        return {**extra, "student": self.chain.view()} if hasattr(self.chain, "view") else extra


if __name__ == "__main__":
    raise SystemExit(main(family="rh_aglt", node_cls=StudentAgltNode))

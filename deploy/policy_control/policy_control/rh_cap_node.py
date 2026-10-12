"""rh_cap_node: RH56F1 오른팔 캡 단계 정책(hdgp open-rh_r_cap, cap_phase grasp · seat) — pour_fj_node 의 rh_cap 계열(10.12 사용자 간단 배포).

  in   robot yaml(한 팔, rh56f1_right_*): 팔 /joint_states · 손 /hand_right/joint_states · 촉각 · 관절 힘
       캡 PoseStamped(base): cap_topic(기본 /objects/cap/pose) — grasp 는 쥘 때까지 live(cup_attach 가 쥔 뒤 손바닥에 붙인다),
         seat 는 reset 순간에만 쓴다(그 뒤는 손바닥에 붙은 캡)
       리시버 PoseStamped(base): receiver_topic(기본 /objects/receiver/pose) — reset 순간 목표(끼움 자리)를 정한다
       /policy_control/joint_target — pd 가 붙잡은 마지막 목표(S0 · S2 경로 재생 끝) = 시작 팔 · 손 목표(없으면 실측, allow_measured_start)
  out  /policy_control/joint_target (팔 7 · 손 6) · /policy_control/<ns>/{obs,action,episode} · status/<노드>
  srv  /policy_control/<ns>/episode/{reset,start,stop,abort} — 단계 성공(phase_success_steps 연속)이면 스스로 stop

    python3 deploy/policy_control/policy_control/rh_cap_node.py --ros-args -r __node:=rh_cap_grasp_node -p ns:=right_capg \\
        -p contract:=deploy/policies/rh56f1/cap/right_capg_j04/rh_cap_contract.json -p robot:=… -p episode_topic:=/policy_control/right/episode
S0(사전 파지 자세) · S2(끼우기 직전 자세)는 이 노드 밖 — 오늘은 사람이 맞춘 자세 또는 pd 경로 재생(미리 계산한 관절 경로).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import codec  # noqa: E402
from policy_control import rh_aglt as A  # noqa: E402
from policy_control import rh_cap as C  # noqa: E402
from policy_control.pour_fj_node import (FAMILIES, NS, Family, PourFjNode, PourFjNodeError, aglt_meas,  # noqa: E402
                                         aglt_target_arrays, main)

# 계열 등록(pour_fj_node.py 는 공용 — 이 모듈을 불러올 때만 rh_cap 이 생긴다)
FAMILIES.setdefault("rh_cap", Family(
    name="rh_cap_node", schema=C.SCHEMA, load=C.load_contract, roles=A.ROLES,
    cups=(("arm", "cup_topic", "/objects/cap/pose"),),
    meas=aglt_meas, chain=C.CapChain, refusals=lambda c, meas, _tol: C.start_refusals(c, meas),
    targets=aglt_target_arrays, errors=(PourFjNodeError, A.RhAgltError, ValueError),
    label=lambda c: f"{c.side().side} arm · cap {c.cap_phase} · 성공 {c.phase_success_steps} 스텝"))


class CapNode(PourFjNode):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from policy_control.raw_poll import poll_subscription

        for name, default in (("receiver_topic", "/objects/receiver/pose"), ("allow_measured_start", False)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if self.contract.cap_phase == "seat":
            self.attach = {}                               # 쥔 캡은 CapChain 이 시작 순간 손바닥에 붙인다
        self._rcv: tuple | None = None
        self._rcv_topic = str(p("receiver_topic"))
        self.create_subscription(PoseStamped, self._rcv_topic, self._on_receiver,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._held_q: dict[str, float] = {}
        self._JointState = JointState
        self._poll_held = poll_subscription(self._poll, JointState, f"{NS}/joint_target",
                                            QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
        self.get_logger().info(f"cap {self.contract.cap_phase} · 리시버 {self._rcv_topic} · 끼움 자리 리시버 축 "
                               f"+{self.contract.seat_offset:.4f} m")

    def _on_receiver(self, msg) -> None:
        try:
            s = codec.decode_pose(msg)
            self._rcv = (np.asarray(s.pos, float), np.asarray(s.quat, float))
        except (codec.CodecError, ValueError) as exc:
            self._errors["receiver"] = str(exc)

    def _drain(self) -> None:
        super()._drain()
        from rclpy.serialization import deserialize_message
        from policy_control.raw_poll import take_all
        for raw, _t in take_all(self._poll_held):
            msg = deserialize_message(raw, self._JointState)
            self._held_q.update(zip(msg.name, (float(v) for v in msg.position)))

    def _held(self, meas: dict):
        s = self.contract.side()
        if not all(j in self._held_q for j in list(s.arm_joints) + list(s.hand_joints)):
            return None
        arm = np.array([self._held_q[j] for j in s.arm_joints])
        err = float(np.abs(arm - meas["arm"].arm_q).max())
        if err > self.reset_tol:
            raise PourFjNodeError(f"pd 가 붙잡은 마지막 목표가 팔 실측에서 {err:.3f} rad(> {self.reset_tol}) — 경로 재생 직후가 아니다")
        return arm, np.array([self._held_q[j] for j in s.hand_joints])

    def _cup_for(self, role: str, advance: bool):
        if self.contract.cap_phase == "seat" and getattr(self.chain, "rel", None) is not None:
            return lambda palm_pos, palm_R, _tact, _st: self.chain.cup_from_palm(palm_pos, palm_R)
        return super()._cup_for(role, advance)

    def _srv_reset(self, req, res):
        if self.chain is not None:
            self.chain.rel = None                          # 새 에피소드 — seat 캡은 FP++ 한 번으로 다시 붙인다
        return super()._srv_reset(req, res)

    def _reset_chain(self, meas: dict) -> None:
        self.chain.fk = self.fk["arm"]
        self.chain.grasp_cfg = self.attach_cfg
        held = self._held(meas)
        if held is None and not bool(self.get_parameter("allow_measured_start").value):
            raise PourFjNodeError("pd 가 붙잡은 마지막 joint_target 이 없다 — S0/S2 경로 재생 뒤에 띄울 것"
                                  "(실측에서 시작하려면 allow_measured_start)")
        self.chain.reset(meas, receiver=self._rcv, held=held)
        self.get_logger().info(f"cap reset · {self.chain.as_dict()} · 시작 목표 "
                               f"{'pd 마지막 joint_target' if held is not None else '실측'}")

    def _after_step(self) -> bool:
        if self.chain.done:
            self._end("stop", f"cap {self.contract.cap_phase} 성공 — {self.contract.phase_success_steps} 스텝 연속 {self.chain.last}")
            return True
        return False

    def _status_extra(self) -> dict:
        return {"receiver": None if self._rcv is None else [round(float(v), 4) for v in self._rcv[0]],
                **({"cap": self.chain.as_dict()} if self.chain is not None else {})}


if __name__ == "__main__":
    raise SystemExit(main(family="rh_cap", node_cls=CapNode))

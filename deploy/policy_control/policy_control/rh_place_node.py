"""rh_place_node: RH56F1 한 팔 컵 홀더 놓기(hdgp open-rh_{r,l}_place) — pour_fj_node 의 rh_place 계열 + 이 노드 고유 부분(10.04).

  in   robot yaml(한 팔, rh56f1_<side>_*): 팔 /joint_states · 손 /hand_<side>/joint_states · 촉각 · 관절 힘
       컵 PoseStamped(base): cup_topic(기본 /objects/cyl60/pose) — reset 순간에만 쓴다(그 뒤는 손바닥에 붙은 컵)
       홀더 PoseStamped(base, latched): holder_topic(기본 /objects/cup_holder_<holder>/pose) — 목표 = 원점 + (0, 0, seat_dz)
       /policy_control/joint_target — pd 가 붙잡고 있는 마지막 목표(aglt 가 낸 것) = 인계 팔 · 손 목표(sim 뱅크 arm_q_target)
  out  /policy_control/joint_target (팔 7 · 손 6) · /policy_control/<ns>/{obs,action,episode} · status/<노드>
  srv  /policy_control/<ns>/episode/{reset,start,stop,abort} — 놓은 뒤 스크립트가 끝나면 스스로 stop

    python3 deploy/policy_control/policy_control/rh_place_node.py --ros-args -r __node:=rh_place_node_right -p ns:=right \\
        -p contract:=deploy/policies/right_rh_place_i09/rh_place_contract.json -p robot:=… [-p holder:=2]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import codec  # noqa: E402
from policy_control.pour_fj_node import NS, Family, PourFjNode, PourFjNodeError, main  # noqa: E402
from policy_control.rh_place import JOINT_FREE_G  # noqa: E402


class PlaceNode(PourFjNode):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from policy_control.raw_poll import poll_subscription

        for name, default in (("holder", -1), ("holder_topic", ""), ("allow_measured_start", False),
                              ("joint_free_g", JOINT_FREE_G), ("require_grasp", True)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if not bool(p("require_grasp")):                   # fake 손은 촉각이 0 — 리허설에서만 끈다
            self.fam = Family(**{**self.fam.__dict__, "refusals": lambda _c, _m, _t: []})
        k = int(p("holder"))
        self.holder_id = self.contract.target_holders[0] if k < 0 else k
        if self.holder_id not in self.contract.target_holders:
            raise PourFjNodeError(f"holder {self.holder_id} 는 학습 목표 홀더 {self.contract.target_holders} 가 아니다")
        self.holder_topic = str(p("holder_topic")) or f"/objects/cup_holder_{self.holder_id}/pose"
        self.attach = {}                                   # 붙은 컵은 PlaceChain 이 든다(인계 순간 한 번, sim attached)
        self._holder: np.ndarray | None = None
        self._held_q: dict[str, float] = {}
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseStamped, self.holder_topic, self._on_holder, latched)
        self._JointState = JointState
        self._poll_held = poll_subscription(self._poll, JointState, f"{NS}/joint_target",
                                            QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
        self.get_logger().info(f"place · 홀더 {self.holder_id} ({self.holder_topic}) · 놓음 손끝 < "
                               f"{self.contract.release_force_n} N · 관절 힘 < {float(p('joint_free_g'))} g")

    # ---------------------------------------------------------------- inputs
    def _on_holder(self, msg) -> None:
        try:
            self._holder = np.asarray(codec.decode_pose(msg).pos, float)
        except (codec.CodecError, ValueError) as exc:
            self._errors["holder"] = str(exc)

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
            return lambda palm_pos, palm_R, _tact, _st: self.chain.cup_from_palm(palm_pos, palm_R)
        return super()._cup_for(role, advance)

    def _srv_reset(self, req, res):
        if self.chain is not None:
            self.chain.rel = None                          # 새 인계 — 컵은 FP++ 한 번으로 다시 붙인다
        return super()._srv_reset(req, res)

    def _reset_chain(self, meas: dict) -> None:
        self.chain.joint_free_g = float(self.get_parameter("joint_free_g").value)
        held = self._held(meas)
        self.chain.reset(meas, holder=self._holder, held=held,
                         allow_measured=bool(self.get_parameter("allow_measured_start").value))
        rel = self.chain.rel[0]
        self.get_logger().info(f"place reset · 자리 {self.chain.as_dict()['seat']} · 시작 목표 "
                               f"{'pd 마지막 joint_target' if held is not None else '실측(allow_measured_start)'} · "
                               f"손바닥 기준 컵 {np.round(rel, 3).tolist()}")

    def _after_step(self) -> bool:
        if self.chain.done:
            self._end("stop", f"placed — 놓음 {self.contract.release_steps} 스텝 + 스크립트 {self.contract.settle_steps} 스텝 끝")
            return True
        return False

    def _status_extra(self) -> dict:
        return {"holder": self.holder_id, "holder_pose": None if self._holder is None else [round(float(v), 4) for v in self._holder],
                **({"place": self.chain.as_dict()} if getattr(self.chain, "as_dict", None) else {})}


if __name__ == "__main__":
    raise SystemExit(main(family="rh_place", node_cls=PlaceNode))

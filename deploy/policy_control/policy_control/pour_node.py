"""pour_node: rclpy shell of the bimanual pour chain (obs -> policy -> decoder -> two fabrics) in ONE process.

The pour policy sees both arms in one 223-vector and its two fabrics share one CUDA world, so the
single-arm obs/policy/fabric node split does not apply. All logic lives in ROS-free modules
(pour_node_core.PourInbox, pour_chain.PourChain, pour_fabric.PourFabricPair); this file only wires IO.

  in   robot yaml sources of both sides (joint_state / float_array tip_force), via sources.SourceSet
       <src_cup_topic>, <rcv_cup_topic>  PoseStamped in the robot base frame (cup body origin as in sim)
       /policy_control/pour/fill_level   Float64, how full the SOURCE cup is 0..1 (optional if the contract has a default)
  out  /policy_control/joint_target      JointState canonical names, both arms + both hands, frame_id '<episode>:<seq>'
       /policy_control/{obs,action}      Float64MultiArray (logging)
       /policy_control/episode           latched JSON (this node is the episode master)
       /policy_control/status/pour_node  JSON
  srv  /policy_control/episode/{reset,start,stop,abort}  std_srvs/Trigger

pd_node runs next to it with a control-only DeployContract of the same asset (it never reads the pour contract).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

import numpy as np  # noqa: E402

from policy_control import _paths  # noqa: F401,E402
from policy_control import codec  # noqa: E402
from policy_control.chain import EpisodeEvent  # noqa: E402
from policy_control.episode_master import EpisodeBook  # noqa: E402
from policy_control.pour_chain import PourChain, PourChainError  # noqa: E402
from policy_control.pour_contract import PourContract, PourContractError, load_contract  # noqa: E402
from policy_control.pour_node_core import (  # noqa: E402
    PourInbox, PourNodeError, fabric_home, joint_target_arrays, start_refusals, tip_forces_to_inputs,
)
from policy_control.sources import RobotCfgError, SourceSet, load_robot_cfg, select_side  # noqa: E402

NS = "/policy_control"
NODE = "pour_node"
EVENTS = ("reset", "start", "stop", "abort")
_HANDLED = (PourNodeError, PourChainError, PourContractError, codec.CodecError, RobotCfgError, ValueError)


def contract_home(contract: PourContract) -> dict:
    home: dict = {}
    for s in contract.sides:
        home.update(dict(zip(s.arm_joints, (float(v) for v in s.arm_reset))))
    return home


def build_fks(contract: PourContract) -> dict:
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK

    if contract.asset not in ASSETS:
        raise PourNodeError(f"contract asset {contract.asset!r} is not in contract_assets.ASSETS")
    urdf = ASSETS[contract.asset].urdf
    return {s.role: UrdfChainFK(urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies) for s in contract.sides}


def feed_inbox(inbox: PourInbox, contract: PourContract, sets: dict, now: float) -> list:
    """Copy each side's fresh SourceSet snapshot into the inbox. Returns per-source problems (text)."""
    problems = []
    for s in contract.sides:
        src = sets[s.role]
        st = src.snapshot(now)
        bad = set(st.stale) | set(st.missing)
        problems += [f"{s.role}:{n}" for n in sorted(bad)]
        if "arm" not in bad and st.arm_q is not None:
            if st.arm_qd is None:
                raise PourNodeError(f"{s.role}: arm source has no velocity (arm_qd is an actor input)")
            inbox.put_joints(src.cfg.sources["arm"].joints, st.arm_q, st.arm_qd, now)
        if "ee" not in bad and st.ee_q is not None:
            inbox.put_joints(st.ee_names, st.ee_q, np.zeros(len(st.ee_names)), now)
        if st.tip_force is not None and "tip_force" not in bad:
            inbox.put_forces(s.role, *tip_forces_to_inputs(s, st.tip_names, st.tip_force), now)
    return problems


try:
    from rclpy.node import Node
except ImportError:
    Node = object                        # type: ignore[misc,assignment]


class PourNode(Node):
    def __init__(self, *, policy=None, fabric=None, fks=None, **kw) -> None:
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64, Float64MultiArray, String
        from std_srvs.srv import Trigger

        super().__init__(NODE, **kw)
        for name, default in (("contract", ""), ("robot", ""), ("device", "cuda:0"), ("stale_sec", 0.1),
                              ("src_cup_topic", ""), ("rcv_cup_topic", ""), ("reset_tol_rad", 0.15),
                              ("max_gap_ticks", 3), ("use_fabric", True)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        cpath, rpath = Path(str(p("contract"))), Path(str(p("robot")))
        if not cpath.is_file() or not rpath.is_file():
            raise PourNodeError(f"parameters 'contract' and 'robot' must be existing files (got {cpath!r}, {rpath!r})")
        cups = {"src": str(p("src_cup_topic")), "rcv": str(p("rcv_cup_topic"))}
        if not all(cups.values()):
            raise PourNodeError("parameters src_cup_topic and rcv_cup_topic are required (PoseStamped, robot base frame)")
        self.contract = load_contract(cpath)
        self.device = str(p("device"))
        self.reset_tol, self.max_gap = float(p("reset_tol_rad")), int(p("max_gap_ticks"))
        self._use_fabric, self._fabric, self._policy = bool(p("use_fabric")), fabric, policy
        robot_cfg = load_robot_cfg(rpath)
        self.sets = {s.role: SourceSet(select_side(robot_cfg, s.side)) for s in self.contract.sides}
        self.inbox = PourInbox(self.contract, float(p("stale_sec")))
        self.fks = fks if fks is not None else build_fks(self.contract)
        self.chain: PourChain | None = None
        self.book = EpisodeBook(contract_home(self.contract))
        self._seq, self._gap, self._errors = 0, 0, {}
        # feed_inbox 가 알려주는 소스별 결손/스테일. **거부 사유로 쓰지 않는다** — 좌우가 같은 토픽
        # (`/joint_states`)을 쓰면 한쪽 SourceSet 스냅샷이 결손으로 보여도 인박스에는 값이 들어와 있다.
        # 기동 가부는 실제 측정을 보는 `start_refusals` 가 정한다. 여기 값은 진단 표시용이다.
        self._src_problems: list = []

        chain_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._String, self._msgs = String, {"joint_state": JointState, "float_array": Float64MultiArray}
        self._pub_target = self.create_publisher(JointState, f"{NS}/joint_target", chain_qos)
        self._pub_obs = self.create_publisher(Float64MultiArray, f"{NS}/obs", chain_qos)
        self._pub_action = self.create_publisher(Float64MultiArray, f"{NS}/action", chain_qos)
        self._pub_episode = self.create_publisher(String, f"{NS}/episode", latched)
        self._pub_status = self.create_publisher(String, f"{NS}/status/{NODE}", QoSProfile(depth=10))
        for s in self.contract.sides:
            for src in self.sets[s.role].cfg.sources.values():
                if src.type in self._msgs and (src.role or src.name) in ("arm", "ee", "tip_force"):
                    self.create_subscription(self._msgs[src.type], src.topic, self._source_cb(s.role, src),
                                             qos_profile_sensor_data)
        for role, topic in cups.items():
            self.create_subscription(PoseStamped, topic, self._cup_cb(role), qos_profile_sensor_data)
        self.create_subscription(Float64, f"{NS}/pour/fill_level", self._on_fill, chain_qos)
        for name in EVENTS:
            self.create_service(Trigger, f"{NS}/episode/{name}", getattr(self, f"_srv_{name}"))
        self.create_timer(1.0 / float(self.contract.policy_hz), self._on_tick)
        self.get_logger().info(f"pour_node up · {self.contract.task} · obs {self.contract.obs_dim} act "
                               f"{self.contract.action_dim} · {self.contract.policy_hz:.1f} Hz · cups {cups}")

    # ---------------------------------------------------------------- inputs
    def _note(self, key: str, exc) -> None:
        self._errors[key] = str(exc)
        self.get_logger().warning(f"{key}: {exc}", throttle_duration_sec=1.0)

    def _source_cb(self, role: str, src):
        def cb(msg) -> None:
            try:
                if src.type == "joint_state":
                    self.sets[role].update_from_joint_state(src.name, codec.decode_joint_state(msg), time.monotonic())
                else:
                    self.sets[role].update_from_float_array(src.name, codec.decode_float_array(msg), time.monotonic())
                self._errors.pop(f"{role}:{src.name}", None)
            except _HANDLED as exc:
                self._note(f"{role}:{src.name}", exc)
        return cb

    def _cup_cb(self, role: str):
        def cb(msg) -> None:
            try:
                pose = codec.decode_pose(msg)
                self.inbox.put_cup(role, pose.pos, pose.quat, time.monotonic())
                self._errors.pop(f"cup:{role}", None)
            except _HANDLED as exc:
                self._note(f"cup:{role}", exc)
        return cb

    def _on_fill(self, msg) -> None:
        try:
            self.inbox.put_fill(float(msg.data), time.monotonic())
        except _HANDLED as exc:
            self._note("fill_level", exc)

    # ---------------------------------------------------------------- chain
    def _make_chain(self, home: dict) -> PourChain:
        if self._policy is None:
            from policy_control.pour_policy import PourPolicy
            self._policy = PourPolicy(self.contract, self.device)
        if self._fabric is None and self._use_fabric:
            from policy_control.pour_fabric import PourFabricPair
            self._fabric = PourFabricPair(self.contract, self.device, home)
        return PourChain(self.contract, self._policy, self.fks, fabric=self._fabric)

    def _measure(self):
        """인박스를 채우고 측정값을 낸다. 소스별 결손/스테일은 버리지 않고 status·거부 사유로 남긴다."""
        now = time.monotonic()
        self._src_problems = feed_inbox(self.inbox, self.contract, self.sets, now)
        return self.inbox.measure(now)

    # ---------------------------------------------------------------- services
    def _reply(self, res, ok: bool, reasons) -> object:
        res.success = bool(ok)
        res.message = json.dumps({"ok": bool(ok), "reasons": [str(r) for r in reasons]})
        return res

    def _emit(self, event: EpisodeEvent) -> None:
        body = {**event.as_dict(), "t_ns": self.get_clock().now().nanoseconds}
        self._pub_episode.publish(self._String(data=json.dumps(body)))
        self.get_logger().info(f"episode {event.episode} {event.event} {list(event.reasons)}")

    def _srv_reset(self, _req, res):
        try:
            m = self._measure()
            home = fabric_home(self.contract, m)
            if self.chain is None:
                self.chain = self._make_chain(home)
            self.chain.reset(home if self._fabric is not None else None)
        except _HANDLED as exc:
            return self._reply(res, False, [f"reset: {exc}"])
        self._seq, self._gap = 0, 0
        self.inbox.hold_fill(False)
        event, _ = self.book.reset()
        self._emit(event)
        return self._reply(res, True, [])

    def _srv_start(self, _req, res):
        try:
            refusals = start_refusals(self.contract, self._measure(), self.reset_tol)
        except _HANDLED as exc:
            return self._reply(res, False, [f"start: {exc}"])
        if refusals:
            return self._reply(res, False, refusals)
        event, reasons = self.book.start()
        if event is None:
            return self._reply(res, False, reasons)
        self.inbox.hold_fill(True)
        self._emit(event)
        return self._reply(res, True, [])

    def _end(self, kind: str, reason: str) -> None:
        self.inbox.hold_fill(False)
        event, _ = self.book.end(kind, reason)
        self._emit(event)

    def _srv_stop(self, _req, res):
        self._end("stop", "user stop")
        return self._reply(res, True, [])

    def _srv_abort(self, _req, res):
        self._end("abort", "abort requested")
        return self._reply(res, True, [])

    # ---------------------------------------------------------------- tick
    def _status(self, ok: bool, reasons, extra=None) -> None:
        # t_ns 는 다른 네 노드와 같은 규약이다 — status_join 의 지연 계산이 이 값을 쓴다.
        # 빠뜨리면 pour 체인만 지연이 영영 NaN 으로 나온다(09.21 fake 런에서 드러났다).
        body = {"node": NODE, "phase": self.book.phase, "episode": self.book.episode, "seq": self._seq,
                "ok": bool(ok), "reasons": [str(r) for r in reasons],
                "sources": list(self._src_problems),
                # 거짓이면 joint_target 을 내지 않는다(행동만 계산) — pd 는 제 내부 목표를 따른다. 화면이 그 사실을 말할 수 있게 싣는다.
                "use_fabric": self._use_fabric,
                # 입력별 수신 상태(live/stale/missing/held/off). ok·reasons 는 틱이 거부된 뒤에야 말하므로
                # 에피소드 밖(IDLE)에서도 "입력이 들어오고 있는가"를 화면이 볼 수 있게 매 status 에 싣는다.
                "inputs": self.inbox.inputs(time.monotonic()),
                "t_pub_ns": self.get_clock().now().nanoseconds, **(extra or {})}
        self._pub_status.publish(self._String(data=json.dumps(body)))

    def _on_tick(self) -> None:
        if self.book.phase != "running" or self.chain is None:
            try:                                                          # IDLE 에서도 입력 나이를 최신으로 둔다
                self._src_problems = feed_inbox(self.inbox, self.contract, self.sets, time.monotonic())
            except _HANDLED as exc:
                self._status(False, [*self._errors.values(), str(exc)])
                return
            self._status(True, list(self._errors.values()))
            return
        t0 = time.perf_counter()
        try:
            step = self.chain.step(self._measure())
            names, q, qd = joint_target_arrays(self.contract, step) if step.joint_targets is not None else ((), None, None)
        except _HANDLED as exc:
            self._gap += 1
            self._status(False, [str(exc)], {"gap": self._gap})
            if self._gap > self.max_gap:
                self._end("abort", f"{self._gap} ticks without a valid step: {exc}")
            return
        self._gap = 0
        self._seq += 1
        self._pub_obs.publish(codec.encode_float_array(step.obs, ["obs"], [step.obs.size], self._seq))
        self._pub_action.publish(codec.encode_action(step.action, self._seq))
        if q is not None:
            self._pub_target.publish(codec.encode_joint_target(names, q, qd, str(self.book.episode), self._seq))
        self._status(True, [], {"active": bool(step.active), "gates": {k: float(v) for k, v in step.gates.items()},
                                "grasped": {k: bool(v) for k, v in step.grasped.items()},
                                "proc_ms": (time.perf_counter() - t0) * 1e3})


def main(argv=None) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=argv)
    node = None
    try:
        node = PourNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

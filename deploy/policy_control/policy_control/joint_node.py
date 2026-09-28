"""joint_node: rclpy shell of the joint-family chain (obs -> policy -> decoder) for ONE arm, in ONE process.

The policy commands arm joints (increment on the previous target) and hand joints (absolute) directly — no
fabric. All logic lives in ROS-free modules (joint_chain.JointChain, joint_node_core); this file only wires IO.

  in   robot yaml sources of the contract side: arm/ee joint_state, object pose (base frame), via sources.SourceSet
  out  /policy_control/joint_target      JointState canonical names, arm + hand (+ welded joints), frame_id '<episode>:<seq>'
       /policy_control/{obs,action}      Float64MultiArray (logging)
       /policy_control/episode           latched JSON (this node is the episode master)
       /policy_control/status/joint_node JSON
  srv  /policy_control/episode/{reset,start,stop,abort}  std_srvs/Trigger

pd_node runs next to it with a control-only DeployContract of the deploy asset (it never reads the joint contract).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import _paths  # noqa: F401,E402
from policy_control import codec  # noqa: E402
from policy_control.chain import EpisodeEvent  # noqa: E402
from policy_control.episode_master import EpisodeBook  # noqa: E402
from policy_control.joint_chain import JointChain, JointChainError  # noqa: E402
from policy_control.joint_contract import JointContract, JointContractError, load_contract  # noqa: E402
from policy_control.joint_decoder import JointDecodeError  # noqa: E402
from policy_control.joint_node_core import (  # noqa: E402
    JointNodeError, joint_target_arrays, measure_from_state, start_refusals,
)
from policy_control.joint_obs import JointObsError  # noqa: E402
from policy_control.sources import RobotCfgError, SourceSet, load_robot_cfg, select_side  # noqa: E402

NS = "/policy_control"
NODE = "joint_node"
EVENTS = ("reset", "start", "stop", "abort")
_HANDLED = (JointNodeError, JointChainError, JointContractError, JointDecodeError, JointObsError, codec.CodecError,
            RobotCfgError, ValueError)


def contract_home(c: JointContract) -> dict:
    return dict(zip(c.arm_joints, (float(v) for v in c.arm_reset)))


def build_fk(c: JointContract):
    """관측 FK 는 **학습 자산** URDF(용접 관절 그대로)로 푼다 — 학습 기하 그대로."""
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK

    if c.train_asset not in ASSETS:
        raise JointNodeError(f"train asset {c.train_asset!r} is not in contract_assets.ASSETS")
    return UrdfChainFK(ASSETS[c.train_asset].urdf, c.arm_joints, c.hand_joints, c.palm_body, c.tip_bodies)


try:
    from rclpy.node import Node
except ImportError:
    Node = object                        # type: ignore[misc,assignment]


class JointNode(Node):
    def __init__(self, *, policy=None, fk=None, **kw) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String
        from std_srvs.srv import Trigger

        super().__init__(NODE, **kw)
        for name, default in (("contract", ""), ("robot", ""), ("device", "cuda:0"), ("reset_tol_rad", 0.15),
                              ("max_gap_ticks", 3), ("goal_offset", [0.0, 0.0, 0.0]), ("use_goal_offset", False),
                              ("publish_target", True)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        cpath, rpath = Path(str(p("contract"))), Path(str(p("robot")))
        if not cpath.is_file() or not rpath.is_file():
            raise JointNodeError(f"parameters 'contract' and 'robot' must be existing files (got {cpath!r}, {rpath!r})")
        self.contract = load_contract(cpath)
        self.device = str(p("device"))
        self.reset_tol, self.max_gap = float(p("reset_tol_rad")), int(p("max_gap_ticks"))
        self.goal_offset = [float(v) for v in p("goal_offset")] if bool(p("use_goal_offset")) else None
        self._publish, self._policy = bool(p("publish_target")), policy
        cfg = select_side(load_robot_cfg(rpath), self.contract.side)
        self.src = SourceSet(cfg)
        self.arm_names = tuple(cfg.sources["arm"].joints)
        self.fk = fk if fk is not None else build_fk(self.contract)
        self.chain: JointChain | None = None
        self.book = EpisodeBook(contract_home(self.contract))
        self._seq, self._gap, self._errors = 0, 0, {}

        chain_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._String = String
        msgs = {"joint_state": JointState, "float_array": Float64MultiArray, "pose": PoseStamped}
        self._pub_target = self.create_publisher(JointState, f"{NS}/joint_target", chain_qos)
        self._pub_obs = self.create_publisher(Float64MultiArray, f"{NS}/obs", chain_qos)
        self._pub_action = self.create_publisher(Float64MultiArray, f"{NS}/action", chain_qos)
        self._pub_episode = self.create_publisher(String, f"{NS}/episode", latched)
        self._pub_status = self.create_publisher(String, f"{NS}/status/{NODE}", QoSProfile(depth=10))
        for src in cfg.sources.values():
            if src.type in msgs and (src.role or src.name) in ("arm", "ee", "object"):
                self.create_subscription(msgs[src.type], src.topic, self._source_cb(src), qos_profile_sensor_data)
        for name in EVENTS:
            self.create_service(Trigger, f"{NS}/episode/{name}", getattr(self, f"_srv_{name}"))
        self.create_timer(1.0 / float(self.contract.policy_hz), self._on_tick)
        order = "ASSUMED" if self.contract.assumed_order else "measured"
        self.get_logger().info(f"joint_node up · {self.contract.task} · {self.contract.side} · obs {self.contract.obs_dim} "
                               f"act {self.contract.action_dim} · {self.contract.policy_hz:.1f} Hz · hand obs order {order}")

    # ---------------------------------------------------------------- inputs
    def _note(self, key: str, exc) -> None:
        self._errors[key] = str(exc)
        self.get_logger().warning(f"{key}: {exc}", throttle_duration_sec=1.0)

    def _source_cb(self, src):
        def cb(msg) -> None:
            try:
                now = time.monotonic()
                if src.type == "joint_state":
                    self.src.update_from_joint_state(src.name, codec.decode_joint_state(msg), now)
                elif src.type == "pose":
                    self.src.update_from_pose(src.name, codec.decode_pose(msg), now)
                else:
                    self.src.update_from_float_array(src.name, codec.decode_float_array(msg), now)
                self._errors.pop(src.name, None)
            except _HANDLED as exc:
                self._note(src.name, exc)
        return cb

    def _measure(self):
        return measure_from_state(self.contract, self.src.snapshot(time.monotonic()), self.arm_names)

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
            if self.chain is None:
                if self._policy is None:
                    from policy_control.joint_policy import JointPolicy
                    self._policy = JointPolicy(self.contract, self.device)
                self.chain = JointChain(self.contract, self._policy, self.fk)
            goal = self.chain.reset(m.obj, self.goal_offset)
        except _HANDLED as exc:
            return self._reply(res, False, [f"reset: {exc}"])
        self._seq, self._gap = 0, 0
        event, _ = self.book.reset()
        self._emit(event)
        return self._reply(res, True, [f"goal {[round(float(v), 3) for v in goal.pos]}"])

    def _srv_start(self, _req, res):
        if self.chain is None:
            return self._reply(res, False, ["start: reset first"])
        try:
            refusals = start_refusals(self.contract, self._measure(), self.reset_tol)
        except _HANDLED as exc:
            return self._reply(res, False, [f"start: {exc}"])
        if refusals:
            return self._reply(res, False, refusals)
        event, reasons = self.book.start()
        if event is None:
            return self._reply(res, False, reasons)
        self._emit(event)
        return self._reply(res, True, [])

    def _end(self, kind: str, reason: str) -> None:
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
        body = {"node": NODE, "phase": self.book.phase, "episode": self.book.episode, "seq": self._seq,
                "ok": bool(ok), "reasons": [str(r) for r in reasons], "publish_target": self._publish,
                "hand_obs_order": "assumed" if self.contract.assumed_order else "measured",
                "t_pub_ns": self.get_clock().now().nanoseconds, **(extra or {})}
        self._pub_status.publish(self._String(data=json.dumps(body)))

    def _on_tick(self) -> None:
        if self.book.phase != "running" or self.chain is None:
            try:
                self._measure()
                self._status(True, list(self._errors.values()))
            except _HANDLED as exc:
                self._status(False, [*self._errors.values(), str(exc)])
            return
        t0 = time.perf_counter()
        try:
            step = self.chain.step(self._measure())
            names, q, qd = joint_target_arrays(self.contract, step)
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
        if self._publish:
            self._pub_target.publish(codec.encode_joint_target(names, q, qd, str(self.book.episode), self._seq))
        self._status(True, [], {"proc_ms": (time.perf_counter() - t0) * 1e3,
                                "goal": [round(float(v), 4) for v in step.goal.pos]})


def main(argv=None) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=argv)
    node = None
    try:
        node = JointNode()
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

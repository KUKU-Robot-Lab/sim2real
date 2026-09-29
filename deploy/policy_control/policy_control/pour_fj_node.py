"""pour_fj_node: RH56F1 양팔 붓기 정책(hdgp open-rh_b_pour_fj) — 관측 → 정책 → 디코더를 한 프로세스에서, 양팔을 한 번에.

09.29 사용자: pour_fj 양손 정책이 곧 나온다 — 미리 준비. 로직은 policy_control/pour_fj.py(순수)에 있고 여기는 배선만.

  in   robot yaml(양팔, rh56f1_bi_*): 팔 /joint_states · 손 /hand_<side>/joint_states · 촉각 /hand_<side>/tip_forces
       컵 두 개 PoseStamped(base): 파라미터 cup_src_topic · cup_rcv_topic
  out  /policy_control/joint_target  JointState — 오른팔 7 · 오른손 6 · 왼팔 7 · 왼손 6 (pd_node 가 팔마다 자기 것만 걸러 쓴다)
       /policy_control/{obs,action} · /policy_control/episode(latched) · /policy_control/status/pour_fj_node
  srv  /policy_control/episode/{reset,start,stop,abort}

학습과 같은 순서: 관측(지금 측정 + 직전 q* + 직전 행동) → 행동 → q* 갱신 → 발행. 처음 hold_steps 스텝은 팔 q* = 시작 자세.
실기에서 붓기 성공(비드)은 잴 수 없다 — 에피소드는 시간(episode_s) · 운영자 stop 으로 끝난다.
접촉 동결: 학습은 손가락 중간 · 원위 마디의 컵 접촉 > 1 N 이면 닫기를 멈춘다. 실기는 손끝 촉각 > 같은 임계로 대신하고,
실제 멈춤은 RH56F1 펌웨어 힘 제한이 한다(forceSet — 실측 전).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import _paths  # noqa: F401,E402
from policy_control import codec  # noqa: E402
from policy_control import pour_fj as F  # noqa: E402
from policy_control.episode_master import EpisodeBook  # noqa: E402
from policy_control.joint_obs import quat_to_matrix  # noqa: E402
from policy_control.sources import SourceSet, load_robot_cfg, select_side  # noqa: E402

NS = "/policy_control"
NODE = "pour_fj_node"
EVENTS = ("reset", "start", "stop", "abort")
CUP_STALE_S = 0.5
START_TILT_MAX_DEG = 15.0


class PourFjNodeError(RuntimeError):
    pass


# ---------------------------------------------------------------- ROS 없는 절반
def side_meas(c: F.FjContract, role: str, st, arm_names, fk, cup) -> F.FjSideMeas:
    """sources.RobotState(한 팔) + 컵 (pos, quat) → FjSideMeas. 결손 · stale 이면 PourFjNodeError."""
    s = c.sides[role]
    bad = set(st.stale) | set(st.missing)
    for need in ("arm", "ee"):
        if need in bad:
            raise PourFjNodeError(f"{s.side}: source {need!r} is {'stale' if need in st.stale else 'missing'}")
    if st.arm_q is None or st.arm_qd is None or st.ee_q is None:
        raise PourFjNodeError(f"{s.side}: 팔 위치 · 속도 · 손 위치가 필요하다")
    arm = dict(zip(arm_names, st.arm_q))
    arm_d = dict(zip(arm_names, st.arm_qd))
    hand = dict(zip(st.ee_names, st.ee_q))
    missing = [j for j in list(s.arm_joints) + list(s.hand_joints) if j not in {**arm, **hand}]
    if missing:
        raise PourFjNodeError(f"{s.side}: 소스에 없는 관절 {missing}")
    if cup is None:
        raise PourFjNodeError(f"{role} 컵 자세가 없다(또는 {CUP_STALE_S} s 넘게 끊겼다)")
    arm_q = np.array([arm[j] for j in s.arm_joints], float)
    hand_q = np.array([hand[j] for j in s.hand_joints], float)
    pose = fk.palm_pose(arm_q, hand_q)
    tact = np.zeros(5) if st.tip_force is None or "tip_force" in bad else np.asarray(st.tip_force, float).reshape(5, -1)[:, 0]
    return F.FjSideMeas(arm_q=arm_q, arm_qd=np.array([arm_d[j] for j in s.arm_joints], float),
                        hand_q={j: float(hand[j]) for j in s.hand_joints}, palm_pos=np.asarray(pose.palm_pos, float),
                        palm_R=quat_to_matrix(pose.palm_quat), tips=np.asarray(pose.tips, float).reshape(5, 3),
                        cup_pos=np.asarray(cup[0], float), cup_quat=np.asarray(cup[1], float), tactile_n=tact)


def start_refusals(c: F.FjContract, meas: dict, tol: float) -> list:
    """팔이 학습 시작 자세 tol 안 · 컵 두 개가 서 있는가(빈 목록 = 시작)."""
    out = []
    for r in F.ROLES:
        err = float(np.abs(meas[r].arm_q - np.asarray(c.sides[r].arm_home)).max())
        if err > tol:
            out.append(f"{c.sides[r].side} arm is {err:.3f} rad from the training start pose (tol {tol})")
        tilt = float(np.degrees(np.arccos(np.clip(quat_to_matrix(meas[r].cup_quat)[2, 2], -1.0, 1.0))))
        if tilt > START_TILT_MAX_DEG:
            out.append(f"{r} cup tilt {tilt:.1f} deg (> {START_TILT_MAX_DEG:.0f}) — a standing cup is expected")
    return out


def target_arrays(c: F.FjContract, targets: dict) -> tuple[tuple, np.ndarray, np.ndarray]:
    names, q = [], []
    for r in F.ROLES:
        s = c.sides[r]
        names += list(s.arm_joints) + list(s.hand_joints)
        q += list(targets[r][0]) + list(targets[r][1])
    arr = np.asarray(q, float)
    if not np.all(np.isfinite(arr)):
        raise PourFjNodeError("non-finite joint target")
    return tuple(names), arr, np.zeros(arr.size)


class PourFjChain:
    """관측 → 정책 → 디코더 한 스텝. 순수(정책 · FK 는 주입)."""

    def __init__(self, c: F.FjContract, policy) -> None:
        self.c, self.policy = c, policy
        self.dec = F.FjDecoder(c)
        self.prev = np.zeros(c.action_dim)
        self.step_i = 0

    def reset(self) -> None:
        self.dec.reset()
        self.prev = np.zeros(self.c.action_dim)
        self.step_i = 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def step(self, meas: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        obs = F.build_obs(self.c, meas, self.dec, self.prev)
        a = np.clip(np.asarray(self.policy.forward(obs), float), -self.c.action_clip, self.c.action_clip)
        touch = {r: list(np.asarray(meas[r].tactile_n) > self.c.freeze_threshold_n) for r in F.ROLES}
        targets = self.dec.step(a, active=self.step_i >= self.c.hold_steps, touch=touch)
        self.prev = np.clip(a, -1.0, 1.0)
        self.step_i += 1
        return obs, a, targets


# ---------------------------------------------------------------- ROS
try:
    from rclpy.node import Node
except ImportError:
    Node = object                        # type: ignore[misc,assignment]


class PourFjNode(Node):
    def __init__(self, *, policy=None, **kw) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String
        from std_srvs.srv import Trigger
        from policy_control.contract_assets import ASSETS
        from policy_control.fk_numpy import UrdfChainFK

        super().__init__(NODE, **kw)
        for name, default in (("contract", ""), ("robot", ""), ("device", "cpu"), ("reset_tol_rad", 0.15),
                              ("max_gap_ticks", 3), ("publish_target", True), ("max_episode_s", -1.0),
                              ("cup_src_topic", "/objects/cup_src/pose"), ("cup_rcv_topic", "/objects/cup_rcv/pose")):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        cpath, rpath = Path(str(p("contract"))), Path(str(p("robot")))
        if not cpath.is_file() or not rpath.is_file():
            raise PourFjNodeError(f"parameters 'contract' and 'robot' must be existing files (got {cpath}, {rpath})")
        self.contract = F.load_contract(cpath)
        cfg = load_robot_cfg(rpath)
        self.srcs = {r: SourceSet(select_side(cfg, self.contract.sides[r].side)) for r in F.ROLES}
        self.arm_names = {r: tuple(select_side(cfg, self.contract.sides[r].side).sources["arm"].joints) for r in F.ROLES}
        urdf = ASSETS[self.contract.asset].urdf
        self.fk = {r: UrdfChainFK(urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
                   for r, s in self.contract.sides.items()}
        self.device, self._policy = str(p("device")), policy
        self.reset_tol, self.max_gap, self._publish = float(p("reset_tol_rad")), int(p("max_gap_ticks")), bool(p("publish_target"))
        m = float(p("max_episode_s"))
        self.max_s = self.contract.episode_s if m < 0 else m
        self.chain: PourFjChain | None = None
        self.book = EpisodeBook({})
        self.cups: dict = {}
        self._seq, self._gap, self._errors, self._t_start = 0, 0, {}, 0.0

        chain_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._String = String
        self._pub_target = self.create_publisher(JointState, f"{NS}/joint_target", chain_qos)
        self._pub_obs = self.create_publisher(Float64MultiArray, f"{NS}/obs", chain_qos)
        self._pub_action = self.create_publisher(Float64MultiArray, f"{NS}/action", chain_qos)
        self._pub_episode = self.create_publisher(String, f"{NS}/episode", latched)
        self._pub_status = self.create_publisher(String, f"{NS}/status/{NODE}", QoSProfile(depth=10))
        msgs = {"joint_state": JointState, "float_array": Float64MultiArray}
        for r in F.ROLES:
            for src in select_side(cfg, self.contract.sides[r].side).sources.values():
                if src.type in msgs and (src.role or src.name) in ("arm", "ee", "tip_force"):
                    self.create_subscription(msgs[src.type], src.topic, self._source_cb(r, src), qos_profile_sensor_data)
        for role, topic in (("src", str(p("cup_src_topic"))), ("rcv", str(p("cup_rcv_topic")))):
            self.create_subscription(PoseStamped, topic, self._cup_cb(role), qos_profile_sensor_data)
        for name in EVENTS:
            self.create_service(Trigger, f"{NS}/episode/{name}", getattr(self, f"_srv_{name}"))
        self.create_timer(1.0 / float(self.contract.policy_hz), self._on_tick)
        self.get_logger().info(f"pour_fj_node up · obs {self.contract.obs_dim} act {self.contract.action_dim} · "
                               f"{self.contract.policy_hz:.0f} Hz · arm {self.contract.arm_mode} · hand obs order "
                               f"{self.contract.hand_obs_order_source.split(':')[0]}")

    # ---------------------------------------------------------------- inputs
    def _source_cb(self, role, src):
        def cb(msg) -> None:
            try:
                now = time.monotonic()
                if src.type == "joint_state":
                    self.srcs[role].update_from_joint_state(src.name, codec.decode_joint_state(msg), now)
                else:
                    self.srcs[role].update_from_float_array(src.name, codec.decode_float_array(msg), now)
                self._errors.pop(f"{role}.{src.name}", None)
            except (codec.CodecError, ValueError) as exc:
                self._errors[f"{role}.{src.name}"] = str(exc)
        return cb

    def _cup_cb(self, role):
        def cb(msg) -> None:
            try:
                s = codec.decode_pose(msg)
                self.cups[role] = (time.monotonic(), s.pos, s.quat)
            except (codec.CodecError, ValueError) as exc:
                self._errors[f"cup_{role}"] = str(exc)
        return cb

    def _cup(self, role):
        got = self.cups.get(role)
        return None if got is None or time.monotonic() - got[0] > CUP_STALE_S else (got[1], got[2])

    def _measure(self) -> dict:
        now = time.monotonic()
        return {r: side_meas(self.contract, r, self.srcs[r].snapshot(now), self.arm_names[r], self.fk[r], self._cup(r))
                for r in F.ROLES}

    # ---------------------------------------------------------------- services
    def _reply(self, res, ok: bool, reasons):
        res.success = bool(ok)
        res.message = json.dumps({"ok": bool(ok), "reasons": [str(r) for r in reasons]})
        return res

    def _emit(self, event) -> None:
        body = {**event.as_dict(), "t_ns": self.get_clock().now().nanoseconds}
        self._pub_episode.publish(self._String(data=json.dumps(body)))
        self.get_logger().info(f"episode {event.episode} {event.event} {list(event.reasons)}")

    def _srv_reset(self, _req, res):
        try:
            self._measure()
            if self.chain is None:
                if self._policy is None:
                    from policy_control.joint_policy import JointPolicy
                    self._policy = JointPolicy(self.contract, self.device)
                self.chain = PourFjChain(self.contract, self._policy)
            self.chain.reset()
        except (PourFjNodeError, F.PourFjError, ValueError) as exc:
            return self._reply(res, False, [f"reset: {exc}"])
        self._seq, self._gap = 0, 0
        event, _ = self.book.reset()
        self._emit(event)
        return self._reply(res, True, [])

    def _srv_start(self, _req, res):
        if self.chain is None:
            return self._reply(res, False, ["start: reset first"])
        try:
            refusals = start_refusals(self.contract, self._measure(), self.reset_tol)
        except (PourFjNodeError, F.PourFjError, ValueError) as exc:
            return self._reply(res, False, [f"start: {exc}"])
        if refusals:
            return self._reply(res, False, refusals)
        event, reasons = self.book.start()
        if event is None:
            return self._reply(res, False, reasons)
        self._t_start = time.monotonic()
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
        body = {"node": NODE, "phase": self.book.phase, "episode": self.book.episode, "seq": self._seq, "ok": bool(ok),
                "reasons": [str(r) for r in reasons], "publish_target": self._publish,
                "hand_obs_order": self.contract.hand_obs_order_source.split(":")[0],
                "t_pub_ns": self.get_clock().now().nanoseconds, **(extra or {})}
        self._pub_status.publish(self._String(data=json.dumps(body)))

    def _on_tick(self) -> None:
        if self.book.phase != "running" or self.chain is None:
            try:
                self._measure()
                self._status(True, list(self._errors.values()))
            except (PourFjNodeError, F.PourFjError, ValueError) as exc:
                self._status(False, [*self._errors.values(), str(exc)])
            return
        t0 = time.perf_counter()
        try:
            obs, action, targets = self.chain.step(self._measure())
            names, q, qd = target_arrays(self.contract, targets)
        except (PourFjNodeError, F.PourFjError, ValueError) as exc:
            self._gap += 1
            self._status(False, [str(exc)], {"gap": self._gap})
            if self._gap > self.max_gap:
                self._end("abort", f"{self._gap} ticks without a valid step: {exc}")
            return
        self._gap = 0
        self._seq += 1
        self._pub_obs.publish(codec.encode_float_array(obs, ["obs"], [obs.size], self._seq))
        self._pub_action.publish(codec.encode_action(action, self._seq))
        if self._publish:
            self._pub_target.publish(codec.encode_joint_target(names, q, qd, str(self.book.episode), self._seq))
        self._status(True, [], {"proc_ms": (time.perf_counter() - t0) * 1e3, "step": self.chain.step_i,
                                "hold": self.chain.step_i <= self.contract.hold_steps})
        if self.max_s > 0 and time.monotonic() - self._t_start >= self.max_s:
            self._end("stop", f"episode time >= {self.max_s:.1f} s (학습 episode_length_s)")


def main(argv=None) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=argv)
    node = None
    try:
        node = PourFjNode()
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

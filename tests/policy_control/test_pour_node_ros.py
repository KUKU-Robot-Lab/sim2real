"""pour_node over ROS (isolated domain, no hardware): golden-trace measurements in, joint_target out.

Policy is the recorded trace action table and the fabric is a fake, so neither torch nor CUDA is needed.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from policy_control.pour_chain import RecordedPolicy
from policy_control.pour_contract import save_contract
from pour_trace_util import contract, make_fk, trace

SIM2REAL = Path(__file__).resolve().parents[2]
ROBOT = SIM2REAL / "policy_control/config/robots/dg5f_m_bi_real.yaml"
NS = "/policy_control"
ENV = 0


class _JT:
    def __init__(self, q):
        self.q_arm, self.qd_arm = np.asarray(q, float), np.zeros(len(q))


class _Fabric:
    """Holds the arm at home; records hold flags."""

    def __init__(self, c):
        self.c, self.holds, self.home = c, [], None

    def reset(self, home):
        self.home = {r: np.asarray(v, float) for r, v in home.items()}

    def step(self, palm, hand, hold=False):
        self.holds.append(bool(hold))
        return {s.role: _JT(self.home[s.role][:len(s.arm_joints)]) for s in self.c.sides}


class _Spinner:
    def __init__(self, context, *nodes):
        from rclpy.executors import SingleThreadedExecutor

        self.executor = SingleThreadedExecutor(context=context)
        for n in nodes:
            self.executor.add_node(n)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.executor.spin_once(timeout_sec=0.02)

    def close(self):
        self._stop.set()
        self.thread.join(timeout=2.0)
        self.executor.shutdown()


@pytest.fixture
def rig(ros, tmp_path):
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Float64MultiArray, String
    from std_srvs.srv import Trigger

    from policy_control import pour_node as P

    c = contract()
    z, meta = trace()
    cpath = tmp_path / "pour_contract.json"
    save_contract(c, cpath)
    fab = _Fabric(c)
    node = P.PourNode(
        context=ros, policy=RecordedPolicy(z["actions"][:, ENV]), fabric=fab,
        fks={s.role: make_fk(s) for s in c.sides},
        parameter_overrides=[Parameter("contract", value=str(cpath)), Parameter("robot", value=str(ROBOT)),
                             Parameter("src_cup_topic", value="/test/src_cup"),
                             Parameter("rcv_cup_topic", value="/test/rcv_cup"),
                             Parameter("stale_sec", value=1.0), Parameter("reset_tol_rad", value=0.2)])
    peer = Node("pour_peer", context=ros)
    got = {"target": [], "status": [], "obs": []}
    peer.create_subscription(JointState, f"{NS}/joint_target", got["target"].append, 10)
    peer.create_subscription(Float64MultiArray, f"{NS}/obs", got["obs"].append, 10)
    peer.create_subscription(String, f"{NS}/status/{P.NODE}", lambda m: got["status"].append(json.loads(m.data)), 10)
    pubs = {}

    def pub(kind, topic):
        if topic not in pubs:
            pubs[topic] = peer.create_publisher(kind, topic, qos_profile_sensor_data)
        return pubs[topic]

    qpos = dict(zip(meta["joint_names"], z["joint_pos"][0, ENV]))
    qvel = dict(zip(meta["joint_names"], z["joint_vel"][0, ENV]))

    def publish_inputs(with_rcv_cup=True):
        now = peer.get_clock().now().to_msg()
        for s in c.sides:
            srcs = node.sets[s.role]
            for key in ("arm", "ee"):
                cfg = srcs.cfg.sources[key]
                names, signs = srcs._msg_names[key]
                m = JointState()
                m.header.stamp = now
                m.name = list(names)
                m.position = [float(qpos[j] * g) for j, g in zip(cfg.joints, signs)]
                m.velocity = [float(qvel[j] * g) for j, g in zip(cfg.joints, signs)]
                pub(JointState, cfg.topic).publish(m)
            tf = srcs.cfg.sources["tip_force"]
            pub(Float64MultiArray, tf.topic).publish(Float64MultiArray(data=[0.0] * (len(tf.tips) * tf.axes)))
            if s.role == "rcv" and not with_rcv_cup:
                continue
            p = PoseStamped()
            p.header.stamp, p.header.frame_id = now, "base_link"
            pos, quat = z[f"{s.role}_cup_pos"][0, ENV], z[f"{s.role}_cup_quat"][0, ENV]
            p.pose.position.x, p.pose.position.y, p.pose.position.z = (float(v) for v in pos)
            (p.pose.orientation.w, p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z) = (
                float(v) for v in quat)
            pub(PoseStamped, f"/test/{s.role}_cup").publish(p)
        pubs.setdefault("fill", peer.create_publisher(Float64, f"{NS}/pour/fill_level", 10)).publish(
            Float64(data=float(z["fill_level"][0, ENV])))

    clients = {n: peer.create_client(Trigger, f"{NS}/episode/{n}") for n in P.EVENTS}
    spin = _Spinner(ros, node, peer)

    def call(name, timeout=5.0):
        assert clients[name].wait_for_service(timeout_sec=timeout), name
        fut = clients[name].call_async(Trigger.Request())
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        assert fut.done(), name
        return {"success": fut.result().success, **json.loads(fut.result().message)}

    def pump(seconds, **kw):
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            publish_inputs(**kw)
            time.sleep(0.01)

    yield {"c": c, "call": call, "pump": pump, "got": got, "fab": fab, "node": node}
    spin.close()
    node.destroy_node()
    peer.destroy_node()


def test_reset_refused_without_measurements(rig):
    r = rig["call"]("reset")
    assert not r["success"] and r["reasons"]


def test_reset_start_publishes_both_arms_and_hands(rig):
    c = rig["c"]
    rig["pump"](0.5)
    r = rig["call"]("reset")
    assert r["success"], r
    assert rig["call"]("start")["success"]
    rig["pump"](1.0)
    assert rig["call"]("stop")["success"]
    msgs = rig["got"]["target"]
    assert len(msgs) >= 10, len(msgs)
    want = [j for s in c.sides for j in list(s.arm_joints) + list(s.hand_joints)]
    m = msgs[-1]
    assert list(m.name) == want and len(m.position) == 54 and np.all(np.isfinite(m.position))
    ep, _, seq = m.header.frame_id.partition(":")
    assert ep == "1" and int(seq) >= 9
    assert len(rig["got"]["obs"][-1].data) == c.obs_dim
    assert rig["fab"].holds and rig["fab"].home is not None


def test_missing_rcv_cup_never_emits_targets(rig):
    rig["pump"](0.5, with_rcv_cup=False)
    r = rig["call"]("reset")
    assert not r["success"] and any("rcv" in x for x in r["reasons"]), r
    assert rig["got"]["target"] == []

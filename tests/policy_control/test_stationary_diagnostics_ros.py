"""Exercise the real ROS shell on the isolated test domain, without motor publication."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.ros


def test_right_pd_publishes_same_tick_diagnostics(ros):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
    from policy_control.pd_node import PdNode
    from policy_control.codec import JointSample

    root = Path(__file__).resolve().parents[2]
    params = {
        "contract": root
        / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json",
        "robot": root / "deploy/policy_control/config/robots/rh56f1_right_fake.yaml",
        "pd_config": root / "deploy/policy_control/config/pd_rh56f1_stationary.yaml",
        "sides": "right",
    }
    node = PdNode(
        context=ros,
        parameter_overrides=[Parameter(k, value=str(v)) for k, v in params.items()],
    )
    observer = rclpy.create_node("stationary_diagnostic_observer", context=ros)
    executor = SingleThreadedExecutor(context=ros)
    executor.add_node(observer)
    seen = []
    observer.create_subscription(
        String,
        "/policy_control/pd_right/diagnostics",
        lambda msg: seen.append(json.loads(msg.data)),
        10,
    )
    try:
        unit = node.units["right"]
        assert not unit.execute
        params_xml = "".join(
            f'<param name="{kind}{i + 1}">{v}</param>'
            for kind, values in [("kp", unit.kp), ("kd", unit.kd)]
            for i, v in enumerate(values)
        )
        unit.on_robot_description(
            '<robot><ros2_control name="openarm_right_hardware_interface"><hardware>'
            + params_xml
            + "</hardware></ros2_control></robot>"
        )
        q = unit.home_arm
        unit.stage.engage(q)
        for seq in range(100):
            now = time.monotonic()
            unit.on_joint_state(
                ["arm"],
                JointSample(
                    tuple(f"openarm_right_joint{i}" for i in range(1, 8)),
                    q,
                    np.zeros(7),
                    np.zeros(7),
                    321.0,
                ),
                now,
            )
            n = len(unit.hand_joints)
            unit.on_joint_state(
                ["ee"],
                JointSample(
                    tuple(unit.hand_joints), np.zeros(n), np.zeros(n), None, 321.0
                ),
                now,
            )
            unit.on_temperatures(
                {f"openarm_right_joint{i}": 30.0 for i in range(1, 8)}, now
            )
            unit.on_contact(False, now)
            unit.take_target(
                JointSample(tuple(unit.arm_joints), q, np.zeros(7), None, 321.0),
                seq,
                now,
            )
            node._tick(now, time.perf_counter())
            executor.spin_once(timeout_sec=0.01)
            if seen:
                break
        assert seen
        row = seen[-1]
        assert row["arm_source_stamp"] == 321.0
        assert row["motor_position"] == pytest.approx(q)
        assert row["torque_integral"] == [0.0] * 7
        assert row["contact"] is False
    finally:
        executor.shutdown()
        observer.destroy_node()
        node.destroy_node()

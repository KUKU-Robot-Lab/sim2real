#!/usr/bin/env python3
"""Conservative hand-contact gate after operator-confirmed empty free space.

Publishes no robot targets. Unknown/stale is sent as contact=True. This is a
hand sensor guard, not certification of whole-arm collision detection.
Thresholds: touch_probe's 20 g change and the existing firmware's 20 raw counts.
Baseline is fixed for this process; a trip is latched until process restart.
"""
import argparse
import json
import time

import numpy as np


def assess(force, baseline, touch, status):
    force, baseline, touch = [np.asarray(v, float) for v in (force, baseline, touch)]
    if force.shape != (6,) or baseline.shape != (6,) or touch.shape != (10,):
        return "invalid_sensor_shape"
    if not all(np.isfinite(v).all() for v in (force, baseline, touch)):
        return "nonfinite_sensor"
    errors = status.get("error")
    if (status.get("op") is not True or status.get("node_ok") is not True
            or status.get("stopping") is not False or status.get("al_code") != 0
            or not isinstance(errors, list) or len(errors) != 6 or any(errors)):
        return "hand_driver_fault"
    if np.max(np.abs(force - baseline)) >= 20:
        return "joint_force_change_20g"
    if np.max(np.abs(touch)) >= 20:
        return "finger_touch_20_raw_counts"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=("left", "right"), required=True)
    p.add_argument("--operator-confirmed-free", action="store_true", required=True)
    args = p.parse_args()
    import rclpy
    from std_msgs.msg import Bool, Float64MultiArray, String
    from rh56f1_interfaces.msg import TouchData1
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = rclpy.create_node(f"stationary_contact_guard_{args.side}")
    ns = f"/hand_{args.side}"
    latest, received, samples = {}, {}, []
    baseline, latched = None, None
    started = time.monotonic()
    def update(key, value):
        latest[key], received[key] = value, time.monotonic()
        if key == "force" and baseline is None:
            samples.append(value)
    node.create_subscription(Float64MultiArray, ns + "/joint_forces",
                             lambda m: update("force", list(m.data)), qos_profile_sensor_data)
    node.create_subscription(TouchData1, ns + "/touch_data",
                             lambda m: update("touch", list(m.finger_forces) + list(m.finger_tangentials)),
                             qos_profile_sensor_data)
    def on_status(msg):
        try:
            update("status", json.loads(msg.data))
        except (ValueError, TypeError):
            update("status", {})
    node.create_subscription(String, ns + "/ecat_status", on_status, 10)
    out = node.create_publisher(Bool, f"/policy_control/pd_{args.side}/contact", 10)
    diag = node.create_publisher(String, f"/policy_control/pd_{args.side}/contact_guard", 10)
    def tick():
        nonlocal baseline, latched
        now = time.monotonic()
        stale = any(now - received.get(k, 0) > age for k, age in
                    (("force", .1), ("touch", .1), ("status", 2.0)))
        reason = "sensor_stale_or_missing" if stale else None
        if not stale and baseline is None and now - started >= 3:
            data = np.asarray(samples[-500:], float)
            if data.shape[0] < 100 or data.shape[1:] != (6,) or not np.isfinite(data).all():
                latched = "invalid_baseline"
            elif np.max(np.std(data, axis=0)) > 5 or np.max(np.abs(np.mean(data, axis=0))) > 400:
                latched = "unstable_or_loaded_baseline"
            else:
                baseline = np.mean(data, axis=0)
                node.get_logger().info("fixed free-space force baseline: " + json.dumps(baseline.tolist()))
        if baseline is not None:
            if not stale:
                reason = assess(latest["force"], baseline, latest["touch"], latest["status"])
            if reason:
                latched = reason
        else:
            reason = reason or "baseline_pending"
        reason = latched or reason
        out.publish(Bool(data=reason is not None))
        diag.publish(String(data=json.dumps({"contact_or_unknown": reason is not None,
            "reason": reason, "operator_confirmed_free": True,
            "scope": "hand_sensors_only", "baseline_g": None if baseline is None else baseline.tolist()})))
    node.create_timer(.02, tick)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():  # SIGTERM 이 이미 context 를 닫았을 수 있다
            rclpy.shutdown()


if __name__ == "__main__":
    main()

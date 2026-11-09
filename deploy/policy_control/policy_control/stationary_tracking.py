"""RH56F1 stationary experiments: pure planning, timestamp matching and verdicts.

FK distances describe model tracking, never optical/physical accuracy.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

JOINTS_BY_SIDE = {
    "right": tuple(f"r_aj_{i}" for i in range(1, 8)),
    "left": tuple(f"l_aj_{i}" for i in range(1, 8)),
}
JOINTS = JOINTS_BY_SIDE["right"]
MATCH_S = 0.020
# 표본 간격 한도는 정렬 한도와 따로 둔다 — 120 Hz pd 가 한 틱 늦으면 20 ms 를 넘는다(11.09 실측 20.1 ms)
GAP_S = 0.050
# 2026-11-08 static-friction block (experiment doc "2026-11-08 실기 대기 목록").
FRICTION_STAGES = ("F1", "F2", "F3", "F4")


def arm_joint_names(side="right"):
    try:
        return JOINTS_BY_SIDE[side]
    except KeyError:
        raise ValueError(f"unsupported arm side: {side}") from None


def validate_live_diagnostics(row):
    if any(row.get(key) is not False for key in ("contact", "stale", "protected")):
        raise ValueError("contact/unknown/stale/protection")
    for key in ("temperature", "torque_feedback", "measured", "velocity"):
        values = np.asarray(row.get(key), dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"required finite seven-joint state missing: {key}")


def capture_pose(events, side="right"):
    """A pose capture never requires PD engagement or any target publication."""

    if side not in ("right", "left"):
        raise ValueError(f"unsupported arm side: {side}")

    requested_side = side
    prefix = "r" if requested_side == "right" else "l"
    joints = tuple(f"{prefix}_aj_{i}" for i in range(1, 8))

    # 11.09 F4: publishers stamp headers on different clocks (arm -0.8 s, hand -2 s
    # behind receive time), so a still-pose snapshot pairs messages by receive time.
    def stamp(e):
        return e["recv_wall_ns"] / 1e9

    arms = [
        e
        for e in events
        if e["topic"] in ("/joint_states", "/pd_state_broadcaster/joint_states")
    ]
    if not arms:
        raise ValueError("arm state missing")
    heads = [e for e in events if e["topic"] == "/head/joint_states"]
    head = max(heads, key=stamp) if heads else None
    arm = (
        min(arms, key=lambda e: abs(stamp(e) - stamp(head)))
        if head is not None
        else max(arms, key=stamp)
    )
    message = arm["message"]
    names = (
        joints
        if set(joints) <= set(message["name"])
        else [f"openarm_{requested_side}_joint{i}" for i in range(1, 8)]
    )
    q = named_values(message, names).tolist()
    scene = dict(zip(message["name"], message["position"]))
    for arm_side in ("right", "left"):
        for i in range(1, 8):
            source = f"openarm_{arm_side}_joint{i}"
            if source in scene:
                scene[f"{arm_side[0]}_aj_{i}"] = scene.pop(source)
    hand = None
    for hand_side in ("right", "left"):
        samples = [e for e in events if e["topic"] == f"/hand_{hand_side}/joint_states"]
        if samples:
            event = min(samples, key=lambda e: abs(stamp(e) - stamp(arm)))
            if abs(stamp(event) - stamp(arm)) <= MATCH_S:
                msg = event["message"]
                values = dict(zip(msg["name"], named_values(msg, msg["name"]).tolist()))
                scene.update(values)
                if hand_side == requested_side:
                    hand = values
    if head is not None:
        if abs(stamp(head) - stamp(arm)) <= MATCH_S:
            msg = head["message"]
            scene.update(dict(zip(msg["name"], named_values(msg, msg["name"]).tolist())))
    if hand is None:
        raise ValueError(f"{requested_side} hand state missing or unmatched >20ms")
    return {"q": q, "hand": hand, "scene": scene, "stamp": stamp(arm)}


def validate_stage(stage, kp, kd, base_kp, base_kd, compensation):
    from .stationary_compensation import CompensationConfig

    if stage not in ("A", "B", "C", "D", "E", "F", "G") + FRICTION_STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    c = CompensationConfig.from_mapping(compensation)
    kp, kd, base_kp, base_kd = [
        np.asarray(v, float) for v in (kp, kd, base_kp, base_kd)
    ]
    ratio = kp / base_kp
    if stage in ("A", "B", "C", "D") and c.enabled:
        raise ValueError("A-D require additional compensation OFF")
    if stage == "A" and (not np.allclose(kp, base_kp) or not np.allclose(kd, base_kd)):
        raise ValueError("A requires baseline gains")
    if stage == "B" and (
        np.sum(~np.isclose(ratio, 1)) > 1
        or not np.allclose(kd, base_kd)
        or not all(any(np.isclose(x, v) for v in (1.0, 1.25, 1.5, 2.0)) for x in ratio)
    ):
        raise ValueError("B changes only one Kp at 1/1.25/1.5/2; Kd fixed")
    if stage == "C" and not (
        np.allclose(kd, base_kd) or np.allclose(kd, base_kd * np.sqrt(ratio))
    ):
        raise ValueError("C requires base Kd or base Kd*sqrt(Kp ratio)")
    if stage == "E" and (not c.gravity_enabled or c.integral_enabled):
        raise ValueError(
            "E requires gravity correction, optional friction, no integral"
        )
    if stage == "F" and (
        not c.integral_enabled
        or not any(np.allclose(c.ki, kp / ti) for ti in (10.0, 5.0, 2.5))
    ):
        raise ValueError("F requires Ki=Kp/Ti for Ti=10/5/2.5s")
    if stage in FRICTION_STAGES and (
        not np.allclose(kp, base_kp) or not np.allclose(kd, base_kd)
    ):
        raise ValueError(f"{stage} requires baseline gains")
    if stage == "F1" and c.integral_enabled:
        raise ValueError("F1 compares static friction off/on with the integral OFF")
    if stage == "F2" and not (
        c.integral_enabled
        and c.static_enabled
        and np.allclose(c.integral_limit[:4], 1.0)
    ):
        raise ValueError("F2 needs static friction + integral with a 1.0 N·m j1~j4 limit")
    if stage in ("F3", "F4") and c.enabled:
        raise ValueError(f"{stage} measures uncompensated error: compensation OFF")
    # F2 sweeps the j1~j4 rate up to 2x the initial 0.2 N·m/s budget (user, 11.08).
    rate_cap = [0.4 if stage == "F2" else 0.2] * 4 + [0.05] * 3
    if np.any(np.asarray(c.integral_limit) > [1.0] * 4 + [0.3] * 3) or np.any(
        np.asarray(c.integral_rate) > rate_cap
    ):
        raise ValueError(
            "experiment integral limits/rates must not exceed initial budget"
        )
    return {"kp_multiplier": ratio.tolist(), "stage": stage}


def aligned_samples(events, side="right"):
    """Prefer same-tick diagnostics; old topics are joined by stamps within 20 ms.

    Never substitute motor commands for missing user targets. Legacy records do
    not establish contact/temperature completeness and therefore cannot pass.
    """
    if side not in JOINTS_BY_SIDE:
        raise ValueError(f"unsupported arm side: {side}")
    joints = JOINTS_BY_SIDE[side]
    diagnostics = [
        e for e in events if e.get("topic") == f"/policy_control/pd_{side}/diagnostics"
    ]
    if diagnostics:
        rows = [json.loads(e["message"]["data"]) for e in diagnostics]
        return rows, {
            "legacy": False,
            "matched": len(rows),
            "unmatched_fraction": 0.0,
            "max_matching_interval_s": MATCH_S,
        }

    def topic(name):
        return sorted([e for e in events if e.get("topic") == name], key=stamp)

    def stamp(e):
        return (e.get("header_stamp_ns") or e["recv_wall_ns"]) / 1e9

    applied = topic(f"/policy_control/pd_{side}/applied")
    arms = topic("/pd_state_broadcaster/joint_states") or topic("/joint_states")
    hands = topic(f"/hand_{side}/joint_states")
    users = topic("/policy_control/joint_target")
    statuses = topic(f"/policy_control/status/pd_{side}")
    query = [stamp(e) for e in applied]
    ai, ag = nearest_indices([stamp(e) for e in arms], query)
    hi, hg = nearest_indices([stamp(e) for e in hands], query)
    ui, _ = nearest_indices([stamp(e) for e in users], query)
    si, _ = nearest_indices([stamp(e) for e in statuses], query)
    rows = []
    for i, e in enumerate(applied):
        if ai[i] < 0 or hi[i] < 0:
            continue
        try:
            arm = arms[ai[i]]["message"]
            names = (
                joints
                if all(j in arm["name"] for j in joints)
                else [f"openarm_{side}_joint{j}" for j in range(1, 8)]
            )
            motor = named_values(e["message"], joints)
            hand = hands[hi[i]]["message"]
            status = (
                json.loads(statuses[si[i]]["message"]["data"]) if si[i] >= 0 else {}
            )
            internal = status.get("target") == "internal"
            rows.append(
                {
                    "stamp": query[i],
                    "joint_names": list(joints),
                    "user_target": named_values(
                        users[ui[i]]["message"], joints
                    ).tolist()
                    if ui[i] >= 0
                    else None,
                    "tracking_target": motor.tolist() if internal else None,
                    "tracking_target_known": status.get("target")
                    in ("internal", "external"),
                    "tracking_target_source": "internal_hold_motor_position"
                    if internal
                    else "user_target",
                    "pd_target": None,
                    "motor_position": motor.tolist(),
                    "measured": named_values(arm, names).tolist(),
                    "hand_names": hand["name"],
                    "measured_hand": hand["position"],
                    "contact": None,
                    "stale": False,
                    "protected": None,
                    "target_changed": False,
                    "matching_gap_s": float(max(ag[i], hg[i])),
                }
            )
        except (ValueError, KeyError):
            continue
    return rows, {
        "legacy": True,
        "matched": len(rows),
        "queries": len(applied),
        "unmatched_fraction": 1.0 - len(rows) / len(applied) if applied else 1.0,
        "status_matched": int(np.sum(si >= 0)),
        "status_unmatched_fraction": float(np.mean(si < 0)) if len(si) else 1.0,
        "max_matching_interval_s": MATCH_S,
        "time_basis": "original stamp; receive wall time only when original absent",
    }


def analyze_samples(
    samples, fk, *, path_end=None, alignment=None, expected_target=None, observe_s=10.0
):
    reasons = []
    if path_end is None:
        reasons.append("planned_path_end_missing")
        path_end = samples[0]["stamp"] if samples else 0.0
    if alignment and alignment.get("legacy"):
        reasons.append("legacy_missing_same_tick_and_safety_state")
    t, tip, targets, measured, deltas = [], [], [], [], []
    final_target = None
    for row in samples:
        try:
            if row.get("tracking_target_known") is False:
                reasons.append("accepted_target_status_unmatched")
                continue
            q = np.asarray(row["measured"], float)
            target = row.get("tracking_target") or row.get("user_target")
            if target is None:
                reasons.append("user_target_missing")
                continue
            target = np.asarray(target, float)
            accepted_target = target.copy()
            if expected_target is not None and row["stamp"] >= path_end:
                target = np.asarray(expected_target, float)
                if row["stamp"] > path_end + MATCH_S and not np.allclose(
                    accepted_target, target, atol=1e-10, rtol=0
                ):
                    reasons.append("target_changed")
            hand = dict(zip(row["hand_names"], row["measured_hand"]))
            delta = (fk(q, hand) - fk(target, hand)) * 1000.0
            t.append(row["stamp"])
            tip.append(float(np.linalg.norm(delta)))
            targets.append(target)
            measured.append(q)
            deltas.append(delta)
            if row["stamp"] >= path_end:
                if final_target is None:
                    final_target = target.copy()
                elif expected_target is None and not np.allclose(
                    target, final_target, atol=1e-10, rtol=0
                ):
                    reasons.append("target_changed")
                for flag in ("stale", "protected"):
                    if row.get(flag) is not False:
                        reasons.append(flag)
                if row.get("contact") is not False:
                    reasons.append("contact_or_unknown")
                if any(v is None for v in row.get("temperature", [None])):
                    reasons.append("temperature_missing")
                if any(v is None for v in row.get("torque_feedback", [None])):
                    reasons.append("torque_feedback_missing")
        except (KeyError, ValueError, TypeError):
            reasons.append("missing_required_state")
    result = evaluate(
        t,
        tip,
        np.asarray(targets).reshape(-1, 7),
        np.asarray(measured).reshape(-1, 7),
        path_end=path_end,
        invalid_reasons=reasons,
        observe_s=observe_s,
    )
    result["alignment"] = alignment
    result["samples_selected"] = len(samples)
    result["samples_used"] = len(t)
    result["required_sample_loss_fraction"] = (
        1.0 - len(t) / len(samples) if samples else 1.0
    )
    result["tip_delta_median_xyz_mm"] = (
        np.median(deltas, axis=0).tolist() if deltas else None
    )
    if targets:
        names = samples[-1]["hand_names"]
        hand_rows = [dict(zip(s["hand_names"], s["measured_hand"])) for s in samples]
        median_hand = {
            n: float(np.median([h[n] for h in hand_rows if n in h])) for n in names
        }
        aggregate_delta = (
            fk(np.median(measured, axis=0), median_hand)
            - fk(np.median(targets, axis=0), median_hand)
        ) * 1000
        result["fk_of_median_joint_delta_xyz_mm"] = aggregate_delta.tolist()
        result["fk_of_median_joint_distance_mm"] = float(
            np.linalg.norm(aggregate_delta)
        )
    result["sensor_resolution_note"] = (
        "0.03/0.06 deg hysteresis; verify actual encoder quantization before tuning"
    )
    result["physical_tip_minus_fk_mm"] = None
    result["torque_rms_nm"] = {}
    for field in (
        "torque_pd",
        "torque_feedforward",
        "torque_estimated_total",
        "torque_feedback",
    ):
        values = [
            r[field]
            for r in samples
            if field in r
            and len(r[field]) == 7
            and all(v is not None for v in r[field])
        ]
        if values:
            result["torque_rms_nm"][field] = np.sqrt(
                np.mean(np.asarray(values) ** 2, axis=0)
            ).tolist()
    additional = [
        np.asarray(r["torque_gravity_correction"])
        + r["torque_friction"]
        + r["torque_integral"]
        + r.get("torque_static_friction", 0.0)
        for r in samples
        if all(
            k in r
            for k in ("torque_gravity_correction", "torque_friction", "torque_integral")
        )
    ]
    result["additional_torque_rms"] = (
        float(np.sqrt(np.mean(np.asarray(additional) ** 2))) if additional else None
    )
    temperatures = [
        r["temperature"]
        for r in samples
        if "temperature" in r
        and len(r["temperature"]) == 7
        and all(v is not None for v in r["temperature"])
    ]
    result["temperature_max_c"] = (
        np.max(temperatures, axis=0).tolist() if temperatures else None
    )
    if measured:
        resolution = []
        for values in np.asarray(measured).T:
            changes = np.diff(np.unique(values))
            changes = changes[changes > 1e-10]
            resolution.append(
                float(np.rad2deg(np.min(changes))) if len(changes) else None
            )
        result["observed_minimum_joint_increment_deg"] = resolution
        errors = np.asarray(measured) - np.asarray(targets)
        initial_sign = np.sign(errors[0])
        result["joint_overshoot_from_initial_error_side_deg"] = np.rad2deg(
            np.maximum(0.0, np.max(-errors * initial_sign, axis=0))
        ).tolist()
    return result, np.asarray(t), np.asarray(tip)


def nearest_indices(source, query, max_gap=MATCH_S):
    source, query = np.asarray(source, float), np.asarray(query, float)
    if not np.isfinite(source).all() or np.any(np.diff(source) < 0):
        raise ValueError("source times must be finite and sorted")
    if not np.isfinite(query).all() or not 0 < max_gap <= MATCH_S:
        raise ValueError("invalid query/max matching interval")
    if not len(source):
        return np.full(query.shape, -1, int), np.full(query.shape, np.inf)
    right = np.clip(np.searchsorted(source, query), 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    idx = np.where(
        np.abs(source[left] - query) <= np.abs(source[right] - query), left, right
    )
    gap = np.abs(source[idx] - query)
    return np.where(gap <= max_gap + 1e-12, idx, -1), gap


def named_values(message, names, field="position"):
    keys, vals = message["name"], message[field]
    if len(keys) != len(set(keys)) or len(keys) != len(vals):
        raise ValueError("duplicate names or incomplete joint vector")
    mapped = dict(zip(keys, vals))
    try:
        out = np.asarray([mapped[j] for j in names], float)
    except KeyError as exc:
        raise ValueError(f"missing joint {exc}") from exc
    if not np.isfinite(out).all():
        raise ValueError("nonfinite joint data")
    return out


def evaluate(t, tip_error_mm, target, measured, *, path_end, invalid_reasons=(), observe_s=10.0):
    """observe_s: 경로 끝 뒤 관측해야 하는 시간 — 정지형 10 s, sweep/step 은 붙들기 길이(11.10)."""
    t, tip, target, measured = map(
        lambda x: np.asarray(x, float), (t, tip_error_mm, target, measured)
    )
    if (
        t.ndim != 1
        or tip.shape != t.shape
        or target.shape != (len(t), 7)
        or measured.shape != target.shape
    ):
        raise ValueError("expected times/errors [N], joint arrays [N,7]")
    reasons = list(invalid_reasons)
    if (
        not np.isfinite(path_end)
        or not len(t)
        or any(not np.isfinite(v).all() for v in (t, tip, target, measured))
    ):
        reasons.append("missing_required_state")
    if len(t) and (np.any(np.diff(t) <= 0) or np.any(np.diff(t) > GAP_S)):
        reasons.append("sample_gap_or_nonmonotonic_time")
    if len(t) and (t[0] > path_end + MATCH_S or t[-1] < path_end + observe_s):
        reasons.append("insufficient_observation")
    err = np.degrees(measured - target)
    tracking, settling, joint_pass, hold_ptp = False, None, False, None
    if not reasons:
        for i in np.flatnonzero((t >= path_end) & (t <= path_end + 3) & (tip <= 1)):
            j = int(np.searchsorted(t, t[i] + observe_s - 1e-9))
            if j >= len(t):
                continue
            if np.all(tip[i : j + 1] <= 1):
                tracking, settling = True, float(t[i] - path_end)
                hold_ptp = np.degrees(np.ptp(measured[i : j + 1], axis=0)).tolist()
                joint_pass = bool(
                    np.all(np.abs(err[i : j + 1]) <= 0.1)
                    and np.all(np.asarray(hold_ptp) <= 0.05)
                )
                break
    valid_err = err[np.isfinite(err).all(axis=1)]
    finite_tip = tip[np.isfinite(tip)]
    return {
        "eligible": not reasons,
        "excluded_reasons": sorted(set(reasons)),
        "tracking_pass": tracking,
        "joint_diagnostics_pass": joint_pass,
        "pass": bool(tracking and joint_pass and not reasons),
        "settling_s": settling,
        "joint_peak_to_peak_deg": hold_ptp,
        "joint_error_median_deg": np.median(valid_err, axis=0).tolist()
        if len(valid_err)
        else None,
        "joint_abs_error_max_deg": np.max(np.abs(valid_err), axis=0).tolist()
        if len(valid_err)
        else None,
        "tip_error_max_mm": float(np.max(finite_tip)) if len(finite_tip) else None,
        "tip_error_median_mm": float(np.median(finite_tip))
        if len(finite_tip)
        else None,
        "physical_accuracy_measured": False,
    }


def prepare_trials(
    poses, lo, hi, smooth, collision_check, *, stage="A", joints=JOINTS,
    offset_deg=2.0, moved_joints=7, repeats=3, pose_ids=None,
):
    if stage not in tuple("ABCDEFG") + FRICTION_STAGES:
        raise ValueError("stage must be A..G or F1..F4")
    if not (0 < offset_deg <= 5 and 1 <= moved_joints <= 7 and repeats >= 1):
        raise ValueError("offset_deg in (0, 5], moved_joints 1..7, repeats >= 1")
    if pose_ids is not None and not set(pose_ids) <= {p["id"] for p in poses}:
        raise ValueError(f"unknown pose ids: {sorted(set(pose_ids) - {p['id'] for p in poses})}")
    if len(poses) != 6 or len({p["id"] for p in poses}) != 6:
        raise ValueError("six uniquely named safe poses required")
    if (
        sum(p["split"] == "fit" for p in poses) != 3
        or sum(p["split"] == "holdout" for p in poses) != 3
    ):
        raise ValueError("freeze three fit and three holdout poses")
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if (
        lo.shape != (7,)
        or hi.shape != (7,)
        or not np.isfinite([lo, hi]).all()
        or np.any(lo >= hi)
    ):
        raise ValueError("invalid limits")
    trials, excluded = [], []
    for p in poses:
        q = np.asarray(p["q"], float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("pose must have seven finite radians")
        if p["split"] != ("holdout" if stage == "G" else "fit"):
            continue
        if pose_ids is not None and p["id"] not in pose_ids:
            continue
        for joint in range(moved_joints):
            for sign in (-1, 1):
                start = q.copy()
                start[joint] += sign * np.radians(offset_deg)
                for repeat in range(repeats):
                    row = {
                        "id": f"{stage}_{p['id']}_j{joint + 1}_{sign:+d}_r{repeat + 1}",
                        "pose": p["id"],
                        "split": p["split"],
                        "stage": stage,
                        "joint": joint,
                        "direction": sign,
                        "repeat": repeat + 1,
                        "offset_deg": offset_deg,
                        "start": start.tolist(),
                        "target": q.tolist(),
                        "hold_s": 15.0,
                    }
                    if np.any(np.minimum(start, q) < lo) or np.any(
                        np.maximum(start, q) > hi
                    ):
                        excluded.append({**row, "reason": "joint_limit"})
                        continue
                    frames = np.asarray(smooth(np.stack([start, q])), float)
                    check = collision_check(frames)
                    if not check.get("ok"):
                        excluded.append({**row, "reason": "collision", "check": check})
                        continue
                    trials.append(
                        {**row, "frames": frames.tolist(), "collision": check}
                    )
    return {
        "schema": "stationary_plan_v1",
        "stage": stage,
        "offset_deg": offset_deg,
        "moved_joints": moved_joints,
        "repeats": repeats,
        "joint_names": list(joints),
        "trials": trials,
        "excluded": excluded,
        "commands_sent": 0,
    }


#: 11.09 고속 측정 — stationary(2° 접근 + 15 s) 대신 한 관절 한 trial 로 묶은 궤적.
#: sweep = 저속 왕복(쿨롱 · 점성 · 반전 걸림 식별), step = 계단 + 짧은 붙들기(보상 설정 비교).
MOTIONS = ("stationary", "sweep", "step")
#: 끝 붙들기 — 궤적 안에 붙들기가 들어 있어 끝에는 짧게만 둔다
MOTION_HOLD_S = 1.0
SWEEP_VMAX = 0.5  # sweep 상한 [rad/s] = pd max_vel(단계 승인값). 올리려면 pd yaml max_vel 과 함께 (11.10 사용자 0.1→0.3→0.5)
SWEEP_AMAX = 2.0  # sweep 가속 상한 [rad/s²] — 15° 반 왕복(0.26 rad)에서도 0.5 rad/s 에 닿는 값 (11.10, 접근·복귀는 0.1 그대로)
#: ★11.10 사용자 "한 번에": sweep 은 pose 하나의 관절들을 한 trial 로 이어 붙인다(pd 연결·bag·판정 1 회)
SWEEP_JOINT_GAP_S = 0.5


def _hold(q, seconds, dt):
    return np.repeat(np.asarray(q, float)[None, :], max(1, int(round(seconds / dt))), axis=0)


def motion_frames(q, joint, motion, parametrize, *, dt, amplitude_deg, speeds, cycles,
                  step_deg, step_hold_s):
    """pose q 에서 시작해 q 로 끝나는 한 관절 궤적 (N,7). parametrize(waypoints, vmax) → 프레임
    (꼭짓점마다 cosine 램프로 멈췄다 출발 — 반전마다 속도 0 을 지난다)."""
    q = np.asarray(q, float)

    def at(deg):
        p = q.copy()
        p[joint] += np.radians(deg)
        return p

    if motion == "sweep":
        parts = [q[None, :]]
        for v in speeds:
            wps = [q] + [at(s * amplitude_deg) for _ in range(cycles) for s in (1, -1)] + [q]
            parts.append(np.asarray(parametrize(np.stack(wps), float(v)), float)[1:])
            parts.append(_hold(q, 0.5, dt))     # 속도 단계 사이 — 정지 구간도 남긴다
        return np.vstack(parts)
    if motion == "step":
        parts = [q[None, :]]
        for target in (at(step_deg), q, at(-step_deg), q):
            parts.append(np.asarray(parametrize(np.stack([parts[-1][-1], target]), None), float)[1:])
            parts.append(_hold(target, step_hold_s, dt))
        return np.vstack(parts)
    raise ValueError(f"unknown motion {motion!r}")


def prepare_motion_trials(
    poses, lo, hi, parametrize, collision_check, *, stage, motion, dt, joints=JOINTS,
    moved_joints=4, pose_ids=None, amplitude_deg=(15.0, 10.0, 6.0, 3.0), speeds=(0.05, 0.1, 0.2, 0.3),
    cycles=1, step_deg=2.0, step_hold_s=5.0, repeats=1, chain=False,
):
    """한 관절 한 trial(반복은 repeats). 시작 = 끝 = 고정 pose → 복귀는 접근 경로만 되짚는다.

    ★11.10: amplitude_deg 는 후보 목록(큰 것부터) — 관절마다 관절 한계·테이블·자기충돌 검사를
    통과하는 가장 큰 진폭을 고른다. 하나도 통과 못 하면 그 관절은 excluded."""
    amps = sorted({float(a) for a in np.atleast_1d(amplitude_deg)}, reverse=True)
    if motion not in MOTIONS[1:]:
        raise ValueError(f"motion must be one of {MOTIONS[1:]}")
    if stage not in FRICTION_STAGES:
        raise ValueError("sweep/step use friction stages F1..F4")
    if not (amps and all(0 < a <= 20 for a in amps) and 0 < step_deg <= 5 and 1 <= moved_joints <= 7
            and 1 <= cycles <= 5 and 0 < step_hold_s <= 15 and repeats >= 1):
        raise ValueError("amplitude in (0, 20], step in (0, 5] deg, joints 1..7, cycles 1..5, hold (0, 15] s")
    if not speeds or any(not 0 < v <= SWEEP_VMAX for v in speeds):
        raise ValueError(f"sweep speeds in (0, {SWEEP_VMAX}] rad/s")
    if chain and motion != "sweep":
        raise ValueError("chain is only for sweep")
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    trials, excluded = [], []
    for p in poses:
        if p["split"] != "fit" or (pose_ids is not None and p["id"] not in pose_ids):
            continue
        q = np.asarray(p["q"], float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("pose must have seven finite radians")
        for joint in range(moved_joints):
            tried = []
            for amp in (amps if motion == "sweep" else amps[:1]):
                frames = motion_frames(q, joint, motion, parametrize, dt=dt, amplitude_deg=amp,
                                       speeds=speeds, cycles=cycles, step_deg=step_deg,
                                       step_hold_s=step_hold_s)
                if np.any(frames.min(0) < lo) or np.any(frames.max(0) > hi):
                    tried.append({"amplitude_deg": amp, "reason": "joint_limit"})
                    continue
                check = collision_check(frames)
                if not check.get("ok"):
                    tried.append({"amplitude_deg": amp, "reason": "collision", "check": check})
                    continue
                break
            else:
                amp, frames = None, None
            for repeat in range(repeats):
                row = {
                    "id": f"{stage}{motion[:2]}_{p['id']}_j{joint + 1}_r{repeat + 1}",
                    "pose": p["id"], "split": p["split"], "stage": stage, "motion": motion,
                    "joint": joint, "direction": 0, "repeat": repeat + 1,
                    "offset_deg": (amp if motion == "sweep" else step_deg),
                    "start": q.tolist(), "target": q.tolist(), "hold_s": MOTION_HOLD_S,
                    "amplitude_tried": tried,
                }
                if frames is None:
                    excluded.append({**row, "reason": tried[-1]["reason"] if tried else "no_amplitude"})
                    continue
                trials.append({**row, "frames": frames.tolist(), "collision": check})
    if chain:
        trials = _chain_trials(trials, stage, dt)
    return {
        "schema": "stationary_plan_v1", "stage": stage, "motion": motion,
        "offset_deg": amps[0] if motion == "sweep" else step_deg,
        "amplitude_candidates_deg": amps,
        "moved_joints": moved_joints, "repeats": repeats, "joint_names": list(joints),
        "speeds": [float(v) for v in speeds], "cycles": cycles, "step_hold_s": step_hold_s,
        "trials": trials, "excluded": excluded, "commands_sent": 0,
    }


def _chain_trials(trials, stage, dt):
    """같은 pose·repeat 의 관절별 sweep 을 순서대로 이어 한 trial 로 만든다. 각 조각은 pose 에서
    시작·끝나고 개별로 관절 한계·충돌 검사를 통과했으므로 이어 붙여도 같은 기하를 지난다.
    segments[i] = 관절 하나의 프레임 구간 [start, end) — 분석은 이것으로 관절을 나눈다."""
    groups = {}
    for t in trials:
        groups.setdefault((t["pose"], t["repeat"]), []).append(t)
    out = []
    for (pose, repeat), rows in groups.items():
        rows = sorted(rows, key=lambda t: t["joint"])
        q = np.asarray(rows[0]["start"], float)
        parts, segments, n = [q[None, :]], [], 1
        for t in rows:
            f = np.asarray(t["frames"], float)[1:]
            segments.append({"joint": t["joint"], "start": n, "end": n + len(f),
                             "amplitude_deg": t["offset_deg"], "amplitude_tried": t["amplitude_tried"]})
            parts.append(f)
            n += len(f)
            gap = _hold(q, SWEEP_JOINT_GAP_S, dt)
            parts.append(gap)
            n += len(gap)
        checks = [t["collision"] for t in rows]
        out.append({
            **{k: v for k, v in rows[0].items() if k not in ("frames", "collision", "amplitude_tried")},
            "id": f"{stage}sw_{pose}_all_r{repeat}",
            "joint": None, "joints": [t["joint"] for t in rows],
            "offset_deg": max(t["offset_deg"] for t in rows),
            "segments": segments,
            "amplitude_tried": [],
            "frames": np.vstack(parts).tolist(),
            "collision": {
                "ok": all(c.get("ok") for c in checks),
                "min_clear_m": min(float(c.get("min_clear_m", np.inf)) for c in checks),
                "samples": sum(int(c.get("samples", 0)) for c in checks),
                "per_joint": checks,
            },
        })
    return out


def select_candidate(rows):
    eligible = [r for r in rows if r.get("pass") is True]
    return (
        min(
            eligible,
            key=lambda r: (max(r["kp_multiplier"]), r["additional_torque_rms"]),
        )
        if eligible
        else None
    )


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def new_run(out, inputs, metadata, *, repo):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)

    def git(*args):
        p = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        )
        return p.stdout.strip() if p.returncode == 0 else None

    manifest = {
        "schema": "stationary_run_v1",
        "created_wall_ns": time.time_ns(),
        "code_commit": git("rev-parse", "HEAD"),
        "code_status": git("status", "--porcelain"),
        "inputs": {
            k: {"path": str(Path(v).resolve()), "sha256": file_hash(v)}
            for k, v in inputs.items()
        },
        "joint_order": list(arm_joint_names(metadata.get("side", "right"))),
        "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID"),
        "commands_sent": 0,
        "metadata": metadata,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def write_report(out, summary, t, tip):
    """Write only new files, with explicit evidence categories and no guessed causes."""
    out = Path(out)
    with (out / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    with (out / "report.md").open("x") as f:
        f.write("# RH56F1 정지 추종 분석\n\n")
        f.write(
            "## 관측 사실\n\n수신 시각과 원본 timestamp는 events.jsonl에 보존한다.\n\n"
        )
        f.write(
            "## 계산 결과\n\n```json\n"
            + json.dumps(summary, indent=2, ensure_ascii=False)
            + "\n```\n\n"
        )
        f.write(
            "## 원인 가설\n\n중력·마찰의 개별 기여도는 이 기록만으로 확정하지 않는다.\n\n"
        )
        f.write(
            "## 합격 여부\n\n"
            + ("통과" if summary.get("pass") else "미통과 또는 판정 제외")
            + "\n\n"
        )
        f.write(
            "FK 추종 차이는 모델상의 계산이며 실제 공간의 1 mm 정확도를 증명하지 않는다. applied.effort는 feedforward 토크다.\n"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from typing import cast

    fig = plt.figure(figsize=(9, 3))
    ax = cast(Axes, fig.add_subplot(111))
    ax.plot(np.asarray(t) - t[0] if len(t) else [], tip)
    ax.axhline(1, color="red", linestyle="--")
    ax.set(xlabel="Recorded time (s)", ylabel="FK tracking error (mm)")
    fig.tight_layout()
    fig.savefig(out / "tracking.png")
    plt.close(fig)


# ★11.10 사용자: 실험기가 Python JSONL(events.jsonl)을 쓰지 않고 rh56f1_record.sh 의 bag(arm/hand/pd) 하나로 남긴다.
# 분석은 bag 을 읽어 같은 이벤트 꼴({topic, recv_wall_ns, header_stamp_ns, frame_id, message})로 되돌린다.
# recv_wall_ns = 기록기가 받은 벽시계 시각(bag 타임스탬프) — 예전 실험기 콜백의 time.time_ns() 와 같은 뜻.
def run_bag_dirs(run):
    """runs/<trial>/bag/<HHMMSS>_<name>/{arm,hand,pd} — 가장 늦은 묶음 하나."""
    found = sorted(p for p in (Path(run) / "bag").glob("*") if (p / "arm").is_dir())
    return [found[-1] / k for k in ("arm", "hand", "pd") if (found[-1] / k).is_dir()] if found else []


def read_bag_events(dirs, topics=None):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rosidl_runtime_py.utilities import get_message

    wanted = None if topics is None else set(topics)
    events = []
    for d in dirs:
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(d), storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        names = {t.name: t.type for t in reader.get_all_topics_and_types()}
        if wanted is not None:
            keep = sorted(wanted & set(names))
            if not keep:   # 빈 필터 = 전부 읽기(rosbag2) — 이 bag 엔 볼 토픽이 없다
                continue
            reader.set_filter(rosbag2_py.StorageFilter(topics=keep))
            names = {k: names[k] for k in keep}
        # 볼 토픽 타입만 불러온다 — 손 bag 의 rh56f1_interfaces 없이도 분석 가능
        types = {k: get_message(v) for k, v in names.items()}
        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            message = message_to_ordereddict(deserialize_message(raw, types[topic]))
            header = message.get("header", {})
            stamp = header.get("stamp")
            events.append({
                "topic": topic,
                "recv_wall_ns": int(t_ns),
                "recv_mono_ns": None,
                "header_stamp_ns": None if stamp is None
                else int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"]),
                "frame_id": header.get("frame_id"),
                "message": message,
            })
    events.sort(key=lambda e: e["recv_wall_ns"])
    return events


def load_run_events(run, topics=None):
    """실험 한 번의 이벤트 — bag 이 있으면 bag, 없으면 예전 events.jsonl(11.10 이전 실험)."""
    run = Path(run)
    dirs = run_bag_dirs(run)
    if dirs:
        return read_bag_events(dirs, topics)
    events = []
    with (run / "events.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                e = json.loads(line)
                if topics is None or e["topic"] in topics:
                    events.append(e)
    return events

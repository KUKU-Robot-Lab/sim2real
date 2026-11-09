#!/usr/bin/env python3
"""RH56F1 capture/prepare/run/analyze. Only run --execute can publish commands."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths, stationary_tracking as S
from policy_control.pd_law import load_pd_config
from policy_control.pd_gains import load_and_check
from policy_control.contract import load_contract, load_driver_gains
from policy_control.fk_numpy import UrdfTree
import run_paths  # noqa: E402 — _paths 가 scripts/ 를 경로에 넣은 뒤

ROOT = Path(__file__).resolve().parents[3]
URDF = _paths.RL_WS / "hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
START_TOL_RAD = 0.01


def dump(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_events(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def tip_fk(urdf, tip, side="right"):
    tree = UrdfTree(urdf)
    joints = S.arm_joint_names(side)

    def fk(q, hand):
        values = tree.expand_mimic({**hand, **dict(zip(joints, q))})
        return tree.transforms([tip], values)[tip][:3, 3]

    return fk


def mirror_pose_set_to_side(poses, scene, urdf, limits):
    """Map captured right index-tip positions across the base Y=0 plane to left-arm IK."""
    from policy_control.hdgp_frozen import ARM_MIRROR_SIGN

    tree = UrdfTree(urdf)
    right_joints = S.arm_joint_names("right")
    left_joints = S.arm_joint_names("left")
    right_tip, left_tip = "r_hl_index_tip", "l_hl_index_tip"
    right_hand_keys = set(poses[0].get("hand", {}))
    left_hand = {
        key: float(value)
        for key, value in scene.items()
        if key.startswith("l_hj_")
    }
    if not right_hand_keys or not left_hand:
        raise ValueError("mirror planning requires measured right and left hand states")
    if len(left_hand) != len(right_hand_keys):
        raise ValueError("right and left hand state dimensions differ")
    lo = np.array([limits[j][0] for j in left_joints], dtype=float)
    hi = np.array([limits[j][1] for j in left_joints], dtype=float)
    signs = np.asarray(ARM_MIRROR_SIGN, dtype=float)

    def transform(side, q, hand):
        joints = left_joints if side == "left" else right_joints
        tip = left_tip if side == "left" else right_tip
        values = tree.expand_mimic({**hand, **dict(zip(joints, q))})
        return tree.transforms([tip], values)[tip]

    mirrored = []
    for pose in poses:
        right_hand = {key: float(value) for key, value in pose["hand"].items()}
        right_q = np.asarray(pose["q"], dtype=float)
        target = transform("right", right_q, right_hand)[:3, 3].copy()
        target[1] *= -1.0
        q = signs * right_q
        q = np.clip(q, lo + 1e-6, hi - 1e-6)
        for _ in range(200):
            current = transform("left", q, left_hand)[:3, 3]
            error = target - current
            if np.linalg.norm(error) <= 5e-5:
                break
            jac = np.empty((3, 7), dtype=float)
            for j in range(7):
                perturbed = q.copy()
                perturbed[j] += 1e-5
                jac[:, j] = (
                    transform("left", perturbed, left_hand)[:3, 3] - current
                ) / 1e-5
            delta = jac.T @ np.linalg.solve(
                jac @ jac.T + 1e-5 * np.eye(3), error
            )
            delta = np.clip(delta, -0.05, 0.05)
            q = np.clip(q + delta, lo + 1e-6, hi - 1e-6)
        residual = float(np.linalg.norm(target - transform("left", q, left_hand)[:3, 3]))
        if residual > 5e-5:
            raise ValueError(
                f"{pose['id']}: mirrored left fingertip IK residual {residual * 1000:.3f} mm"
            )
        mirrored.append(
            {
                **pose,
                "id": pose["id"].replace("right_", "left_mirror_"),
                "q": q.tolist(),
                "hand": left_hand,
                "mirrored_from": pose["id"],
                "mirror_tip_residual_mm": residual * 1000.0,
            }
        )
    return mirrored


def analyze(args):
    source = args.events.resolve()
    tip = args.tip or f"{'r' if args.side == 'right' else 'l'}_hl_index_tip"
    S.new_run(
        args.out,
        {"events": source, "urdf": args.urdf},
        {
            "mode": "analyze",
            "tip": tip,
            "side": args.side,
            "planned_path_end": args.path_end,
            "window_start": args.start_time,
        },
        repo=ROOT,
    )
    # Preserve the exact original events without allocating a second in-memory copy on disk.
    (args.out / "events.jsonl").symlink_to(os.path.relpath(source, args.out))
    rows, alignment = S.aligned_samples(load_events(source), side=args.side)
    if args.start_time is not None:
        rows = [r for r in rows if r["stamp"] >= args.start_time]
    summary, t, tip = S.analyze_samples(
        rows, tip_fk(args.urdf, tip, args.side), path_end=args.path_end, alignment=alignment
    )
    S.write_report(args.out, summary, t, tip)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def capture(args):
    # engage 직후·이전 trial 복귀 직후의 팔은 아직 처지는 중이다 — 찍은 뒤 시작 전까지 0.11 deg 넘게 움직이면
    # run 의 시작 자세 검사(START_TOL_RAD)에 막힌다(11.09 r3). 멈출 시간을 두고 찍는다.
    if args.settle_s > 0:
        time.sleep(args.settle_s)
    S.new_run(
        args.out,
        {"urdf": args.urdf},
        {"mode": "capture", "seconds": args.seconds, "side": args.side},
        repo=ROOT,
    )
    raw = args.out / "raw"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("stationary_recorder.py")),
            "--side",
            args.side,
            "--seconds",
            str(args.seconds),
            "--out",
            str(raw),
        ],
        check=True,
    )
    (args.out / "events.jsonl").symlink_to("raw/events.jsonl")
    events = load_events(raw / "events.jsonl")
    rows, alignment = S.aligned_samples(events, side=args.side)
    summary = {
        "pass": False,
        "eligible": False,
        "excluded_reasons": ["capture_only"],
        "alignment": alignment,
        "commands_sent": 0,
    }
    try:
        pose = S.capture_pose(events, side=args.side)
    except ValueError as exc:
        summary["excluded_reasons"].append(str(exc))
        pose = None
    if pose is not None:
        dump(
            args.out / "snapshot.json",
            {
                "id": args.pose_id,
                "split": args.split,
                **pose,
                "source": str(args.out / "events.jsonl"),
                "note": "Operator must verify free-space pose; capture does not certify collision safety.",
            },
        )
    S.write_report(args.out, summary, [], [])
    return 0 if pose is not None else 1


def prepare(args):
    import plan_home_path as P
    import home_path_world as W

    poses = json.loads(args.poses.read_text())
    cfg = load_pd_config(args.pd_config)
    contract = load_contract(args.contract)
    gains = load_and_check(cfg.gains, contract, side=args.side)
    base = load_driver_gains(cfg.gains.yaml)
    block = S.validate_stage(
        args.stage,
        gains.real_kp,
        gains.real_kd,
        [base[i][0] for i in range(1, 8)],
        [base[i][1] for i in range(1, 8)],
        cfg.stationary_compensation,
    )
    if not cfg.gains.require_live_description or cfg.side != args.side:
        raise ValueError(
            f"prepare requires {args.side} stationary_tracking experiment profile"
        )
    # Scene is a complete, captured canonical pose including the other arm and both hands.
    scene = json.loads(args.scene.read_text())
    other_side = "left" if args.side == "right" else "right"
    if not all(f"{other_side[0]}_aj_{i}" in scene for i in range(1, 8)):
        raise ValueError(
            f"scene must include measured {other_side} arm; no automatic home/bc05 assumption"
        )
    tree = UrdfTree(args.urdf)
    joints = S.arm_joint_names(args.side)
    required = {
        j.name for j in tree.joints.values() if j.type != "fixed" and j.mimic is None
    } - set(joints)
    if not required <= set(scene):
        raise ValueError(
            f"collision scene missing movable joints: {sorted(required - set(scene))}"
        )
    limits = W.load_profile_limits(args.profile)
    lo, hi = (
        np.array([limits[j][0] for j in joints]),
        np.array([limits[j][1] for j in joints]),
    )
    if args.mirror_right:
        if args.side != "left":
            raise ValueError("--mirror-right is only valid with --side left")
        poses = mirror_pose_set_to_side(poses, scene, args.urdf, limits)
    hand = poses[0].get("hand", {})
    hand_tol_rad = 0.005
    hand_deltas = [
        abs(float(p.get("hand", {}).get(j, float("inf"))) - float(q))
        for p in poses
        for j, q in hand.items()
    ]
    max_hand_delta = max(hand_deltas, default=float("inf"))
    if (
        not hand
        or any(set(p.get("hand", {})) != set(hand) for p in poses)
        or max_hand_delta > hand_tol_rad
    ):
        raise ValueError(
            f"all six poses must share a fixed measured hand posture within {hand_tol_rad} rad; "
            f"max delta={max_hand_delta:.6f} rad"
        )
    scene.update(hand)
    world = W.build_world(
        W.WorldSpec(urdf=args.urdf, env_yaml=args.env_yaml, side=args.side)
    )

    def check(frames):
        # Checker leaves its start-pose exclusions on the shared world; a stale set makes the
        # next Checker miss them and flip ok/fail on identical paths. Start each from none.
        world.excluded_pairs = set()
        checker = P.Checker(world, [scene], 0.02, frames[0], (lo, hi))
        checker.lenient = False  # no contact/escape exemptions for this experiment
        report = checker.check_path(frames)
        return {
            "ok": bool(report["ok"]),
            "min_clear_m": float(report["min_clear"]),
            "samples": report["samples"],
            "first_failures": report["fails"][:3],
            "excluded_structural_pairs": checker.always_contact,
        }

    dt = 0.01

    # Existing cosine ramp: peak acceleration = vmax*pi/(2*ramp) <= 0.1 rad/s².
    def smooth(path, vmax=None):
        # 접근·복귀 = --speed(가속 0.1), 고속 sweep 단계 = SWEEP_VMAX·SWEEP_AMAX 까지(11.10 사용자 0.3 rad/s)
        if vmax is None:
            return P.time_parametrize(path, vmax=args.speed, dt=dt, ramp=max(1.0, args.speed * np.pi / 0.2))
        v = min(float(vmax), S.SWEEP_VMAX)
        # 여유 5 % — 이산화한 cosine 램프의 2차 차분이 상한을 살짝 넘지 않게
        return P.time_parametrize(path, vmax=v, dt=dt, ramp=v * np.pi / (2 * 0.95 * S.SWEEP_AMAX))

    if args.motion == "stationary":
        plan = S.prepare_trials(
            poses, lo, hi, smooth, check, stage=args.stage, joints=joints,
            offset_deg=args.offset_deg, moved_joints=args.moved_joints,
            repeats=args.repeats, pose_ids=args.only_pose,
        )
    else:
        # 11.09 고속 측정: 한 관절 한 trial — sweep(저속 왕복) / step(계단 + 짧은 붙들기)
        plan = S.prepare_motion_trials(
            poses, lo, hi, smooth, check, stage=args.stage, motion=args.motion, dt=dt,
            joints=joints, moved_joints=args.moved_joints, pose_ids=args.only_pose,
            amplitude_deg=args.amplitude_deg, speeds=tuple(args.sweep_speeds),
            cycles=args.cycles, step_deg=args.offset_deg, step_hold_s=args.step_hold,
            repeats=args.repeats, chain=(args.motion == "sweep" and not args.per_joint),
        )
    initial_q = np.asarray([scene[j] for j in joints], dtype=float)
    world.excluded_pairs = set()
    route_checker = P.Checker(world, [scene], 0.02, initial_q, (lo, hi))
    route_checker.lenient = False
    runnable, route_excluded = [], []
    home_q = initial_q.copy()
    ramp = max(1.0, args.speed * np.pi / 0.2)

    def safe_route(start_q, goal_q):
        direct = smooth(np.stack([start_q, goal_q]))
        direct_report = route_checker.check_path(direct)
        if direct_report.get("ok"):
            return direct, direct_report
        waypoints = P.plan_j1j4(
            route_checker,
            start_q,
            goal_q,
            lo,
            hi,
            vmax=args.speed,
            dt=dt,
            ramp=ramp,
        )
        if waypoints is None:
            return None, None
        route = P.time_parametrize(waypoints, args.speed, dt, ramp)
        report = route_checker.check_path(route)
        return (route, report) if report.get("ok") else (None, report)

    for trial in plan["trials"]:
        start_q = np.asarray(trial["start"], dtype=float)
        route, route_check = safe_route(home_q, start_q)
        if route is None:
            route_excluded.append(
                {
                    **trial,
                    "reason": "no_safe_approach_from_captured_home",
                    "approach_check": None if route_check is None else {
                        "fails": route_check["fails"][:3],
                        "min_clear_m": float(route_check["min_clear"]),
                    },
                }
            )
            continue
        # Reverse the already checked target path and home approach. With a static
        # collision scene, reversal preserves the same collision-free geometry.
        target_frames = np.asarray(trial["frames"], dtype=float)
        if trial.get("motion", "stationary") == "stationary":
            return_route = np.vstack([target_frames[::-1], route[-2::-1]])
        else:
            # sweep/step 은 pose 에서 끝난다 — 접근 경로만 되짚는다. 첫 줄은 동작 끝(실행기는 return[1:] 을 붙이고 2 줄 이상을 요구)
            return_route = np.vstack([target_frames[-1:], route[::-1]])
        trial["approach_start_q"] = home_q.tolist()
        trial["approach_frames"] = route.tolist()
        trial["approach_collision"] = {
            "min_clear_m": float(route_check["min_clear"]),
            "samples": route_check["samples"],
        }
        trial["return_frames"] = return_route.tolist()
        trial["return_collision"] = {
            "min_clear_m": min(
                float(trial["collision"]["min_clear_m"]),
                float(route_check["min_clear"]),
            ),
            "samples": int(trial["collision"]["samples"])
            + int(route_check["samples"]),
            "verified_by_reversing_checked_paths": True,
        }
        runnable.append(trial)
    plan["trials"] = runnable
    plan["excluded"].extend(route_excluded)
    plan.update(
        side=args.side,
        mirror_right=args.mirror_right,
        dt=dt,
        speed=args.speed,
        acceleration_limit=0.1 if args.motion == "stationary" else S.SWEEP_AMAX,
        scene=scene,
        fixed_hand=hand,
        fixed_hand_tolerance_rad=hand_tol_rad,
        max_measured_hand_delta_rad=max_hand_delta,
        scene_joints=sorted(required),
        block=block,
        kp=gains.real_kp,
        kd=gains.real_kd,
        compensation=cfg.stationary_compensation,
        pd_config=str(args.pd_config.resolve()),
        contract=str(args.contract.resolve()),
    )
    inputs = {
        k: getattr(args, k).resolve()
        for k in (
            "poses",
            "scene",
            "pd_config",
            "contract",
            "profile",
            "urdf",
            "env_yaml",
        )
    }
    inputs["gains"] = cfg.gains.arm_gains_override or cfg.gains.yaml
    manifest = S.new_run(
        args.out,
        inputs,
        {"mode": "prepare", "stage": args.stage, "side": args.side, "mirror_right": args.mirror_right},
        repo=ROOT,
    )
    plan["inputs"] = manifest["inputs"]
    dump(args.out / "plan.json", plan)
    (args.out / "events.jsonl").touch(exist_ok=False)
    S.write_report(
        args.out,
        {
            "pass": False,
            "eligible": False,
            "commands_sent": 0,
            "trials": len(plan["trials"]),
            "excluded": plan["excluded"],
        },
        [],
        [],
    )
    print(
        f"Prepared {len(plan['trials'])} runnable trials; {len(plan['excluded'])} excluded. No commands sent."
    )
    return 0


def run(args):
    plan = json.loads(args.plan.read_text())
    args.side = plan.get("side", "right")
    trials = [t for t in plan["trials"] if t["id"] == args.trial]
    if len(trials) != 1:
        raise ValueError(
            "select exactly one prepared --trial; no automatic candidate progression"
        )
    trial = trials[0]
    manifest = S.new_run(
        args.out,
        {
            "plan": args.plan,
            **{key: value["path"] for key, value in plan.get("inputs", {}).items()},
        },
        {
            "mode": "run",
            "side": args.side,
            "trial": trial["id"],
            "stage": plan["stage"],
            "execute_requested": args.execute,
            "kp": plan["kp"],
            "kd": plan["kd"],
            "compensation": plan["compensation"],
            "hand": plan["fixed_hand"],
        },
        repo=ROOT,
    )
    if not args.execute:
        (args.out / "events.jsonl").touch(exist_ok=False)
        S.write_report(
            args.out,
            {
                "pass": False,
                "eligible": False,
                "commands_sent": 0,
                "excluded_reasons": ["dry_run"],
                "trial": trial["id"],
            },
            [],
            [],
        )
        print(
            "Dry run: 0 commands. Existing side-specific PD must be TRACKING; the prepared plan moves automatically from its captured scene."
        )
        return 0
    for entry in plan["inputs"].values():
        if S.file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError(f"prepared input changed: {entry['path']}")
    if os.environ.get("ROS_DOMAIN_ID") not in ("50", "97", "99"):
        raise ValueError("explicit ROS_DOMAIN_ID 50 (real), 97/99 (fake) required")
    return execute_trial(args, plan, trial, manifest)


RECORD_SH = Path(__file__).with_name("rh56f1_record.sh")
BAG_START_TIMEOUT_S = 15.0


def bag_topics(side):
    """분석(S.aligned_samples · 마찰 스크립트)이 읽는 토픽."""
    return (
        "/joint_states", "/head/joint_states", f"/hand_{side}/joint_states",
        "/policy_control/joint_target", f"/policy_control/status/pd_{side}",
        f"/policy_control/pd_{side}/diagnostics",
    )


class _Latest:
    """토픽별 마지막 메시지 — 사전 변환은 읽을 때 한 번만(750 Hz 콜백에서 변환하지 않는다)."""

    def __init__(self):
        self._raw, self._dict = {}, {}

    def put(self, topic, msg):
        self._raw[topic] = msg
        self._dict.pop(topic, None)

    def __contains__(self, topic):
        return topic in self._raw

    def __getitem__(self, topic):
        if topic not in self._dict:
            from rosidl_runtime_py.convert import message_to_ordereddict

            self._dict[topic] = message_to_ordereddict(self._raw[topic])
        return self._dict[topic]

    def get(self, topic, default=None):
        return self[topic] if topic in self._raw else default


class TrialBag:
    """rh56f1_record.sh 로 runs/<trial>/bag/<시각>_<이름>/{arm,hand,pd} 를 남긴다(11.10 사용자: 실기 기록은 bag 만).

    arm bag 에 /joint_states 외에 머리 · 목표 · pd 상태를 더하고(EXTRA), 손 bag 은 그 손만 싣는다.
    """

    def __init__(self, out, side):
        self.out, self.side = Path(out), side
        self.name = f"st_{side}_{self.out.name}"
        self.env = dict(
            os.environ,
            RH56F1_RECORD_DIR=str(self.out / "bag" / "pid"),
            RH56F1_RECORD_OUT=str(self.out / "bag"),
            RH56F1_RECORD_NO_INFO="1",
            HAND_SIDES=side,
            EXTRA=" ".join(("/head/joint_states", "/policy_control/joint_target",
                           f"/policy_control/status/pd_{side}", f"/policy_control/{side}/episode")),
        )
        self.started = False

    def _logs(self):
        d = self.out / "bag" / "pid" / self.name
        return {k: d / f"{k}.log" for k in ("arm", "hand", "pd")}

    def start(self, idle=lambda: time.sleep(0.1)):
        subprocess.run(["bash", str(RECORD_SH), "start", self.name], env=self.env, check=True,
                       stdout=subprocess.DEVNULL)
        self.started = True
        need = {
            "arm": ("/joint_states", "/head/joint_states", "/policy_control/joint_target",
                    f"/policy_control/status/pd_{self.side}", f"/policy_control/{self.side}/episode"),
            "hand": (f"/hand_{self.side}/joint_states",),
            "pd": (f"/policy_control/pd_{self.side}/diagnostics",),
        }
        deadline = time.monotonic() + BAG_START_TIMEOUT_S
        while True:
            missing = []
            for k, path in self._logs().items():
                text = path.read_text(errors="replace") if path.exists() else ""
                missing += [t for t in need[k] if f"Subscribed to topic '{t}'" not in text]
            if not missing:
                return
            if time.monotonic() > deadline:
                raise ValueError("bag recorder not subscribed: " + ", ".join(missing))
            idle()

    def stop(self):
        if self.started:
            subprocess.run(["bash", str(RECORD_SH), "stop", self.name], env=self.env, check=False,
                           stdout=subprocess.DEVNULL)
            self.started = False


def execute_trial(args, plan, trial, manifest):
    """Publish only existing PD target/episode messages. No engage/release/CAN loop."""
    joint_names = tuple(plan.get("joint_names", S.arm_joint_names(args.side)))
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    from policy_control import codec
    import stationary_recorder as recorder
    from replay_to_pd import replay_episode_id

    rclpy.init()
    node = rclpy.create_node(f"stationary_tracking_{args.side}")
    other_side = "left" if args.side == "right" else "right"
    latest, received = _Latest(), {}
    count, failure, path_end = 0, None, None
    gc_was_enabled, gc_paused = gc.isenabled(), False
    episode_commands = 0
    episode = replay_episode_id()
    manifest["metadata"]["episode"] = episode
    bag = None

    def callback(topic, msg):
        # ★11.10: 기록은 bag(rh56f1_record.sh)이 한다 — 여기선 게이트용 최신값만(변환은 읽을 때).
        latest.put(topic, msg)
        received[topic] = time.monotonic()

    qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
    pd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    diag_topic = f"/policy_control/pd_{args.side}/diagnostics"
    status_topic = f"/policy_control/status/pd_{args.side}"
    for topic, typename in recorder.topic_spec(args.side).items():
        if topic == "/robot_description":
            selected_qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        elif topic in (diag_topic, status_topic):
            # Match pd_node and board_jog; preserve the live PD gate streams.
            selected_qos = pd_qos
        else:
            selected_qos = qos
        node.create_subscription(
            get_message(typename),
            topic,
            lambda m, t=topic: callback(t, m),
            selected_qos,
        )
    target_pub = node.create_publisher(JointState, "/policy_control/joint_target", 10)
    latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ep_pub = node.create_publisher(String, f"/policy_control/{args.side}/episode", latched)
    from policy_control.sources import load_profile

    profile = load_profile(Path(plan["inputs"]["profile"]["path"]))
    tree = UrdfTree(plan["inputs"]["urdf"]["path"])

    def current():
        now = time.monotonic()
        stale_topics = [
            topic
            for topic in (diag_topic, status_topic)
            if now - received.get(topic, 0) > 0.1
        ]
        if stale_topics:
            raise ValueError("PD stream stale or missing: " + ", ".join(stale_topics))
        d, status = (
            json.loads(latest[diag_topic]["data"]),
            json.loads(latest[status_topic]["data"]),
        )
        if (
            not status.get("execute")
            or not status.get("ok")
            or status.get("phase") != "TRACKING"
            or status.get("estop")
            or not d["gains"].get("runtime", {}).get("verified")
        ):
            raise ValueError("PD not ready or runtime gains unverified")
        S.validate_live_diagnostics(d)
        for key in ("kp", "kd"):
            if not np.allclose(d["gains"]["real_" + key], plan[key], rtol=0, atol=1e-6):
                raise ValueError("live gain mismatch")
        if d.get("compensation_config") != plan["compensation"]:
            raise ValueError("live compensation configuration mismatch")
        hand = tree.expand_mimic(dict(zip(d["hand_names"], d["measured_hand"])))
        if any(
            j not in hand or abs(hand[j] - q) > 0.02
            for j, q in plan["fixed_hand"].items()
        ):
            raise ValueError("hand posture changed")
        live = {}
        for topic in (
            "/joint_states",
            "/head/joint_states",
            "/hand_left/joint_states",
            "/hand_right/joint_states",
        ):
            if now - received.get(topic, 0) <= 0.1:
                message = latest[topic]
                live.update(zip(message.get("name", []), message.get("position", [])))
        for joint in plan["scene_joints"]:
            item = profile.get(joint)
            if item is None:
                if joint not in live:
                    raise ValueError(f"collision scene state missing: {joint}")
                current_value = live[joint]
            else:
                if item["source"] not in live:
                    raise ValueError(f"collision scene state missing: {joint}")
                current_value = live[item["source"]] * item["sign"]
            if abs(current_value - plan["scene"][joint]) > 0.02:
                raise ValueError(f"collision scene changed: {joint}")
        if count:
            observed = (
                latest.get("/policy_control/joint_target", {})
                .get("header", {})
                .get("frame_id", "")
            )
            if observed and observed.split(":")[0] != str(episode):
                raise ValueError("concurrent target publisher detected")
        return d

    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        d = current()
        publishers = node.get_publishers_info_by_topic("/policy_control/joint_target")
        if len(publishers) != 1 or publishers[0].node_name != node.get_name():
            raise ValueError(
                "another target publisher/jog is active; concurrent correction is forbidden"
            )
        # 0.01 rad: 정지 중 손목(j7)이 capture 후 수 초 사이 0.003 rad 흘렀다(11.09 F1offSw a02).
        # 첫 목표가 최대 0.01 rad 튀어도 Kp 70 기준 <1 N·m.
        start_gap = np.abs(np.asarray(d["measured"]) - trial["approach_start_q"])
        if np.max(start_gap) > START_TOL_RAD:
            raise ValueError(
                "live arm does not match the captured-home approach start: "
                f"max gap {np.max(start_gap):.4f} rad at j{int(np.argmax(start_gap)) + 1}"
            )
        # Another arm moving invalidates the prechecked scene; check live source names.
        state = latest.get("/joint_states", {})
        if time.monotonic() - received.get("/joint_states", 0) > 0.1:
            raise ValueError("other arm scene state missing")
        source = dict(zip(state.get("name", []), state.get("position", [])))
        for i in range(1, 8):
            name = f"openarm_{other_side}_joint{i}"
            if (
                name not in source
                or abs(source[name] - plan["scene"][f"{other_side[0]}_aj_{i}"]) > 0.002
            ):
                raise ValueError("other arm differs from collision-checked scene")
        approach = np.asarray(trial["approach_frames"], float)
        path_frames = np.asarray(trial["frames"], float)
        motion_frames = np.vstack([approach[:-1], path_frames])
        return_frames = np.asarray(trial["return_frames"], float)
        dt = plan["dt"]
        if (
            motion_frames.ndim != 2
            or motion_frames.shape[1] != 7
            or len(motion_frames) < 2
            or not np.isfinite(motion_frames).all()
            or return_frames.ndim != 2
            or return_frames.shape[1] != 7
            or len(return_frames) < 2
            or not np.isfinite(return_frames).all()
            or dt != 0.01
            or trial["hold_s"] != (
                15.0 if trial.get("motion", "stationary") == "stationary" else S.MOTION_HOLD_S
            )
        ):
            raise ValueError("invalid prepared trial dimensions/timing")
        end_index = len(motion_frames) - 1
        hold_frames = np.repeat(
            motion_frames[-1][None, :], int(trial["hold_s"] / dt), axis=0
        )
        frames = np.vstack([motion_frames, hold_frames, return_frames[1:]])
        if np.max(np.abs(np.diff(frames, axis=0) / dt)) > S.SWEEP_VMAX + 1e-8:
            raise ValueError("prepared trial exceeds velocity limit")
        if (
            len(frames) > 2
            and np.max(np.abs(np.diff(frames, n=2, axis=0) / dt**2))
            > (0.1 if trial.get("motion", "stationary") == "stationary" else S.SWEEP_AMAX) + 1e-6
        ):
            raise ValueError("prepared trial exceeds acceleration limit")
        # ★11.10: 기록은 bag 하나 — 첫 명령 전에 기록기가 필요한 토픽을 모두 구독했는지 확인한다.
        bag = TrialBag(args.out, args.side)
        # ★11.10: 기록기 구독을 기다리는 수 초 동안에도 spin 해야 PD 게이트 값(0.1 s)이 낡지 않는다.
        bag.start(idle=lambda: rclpy.spin_once(node, timeout_sec=0.1))
        # Retained recording rows make a generation-2 collection exceed the
        # 20 ms command deadline. Collect before motion and defer cyclic GC
        # until after the stop/abort confirmation; reference counting stays on.
        if gc_was_enabled:
            gc.collect()
            gc.disable()
            gc_paused = True
        # 첫 명령 직전 최신값으로 다시 확인(시작 자세 · 장면 · PD 상태).
        refresh = time.monotonic() + 0.3
        while time.monotonic() < refresh:
            rclpy.spin_once(node, timeout_sec=0.02)
        d = current()
        start_gap = np.abs(np.asarray(d["measured"]) - trial["approach_start_q"])
        if np.max(start_gap) > START_TOL_RAD:
            raise ValueError(
                "live arm does not match the captured-home approach start: "
                f"max gap {np.max(start_gap):.4f} rad at j{int(np.argmax(start_gap)) + 1}"
            )
        ep_pub.publish(String(data=json.dumps({"event": "reset", "episode": episode})))
        episode_commands += 1
        velocity = np.gradient(frames, dt, axis=0)
        velocity[0] = velocity[-1] = 0.0
        start = time.monotonic()
        path_end = time.time() + end_index * dt
        for i, (q, v) in enumerate(zip(frames, velocity)):
            while time.monotonic() < start + i * dt:
                rclpy.spin_once(
                    node,
                    timeout_sec=min(0.002, max(0.0, start + i * dt - time.monotonic())),
                )
            if time.monotonic() - (start + i * dt) > 0.02:
                raise ValueError("command scheduling delay >20ms; stop trial")
            current()
            target_pub.publish(
                codec.encode_joint_target(
                    joint_names, q, v, str(episode), i, stamp=time.time()
                )
            )
            count += 1
            rclpy.spin_once(node, timeout_sec=0.0)
    except (ValueError, KeyError, TypeError, RuntimeError, KeyboardInterrupt) as exc:
        failure = str(exc) or "operator_interrupt"
    finally:
        if episode_commands:
            # Existing stop/abort makes PD hold; never releases supporting torque.
            ep_pub.publish(
                String(
                    data=json.dumps(
                        {"event": "abort" if failure else "stop", "episode": episode}
                    )
                )
            )
            episode_commands += 1
            until = time.monotonic() + 0.5
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.01)
        if bag is not None:
            bag.stop()
        node.destroy_node()
        rclpy.shutdown()
        if gc_paused and gc_was_enabled:
            gc.enable()
        manifest["commands_sent"] = count
        manifest["episode_commands_sent"] = episode_commands
        manifest["metadata"]["runtime_gains"] = (
            json.loads(latest[diag_topic]["data"]).get("gains")
            if diag_topic in latest
            else None
        )
        manifest["metadata"].update(planned_path_end=path_end, failure=failure)
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = S.load_run_events(args.out, topics=bag_topics(args.side)) if bag is not None else []
    samples, alignment = S.aligned_samples(rows, side=args.side)
    # Discard stop/abort confirmation ticks from the timed trial window.
    if path_end is not None:
        samples = [s for s in samples if s["stamp"] <= path_end + trial["hold_s"]]
    summary, t, tip = S.analyze_samples(
        samples,
        tip_fk(
            Path(plan["inputs"]["urdf"]["path"]),
            args.tip or f"{'r' if args.side == 'right' else 'l'}_hl_index_tip",
            args.side,
        ),
        path_end=path_end,
        alignment=alignment,
        expected_target=trial["target"],
        observe_s=10.0 if trial.get("motion", "stationary") == "stationary" else max(0.5, trial["hold_s"] - 0.05),
    )
    if failure:
        summary.update(eligible=False, **{"pass": False})
        summary["excluded_reasons"].append(failure)
    summary["commands_sent"] = count
    S.write_report(args.out, summary, t, tip)
    if failure:
        print(
            f"Trial stopped: {failure}; commands_sent={count}; report={args.out / 'report.md'}",
            file=sys.stderr,
        )
    else:
        print(f"Trial complete; commands_sent={count}; report={args.out / 'report.md'}")
    return 1 if failure else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    for mode in ("capture", "prepare", "run", "analyze"):
        p = subs.add_parser(mode)
        p.add_argument("--out", type=Path, default=None, help="New directory only")
        p.add_argument("--tip", default=None)
        if mode != "run":
            p.add_argument("--urdf", type=Path, default=URDF)
        if mode == "capture":
            p.add_argument("--side", choices=("right", "left"), default="right")
            p.add_argument("--seconds", type=float, default=15.0)
            p.add_argument("--pose-id", default="captured_pose")
            p.add_argument("--settle-s", type=float, default=0.0, help="wait before recording (arm sag settles)")
            p.add_argument("--split", choices=("fit", "holdout"), default="fit")
        if mode == "analyze":
            p.add_argument("--side", choices=("right", "left"), default="right")
            p.add_argument("--events", type=Path, required=True)
            p.add_argument(
                "--path-end",
                type=float,
                help="Planned path end, wall seconds; absent means excluded",
            )
            p.add_argument(
                "--start-time",
                type=float,
                help="Explicit historical window start in wall seconds",
            )
        if mode == "prepare":
            for name in (
                "poses",
                "scene",
                "pd-config",
                "contract",
                "profile",
                "env-yaml",
            ):
                p.add_argument("--" + name, type=Path, required=True)
            p.add_argument("--side", choices=("right", "left"), default="right")
            p.add_argument("--mirror-right", action="store_true", help="mirror right fingertip poses into the selected left-arm plan")
            p.add_argument("--stage", choices=list("ABCDEFG") + list(S.FRICTION_STAGES), default="A")
            p.add_argument("--offset-deg", type=float, default=2.0, help="approach offset from the target [deg]")
            p.add_argument("--moved-joints", type=int, default=7, help="vary j1..jN only")
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--only-pose", action="append", default=None, help="keep only these pose ids (after mirroring)")
            p.add_argument("--speed", type=float, default=0.05)
            p.add_argument("--motion", choices=S.MOTIONS, default="stationary",
                           help="stationary=2° 접근+장시간 붙들기, sweep=저속 왕복, step=계단+짧은 붙들기")
            p.add_argument("--amplitude-deg", type=float, nargs="+", default=[15.0, 10.0, 6.0, 3.0],
                           help="sweep 진폭 후보 [deg] — 관절마다 충돌·한계를 통과하는 가장 큰 값(11.10)")
            p.add_argument("--sweep-speeds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
                           help="sweep 속도 단계 [rad/s]")
            p.add_argument("--cycles", type=int, default=1, help="sweep 속도당 왕복 수")
            p.add_argument("--per-joint", action="store_true",
                           help="sweep 을 관절마다 따로 trial 로(기본: pose 하나의 관절을 한 trial 로 이어서)")
            p.add_argument("--step-hold", type=float, default=5.0, help="step 붙들기 [s]")
        if mode == "run":
            p.add_argument("--plan", type=Path, required=True)
            p.add_argument("--trial", required=True)
            p.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = run_paths.run_dir("stationary", args.mode, mkdir=False)   # "New directory only" — 만드는 건 아래 쓰는 쪽
    if args.mode == "prepare" and not 0 < args.speed <= 0.05:
        parser.error("speed must be in (0, 0.05] rad/s")
    return globals()[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())

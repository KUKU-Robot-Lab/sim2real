#!/usr/bin/env python3
"""RH56F1 정지 마찰 실험의 남은 trial 을 한 프로세스로 — trial 마다 capture → rebase → run, 처음 실패에서 멈춘다.

trial 마다 stationary_tracking.py capture · rebase_trial.py · stationary_contact_guard.py + run 을 새로 띄우던
기동 비용(11.09 F1 실측 trial 46.6 s 중 약 21 s)을 없앤다. 검사와 산출물은 그 파일들과 같다(고치지 않고 import).
다른 점은 넷이다.
- ROS 노드와 구독을 하나로 계속 쓴다. 새 노드의 발견 대기(run 의 5 s · 감시 sleep 5 s)가 없다.
- settle 은 고정 3 s 대신 팔이 멈췄다는 판정(2 s 동안 관절 폭 0.8 mrad 이하, 엔코더 한 칸 0.38 mrad)에서 끝난다.
  capture 기록은 그 2 s 창이다. rebase 뒤 팔이 0.002 rad 넘게 움직였으면 다시 capture 한다(최대 3 번).
  시작 자세 0.002 rad 검사는 run 에서 그대로 한다.
- 접촉 감시는 이 프로세스가 trial 직전 정지 구간의 힘으로 기준선을 다시 잡는다(규칙 = stationary_contact_guard).
  기준선은 정지 판정 뒤에 받은 힘 중 최근 3 s 만, 그 3 s 동안 팔이 멈춰 있을 때만 잡는다. 모자라면 기다리고
  8 s 를 넘으면 멈춘다. 감시가 기준선을 잡지 않은 동안(trial 사이)은 contact=True(baseline_pending)를 낸다.
- rebase 는 rebase_trial.py 를 그대로 돌리고 충돌 world 만 한 번 만든다(매번 갓 만든 상태로 되돌린다).
  분석 · 그림은 다음 trial 의 settle 과 겹쳐 돌고 capture 전에 끝난다(기다리는 동안도 구독을 돈다).
  실패한 trial 은 사유를 먼저 찍고 분석한 뒤 멈춘다.
- SIGINT · SIGTERM 은 한 번만 KeyboardInterrupt 가 된다. 두 번째부터는 무시하고, trial 의 abort · manifest ·
  해제를 끝낸 뒤 멈춘다.

--execute 없이는 ROS 를 띄우지 않고 할 일만 찍는다(로봇 명령 0).
진행 줄 '@@ <trial> <capture|rebase|trial>' 은 콘솔이 진행 표시에 쓴다. 실패하면 마지막 줄이 사유다.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import runpy
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stationary_tracking as cli  # noqa: E402 — tools/ CLI(경로 · dump · tip_fk), policy_control 경로도 넣는다
import stationary_recorder as recorder  # noqa: E402
import home_path_world as W  # noqa: E402
from stationary_contact_guard import assess  # noqa: E402
from policy_control import stationary_tracking as S  # noqa: E402

REBASE = cli.ROOT / "analysis/friction/common/scripts/rebase_trial.py"
SETTLE_WINDOW_S, SETTLE_TOL_RAD, SETTLE_TIMEOUT_S = 2.0, 0.0008, 8.0   # 창 = 원래 capture --seconds 2
GUARD_WINDOW_S, GUARD_TIMEOUT_S = 3.0, 8.0   # 기준선 = 정지한 최근 3 s 의 힘(원래 감시 기동 뒤 settle 3 s 와 같은 길이)
RECAPTURE_TOL_RAD, RECAPTURE_MAX = 0.002, 3
KEEP_S = 5.0                 # 굴리는 버퍼 = 원래 run 이 동작 전 5 s 동안 받던 줄(events 앞부분)
GUARD_N, GUARD_MIN_N = 500, 100
READY_TIMEOUT_S = 5.0
REPORT_LOCK = threading.Lock()   # pyplot 은 스레드 안전이 아니다 — capture 와 앞 trial 분석이 그림을 같이 그리지 않게


def arm_q(row, side):
    names = row["message"].get("name", [])
    try:
        return [row["message"]["position"][names.index(f"openarm_{side}_joint{i}")] for i in range(1, 8)]
    except (ValueError, IndexError):
        return None


def settled(rows, side, now_ns, window_s=SETTLE_WINDOW_S, tol=SETTLE_TOL_RAD):
    """버퍼(앞 trial 끝 뒤의 줄)가 window_s 를 덮고 그 창의 /joint_states 팔 관절 폭이 모두 tol 이하."""
    js = [(r["recv_mono_ns"], q) for r in rows
          if r["topic"] == "/joint_states" and (q := arm_q(r, side)) is not None]
    start = now_ns - window_s * 1e9
    if not js or js[0][0] > start:
        return False
    q = np.array([q for t, q in js if t >= start])
    return len(q) >= 10 and float(np.max(np.ptp(q, axis=0))) <= tol


def guard_baseline(samples):
    """stationary_contact_guard 의 기준선 규칙 → (기준선, None) 또는 (None, 잠금 사유)."""
    try:
        data = np.asarray(samples[-GUARD_N:], float)
    except ValueError:
        return None, "invalid_baseline"
    if data.ndim != 2 or data.shape[0] < GUARD_MIN_N or data.shape[1:] != (6,) or not np.isfinite(data).all():
        return None, "invalid_baseline"
    if np.max(np.std(data, axis=0)) > 5 or np.max(np.abs(np.mean(data, axis=0))) > 400:
        return None, "unstable_or_loaded_baseline"
    return np.mean(data, axis=0), None


def trial_frames(trial, dt):
    """stationary_tracking.execute_trial 과 같은 프레임 · 시간 · 속도 · 가속 검사 → (frames, end_index)."""
    approach = np.asarray(trial["approach_frames"], float)
    path_frames = np.asarray(trial["frames"], float)
    motion_frames = np.vstack([approach[:-1], path_frames])
    return_frames = np.asarray(trial["return_frames"], float)
    if (
        motion_frames.ndim != 2 or motion_frames.shape[1] != 7 or len(motion_frames) < 2
        or not np.isfinite(motion_frames).all()
        or return_frames.ndim != 2 or return_frames.shape[1] != 7 or len(return_frames) < 2
        or not np.isfinite(return_frames).all()
        or dt != 0.01 or trial["hold_s"] != 15.0
    ):
        raise ValueError("invalid prepared trial dimensions/timing")
    end_index = len(motion_frames) - 1
    hold_frames = np.repeat(motion_frames[-1][None, :], int(trial["hold_s"] / dt), axis=0)
    frames = np.vstack([motion_frames, hold_frames, return_frames[1:]])
    if np.max(np.abs(np.diff(frames, axis=0) / dt)) > 0.05 + 1e-8:
        raise ValueError("prepared trial exceeds velocity limit")
    if len(frames) > 2 and np.max(np.abs(np.diff(frames, n=2, axis=0) / dt**2)) > 0.1 + 1e-6:
        raise ValueError("prepared trial exceeds acceleration limit")
    return frames, end_index


class Rebaser:
    """rebase_trial.py 를 이 프로세스에서 그대로 — 충돌 world 만 한 번 만든다(11.09 실측 rebase 2.3 s 중 2.1 s).
    Checker 가 바꾸는 것(data · excluded_pairs)을 매번 build_world 직후 상태로 되돌린다."""

    def __init__(self):
        self._build, self._worlds = W.build_world, {}

    def _world(self, spec):
        import mujoco
        key = repr(spec)
        if key not in self._worlds:
            self._worlds[key] = self._build(spec)
        world = self._worlds[key]
        mujoco.mj_resetData(world.model, world.data)
        world.excluded_pairs = set()
        return world

    def __call__(self, plan, snapshot, trial, out):
        argv, build = sys.argv, W.build_world
        sys.argv = [str(REBASE), "--plan", str(plan), "--snapshot", str(snapshot), "--trial", trial, "--out", str(out)]
        W.build_world = self._world
        try:
            runpy.run_path(str(REBASE), run_name="__main__")
        finally:
            sys.argv, W.build_world = argv, build
        return Path(out) / "plan.json"


def analyze_run(out, rows, plan, trial, path_end, failure, count, side):
    """stationary_tracking.execute_trial 끝의 분석 · 보고와 같다."""
    samples, alignment = S.aligned_samples(rows, side=side)
    if path_end is not None:          # stop/abort 확인 틱은 trial 창에서 뺀다
        samples = [s for s in samples if s["stamp"] <= path_end + trial["hold_s"]]
    summary, t, tip = S.analyze_samples(
        samples,
        cli.tip_fk(Path(plan["inputs"]["urdf"]["path"]), f"{'r' if side == 'right' else 'l'}_hl_index_tip", side),
        path_end=path_end, alignment=alignment, expected_target=trial["target"])
    if failure:
        summary.update(eligible=False, **{"pass": False})
        summary["excluded_reasons"].append(failure)
    summary["commands_sent"] = count
    with REPORT_LOCK:
        S.write_report(out, summary, t, tip)
    return summary


class Analysis(threading.Thread):
    def __init__(self, *args):
        super().__init__(daemon=False)
        self.args, self.error = args, None

    def run(self):
        try:
            analyze_run(*self.args)
        except Exception as exc:          # noqa: BLE001 — 주 스레드가 join 뒤 다시 올린다
            self.error = exc

    def finish(self):
        self.join()
        if self.error is not None:
            raise RuntimeError(f"analysis of {self.args[0]} failed: {self.error!r}") from self.error


class Batch:
    """노드 하나 · 구독 하나로 여러 trial. 구독 QoS 는 run 의 것(stationary_tracking.execute_trial)과 같다."""

    def __init__(self, side):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from rosidl_runtime_py.utilities import get_message
        from rosidl_runtime_py.convert import message_to_ordereddict
        from std_msgs.msg import Bool, String
        from sensor_msgs.msg import JointState

        self.rclpy, self.Bool, self.String, self._to_dict = rclpy, Bool, String, message_to_ordereddict
        self.side, self.other = side, "left" if side == "right" else "right"
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)     # SIGINT · SIGTERM 은 main 이 받는다
        self.node = rclpy.create_node(f"stationary_tracking_{side}")
        self.latest, self.received, self.latched = {}, {}, {}
        self.buffer, self.force = collections.deque(), collections.deque()
        self.sink, self.guard = None, None
        self.diag_topic = f"/policy_control/pd_{side}/diagnostics"
        self.status_topic = f"/policy_control/status/pd_{side}"
        self.hand = f"/hand_{side}"
        qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
        pd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        for topic, typename in recorder.topic_spec(side).items():
            if topic == "/robot_description":
                selected = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
            elif topic in (self.diag_topic, self.status_topic):
                selected = pd_qos
            else:
                selected = qos
            self.node.create_subscription(get_message(typename), topic, lambda m, t=topic: self._on(t, m), selected)
        self.target_pub = self.node.create_publisher(JointState, "/policy_control/joint_target", 10)
        latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ep_pub = self.node.create_publisher(String, f"/policy_control/{side}/episode", latched)
        self.contact_pub = self.node.create_publisher(Bool, f"/policy_control/pd_{side}/contact", 10)
        self.guard_pub = self.node.create_publisher(String, f"/policy_control/pd_{side}/contact_guard", 10)
        self.node.create_timer(.02, self._tick)

    def close(self):
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()

    def _on(self, topic, msg):
        mono = time.monotonic_ns()
        message = self._to_dict(msg)
        row = recorder.event(topic, message, wall_ns=time.time_ns(), mono_ns=mono)
        self.latest[topic], self.received[topic] = message, mono / 1e9
        self.buffer.append(row)
        while mono - self.buffer[0]["recv_mono_ns"] > KEEP_S * 1e9:
            self.buffer.popleft()
        if recorder.is_latched(topic):
            self.latched[topic] = row
        if topic == self.hand + "/joint_forces":
            self.force.append((mono / 1e9, list(message["data"])))
            while mono / 1e9 - self.force[0][0] > KEEP_S:
                self.force.popleft()
        if self.sink is not None:
            rows, stream = self.sink
            rows.append(row)
            stream.write(json.dumps(row) + "\n")

    def _tick(self):
        """stationary_contact_guard.tick — 기준선이 있을 때만(감시 켜진 동안) 낸다."""
        g = self.guard
        if g is None:           # 기준선 전 = 원래 감시의 baseline_pending(contact=True) — trial 사이에도 낸다
            self.contact_pub.publish(self.Bool(data=True))
            self.guard_pub.publish(self.String(data=json.dumps({
                "contact_or_unknown": True, "reason": "baseline_pending", "operator_confirmed_free": True,
                "scope": "hand_sensors_only", "baseline_g": None})))
            return
        now = time.monotonic()
        keys = ((self.hand + "/joint_forces", .1), (self.hand + "/touch_data", .1), (self.hand + "/ecat_status", 2.0))
        stale = any(now - self.received.get(k, 0) > age for k, age in keys)
        reason = "sensor_stale_or_missing" if stale else None
        if not stale and g["baseline"] is not None:
            touch = self.latest[self.hand + "/touch_data"]
            try:
                status = json.loads(self.latest[self.hand + "/ecat_status"]["data"])
            except (ValueError, TypeError):
                status = {}
            reason = assess(list(self.latest[self.hand + "/joint_forces"]["data"]), g["baseline"],
                            list(touch["finger_forces"]) + list(touch["finger_tangentials"]),
                            status if isinstance(status, dict) else {})
        if reason:
            g["latched"] = reason
        reason = g["latched"] or reason
        self.contact_pub.publish(self.Bool(data=reason is not None))
        self.guard_pub.publish(self.String(data=json.dumps({
            "contact_or_unknown": reason is not None, "reason": reason, "operator_confirmed_free": True,
            "scope": "hand_sensors_only", "baseline_g": None if g["baseline"] is None else g["baseline"].tolist()})))

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=min(.02, max(0., end - time.monotonic())))

    def spin_until(self, pred, timeout_s, what):
        """pred 는 0.05 s 마다 본다(spin_once 는 메시지 하나씩이라 매번 보면 settled 가 초당 수천 번 돈다)."""
        end = time.monotonic() + timeout_s
        while not pred():
            if time.monotonic() > end:
                raise ValueError(what)
            self.spin(.05)

    def spin_while(self, alive):
        while alive():
            self.spin(.05)

    def call_spinning(self, fn, *args):
        """fn 을 스레드에서 돌리고 끝날 때까지 구독을 돈다(rebase 동안 버퍼 · 감시 줄이 끊기지 않게)."""
        out = {}

        def target():
            try:
                out["value"] = fn(*args)
            except BaseException as exc:      # noqa: BLE001 — 주 스레드가 다시 올린다
                out["error"] = exc
        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        self.spin_while(worker.is_alive)
        if "error" in out:
            raise out["error"]
        return out["value"]

    def moved_since(self, snapshot):
        """capture 의 팔 q 와 지금 /joint_states 의 최대 차(rad). 없으면 inf."""
        self.spin(.05)
        q = None if "/joint_states" not in self.latest else arm_q({"message": self.latest["/joint_states"]}, self.side)
        if q is None:
            return float("inf")
        return float(np.max(np.abs(np.asarray(q) - json.loads(Path(snapshot).read_text())["q"])))

    def reset(self):
        """다음 trial 의 settle · capture · 감시 기준선은 이 뒤의 줄만 본다."""
        self.buffer.clear()
        self.force.clear()

    def wait_ready(self):
        fresh = lambda: all(time.monotonic() - self.received.get(t, 0) <= .1    # noqa: E731
                            for t in (self.diag_topic, self.status_topic))
        self.spin_until(fresh, READY_TIMEOUT_S, "PD stream stale or missing")

    def capture(self, out):
        """stationary_tracking.capture 와 같은 산출물(raw/ = stationary_recorder 형식). 창 = 정지 판정 1 s."""
        self.spin_until(lambda: settled(self.buffer, self.side, time.monotonic_ns(), SETTLE_WINDOW_S), SETTLE_TIMEOUT_S,
                        f"arm did not settle within {SETTLE_TIMEOUT_S:.0f} s "
                        f"(joint range > {SETTLE_TOL_RAD * 1e3:.1f} mrad over {SETTLE_WINDOW_S:.0f} s)")
        self.force.clear()           # 감시 기준선은 정지 판정 뒤의 힘만
        end_ns, start_wall = time.monotonic_ns(), time.time_ns() - int(SETTLE_WINDOW_S * 1e9)
        window = [r for r in self.buffer if r["recv_mono_ns"] >= end_ns - SETTLE_WINDOW_S * 1e9]
        # shortcut: 래치 토픽은 새 구독이 처음 받는 마지막 값을 앞에 둔다(받은 시각은 원래 것), 새 구독과 똑같지는 않다
        events = [r for t, r in self.latched.items() if all(w["topic"] != t for w in window)] + window
        S.new_run(out, {"urdf": cli.URDF}, {"mode": "capture", "seconds": SETTLE_WINDOW_S, "side": self.side},
                  repo=cli.ROOT)
        raw = out / "raw"
        raw.mkdir(parents=True, exist_ok=False)
        spec = recorder.topic_spec(self.side)
        counts = collections.Counter(r["topic"] for r in events)
        with (raw / "events.jsonl").open("x") as stream:
            stream.writelines(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in events)
        recorder.save_npz(raw / "events.npz", events)
        (raw / "metadata.json").write_text(json.dumps({
            "schema": "readonly_ros_probe_v1", "side": self.side, "seconds": SETTLE_WINDOW_S,
            "start_wall_ns": start_wall, "ROS_DOMAIN_ID": os.environ["ROS_DOMAIN_ID"],
            "FASTRTPS_DEFAULT_PROFILES_FILE": os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"),
            "topics": spec, "commands_sent": 0,
            "time_note": "per-topic callbacks, not synchronized sampling; align timestamps in analysis",
            "source": "stationary_batch rolling window after settle", "end_wall_ns": time.time_ns(),
            "counts": dict(counts), "n_events": len(events), "missing_topics": [t for t in spec if not counts[t]],
        }, ensure_ascii=False, indent=2) + "\n")
        (out / "events.jsonl").symlink_to("raw/events.jsonl")
        _, alignment = S.aligned_samples(events, side=self.side)
        summary = {"pass": False, "eligible": False, "excluded_reasons": ["capture_only"],
                   "alignment": alignment, "commands_sent": 0}
        try:
            pose = S.capture_pose(events, side=self.side)
        except ValueError as exc:
            summary["excluded_reasons"].append(str(exc))
            pose = None
        if pose is not None:
            cli.dump(out / "snapshot.json", {
                "id": "captured_pose", "split": "fit", **pose, "source": str(out / "events.jsonl"),
                "note": "Operator must verify free-space pose; capture does not certify collision safety."})
        with REPORT_LOCK:
            S.write_report(out, summary, [], [])
        if pose is None:
            raise ValueError("capture failed: " + "; ".join(summary["excluded_reasons"][1:]))
        return out / "snapshot.json"

    def run(self, plan_path, trial_id, out):
        """stationary_tracking.run(--execute) 와 같은 manifest · 입력 해시 검사 → 실행 → (rows, path_end, failure, count)."""
        plan = json.loads(plan_path.read_text())
        if plan.get("side", "right") != self.side:
            raise ValueError(f"plan side {plan.get('side', 'right')} != {self.side}")
        trials = [t for t in plan["trials"] if t["id"] == trial_id]
        if len(trials) != 1:
            raise ValueError("select exactly one prepared --trial; no automatic candidate progression")
        trial = trials[0]
        manifest = S.new_run(
            out,
            {"plan": plan_path, **{key: value["path"] for key, value in plan.get("inputs", {}).items()}},
            {"mode": "run", "side": self.side, "trial": trial["id"], "stage": plan["stage"],
             "execute_requested": True, "kp": plan["kp"], "kd": plan["kd"],
             "compensation": plan["compensation"], "hand": plan["fixed_hand"]},
            repo=cli.ROOT)
        for entry in plan["inputs"].values():
            if S.file_hash(entry["path"]) != entry["sha256"]:
                raise ValueError(f"prepared input changed: {entry['path']}")
        return plan, trial, self._execute(plan, trial, manifest, out)

    def _guard_on(self):
        """정지 판정 뒤 받은 힘이 3 s 를 덮고 그 3 s 동안 팔이 멈춰 있을 때 그 3 s 로 기준선(아니면 기다림 · 8 s 에 멈춤)."""
        self.spin(.05)                      # 쌓인 콜백부터 비운다 — 오래된 힘으로 잡지 않게

        def ready():
            now = time.monotonic()
            return (bool(self.force) and self.force[0][0] <= now - GUARD_WINDOW_S and self.force[-1][0] >= now - .1
                    and sum(t >= now - GUARD_WINDOW_S for t, _ in self.force) >= GUARD_MIN_N
                    and settled(self.buffer, self.side, time.monotonic_ns(), window_s=GUARD_WINDOW_S))
        self.spin_until(ready, GUARD_TIMEOUT_S,
                        f"guard baseline: no {GUARD_WINDOW_S:.0f} s of fresh hand force while the arm is still")
        start = time.monotonic() - GUARD_WINDOW_S
        baseline, latched = guard_baseline([f for t, f in self.force if t >= start])
        self.guard = {"baseline": baseline, "latched": latched}
        if baseline is not None:
            self.node.get_logger().info("fixed free-space force baseline: " + json.dumps(baseline.tolist()))

    def _execute(self, plan, trial, manifest, out):
        """stationary_tracking.execute_trial 과 같은 검사 · 발행. 다른 점: 노드가 이미 떠 있어 5 s 대기 대신
        감시 기준선을 잡고 pd 진단이 접촉 없음을 볼 때까지(최대 2 s)만 돈다."""
        from policy_control import codec
        from policy_control.fk_numpy import UrdfTree
        from policy_control.sources import load_profile
        from replay_to_pd import replay_episode_id

        side, other_side, rclpy, node = self.side, self.other, self.rclpy, self.node
        latest, received = self.latest, self.received
        diag_topic, status_topic = self.diag_topic, self.status_topic
        joint_names = tuple(plan.get("joint_names", S.arm_joint_names(side)))
        count, failure, path_end = 0, None, None
        gc_was_enabled, gc_paused = gc.isenabled(), False
        episode_commands = 0
        episode = replay_episode_id()
        manifest["metadata"]["episode"] = episode
        profile = load_profile(Path(plan["inputs"]["profile"]["path"]))
        tree = UrdfTree(plan["inputs"]["urdf"]["path"])
        rows = list(self.buffer)           # 동작 전 정지 구간 — 원래 run 이 처음 5 s 동안 받던 줄
        stream = (out / "events.jsonl").open("x")
        stream.writelines(json.dumps(r) + "\n" for r in rows)
        latest.pop("/policy_control/joint_target", None)   # 앞 trial 의 우리 목표가 '다른 발행자' 로 보이지 않게
        self.sink = (rows, stream)

        def current():
            now = time.monotonic()
            stale_topics = [topic for topic in (diag_topic, status_topic) if now - received.get(topic, 0) > 0.1]
            if stale_topics:
                raise ValueError("PD stream stale or missing: " + ", ".join(stale_topics))
            d, status = json.loads(latest[diag_topic]["data"]), json.loads(latest[status_topic]["data"])
            if (not status.get("execute") or not status.get("ok") or status.get("phase") != "TRACKING"
                    or status.get("estop") or not d["gains"].get("runtime", {}).get("verified")):
                raise ValueError("PD not ready or runtime gains unverified")
            S.validate_live_diagnostics(d)
            for key in ("kp", "kd"):
                if not np.allclose(d["gains"]["real_" + key], plan[key], rtol=0, atol=1e-6):
                    raise ValueError("live gain mismatch")
            if d.get("compensation_config") != plan["compensation"]:
                raise ValueError("live compensation configuration mismatch")
            hand = tree.expand_mimic(dict(zip(d["hand_names"], d["measured_hand"])))
            if any(j not in hand or abs(hand[j] - q) > 0.02 for j, q in plan["fixed_hand"].items()):
                raise ValueError("hand posture changed")
            live = {}
            for topic in ("/joint_states", "/head/joint_states", "/hand_left/joint_states", "/hand_right/joint_states"):
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
                observed = latest.get("/policy_control/joint_target", {}).get("header", {}).get("frame_id", "")
                if observed and observed.split(":")[0] != str(episode):
                    raise ValueError("concurrent target publisher detected")
            return d

        def contact_clear():
            try:
                return json.loads(latest[diag_topic]["data"]).get("contact") is False
            except (KeyError, ValueError, TypeError, AttributeError):
                return False

        try:
            self._guard_on()
            try:        # 감시가 낸 접촉 없음이 pd 진단에 닿을 때까지 — 안 닿으면 아래 current() 가 사유를 낸다
                self.spin_until(contact_clear, 2.0, "contact not clear")
            except ValueError:
                self.node.get_logger().warning(
                    "pd diagnostics contact not clear after 2 s (guard reason: "
                    f"{self.guard and (self.guard['latched'] or 'none')}, diag: {latest.get(diag_topic, {}).get('data', 'missing')[:200]})")
            d = current()
            publishers = node.get_publishers_info_by_topic("/policy_control/joint_target")
            if len(publishers) != 1 or publishers[0].node_name != node.get_name():
                raise ValueError("another target publisher/jog is active; concurrent correction is forbidden")
            if np.max(np.abs(np.asarray(d["measured"]) - trial["approach_start_q"])) > 0.002:
                raise ValueError("live arm does not match the captured-home approach start")
            # Another arm moving invalidates the prechecked scene; check live source names.
            state = latest.get("/joint_states", {})
            if time.monotonic() - received.get("/joint_states", 0) > 0.1:
                raise ValueError("other arm scene state missing")
            source = dict(zip(state.get("name", []), state.get("position", [])))
            for i in range(1, 8):
                name = f"openarm_{other_side}_joint{i}"
                if name not in source or abs(source[name] - plan["scene"][f"{other_side[0]}_aj_{i}"]) > 0.002:
                    raise ValueError("other arm differs from collision-checked scene")
            dt = plan["dt"]
            frames, end_index = trial_frames(trial, dt)
            # Retained recording rows make a generation-2 collection exceed the 20 ms command deadline.
            if gc_was_enabled:
                gc.collect()
                gc.disable()
                gc_paused = True
            self.ep_pub.publish(self.String(data=json.dumps({"event": "reset", "episode": episode})))
            episode_commands += 1
            velocity = np.gradient(frames, dt, axis=0)
            velocity[0] = velocity[-1] = 0.0
            start = time.monotonic()
            path_end = time.time() + end_index * dt
            for i, (q, v) in enumerate(zip(frames, velocity)):
                while time.monotonic() < start + i * dt:
                    rclpy.spin_once(node, timeout_sec=min(0.002, max(0.0, start + i * dt - time.monotonic())))
                if time.monotonic() - (start + i * dt) > 0.02:
                    raise ValueError("command scheduling delay >20ms; stop trial")
                current()
                self.target_pub.publish(codec.encode_joint_target(joint_names, q, v, str(episode), i, stamp=time.time()))
                count += 1
                rclpy.spin_once(node, timeout_sec=0.0)
        except (Exception, KeyboardInterrupt) as exc:        # noqa: BLE001 — 무엇이든 failure 로 abort 를 낸다
            failure = (str(exc) if isinstance(exc, (ValueError, KeyError, TypeError, RuntimeError))
                       else f"{type(exc).__name__}: {exc}") if str(exc) else (
                "operator_interrupt" if isinstance(exc, KeyboardInterrupt) else type(exc).__name__)
        finally:
            SIG["defer"] = True             # 첫 신호가 여기서 오면 abort · manifest 를 끝낸 뒤 main 이 멈춘다
            if episode_commands:
                # Existing stop/abort makes PD hold; never releases supporting torque.
                self.ep_pub.publish(self.String(data=json.dumps(
                    {"event": "abort" if failure else "stop", "episode": episode})))
                episode_commands += 1
                self.spin(0.5)
            self.guard, self.sink = None, None
            stream.close()
            if gc_paused and gc_was_enabled:
                gc.enable()
            manifest["commands_sent"] = count
            manifest["episode_commands_sent"] = episode_commands
            try:
                gains = json.loads(latest[diag_topic]["data"]).get("gains") if diag_topic in latest else None
            except (ValueError, TypeError, AttributeError):
                gains = None
            manifest["metadata"]["runtime_gains"] = gains
            manifest["metadata"].update(planned_path_end=path_end, failure=failure)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            self.reset()
            SIG["defer"] = False
        return rows, path_end, failure, count


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=("right", "left"), required=True)
    p.add_argument("--plan", type=Path, required=True, help="실험 setup/plan/plan.json(rebase 원본)")
    p.add_argument("--trial", nargs=3, action="append", required=True, metavar=("ID", "REBASE_DIR", "RUN_DIR"),
                   help="차례대로 돈다. REBASE_DIR 에 capture/ · plan/, RUN_DIR 에 run 산출물")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    args.trial = [(t, Path(r), Path(o)) for t, r, o in args.trial]
    ids = [t for t, _, _ in args.trial]
    if len(set(ids)) != len(ids):
        p.error("trial ids repeat")
    for _, r, o in args.trial:
        for path in (r / "capture", r / "plan", o):
            if path.exists():
                p.error(f"{path} exists — runs/<trial> and rebase attempts are never rewritten")
    if not args.plan.is_file():
        p.error(f"{args.plan} missing")
    return args


SIG = {"count": 0, "defer": False}


def _interrupt(*_):
    """첫 신호만 KeyboardInterrupt(정리 중이면 미룬다), 두 번째부터 무시 — abort · manifest · 해제가 끝나게."""
    SIG["count"] += 1
    if SIG["count"] == 1 and not SIG["defer"]:
        raise KeyboardInterrupt


def _check_interrupt():
    if SIG["count"]:
        raise KeyboardInterrupt


def main(argv=None):
    args = parse(argv)
    if not args.execute:
        for t, r, o in args.trial:
            print(f"{t}: capture {r / 'capture'} → rebase {r / 'plan'} → run {o}")
        print(f"Dry run: 0 commands, {len(args.trial)} trial(s).")
        return 0
    if os.environ.get("ROS_DOMAIN_ID") not in ("50", "97", "99"):
        raise SystemExit("explicit ROS_DOMAIN_ID 50 (real), 97/99 (fake) required")
    for sig in (signal.SIGINT, signal.SIGTERM):     # 콘솔 정지는 그룹 SIGTERM — 도는 trial 을 abort(=pd hold) 로 끝낸다
        signal.signal(sig, _interrupt)
    batch, rebase, pending, step, tid = None, Rebaser(), None, "start", ""
    try:
        batch = Batch(args.side)
        batch.wait_ready()
        for tid, rdir, run_dir in args.trial:
            _check_interrupt()
            if pending is not None:       # 앞 trial 분석은 capture 전에 끝낸다(20 ms 명령 마감) — 그동안도 구독을 돈다
                step = "analysis"
                batch.spin_while(pending.is_alive)
                pending.finish()
                pending = None
            plan_path = None
            for attempt in range(RECAPTURE_MAX):
                capture = rdir / "capture" if attempt == 0 else rdir / f"capture_{attempt + 1}"
                step = "capture"
                print(f"@@ {tid} capture", flush=True)
                snapshot = batch.capture(capture)
                step = "rebase"
                print(f"@@ {tid} rebase", flush=True)
                if attempt:               # 앞 시도 plan 은 지우지 않고 옆에 둔다(capture 와 짝)
                    (rdir / "plan").rename(rdir / f"plan_{attempt}")
                plan_path = batch.call_spinning(rebase, args.plan, snapshot, tid, rdir / "plan")
                moved = batch.moved_since(snapshot)
                if moved <= RECAPTURE_TOL_RAD:
                    break
                print(f"arm moved {moved * 1e3:.1f} mrad during rebase — capture again", flush=True)
            else:
                raise ValueError(f"arm kept moving: {RECAPTURE_MAX} captures, last move {moved * 1e3:.1f} mrad")
            _check_interrupt()
            step = "trial"
            print(f"@@ {tid} trial", flush=True)
            plan, trial, (rows, path_end, failure, count) = batch.run(plan_path, tid, run_dir)
            job = Analysis(run_dir, rows, plan, trial, path_end, failure, count, args.side)
            if failure:
                print(f"Trial stopped: {failure}; commands_sent={count}", file=sys.stderr, flush=True)
                job.run()
                job.finish()
                print(f"Trial stopped: {failure}; commands_sent={count}; report={run_dir / 'report.md'}",
                      file=sys.stderr, flush=True)
                return 1
            print(f"Trial complete; commands_sent={count}; report={run_dir / 'report.md'}", flush=True)
            job.start()
            pending = job
        if pending is not None:
            batch.spin_while(pending.is_alive)
            pending.finish()
            pending = None
        print("@@ done", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:          # noqa: BLE001 — 사유 한 줄을 남기고 멈춘다
        print(f"{step} {tid}: {type(exc).__name__}: {exc}" if str(exc) else f"{step} {tid}: {type(exc).__name__}",
              file=sys.stderr, flush=True)
        return 1
    finally:
        if pending is not None:
            pending.join()                # 끝난 trial 의 summary.json 은 남긴다
        if batch is not None:
            batch.close()


if __name__ == "__main__":
    raise SystemExit(main())

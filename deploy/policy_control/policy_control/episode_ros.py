"""에피소드 실행기의 ROS 쪽 — 정책 노드 · pd · 인지에 이미 있는 길만 쓴다(새로 관절 명령을 내지 않는다).

  정책   /policy_control/<팔>/episode/{reset,start,stop} (rh_aglt_node · rh_place_node, ns = 팔) · 이벤트 /policy_control/<팔>/episode
         aglt 는 reset 뒤 /policy_control/<팔>/goal 에 SETTING 을 주고(노드 stop_on_target 이 도달 때 끝낸다), place 는 그대로.
  궤적   rehome = plan_rehome(실측 → RRT, 로봇 세계) → check_path_start → replay_to_pd --execute → pd goto_home 정착.
         물체를 든 손이 있는 팔은 거부한다(계획기가 든 컵을 모른다).
  배치   snapshot = 물체 토픽(FP++)을 창(1.5 s) 동안 모아 정지 · 신선이면 중앙값을 기록 → /episode/objects/<이름>/pose 로
         30 Hz 다시 낸다(집기 전 정지 컵 — 10.04 사용자 "처음에 FP++ 로 각 컵 배치를 한 번에 기록"). 집힌 물체는 그만 낸다.
  홀더   고정(10.04 사용자) — holder_poses 파일(cup_holder_pose_node --write)을 /objects/cup_holder_<id>/pose 로 latched 발행.
  안전   pd 상태(HOLD · estop)를 보고 있다가 정책 중이면 바로 끝낸다. safe_stop = 떠 있는 정책 노드 episode/stop(pd 가 붙든다) ·
         돌던 재생 프로세스 SIGTERM.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from policy_control import _paths
from policy_control.episode_failure import NodeResult
from policy_control.episode_spec import EMPTY, Episode, Node
from policy_control.rh_place import HOLDER_FLOOR_Z

NS = "/policy_control"
OBJECT_RELAY = "/episode/objects/{}/pose"
SNAPSHOT_WINDOW_S = 1.5
SNAPSHOT_STILL_M = 0.005          # 창 안 위치 퍼짐 — 넘으면 컵이 움직이는 중
SNAPSHOT_MIN_FRAMES = 5
POLICY_MARGIN_S = 10.0            # 정책 에피소드 시간 + 이만큼 안에 이벤트가 안 오면 stop → timeout
SEAT_DZ_CYL60 = HOLDER_FLOOR_Z + 0.085
TOOLS = Path(__file__).resolve().parents[1] / "tools"
SIDES = ("right", "left")


def _q(msg) -> tuple:
    p, o = msg.pose.position, msg.pose.orientation
    return (p.x, p.y, p.z), (o.w, o.x, o.y, o.z)


def load_holder_poses(path: Path) -> dict[int, tuple]:
    """holder_poses 파일 → {마커 id: (pos, yaw)}. 없으면 빈 dict."""
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {int(h["marker_id"]): (tuple(float(v) for v in h["position"]), float(h.get("yaw_rad", 0.0)))
            for h in (raw.get("holders") or {}).values()}


def snapshot_of(frames: list) -> tuple | None:
    """창 안 프레임들 → (pos 중앙값, 마지막 quat) — 프레임이 모자라거나 움직이면 None."""
    if len(frames) < SNAPSHOT_MIN_FRAMES:
        return None
    pos = np.array([f[0] for f in frames], float)
    if float(np.max(np.linalg.norm(pos - np.median(pos, axis=0), axis=1))) > SNAPSHOT_STILL_M:
        return None
    return tuple(float(v) for v in np.median(pos, axis=0)), tuple(frames[-1][1])


def cup_in_holder(cup_pos, holder_pos, seat_dz: float, xy_tol: float = 0.035, z_tol: float = 0.03) -> bool:
    c, h = np.asarray(cup_pos, float), np.asarray(holder_pos, float)
    return bool(np.linalg.norm(c[:2] - h[:2]) < xy_tol and abs(c[2] - (h[2] + seat_dz)) < z_tol)


class RosExecutor:
    """EpisodeManager 의 실행기 — rclpy 노드(콜백은 다른 스레드의 executor 가 돌린다)를 받아 쓴다."""

    def __init__(self, node, ep: Episode, *, robot: str = "rh56f1", run_dir: Path | None = None,
                 contract: str = "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json") -> None:
        from geometry_msgs.msg import Point, PoseStamped
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self.node, self.ep, self.robot = node, ep, robot
        self.run_dir = run_dir or (_paths.SIM2REAL / "logs" / "episode" / time.strftime("%Y%m%d_%H%M%S"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.contract = str(_paths.SIM2REAL / contract)
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._events: dict[str, dict] = {}
        self._status: dict[str, dict] = {}
        self._pd: dict[str, dict] = {}
        self._frames: dict[str, list] = {name: [] for name in ep.objects}
        self._goal_result: dict[str, dict] = {}
        self._relay: dict[str, tuple] = {}            # 물체 → (pos, quat) — snapshot 뒤 집힐 때까지
        self._proc: subprocess.Popen | None = None
        self._Point, self._Trigger, self._Pose = Point, Trigger, PoseStamped
        #: 서비스 응답은 다른 콜백(예: 노드의 stop 서비스) 안에서도 받아야 한다 — 서로 막지 않는 그룹
        self._cbg = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for side in SIDES:
            node.create_subscription(String, f"{NS}/{side}/episode", self._on_json(self._events, side), latched)
            node.create_subscription(String, f"{NS}/{side}/goal_result", self._on_json(self._goal_result, side), latched)
            node.create_subscription(String, f"{NS}/status/pd_{side}", self._on_json(self._pd, side), QoSProfile(depth=10))
            for kind in ("rh_aglt_node", "rh_place_node"):
                name = f"{kind}_{side}"
                node.create_subscription(String, f"{NS}/status/{name}", self._on_json(self._status, name), QoSProfile(depth=10))
        for name, obj in ep.objects.items():
            node.create_subscription(PoseStamped, obj["topic"], self._on_frame(name), qos_profile_sensor_data)
        self._goal_pub = {s: node.create_publisher(Point, f"{NS}/{s}/goal", QoSProfile(depth=10)) for s in SIDES}
        self._relay_pub = {name: node.create_publisher(PoseStamped, OBJECT_RELAY.format(name), 10) for name in ep.objects}
        self._holder_pub = {}
        self.holders = load_holder_poses(_paths.SIM2REAL / ep.holder_poses) if ep.holder_poses else {}
        for hid in sorted(set(ep.holders.values())):
            self._holder_pub[hid] = node.create_publisher(PoseStamped, f"/objects/cup_holder_{hid}/pose", latched)
        self.publish_holders()
        node.create_timer(1.0 / 30.0, self._relay_tick)

    # ---------------------------------------------------------------- 입력
    def _on_json(self, store: dict, key: str):
        def cb(msg) -> None:
            try:
                body = json.loads(msg.data)
            except ValueError:
                return
            with self._lock:
                store[key] = {**body, "_t": time.monotonic()}
        return cb

    def _on_frame(self, name: str):
        def cb(msg) -> None:
            with self._lock:
                fr = self._frames[name]
                fr.append((*_q(msg), time.monotonic()))
                cutoff = time.monotonic() - SNAPSHOT_WINDOW_S
                while fr and fr[0][2] < cutoff:
                    fr.pop(0)
        return cb

    def _pose_msg(self, pos, quat, frame: str = "base_link"):
        m = self._Pose()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = frame
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (float(v) for v in quat)
        return m

    def publish_holders(self) -> list[str]:
        missing = []
        for hid, pub in self._holder_pub.items():
            if hid not in self.holders:
                missing.append(f"holder {hid}")
                continue
            pos, yaw = self.holders[hid]
            pub.publish(self._pose_msg(pos, (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))))
        return missing

    def _relay_tick(self) -> None:
        with self._lock:
            relay = dict(self._relay)
        for name, (pos, quat) in relay.items():
            self._relay_pub[name].publish(self._pose_msg(pos, quat))

    # ---------------------------------------------------------------- 서비스
    def _call(self, path: str, timeout: float = 5.0) -> tuple[bool, list]:
        cli = self.node.create_client(self._Trigger, path, callback_group=self._cbg)
        try:
            if not cli.wait_for_service(timeout_sec=2.0):
                return False, [f"{path}: 서비스가 없다(노드가 떠 있는가)"]
            fut = cli.call_async(self._Trigger.Request())
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < timeout:
                time.sleep(0.02)
            if not fut.done():
                return False, [f"{path}: 응답 없음({timeout:.0f} s)"]
            res = fut.result()
            try:
                body = json.loads(res.message)
                return bool(body.get("ok", res.success)), list(body.get("reasons", []))
            except ValueError:
                return bool(res.success), [res.message] if res.message else []
        finally:
            self.node.destroy_client(cli)

    def _pd_fault(self, sides) -> str | None:
        with self._lock:
            for s in sides:
                st = self._pd.get(s) or {}
                if st.get("estop") or st.get("phase") == "HOLD":
                    return f"pd_{s} {'estop' if st.get('estop') else 'HOLD'} {st.get('reasons', '')}"
        return None

    # ---------------------------------------------------------------- 실행기 계약
    def snapshot(self, node: Node, world) -> tuple[NodeResult, dict]:
        deadline = time.monotonic() + SNAPSHOT_WINDOW_S + 3.0
        names = [n for n in self.ep.objects if (world.objects.get(n) or {}).get("at", "table") == "table"]
        got, why = {}, {}
        while time.monotonic() < deadline and len(got) < len(names) and not self.cancel.is_set():
            time.sleep(0.1)
            with self._lock:
                frames = {n: list(self._frames[n]) for n in names}
            for n in names:
                fresh = [f for f in frames[n] if time.monotonic() - f[2] < SNAPSHOT_WINDOW_S]
                snap = snapshot_of(fresh)
                if snap is not None:
                    got[n] = snap
                else:
                    why[n] = f"{len(fresh)} 프레임" + (" · 움직임" if len(fresh) >= SNAPSHOT_MIN_FRAMES else "")
        missing = [n for n in names if n not in got]
        if missing:
            return NodeResult("refused", "cup pose missing — " + ", ".join(f"{n}({why.get(n, '없음')})" for n in missing)), {}
        with self._lock:
            self._relay.update(got)
        (self.run_dir / "scene.json").write_text(json.dumps({"at": time.strftime("%F %T"), "objects": got}, indent=2))
        return NodeResult("completed", "", signals={"objects": sorted(got)}), got

    def run_policies(self, node: Node, plans: list, world) -> list:
        out: list = [None] * len(plans)

        def one(i, p):
            try:
                out[i] = self._policy(p)
            except Exception as exc:                  # 스레드 안 예외는 결과로
                out[i] = NodeResult("error", f"{type(exc).__name__}: {exc}", role=p.role)
        threads = [threading.Thread(target=one, args=(i, p), daemon=True) for i, p in enumerate(plans)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for p, res in zip(plans, out):               # 집힌 물체는 더 다시 내지 않는다(이제 손바닥 FK)
            if p.kind == "aglt" and res is not None and res.status == "completed" and p.target_object:
                with self._lock:
                    self._relay.pop(p.target_object, None)
        return out

    def _policy(self, p) -> NodeResult:
        side = p.side
        node_name = {"aglt": "rh_aglt_node", "place": "rh_place_node"}.get(p.kind)
        if node_name is None:
            return NodeResult("refused", f"{p.kind} 정책 노드가 아직 없다", role=p.role)
        status_key = f"{node_name}_{side}"
        t0 = time.monotonic()
        with self._lock:
            last_ep = int((self._events.get(side) or {}).get("episode", 0))
        ok, why = self._call(f"{NS}/{side}/episode/reset")
        if not ok:
            return NodeResult("refused", "; ".join(why), role=p.role)
        if p.kind == "aglt" and p.setting is not None:
            with self._lock:
                self._goal_result.pop(side, None)
            x, y, z = p.setting
            self._goal_pub[side].publish(self._Point(x=float(x), y=float(y), z=float(z)))
            deadline = time.monotonic() + 3.0
            res = None
            while time.monotonic() < deadline:
                with self._lock:
                    res = self._goal_result.get(side)
                if res and [round(v, 4) for v in res.get("request", [])] == [round(float(v), 4) for v in p.setting]:
                    break
                time.sleep(0.05)
            if not res or not res.get("ok"):
                return NodeResult("refused", f"goal {p.setting} refused: {(res or {}).get('reasons', 'no answer')}", role=p.role)
        ok, why = self._call(f"{NS}/{side}/episode/start")
        if not ok:
            return NodeResult("refused", "; ".join(why), role=p.role)
        limit = 15.0 + POLICY_MARGIN_S
        while True:
            time.sleep(0.05)
            if self.cancel.is_set():
                return NodeResult("aborted", "episode runner stop", time.monotonic() - t0, p.role)
            fault = self._pd_fault([side])
            if fault:
                self._call(f"{NS}/{side}/episode/stop")
                return NodeResult("completed", fault, time.monotonic() - t0, p.role, {"pd_fault": fault})
            with self._lock:
                ev = dict(self._events.get(side) or {})
                st = dict(self._status.get(status_key) or {})
            if int(ev.get("episode", 0)) > last_ep and ev.get("event") in ("stop", "abort"):
                break
            if time.monotonic() - t0 > limit:
                self._call(f"{NS}/{side}/episode/stop")
                return NodeResult("timeout", f"no stop event in {limit:.0f} s", time.monotonic() - t0, p.role)
        reasons = "; ".join(ev.get("reasons", []))
        dur = time.monotonic() - t0
        if ev.get("event") == "abort":
            return NodeResult("aborted", reasons, dur, p.role)
        if "episode time" in reasons:
            return NodeResult("timeout", reasons, dur, p.role, self._signals(p, st))
        return NodeResult("completed", reasons, dur, p.role, self._signals(p, st))

    def _signals(self, p, st: Mapping) -> dict:
        if p.kind == "aglt":
            cup = (st.get("cup") or {}).get("arm") or {}
            return {"attached": cup.get("source") == "attached" if cup else None,
                    "dropped": bool(cup.get("releases", 0)), "setting_err_m": st.get("kp_dist"), "reached": st.get("reached")}
        if p.kind == "place":
            pl = st.get("place") or {}
            empty = pl.get("phase") == "done"
            sig = {"hand_empty": empty, "opened": pl.get("opened"), "tact_max": pl.get("tact_max"), "jf_max": pl.get("jf_max")}
            pos = self._fresh_object(p.source_object)
            hold = self.holders.get(p.holder_id) if p.holder_id is not None else None
            if pos is not None and hold is not None:
                sig["cup_in_holder"] = cup_in_holder(pos, hold[0], SEAT_DZ_CYL60)
                sig["cup_pos"] = [round(v, 4) for v in pos]
            return sig
        return {}

    def _fresh_object(self, name: str | None, wait_s: float = 2.0):
        """놓은 뒤 물체 자리 — 그 물체의 FP++ 토픽 새 프레임(창 안 중앙값). 없으면 None(판정 보류)."""
        if not name or name not in self._frames:
            return None
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            with self._lock:
                fr = [f for f in self._frames[name] if time.monotonic() - f[2] < 0.5]
            if len(fr) >= 3:
                return tuple(float(v) for v in np.median(np.array([f[0] for f in fr]), axis=0))
            time.sleep(0.1)
        return None

    def run_trajectory(self, node: Node, traj, world) -> NodeResult:
        if traj.kind == "noop":
            return NodeResult("completed", "noop")
        t0 = time.monotonic()
        for side in traj.sides:
            if world.hand(side) != EMPTY:
                return NodeResult("refused", f"{side} hand holds {world.hand(side)} — rehome 계획기가 든 물체를 모른다")
            out = self.run_dir / f"rehome_{side}_{node.id}.npz"
            joints = ",".join(f"{side[0]}_aj_{i}" for i in range(1, 8))
            steps = [
                [sys.executable, str(TOOLS / "plan_rehome.py"), "--side", side, "--robot", self.robot, "--out", str(out)],
                [sys.executable, str(TOOLS / "check_path_start.py"), "--npz", str(out), "--contract", self.contract],
                [sys.executable, str(TOOLS / "replay_to_pd.py"), "--npz", str(out), "--joints", joints, "--rate-scale", "1.0",
                 "--execute"],
                [sys.executable, str(TOOLS / "episode_ctl.py"), "--side", side, "--only", "pd_goto_home",
                 "--service-timeout", "45", "--execute", "--approve", "pd_goto_home"],
            ]
            for argv in steps:
                rc, tail = self._run(argv)
                if self.cancel.is_set():
                    return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
                if rc != 0:
                    return NodeResult("error", f"{Path(argv[1]).name} rc {rc}: {tail}", time.monotonic() - t0)
        return NodeResult("completed", "", time.monotonic() - t0)

    def _run(self, argv: list) -> tuple[int, str]:
        log = self.run_dir / "tools.log"
        with open(log, "a") as fh:
            fh.write(f"\n$ {' '.join(argv)}\n")
            fh.flush()
            self._proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
            rc = self._proc.wait()
            self._proc = None
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
        return rc, tail[0][-300:]

    def prepare(self, step: str, plans: list, world) -> NodeResult:
        if step == "open_hand":
            for side in sorted({p.side for p in plans if p.side in SIDES}):
                rc, tail = self._run([sys.executable, str(TOOLS / "trigger.py"), "pd/hand_release", "--side", side, "--execute"])
                if rc != 0:
                    return NodeResult("error", f"hand_release {side}: {tail}")
            time.sleep(1.0)
            return NodeResult("completed", "")
        if step == "refresh_perception":
            missing = self.publish_holders()
            if missing:
                self.holders = load_holder_poses(_paths.SIM2REAL / self.ep.holder_poses)
                missing = self.publish_holders()
            if missing:
                return NodeResult("refused", f"holder pose missing: {missing} ({self.ep.holder_poses})")
            time.sleep(1.0)
            return NodeResult("completed", "")
        return NodeResult("refused", f"모르는 준비 {step}")

    def safe_stop(self, reason: str) -> None:
        self.cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        for side in SIDES:
            self._call(f"{NS}/{side}/episode/stop", timeout=3.0)

    def joints(self) -> dict:
        return {}

    def clear_cancel(self) -> None:
        self.cancel.clear()

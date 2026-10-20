"""에피소드 실행기의 ROS 쪽 — 정책 노드 · pd · 인지에 이미 있는 길만 쓴다(새로 관절 명령을 내지 않는다).

  정책   /policy_control/<팔>/episode/{reset,start,stop} (rh_aglt_node · rh_place_node, ns = 팔) · 이벤트 /policy_control/<팔>/episode
         aglt 는 reset 뒤 /policy_control/<팔>/goal 에 SETTING 을 주고(노드 stop_on_target 이 도달 때 끝낸다), place 는 그대로.
  궤적   rehome = plan_rehome(실측 → RRT, 로봇 세계) → check_path_start → replay_to_pd --execute → pd goto_home 정착.
         물체를 든 손이 있는 팔은 거부한다(계획기가 든 컵을 모른다).
         carry · tilt(10.17) = plan_<kind>.py(실측 → 계획 · 검사, 움직이지 않음) → replay_to_pd(그 팔 관절) — 손 상태(cups)가 다르면 거부.
  손     hand open = pd hand_release(빈손에서만) · hand grasp = tools/hand_grasp.py(팔 제자리, 닿을 때까지 닫고 쥔 판정) — 10.17 경로 B.
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
import re
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
from policy_control.episode_spec import EMPTY, PLANNED_TRAJ, TRAJ_ASSETS, Episode, Node, episode_sides, hand_contract_file
from policy_control.rh_place import HOLDER_FLOOR_Z

NS = "/policy_control"
OBJECT_RELAY = "/episode/objects/{}/pose"
#: ★10.19 팔마다 띄운 실행기(episode_runner_<팔>)는 자기 이름 아래로 다시 낸다 — 두 팔 에피소드가 같은 물체 이름(CUP)을 써도
#  /episode/objects/CUP/pose 에 두 실행기가 섞여 내지 않게. 기본 실행기(두 팔 tilt 등)는 그대로.
DEFAULT_RUNNER = "episode_runner"


def relay_topic(runner: str, name: str) -> str:
    """snapshot 재발행 토픽 — 기본 실행기는 /episode/objects/<이름>/pose, 그 밖은 /episode/<실행기 꼬리>/objects/<이름>/pose."""
    if runner in ("", DEFAULT_RUNNER):
        return OBJECT_RELAY.format(name)
    tail = runner[len(DEFAULT_RUNNER) + 1:] if runner.startswith(DEFAULT_RUNNER + "_") else runner
    return f"/episode/{tail}/objects/{name}/pose"


def live_topic(topic: str) -> str | None:
    """FP++ 물체 snapshot 토픽 …/pose → 같은 물체의 live 추적 토픽 …/live_pose(object_pose_node 규약). 아니면 None."""
    return topic[: -len("/pose")] + "/live_pose" if topic.endswith("/pose") else None


SNAPSHOT_WINDOW_S = 1.5
SNAPSHOT_STILL_M = 0.005          # 창 안 위치 퍼짐 — 넘으면 컵이 움직이는 중
SNAPSHOT_MIN_FRAMES = 5
LIFT_RISE_MIN_M = 0.07            # ★10.19 1-B lift 판정 — 손바닥 10 cm 중 컵이 7 cm 넘게(결정 10 의 7 cm 위와 같은 선)
POLICY_MARGIN_S = 10.0            # 정책 에피소드 시간 + 이만큼 안에 이벤트가 안 오면 stop → timeout
#: ★10.14 실기 183230: FP++ live 추적이 품질 검사(MASK_TOO_SMALL) 3번 실패마다 다시 등록하며 2.8~3.5 s 끊긴다(5~6 s 마다).
#  정책 reset 은 0.5 s 안에 받은 live 컵만 받아서, 그 사이에 부르면 거부되고 에피소드가 안전 정지했다.
#  reset 은 로봇을 움직이지 않는다 — 이 거부만 이 시간 안에서 다시 부른다(다른 거부는 그대로 판정으로 간다).
RESET_CUP_STALE_TEXT = "FP++ live 컵 자세가"
RESET_CUP_RETRY_S = 6.0
RESET_CUP_RETRY_STEP_S = 0.2


def reset_until_cup_fresh(call, *, retry_s: float = RESET_CUP_RETRY_S, step_s: float = RESET_CUP_RETRY_STEP_S,
                          sleep=time.sleep, clock=time.monotonic, cancelled=lambda: False) -> tuple[bool, list]:
    """reset 을 부르고, live 컵이 끊겨 거부되면 retry_s 안에서 다시 부른다. 마지막 답을 그대로 돌려준다."""
    deadline = clock() + retry_s
    while True:
        ok, why = call()
        stale = not ok and any(RESET_CUP_STALE_TEXT in str(w) for w in why)
        if not stale or cancelled() or clock() >= deadline:
            return ok, why
        sleep(step_s)
SEAT_DZ_CYL60 = HOLDER_FLOOR_Z + 0.085
HAND_OPEN_WAIT_S = 3.0            # 손 펴기 — RH56F1 전 행정 약 1 s(벤더 speedSet 기본) + 여유. 그 안에 실측이 펴져야 한다
HAND_OPEN_TOL_RAD = 0.15          # 펴짐 = 엄지 굽힘 · 네 손가락이 0(편 끝)에서 이 안(엄지 회전은 보지 않는다). pd 홈 손 경로 검사와 같은 값
                                  # (pd 는 손 목표를 한계 안쪽 0.05 로 물린다 — fake 실측 0.07~0.09 에서 섰다)
HAND_OPEN_AXES = ("thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
HAND_STATE = "/hand_{side}/joint_states"   # rh56f1_state_node — canonical r_hj_* · l_hj_*
TOOLS = Path(__file__).resolve().parents[1] / "tools"
SIDES = ("right", "left")
NODES = {"aglt": "rh_aglt_node", "place": "rh_place_node"}


def service_ns(kind: str, side: str) -> str:
    """정책 노드 서비스 ns — 한 팔에 둘이 같이 떠서 aglt = <팔>, place = <팔>_place(미션 에피소드 단계와 같다).
    이벤트는 둘 다 /policy_control/<팔>/episode 에 node 이름을 실어 낸다(pd 가 그 팔에 적용)."""
    return side if kind == "aglt" else f"{side}_{kind}"


def _q(msg) -> tuple:
    p, o = msg.pose.position, msg.pose.orientation
    return (p.x, p.y, p.z), (o.w, o.x, o.y, o.z)


def real_object_name(topic: str) -> str:
    """/objects/<실물>/pose → 실물 이름(config/objects 등록 이름). 아니면 ""."""
    parts = [x for x in str(topic).split("/") if x]
    return parts[1] if len(parts) == 3 and parts[0] == "objects" else ""


#: 홀더 번호 = 이름 끝 숫자(테이블 왼쪽부터, config/cup_holders.yaml) — 에피소드 holders · /objects/cup_holder_N/pose 와 같은 번호.
HOLDER_NAME = re.compile(r"cup_holder_(\d+)")


def load_holder_poses(path: Path) -> dict[int, tuple]:
    """holder_poses 파일 → {홀더 번호: (pos, yaw)}. 없으면 빈 dict.
    ★10.20 번호는 이름에서 읽는다 — marker_id 가 아니다(사용자 "순서를 다시 0,1,2로", 마커 짝은 config/cup_holders.yaml)."""
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[int, tuple] = {}
    for name, h in (raw.get("holders") or {}).items():
        m = HOLDER_NAME.fullmatch(str(name))
        if m is None:
            raise ValueError(f"{path}: 홀더 이름 {name!r} 은 cup_holder_<번호> 꼴이어야 한다")
        out[int(m.group(1))] = (tuple(float(v) for v in h["position"]), float(h.get("yaw_rad", 0.0)))
    return out


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
                 contract: str = "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json",
                 runner: str = DEFAULT_RUNNER) -> None:
        from geometry_msgs.msg import Point, PoseStamped
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self.node, self.ep, self.robot, self.runner = node, ep, robot, runner
        # ★10.19 이 에피소드가 움직이는 팔 — safe_stop 은 이 팔의 정책만 멈춘다(다른 팔 실행기가 따로 도는 정책을 건드리지 않는다).
        #   못 찾으면 두 팔(옛 동작).
        self.sides = tuple(episode_sides(ep)) or SIDES
        self.run_dir = run_dir or (_paths.SIM2REAL / "logs" / "episode" / time.strftime("%Y%m%d_%H%M%S"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.contract = str(_paths.SIM2REAL / contract)
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._events: dict[str, dict] = {}
        self._status: dict[str, dict] = {}
        self._pd: dict[str, dict] = {}
        self._frames: dict[str, list] = {name: [] for name in ep.objects}
        self._live: dict[str, list] = {name: [] for name in ep.objects}   # ★10.20 lift 판정용 FP++ live 추적
        self._hand: dict[str, dict] = {s: {} for s in SIDES}   # ★10.17 손 실측(손 펴짐 확인) — 이름 → (rad, 시각)
        self._goal_result: dict[str, dict] = {}
        self._relay: dict[str, tuple] = {}            # 물체 → (pos, quat) — snapshot 뒤 집힐 때까지
        self._proc: subprocess.Popen | None = None
        self._Point, self._Trigger, self._Pose = Point, Trigger, PoseStamped
        #: 서비스 응답은 다른 콜백(예: 노드의 stop 서비스) 안에서도 받아야 한다 — 서로 막지 않는 그룹
        self._cbg = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for side in SIDES:
            node.create_subscription(String, f"{NS}/{side}/episode", self._on_event, latched)
            node.create_subscription(String, f"{NS}/{side}/goal_result", self._on_json(self._goal_result, side), latched)
            node.create_subscription(String, f"{NS}/status/pd_{side}", self._on_json(self._pd, side), QoSProfile(depth=10))
            for kind in ("rh_aglt_node", "rh_place_node"):
                name = f"{kind}_{side}"
                node.create_subscription(String, f"{NS}/status/{name}", self._on_json(self._status, name), QoSProfile(depth=10))
        for name, obj in ep.objects.items() if any(n.type == "snapshot" for n in ep.nodes) else ():   # 경로 B 는 FP++ 를 안 받는다
            node.create_subscription(PoseStamped, obj["topic"], self._on_frame(name), qos_profile_sensor_data)
        #: ★10.20 실기 1-B: 들림 판정은 live 추적으로 본다. …/pose 는 locate 때 한 번 찍은 자리를 계속 다시 내서
        #  032511 shaker 가 live 로 9.6 cm 들렸는데 before = after 로 0.0 cm 판정 → 들고 멈춤.
        for name in {t.object for t in ep.trajectories.values() if t.kind == "lift" and t.object}:
            topic = live_topic(ep.objects.get(name, {}).get("topic", ""))
            if topic:
                node.create_subscription(PoseStamped, topic, self._on_frame(name, self._live), qos_profile_sensor_data)
        if any(n.type == "hand" for n in ep.nodes):
            from sensor_msgs.msg import JointState
            for side in SIDES:
                node.create_subscription(JointState, HAND_STATE.format(side=side), self._on_hand(side), qos_profile_sensor_data)
        self._goal_pub = {s: node.create_publisher(Point, f"{NS}/{s}/goal", QoSProfile(depth=10)) for s in SIDES}
        self._relay_pub = {name: node.create_publisher(PoseStamped, relay_topic(runner, name), 10) for name in ep.objects}
        self._holder_pub = {}
        self.holders = load_holder_poses(_paths.SIM2REAL / ep.holder_poses) if ep.holder_poses else {}
        for hid in sorted(set(ep.holders.values())):
            self._holder_pub[hid] = node.create_publisher(PoseStamped, f"/objects/cup_holder_{hid}/pose", latched)
        self.publish_holders()
        node.create_timer(1.0 / 30.0, self._relay_tick)

    # ---------------------------------------------------------------- 입력
    def _on_event(self, msg) -> None:
        """팔 이벤트 토픽 — 정책 노드 둘이 같이 내므로 본문 node 로 가른다."""
        try:
            body = json.loads(msg.data)
        except ValueError:
            return
        with self._lock:
            self._events[str(body.get("node", ""))] = {**body, "_t": time.monotonic()}

    def _on_json(self, store: dict, key: str):
        def cb(msg) -> None:
            try:
                body = json.loads(msg.data)
            except ValueError:
                return
            with self._lock:
                store[key] = {**body, "_t": time.monotonic()}
        return cb

    def _on_hand(self, side: str):
        def cb(msg) -> None:
            now = time.monotonic()
            with self._lock:
                self._hand[side].update({n: (float(p), now) for n, p in zip(msg.name, msg.position)})
        return cb

    def _hand_open_err(self, side: str) -> float | None:
        """펴짐 오차 max |q| [rad](엄지 굽힘 · 네 손가락) — 실측이 0.5 s 안에 없으면 None."""
        names = [f"{side[0]}_hj_{a}" for a in HAND_OPEN_AXES]
        with self._lock:
            got = [self._hand[side].get(n) for n in names]
        if any(g is None or time.monotonic() - g[1] > 0.5 for g in got):
            return None
        return max(abs(g[0]) for g in got)

    def _on_frame(self, name: str, store: dict | None = None):
        store = self._frames if store is None else store

        def cb(msg) -> None:
            with self._lock:
                fr = store[name]
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
    def _call(self, path: str, timeout: float = 5.0, wait: float = 2.0) -> tuple[bool, list]:
        cli = self.node.create_client(self._Trigger, path, callback_group=self._cbg)
        try:
            if not cli.wait_for_service(timeout_sec=wait):
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

    def _set_object(self, node: str, name: str, timeout: float = 5.0, wait: float = 2.0) -> tuple[bool, list]:
        """정책 노드 파라미터 object(다물체 aglt 정책의 물체 이름)를 SetParameters 로 — 다음 reset 이 읽는다."""
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters

        path = f"{node}/set_parameters"
        cli = self.node.create_client(SetParameters, path, callback_group=self._cbg)
        try:
            if not cli.wait_for_service(timeout_sec=wait):
                return False, [f"{path}: 서비스가 없다(노드가 떠 있는가)"]
            req = SetParameters.Request()
            req.parameters.append(Parameter(name="object", value=ParameterValue(
                type=ParameterType.PARAMETER_STRING, string_value=name)))
            fut = cli.call_async(req)
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < timeout:
                time.sleep(0.02)
            if not fut.done():
                return False, [f"{path}: 응답 없음({timeout:.0f} s)"]
            bad = [r.reason for r in fut.result().results if not r.successful]
            return (False, [f"{path} object={name}: {bad}"]) if bad else (True, [])
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
        node_name = NODES.get(p.kind)
        if node_name is None:
            return NodeResult("refused", f"{p.kind} 정책 노드가 아직 없다", role=p.role)
        status_key = f"{node_name}_{side}"
        ns = service_ns(p.kind, side)
        self._wait_node_ready(status_key)                 # 연속 실행: snapshot 직후 재발행 컵이 노드에 닿을 시간
        t0 = time.monotonic()
        with self._lock:
            last_ep = int((self._events.get(status_key) or {}).get("episode", 0))
        if p.kind in ("aglt", "place") and getattr(p, "policy_object", None):   # ★10.12 다물체 정책 — reset 이 물체 one-hot 을 고른다
            ok, why = self._set_object(f"/{node_name}_{side}", str(p.policy_object))
            if not ok:
                return NodeResult("refused", "; ".join(why), role=p.role)
        ok, why = reset_until_cup_fresh(lambda: self._call(f"{NS}/{ns}/episode/reset"), cancelled=self.cancel.is_set)
        if not ok:
            return NodeResult("refused", "; ".join(why), role=p.role)
        t0 = time.monotonic()                             # 정책 시간 한도는 reset 이 받아들여진 뒤부터 센다
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
        ok, why = self._call(f"{NS}/{ns}/episode/start")
        if not ok:
            return NodeResult("refused", "; ".join(why), role=p.role)
        limit = float(getattr(p, "max_s", 15.0)) + POLICY_MARGIN_S     # ★10.17 경로 C 는 노드 한도가 길다
        while True:
            time.sleep(0.05)
            if self.cancel.is_set():
                return NodeResult("aborted", "episode runner stop", time.monotonic() - t0, p.role)
            fault = self._pd_fault([side])
            if fault:
                self._call(f"{NS}/{ns}/episode/stop")
                return NodeResult("completed", fault, time.monotonic() - t0, p.role, {"pd_fault": fault})
            with self._lock:
                ev = dict(self._events.get(status_key) or {})
                st = dict(self._status.get(status_key) or {})
            if int(ev.get("episode", 0)) > last_ep and ev.get("event") in ("stop", "abort"):
                break
            if time.monotonic() - t0 > limit:
                self._call(f"{NS}/{ns}/episode/stop")
                return NodeResult("timeout", f"no stop event in {limit:.0f} s", time.monotonic() - t0, p.role)
        reasons = "; ".join(ev.get("reasons", []))
        dur = time.monotonic() - t0
        if ev.get("event") == "abort":
            return NodeResult("aborted", reasons, dur, p.role)
        if "user stop" in reasons:                    # 사람이 정지 바로 멈췄다 — 복구하지 않고 실행기도 멈춘다
            return NodeResult("aborted", f"operator stopped the policy node ({reasons})", dur, p.role)
        if "episode time" in reasons:
            return NodeResult("timeout", reasons, dur, p.role, self._signals(p, st))
        return NodeResult("completed", reasons, dur, p.role, self._signals(p, st))

    def _wait_node_ready(self, status_key: str, timeout: float = 5.0) -> None:
        """정책 노드 status 가 ok(측정 · 컵이 다 들어옴)일 때까지 잠깐 기다린다 — 안 되면 그대로 reset(거부 사유가 판정한다)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.cancel.is_set():
            with self._lock:
                st = dict(self._status.get(status_key) or {})
            if st.get("ok") and time.monotonic() - float(st.get("_t", 0)) < 1.0:
                return
            time.sleep(0.1)

    def _wait_pd_tracking(self, side: str, timeout: float = 5.0) -> bool:
        """정책이 끝난 직후 pd 는 마지막 목표가 0.25 s 넘게 늙어 잠깐 watchdog HOLD 로 갔다가 정지 이벤트의 붙들기로 풀린다
        (10.04 fake 연속 실행). 손 · 궤적 명령 전에 TRACKING 으로 돌아오기를 기다린다 — 안 돌아오면 False(진짜 HOLD)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.cancel.is_set():
            with self._lock:
                st = dict(self._pd.get(side) or {})
            if st.get("phase") == "TRACKING" and not st.get("estop") and time.monotonic() - float(st.get("_t", 0)) < 1.0:
                return True
            time.sleep(0.1)
        return False

    def _pd_view(self, side: str) -> dict:
        with self._lock:
            st = dict(self._pd.get(side) or {})
        return {"pd_phase": st.get("phase"), "pd_reasons": st.get("reasons")}

    def _signals(self, p, st: Mapping) -> dict:
        return {**self._signals_of(p, st), **self._pd_view(p.side)}

    def _signals_of(self, p, st: Mapping) -> dict:
        if p.kind == "aglt":
            cup = (st.get("cup") or {}).get("arm") or {}
            # ★10.14 붙이기를 끈 노드(cup_attach switch False · 위치제어)는 '안 붙음'이 쥠 실패가 아니다 — 모름(None)
            return {"attached": cup.get("source") == "attached" if cup and cup.get("switch", True) else None,
                    "dropped": bool(cup.get("releases", 0)), "setting_err_m": st.get("kp_dist"), "reached": st.get("reached"),
                    "live_off_m": cup.get("check_max_m")}         # ★10.13 붙은 컵 ↔ FP++ 실측 최대 거리(episode_failure)
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

    def _fresh_object(self, name: str | None, wait_s: float = 2.0, store: dict | None = None):
        """놓은 뒤 물체 자리 — 그 물체의 FP++ 토픽 새 프레임(창 안 중앙값). 없으면 None(판정 보류)."""
        store = self._frames if store is None else store
        if not name or name not in store:
            return None
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            with self._lock:
                fr = [f for f in store[name] if time.monotonic() - f[2] < 0.5]
            if len(fr) >= 3:
                return tuple(float(v) for v in np.median(np.array([f[0] for f in fr]), axis=0))
            time.sleep(0.1)
        return None

    def _live_object(self, name: str | None, wait_s: float = 2.0):
        """lift 판정용 물체 자리 — FP++ live 추적(…/live_pose) 새 프레임 중앙값. 없으면 None(판정 보류)."""
        return self._fresh_object(name, wait_s, self._live)

    def run_trajectory(self, node: Node, traj, world) -> NodeResult:
        if traj.kind == "noop":
            return NodeResult("completed", "noop")
        if traj.kind in PLANNED_TRAJ:
            return self._run_planned(node, traj, world)
        t0 = time.monotonic()
        for side in traj.sides:
            if not self._wait_pd_tracking(side):
                pd = self._pd_view(side)
                return NodeResult("error", f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}",
                                  signals={"pd_fault": pd["pd_phase"] or "unknown"})
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
            for k, argv in enumerate(steps):
                rc, tail = self._run(argv)
                if self.cancel.is_set():
                    return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
                if rc != 0:                                # 계획 · 시작점 검사 실패 = 안전한 경로 없음(움직이지 않았다)
                    status = "refused" if k < 2 else "error"
                    return NodeResult(status, f"{'no safe path — ' if k < 2 else ''}{Path(argv[1]).name} rc {rc}: {tail}",
                                      time.monotonic() - t0)
        return NodeResult("completed", "", time.monotonic() - t0)

    def _run_planned(self, node: Node, traj, world) -> NodeResult:
        """10.17 carry(ready_carry → 붓기 시작 자세, 한 팔씩 또는 두 팔) · tilt(D4 기본 궤적 + 복귀, 두 팔): 실측에서 계획 · 검사
        (움직이지 않음) → 그 팔 관절을 한 시각표로 재생. 계획 · 검사 실패(rc ≠ 0)면 움직이지 않고 refused. 손 목표는 그대로
        (팔 관절만 보낸다). cups = 그 팔 손이 컵을 쥐었을 때만(False = 빈손일 때만 — 경로 B 는 컵 없이 시작 자세로 간다)."""
        t0 = time.monotonic()
        sides = [s for s in SIDES if s in traj.sides]
        for side in sides:
            if not self._wait_pd_tracking(side):
                pd = self._pd_view(side)
                return NodeResult("error", f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}",
                                  signals={"pd_fault": pd["pd_phase"] or "unknown"})
            held = world.hand(side)
            if traj.cups and held == EMPTY:
                return NodeResult("refused", f"{traj.kind}: {side} 손이 비었다 — 컵을 쥔 상태에서만(손 안 컵 자리 기준)")
            if not traj.cups and held != EMPTY:
                return NodeResult("refused", f"{traj.kind}(빈손): {side} 손에 {held} — 빈손 이송은 손이 빈 때만")
        asset = TRAJ_ASSETS / traj.asset
        cfg = yaml.safe_load((asset / "policy.yaml").read_text(encoding="utf-8")) or {}
        out = self.run_dir / f"{traj.kind}_{node.id}"
        joints = ",".join(f"{s[0]}_aj_{i}" for s in sides for i in range(1, 8))
        side_arg = ["--sides", ",".join(sides)] if traj.kind == "carry" else []
        before = None
        if traj.kind == "reach":                      # ★10.19 1-B: 지금 FP++ 컵 자세(창 안 중앙값, 멈춰 있어야)로 계획
            snap = self._still_object(traj.object)
            if snap is None:
                return NodeResult("refused", f"reach: {traj.object} 컵 자세가 없거나 움직인다({SNAPSHOT_WINDOW_S} s 창, "
                                             f"{SNAPSHOT_STILL_M * 1000:.0f} mm) — FP++ 확인", time.monotonic() - t0)
            real = real_object_name(self.ep.objects[traj.object].get("topic", ""))
            if not real:
                return NodeResult("refused", f"reach: {traj.object} 토픽에서 실물 이름을 못 읽었다", time.monotonic() - t0)
            side_arg = ["--side", sides[0], "--object", real, "--cup", ",".join(f"{v:.5f}" for v in snap[0]),
                        "--quat", ",".join(f"{v:.6f}" for v in snap[1]), "--stop", traj.stop]   # ★10.20 pregrasp = 경로 D
        elif traj.kind == "lift":
            side_arg = ["--side", sides[0]]
            if traj.object:
                before = self._live_object(traj.object)
        steps = [
            [sys.executable, str(TOOLS / f"plan_{traj.kind}.py"), "--asset", str(asset), "--out", str(out), *side_arg],
            [sys.executable, str(TOOLS / "replay_to_pd.py"), "--npz", str(out / f"{traj.kind}_cmd.npz"), "--joints", joints,
             "--rate-scale", repr(float(traj.rate_scale)), "--max-start-gap", repr(float(cfg.get("start_gap_rad", 0.05))),
             "--execute"],
        ]
        for k, argv in enumerate(steps):
            rc, tail = self._run(argv)
            if self.cancel.is_set():
                return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
            if rc != 0:                                # 계획 · 검사 실패 = 움직이지 않았다
                status = "refused" if k == 0 else "error"
                return NodeResult(status, f"{Path(argv[1]).name} rc {rc}: {tail}", time.monotonic() - t0)
        if traj.kind == "lift" and traj.object:       # ★10.19 1-B: 컵이 손과 같이 올라갔는가(유지 끝 FP++ — 없으면 판정 보류)
            after = self._live_object(traj.object)
            if before is None or after is None:
                return NodeResult("completed", f"들기 끝 · FP++ 컵 {'전' if before is None else '뒤'} 프레임 없음 — 들림 판정 보류",
                                  time.monotonic() - t0, signals={"cup_rise_m": None})
            rise = float(after[2] - before[2])
            sig = {"cup_rise_m": round(rise, 4), "cup_before": [round(v, 4) for v in before], "cup_after": [round(v, 4) for v in after]}
            if rise < LIFT_RISE_MIN_M:
                return NodeResult("aborted", f"컵이 {rise * 100:.1f} cm 만 올라갔다(< {LIFT_RISE_MIN_M * 100:.0f} cm) — 못 쥐었거나 놓쳤다",
                                  time.monotonic() - t0, signals=sig)
            return NodeResult("completed", f"컵 {rise * 100:.1f} cm 들림 · 유지 끝", time.monotonic() - t0, signals=sig)
        return NodeResult("completed", "", time.monotonic() - t0)

    def _still_object(self, name: str, wait_s: float = 3.0):
        """reach 계획용 컵 자세 — 창(SNAPSHOT_WINDOW_S) 안 프레임이 멈춰 있으면 (pos 중앙값, 마지막 quat wxyz), 아니면 None."""
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not self.cancel.is_set():
            with self._lock:
                fr = [f for f in self._frames.get(name, []) if time.monotonic() - f[2] < SNAPSHOT_WINDOW_S]
            snap = snapshot_of(fr)
            if snap is not None:
                return snap
            time.sleep(0.1)
        return None

    def run_hand(self, node: Node, world) -> NodeResult:
        """★10.17 경로 B: open = 손을 계약 홈 손(편 손)으로(pd hand_release, 팔은 pd 가 그 자리에 붙든 채) — 빈손에서만.
        grasp = 한 손을 닿을 때까지 닫고 쥔 판정 · 3 s 유지(tools/hand_grasp.py, 팔 제자리). 못 쥐면 aborted(not grasped)."""
        t0 = time.monotonic()
        for side in node.sides:
            if not self._wait_pd_tracking(side):
                pd = self._pd_view(side)
                return NodeResult("error", f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}",
                                  signals={"pd_fault": pd["pd_phase"] or "unknown"})
            if world.hand(side) != EMPTY:
                return NodeResult("refused", f"hand {node.action}: {side} 손에 {world.hand(side)} — 빈손에서만(펴면 떨어진다)")
        if node.action == "open":
            for side in node.sides:
                rc, tail = self._run([sys.executable, str(TOOLS / "trigger.py"), "pd/hand_release", "--side", side, "--execute"])
                if self.cancel.is_set():
                    return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
                if rc != 0:
                    pd = self._pd_view(side)
                    return NodeResult("error", f"hand_release {side} 거부 · pd {pd['pd_phase']} {pd['pd_reasons']}: {tail}",
                                      time.monotonic() - t0)
            deadline = time.monotonic() + HAND_OPEN_WAIT_S        # 리뷰: 대기만 하지 않고 실측이 펴졌는지 본다
            errs: dict = {}
            while time.monotonic() < deadline and not self.cancel.is_set():
                errs = {side: self._hand_open_err(side) for side in node.sides}
                if all(e is not None and e <= HAND_OPEN_TOL_RAD for e in errs.values()):
                    return NodeResult("completed", f"손 펴짐 {errs}", time.monotonic() - t0)
                time.sleep(0.1)
            if self.cancel.is_set():
                return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
            return NodeResult("error", f"손이 {HAND_OPEN_WAIT_S:.0f} s 안에 안 펴졌다 — max|q| {errs}(허용 {HAND_OPEN_TOL_RAD} rad, "
                                       f"None = 손 실측 없음)", time.monotonic() - t0)
        side = node.sides[0]
        contract = hand_contract_file(node.hand_contract, _paths.SIM2REAL / "deploy" / "policies")
        if not contract.is_file():
            return NodeResult("refused", f"hand grasp: {contract} 가 없다(hand_contract {node.hand_contract})")
        rc, tail = self._run([sys.executable, str(TOOLS / "hand_grasp.py"), "--side", side, "--contract", str(contract),
                              "--cfg", json.dumps(dict(node.cfg)), "--out", str(self.run_dir / f"hand_{node.id}"), "--execute"])
        dur = time.monotonic() - t0
        if self.cancel.is_set():
            return NodeResult("aborted", "episode runner stop", dur)
        if rc == 0:
            return NodeResult("completed", tail, dur)
        if rc == 2:                                    # 못 쥠 — 들지 않았다(손은 그 자리, pd 가 붙든다)
            return NodeResult("aborted", f"hand_grasp rc 2: {tail}", dur)
        if rc == 3:
            return NodeResult("error", f"hand_grasp rc 3: {tail}", dur, signals={"pd_fault": self._pd_view(side)["pd_phase"] or "pd"})
        return NodeResult("error", f"hand_grasp rc {rc}: {tail}", dur)

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
                if not self._wait_pd_tracking(side):
                    pd = self._pd_view(side)
                    return NodeResult("error", f"pd_{side} 가 TRACKING 으로 안 돌아온다 · {pd['pd_phase']} {pd['pd_reasons']}",
                                      signals={"pd_fault": pd["pd_phase"] or "unknown"})
                rc, tail = self._run([sys.executable, str(TOOLS / "trigger.py"), "pd/hand_release", "--side", side, "--execute"])
                if rc != 0:
                    pd = self._pd_view(side)
                    return NodeResult("error", f"hand_release {side} 거부 · pd {pd['pd_phase']} {pd['pd_reasons']}: {tail}",
                                      signals={"pd_fault": pd["pd_phase"]} if pd["pd_phase"] == "HOLD" else {})
            time.sleep(1.0)
            return NodeResult("completed", "")
        if step == "refresh_objects":
            names = [p.target_object or p.source_object for p in plans if (p.target_object or p.source_object)]
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with self._lock:
                    fresh = all(n in self._relay or any(time.monotonic() - f[2] < 0.5 for f in self._frames.get(n, []))
                                for n in names)
                if fresh:
                    time.sleep(1.0)                        # 정책 노드가 받을 시간
                    return NodeResult("completed", "")
                time.sleep(0.1)
            return NodeResult("refused", f"cup pose missing — {names} 프레임이 5 s 동안 없다")
        if step == "refresh_holders":
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
        for side in self.sides:                       # ★10.19 이 에피소드의 팔만 — 다른 팔 실행기의 정책은 그대로
            for kind in NODES:
                self._call(f"{NS}/{service_ns(kind, side)}/episode/stop", timeout=2.0, wait=0.3)

    def joints(self) -> dict:
        return {}

    def clear_cancel(self) -> None:
        self.cancel.clear()

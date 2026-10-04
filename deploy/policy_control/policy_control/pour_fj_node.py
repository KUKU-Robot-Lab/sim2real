"""pour_fj_node: RH56F1 정책 노드 — 관측 → 정책 → 디코더를 한 프로세스에서. 계열(family)을 계약 schema 로 고른다.

  pour_fj  양팔 붓기(hdgp open-rh_b_pour_fj) — 노드 이름 pour_fj_node, 컵 둘
  rh_aglt  한 팔 접근 · 파지 · 들기 · 이송(hdgp open-rh_{r,l}_aglt, 09.30) — 노드 이름 rh_aglt_node(tools 없이
           policy_control/rh_aglt_node.py 로 띄운다), 컵 하나 + 리셋 때 정한 목표
손 행동 법칙은 두 계열 모두 policy_control/rh56f1_hand.py 한 곳이다.

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
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import _paths  # noqa: F401,E402
from policy_control import codec  # noqa: E402
from policy_control.cup_attach import AttachCfg, CupAttach, grasp_signal  # noqa: E402
from policy_control import pour_fj as F  # noqa: E402
from policy_control import rh_aglt as A  # noqa: E402
from policy_control import rh_aglt_goals as G  # noqa: E402
from policy_control.episode_master import EpisodeBook  # noqa: E402
from policy_control.joint_obs import quat_to_matrix  # noqa: E402
from policy_control.sources import SourceSet, load_robot_cfg, select_side  # noqa: E402
from policy_control.lean_node import LeanNodeMixin, lean_node_kwargs  # noqa: E402

NS = "/policy_control"
NODE = "pour_fj_node"
EVENTS = ("reset", "start", "stop", "abort")
CUP_STALE_S = 0.5
START_TILT_MAX_DEG = 15.0


class PourFjNodeError(RuntimeError):
    pass


# ---------------------------------------------------------------- ROS 없는 절반
def _side_raw(s, st, arm_names, fk, cup, role: str) -> dict:
    """sources.RobotState(한 팔) + 컵 (pos, quat) → 측정 칸들. s 는 계열의 한 팔(arm_joints · hand_joints …)."""
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
    arm_q = np.array([arm[j] for j in s.arm_joints], float)
    hand_q = np.array([hand[j] for j in s.hand_joints], float)
    pose = fk.palm_pose(arm_q, hand_q)
    tact = np.zeros(5) if st.tip_force is None or "tip_force" in bad else np.asarray(st.tip_force, float).reshape(5, -1)[:, 0]
    jf = None if getattr(st, "joint_force", None) is None or "joint_force" in bad else np.asarray(st.joint_force, float).reshape(-1)
    if callable(cup):                               # ★10.04 쥔 뒤에는 손바닥 FK 로(cup_attach) — 손바닥 · 촉각 · 시각을 본 뒤 고른다
        cup = cup(np.asarray(pose.palm_pos, float), quat_to_matrix(pose.palm_quat), tact, st)
    if cup is None:
        raise PourFjNodeError(f"{role} 컵 자세가 없다(또는 {CUP_STALE_S} s 넘게 끊겼다)")
    return dict(arm_q=arm_q, arm_qd=np.array([arm_d[j] for j in s.arm_joints], float),
                hand_q={j: float(hand[j]) for j in s.hand_joints}, palm_pos=np.asarray(pose.palm_pos, float),
                palm_R=quat_to_matrix(pose.palm_quat), tips=np.asarray(pose.tips, float).reshape(5, 3),
                cup_pos=np.asarray(cup[0], float), cup_quat=np.asarray(cup[1], float), tactile_n=tact, joint_force=jf)


def side_meas(c: F.FjContract, role: str, st, arm_names, fk, cup) -> F.FjSideMeas:
    """sources.RobotState(한 팔) + 컵 (pos, quat) → FjSideMeas. 결손 · stale 이면 PourFjNodeError."""
    return F.FjSideMeas(**_side_raw(c.sides[role], st, arm_names, fk, cup, role))


def aglt_meas(c: A.RaContract, role: str, st, arm_names, fk, cup) -> A.RaMeas:
    return A.RaMeas(**_side_raw(c.side(role), st, arm_names, fk, cup, role))


def _cup_tilt_deg(quat) -> float:
    return float(np.degrees(np.arccos(np.clip(quat_to_matrix(quat)[2, 2], -1.0, 1.0))))


def start_refusals(c: F.FjContract, meas: dict, tol: float) -> list:
    """팔이 학습 시작 자세 tol 안 · 컵 두 개가 서 있는가(빈 목록 = 시작)."""
    out = []
    for r in F.ROLES:
        err = float(np.abs(meas[r].arm_q - np.asarray(c.sides[r].arm_home)).max())
        if err > tol:
            out.append(f"{c.sides[r].side} arm is {err:.3f} rad from the training start pose (tol {tol})")
        tilt = _cup_tilt_deg(meas[r].cup_quat)
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


def aglt_start_refusals(c: A.RaContract, meas: dict, tol: float) -> list:
    m, s = meas["arm"], c.side()
    out = []
    err = float(np.abs(m.arm_q - np.asarray(s.arm_home)).max())
    if err > tol:
        out.append(f"{s.side} arm is {err:.3f} rad from the training start pose (tol {tol})")
    tilt = _cup_tilt_deg(m.cup_quat)
    if tilt > START_TILT_MAX_DEG:
        out.append(f"cup tilt {tilt:.1f} deg (> {START_TILT_MAX_DEG:.0f}) — a standing cup is expected")
    return out


def aglt_target_arrays(c: A.RaContract, targets: dict) -> tuple[tuple, np.ndarray, np.ndarray]:
    s = c.side()
    arr = np.concatenate([targets["arm"][0], targets["arm"][1]]).astype(float)
    if not np.all(np.isfinite(arr)):
        raise PourFjNodeError("non-finite joint target")
    return tuple(s.arm_joints) + tuple(s.hand_joints), arr, np.zeros(arr.size)


class AgltChain:
    """rh_aglt 한 스텝 — 관측(목표 포함) → 정책 → 디코더. 첫 목표는 reset 때 컵 자세로, 그 뒤는 사용자 입력(rh_aglt_goals)."""

    def __init__(self, c: A.RaContract, policy) -> None:
        self.c, self.policy = c, policy
        self.dec = A.RaDecoder(c)
        self.prev = np.zeros(c.action_dim)
        self.step_i = 0
        self.goals: G.GoalBook | None = None
        self.grasp_cfg: AttachCfg | None = None     # 노드가 넣는다(부착과 같은 쥠 신호) — 없으면 손끝 촉각 규칙

    @property
    def goal(self) -> A.Goal | None:
        return None if self.goals is None else self.goals.goal

    def request_goal(self, target) -> list:
        if self.goals is None:
            return ["goal: reset first (the first goal comes from the cup pose at reset)"]
        return self.goals.request(target)

    def reset(self, meas: dict | None = None) -> None:
        if meas is None:
            raise PourFjNodeError("rh_aglt reset needs the cup pose (goal = cup at reset + offset)")
        self.dec.reset()
        self.prev = np.zeros(self.c.action_dim)
        self.step_i = 0
        self.goals = G.GoalBook.start(self.c, meas["arm"].cup_pos, meas["arm"].cup_quat)
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def step(self, meas: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        m = meas["arm"]
        obs = A.build_obs(self.c, m, self.dec, self.goal, self.prev)
        a = np.clip(np.asarray(self.policy.forward(obs), float), -self.c.action_clip, self.c.action_clip)
        active = self.step_i >= self.c.hold_steps
        targets = self.dec.step(a, active=active, tactile_n=m.tactile_n)
        if active:                                  # 학습 GoalState.step 도 대기 중에는 세지 않는다
            g = None
            if self.grasp_cfg is not None:                 # 배포: 부착과 같은 쥠 신호(손끝 또는 관절 힘), 손끝 문턱은 계약 값
                g = grasp_signal(m.tactile_n, m.joint_force, replace(self.grasp_cfg, force_n=self.c.grasp_threshold_n))
            self.goals.step(m.cup_pos, m.cup_quat, m.tactile_n, is_grasped=g)
        self.prev = np.clip(a, -1.0, 1.0)
        self.step_i += 1
        return obs, a, targets


class PourFjChain:
    """관측 → 정책 → 디코더 한 스텝. 순수(정책 · FK 는 주입)."""

    def __init__(self, c: F.FjContract, policy) -> None:
        self.c, self.policy = c, policy
        self.dec = F.FjDecoder(c)
        self.prev = np.zeros(c.action_dim)
        self.step_i = 0

    def reset(self, meas: dict | None = None, hand_start: dict | None = None) -> None:
        """hand_close_margin 계약(b16~): 인계 순간의 손 **실측** 관절각을 q*_0 로 하고 디코더 q* 도 거기서 시작한다.

        sim 과 같다 — 인계 뱅크(build_pour_handoff_bank --targets_from_state)의 손 목표가 수집 순간 실제 관절 위치이고
        env 가 그 값을 닫기 상한 기준으로 쓴다(T2R Pouring 10.04). 실기는 손가락이 컵에 막혀 목표보다 덜 닫히므로
        파지 정책의 마지막 목표를 쓰면 이미 margin 넘게 조인 상태로 시작할 수 있다. hand_start 를 주면 그것이 이긴다(시험용).
        """
        handoff = bool(getattr(self.c, "bank_start", False)) or self.c.hand_close_margin_rad > 0.0
        arm_start = None
        if handoff:
            if meas is None or set(meas) != set(F.ROLES):
                raise PourFjNodeError("이 붓기 계약(인계 뱅크 시작 · hand_close_margin)은 인계 순간 두 팔 · 두 손의 실측 관절각이 필요하다")
            if hand_start is None:
                hand_start = {r: [float(meas[r].hand_q[j]) for j in self.c.sides[r].hand_joints] for r in F.ROLES}
            arm_start = {r: [float(v) for v in meas[r].arm_q] for r in F.ROLES}   # sim 뱅크 --targets_from_state
        self.dec.reset(hand_start, arm_start)
        self.prev = np.zeros(self.c.action_dim)
        # sim start_bank.restore: episode_length_buf = hold_steps → 인계 시작은 hold 를 건너뛰고 첫 스텝부터 정책(LSTM 은 0)
        self.step_i = self.c.hold_steps if handoff else 0
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


# ---------------------------------------------------------------- 계열
class Family:
    def __init__(self, *, name, schema, load, roles, cups, meas, chain, refusals, targets, errors, label):
        self.name, self.schema, self.load, self.roles, self.cups = name, schema, load, roles, cups
        self.meas, self.chain, self.refusals, self.targets, self.errors, self.label = (meas, chain, refusals, targets,
                                                                                      errors, label)


FAMILIES = {
    "pour_fj": Family(name="pour_fj_node", schema=F.SCHEMA, load=F.load_contract, roles=F.ROLES,
                      cups=(("src", "cup_src_topic", "/objects/cup_src/pose"), ("rcv", "cup_rcv_topic", "/objects/cup_rcv/pose")),
                      meas=side_meas, chain=PourFjChain, refusals=start_refusals, targets=target_arrays,
                      errors=(PourFjNodeError, F.PourFjError, ValueError),
                      label=lambda c: f"arm {c.arm_mode} · hand obs order {c.hand_obs_order_source.split(':')[0]}"),
    "rh_aglt": Family(name="rh_aglt_node", schema=A.SCHEMA, load=A.load_contract, roles=A.ROLES,
                      cups=(("arm", "cup_topic", "/objects/aglt_cup_s065/pose"),),
                      meas=aglt_meas, chain=AgltChain, refusals=aglt_start_refusals, targets=aglt_target_arrays,
                      errors=(PourFjNodeError, A.RhAgltError, ValueError),
                      label=lambda c: f"{c.side().side} arm · goal +{c.goal_offset}"
                                        f"{' · goal input on' if G.has_goal_spec(c) else ' · goal input off (old contract)'}"),
}


def family_of(contract_path: Path) -> str:
    schema = json.loads(Path(contract_path).read_text()).get("schema")
    for k, fam in FAMILIES.items():
        if fam.schema == schema:
            return k
    raise PourFjNodeError(f"{contract_path}: 모르는 계약 schema {schema!r} (pour_fj · rh_aglt)")


# ---------------------------------------------------------------- ROS
try:
    from rclpy.node import Node
except ImportError:
    Node = object                        # type: ignore[misc,assignment]


class PourFjNode(LeanNodeMixin, Node):
    def __init__(self, *, policy=None, family: str = "pour_fj", **kw) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data as _sensor
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String
        from std_srvs.srv import Trigger
        from policy_control.contract_assets import ASSETS
        from policy_control.fk_numpy import UrdfChainFK

        self.fam = FAMILIES[family]
        super().__init__(self.fam.name, **{**lean_node_kwargs(), **kw})   # ★10.03 CPU(lean_node)
        self.node_name = self.get_name()          # -r __node:=rh_aglt_node_right 로 팔마다 이름을 가른다
        for name, default in (("contract", ""), ("robot", ""), ("device", "cpu"), ("reset_tol_rad", 0.15),
                              ("max_gap_ticks", 3), ("publish_target", True), ("max_episode_s", -1.0), ("ns", ""),
                              ("cup_attach", True), ("attach_force_n", AttachCfg.force_n),
                              ("attach_joint_force_g", AttachCfg.joint_force_g), ("attach_signal", AttachCfg.signal),
                              ("attach_max_palm_dist_m", AttachCfg.max_palm_dist_m),
                              ("attach_after_s", AttachCfg.attach_after_s), ("release_steps", AttachCfg.release_steps),
                              *((param, topic) for _, param, topic in self.fam.cups)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        ns = str(p("ns")).strip("/")
        #: 한 세션에서 정책을 팔마다 동시에(09.30 사용자): ns 를 주면 에피소드 서비스 · 토픽 · 관측 · 행동을 /policy_control/<ns>/ 아래로.
        #  pd 는 /policy_control/<side>/episode 를 그 팔에만 적용한다 — ns 는 팔 이름(right · left)으로 준다. joint_target 은 공용.
        self.base = f"{NS}/{ns}" if ns else NS
        cpath, rpath = Path(str(p("contract"))), Path(str(p("robot")))
        if not cpath.is_file() or not rpath.is_file():
            raise PourFjNodeError(f"parameters 'contract' and 'robot' must be existing files (got {cpath}, {rpath})")
        if family_of(cpath) != family:
            raise PourFjNodeError(f"{cpath}: {family_of(cpath)} 계약 — 이 노드는 {family}")
        self.contract = self.fam.load(cpath)
        cfg = load_robot_cfg(rpath)
        self.srcs = {r: SourceSet(select_side(cfg, self.contract.sides[r].side)) for r in self.fam.roles}
        self.arm_names = {r: tuple(select_side(cfg, self.contract.sides[r].side).sources["arm"].joints)
                          for r in self.fam.roles}
        urdf = ASSETS[self.contract.asset].urdf
        self.fk = {r: UrdfChainFK(urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
                   for r, s in self.contract.sides.items()}
        self.device, self._policy = str(p("device")), policy
        self.reset_tol, self.max_gap, self._publish = float(p("reset_tol_rad")), int(p("max_gap_ticks")), bool(p("publish_target"))
        m = float(p("max_episode_s"))
        self.max_s = self.contract.episode_s if m < 0 else m
        if m < 0 and getattr(self.contract, "bank_start", False):    # sim 은 hold_steps 만큼 이미 흐른 채로 시작한다
            self.max_s = self.contract.episode_s - self.contract.hold_steps / float(self.contract.policy_hz)
        self.chain = None
        self.book = EpisodeBook({})
        self.cups: dict = {}
        # ★10.04 사용자: FP++ 는 정지한 컵만 — 쥔 뒤 컵 자세는 손바닥 기준 상대 자세 + 손바닥 FK(cup_attach).
        #   sim(rh_aglt hdgp 2721a946)과 같은 규칙: 엄지 · 다른 손가락 손끝 > 1 N 이 이어지는 동안 파지 시작 뒤 100 ms 넘어 찍힌
        #   FP++ 프레임이 오면 그 프레임 시각의 손바닥 FK 로 붙이고, 15 스텝(250 ms) 끊기면 뗀다.
        self.attach_cfg = AttachCfg(force_n=float(p("attach_force_n")), joint_force_g=float(p("attach_joint_force_g")),
                                    signal=str(p("attach_signal")), attach_after_s=float(p("attach_after_s")),
                                    max_palm_dist_m=float(p("attach_max_palm_dist_m")),
                                    release_steps=int(p("release_steps")))
        self.attach = {r: CupAttach(self.attach_cfg) for r in self.fam.roles} if bool(p("cup_attach")) else {}
        self._seq, self._gap, self._errors, self._t_start = 0, 0, {}, 0.0

        chain_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._String = String
        self._Pose = PoseStamped
        self._pub_target = self.create_publisher(JointState, f"{NS}/joint_target", chain_qos)
        self._pub_obs = self.create_publisher(Float64MultiArray, f"{self.base}/obs", chain_qos)
        self._pub_action = self.create_publisher(Float64MultiArray, f"{self.base}/action", chain_qos)
        self._pub_episode = self.create_publisher(String, f"{self.base}/episode", latched)
        self._pub_status = self.create_publisher(String, f"{NS}/status/{self.node_name}", QoSProfile(depth=10))
        # ★10.04 CPU: 상태 · 컵 토픽은 실행기 밖 폴링 노드에서 받는다 — 정책 틱(60 Hz) · 서비스에서 쌓인 것을 비우고 마지막 것만 푼다.
        #   팔 250 + 손 250 + 촉각 250 + 관절 힘 250 Hz 를 메시지마다 깨어나 받던 비용(pd 와 같은 rclpy wait set 비용)을 없앤다.
        #   받은 시각은 rmw received_timestamp(낡음 판정 그대로). raw_poll 참고.
        from policy_control.raw_poll import make_poll_node, poll_subscription
        msgs = {"joint_state": JointState, "float_array": Float64MultiArray}
        self._poll = make_poll_node(f"{self.node_name}_poll", context=self.context)
        self._poll_srcs = []
        for r in self.fam.roles:
            for src in select_side(cfg, self.contract.sides[r].side).sources.values():
                if src.type in msgs and (src.role or src.name) in ("arm", "ee", "tip_force", "joint_force"):
                    self._poll_srcs.append((r, src, msgs[src.type], poll_subscription(self._poll, msgs[src.type], src.topic, _sensor)))
        self._poll_cups = {role: poll_subscription(self._poll, PoseStamped, str(p(param)), _sensor)
                           for role, param, _ in self.fam.cups}
        for name in EVENTS:
            self.create_service(Trigger, f"{self.base}/episode/{name}", getattr(self, f"_srv_{name}"))
        if family == "rh_aglt":                     # 10.01 사용자: 목표 직접 입력(base 좌표 m) — 결과는 goal_result(latched JSON)
            from geometry_msgs.msg import Point
            self._pub_goal_result = self.create_publisher(String, f"{self.base}/goal_result", latched)
            self.create_subscription(Point, f"{self.base}/goal", self._on_goal, QoSProfile(depth=10))
        self.create_timer(1.0 / float(self.contract.policy_hz), self._on_tick)
        self.get_logger().info(f"{self.node_name} up · obs {self.contract.obs_dim} act {self.contract.action_dim} · "
                               f"{self.contract.policy_hz:.0f} Hz · {self.fam.label(self.contract)}")

    # ---------------------------------------------------------------- inputs
    def _apply_source(self, role, src, msg, now: float) -> None:
        try:
            if src.type == "joint_state":
                self.srcs[role].update_from_joint_state(src.name, codec.decode_joint_state(msg), now)
            else:
                self.srcs[role].update_from_float_array(src.name, codec.decode_float_array(msg), now)
            self._errors.pop(f"{role}.{src.name}", None)
        except (codec.CodecError, ValueError) as exc:
            self._errors[f"{role}.{src.name}"] = str(exc)

    def _apply_cup(self, role, msg, now: float) -> None:
        try:
            s = codec.decode_pose(msg)
            self.cups[role] = (now, s.pos, s.quat, s.stamp)   # stamp = FP++ 영상 시각(시스템 시계 초)
        except (codec.CodecError, ValueError) as exc:
            self._errors[f"cup_{role}"] = str(exc)

    def _drain(self) -> None:
        """폴링 구독마다 쌓인 것을 비우고 마지막 메시지만 풀어 소스 · 컵에 넣는다(받은 시각 그대로)."""
        from rclpy.serialization import deserialize_message
        from policy_control.raw_poll import take_last
        for role, src, typ, sub in self._poll_srcs:
            got = take_last(sub)
            if got is not None:
                self._apply_source(role, src, deserialize_message(got[0], typ), got[1])
        for role, sub in self._poll_cups.items():
            got = take_last(sub)
            if got is not None:
                self._apply_cup(role, deserialize_message(got[0], self._Pose), got[1])

    def destroy_node(self) -> None:
        poll, self._poll = getattr(self, "_poll", None), None
        if poll is not None:
            poll.destroy_node()
        super().destroy_node()

    def _cup(self, role):
        got = self.cups.get(role)
        return None if got is None or time.monotonic() - got[0] > CUP_STALE_S else (got[1], got[2], got[3])

    def _cup_for(self, role: str, advance: bool):
        """컵 자세를 고르는 함수 — 붙어 있으면 손바닥 FK. advance=True(달리는 정책 스텝)일 때만 판정 카운터가 움직인다."""
        live = self._cup(role)
        est = self.attach.get(role)
        if est is None:
            return live
        if advance:
            def step(palm_pos, palm_R, tact, st):
                t_palm = st.stamps.get("arm") or self.get_clock().now().nanoseconds * 1e-9   # 팔 상태 stamp = 손바닥 FK 시각
                jf = None if st.joint_force is None or "joint_force" in st.stale else st.joint_force
                before = est.source
                out = est.step(grasp_signal(tact, jf, self.attach_cfg), float(t_palm), palm_pos, palm_R, live)
                if est.source != before:                  # 붙고 뗀 순간을 로그로 — 상태(status.cup)에도 같은 값이 계속 실린다
                    self.get_logger().info(f"cup {role}: {before} → {est.source} · {est.as_dict()}")
                return out
            return step
        return lambda palm_pos, palm_R, _tact, _st: est.peek(palm_pos, palm_R, live)

    def _measure(self, advance: bool = False) -> dict:
        now = time.monotonic()
        return {r: self.fam.meas(self.contract, r, self.srcs[r].snapshot(now), self.arm_names[r], self.fk[r],
                                 self._cup_for(r, advance))
                for r in self.fam.roles}

    def _on_goal(self, msg) -> None:
        target = [float(msg.x), float(msg.y), float(msg.z)]
        reasons = (["goal: reset first (the first goal comes from the cup pose at reset)"] if self.chain is None
                   else self.chain.request_goal(target))
        body = {"ok": not reasons, "reasons": reasons, "request": [round(v, 4) for v in target],
                **(self.chain.goals.as_dict() if getattr(self.chain, "goals", None) is not None else {})}
        self._pub_goal_result.publish(self._String(data=json.dumps(body)))
        if reasons:                                 # rclpy: 한 호출 자리에서 로그 등급을 바꾸면 ValueError
            self.get_logger().warning(f"goal {body['request']} refused {reasons}")
        else:
            self.get_logger().info(f"goal {body['request']} accepted · now {body.get('goal')} · queue {body.get('queue')}")

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
        self._drain()
        for est in self.attach.values():           # 새 에피소드 — 컵은 다시 FP++ 부터
            est.reset()
        try:
            meas = self._measure()
            if self.chain is None:
                if self._policy is None:
                    from policy_control.joint_policy import JointPolicy
                    self._policy = JointPolicy(self.contract, self.device)
                self.chain = self.fam.chain(self.contract, self._policy)
                if hasattr(self.chain, "grasp_cfg") and self.attach:
                    self.chain.grasp_cfg = self.attach_cfg
            self.chain.reset(meas)                      # margin 계약은 meas 의 손 실측각이 q*_0(없으면 측정에서 이미 거부)
            q0 = getattr(self.chain.dec, "hand_q0", None) if hasattr(self.chain, "dec") else None
            if q0:
                self.get_logger().info(f"인계 손 q*_0(실측) { {r: [round(float(v), 3) for v in q] for r, q in q0.items()} }")
        except self.fam.errors as exc:
            return self._reply(res, False, [f"reset: {exc}"])
        self._seq, self._gap = 0, 0
        event, _ = self.book.reset()
        self._emit(event)
        return self._reply(res, True, [])

    def _srv_start(self, _req, res):
        self._drain()
        if self.chain is None:
            return self._reply(res, False, ["start: reset first"])
        try:
            refusals = self.fam.refusals(self.contract, self._measure(), self.reset_tol)
        except self.fam.errors as exc:
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
        body = {"node": self.node_name, "phase": self.book.phase, "episode": self.book.episode, "seq": self._seq, "ok": bool(ok),
                "reasons": [str(r) for r in reasons], "publish_target": self._publish,
                "hand_obs_order": self.contract.hand_obs_order_source.split(":")[0],
                **(self.chain.goals.as_dict() if getattr(self.chain, "goals", None) is not None else {}),
                "cup": {r: e.as_dict() for r, e in self.attach.items()},
                "t_pub_ns": self.get_clock().now().nanoseconds, **(extra or {})}
        self._pub_status.publish(self._String(data=json.dumps(body)))

    def _on_tick(self) -> None:
        self._drain()
        if self.book.phase != "running" or self.chain is None:
            try:
                self._measure()
                self._status(True, list(self._errors.values()))
            except self.fam.errors as exc:
                self._status(False, [*self._errors.values(), str(exc)])
            return
        t0 = time.perf_counter()
        try:
            obs, action, targets = self.chain.step(self._measure(advance=True))
            names, q, qd = self.fam.targets(self.contract, targets)
        except self.fam.errors as exc:
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


def main(argv=None, family: str = "pour_fj") -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=argv)
    node = None
    try:
        node = PourFjNode(family=family)
        from policy_control.cpu_plan import keep_off_rt
        node.get_logger().info(keep_off_rt())   # ★10.03 실시간 코어를 비켜 간다
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:                      # SIGTERM 으로 컨텍스트가 먼저 닫힌 뒤 타이머가 한 번 더 발행하면 RCLError — 정상 종료
        if rclpy.ok() or "context is invalid" not in str(exc):
            raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

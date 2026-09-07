"""pd_arm — pd 노드의 **한 팔 단위**(ArmUnit): 계약 `sides[side]` + robot yaml `groups` + pd yaml 을
한 팔 몫의 백엔드·법칙·FSM·중력·게인 게이트·hold/blend/thermal 로 묶는다 (플랜 §4.4).

노드(pd_node)는 팔 단위를 `sides` 순서(우 → 좌, 양팔 리셋 규약)로 들고 tick/서비스를 돌린다. 한 팔의
HOLD 는 그 팔만 세운다(FSM 이 팔마다 있다); estop/release 는 노드가 모든 팔에 건다.

락은 노드가 든다 — 여기 메서드는 두 부류다: **락 아래에서 부르는 것**(tick·take_target·on_episode·
engage_stage·start_home·start_release·zero_release·engage_refusals) 과 **블로킹이라 락 밖에서 부르는 것**
(apply_hand_gains·switch_engage·switch_release·read_reference·list_controllers).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .chain import PdStage, PdTarget, StageStatus
from .codec import CodecError, JointSample, select_joints
from .contract import SIDES, DeployContract, GainMismatch, SideCfg, side_of_joint
from .controller_switch import ControllerSwitch, read_jtc_reference, source_arm_joints
from .pd_backends import (ArmForwardBackend, Dg5fJtcBackend, GripperCmd, GripperJtcBackend, HandCmd,
                          HandGainsClient, hand_controller_name)
from .pd_gains import GainsError, expected_hand_gains, load_and_check
from .pd_gravity import make_gravity
from .pd_law import (PdCommand, PdConfig, blend_engage, blend_fraction, blend_release, law_cfg_from_config,
                     limits_from_profile)
from .pd_state import (EngageCheck, Phase, engage_refusals, hold_is_self_clearing, thermal_act_joints,
                       thermal_init, thermal_levels, thermal_stale_joints, thermal_step,
                       thermal_unknown_joints)
from .sources import RobotCfg, SourceSet, select_side

from jtc_bridge_core import JointRemap  # noqa: E402  (scripts/)

HOLD_SEQ = -2                      # 내부(홈/정지) 목표의 seq — 외부 목표(≥0)·없음(−1)과 구분
JTC_REF_MAX_AGE_SEC = 0.5
SIDE_ORDER = ("right", "left")     # 양팔 리셋 규약: 우팔 먼저
_ENGAGED = (Phase.RAMPING, Phase.TRACKING, Phase.HOLD)
_MOVING = (Phase.RAMPING, Phase.TRACKING)


class PdArmError(RuntimeError):
    """한 팔의 배선/설정 오류(기동 거부)."""


@dataclass(frozen=True)
class Blend:
    kind: str                      # engage | release
    t0: float


@dataclass(frozen=True)
class Hold:
    """내부 목표: goto_home(settle) 또는 stop/abort 뒤 현재 세트포인트 유지."""

    q: np.ndarray                  # 팔 canonical
    hand: np.ndarray | None        # 손 지령 관절 순
    bias: np.ndarray
    settle: bool
    settled: bool = False
    err: float = float("nan")


@dataclass(frozen=True)
class SideBackends:
    arm: ArmForwardBackend
    switch: ControllerSwitch
    gripper: GripperJtcBackend | None
    hand: Dg5fJtcBackend | None
    hand_gains: HandGainsClient | None
    hand_joint: str | None         # gripper 백엔드의 canonical 관절


@dataclass(frozen=True)
class TickResult:
    cmd: PdCommand | None          # None = 송출 없음(IDLE 또는 실측 없음)
    hand_written: np.ndarray | None
    status: StageStatus
    error: str | None = None       # 실측 결손/스테일 등 이번 tick 의 사유


# ------------------------------------------------------------------ side selection / wiring
def select_sides(requested: str, contract: DeployContract, robot_cfg: RobotCfg) -> list[str]:
    """ROS 파라미터 `sides`(쉼표 목록, '' = 자동) → 우→좌 순 목록. 자동 = 계약과 yaml 양쪽에 있는 팔 전부."""
    yaml_sides, contract_sides = set(robot_cfg.sides), set(contract.sides)
    wanted = [s.strip() for s in str(requested).split(",") if s.strip()]
    if not wanted:
        wanted = [s for s in SIDE_ORDER if s in yaml_sides and s in contract_sides]
        if not wanted:
            raise PdArmError(f"no side is in both the robot yaml {sorted(yaml_sides)} and the contract "
                             f"{sorted(contract_sides)}")
    if len(set(wanted)) != len(wanted):
        raise PdArmError(f"sides {wanted} has duplicates")
    for s in wanted:
        if s not in SIDES:
            raise PdArmError(f"side {s!r} not in {SIDES}")
        if s not in yaml_sides:
            raise PdArmError(f"side {s!r} is not wired in the robot yaml (has {sorted(yaml_sides)})")
        if s not in contract_sides:
            raise PdArmError(f"side {s!r} is not in the contract (has {sorted(contract_sides)})")
    return [s for s in SIDE_ORDER if s in wanted]


def hand_command_joints(side_cfg: SideCfg) -> list[str]:
    """pd 가 손에 보내는 canonical 관절: 정책 계약이면 action.hand.joints(디코더 출력), 제어 전용이면 side.hand_joints."""
    if side_cfg.hand is not None:
        return list(side_cfg.hand.joints)
    return list(side_cfg.hand_joints)


def side_groups(robot_cfg: RobotCfg, side_cfg: SideCfg) -> dict:
    """계약 `sides.<side>.pd_groups` ∩ robot yaml 그룹.

    yaml 이 선언된 그룹의 **부분집합**만 들고 있어도 된다 — 일부만 제어하겠다는 의도적 선택이다
    (2026-09-07: 손 PD 시험 동안 팔 그룹을 뺀 `*_hand_only.yaml`). 위험한 방향은 그 반대다:
    계약이 모르는 그룹을 yaml 이 들고 오면 pd 가 계약 밖 하드웨어를 움직이게 되므로 거부한다.
    """
    known = set(side_cfg.pd_groups)
    # 양팔 yaml 은 반대편 팔의 그룹도 담는다 — **이 팔 소속**(groups.<g>.side)만 본다.
    extra = [g for g, cfg in robot_cfg.groups.items()
             if str((cfg or {}).get("side") or "") == side_cfg.side and g not in known]
    if extra:
        raise PdArmError(f"robot yaml groups {extra} 는 계약 sides.{side_cfg.side}.pd_groups "
                         f"{sorted(known)} 에 없다")
    picked = {g: robot_cfg.groups[g] for g in side_cfg.pd_groups if g in robot_cfg.groups}
    if not picked:
        raise PdArmError(f"robot yaml 에 sides.{side_cfg.side}.pd_groups {sorted(known)} 중 아무것도 없다")
    return picked


def build_side_backends(node, groups: dict, cfg: PdConfig, side_cfg: SideCfg, hand_joints: list[str],
                        profile: dict, *, execute: bool) -> SideBackends:
    """robot yaml groups(한 팔 몫) → 백엔드. 그리퍼/손은 pd yaml 블록과 맞아야 한다.

    팔 그룹은 **선택**이다 — 없으면 손만 제어한다(2026-09-07: 손 PD 시험 동안 팔을 끈다).
    그때 arm·switch 는 None 이고 팔 지령·컨트롤러 교대가 생략된다."""
    arm = switch = gripper = hand = gains = None
    hand_joint = None
    for name, g in groups.items():
        kind = g.get("backend")
        if kind == "arm_forward":
            side = str(g["side"])
            if side != side_cfg.side:
                raise PdArmError(f"groups.{name}.side {side!r} ≠ contract side {side_cfg.side!r}")
            remap = JointRemap(list(side_cfg.arm_joints), source_arm_joints(side), profile)
            arm = ArmForwardBackend(node, side, remap, execute=execute)
            switch = ControllerSwitch(node, side, list(g["forward"]), str(g["jtc"]), execute=execute)
        elif kind == "jtc_single_point":
            gripper, hand_joint = _gripper_backend(node, name, g, cfg, hand_joints, profile, execute)
        elif kind == "dg5f_jtc":
            hand, gains = _hand_backend(node, name, g, cfg, hand_joints, profile, execute)
        else:
            raise PdArmError(f"groups.{name}: unknown backend {kind!r}")
    if (arm is None) != (switch is None):
        raise PdArmError(f"side {side_cfg.side}: arm_forward group is half-built ({sorted(groups)})")
    if arm is None and gripper is None and hand is None:
        raise PdArmError(f"side {side_cfg.side}: no controllable group among {sorted(groups)}")
    return SideBackends(arm=arm, switch=switch, gripper=gripper, hand=hand, hand_gains=gains, hand_joint=hand_joint)


def _gripper_backend(node, name, g, cfg, hand_joints, profile, execute):
    if cfg.gripper is None or len(hand_joints) != 1:
        raise PdArmError(f"groups.{name}: jtc_single_point needs the pd yaml gripper block and one hand joint")
    joint = hand_joints[0]
    if profile[joint]["source"] != str(g["joint"]):
        raise PdArmError(f"groups.{name}.joint {g['joint']!r} ≠ contract hand joint {joint} source "
                         f"{profile[joint]['source']!r}")
    if abs(float(g.get("close_overtravel_m", cfg.gripper.close_overtravel_m)) - cfg.gripper.close_overtravel_m) > 1e-9:
        raise PdArmError(f"groups.{name}.close_overtravel_m ≠ pd yaml gripper.close_overtravel_m")
    backend = GripperJtcBackend(node, str(g["topic"]), str(g["joint"]), cfg.gripper.close_overtravel_m,
                                cfg.gripper.max_vel, lower=float(profile[joint]["lower"]),
                                upper=float(profile[joint]["upper"]), execute=execute)
    return backend, joint


def _hand_backend(node, name, g, cfg, hand_joints, profile, execute):
    if cfg.hand is None:
        raise PdArmError(f"groups.{name}: dg5f_jtc needs the pd yaml hand block")
    if abs(float(g.get("pid_p", cfg.hand.pid_p)) - cfg.hand.pid_p) > 1e-9:
        raise PdArmError(f"groups.{name}.pid_p ≠ pd yaml hand.pid_p")
    sources = [profile[j]["source"] for j in hand_joints]
    backend = Dg5fJtcBackend(node, str(g["topic"]), JointRemap(hand_joints, sources, profile), cfg.hand.max_vel,
                             execute=execute)
    try:
        controller = hand_controller_name(str(g["topic"]), g.get("namespace"))
    except ValueError as exc:
        raise PdArmError(f"groups.{name}: {exc}") from exc
    gains = HandGainsClient(node, controller, sources, timeout_sec=3.0, execute=execute)
    return backend, gains


def _side_thermal(cfg: PdConfig, side: str) -> tuple:
    return tuple(r for r in cfg.thermal if side_of_joint(r.joint) == side)


# ================================================================== one arm
class ArmUnit:
    """한 팔: 실측(SourceSet, 한 팔 view) → PdStage.tick → 백엔드. 상태 변경은 노드 락 아래에서만."""

    def __init__(self, node, *, side_cfg: SideCfg, robot_cfg: RobotCfg, cfg: PdConfig, contract: DeployContract,
                 profile: dict, stage_name: str, execute: bool) -> None:
        self.side = side_cfg.side
        self.side_cfg, self.cfg, self.contract, self.profile = side_cfg, cfg, contract, profile
        self.execute = bool(execute)
        self.arm_joints = list(side_cfg.arm_joints)
        self.hand_joints = hand_command_joints(side_cfg)
        self.robot_cfg = select_side(robot_cfg, self.side)
        self._setup_law(stage_name)
        self.backends = build_side_backends(node, side_groups(robot_cfg, side_cfg), cfg, side_cfg, self.hand_joints,
                                            profile, execute=self.execute)
        self.sources = SourceSet(self.robot_cfg)
        self.thermal_rules = _side_thermal(cfg, self.side)
        self.thermal = thermal_init(self.thermal_rules)
        self.t_ee_recv: float | None = None
        self.target: PdTarget | None = None
        self.hand_target: np.ndarray | None = None
        self.hold: Hold | None = None
        self.blend: Blend | None = None
        self.switch_failed = False
        self.thermal_retreat = False   # 발열 HOLD 에서 저부하 자세로 내려가는 중
        self.efforts: dict[str, float] = {}
        self.temps: dict[str, float] = {}   # 관절 → 모터 온도 [°C] (없으면 키가 없다)
        self.t_temp_recv: float | None = None
        self.t_arm_recv: float | None = None

    # ---------------------------------------------------------------- setup
    def _setup_law(self, stage_name: str) -> None:
        lower, upper, vel = limits_from_profile(self.robot_cfg.joint_profile, self.arm_joints)
        if np.any(np.abs(vel - self.cfg.lead_vel) > 1e-9):
            raise PdArmError(f"pd yaml lead_vel {self.cfg.lead_vel} ≠ profile joint velocity {vel.tolist()}")
        if list(self.side_cfg.sim_gains.joints) != self.arm_joints:
            raise PdArmError(f"side {self.side}: sim_gains.joints ≠ arm_joints")
        self.kp = np.asarray(self.side_cfg.sim_gains.kp, dtype=float)
        self.gains_ok, self.gains_report = self._check_gains()
        self.gravity_fn = make_gravity(self.cfg.gravity, self.contract, side=self.side)
        ramp = law_cfg_from_config(self.cfg, self.contract, "ramp", lower, upper, side=self.side)
        track = law_cfg_from_config(self.cfg, self.contract, stage_name, lower, upper, side=self.side)
        self.dt = 1.0 / self.cfg.pd_hz
        self.stage = PdStage(ramp_cfg=ramp, track_cfg=track, watchdog_sec=self.cfg.watchdog_sec,
                             abort_tracking=self.cfg.abort_tracking, ramp_tol=self.cfg.settle.tol,
                             release_zero_ticks=self.cfg.release_zero_ticks, dt=self.dt, gravity_fn=self.gravity_fn)
        self.home_arm = np.asarray(self.side_cfg.home_arm, dtype=float)
        home_hand = self.side_cfg.home_hand
        self.home_hand = np.array([float(home_hand[j]) for j in self.hand_joints]) \
            if self.hand_joints and all(j in home_hand for j in self.hand_joints) else None

    def _check_gains(self) -> tuple[bool, dict]:
        try:
            rep = load_and_check(self.cfg.gains, self.contract, side=self.side)
        except GainMismatch as exc:
            return False, {"ok": False, "reasons": [str(exc)], "kd_note": "", "accepted_mismatch": False}
        except GainsError as exc:
            raise PdArmError(f"gains: {exc}") from exc
        return True, {"ok": rep.ok, "reasons": rep.reasons, "kd_note": rep.kd_note,
                      "impossible_kd": list(rep.impossible_kd), "accepted_mismatch": rep.accepted_mismatch}

    @property
    def phase(self) -> Phase:
        return self.stage.state.fsm.phase

    @property
    def engaged(self) -> bool:
        return self.phase in _ENGAGED

    def joint_topics(self) -> dict:
        """{topic: [role, …]} — 이 팔의 arm/ee 소스 토픽(노드가 구독을 합친다)."""
        out: dict = {}
        for r in ("arm", "ee"):
            out.setdefault(self.robot_cfg.sources[r].topic, []).append(r)
        return out

    def close(self) -> None:
        if self.backends.switch is not None:
            self.backends.switch.close()
        if self.backends.hand_gains is not None:
            self.backends.hand_gains.close()

    # ---------------------------------------------------------------- inputs (under lock)
    def on_joint_state(self, roles: list[str], sample: JointSample, now: float) -> None:
        for role in roles:
            self.sources.update_from_joint_state(role, sample, now)
        if "arm" in roles:
            self.t_arm_recv = now
            self.efforts = self._arm_efforts(sample)
        if "ee" in roles:
            self.t_ee_recv = now

    def on_temperatures(self, by_source: Mapping[str, float], now: float) -> None:
        """드라이버가 올린 모터 서미스터(°C)를 canonical 관절 이름으로 옮긴다.

        `/dynamic_joint_states` 의 `temperature_rotor`/`temperature_mos` 에서 온다. 관절당
        더 뜨거운 쪽을 쓴다 — 로터가 먼저 뜨는 경우도, 드라이버가 먼저 뜨는 경우도 있다.
        """
        self.temps = {c: float(by_source[self.profile[c]["source"]]) for c in self.arm_joints
                      if self.profile[c]["source"] in by_source}
        self.t_temp_recv = now

    def _fresh_temps(self, now: float) -> dict[str, float]:
        """온도가 끊기면 **빈 dict** 를 돌려준다 — 마지막 값을 계속 믿으면 보호가 조용히 꺼진다."""
        stale = self.robot_cfg.sources["arm"].stale_sec
        if self.t_temp_recv is None or (now - self.t_temp_recv) > stale:
            return {}
        return self.temps

    def _arm_efforts(self, sample: JointSample) -> dict[str, float]:
        if sample.effort is None:
            return {}
        idx = {n: i for i, n in enumerate(sample.names)}
        return {c: float(sample.effort[idx[self.profile[c]["source"]]]) for c in self.arm_joints
                if self.profile[c]["source"] in idx}

    def take_target(self, sample: JointSample, seq: int, now: float) -> bool:
        """joint_target 한 건에서 이 팔의 관절을 이름으로 뽑는다. 팔 관절이 다 없으면 False(다른 팔의 목표)."""
        try:
            q, qd = select_joints(sample, self.arm_joints)
        except CodecError:
            return False
        if qd is None:
            raise CodecError(f"joint_target({self.side}): velocity 가 없다")
        n = len(self.arm_joints)
        self.target = PdTarget(q=q.copy(), qd=qd.copy(), tau_ff=np.zeros(n), seq=int(seq), t_recv=now)
        if self.hand_joints and all(j in sample.names for j in self.hand_joints):
            self.hand_target = select_joints(sample, self.hand_joints)[0].copy()
        self.hold = None                                    # 외부 목표가 내부 유지를 대체한다
        return True

    def on_episode(self, event: str, episode: int) -> None:
        if event == "reset":
            self.stage.new_episode(episode)
            if self.hold is not None:                       # 정착 bias 제거·홈 유지(sim 리셋 직후와 동일)
                self.hold = replace(self.hold, bias=np.zeros_like(self.hold.bias), settle=False)
        elif event in ("stop", "abort") and self.stage.state.law is not None:
            self.hold = Hold(q=self.stage.state.law.q_setpoint.copy(), hand=self.hand_target,
                             bias=np.zeros(len(self.arm_joints)), settle=False)
            self.target = None

    # ---------------------------------------------------------------- tick (under lock)
    def tick(self, now: float, estop: bool) -> TickResult:
        state = self.sources.snapshot(now)
        q_m, qd_m, why = self._measured(state)
        self._advance_blend(now)
        if q_m is None:
            return TickResult(cmd=None, hand_written=None, status=self.idle_status(), error=why)
        target = self._select_target(now, q_m)
        out = self.stage.tick(target, q_m, qd_m, now, estop=estop,
                              thermal_act=thermal_act_joints(self.thermal, self.thermal_rules),
                              switch_failed=self.switch_failed,
                              thermal_stale=thermal_stale_joints(self.thermal, self.thermal_rules),
                              thermal_retreat=self.thermal_retreat)
        cmd = self._blend_cmd(out.cmd, q_m, now)
        hand_written = self._write(cmd, out.state.fsm.phase, state)
        self.thermal = thermal_step(self.thermal, self.thermal_rules,
                                    {r.joint: self.efforts.get(r.joint, 0.0) for r in self.thermal_rules}, self.dt,
                                    temps=self._fresh_temps(now))
        if self.thermal_retreat and self.home_settled():
            self.thermal_retreat = False     # 저부하 자세에 닿았다 — 가드를 다시 켠다
        return TickResult(cmd=cmd, hand_written=hand_written, status=out.status)

    def measured_q(self, now: float) -> np.ndarray | None:
        return self._measured(self.sources.snapshot(now))[0]

    def _measured(self, state):
        """pd 는 arm/ee 만 쓴다(object 등 다른 필수 소스의 결손은 obs 노드 몫).

        ★손 단독(팔 백엔드 없음)이면 **arm 을 요구하지 않는다**. 팔은 우리 지령을 안 받으므로
        그 추종오차는 우리가 판정할 대상이 아니다(2026-09-07: 손 드라이버만 띄운 시험에서
        `joint state missing ['arm']` 으로 손 지령까지 막혔다). q_meas 는 직전 목표를 그대로 써
        팔 법칙을 무해한 통과로 만든다 — 어디에도 발행되지 않는다.
        """
        roles = ("arm", "ee") if self.backends.arm is not None else ("ee",)
        missing = [r for r in roles if r in state.missing]
        stale = [r for r in roles if r in state.stale]
        if missing:
            return None, None, f"{self.side}: joint state missing {missing}"
        if stale:
            return None, None, f"{self.side}: joint state stale {stale}"
        if self.backends.arm is None:
            n = len(self.arm_joints)
            q = self.target.q if self.target is not None else np.zeros(n)
            return np.asarray(q, dtype=float).reshape(-1)[:n], np.zeros(n), None
        return state.arm_q, state.arm_qd, None

    def _select_target(self, now: float, q_m: np.ndarray) -> PdTarget | None:
        if self.hold is None:
            return self.target
        self.hold = self._settle(self.hold, q_m)
        return PdTarget(q=self.hold.q + self.hold.bias, qd=np.zeros(len(q_m)), tau_ff=np.zeros(len(q_m)),
                        seq=HOLD_SEQ, t_recv=now)

    def _settle(self, hold: Hold, q_m: np.ndarray) -> Hold:
        """goto_home 정착: bias += gain·(home − q) (clamp), |q − home| < tol 이면 settled (left_inference_node 규약)."""
        err = float(np.abs(q_m - hold.q).max())
        if not hold.settle or hold.settled or self.phase is not Phase.TRACKING:
            return replace(hold, err=err)
        if err < self.cfg.settle.tol:
            return replace(hold, settled=True, err=err)
        g = self.side_cfg.gravity
        if g.mode != "integral_droop" or g.gain is None:
            return replace(hold, err=err)
        bias = np.clip(hold.bias + float(g.gain) * (hold.q - q_m), -self.cfg.settle.clamp, self.cfg.settle.clamp)
        return replace(hold, bias=bias, err=err)

    # ---------------------------------------------------------------- blend
    def _advance_blend(self, now: float) -> None:
        b = self.blend
        if b is None or now - b.t0 < self.cfg.blend_sec:
            return
        self.blend = None
        if b.kind == "release" and self.engaged:
            self.stage.release()

    def _blend_cmd(self, cmd: PdCommand | None, q_m: np.ndarray, now: float) -> PdCommand | None:
        """ref = 법칙 q_cmd + G/kp 로 두면 blend_engage 는 s=0 에서 (ref, 0) = JTC 가 들고 있던 것, s=1 에서
        (q_cmd, G) = 법칙 정상 출력 — 사이에서 kp(q*−q)+τ 가 일정하다. release 는 s 를 거꾸로."""
        b = self.blend
        if b is None or cmd is None:
            return cmd
        s = blend_fraction(now - b.t0, self.cfg.blend_sec)
        g = np.asarray(self.gravity_fn(q_m), dtype=float)
        blend = blend_engage if b.kind == "engage" else blend_release
        q_star, tau = blend(cmd.q + g / self.kp, g, self.kp, s)     # G=0(integral_droop) 이면 항등
        return replace(cmd, q=q_star, tau=tau)

    # ---------------------------------------------------------------- output
    def _write(self, cmd: PdCommand | None, phase: Phase, state):
        if cmd is None:
            return None
        if self.backends.arm is not None:            # ★손 단독(팔 그룹 없음)에서는 팔 지령이 없다
            self.backends.arm.write(cmd)
        if phase not in _MOVING:
            return None
        hand = self.hold.hand if (self.hold is not None and self.hold.hand is not None) else self.hand_target
        if hand is None:
            return None
        return self._write_hand(hand, state)

    def _write_hand(self, hand: np.ndarray, state):
        b = self.backends
        if b.gripper is not None:
            idx = list(state.ee_names).index(b.hand_joint)
            w = b.gripper.write(GripperCmd(q_star=float(hand[0]), q_meas=float(state.ee_q[idx]), dt=self.dt))
            return np.array([w.q_cmd])
        if b.hand is not None:
            meas = np.asarray([state.ee_q[list(state.ee_names).index(j)] for j in self.hand_joints], dtype=float) \
                if all(j in state.ee_names for j in self.hand_joints) else None
            b.hand.write(HandCmd(q_star=hand, qd_star=None, dt=self.dt, q_meas=meas))
            return hand.copy()
        return None

    def idle_status(self) -> StageStatus:
        st = self.stage.state
        return StageStatus(node="pd", phase=st.fsm.phase.value, episode=st.episode,
                           seq=-1 if st.target is None else int(st.target.seq), ok=st.fsm.phase is not Phase.HOLD,
                           reasons=() if st.fsm.hold_reason is None else tuple(st.fsm.hold_reason.split("; ")),
                           proc_ms=0.0)

    def extras(self) -> dict:
        """status 의 팔별 부가 필드."""
        return {"side": self.side, "gains": self.gains_report, "thermal": thermal_levels(self.thermal, self.thermal_rules),
                "blend": None if self.blend is None else self.blend.kind,
                "hold": None if self.hold is None else {"settle": self.hold.settle, "settled": self.hold.settled,
                                                        "err": self.hold.err},
                "target": "internal" if self.hold is not None else ("external" if self.target is not None else None)}

    # ---------------------------------------------------------------- services
    def list_controllers(self) -> dict:
        """블로킹 — 락 밖."""
        if self.backends.switch is None:            # 손 단독: 팔 컨트롤러 교대가 없다
            return {}
        return self.backends.switch.list() if self.execute else {}

    def engage_refusals(self, states: dict, estop: bool, now: float) -> list[str]:
        # 손 단독이면 팔 상태가 안 온다 — 신선도는 손(ee) 수신으로 판정한다.
        recv = self.t_arm_recv if self.backends.arm is not None else self.t_ee_recv
        age = None if recv is None else now - recv
        check = EngageCheck(execute=self.execute, state_age_sec=age, stale_sec=self.robot_cfg.sources["arm"].stale_sec,
                            gains_ok=self.gains_ok, accept_sim_mismatch=self.cfg.gains.accept_sim_mismatch,
                            gravity_conflict=None,
                            effort_controller_active=(self.backends.switch is not None
                                                      and states.get(self.backends.switch.forward[2]) == "active"),
                            estop_latched=estop, phase=self.phase,
                            thermal_unknown=thermal_unknown_joints(self.thermal, self.thermal_rules))
        return engage_refusals(check)

    def apply_hand_gains(self) -> tuple[bool, list[str]]:
        """블로킹 — 락 밖. 손이 없으면 (True, [])."""
        if self.backends.hand_gains is None:
            return True, []
        hg = expected_hand_gains(self.cfg)
        return self.backends.hand_gains.check_and_apply(hg.pid_p, hg.pid_d)

    def switch_engage(self) -> tuple[bool, list[str]]:
        """블로킹 — 락 밖: forward 3종 load+configure → STRICT switch.

        손 단독(팔 그룹 없음)이면 교대할 팔 컨트롤러가 없다 — 손은 자기 controller_manager
        (Modbus TCP)에서 JTC 로 돌기 때문이다."""
        sw = self.backends.switch
        if sw is None:
            return True, ["hand-only: no arm controller switch"]
        ok, notes = sw.ensure_loaded_inactive(sw.forward)
        if not ok:
            return False, notes
        ok, why = sw.engage()
        return ok, notes + why

    def read_reference(self, node) -> np.ndarray | None:
        """블로킹 — 락 밖: JTC controller_state 의 세트포인트(source 순) 또는 None."""
        return read_jtc_reference(node, self.side, JTC_REF_MAX_AGE_SEC) if self.execute else None

    def seed_from(self, ref_src: np.ndarray | None, now: float) -> tuple[np.ndarray, str]:
        """JTC reference(source 순) → canonical, 없거나 낡으면 실측. 모델 중력이면 ref − G/kp (블렌드 종점)."""
        q_m = self.measured_q(now)
        if q_m is None:
            raise PdArmError(f"{self.side}: no measured joint state to seed from")
        if ref_src is None:
            ref, note = q_m, f"{self.side}: seed measured q (no fresh JTC reference)"
        else:
            src = source_arm_joints(self.side)
            idx = {s: i for i, s in enumerate(src)}
            ref = np.array([float(ref_src[idx[self.profile[c]["source"]]]) * float(self.profile[c]["sign"])
                            for c in self.arm_joints])
            note = f"{self.side}: seed JTC reference"
        return ref - np.asarray(self.gravity_fn(q_m), dtype=float) / self.kp, note

    def engage_stage(self, q_seed: np.ndarray, now: float) -> str:
        self.stage.engage(q_seed)
        self.blend = Blend("engage", now)
        self.hold, self.target, self.switch_failed = None, None, False
        return self.phase.value

    def start_home(self) -> None:
        self.hold = Hold(q=self.home_arm.copy(), hand=self.home_hand, bias=np.zeros(len(self.home_arm)), settle=True)
        self.target = None

    def start_thermal_retreat(self) -> bool:
        """자기해제형 HOLD(발열·워치독)에서 홈으로 내려가는 것을 허용한다.

        ★09.07 교착: 발열 HOLD 는 세트포인트를 얼려 팔을 **내리는 것까지** 막았다. 식으려면
        내려야 하는데 내릴 수가 없어, 팔을 든 채 release 하고 옛 JTC 경로로 내려야 했다.
        후퇴 중에는 발열 사유가 HOLD 를 다시 걸지 않는다(`FaultInputs.thermal_retreat`).
        """
        if not hold_is_self_clearing(self.stage.state.fsm):
            return False
        self.thermal_retreat = True
        self.start_home()
        return True

    def home_settled(self) -> bool:
        return self.hold is None or self.hold.settled

    def start_release(self, now: float) -> None:
        if self.blend is None or self.blend.kind != "release":
            self.blend = Blend("release", now)

    def zero_release(self) -> None:
        if self.backends.arm is not None:
            self.backends.arm.zero_release()
        for b in (self.backends.gripper, self.backends.hand):
            if b is not None:
                b.zero_release()

    def switch_release(self) -> tuple[bool, list[str]]:
        """블로킹 — 락 밖. 손 단독이면 교대할 팔 컨트롤러가 없다."""
        if self.backends.switch is None:
            return True, ["hand-only: no arm controller switch"]
        return self.backends.switch.release()

"""pd 노드 상태기계(순수, ROS 무의존) — 플랜 §4.4.

    IDLE ──engage──▶ RAMPING ──ramp_done──▶ TRACKING
                        │ fault               │ fault
                        ▼                     ▼
                      HOLD(reason) ◀──fault── HOLD
      RAMPING/TRACKING/HOLD ──release──▶ RELEASING ──zero_tick×N──▶ IDLE

HOLD 는 세트포인트 동결·q̇*=0·τ_ff 유지(급감 금지)를 뜻한다 — 그 해석은 `law_flags` 가
pd_law 의 (advance, tracking) 두 플래그로 넘긴다. engage 거부 사유와 fault 사유는 각각
한 리스트로 모아 status 에 그대로 실린다(어느 캡이 걸렸는지 사후에 알 수 있게).

모든 상태는 frozen dataclass 이고 전이는 새 객체를 돌려준다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence


class TransitionError(RuntimeError):
    """The requested event is not legal in the current phase."""


class Phase(Enum):
    IDLE = "IDLE"
    RAMPING = "RAMPING"
    TRACKING = "TRACKING"
    HOLD = "HOLD"
    RELEASING = "RELEASING"


EVENTS = ("engage", "ramp_done", "fault", "release", "zero_tick", "target_fresh")
_MOVING = (Phase.RAMPING, Phase.TRACKING)
_ENGAGED = (Phase.RAMPING, Phase.TRACKING, Phase.HOLD)


@dataclass(frozen=True)
class FsmState:
    phase: Phase
    hold_reason: str | None
    zero_ticks: int                 # RELEASING 에서 q̇*=0·τ=0 을 몇 번 송출했나
    release_zero_ticks: int         # IDLE 로 가기 위해 필요한 횟수(설정)


def initial_fsm(release_zero_ticks: int) -> FsmState:
    if int(release_zero_ticks) < 1:
        raise ValueError(f"release_zero_ticks must be >= 1, got {release_zero_ticks}")
    return FsmState(phase=Phase.IDLE, hold_reason=None, zero_ticks=0,
                    release_zero_ticks=int(release_zero_ticks))


def transition(state: FsmState, event: str, reason: str | None = None) -> FsmState:
    """Apply *event* and return the new state; illegal pairs raise TransitionError."""
    if event not in EVENTS:
        raise TransitionError(f"unknown event {event!r}; expected one of {EVENTS}")
    phase = state.phase
    if event == "engage" and phase is Phase.IDLE:
        return replace(state, phase=Phase.RAMPING, hold_reason=None, zero_ticks=0)
    if event == "ramp_done" and phase is Phase.RAMPING:
        return replace(state, phase=Phase.TRACKING)
    if event == "fault" and phase in _ENGAGED:
        return _hold(state, reason)
    if event == "release" and phase in _ENGAGED:
        return replace(state, phase=Phase.RELEASING, zero_ticks=0)
    if event == "zero_tick" and phase is Phase.RELEASING:
        return _zero_tick(state)
    if event == "target_fresh":
        return _target_fresh(state)
    raise TransitionError(f"event {event!r} is not legal in phase {phase.value}")


def _hold(state: FsmState, reason: str | None) -> FsmState:
    if not reason:
        raise TransitionError("a fault needs a reason")
    # 사유는 **종류**(콜론 앞)별로 하나만 남긴다 — 'watchdog: target stale 0.252 s' 처럼 숫자만
    # 다른 사유가 100 Hz 로 쌓여 문자열이 무한히 자라는 것을 막는다(첫 메시지를 보존).
    reasons = [] if state.hold_reason is None else state.hold_reason.split("; ")
    kinds = {r.split(":", 1)[0] for r in reasons}
    if reason.split(":", 1)[0] not in kinds:
        reasons = [*reasons, reason]
    return replace(state, phase=Phase.HOLD, hold_reason="; ".join(reasons))


WATCHDOG_KIND = "watchdog"

# 조건이 사라지면 스스로 풀리는 사유들(사유 문자열의 콜론 앞 접두사).
# 나머지(추종오차·관절한계·τ 캡·estop·switch)는 사람이 판단할 사유라 남긴다.
SELF_CLEARING_KINDS = (WATCHDOG_KIND, "thermal")


def _kinds(state: FsmState) -> list[str]:
    return [] if not state.hold_reason else [r.split(":", 1)[0].strip()
                                             for r in state.hold_reason.split("; ")]


def clear_reasons(state: FsmState, resolved: Sequence[str]) -> FsmState:
    """Drop every HOLD reason whose kind is in *resolved*; resume if none remain.

    발열이 식거나 목표가 돌아오는 것처럼 **측정으로 사라진** 조건을 되돌리는 단일 경로다.
    남는 사유가 하나라도 있으면 HOLD 를 유지한다(사람이 판단할 사유를 온도가 대신 풀 수 없다).
    RAMPING 으로 돌아가 세트포인트가 실측에서 다시 접근하게 한다(TRACKING 직행 금지).
    """
    if state.phase is not Phase.HOLD or not state.hold_reason:
        return state
    resolved = set(resolved)
    keep = [r for r in state.hold_reason.split("; ") if r.split(":", 1)[0].strip() not in resolved]
    if keep:
        return replace(state, hold_reason="; ".join(keep))
    return replace(state, phase=Phase.RAMPING, hold_reason=None, zero_ticks=0)


def hold_is_self_clearing(state: FsmState) -> bool:
    """HOLD 사유가 전부 자기해제형인가 — 저부하 자세로 **후퇴하는 것**이 허용되는 조건.

    ★09.07 교착: 발열 HOLD 가 팔을 내리는 것까지 막았다. 식으려면 내려야 하는데 내릴 수가
    없어, 팔을 든 채 release 하고 옛 JTC 경로로 내려야 했다. 자기해제형만 남은 HOLD 에서는
    goto_home(0.1 rad/s 램프)을 허용해 스스로 빠져나오게 한다.
    """
    kinds = _kinds(state)
    return (state.phase is Phase.HOLD and bool(kinds)
            and all(k.startswith(SELF_CLEARING_KINDS) for k in kinds))


def _target_fresh(state: FsmState) -> FsmState:
    """목표가 다시 들어왔다. **워치독만이 사유이면** HOLD 를 푼다.

    ★2026-09-07 우팔 실기: engage 와 스트림 시작 사이 공백으로 워치독 HOLD 에 들어간 뒤,
    104 초짜리 자세 이동이 목표 2612 개를 전부 보냈고 pd 도 다 받았는데(seq 2611) 세트포인트가
    얼어 있어 팔이 안 움직였다. 워치독은 "목표가 없다"는 양성 조건이라 조건이 사라지면 재개하는
    것이 옳다. 추종오차·관절한계·τ·발열·estop 은 사람이 판단할 사유이므로 남긴다.
    RAMPING 으로 돌아가 세트포인트가 실측에서 다시 접근하게 한다(TRACKING 직행 금지).
    """
    if set(_kinds(state)) != {WATCHDOG_KIND}:
        return state
    return clear_reasons(state, (WATCHDOG_KIND,))


def _zero_tick(state: FsmState) -> FsmState:
    ticks = state.zero_ticks + 1
    if ticks >= state.release_zero_ticks:
        return replace(state, phase=Phase.IDLE, hold_reason=None, zero_ticks=0)
    return replace(state, zero_ticks=ticks)


def law_flags(phase: Phase) -> tuple[bool, bool]:
    """(advance, tracking) for pd_law: setpoint may move / policy feed-forward allowed."""
    return (phase in _MOVING, phase is Phase.TRACKING)


# ------------------------------------------------------------------ engage refusals
@dataclass(frozen=True)
class EngageCheck:
    execute: bool
    state_age_sec: float | None     # None = 실측을 한 번도 못 받음
    stale_sec: float
    gains_ok: bool
    accept_sim_mismatch: bool
    gravity_conflict: str | None
    effort_controller_active: bool
    estop_latched: bool
    phase: Phase
    thermal_unknown: Sequence[str] = ()   # 온도 근거인데 서미스터가 한 번도 안 왔다


def engage_refusals(check: EngageCheck) -> list[str]:
    """Every reason engage must be refused, in one list (empty = allowed)."""
    reasons: list[str] = []
    if not check.execute:
        reasons.append("execute is false (dry run) — set execute:=true to touch controllers")
    if check.state_age_sec is None or check.state_age_sec > check.stale_sec:
        reasons.append(f"joint state stale (age {check.state_age_sec} s > {check.stale_sec} s)")
    if not check.gains_ok and not check.accept_sim_mismatch:
        reasons.append("driver kp != trained kp (gain mismatch); accept_sim_mismatch not set")
    if check.gravity_conflict:
        reasons.append(f"gravity mode conflict: {check.gravity_conflict}")
    if check.effort_controller_active:
        reasons.append("forward effort controller already active (gravity_comp_node?)")
    if check.estop_latched:
        reasons.append("estop latched")
    for joint in check.thermal_unknown:
        reasons.append(f"thermal rule {joint} needs motor temperature but none has arrived "
                       "(driver does not export temperature_rotor?)")
    if check.phase is not Phase.IDLE:
        reasons.append(f"phase {check.phase.value} is not IDLE")
    return reasons


# ------------------------------------------------------------------ fault detection
@dataclass(frozen=True)
class FaultInputs:
    target_age_sec: float | None    # None = 아직 첫 목표 없음(watchdog 대상 아님)
    watchdog_sec: float
    tracking_err: float             # max |q_setpoint − q_meas| [rad]
    abort_tracking: float
    target_clipped: bool            # 목표가 관절 한계 밖이었다(pd_law 'position')
    effort_fault: bool              # τ 합이 cap 초과(pd_law)
    estop_latched: bool
    thermal_act: Sequence[str]      # thermal_act_joints(...) 결과
    switch_failed: bool
    thermal_stale: Sequence[str] = ()   # 온도 규칙인데 센서가 안 옴 — '차갑다'로 읽으면 보호가 꺼진다
    thermal_retreat: bool = False       # 지금 저부하 자세로 내려가는 중 = 발열은 이미 조치 중


def detect_faults(inp: FaultInputs) -> list[str]:
    """Reasons that must send the FSM to HOLD this tick (empty = none)."""
    reasons: list[str] = []
    if inp.target_age_sec is not None and inp.target_age_sec > inp.watchdog_sec:
        reasons.append(f"watchdog: target stale {inp.target_age_sec:.3f} s > {inp.watchdog_sec} s")
    if inp.tracking_err > inp.abort_tracking:
        reasons.append(f"tracking error {inp.tracking_err:.3f} rad > {inp.abort_tracking}")
    if inp.target_clipped:
        reasons.append("joint limit: target outside profile bounds")
    if inp.effort_fault:
        reasons.append("effort cap exceeded (tau zeroed)")
    if inp.estop_latched:
        reasons.append("estop latched")
    if not inp.thermal_retreat:
        # 후퇴 중에 발열이 다시 HOLD 를 걸면 팔은 **든 채로** 얼어붙는다(09.07 교착).
        # 이미 저부하 자세로 내려가는 중 = 발열에 대한 조치가 진행 중이므로 두 사유를 다 누른다.
        for joint in inp.thermal_act:
            reasons.append(f"thermal {joint}: over temperature")
        for joint in inp.thermal_stale:
            reasons.append(f"thermal sensor {joint}: temperature missing")
    if inp.switch_failed:
        reasons.append("controller switch failed")
    return reasons


# ------------------------------------------------------------------ thermal
@dataclass(frozen=True)
class ThermalRule:
    """한 관절의 발열 판정 근거. **temp 또는 effort 중 정확히 하나**를 쓴다.

    temp   — 모터 서미스터(`t_rotor`/`t_mos`, °C). `temp_act_c` 에서 걸고 `temp_clear_c`
             아래로 내려가야 풀린다(히스테리시스). 조건이 사라지면 스스로 풀리는 사유다.
    effort — 온도가 없는 소스(fake plant)용 옛 대리지표. |τ| 가 문턱을 넘은 **시간**을
             적분한다. 09.05~09.07 에 유일한 수단이었고 임계값은 고장 사례 한 건에서 나왔다.
    """

    joint: str
    effort_nm: float | None = None
    act_sec: float | None = None
    warn_sec: float | None = None
    temp_act_c: float | None = None
    temp_clear_c: float | None = None
    temp_warn_c: float | None = None

    def __post_init__(self) -> None:
        has_temp, has_effort = self.temp_act_c is not None, self.effort_nm is not None
        if has_temp == has_effort:
            raise ValueError(f"thermal rule {self.joint}: give exactly one of temp_act_c / effort_nm")
        if has_temp:
            self._check_temp()
        else:
            self._check_effort()

    def _check_temp(self) -> None:
        if self.temp_clear_c is None:
            raise ValueError(f"thermal rule {self.joint}: temp_act_c needs temp_clear_c (hysteresis)")
        if self.temp_clear_c >= self.temp_act_c:
            raise ValueError(f"thermal rule {self.joint}: temp_clear_c must be < temp_act_c")
        if self.temp_warn_c is not None and not self.temp_clear_c <= self.temp_warn_c < self.temp_act_c:
            raise ValueError(f"thermal rule {self.joint}: temp_warn_c must be in [clear, act)")
        if self.act_sec is not None or self.warn_sec is not None:
            raise ValueError(f"thermal rule {self.joint}: act_sec/warn_sec belong to the effort basis")

    def _check_effort(self) -> None:
        if self.effort_nm <= 0.0 or not self.act_sec or self.act_sec <= 0.0:
            raise ValueError(f"thermal rule {self.joint}: effort_nm and act_sec must be > 0")
        if self.warn_sec is not None and not 0.0 < self.warn_sec < self.act_sec:
            raise ValueError(f"thermal rule {self.joint}: warn_sec must be in (0, act_sec)")

    @property
    def basis(self) -> str:
        return "temp" if self.temp_act_c is not None else "effort"


@dataclass(frozen=True)
class ThermalState:
    hot_sec: tuple                  # effort 근거: 문턱 위에 머문 누적 시간 [s]
    latched: tuple = ()             # temp 근거: 지금 act 로 걸려 있나
    last_c: tuple = ()              # temp 근거: 이번 tick 에 온 온도(없으면 nan)
    ever: tuple = ()                # temp 근거: 온도를 **한 번이라도** 받은 적이 있나

    def _sized(self, n: int) -> "ThermalState":
        """옛 호출부(hot_sec 만 주는 곳)를 위해 나머지 칸을 채운다."""
        if len(self.latched) == n and len(self.last_c) == n and len(self.ever) == n:
            return self
        return ThermalState(hot_sec=self.hot_sec, latched=tuple(False for _ in range(n)),
                            last_c=tuple(float("nan") for _ in range(n)),
                            ever=tuple(False for _ in range(n)))


_RULE_KEYS = {"joint", "effort_nm", "act_sec", "warn_sec", "temp_act_c", "temp_clear_c", "temp_warn_c"}


def thermal_rules_from_config(items: Sequence[Mapping]) -> tuple[ThermalRule, ...]:
    rules = []
    for item in items:
        unknown = set(item) - _RULE_KEYS
        if unknown or "joint" not in item:
            raise ValueError(f"thermal rule {dict(item)}: unknown {sorted(unknown)}, joint required")
        kw = {k: (None if item.get(k) is None else float(item[k])) for k in _RULE_KEYS - {"joint"}}
        rules.append(ThermalRule(joint=str(item["joint"]), **kw))
    return tuple(rules)


def thermal_init(rules: Sequence[ThermalRule]) -> ThermalState:
    n = len(rules)
    return ThermalState(hot_sec=tuple(0.0 for _ in range(n)), latched=tuple(False for _ in range(n)),
                        last_c=tuple(float("nan") for _ in range(n)),
                        ever=tuple(False for _ in range(n)))


def thermal_step(state: ThermalState, rules: Sequence[ThermalRule], efforts: Mapping[str, float],
                 dt: float, temps: Mapping[str, float] | None = None) -> ThermalState:
    """Advance both bases one tick; *temps* holds whatever the driver reported this tick."""
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if len(state.hot_sec) != len(rules):
        raise ValueError(f"thermal state has {len(state.hot_sec)} entries for {len(rules)} rules")
    state = state._sized(len(rules))
    temps = {} if temps is None else temps
    hot, latched, last, ever = [], [], [], []
    for rule, sec, was, seen_before in zip(rules, state.hot_sec, state.latched, state.ever):
        if rule.basis == "temp":
            hot.append(sec)
            t = temps.get(rule.joint)
            last.append(float("nan") if t is None else float(t))
            latched.append(was if t is None else _latch(rule, float(t), was))
            ever.append(bool(seen_before or t is not None))
            continue
        if rule.joint not in efforts:
            raise KeyError(f"thermal rule joint {rule.joint!r} missing from efforts")
        above = abs(float(efforts[rule.joint])) > rule.effort_nm
        hot.append(sec + dt if above else max(0.0, sec - dt))
        latched.append(False)
        last.append(float("nan"))
        ever.append(True)
    return ThermalState(hot_sec=tuple(hot), latched=tuple(latched), last_c=tuple(last), ever=tuple(ever))


def _latch(rule: ThermalRule, t: float, was: bool) -> bool:
    if t >= rule.temp_act_c:
        return True
    if t <= rule.temp_clear_c:
        return False
    return was


def thermal_levels(state: ThermalState, rules: Sequence[ThermalRule]) -> dict[str, str]:
    """joint → 'ok' | 'warn' | 'act' | 'stale'."""
    state = state._sized(len(rules))
    levels = {}
    for rule, sec, hot, t, ever in zip(rules, state.hot_sec, state.latched, state.last_c, state.ever):
        if rule.basis == "temp":
            levels[rule.joint] = _temp_level(rule, hot, t, ever)
        elif sec >= rule.act_sec:
            levels[rule.joint] = "act"
        elif rule.warn_sec is not None and sec >= rule.warn_sec:
            levels[rule.joint] = "warn"
        else:
            levels[rule.joint] = "ok"
    return levels


def _temp_level(rule: ThermalRule, latched: bool, t: float, ever: bool) -> str:
    """'unknown' 과 'stale' 은 다른 사건이다.

    unknown — 온도를 한 번도 못 받았다. 드라이버가 서미스터를 export 안 하는 구형 브링업이거나
              fake 플랜트다. **engage 를 거부**해서 사람이 즉시 알게 한다(조용히 HOLD 하지 않는다).
    stale   — 오다가 끊겼다. 운전 중 사건이므로 HOLD(자기해제형: 다시 오면 풀린다).
    """
    if latched:
        return "act"
    if t != t:                                  # nan = 이번 tick 에 온도가 안 왔다
        return "stale" if ever else "unknown"
    if rule.temp_warn_c is not None and t >= rule.temp_warn_c:
        return "warn"
    return "ok"


def thermal_act_joints(state: ThermalState, rules: Sequence[ThermalRule]) -> tuple[str, ...]:
    return tuple(j for j, lvl in thermal_levels(state, rules).items() if lvl == "act")


def thermal_stale_joints(state: ThermalState, rules: Sequence[ThermalRule]) -> tuple[str, ...]:
    return tuple(j for j, lvl in thermal_levels(state, rules).items() if lvl == "stale")


def thermal_unknown_joints(state: ThermalState, rules: Sequence[ThermalRule]) -> tuple[str, ...]:
    return tuple(j for j, lvl in thermal_levels(state, rules).items() if lvl == "unknown")

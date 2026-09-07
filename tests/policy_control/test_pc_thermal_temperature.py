"""발열 가드를 실제 온도로 판정한다 — 토크×시간 대리지표의 은퇴 (ROS 없음).

2026-09-07 확인: DM 모터는 MIT 피드백 프레임마다 `t_rotor`/`t_mos` 를 실어 보내고
`openarm_can` 이 이미 디코딩하고 있었는데, ros2_control 하드웨어 인터페이스가
position/velocity/effort 셋만 export 해서 ROS 로 안 올라왔다. 그래서 09.05 에
"3.17 N·m 로 18 분이면 탄다"는 **한 번의 고장 사례**로 1.5 N·m/300 s 를 지어냈다.
같은 관절인데 좌 5.0 / 우 1.5 로 갈린 것이 그 임의성의 증거다.

이제 온도가 올라오므로 규칙은 두 갈래다:
  · temp   — temp_act_c 에서 걸고 temp_clear_c 아래로 내려가면 **스스로 풀린다**(히스테리시스)
  · effort — 온도가 없는 곳(fake plant)용 옛 토크 적분

★교착: 발열 HOLD 는 팔을 **내리는 것까지** 막았다. 식으려면 내려야 하는데 내릴 수가 없다.
그래서 (a) 발열 사유는 자기해제형이고, (b) 자기해제형만 남은 HOLD 에서는 goto_home 을
허용하며, (c) 그 후퇴 중에는 발열 사유가 HOLD 를 다시 걸지 않는다.
"""
from __future__ import annotations

import pytest

from policy_control import pd_state as S

pytestmark = pytest.mark.unit

TEMP_RULE = S.ThermalRule(joint="r_aj_7", temp_act_c=70.0, temp_clear_c=55.0, temp_warn_c=60.0)
EFFORT_RULE = S.ThermalRule(joint="l_aj_7", effort_nm=1.5, act_sec=3.0)


# ---------------------------------------------------------------- rule shape
def test_a_rule_must_pick_exactly_one_basis():
    with pytest.raises(ValueError):                       # 아무 근거도 없음
        S.ThermalRule(joint="j")
    with pytest.raises(ValueError):                       # 두 근거를 섞었다
        S.ThermalRule(joint="j", effort_nm=1.5, act_sec=3.0, temp_act_c=70.0, temp_clear_c=55.0)
    with pytest.raises(ValueError):                       # clear 없이 act 만
        S.ThermalRule(joint="j", temp_act_c=70.0)
    with pytest.raises(ValueError):                       # clear 가 act 보다 높다 = 히스테리시스 아님
        S.ThermalRule(joint="j", temp_act_c=55.0, temp_clear_c=70.0)
    with pytest.raises(ValueError):                       # warn 이 대역 밖
        S.ThermalRule(joint="j", temp_act_c=70.0, temp_clear_c=55.0, temp_warn_c=90.0)
    assert TEMP_RULE.basis == "temp"
    assert EFFORT_RULE.basis == "effort"


# ---------------------------------------------------------------- hysteresis
def test_temperature_latches_at_act_and_only_releases_below_clear():
    rules = (TEMP_RULE,)
    st = S.thermal_init(rules)

    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={"r_aj_7": 69.0})
    assert S.thermal_act_joints(st, rules) == ()
    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={"r_aj_7": 70.0})
    assert S.thermal_act_joints(st, rules) == ("r_aj_7",)

    # 60 °C 는 act 아래지만 clear 위 — 걸린 채로 유지된다(채터링 방지)
    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={"r_aj_7": 60.0})
    assert S.thermal_act_joints(st, rules) == ("r_aj_7",)

    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={"r_aj_7": 55.0})
    assert S.thermal_act_joints(st, rules) == ()


def test_warn_band_reports_between_clear_and_act():
    rules = (TEMP_RULE,)
    st = S.thermal_init(rules)
    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={"r_aj_7": 62.0})
    assert S.thermal_levels(st, rules) == {"r_aj_7": "warn"}
    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={"r_aj_7": 50.0})
    assert S.thermal_levels(st, rules) == {"r_aj_7": "ok"}


def test_effort_basis_still_integrates_for_sources_without_a_thermistor():
    rules = (EFFORT_RULE,)
    st = S.thermal_init(rules)
    for _ in range(30):
        st = S.thermal_step(st, rules, efforts={"l_aj_7": 2.0}, dt=0.1, temps={})
    assert S.thermal_act_joints(st, rules) == ("l_aj_7",)


def test_the_two_bases_coexist_and_do_not_read_each_other():
    rules = (TEMP_RULE, EFFORT_RULE)
    st = S.thermal_init(rules)
    # 온도 규칙 관절에 큰 토크를 줘도 온도가 낮으면 안 걸린다
    st = S.thermal_step(st, rules, efforts={"r_aj_7": 99.0, "l_aj_7": 0.0}, dt=0.1,
                        temps={"r_aj_7": 30.0})
    assert S.thermal_act_joints(st, rules) == ()


# ---------------------------------------------------------------- sensor loss
def test_never_arriving_and_dropping_out_are_different_events():
    """온도가 0.0 으로 오거나 아예 없을 때 '차갑다'로 읽으면 보호가 조용히 꺼진다.

    한 번도 안 온 것(구형 브링업·fake 플랜트)은 **engage 거부** 감이고,
    오다가 끊긴 것은 운전 중 사건이라 HOLD 감이다. 둘을 뭉치면 둘 다 틀린 대응이 된다.
    """
    rules = (TEMP_RULE,)
    st = S.thermal_init(rules)
    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={})
    assert S.thermal_levels(st, rules) == {"r_aj_7": "unknown"}
    assert S.thermal_unknown_joints(st, rules) == ("r_aj_7",)
    assert S.thermal_stale_joints(st, rules) == ()          # 아직 HOLD 감은 아니다

    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={"r_aj_7": 40.0})
    assert S.thermal_unknown_joints(st, rules) == () and S.thermal_stale_joints(st, rules) == ()

    st = S.thermal_step(st, rules, efforts={}, dt=0.1, temps={})   # 오다가 끊겼다
    assert S.thermal_levels(st, rules) == {"r_aj_7": "stale"}
    assert S.thermal_stale_joints(st, rules) == ("r_aj_7",)


def test_engage_is_refused_when_a_temp_rule_has_no_thermistor():
    base = dict(execute=True, state_age_sec=0.0, stale_sec=0.5, gains_ok=True, accept_sim_mismatch=False,
                gravity_conflict=None, effort_controller_active=False, estop_latched=False, phase=S.Phase.IDLE)
    assert S.engage_refusals(S.EngageCheck(**base)) == []
    why = S.engage_refusals(S.EngageCheck(**base, thermal_unknown=("r_aj_7",)))
    assert len(why) == 1 and "temperature" in why[0]


def test_sensor_loss_is_a_fault_reason():
    reasons = S.detect_faults(S.FaultInputs(
        target_age_sec=0.0, watchdog_sec=0.25, tracking_err=0.0, abort_tracking=0.3,
        target_clipped=False, effort_fault=False, estop_latched=False,
        thermal_act=(), thermal_stale=("r_aj_7",), switch_failed=False))
    assert len(reasons) == 1
    assert "thermal sensor r_aj_7" in reasons[0]


# ---------------------------------------------------------------- self-clearing HOLD
def _held(*reasons: str) -> S.FsmState:
    st = S.transition(S.initial_fsm(release_zero_ticks=5), "engage")
    for r in reasons:
        st = S.transition(st, "fault", r)
    return st


def test_thermal_hold_clears_itself_once_the_joint_has_cooled():
    st = _held("thermal r_aj_7: rotor 71 C above 70.0 C")
    assert st.phase is S.Phase.HOLD
    st = S.clear_reasons(st, ("thermal r_aj_7",))
    assert st.phase is S.Phase.RAMPING       # 실측에서 세트포인트를 다시 접근시킨다
    assert st.hold_reason is None


def test_cooling_does_not_clear_a_hold_a_human_must_judge():
    st = _held("thermal r_aj_7: rotor 71 C above 70.0 C", "tracking error 0.400 rad > 0.3")
    st = S.clear_reasons(st, ("thermal r_aj_7",))
    assert st.phase is S.Phase.HOLD
    assert "tracking" in st.hold_reason and "thermal" not in st.hold_reason


def test_watchdog_recovery_still_works_through_the_general_path():
    st = _held("watchdog: target stale 0.400 s > 0.25 s")
    assert S.transition(st, "target_fresh").phase is S.Phase.RAMPING


# ---------------------------------------------------------------- deadlock escape
def test_a_thermal_only_hold_may_retreat_but_a_judged_one_may_not():
    assert S.hold_is_self_clearing(_held("thermal r_aj_7: too hot"))
    assert S.hold_is_self_clearing(_held("thermal sensor r_aj_7: temperature missing"))
    assert S.hold_is_self_clearing(_held("watchdog: target stale"))
    assert not S.hold_is_self_clearing(_held("estop latched"))
    assert not S.hold_is_self_clearing(_held("thermal r_aj_7: too hot", "estop latched"))
    # 움직이는 중에는 후퇴 개념이 없다
    assert not S.hold_is_self_clearing(S.transition(S.initial_fsm(3), "engage"))


def test_retreat_suppresses_the_thermal_fault_so_the_arm_can_come_down():
    """후퇴 중에 발열이 다시 HOLD 를 걸면 팔은 든 채로 다시 얼어붙는다 — 그게 09.07 교착이었다."""
    inp = dict(target_age_sec=0.0, watchdog_sec=0.25, tracking_err=0.0, abort_tracking=0.3,
               target_clipped=False, effort_fault=False, estop_latched=False,
               thermal_act=("r_aj_7",), thermal_stale=(), switch_failed=False)
    assert S.detect_faults(S.FaultInputs(**inp)) != []
    assert S.detect_faults(S.FaultInputs(**inp, thermal_retreat=True)) == []
    # 후퇴는 발열만 눌러준다 — estop 은 그대로 잡는다
    still = S.detect_faults(S.FaultInputs(**{**inp, "estop_latched": True}, thermal_retreat=True))
    assert len(still) == 1 and "estop" in still[0]

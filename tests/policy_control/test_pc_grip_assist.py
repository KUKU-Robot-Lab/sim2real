"""★10.13 배포 전용 손 보조(rh_aglt.GripAssist · AgltChain.lift_grasped) — 사용자 결정 "a, b 둘 다 진행, 제어가 정책을 충실히 따라가게".

실기 bag(logs/bags/20261013_164845 오른손 · 165303 왼손)에서 본 두 상황을 손 실측 모형으로 만든다:
  - 엄지 굽힘은 편 끝 0.05 rad 에 붙어 있고(명령 0.00~0.04 rad 보다 더 굽은 값), 컵이 누르면 관절 힘만 오른다.
  - 검지는 컵(0.45 rad)에서 막힌다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control import pour_fj_node as N
from policy_control import rh_aglt as A
from policy_control import rh_aglt_goals as G

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy" / "policies"
RG3S = {"right": POL / "rh56f1/aglt/right_rg3s/rh_aglt_contract.json",
        "left": POL / "rh56f1/aglt/left_rg3s/rh_aglt_contract.json"}
THUMB_FLOOR, CUP_INDEX = 0.05, 0.45
I_THUMB_FLEX, I_INDEX = 1, 2                          # hand_joints 순(엄지 회전 · 엄지 굽힘 · 검지 · 중지 · 약지 · 새끼)


@pytest.fixture(scope="module", params=["right", "left"])
def c(request):
    c = A.load_contract(RG3S[request.param])
    assert c.hand_block and c.grasp_hold_rule == "block_follow_open"   # 이 보조는 막힘 계약 위에서만 뜻이 있다
    return c


def _close_on_cup(dec: A.RaDecoder, *, ticks: int, thumb_g: float, finger_cup: bool = True) -> list[bool]:
    """검지만 닫으라는 행동(엄지 굽힘은 편 쪽) — 실측: 엄지 굽힘은 편 끝, 검지는 컵에서 막힌다. 매 틱 쥔 판정."""
    a = -np.ones(13)
    a[7 + I_INDEX] = 1.0                                # 검지 → 쥔 자세(1.08 rad) 쪽
    a[7 + 0] = 1.0                                      # 엄지 회전은 열린 쪽(행동 +1 = q_open 1.57 쪽이 아니어도 무관)
    out = []
    for _ in range(ticks):
        q_star = dec.state["arm"].hand_target
        q = q_star.copy()
        q[I_THUMB_FLEX] = max(q_star[I_THUMB_FLEX], THUMB_FLOOR)
        if finger_cup:
            q[I_INDEX] = min(q_star[I_INDEX], CUP_INDEX)
        jf = np.array([thumb_g, 0.0, 150.0 if finger_cup else 0.0, 0.0, 0.0, 0.0])
        dec.step(a, active=True, hand_q=q, joint_force=jf)
        out.append(bool(dec.grasped()))
    return out


def test_by_default_the_thumb_at_its_open_stop_never_counts_as_grasped(c):
    """기본(sim 과 같은 판정): 엄지 래치가 없으면 손가락이 막혀도 쥔 판정이 없다 — 실기 왼손 0/899 틱의 재현."""
    dec = A.RaDecoder(c)
    assert not any(_close_on_cup(dec, ticks=240, thumb_g=130.0))


def test_thumb_support_force_with_a_blocked_finger_is_grasped(c):
    dec = A.RaDecoder(c, A.GripAssist(thumb_support_g=80.0))
    g = _close_on_cup(dec, ticks=240, thumb_g=130.0)
    assert g[-1] and sum(g) > 100


def test_thumb_force_without_a_blocked_finger_is_not_grasped(c):
    """빈손에서 엄지에 힘이 걸려도(받침만) 손가락 막힘이 없으면 쥔 판정이 아니다."""
    dec = A.RaDecoder(c, A.GripAssist(thumb_support_g=80.0))
    assert not any(_close_on_cup(dec, ticks=240, thumb_g=130.0, finger_cup=False))


def test_thumb_force_below_the_threshold_is_not_support(c):
    dec = A.RaDecoder(c, A.GripAssist(thumb_support_g=80.0))
    assert not any(_close_on_cup(dec, ticks=240, thumb_g=30.0))


def test_a_missing_joint_force_is_not_support(c):
    dec = A.RaDecoder(c, A.GripAssist(thumb_support_g=80.0))
    a = -np.ones(13)
    a[7 + I_INDEX] = 1.0
    for _ in range(240):
        q = dec.state["arm"].hand_target.copy()
        q[I_THUMB_FLEX], q[I_INDEX] = max(q[I_THUMB_FLEX], THUMB_FLOOR), min(q[I_INDEX], CUP_INDEX)
        dec.step(a, active=True, hand_q=q, joint_force=None)
    assert not dec.grasped()


def test_without_unfreeze_the_finger_target_stays_at_first_contact(c):
    """막힘 고정은 래치까지 걸린 틱(lag · n_ticks)만큼 접촉점을 지난 자리에서 멈춘다 — 실기 왼손 목표 0.66 · 실측 0.55 와 같은 꼴."""
    dec = A.RaDecoder(c, A.GripAssist(thumb_support_g=80.0))
    _close_on_cup(dec, ticks=200, thumb_g=130.0)
    held = dec.state["arm"].hand_target[I_INDEX]
    _close_on_cup(dec, ticks=40, thumb_g=130.0)
    assert dec.state["arm"].hand_target[I_INDEX] == pytest.approx(held)              # 더 닫지 않는다
    assert held < c.side().hand_grip[I_INDEX] - 0.3                                   # 쥔 자세(1.08)에 한참 못 미친다


def test_unfreeze_on_grasp_lets_the_finger_follow_the_policy(c):
    """A: 쥔 뒤에는 고정을 풀어 정책 목표(검지 쥔 자세 1.08 rad)까지 닫는다 — 쥐기 전에는 기본과 같다."""
    base = A.RaDecoder(c, A.GripAssist(thumb_support_g=80.0))
    dec = A.RaDecoder(c, A.GripAssist(unfreeze_on_grasp=True, thumb_support_g=80.0))
    g = _close_on_cup(dec, ticks=240, thumb_g=130.0)
    _close_on_cup(base, ticks=240, thumb_g=130.0)
    first = g.index(True)
    assert first > 0
    assert dec.state["arm"].hand_target[I_INDEX] > base.state["arm"].hand_target[I_INDEX] + 0.3
    assert dec.state["arm"].hand_target[I_INDEX] == pytest.approx(c.side().hand_grip[I_INDEX], abs=0.02)


def test_unfreeze_does_nothing_before_a_grasp(c):
    """쥔 판정이 한 번도 없으면 A 를 켜도 목표가 기본과 같다(엄지 받침 없이 = 지금 판정)."""
    a_on = A.RaDecoder(c, A.GripAssist(unfreeze_on_grasp=True))
    a_off = A.RaDecoder(c)
    _close_on_cup(a_on, ticks=240, thumb_g=130.0)
    _close_on_cup(a_off, ticks=240, thumb_g=130.0)
    np.testing.assert_allclose(a_on.state["arm"].hand_target, a_off.state["arm"].hand_target)


def test_reset_clears_the_thumb_support(c):
    dec = A.RaDecoder(c, A.GripAssist(thumb_support_g=80.0))
    _close_on_cup(dec, ticks=240, thumb_g=130.0)
    dec.reset()
    assert not dec.grasped() and dec._support is False


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_a_bad_thumb_threshold_is_refused(bad):
    with pytest.raises(A.RhAgltError):
        A.GripAssist(thumb_support_g=bad)


def _chain_with_cup(c, cup_reset_z: float) -> N.AgltChain:
    ch = N.AgltChain(c, policy=None)
    ch.goals = G.GoalBook.start(c, np.array([0.3, -0.1, cup_reset_z]), np.array([1.0, 0, 0, 0]), freeze=ch.freeze)
    return ch


def _meas(cup, palm) -> A.RaMeas:
    return A.RaMeas(arm_q=np.zeros(7), arm_qd=np.zeros(7), hand_q={}, palm_pos=np.asarray(palm, float), palm_R=np.eye(3),
                    tips=np.zeros((5, 3)), cup_pos=np.asarray(cup, float), cup_quat=np.array([1.0, 0, 0, 0]),
                    tactile_n=np.zeros(5))


def test_lift_grasp_is_off_by_default(c):
    ch = _chain_with_cup(c, 0.26)
    assert ch.lift_grasp_palm_m == 0.0
    assert not ch.lift_grasped(_meas([0.3, -0.1, 0.38], [0.3, -0.05, 0.38]))


def test_lift_grasp_needs_the_lift_and_the_cup_near_the_palm(c):
    """B: 왼손 165303 — 쉐이커 10~14 cm 들림 · 컵-손바닥 4~8 cm → 쥔 것으로 본다. 덜 들렸거나 손에서 멀면 아니다."""
    ch = _chain_with_cup(c, 0.26)
    ch.lift_grasp_palm_m = 0.08
    assert ch.lift_grasped(_meas([0.3, -0.1, 0.36], [0.3, -0.05, 0.36]))          # 10 cm 들림 · 5 cm
    assert not ch.lift_grasped(_meas([0.3, -0.1, 0.32], [0.3, -0.05, 0.32]))      # 6 cm 들림(< freeze_lift_m 7 cm)
    assert not ch.lift_grasped(_meas([0.3, -0.1, 0.36], [0.3, 0.05, 0.36]))       # 손바닥에서 15 cm


def _fake_node(c, **params):
    from types import SimpleNamespace
    vals = {"object": "", "freeze_rule": "lift_hold", "freeze_lift_m": 0.07, "freeze_hold_steps": 30,
            "hand_unfreeze_on_grasp": False, "thumb_support_g": 0.0, "lift_grasp_palm_m": 0.0, **params}
    chain = N.AgltChain(c, policy=None)
    chain.reset = lambda meas: None                   # 체인 리셋 자체는 다른 테스트가 본다 — 여기선 파라미터 배선만
    return SimpleNamespace(chain=chain, get_parameter=lambda n: SimpleNamespace(value=vals[n])), chain


def test_the_node_wires_the_assist_parameters_into_the_chain_at_reset(c):
    node, chain = _fake_node(c, hand_unfreeze_on_grasp=True, thumb_support_g=80.0, lift_grasp_palm_m=0.08)
    N.PourFjNode._reset_chain(node, {})
    assert chain.dec.assist == A.GripAssist(True, 80.0) and chain.lift_grasp_palm_m == 0.08


def test_the_node_defaults_leave_the_sim_behaviour(c):
    node, chain = _fake_node(c)
    N.PourFjNode._reset_chain(node, {})
    assert chain.dec.assist == A.GripAssist() and chain.lift_grasp_palm_m == 0.0


@pytest.mark.parametrize("bad", [{"thumb_support_g": -5.0}, {"lift_grasp_palm_m": -0.1}, {"lift_grasp_palm_m": float("nan")}])
def test_bad_assist_parameters_refuse_the_reset(c, bad):
    node, _ = _fake_node(c, **bad)
    with pytest.raises(A.RhAgltError):
        N.PourFjNode._reset_chain(node, {})

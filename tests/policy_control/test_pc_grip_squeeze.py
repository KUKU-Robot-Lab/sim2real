"""★10.19 배포 전용 쥠 보조 D(rh_aglt.GripAssist squeeze) — 사용자: "sim 은 세게 쥐면 물리가 터져 적당히 쥐게 학습했다.
실기는 그럴 이유가 없으니 제어로 더 세게 쥐게만 하면 된다" · "정책은 원래 상태를 보고, 제어가 움직임을 보조해 더 닫게".

근거 bag 20261019_133007(왼손 neo03 · source240): 정책 목표는 쥔 자세의 0.7~0.85 까지 가고 손가락이 그 목표에 그대로 닿아
힘 2~50 g(덜 쥠) · 10 cm 든 뒤 정책이 검지 · 중지를 풀어 떨굼.
손 모형: 손가락은 보낸 명령을 따라가다 병 표면(CUP_Q)에서 멈춘다(실측 = min(명령, CUP_Q)), 헛잡음이면 끝까지 따라간다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control import pour_fj_node as N
from policy_control import rh_aglt as A

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy" / "policies" / "rh56f1" / "aglt"
RET = REPO / "deploy" / "policies_retired" / "rh56f1" / "aglt"   # ★10.20 옛 neo03(시험 고정본)
F = [2, 3, 4, 5]                                     # hand_joints 순 — 검지 · 중지 · 약지 · 새끼
CUP_Q = np.array([0.0, 0.0, 0.85, 0.85, 0.7, 0.7])   # 병에 닿는 손가락 각(엄지는 쓰지 않음) — 정책 목표(frac 0.7)보다 약 0.1 rad 안쪽
HZ = 60


@pytest.fixture(scope="module", params=["left_neo03", "right_neo03m"])
def c(request):
    c = A.select_object(A.load_contract(RET / request.param / "rh_aglt_contract.json"), "source240_real")
    assert c.hand_block and c.arm_hold_rise_m > 0.0
    return c


def _act(finger: float) -> np.ndarray:
    a = np.zeros(13)
    a[7:] = -1.0
    a[7 + np.array(F)] = finger                        # 손가락 행동(−1 편 · +1 쥔 자세) — frac = (a + 1) / 2
    return a


def _run(dec: A.RaDecoder, acts, *, cup: bool = True, palm_m: float = 0.04, cup_z=None, q0=None) -> list[np.ndarray]:
    """손 모형으로 굴린다 → 매 스텝 보낸 손 명령. 실측 = 직전 명령(병이면 CUP_Q 에서 멈춤)."""
    q = np.array(dec.c.side().hand_open, float) if q0 is None else np.array(q0, float)
    out = []
    for i, a in enumerate(acts):
        z = 0.3 if cup_z is None else cup_z(i)
        t = dec.step(a, active=True, hand_q=q, joint_force=np.zeros(6), obj_palm_m=palm_m, cup_z=z)["arm"][1]
        out.append(t.copy())
        q = np.minimum(t, CUP_Q) if cup else t.copy()
        q[1] = max(q[1], 0.05)                         # 엄지 굽힘 편 끝(실기 0.05)
    return out


def test_off_by_default_the_decoder_is_unchanged(c):
    acts = [_act(0.2)] * 120
    a, b = A.RaDecoder(c), A.RaDecoder(c, A.GripAssist(squeeze=False))
    for x, y in zip(_run(a, acts), _run(b, acts)):
        assert np.array_equal(x, y)
    assert a.squeeze_view() == {}


def test_the_boost_rides_on_the_sent_command_and_the_policy_keeps_its_own_target(c):
    """정책은 frac 0.7 까지만 닫는다(병 표면보다 약 0.1 rad 덜) → 보내는 명령만 정책 목표 + 0.2 rad, 손가락은 병을 민다.
    정책 자기 목표(law 상태 · 관측 hand_err 의 q*)는 보조가 없을 때와 같다."""
    dec, plain = A.RaDecoder(c, A.GripAssist(squeeze=True)), A.RaDecoder(c)
    out, base = _run(dec, [_act(0.4)] * 180), _run(plain, [_act(0.4)] * 180)
    assert dec.squeeze_on and dec.squeeze_counts["on"] == 1
    pol = dec.state["arm"].hand_target
    assert np.allclose(pol, plain.state["arm"].hand_target)                 # 정책 목표는 그대로
    assert np.allclose(base[-1], pol)                                        # 보조가 없으면 보내는 것 = 정책 목표
    assert np.all(pol[F] < CUP_Q[F])                                         # 정책만이면 병에 닿지 않는다(힘 0)
    assert np.allclose(out[-1][F] - pol[F], dec.assist.squeeze_boost_rad)    # 보조 = 정책 목표 + 0.2 rad
    assert np.array_equal(out[-1][:2], pol[:2])                              # 엄지는 정책 그대로
    v = dec.squeeze_view()["squeeze"]
    assert v["pushing"] == 4 and v["boost_rad"] == pytest.approx(0.2)
    assert dec.grasped()                                                     # 둘 이상 밀고 있음 = 쥠(엄지 래치 없이)


def test_the_lead_never_passes_the_pd_cap(c):
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True, squeeze_boost_rad=0.4))
    q_cup = np.array([0.0, 0.0, 0.5, 0.5, 0.4, 0.4])                         # 병이 정책 목표보다 훨씬 안쪽
    q = np.array(c.side().hand_open, float)
    for _ in range(180):
        t = dec.step(_act(0.4), active=True, hand_q=q, joint_force=np.zeros(6), obj_palm_m=0.04, cup_z=0.3)["arm"][1]
        q = np.minimum(t, q_cup)
    assert np.allclose(t[F], np.maximum(dec.state["arm"].hand_target[F], q_cup[F] + c.hand_lead_cap_rad))


def test_far_from_the_cup_or_a_light_close_does_not_engage(c):
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    _run(dec, [_act(0.6)] * 60, palm_m=0.2)                                  # 손바닥에서 20 cm
    assert not dec.squeeze_on
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    _run(dec, [_act(-0.2)] * 120)                                            # frac 0.4 < 0.5
    assert not dec.squeeze_on
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    _run(dec, [_act(0.6)] * 30, palm_m=float("nan"))                         # 컵 거리 모름 = 켜지 않음
    assert not dec.squeeze_on


def test_before_a_lift_the_boost_shrinks_as_the_policy_opens(c):
    """들리기 전 정책이 frac 0.3 으로 풀면 보조도 (0.3 − 0.15)/(0.5 − 0.15) 배로 준다 — 정책의 다시 잡기를 막지 않는다."""
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    out = _run(dec, [_act(0.4)] * 90 + [_act(-0.4)] * 120)
    pol = dec.state["arm"].hand_target
    assert dec.squeeze_on
    assert np.allclose(out[-1][F] - pol[F], 0.2 * (0.3 - 0.15) / (0.5 - 0.15), atol=2e-3)


def test_a_policy_that_opens_for_a_second_without_a_lift_gets_its_hand_back(c):
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    _run(dec, [_act(0.4)] * 90 + [_act(-1.0)] * 30)
    assert dec.squeeze_on                                                    # 정책 목표가 0.15 아래로 1 s 가기 전
    _run(dec, [_act(-1.0)] * 90)
    assert not dec.squeeze_on and dec.squeeze_counts["open"] == 1
    out = _run(dec, [_act(-1.0)] * 30, q0=CUP_Q)
    assert np.allclose(out[-1], dec.state["arm"].hand_target) and np.all(out[-1][F] < 0.1)


def test_a_lifted_cup_is_never_released_on_the_policy_open(c):
    """bag: 10 cm 든 뒤 정책이 검지 · 중지를 풀어 떨굼 — 팔 들림 래치 동안 보낸 명령보다 펴지 않는다."""
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    lift = lambda i: 0.3 if i < 90 else 0.3 + 0.1                           # 90 스텝부터 관측 컵 10 cm 위 → 팔 들림 래치
    out = _run(dec, [_act(0.4)] * 90 + [_act(-1.0)] * 240, cup_z=lift)
    assert dec.arm_latched and dec.squeeze_on and dec.squeeze_counts["open"] == 0
    assert np.all(out[-1][F] >= out[89][F] - 1e-9) and np.all(out[-1][F] > CUP_Q[F])
    assert np.all(dec.state["arm"].hand_target[F] < 0.1)                    # 정책은 폈다고 보고 있다(목표는 그대로)


def test_a_miss_releases_and_waits_for_the_policy_to_open_before_trying_again(c):
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    out = _run(dec, [_act(0.4)] * 150, cup=False)                            # 병 없음 → 명령까지 막힘 없이 닫힘 → 헛잡음
    assert not dec.squeeze_on and dec.squeeze_counts["miss"] == 1
    assert np.allclose(out[-1], dec.state["arm"].hand_target)               # 정책 목표로 돌아간다
    _run(dec, [_act(0.4)] * 30, cup=False)
    assert not dec.squeeze_on                                                # 정책이 아직 닫은 채 — 다시 켜지 않는다
    _run(dec, [_act(-1.0)] * 30 + [_act(0.4)] * 20)
    assert dec.squeeze_on and dec.squeeze_counts["on"] == 2                  # 한 번 편 뒤 다시 닫으면 다시


def test_a_cup_does_not_read_as_a_miss(c):
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    _run(dec, [_act(0.4)] * 600)
    assert dec.squeeze_on and dec.squeeze_counts["miss"] == 0


def test_reset_turns_it_off(c):
    dec = A.RaDecoder(c, A.GripAssist(squeeze=True))
    _run(dec, [_act(0.4)] * 60)
    assert dec.squeeze_on
    dec.reset()
    assert dec.squeeze_view()["squeeze"]["boost_rad"] == 0.0
    assert not dec.squeeze_on and dec.squeeze_counts == {"on": 0, "miss": 0, "open": 0}


@pytest.mark.parametrize("bad", [{"squeeze_trigger_frac": 0.0}, {"squeeze_near_m": -0.1}, {"squeeze_rate_rad_s": float("nan")},
                                 {"squeeze_release_frac": 0.6}, {"squeeze_boost_rad": 0.0}, {"squeeze_boost_rad": 0.8}])
def test_bad_squeeze_settings_are_refused(bad):
    with pytest.raises(A.RhAgltError):
        A.GripAssist(squeeze=True, **bad)


def test_the_node_wires_hand_squeeze(c):
    from types import SimpleNamespace
    vals = {"object": "", "freeze_rule": "lift_hold", "freeze_lift_m": 0.07, "freeze_hold_steps": 30,
            "hand_unfreeze_on_grasp": True, "thumb_support_g": 80.0, "lift_grasp_palm_m": 0.08, "thumb_open_floor_rad": 0.06,
            "approach_grasp": "", "hand_squeeze": True, "squeeze_trigger_frac": 0.6, "squeeze_near_m": 0.08, "squeeze_rate_rad_s": 1.0,
            "squeeze_boost_rad": 0.3}
    chain = N.AgltChain(c, policy=None)
    chain.reset = lambda meas: None
    N.PourFjNode._reset_chain(SimpleNamespace(chain=chain, get_parameter=lambda n: SimpleNamespace(value=vals[n]),
                                              _inhand_run=lambda handoff: None), {})   # ★10.21 추정기는 test_pc_inhand_est
    a = chain.dec.assist
    assert a.squeeze and (a.squeeze_trigger_frac, a.squeeze_near_m, a.squeeze_rate_rad_s, a.squeeze_boost_rad) == (0.6, 0.08, 1.0, 0.3)
    assert chain.dec.squeeze_view()["squeeze"]["on"] is False


def test_rg3_grasp_episodes_turn_it_on_and_path_c_does_not():
    import yaml

    m = yaml.safe_load((REPO / "config/mission_rh56f1_control.yaml").read_text())
    for side in ("right", "left"):
        node = next(c["argv"] for c in m["run"][f"episode_grasp_d_{side}"] if "rh_aglt_node.py" in " ".join(c.get("argv", [])))
        assert "hand_squeeze:=true" in node
        node = next(c["argv"] for c in m["run"][f"episode_grasp_c_{side}"] if "rh_aglt_node.py" in " ".join(c.get("argv", [])))
        assert "hand_squeeze:=true" not in node

"""rh_place 계열(RH56F1 한 팔 컵 홀더 놓기) — 계약 · 인계 시작 · 붙은 컵 · 놓음 → 스크립트 → 끝. 10.04 사용자: 성공한 정책부터 실기에.

hdgp rh_place_r(51013e96): 관측 · 행동 · 디코더는 rh_aglt 와 같고, 목표 = 홀더 자리(seat_pos), 시작 = aglt 가 쥔 상태의 뱅크 행
(팔 · 손 목표 = aglt 마지막 q*), 컵 관측 = 인계 순간 손바닥 기준 상대 자세로 붙인 것(cup_obs_mode attached), 놓음 = 손가락 · 손바닥
접촉 < 1 N 5 스텝 → 45 스텝 스크립트(손 펴기 · 팔은 시작 관절로) → 끝.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control import rh_aglt as A
from policy_control import rh_place as P

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy" / "policies"
RUNS = {"right": POL / "right_rh_place_i09", "left": POL / "left_rh_place_i01"}
URDF = Path.home() / "rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"


def _need(run: Path) -> None:
    if not (run / "params" / "env.yaml").is_file() or not URDF.is_file():
        pytest.skip(f"{run.name} params 또는 RH56F1 URDF 없음")


@pytest.fixture(scope="module", params=["right", "left"])
def c(request) -> P.PlaceContract:
    run = RUNS[request.param]
    _need(run)
    path = run / "rh_place_contract.json"
    if path.is_file():
        return P.load_contract(path)
    from policy_control.pour_profiles import load_profile
    from policy_control import _paths
    return P.build(run, next((run / "nn").glob("*.pth")), load_profile(_paths.RL_WS / "hdgp", "rh56f1_right"), URDF,
                   asset="openarm_rh56f1_bi_rl")


def test_the_contract_is_the_place_run(c):
    s = c.side()
    assert c.schema == P.SCHEMA and c.task == f"open-rh_{s.side[0]}_place"
    assert (c.obs_dim, c.action_dim, c.hold_steps, c.policy_hz, c.episode_s) == (96, 13, 0, pytest.approx(60.0), 15.0)
    assert c.cup_half_height == pytest.approx(0.085)                      # cyl60 — 덤프도 맞다(클래스 기본 물체)
    assert c.seat_dz == pytest.approx(0.060)                              # HOLDER_FLOOR_Z −0.025 − cup_bottom_z −0.085
    assert c.target_holders == ([1, 2] if s.side == "right" else [0, 1])
    assert (c.release_steps, c.settle_steps, c.release_force_n) == (5, 45, 1.0)
    assert c.goal_box_min == [] and c.goal_delta_distance == 0.0         # 목표 입력(aglt)은 쓰지 않는다 — 홀더가 목표


def test_an_aglt_run_is_refused():
    run = POL / "right_rh_aglt_cyl60g"
    _need(run)
    from policy_control.pour_profiles import load_profile
    from policy_control import _paths
    with pytest.raises(A.RhAgltError, match="rh_place"):
        P.build(run, next((run / "nn").glob("*.pth")), load_profile(_paths.RL_WS / "hdgp", "rh56f1_right"), URDF,
                asset="openarm_rh56f1_bi_rl")


def test_seat_is_the_holder_origin_raised_to_the_cup_origin(c):
    g = P.seat_goal(c, [0.38, -0.002, 0.235])
    assert g.pos == pytest.approx([0.38, -0.002, 0.295]) and g.quat == pytest.approx([1, 0, 0, 0])   # sim 자리 z 0.295


def test_the_scripted_open_hand_action_is_the_sim_one(c):
    a = P.open_hand_action(c)
    assert a == pytest.approx([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])        # 엄지 외전만 +1(편 손이 범위 위쪽)
    q = A.hand_law(c).raw(a)
    assert q == pytest.approx(c.side().hand_open)


def _meas(c, *, arm_q=None, palm=(0.30, -0.10, 0.40), R=np.eye(3), cup=(0.30, -0.10, 0.35), tact=(2, 2, 2, 0, 0), jf=None):
    s = c.side()
    return A.RaMeas(arm_q=np.asarray(arm_q if arm_q is not None else s.arm_home, float), arm_qd=np.zeros(7),
                    hand_q=dict(zip(s.hand_joints, s.hand_grip)), palm_pos=np.asarray(palm, float), palm_R=np.asarray(R, float),
                    tips=np.tile(palm, (5, 1)), cup_pos=np.asarray(cup, float), cup_quat=np.array([1.0, 0, 0, 0]),
                    tactile_n=np.asarray(tact, float), joint_force=jf)


class _Zero:
    def forward(self, obs):
        return np.zeros(13)


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def test_reset_starts_from_the_held_targets_and_glues_the_cup_to_the_palm(c):
    ch = P.PlaceChain(c, _Zero())
    held = (np.asarray(c.side().arm_home) + 0.01, np.asarray(c.side().hand_grip) * 0.9)
    ch.reset({"arm": _meas(c)}, holder=[0.38, -0.002, 0.235], held=held)
    st = ch.dec.state["arm"]
    assert st.arm_target == pytest.approx(held[0]) and st.hand_target == pytest.approx(held[1])   # sim 뱅크 arm_q_target · hand_target
    pos, quat = ch.cup_from_palm(np.array([0.40, 0.0, 0.50]), _rz(90))       # 손바닥이 돌고 옮겨 가도 컵은 손바닥에 붙어 따라온다
    assert pos == pytest.approx([0.40, 0.0, 0.45]) and abs(abs(quat[0]) - np.cos(np.radians(45))) < 1e-9
    assert ch.goal.pos == pytest.approx([0.38, -0.002, 0.295])


def test_reset_without_held_targets_uses_the_measured_joints_only_when_told(c):
    ch = P.PlaceChain(c, _Zero())
    with pytest.raises(P.RhPlaceError, match="held"):
        ch.reset({"arm": _meas(c)}, holder=[0.38, -0.002, 0.235], held=None)
    ch.reset({"arm": _meas(c)}, holder=[0.38, -0.002, 0.235], held=None, allow_measured=True)
    assert ch.dec.state["arm"].arm_target == pytest.approx(c.side().arm_home)


def test_release_then_scripted_settle_then_done(c):
    ch = P.PlaceChain(c, _Zero())
    m_held, m_free = _meas(c), _meas(c, tact=(0.2, 0.1, 0, 0, 0))
    ch.reset({"arm": m_held}, holder=[0.38, -0.002, 0.235], held=(np.asarray(c.side().arm_home) + 0.05,
                                                                 np.asarray(c.side().hand_grip)))
    for _ in range(3):
        ch.step({"arm": m_held})
    for k in range(c.release_steps - 1):
        ch.step({"arm": m_free})
        assert not ch.settling
    ch.step({"arm": m_free})                                            # 5 번째 연속 — 놓음
    assert ch.settling and not ch.done
    hand0 = ch.dec.state["arm"].hand_target.copy()
    for _ in range(c.settle_steps - 1):
        _, a, _ = ch.step({"arm": m_free})
        assert a[7:] == pytest.approx(P.open_hand_action(c))           # 정책 대신 스크립트
        assert not ch.done
    ch.step({"arm": m_free})
    assert ch.done
    st = ch.dec.state["arm"]
    assert np.abs(st.hand_target - np.asarray(c.side().hand_open)).max() < np.abs(hand0 - np.asarray(c.side().hand_open)).max()
    assert np.abs(st.arm_target - np.asarray(c.side().arm_home)).max() < 0.05      # 시작 관절(측정) 쪽으로 돌아왔다


def test_a_touch_breaks_the_release_streak(c):
    ch = P.PlaceChain(c, _Zero())
    ch.reset({"arm": _meas(c)}, holder=[0.38, -0.002, 0.235], held=(c.side().arm_home, c.side().hand_grip))
    free, touch = _meas(c, tact=(0, 0, 0, 0, 0)), _meas(c, tact=(0, 1.5, 0, 0, 0))
    for m in (free, free, free, free, touch, free, free, free, free):
        ch.step({"arm": m})
    assert not ch.settling
    jf_free = _meas(c, tact=(0, 0, 0, 0, 0), jf=[400, 0, 0, 0, 0, 0])      # 손끝은 비었어도 관절 힘이 남으면 아직 쥐고 있다
    for _ in range(10):
        ch.step({"arm": jf_free})
    assert not ch.settling


def test_obs_sees_the_seat_as_the_goal_and_the_glued_cup(c):
    ch = P.PlaceChain(c, _Zero())
    holder = [0.38, -0.002, 0.235]
    ch.reset({"arm": _meas(c, cup=(0.38, -0.002, 0.295))}, holder=holder, held=(c.side().arm_home, c.side().hand_grip))
    obs, _, _ = ch.step({"arm": _meas(c)})
    kp = slice(7 + 7 + 7 + 6 + 6 + 3 + 6 + 3 + 3 + 15, 7 + 7 + 7 + 6 + 6 + 3 + 6 + 3 + 3 + 15 + 12)
    assert np.abs(obs[kp]).max() < 1e-9                                 # 붙은 컵이 자리에 있다 → 키포인트 차 0


def test_start_needs_a_grasped_cup(c):
    assert P.start_refusals(c, {"arm": _meas(c, tact=(2, 2, 0, 0, 0))}) == []
    assert any("grasp" in r for r in P.start_refusals(c, {"arm": _meas(c, tact=(0, 0, 0, 0, 0))}))


def test_the_checkpoint_loads_and_acts(c):
    pytest.importorskip("torch")
    from policy_control.joint_policy import JointPolicy
    ch = P.PlaceChain(c, JointPolicy(c, "cpu"))
    ch.reset({"arm": _meas(c)}, holder=[0.38, -0.002, 0.235], held=(c.side().arm_home, c.side().hand_grip))
    for _ in range(5):
        obs, a, t = ch.step({"arm": _meas(c)})
    assert a.size == 13 and np.all(np.isfinite(a)) and np.all(np.isfinite(t["arm"][0]))


def test_the_node_knows_the_place_family(c, tmp_path):
    from policy_control import pour_fj_node as N
    p = tmp_path / "rh_place_contract.json"
    p.write_text(c.to_json())
    assert N.family_of(p) == "rh_place"

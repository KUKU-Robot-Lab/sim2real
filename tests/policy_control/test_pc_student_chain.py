"""11.05 StudentChain — sim aglt_start=ap 학생 운전 순서(대기 → 접근 → 넘김 정지 → 잡기)와 FP++ 컵을 쓰는 자리."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control import rh_aglt as A
from policy_control import rh_aglt_goals as G
from policy_control.student_chain import StudentChain, spec_limits

REPO = Path(__file__).resolve().parents[2]
AGLT = REPO / "deploy" / "policies" / "rh56f1" / "aglt"
AP05 = AGLT / "right_approach_ap05" / "rh_aglt_contract.json"
RCG04 = AGLT / "right_rcg04" / "rh_aglt_contract.json"
BUNDLE = AGLT / "right_student1104e"
CUP = np.array([0.30, -0.15, 0.29])


class FakePolicy:
    """학생 대역 — 들어온 단계 칸 · 직전 행동 칸을 기록하고 정해진 행동을 낸다."""
    handover = {"tmin_s": 2.0, "still_s": 0.5, "still_mps": 0.03}
    policy_cfg = {"dir": BUNDLE, "source": {"spec": "student/spec.json"}}

    def __init__(self) -> None:
        self.calls, self.n_reset = [], 0

    def reset(self) -> None:
        self.n_reset += 1

    def check_inputs(self, depth, age, rgb=None):
        from policy_control.student_policy import StaleDepth

        if not age <= 0.2:
            raise StaleDepth(f"깊이 나이 {age}")

    def step(self, depth, prop, age, rgb=None):
        self.calls.append(np.asarray(prop).copy())
        return np.full(13, 0.5)


def _meas(c, palm, cup=CUP):
    s = c.side()
    m = A.RaMeas(arm_q=np.asarray(s.arm_home, float), arm_qd=np.zeros(7),
                 hand_q={j: q for j, q in zip(s.hand_joints, s.hand_open)}, palm_pos=np.asarray(palm, float),
                 palm_R=np.eye(3), tips=np.tile(palm, (5, 1)), cup_pos=np.asarray(cup, float),
                 cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5))
    return {"arm": m, "depth": (np.zeros((90, 160), np.float32), None, 0.03)}


@pytest.fixture
def chain():
    if not (BUNDLE / "student" / "spec.json").is_file():
        pytest.skip("학생 묶음 없음")
    ch = StudentChain(A.load_contract(AP05), FakePolicy())
    ch.object_name = "source240_real"
    ch.freeze = G.FreezeRule("lift_hold", 0.07, 120)
    ch.reset(_meas(ch.c, (0.2, -0.2, 0.3)))
    return ch


def test_spec_limits_are_the_sim_eval_conditions():
    if not (BUNDLE / "student" / "spec.json").is_file():
        pytest.skip("학생 묶음 없음")
    assert spec_limits(FakePolicy.policy_cfg) == (15.0, 30.0)


def test_ready_start_contract_is_refused():
    with pytest.raises(A.RhAgltError, match="홈 출발"):
        StudentChain(A.load_contract(RCG04), FakePolicy())


def test_wait_approach_stop_grasp_order_and_where_the_cup_is_used(chain):
    c, pol = chain.c, chain.policy
    hold, a = A.ready_hold_action(c), np.full(13, 0.5)
    stage_col, PREV = slice(-2, None), slice(42, 55)        # student_obs.PROP_COLS prev_action
    rows = []
    palm = np.array([0.2, -0.2, 0.3])
    # 대기 hold_steps: 실행 = (팔 0, 손 폄), 단계 approach, 디코더 비활성
    for _ in range(c.hold_steps):
        _, ex, _ = chain.step(_meas(c, palm))
        rows.append(("hold", ex))
    assert all(np.allclose(ex, hold) for _, ex in rows) and chain.phase == "approach"
    # 접근: 손바닥이 움직이면 학생 행동, T_min(2 s) 전에는 멈춰도 넘기지 않는다
    for k in range(60):
        _, ex, _ = chain.step(_meas(c, palm + [0.002 * k, 0, 0]))
        assert np.allclose(ex, a) and chain.phase == "approach"
    assert chain.dec._z0 is None                              # 접근 동안 래치 기준 높이 없음
    still = palm + [0.12, 0, 0]
    n = 0
    while chain.phase == "approach":
        _, ex, _ = chain.step(_meas(c, still))
        n += 1
        assert n < 200
    # 넘긴 틱 포함 0.5 s(30 틱) 정지 — 실행 = hold, 단계 칸 = grasp, 기준 높이는 아직 없음
    assert np.allclose(ex, hold) and chain.end is None
    for _ in range(29):
        _, ex, _ = chain.step(_meas(c, still))
        assert np.allclose(ex, hold) and chain.dec._z0 is None and not chain.grasp_started
    # 첫 잡기 틱: 학생 행동 · 이 틱 컵이 래치 · lift_hold 기준
    moved = CUP + [0.01, 0, 0]
    _, ex, _ = chain.step(_meas(c, still, cup=moved))
    assert np.allclose(ex, a) and chain.grasp_started
    assert chain.dec._z0 == pytest.approx(moved[2]) and np.allclose(chain.goals.cup_reset, moved)
    # 관측: 단계 칸 · 직전 행동 칸(실행 행동)
    assert np.allclose(pol.calls[0][stage_col], [1, 0]) and np.allclose(pol.calls[-1][stage_col], [0, 1])
    assert np.allclose(pol.calls[0][PREV], 0.0) and np.allclose(pol.calls[1][PREV], hold)
    assert np.allclose(pol.calls[-1][PREV], hold)            # 마지막 정지 틱의 실행 행동
    assert len(pol.calls) == chain.step_i                     # 정지 · 대기 틱에도 학생이 돈다


def test_approach_timeout_ends_the_episode_as_abort(chain):
    c = chain.c
    for k in range(c.hold_steps + 15 * 60 + 2):
        chain.step(_meas(c, (0.2 + 0.01 * k, -0.2, 0.3)))     # 계속 움직인다
        if chain.end is not None:
            break
    assert chain.end is not None and chain.end[0] == "abort" and "접근" in chain.end[1]


def test_missing_depth_is_stale(chain):
    from policy_control.student_policy import StaleDepth

    m = _meas(chain.c, (0.2, -0.2, 0.3))
    del m["depth"]
    with pytest.raises(StaleDepth):
        chain.step(m)


def test_stale_depth_tick_does_not_advance_the_approach_clock(chain):
    from policy_control.student_policy import StaleDepth

    c = chain.c
    for _ in range(c.hold_steps + 3):
        chain.step(_meas(c, (0.2, -0.2, 0.3)))
    n, i = chain.hand.n, chain.step_i
    m = _meas(c, (0.2, -0.2, 0.3))
    m["depth"] = (m["depth"][0], None, 0.5)
    with pytest.raises(StaleDepth):
        chain.step(m)
    assert (chain.hand.n, chain.step_i) == (n, i)


def test_goal_requested_before_grasp_is_kept_on_the_grasp_cup(chain):
    c = chain.c
    if not G.has_goal_spec(c):
        pytest.skip("계약에 목표 분포 없음")
    target = (np.asarray(c.goal_box_min, float) + np.asarray(c.goal_box_max, float)) / 2
    assert chain.goals.request(target) == []
    chain.hand.step = lambda palm: ("grasp", False, False)
    chain.step_i = c.hold_steps
    chain.step(_meas(c, (0.42, -0.2, 0.3), cup=CUP + [0.01, 0, 0]))
    assert chain.grasp_started and np.allclose(chain.goals.target, target)
    assert np.allclose(chain.goals.cup_reset, CUP + [0.01, 0, 0])


def test_nan_cup_on_first_grasp_tick_is_refused(chain):
    c = chain.c
    chain.hand.step = lambda palm: ("grasp", False, False)
    chain.step_i = c.hold_steps
    with pytest.raises(A.RhAgltError, match="유한"):
        chain.step(_meas(c, (0.42, -0.2, 0.3), cup=np.full(3, np.nan)))

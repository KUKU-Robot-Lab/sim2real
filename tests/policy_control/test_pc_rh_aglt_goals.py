"""rh_aglt 배포 목표 입력(10.01 사용자) — 학습 목표 분포(hdgp rh_aglt_env_cfg · keypoint_goal) 검사 · 중간 목표 · 달성 판정."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from policy_control import pour_fj_node as N
from policy_control import rh_aglt as A
from policy_control import rh_aglt_goals as G

POL = Path(__file__).resolve().parents[2] / "deploy" / "policies"
RIGHT, LEFT = POL / "rh56f1/aglt/right_mirror_l5" / "rh_aglt_contract.json", POL / "rh56f1/aglt/left_i05" / "rh_aglt_contract.json"
UP = np.array([1.0, 0.0, 0.0, 0.0])
GRIP = (2.0, 1.5, 0.0, 0.0, 0.0)          # 엄지 + 검지
CUP_R = np.array([0.25, -0.20, 0.264865])   # 오른팔 소환 박스 가운데, 정착고


def _right() -> A.RaContract:
    return A.load_contract(RIGHT)


def test_contracts_carry_the_training_goal_distribution_mirrored_for_the_left_arm():
    r, lft = _right(), A.load_contract(LEFT)
    assert r.goal_box_min == pytest.approx([0.10, -0.30, 0.344865]) and r.goal_box_max == pytest.approx([0.40, -0.10, 0.484865])
    assert lft.goal_box_min == pytest.approx([0.10, 0.10, 0.344865]) and lft.goal_box_max == pytest.approx([0.40, 0.30, 0.484865])
    for c in (r, lft):
        assert (c.goal_first_xy_range, c.goal_first_z_range, c.goal_delta_distance) == (0.05, [0.1, 0.18], 0.08)
        assert (c.goal_success_steps, c.goal_tol, c.grasp_threshold_n) == (10, 0.02, 1.0)
        assert G.has_goal_spec(c)


def test_goals_outside_the_box_or_not_finite_are_refused_and_change_nothing():
    b = G.GoalBook.start(_right(), CUP_R, UP)
    before = b.current.copy()
    assert "outside" in b.request([0.25, 0.05, 0.40])[0]          # 왼팔 쪽
    assert "outside" in b.request([0.25, -0.20, 0.60])[0]         # 너무 높다
    assert "finite" in b.request([0.25, float("nan"), 0.40])[0]
    assert np.allclose(b.current, before) and b.queue == [] and b.target is None


def test_a_goal_inside_the_first_region_replaces_the_first_goal():
    b = G.GoalBook.start(_right(), CUP_R, UP)
    assert b.current == pytest.approx(CUP_R + [0, 0, 0.14])
    assert b.request(CUP_R + [0.04, -0.03, 0.11]) == []
    assert b.current == pytest.approx(CUP_R + [0.04, -0.03, 0.11]) and b.queue == []


def test_a_far_first_target_keeps_the_first_goal_and_queues_waypoints_within_delta():
    c = _right()
    b = G.GoalBook.start(c, CUP_R, UP)
    target = np.array([0.38, -0.12, 0.47])
    assert b.request(target) == []
    assert b.current == pytest.approx(CUP_R + [0, 0, 0.14])
    path = [b.current, *b.queue]
    assert path[-1] == pytest.approx(target)
    assert max(np.abs(np.diff(path, axis=0)).max(axis=1)) <= c.goal_delta_distance + 1e-9
    assert all(G.in_box(c, p) for p in path)


def test_success_needs_accumulated_near_steps_now_near_and_a_grasp_then_advances():
    c = _right()
    b = G.GoalBook.start(c, CUP_R, UP)
    b.request([0.38, -0.12, 0.47])
    first, n_queue = b.current.copy(), len(b.queue)
    for _ in range(12):                                         # 근처지만 안 쥠 → 누적만
        assert not b.step(first, UP, (0, 0, 0, 0, 0))
    assert b.near_steps == 12 and b.successes == 0
    assert not b.step(first + [0, 0, 0.05], UP, GRIP)           # 쥐었지만 지금 멀다
    assert b.step(first, UP, GRIP)                               # 누적 ≥ 10 · 근처 · 쥠
    assert b.successes == 1 and np.allclose(b.anchor, first) and len(b.queue) == n_queue - 1 and b.near_steps == 0


def test_after_a_success_a_new_target_is_planned_from_the_reached_goal():
    c = _right()
    b = G.GoalBook.start(c, CUP_R, UP)
    for _ in range(10):
        b.step(b.current, UP, GRIP)
    assert b.successes == 1
    reached = b.anchor.copy()
    assert b.request(reached + [0.05, 0.0, -0.05]) == []         # delta 이내 → 곧바로
    assert b.current == pytest.approx(reached + [0.05, 0.0, -0.05]) and b.queue == []
    assert b.request(reached + [0.0, 0.09, 0.0]) == []           # delta 밖 → 둘로
    assert b.current == pytest.approx(reached + [0.0, 0.045, 0.0]) and len(b.queue) == 1


def test_the_last_goal_is_held_without_counting_again():
    b = G.GoalBook.start(_right(), CUP_R, UP)
    for _ in range(30):
        b.step(b.current, UP, GRIP)
    assert b.successes == 1


def test_first_goal_is_clipped_to_the_box_like_training():
    b = G.GoalBook.start(_right(), [0.42, -0.20, 0.264865], UP)
    assert b.current[0] == pytest.approx(0.40)


def test_old_contracts_without_the_distribution_refuse_goal_input():
    old = replace(_right(), goal_box_min=[], goal_box_max=[])
    b = G.GoalBook.start(old, CUP_R, UP)
    assert "rebuild" in b.request([0.25, -0.2, 0.40])[0]
    assert not b.step(b.current, UP, GRIP)


class _Zero:
    def forward(self, obs):
        return np.zeros(13)


def _meas(c, cup, tact=(0, 0, 0, 0, 0)) -> A.RaMeas:
    s = c.side()
    return A.RaMeas(arm_q=np.asarray(s.arm_home), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, s.hand_open)),
                    palm_pos=np.array([0.2, -0.2, 0.3]), palm_R=np.eye(3), tips=np.tile([0.25, -0.2, 0.28], (5, 1)),
                    cup_pos=np.asarray(cup, float), cup_quat=UP, tactile_n=np.asarray(tact, float))


def test_chain_takes_goals_only_after_reset_and_does_not_count_during_hold():
    c = _right()
    ch = N.AgltChain(c, _Zero())
    assert "reset first" in ch.request_goal([0.25, -0.2, 0.40])[0]
    ch.reset({"arm": _meas(c, CUP_R)})
    goal = ch.goal.pos.copy()
    for _ in range(c.hold_steps):                               # 대기 중에는 목표에 있어도 세지 않는다
        ch.step({"arm": _meas(c, goal, GRIP)})
    assert ch.goals.near_steps == 0
    for _ in range(c.goal_success_steps):
        ch.step({"arm": _meas(c, goal, GRIP)})
    assert ch.goals.successes == 1
    assert A.build_obs(c, _meas(c, goal), ch.dec, ch.goal, ch.prev).size == c.obs_dim


def test_a_deploy_grasp_signal_overrides_the_tip_rule():
    """10.04 배포: 엄지 첫마디 파지는 손끝에 안 잡혀도 관절 힘으로 쥔 것 — 노드가 is_grasped 로 넘긴다(cup_attach.grasp_signal)."""
    c = _right()
    b = G.GoalBook.start(c, CUP_R, UP)
    for _ in range(12):
        assert not b.step(b.current, UP, (0, 0, 0, 0, 0))             # 손끝 0 → 쥠 아님
    assert b.step(b.current, UP, (0, 0, 0, 0, 0), is_grasped=True)    # 관절 힘 판정이 쥠
    assert b.successes == 1


def test_the_aglt_chain_uses_joint_forces_for_the_goal_grasp_when_configured():
    from dataclasses import replace as _replace
    from policy_control import pour_fj_node as N
    from policy_control import rh_aglt as A
    from policy_control.cup_attach import AttachCfg
    c = _right()

    class _P:
        def forward(self, obs):
            return np.zeros(13)

    ch = N.AgltChain(c, _P())
    s = c.side()
    m = A.RaMeas(arm_q=np.asarray(s.arm_home), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, s.hand_open)),
                 palm_pos=np.array([0.2, -0.2, 0.3]), palm_R=np.eye(3), tips=np.tile([0.25, -0.2, 0.28], (5, 1)),
                 cup_pos=np.asarray(CUP_R, float), cup_quat=np.asarray(UP, float), tactile_n=np.zeros(5),
                 joint_force=np.array([450.0, 0.0, 400.0, 0.0, 0.0, 0.0]))
    ch.reset({"arm": m})
    goal = ch.goal.pos.copy()
    held = _replace(m, cup_pos=goal - np.array([0.0, 0.0, 0.0]))       # 목표 자리(키포인트 0 거리)에 컵
    ch.grasp_cfg = AttachCfg()
    for _ in range(c.hold_steps + 12):
        ch.step({"arm": held})
    assert ch.goals.successes >= 1


def test_reached_means_the_requested_target_itself_was_achieved():
    """10.04 에피소드 실행기: aglt 는 SETTING(사용자 목표)에 닿으면 끝난다 — 중간 목표 달성은 아직 아니다."""
    c = A.load_contract(RIGHT)
    book = G.GoalBook.start(c, [0.25, -0.20, 0.29], [1.0, 0.0, 0.0, 0.0])
    assert not book.reached
    assert book.request([0.25, -0.12, 0.41]) == []
    tact = [2.0, 2.0, 0, 0, 0]
    for _ in range(200):
        book.step(book.current, book.quat, tact)
        if book.reached:
            break
    assert book.reached and np.allclose(book.anchor, [0.25, -0.12, 0.41])

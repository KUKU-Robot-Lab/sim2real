"""joint family(팔 관절 증분 + 손 절대, fabric 없음) — right_m15_e800 계약 · 디코더 · 관측 · 체인.

09.28 사용자: 오른팔 첫 실험 정책은 right_m15_e800(cup_pick 세션 s2r 후보). 기존 세 family 는 palm 6D 를 fabric 에
넘기므로 이 정책을 거절했다. 여기 값들은 학습 커밋 d38b4346 의 hdgp 코드에서 읽은 식 · 한계와 대조한다.
시뮬레이터 재현(trace 대조)은 trace 가 생기면 따로 잠근다 — 지금 hand_obs_order 는 가정값이다.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from policy_control import joint_contract as JC
from policy_control.contract_assets import ASSETS
from policy_control.fk_numpy import UrdfChainFK
from policy_control.joint_build import assumed_obs_order, build_joint_contract
from policy_control.joint_chain import JointChain, JointMeasure, RecordedPolicy
from policy_control.joint_decoder import JointDecoder
from policy_control.joint_obs import Pose, keypoints

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
RUN = SIM2REAL / "deploy/policies/right_m15_e800"
HDGP = SIM2REAL.parent / "hdgp"
DEPLOY = "openarm_dg5f-m-short_bi_rl"

needs_run = pytest.mark.skipif(not (RUN / "params/env.yaml").is_file() or not HDGP.is_dir(),
                               reason="right_m15_e800 등록본 또는 hdgp 가 없다")


@pytest.fixture(scope="module")
def c():
    if not (RUN / "params/env.yaml").is_file() or not HDGP.is_dir():
        pytest.skip("right_m15_e800 등록본 또는 hdgp 가 없다")
    if not list((RUN / "nn").glob("*.pth")):
        pytest.skip("체크포인트가 없다(gitignore) — fetch 한 PC 에서만")
    return build_joint_contract(RUN, HDGP, DEPLOY, hand_obs_order=None, order_source="assumed: test")


@pytest.fixture(scope="module")
def fk(c):
    return UrdfChainFK(ASSETS[c.train_asset].urdf, c.arm_joints, c.hand_joints, c.palm_body, c.tip_bodies)


def test_the_contract_carries_the_trained_interface(c):
    assert (c.task, c.side, c.obs_dim, c.action_dim) == ("open-short_r_cup_pick-lstm", "right", 133, 26)
    assert c.policy_hz == pytest.approx(60.0) and c.obs_clip == 5.0 and c.action_clip == 1.0 and c.recurrent
    assert c.train_asset == "openarm_dg5f-m-short-tl_bi_rl" and c.asset == DEPLOY
    assert c.arm_reset == pytest.approx((-1.1974, 0.6707, 0.1866, 1.731, 0.692, 0.0416, 0.946))   # 에피소드 시작, 홈 아님
    assert (c.k_arm, c.arm_ema, c.hand_ema) == (0.05, 0.1, 0.1)
    assert c.welded == {"r_hj_thumb_1": 0.0} and "r_hj_thumb_1" not in c.hand_joints and len(c.hand_joints) == 19
    lim = dict(zip(c.hand_joints, zip(c.hand_lo, c.hand_hi)))
    assert lim["r_hj_thumb_2"] == pytest.approx((-1.5808, -1.5608))          # 대향 고정(프로필 override)
    assert lim["r_hj_index_1"] == pytest.approx((-0.01, 0.01)) and lim["r_hj_pinky_1"] == pytest.approx((0.0, 0.01))
    assert lim["r_hj_index_3"][0] == 0.0 and lim["r_hj_index_2"][1] == pytest.approx(2.00713, abs=1e-5)
    assert dict(zip(c.hand_joints, c.hand_reset))["r_hj_thumb_2"] == pytest.approx(-1.57)
    assert c.keypoint_half_height == pytest.approx(0.09) and c.goal_offset[2] == pytest.approx(0.24625)
    assert c.assumed_order


def test_the_assumed_hand_order_is_the_20_joint_sim_order_without_the_weld(c):
    """pour i24 trace 의 실측 관절 순서(20 관절 short)에서 thumb_1 만 뺀 것과 같아야 한다."""
    meta = SIM2REAL / "deploy/policies/both_pour_i24/trace_meta.json"
    if not meta.is_file():
        pytest.skip("i24 trace_meta 가 없다")
    names = [n for n in json.loads(meta.read_text())["joint_names"] if n.startswith("r_hj_") and n != "r_hj_thumb_1"]
    assert list(c.hand_obs_order) == names == list(assumed_obs_order(c.hand_joints, "r"))


def test_the_contract_round_trips_and_refuses_a_broken_order(c, tmp_path):
    JC.save_contract(c, tmp_path / "joint_contract.json")
    assert JC.load_contract(tmp_path / "joint_contract.json") == c
    with pytest.raises(JC.JointContractError, match="permutation"):
        JC.validate(dataclasses.replace(c, hand_obs_order=c.hand_obs_order[:-1] + ("r_hj_thumb_1",)))
    with pytest.raises(JC.JointContractError, match="dims"):
        JC.validate(dataclasses.replace(c, obs_dim=132))


def test_the_arm_integrates_on_the_previous_target_with_ema_like_hdgp(c):
    d, rng = JointDecoder(c), np.random.default_rng(1)
    q = np.asarray(c.arm_reset)
    lo, hi = np.asarray(c.arm_lo), np.asarray(c.arm_hi)
    for _ in range(50):
        a = rng.uniform(-1.5, 1.5, c.action_dim)
        out = d.step(a)
        ac = np.clip(a, -1, 1)
        q_raw = np.clip(q + 0.05 * ac[:7], lo, hi)
        q = np.clip(0.1 * q_raw + 0.9 * q, lo, hi)
        assert np.allclose(out.arm, q, atol=1e-12) and np.all(np.abs(out.action) <= 1.0)
    assert np.max(np.abs(np.diff([JointDecoder(c).step(np.ones(26)).arm, c.arm_reset], axis=0))) <= 0.005 + 1e-12


def test_the_hand_is_an_ema_toward_the_absolute_target_and_normalizes_back(c):
    d = JointDecoder(c)
    assert d.hand_action_obs()[c.hand_joints.index("r_hj_thumb_2")] == pytest.approx(0.08, abs=1e-6)   # 리셋 관측
    assert d.hand_action_obs()[c.hand_joints.index("r_hj_index_2")] == pytest.approx(-1.0)
    lo, hi = np.asarray(c.hand_lo), np.asarray(c.hand_hi)
    for _ in range(200):
        out = d.step(np.ones(26))
    assert np.allclose(out.hand, hi, atol=1e-6)                                    # a = +1 은 상한으로 수렴
    assert np.allclose(d.hand_action_obs(), 1.0, atol=1e-5)
    d.reset()
    first = d.step(np.concatenate([np.zeros(7), -np.ones(19)])).hand
    assert np.allclose(first, np.clip(0.1 * lo + 0.9 * np.asarray(c.hand_reset), lo, hi))


def test_keypoints_are_axial_offsets_in_the_object_frame(c):
    upright = keypoints(c, Pose(np.array([0.3, -0.2, 0.25]), np.array([1.0, 0, 0, 0])))
    assert np.allclose(upright[:, :2], [0.3, -0.2]) and np.allclose(upright[:, 2], 0.25 + np.array([0.09, -0.09, 0.03, -0.03]))
    tipped = keypoints(c, Pose(np.zeros(3), np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])))   # x 로 90°
    assert np.allclose(tipped[0], [0.0, -0.09, 0.0], atol=1e-12)


def _measure(c, arm, hand, obj):
    pos = {**dict(zip(c.arm_joints, arm)), **dict(zip(c.hand_joints, hand)), "r_hj_thumb_1": 0.0}
    return JointMeasure(pos, {k: 0.0 for k in pos}, obj)


def test_the_chain_builds_the_133_layout_from_fk_and_decoder_state(c, fk):
    cup = Pose(np.array([0.35, -0.15, 0.25]), np.array([1.0, 0, 0, 0]))
    chain = JointChain(c, RecordedPolicy(np.zeros((3, 26))), fk)
    goal = chain.reset(cup)
    assert np.allclose(goal.pos, np.clip(cup.pos + [0, 0, 0.24625], c.goal_box_lo, c.goal_box_hi))
    step = chain.step(_measure(c, c.arm_reset, c.hand_reset, cup))
    o = step.obs
    assert o.shape == (133,) and np.all(np.abs(o) <= 5.0)
    assert np.allclose(o[0:7], c.arm_reset) and np.allclose(o[76:83], c.arm_reset)        # arm_q · cmd_state
    order = [c.hand_joints.index(n) for n in c.hand_obs_order]
    assert np.allclose(o[14:33], np.asarray(c.hand_reset)[order])                       # hand_q 는 시뮬 순서
    pose = fk.palm_pose(c.arm_reset, c.hand_reset)
    assert np.allclose(o[52:55], pose.palm_pos) and np.allclose(o[55:61], np.r_[pose.extra["palm_rot"][:, 0],
                                                                               pose.extra["palm_rot"][:, 1]])
    assert np.allclose(o[61:76], (pose.tips - pose.palm_pos).reshape(-1))
    assert np.allclose(o[95:107].reshape(4, 3), np.array([[0, 0, -0.24625]] * 4), atol=1e-9)   # 물체 − 목표
    assert np.allclose(o[107:114], 0.0)                                                   # 리셋 직후 행동 0


@needs_run
def test_the_lstm_checkpoint_loads_on_cpu_and_answers_26_finite_actions(c):
    torch = pytest.importorskip("torch")
    del torch
    from policy_control.joint_policy import JointPolicy

    try:
        pol = JointPolicy(c, device="cpu")
    except ImportError as exc:
        pytest.skip(f"rl_games 가 없다: {exc}")
    obs = np.zeros(133, dtype=np.float32)
    a1 = pol.forward(obs)
    a2 = pol.forward(obs)
    pol.reset()
    a3 = pol.forward(obs)
    assert a1.shape == (26,) and np.all(np.isfinite(a1))
    assert np.allclose(a1, a3, atol=1e-6) and not np.allclose(a1, a2, atol=1e-7)          # 은닉 상태가 돈다


def test_the_episode_ends_like_training_when_the_goal_is_held_or_time_runs_out():
    """09.28 사용자: 목표에 이송하면 에피소드가 끝나고, 홈으로 돌아와 다시 돌린다 — 학습 규칙(연속 도달 · 에피소드 길이)."""
    from policy_control.joint_node_core import EpisodeEnd

    e = EpisodeEnd(tol=0.0318, steps=10, max_s=15.0)
    e.reset(100.0)
    for k in range(9):
        assert e.update(0.02, 100.0 + k / 60) is None
    assert e.update(0.05, 100.2) is None                 # 한 번 벗어나면 연속이 끊긴다(force_consecutive)
    for k in range(9):
        assert e.update(0.03, 100.3 + k / 60) is None
    assert "goal reached" in e.update(0.03, 100.5)
    e.reset(200.0)
    assert e.update(None, 214.9) is None and "episode time" in e.update(0.5, 215.0)
    off = EpisodeEnd()
    off.reset(0.0)
    assert off.update(0.0, 1e6) is None                   # 0 = 끔(오른팔 등 값을 넘기지 않은 정책)


def test_keypoint_goal_distance_is_the_max_over_keypoints(c):
    from policy_control.joint_node_core import keypoint_goal_dist
    from policy_control.joint_obs import Pose

    up = np.array([1.0, 0.0, 0.0, 0.0])
    a, b = Pose(np.zeros(3), up), Pose(np.array([0.03, 0.0, 0.04]), up)
    assert keypoint_goal_dist(c, a, b) == pytest.approx(0.05)            # 같은 자세면 모든 키포인트가 같이 옮겨진다

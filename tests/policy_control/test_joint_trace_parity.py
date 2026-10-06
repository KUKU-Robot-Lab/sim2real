"""joint family 재현 — 학습 env(Isaac)에서 뽑은 trace 로 배포 코드의 관측 · 디코더 · 정책을 대조한다.

trace 는 deploy/policy_control/tools/isaac_joint_trace.py 가 만든다(학습 전용 관측 노이즈 · 지연을 끈 env). 그래서
env 관측 = 배포가 만드는 깨끗한 관측이어야 한다. 대조하는 것:
  ① meta 의 관절 순서 · 한계 · 시작 자세 = 계약        (손 관측 순서가 가정이 아니라 실측인가)
  ② 매 스텝 관측 133 = 기록된 상태로 배포 코드가 다시 만든 관측
  ③ 디코더가 기록된 행동으로 만든 q* = env 의 팔 · 손 q*
  ④ 체크포인트가 기록된 관측에 같은 행동을 낸다(CPU vs env GPU)
trace 가 없으면 skip — 저장소에 넣지 않는다(gitignore *.npz).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from policy_control.joint_contract import load_contract
from policy_control.joint_decoder import JointDecoder
from policy_control.joint_node import build_fk
from policy_control.joint_obs import JointState, Pose, build_obs

pytestmark = pytest.mark.unit

RUN = Path(__file__).resolve().parents[2] / "deploy/policies/dg5f_m/cup_pick/right_m15"


@pytest.fixture(scope="module")
def data():
    if not (RUN / "trace.npz").is_file() or not (RUN / "joint_contract.json").is_file():
        pytest.skip("trace 가 없다 — isaac_joint_trace.py 로 만든다")
    z = np.load(RUN / "trace.npz")
    meta = json.loads((RUN / "trace_meta.json").read_text())
    return load_contract(RUN / "joint_contract.json"), {k: z[k][:, 0] for k in z.files}, meta


def test_the_trace_meta_agrees_with_the_contract(data):
    c, _, m = data
    assert c.hand_obs_order_source.startswith("measured")
    assert tuple(m["hand_obs_order"]) == c.hand_obs_order
    assert tuple(m["hand_action_order"]) == c.hand_joints and tuple(m["arm_joints"]) == c.arm_joints
    assert np.allclose(m["act_lo"], c.hand_lo, atol=1e-5) and np.allclose(m["act_hi"], c.hand_hi, atol=1e-5)
    assert np.allclose(m["arm_lo"], c.arm_lo, atol=1e-4) and np.allclose(m["arm_hi"], c.arm_hi, atol=1e-4)
    assert np.allclose(m["hand_reset_q"], c.hand_reset, atol=1e-6)
    assert (m["k_arm"], m["arm_ema"], m["hand_ema"]) == (c.k_arm, c.arm_ema, c.hand_ema)
    assert m["palm_body"] == c.palm_body and tuple(m["tip_bodies"]) == c.tip_bodies


def _state(c, d, m, fk, t):
    names = m["joint_names"]
    pos, vel = dict(zip(names, d["q"][t])), dict(zip(names, d["qd"][t]))
    pose = fk.palm_pose([pos[n] for n in c.arm_joints], [pos[n] for n in c.hand_joints])
    return JointState(pos, vel, pose.palm_pos, pose.extra["palm_rot"], pose.tips,
                      Pose(d["obj_pos"][t], d["obj_quat"][t]), Pose(d["goal_pos"][t], d["goal_quat"][t])), pose


def test_fk_matches_the_simulator_palm_and_tips(data):
    c, d, m = data
    fk = build_fk(c)
    for t in range(len(d["obs"])):
        _, pose = _state(c, d, m, fk, t)
        assert np.allclose(pose.palm_pos, d["palm_pos"][t], atol=2e-4), t
        assert np.allclose(pose.tips, d["tips"][t], atol=5e-4), t


def test_every_observation_is_rebuilt_from_the_recorded_state(data):
    c, d, m = data
    fk = build_fk(c)
    lo, hi = np.asarray(c.hand_lo), np.asarray(c.hand_hi)
    worst = 0.0
    for t in range(len(d["obs"])):
        s, _ = _state(c, d, m, fk, t)
        prev = np.zeros(c.n_arm) if d["ep_len"][t] == 0 else np.clip(d["action"][t - 1][:c.n_arm], -1, 1)
        hand_obs = 2.0 * (d["hand_qstar"][t] - lo) / (hi - lo) - 1.0
        obs = build_obs(c, s, d["arm_qstar"][t], prev, hand_obs)
        err = np.abs(obs - d["obs"][t])
        worst = max(worst, float(err.max()))
        assert err.max() < 2e-3, (t, int(err.argmax()), float(err.max()))
    print(f"obs 재현 최대 오차 {worst:.2e}")


def test_the_decoder_reproduces_the_simulator_targets(data):
    c, d, _ = data
    dec = JointDecoder(c)
    assert np.allclose(dec.arm, d["arm_qstar"][0], atol=1e-5) and np.allclose(dec.hand, d["hand_qstar"][0], atol=1e-5)
    for t in range(len(d["action"]) - 1):
        if d["ep_len"][t + 1] == 0:                 # 에피소드 경계 — env 는 시작 자세로 리셋, 디코더도 리셋
            dec.reset()
            continue
        out = dec.step(d["action"][t])
        assert np.allclose(out.arm, d["arm_qstar"][t + 1], atol=1e-5), t
        assert np.allclose(out.hand, d["hand_qstar"][t + 1], atol=1e-5), t


def test_the_checkpoint_answers_the_recorded_actions(data):
    pytest.importorskip("torch")
    from policy_control.joint_policy import JointPolicy

    c, d, _ = data
    pol = JointPolicy(c, device="cpu")
    pol.reset()
    worst = 0.0
    for t in range(len(d["obs"])):
        if t > 0 and d["ep_len"][t] == 0:
            pol.reset()                              # 학습의 zero_rnn_on_done
        a = pol.forward(np.clip(d["obs"][t], -c.obs_clip, c.obs_clip))
        worst = max(worst, float(np.abs(a - d["action"][t]).max()))
    print(f"actor 재현 최대 오차 {worst:.2e}")
    assert worst < 1e-3

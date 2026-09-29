"""체크포인트 (obs, action) 차원 읽기 — tests/test_robot_profile.py 에서 옮겼다(09.29)."""
from __future__ import annotations

# ── RNN 체크포인트의 관측 차원 ─────────────────────────────────────────
# fab_test42(vision-3090) 실측에서 드러났다: actor_mlp 입력 1096 = LSTM 1024 + obs 72.
# 첫 층을 obs 로 읽던 구현은 RNN 정책에 대해 조용히 틀린 값을 냈다.

def test_a_recurrent_checkpoints_obs_comes_from_the_normalizer_not_the_mlp():
    import torch

    from policy_control.checkpoint_dims import _obs_dim_from_state

    state = {
        "a2c_network.actor_mlp.0.weight": torch.zeros(512, 1096),
        "a2c_network.rnn.rnn.weight_ih_l0": torch.zeros(4096, 72),
        "a2c_network.rnn.rnn.weight_hh_l0": torch.zeros(4096, 1024),
        "running_mean_std.running_mean": torch.zeros(72),
    }
    assert _obs_dim_from_state(state, "t") == 72


def test_a_recurrent_checkpoint_without_a_normalizer_falls_back_to_the_rnn_input():
    import torch

    from policy_control.checkpoint_dims import _obs_dim_from_state

    state = {
        "a2c_network.actor_mlp.0.weight": torch.zeros(512, 1096),
        "a2c_network.rnn.rnn.weight_ih_l0": torch.zeros(4096, 72),
    }
    assert _obs_dim_from_state(state, "t") == 72


def test_a_plain_mlp_checkpoint_still_reads_the_first_layer():
    import torch

    from policy_control.checkpoint_dims import _obs_dim_from_state

    state = {"a2c_network.actor_mlp.0.weight": torch.zeros(256, 36)}
    assert _obs_dim_from_state(state, "t") == 36


def test_a_recurrent_checkpoint_with_no_readable_obs_refuses_the_mlp_guess():
    import pytest
    import torch

    from policy_control.checkpoint_dims import _obs_dim_from_state

    state = {
        "a2c_network.actor_mlp.0.weight": torch.zeros(512, 1096),
        "a2c_network.rnn.rnn.weight_hh_l0": torch.zeros(4096, 1024),
    }
    with pytest.raises(KeyError, match="concat_input"):
        _obs_dim_from_state(state, "t")


def test_recurrence_is_detectable_because_deployment_must_carry_hidden_state():
    from policy_control.checkpoint_dims import checkpoint_is_recurrent

    assert checkpoint_is_recurrent({"a2c_network.rnn.rnn.weight_hh_l0": 1})
    assert not checkpoint_is_recurrent({"a2c_network.actor_mlp.0.weight": 1})

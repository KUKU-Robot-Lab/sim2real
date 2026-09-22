"""Pour family: contract is derived from the run dump, obs_clip is applied, and the i11 actor reproduces the trace."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from policy_control.pour_contract import PourContractError, from_dict, to_dict, validate
from policy_control.pour_obs import clip_obs
from pour_trace_util import contract, trace

SIM2REAL = Path(__file__).resolve().parents[2]
FIX = SIM2REAL / "tests/fixtures/policy_control/pour_i11"
CKPT = SIM2REAL.parent / "hdgp/log/server_mirror/pour-fab/t2r_i11/nn/open-short_b_pour_fab.pth"
RESET_ROW = 897  # obs_next[897] is pre-reset, actions[898] is post-reset


def test_contract_values_come_from_the_run_dump():
    c = contract()
    assert (c.obs_dim, c.action_dim, c.mlp_units, c.normalize_input) == (223, 18, (512, 256, 128), True)
    assert c.obs_clip == 5.0 and c.action_clip == 1.0 and c.palm_ema_alpha == 0.25 and c.hold_steps == 45
    assert [(s.role, s.side) for s in c.sides] == [("src", "right"), ("rcv", "left")]


def test_contract_json_round_trip_is_lossless():
    c = contract()
    assert from_dict(to_dict(c)) == c


def test_validate_rejects_inconsistent_dims():
    with pytest.raises(PourContractError):
        validate(dataclasses.replace(contract(), obs_dim=222))


def test_clip_obs_applies_training_wrapper_clip():
    c = contract()
    obs = np.linspace(-9.0, 9.0, c.obs_dim)
    out = clip_obs(c, obs)
    assert out.min() == -c.obs_clip and out.max() == c.obs_clip
    assert obs.max() == 9.0  # input untouched


@pytest.mark.skipif(not CKPT.exists(), reason="i11 checkpoint not mirrored on this host")
def test_actor_reproduces_trace_actions_from_trace_obs():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(SIM2REAL / "scripts"))
    from policy_loader import RLGamesActorPolicy

    c = contract()
    z, _ = trace()
    policy = RLGamesActorPolicy(str(FIX / "params/agent.yaml"), str(CKPT), obs_dim=c.obs_dim,
                                action_dim=c.action_dim, device="cpu", action_clip=None)
    obs = clip_obs(c, z["obs_next"][:RESET_ROW, 0])
    mu = policy.get_action(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()
    err = np.abs(np.clip(mu, -1, 1) - np.clip(z["actions"][1:RESET_ROW + 1, 0], -1, 1)).max()
    assert err < 1e-5

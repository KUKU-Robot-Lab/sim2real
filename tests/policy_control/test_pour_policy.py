"""pour policy loader: PourContract + run dump + checkpoint -> mu(obs). Uses a synthetic (random-weight) checkpoint."""
import sys
from pathlib import Path

import numpy as np
import pytest

from policy_control.pour_build import build_pour_contract
from policy_control.pour_contract import PourContractError
from policy_control.pour_policy import PourPolicy

from pour_trace_util import ASSET, FIX, HDGP, trace

torch = pytest.importorskip("torch")
pytest.importorskip("rl_games")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _fake_ckpt(path):
    import yaml
    from rl_games.algos_torch.model_builder import ModelBuilder
    params = yaml.safe_load((FIX / "params" / "agent.yaml").read_text())["params"]
    net = ModelBuilder().load(params).build({"actions_num": 18, "input_shape": (223,), "num_seqs": 1,
                                             "value_size": 1, "normalize_value": True, "normalize_input": True})
    torch.save({"model": net.state_dict()}, path)
    return path


def _contract(ckpt):
    return build_pour_contract(FIX, FIX / "trace_meta.json", HDGP, ASSET.urdf, ASSET.name, checkpoint=ckpt)


def test_loads_and_maps_trace_obs_to_finite_action(tmp_path):
    ckpt = _fake_ckpt(tmp_path / "fake.pth")
    pol = PourPolicy(_contract(ckpt), device="cpu")
    z, _ = trace()
    a = pol.forward(z["obs_next"][5, 0])
    assert a.shape == (18,) and np.all(np.isfinite(a))
    assert np.allclose(a, pol.forward(z["obs_next"][5, 0]))          # deterministic mu
    with pytest.raises(PourContractError):
        pol.forward(np.zeros(222))


def test_refuses_missing_or_changed_checkpoint(tmp_path):
    with pytest.raises(PourContractError, match="no checkpoint"):
        PourPolicy(_contract(None), device="cpu")
    ckpt = _fake_ckpt(tmp_path / "fake.pth")
    c = _contract(ckpt)
    ckpt.write_bytes(ckpt.read_bytes() + b"x")
    with pytest.raises(PourContractError, match="md5"):
        PourPolicy(c, device="cpu")

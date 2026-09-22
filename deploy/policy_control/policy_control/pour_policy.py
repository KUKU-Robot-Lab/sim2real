"""rl_games actor for the pour family: PourContract (run dump + checkpoint, hash-checked) -> mu(obs).

No action clip here (the decoder clips; prev_actions must carry the raw mu, as in training).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _paths  # noqa: F401  (puts scripts/ on sys.path for policy_loader)
from .pour_contract import PourContract, PourContractError, file_md5, file_sha1


class PourPolicy:
    def __init__(self, contract: PourContract, device: str = "cuda:0") -> None:
        if not contract.checkpoint:
            raise PourContractError("contract has no checkpoint (control/obs-only build)")
        ckpt, agent = Path(contract.checkpoint), Path(contract.run_dir) / "params" / "agent.yaml"
        for p in (ckpt, agent):
            if not p.is_file():
                raise PourContractError(f"missing: {p}")
        if file_md5(ckpt) != contract.checkpoint_md5:
            raise PourContractError(f"checkpoint md5 changed since the contract was built: {ckpt}")
        if file_sha1(agent) != contract.agent_yaml_sha1:
            raise PourContractError(f"agent.yaml changed since the contract was built: {agent}")
        import torch
        from policy_loader import RLGamesActorPolicy
        self._torch, self._device, self._dim = torch, device, int(contract.obs_dim)
        self._policy = RLGamesActorPolicy(str(agent), str(ckpt), obs_dim=contract.obs_dim,
                                          action_dim=contract.action_dim, device=device, action_clip=None)
        if bool(self._policy.model.is_rnn()):
            raise PourContractError("pour family expects an MLP actor; checkpoint network is recurrent")

    def forward(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.size != self._dim or not np.all(np.isfinite(arr)):
            raise PourContractError(f"obs must be {self._dim} finite values, got {arr.size}")
        t = self._torch.as_tensor(arr, device=self._device).unsqueeze(0)
        return self._policy.get_action(t)[0].detach().cpu().numpy().astype(np.float64)

    def reset(self) -> None:
        return None

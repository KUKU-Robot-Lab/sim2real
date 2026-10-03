"""rl_games actor for the joint family: JointContract (run dump + checkpoint, hash-checked) -> mu(obs).

LSTM 이면 은닉 상태를 스텝 사이에 들고, `reset()`(에피소드 시작)에 0 으로 — 학습의 zero_rnn_on_done 과 같다.
행동 clip 은 디코더가 한다(여기서 자르면 관측 action_arm 칸과 어긋날 일은 없지만 한 곳에서만 자른다).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _paths  # noqa: F401  (puts scripts/ on sys.path for policy_loader)
from .joint_contract import JointContract, JointContractError, file_md5, file_sha1


class JointPolicy:
    def __init__(self, contract: JointContract, device: str = "cuda:0") -> None:
        ckpt, agent = Path(contract.checkpoint), Path(contract.run_dir) / "params" / "agent.yaml"
        for p in (ckpt, agent):
            if not p.is_file():
                raise JointContractError(f"missing: {p}")
        if file_md5(ckpt) != contract.checkpoint_md5:
            raise JointContractError(f"checkpoint md5 changed since the contract was built: {ckpt}")
        if file_sha1(agent) != contract.agent_yaml_sha1:
            raise JointContractError(f"agent.yaml changed since the contract was built: {agent}")
        import torch
        from policy_loader import RLGamesActorPolicy, RLGamesLstmActorPolicy
        if str(device).startswith("cpu"):
            # 기본 스레드 수(코어 전부)는 같은 PC 의 학습 · 다른 노드와 겨뤄 한 스텝이 47 ms 까지 늘었다(09.28 fake,
            # 혼자 재면 4 ms). 60 Hz 예산 16.7 ms 안에 두려고 줄인다. 실기는 cuda 로 돈다(2.4 ms).
            from .cpu_plan import current_plan   # POLICY_CPU_THREADS 가 있으면 그 값, 없으면 2(코어가 적으면 1)
            torch.set_num_threads(current_plan().torch_threads)
        cls = RLGamesLstmActorPolicy if contract.recurrent else RLGamesActorPolicy
        self._torch, self._device, self._dim = torch, device, int(contract.obs_dim)
        self._policy = cls(str(agent), str(ckpt), obs_dim=contract.obs_dim, action_dim=contract.action_dim,
                           device=device, action_clip=None)
        if bool(self._policy.model.is_rnn()) != bool(contract.recurrent):
            raise JointContractError("checkpoint network recurrence differs from the contract")
        missing = list(getattr(self._policy, "load_report", ((), ()))[0])
        if missing:                                     # 빠진 가중치 = 초기값으로 도는 정책 — 절대 통과시키지 않는다
            raise JointContractError(f"checkpoint did not fill {len(missing)} model weights: {missing[:5]}")

    def forward(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.size != self._dim or not np.all(np.isfinite(arr)):
            raise JointContractError(f"obs must be {self._dim} finite values, got {arr.size}")
        t = self._torch.as_tensor(arr, device=self._device).unsqueeze(0)
        with self._torch.no_grad():
            return self._policy.get_action(t)[0].detach().cpu().numpy().astype(np.float64)

    def reset(self) -> None:
        if hasattr(self._policy, "reset_states"):
            self._policy.reset_states()

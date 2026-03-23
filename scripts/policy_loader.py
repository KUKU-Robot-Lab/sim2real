#!/usr/bin/env python3
"""rl_games actor policy 로더 (Isaac Sim 의존성 없음).

5g_grasp_right_v7 체크포인트에서 actor network만 로드하여
실환경 추론에 사용합니다.

사용 예시:
    python3 policy_loader.py \
        --agent  /path/to/agent.yaml \
        --ckpt   /path/to/5g_grasp_right-v7.pth

    >>> policy = RLGamesActorPolicy(agent_yaml, ckpt_path)
    >>> obs = torch.randn(1, 106).cuda()
    >>> action = policy.get_action(obs)   # (1, 11)
"""

from __future__ import annotations

import argparse
import yaml
import torch

from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch import torch_ext


# ---------------------------------------------------------------------------
# 체크포인트 state_dict 키 정규화 (compile 여부 무관하게 로드 가능)
# ---------------------------------------------------------------------------

def _adjust_state_dict_keys(ckpt_sd: dict, model_sd: dict) -> dict:
    """ckpt state_dict 키를 model state_dict 키에 맞게 조정합니다."""
    adjusted = {}
    for key, value in ckpt_sd.items():
        if key in model_sd:
            adjusted[key] = value
            continue
        # torch.compile 시 '_orig_mod' 삽입
        parts = key.split(".")
        parts.insert(2, "_orig_mod")
        candidate = ".".join(parts)
        if candidate in model_sd:
            adjusted[candidate] = value
            continue
        # '_orig_mod' 제거 시도
        stripped = key.replace("_orig_mod.", "")
        if stripped in model_sd:
            adjusted[stripped] = value
            continue
        # 매칭 실패 → 원본 키 유지 (load_state_dict strict=False 대비)
        adjusted[key] = value
    return adjusted


# ---------------------------------------------------------------------------
# Policy 클래스
# ---------------------------------------------------------------------------

class RLGamesActorPolicy:
    """rl_games MLP actor policy (non-RNN, continuous action).

    Args:
        agent_yaml_path : agent.yaml 경로
        checkpoint_path : .pth 체크포인트 경로
        obs_dim         : actor 관측 차원 (v7 = 106)
        action_dim      : 액션 차원 (v7 = 11)
        device          : 'cuda:0' 또는 'cpu'
    """

    def __init__(
        self,
        agent_yaml_path: str,
        checkpoint_path: str,
        obs_dim: int = 106,
        action_dim: int = 11,
        device: str = "cuda:0",
    ) -> None:
        self.device = device

        # ---- yaml 파싱 ----
        with open(agent_yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        params = cfg["params"]

        normalize_input = params["config"].get("normalize_input", True)
        normalize_value = params["config"].get("normalize_value", True)

        # ---- 모델 빌드 ----
        model_config = {
            "actions_num":      action_dim,
            "input_shape":      (obs_dim,),
            "num_seqs":         1,
            "value_size":       1,
            "normalize_value":  normalize_value,
            "normalize_input":  normalize_input,
        }
        builder = ModelBuilder()
        network = builder.load(params)
        self.model = network.build(model_config).to(device)
        self.model.eval()

        # ---- 체크포인트 로드 ----
        weights = torch_ext.load_checkpoint(checkpoint_path)
        adjusted = _adjust_state_dict_keys(weights["model"], self.model.state_dict())
        self.model.load_state_dict(adjusted, strict=False)

        # normalize_input=True 일 때 running_mean_std 복원
        # (모델 state_dict 안에 포함되어 있으면 이미 위에서 로드됨.
        #  별도 키로 저장된 경우를 위한 fallback)
        if normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

        # dummy forward → warmup (CUDA kernel 초기화)
        dummy = torch.zeros(1, obs_dim, device=device)
        with torch.inference_mode():
            self._forward(dummy)

        print(f"[policy_loader] Loaded: {checkpoint_path}")
        print(f"[policy_loader] obs_dim={obs_dim}, action_dim={action_dim}, device={device}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _forward(self, obs: torch.Tensor) -> torch.Tensor:
        """model forward → deterministic action (mu)."""
        batch_dict = {
            "is_train": False,
            "obs":      obs,
        }
        res = self.model(batch_dict)
        return res["mus"]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """관측값으로부터 결정론적 action을 반환합니다.

        Args:
            obs: (1, obs_dim) float32 tensor, 임의 장치 가능
                 (내부에서 self.device로 이동)

        Returns:
            (1, action_dim) float32, ∈ [-1, 1]
        """
        obs = obs.to(self.device, dtype=torch.float32)
        mu = self._forward(obs)
        return mu.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# 단독 테스트
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RLGamesActorPolicy 단독 테스트")
    parser.add_argument("--agent", required=True, help="agent.yaml 경로")
    parser.add_argument("--ckpt",  required=True, help=".pth 체크포인트 경로")
    parser.add_argument("--obs_dim",    type=int, default=106)
    parser.add_argument("--action_dim", type=int, default=11)
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--n",          type=int, default=5, help="반복 횟수")
    args = parser.parse_args()

    policy = RLGamesActorPolicy(
        agent_yaml_path=args.agent,
        checkpoint_path=args.ckpt,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        device=args.device,
    )

    import time
    obs = torch.zeros(1, args.obs_dim, device=args.device)
    for i in range(args.n):
        t0 = time.perf_counter()
        action = policy.get_action(obs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"[{i}] action={action.cpu().tolist()}  ({dt:.2f} ms)")


if __name__ == "__main__":
    main()

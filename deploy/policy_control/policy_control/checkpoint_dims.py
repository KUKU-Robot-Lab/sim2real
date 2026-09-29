"""rl_games 체크포인트가 실제로 학습된 (obs, action) 차원 — 계약 빌드가 run 덤프와 대조한다.

09.29: scripts/robot_profile.py(옛 구성 프로필, hdgp 레거시 태스크 삭제로 지움)에서 그대로 옮겼다.
"""
from __future__ import annotations

from pathlib import Path

_ACTOR_FIRST_LAYER = "a2c_network.actor_mlp.0.weight"
_MU_HEAD = "a2c_network.mu.weight"
#: obs 정규화 통계. 길이 = **원 관측 차원**. rl_games 는 정규화를 raw obs 에 걸므로
#: 망 구조(RNN 유무·concat_input)와 무관하게 이것이 가장 곧은 진실원천이다.
_OBS_NORM = "running_mean_std.running_mean"
_RNN_INPUT = "a2c_network.rnn.rnn.weight_ih_l0"
_RNN_HIDDEN = "a2c_network.rnn.rnn.weight_hh_l0"


def checkpoint_is_recurrent(state: dict) -> bool:
    return any(".rnn." in k for k in state)


def _obs_dim_from_state(state: dict, where: str) -> int:
    """관측 차원. **`actor_mlp` 첫 층을 그냥 믿으면 안 된다.**

    rl_games 의 RNN 망은 `concat_input: true` 일 때 actor_mlp 입력이
    `rnn_hidden + obs` 다. fab_test42 실측: actor_mlp 입력 1096 = LSTM 1024 + obs **72**.
    1096 을 obs 로 읽으면 계약 검증이 통과할 리 없는 값으로 조용히 어긋난다.
    """
    if _OBS_NORM in state:
        return int(state[_OBS_NORM].reshape(-1).shape[0])
    if _RNN_INPUT in state:
        # RNN 이 mlp 앞에 있으면 rnn 입력이 곧 obs 다.
        return int(state[_RNN_INPUT].shape[1])
    if _ACTOR_FIRST_LAYER in state:
        if checkpoint_is_recurrent(state):
            raise KeyError(
                f"{where}: RNN 체크포인트인데 obs 차원을 곧게 읽을 텐서가 없다"
                f"({_OBS_NORM} · {_RNN_INPUT} 둘 다 없음). {_ACTOR_FIRST_LAYER} 는 "
                "concat_input 때문에 obs 가 아닐 수 있어 쓰지 않는다."
            )
        return int(state[_ACTOR_FIRST_LAYER].shape[1])
    raise KeyError(f"{where}: obs 차원을 읽을 텐서가 없다")


def checkpoint_contract(checkpoint: Path) -> tuple[int, int]:
    """rl_games 체크포인트가 **실제로** 학습된 (obs, action) 차원.

    짐작한 차원으로 정책을 로드하면 shape mismatch 로 죽거나(운이 좋은 경우)
    잘못된 obs 로 조용히 돈다.
    """
    import torch

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    if _MU_HEAD not in state:
        raise KeyError(
            f"{checkpoint}: 액션 차원을 읽을 {_MU_HEAD} 가 없다\n"
            f"  (rl_games actor 체크포인트가 맞는지 확인할 것)"
        )
    return _obs_dim_from_state(state, str(checkpoint)), int(state[_MU_HEAD].shape[0])

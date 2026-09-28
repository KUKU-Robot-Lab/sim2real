"""palm 지령 스텝당 변화량 상한(`palm_cmd_max_step`) — hdgp i19 부터 학습에 들어간 slew.

09.28 i24 등록 때 계약 빌더가 env.yaml 의 `palm_cmd_max_step: 0.03` 을 읽지 않고 통과시켰고,
디코더는 EMA 만 적용했다. 그 결과 trace 대비 palm 지령 오차 1.54(정규화 단위)로 재현 테스트가 떨어졌다.
hdgp `pour_rules.slew_limit`: cmd = prev + clamp(ema - prev, ±max_step), max_step <= 0 이면 끔.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from policy_control import pour_contract as PC
from policy_control.pour_decoder import PALM_DIM, PourDecoder, SideInputs
from pour_trace_util import FIX, contract

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def base():
    return contract(FIX)


def _inputs(c):
    return {r: SideInputs(1.0, np.zeros(len(c.side(r).fingers)), np.zeros(len(c.side(r).fingers))) for r in c.roles}


def test_old_contracts_without_the_key_load_as_no_limit(base, tmp_path):
    raw = PC.to_dict(base)
    raw.pop("palm_cmd_max_step", None)                  # i18 이전에 만든 계약 파일
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw))
    assert PC.load_contract(path).palm_cmd_max_step == 0.0


def test_a_negative_limit_is_refused(base):
    with pytest.raises(PC.PourContractError, match="palm_cmd_max_step"):
        PC.validate(dataclasses.replace(base, palm_cmd_max_step=-0.01))


def test_the_palm_command_moves_at_most_max_step_per_step_like_hdgp(base):
    c = dataclasses.replace(base, palm_cmd_max_step=0.03)
    dec, ref = PourDecoder(c), {r: np.zeros(PALM_DIM) for r in c.roles}
    rng = np.random.default_rng(0)
    for _ in range(40):
        a = rng.uniform(-1, 1, c.action_dim)
        out = dec.step(a, _inputs(c))
        for i, r in enumerate(c.roles):
            b = i * (len(a) // len(c.roles))
            ema = c.palm_ema_alpha * np.clip(a[b:b + PALM_DIM], -1, 1) + (1 - c.palm_ema_alpha) * ref[r]
            want = ref[r] + np.clip(ema - ref[r], -0.03, 0.03)
            assert np.allclose(out[r].palm_cmd, want, atol=1e-12)
            assert np.all(np.abs(out[r].palm_cmd - ref[r]) <= 0.03 + 1e-12)
            ref[r] = want


def test_zero_means_plain_ema(base):
    c = dataclasses.replace(base, palm_cmd_max_step=0.0)
    a = np.ones(c.action_dim)
    out = PourDecoder(c).step(a, _inputs(c))
    for r in c.roles:
        assert np.allclose(out[r].palm_cmd, c.palm_ema_alpha * np.ones(PALM_DIM))

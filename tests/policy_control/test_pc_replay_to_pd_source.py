"""replay_to_pd 는 자세 이동 궤적(reset_*.npz)도 읽어야 한다 (ROS 없음).

2026-09-07 결정: 자세 이동(차렷↔preset)을 **pd 가 직접 소유**한다. 지금까지는 옛 스택
(`gravity_comp_node` + `shadow_replay` + JTC)으로 옮기고 pd 에 인계했는데, 인계 구간이
무보상이라 8° 내려앉고, pd 는 engage 때 **실측을 시드로 잡아** 그 처짐이 새 목표가 됐다.
pd 자신의 `model_tau_ff` 가 검증모델과 일치(0.99~1.01)하게 고쳐졌으므로 두 시스템을 쓸
이유가 없다 — 차렷에서 engage 하고 궤적을 pd 로 흘리면 인계 자체가 없어진다.

`--npz` 가 `fabric_q` 키를 하드코딩해 `reset_right.npz`(키 `arm_target`)를 못 읽었다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
PRESET = SIM2REAL / "logs/shadow/reset_both/reset_right.npz"
needs_preset = pytest.mark.skipif(not PRESET.exists(), reason="preset npz 없음")


def _tool():
    path = SIM2REAL / "policy_control/tools/replay_to_pd.py"
    spec = importlib.util.spec_from_file_location("replay_to_pd_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    def __init__(self, npz, joints, key=None, env=0, dt=0.02):
        self.npz, self.hdf5, self.joints, self.env, self.dt = npz, None, joints, env, dt
        self.npz_key = key


@needs_preset
def test_reads_the_preset_trajectory_without_naming_its_key():
    mod = _tool()
    q, dt = mod.load_frames(_Args(PRESET, [f"r_aj_{i}" for i in range(1, 8)]))
    assert q.shape == (2605, 7), q.shape
    assert 0.005 < dt < 0.1, dt
    assert np.allclose(q[0], 0.0, atol=0.02), "첫 프레임은 차렷이어야 한다"
    assert abs(q[-1][3] - 2.0) < 1e-3 and abs(q[-1][1] - 0.3) < 1e-3, "끝은 preset(j2 0.3 · j4 2.0)"


@needs_preset
def test_explicit_key_wins():
    mod = _tool()
    q, _ = mod.load_frames(_Args(PRESET, [f"r_aj_{i}" for i in range(1, 8)], key="arm_target"))
    assert q.shape == (2605, 7)


@needs_preset
def test_unknown_key_names_the_available_ones():
    mod = _tool()
    with pytest.raises(SystemExit) as e:
        mod.load_frames(_Args(PRESET, [f"r_aj_{i}" for i in range(1, 8)], key="nope"))
    assert "arm_target" in str(e.value), "무엇을 쓸 수 있는지 알려줘야 한다"

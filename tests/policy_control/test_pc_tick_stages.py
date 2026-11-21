"""★11.10 정책 틱 단계 계측(stages_ms) · 넘친 틱 경고 · rh_aglt 장치 한 곳(AGLT_DEVICE) · JointPolicy 직접 경로."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from policy_control import pour_fj_node as N
from policy_control.tick_budget import TickBudget

REPO = Path(__file__).resolve().parents[2]
OLD_KEYS = {"budget_ms", "ticks", "proc_over", "late", "proc_max_ms", "gap_max_ms"}


def test_budget_without_stages_keeps_the_old_status_keys():
    b = TickBudget(16.7)
    b.record(0.0, 5.0)
    assert set(b.as_dict()) == OLD_KEYS


def test_budget_stage_mean_and_max():
    b = TickBudget(16.7)
    b.record(0.0, 5.0, {"infer": 3.0, "obs": 1.0})
    b.record(0.016, 9.0, {"infer": 7.0, "obs": 1.0})
    d = b.as_dict()
    assert d["stage_mean_ms"] == {"infer": 5.0, "obs": 1.0} and d["stage_max_ms"] == {"infer": 7.0, "obs": 1.0}
    b.reset()
    assert set(b.as_dict()) == OLD_KEYS


def test_tick_stages_split_obs_and_infer():
    s = N.tick_stages(0.0, 0.002, 0.010, 0.011, 0.0125, infer_ms=6.0)
    assert s == pytest.approx({"measure": 2.0, "obs": 2.0, "infer": 6.0, "targets": 1.0, "publish": 1.5})


def _on_tick(budget_ms, chain_ms=0.0):
    out = {"status": [], "warn": []}

    def step(meas):
        me.chain.infer_ms = 0.5
        if chain_ms:
            import time
            time.sleep(chain_ms / 1e3)
        return np.zeros(2), np.zeros(1), {}
    me = NS(_drain=lambda: None, book=NS(phase="running", episode=1), chain=NS(step=step, step_i=1),
            _measure=lambda advance=False: {"arm": None}, cup_fresh={}, contract=NS(hold_steps=0),
            fam=NS(targets=lambda c, t: (["a"], np.zeros(1), np.zeros(1)), errors=(ValueError,)), _gap=0, _seq=0, max_gap=3,
            _budget=TickBudget(budget_ms), _lg_stop=None, _publish=True, _pub_obs=NS(publish=lambda m: None),
            _pub_action=NS(publish=lambda m: None), _publish_debug=lambda: None, _pub_target=NS(publish=lambda m: None),
            _status=lambda ok, reasons, extra=None: out["status"].append(extra),
            _end=lambda *a: None, _after_step=lambda: True, max_s=0, _t_start=0.0,
            get_logger=lambda: NS(warning=lambda msg, **kw: out["warn"].append((msg, kw))))
    me._lift_guard_tick = lambda meas, names, q: {}
    N.PourFjNode._on_tick(me)
    return out


def test_status_carries_stages_and_no_warning_under_budget():
    out = _on_tick(1000.0)
    (extra,) = out["status"]
    assert set(extra["stages_ms"]) == {"measure", "obs", "infer", "targets", "publish"}
    assert extra["stages_ms"]["infer"] == 0.5 and "stage_mean_ms" in extra["tick"]
    assert out["warn"] == []


def test_over_budget_tick_warns_throttled():
    out = _on_tick(1.0, chain_ms=3.0)
    (msg, kw), = out["warn"]
    assert "예산" in msg and kw == {"throttle_duration_sec": 5.0}
    assert out["status"][0]["tick"]["proc_over"] == 1


def _gen():
    spec = importlib.util.spec_from_file_location("make_rh56f1_missions_dev", REPO / "scripts/ops/make_rh56f1_missions.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def _aglt_devices(node) -> list:
    found = []
    if isinstance(node, dict):
        for v in node.values():
            found += _aglt_devices(v)
    elif isinstance(node, list):
        if any(isinstance(x, str) and x.endswith("rh_aglt_node.py") for x in node):
            found += [x for x in node if isinstance(x, str) and x.startswith("device:=")]
        else:
            for v in node:
                found += _aglt_devices(v)
    return found


def test_aglt_device_is_one_switch_and_defaults_to_cpu():
    gen = _gen()
    assert gen.AGLT_DEVICE == "cpu"                     # 기본 = 지금 동작
    cpu = _aglt_devices(gen.mission("real"))
    assert cpu and set(cpu) == {"device:=cpu"}
    gen.AGLT_DEVICE = "cuda"
    cuda = _aglt_devices(gen.mission("real"))
    assert len(cuda) == len(cpu) and set(cuda) == {"device:=cuda"}


def _ap05_contract():
    pytest.importorskip("torch")
    pytest.importorskip("rl_games")
    from policy_control import rh_aglt as A
    c = A.load_contract(REPO / "deploy/policies/rh56f1/aglt/right_approach_ap05/rh_aglt_contract.json")
    if not Path(c.checkpoint).is_file():
        pytest.skip("가중치 없음(.gitignore)")
    return c


def _roll(pol, n=40):
    rng = np.random.default_rng(0)
    pol.reset()
    return np.stack([pol.forward(rng.normal(scale=0.5, size=pol._dim)) for _ in range(n)])


def test_int8_is_off_by_default_and_off_path_is_the_old_get_action(monkeypatch):
    """끔(기본) = 예전과 같은 get_action 경로 · 같은 출력. 켜면 int8 망을 쓰고 출력이 조금 다르다(작은 차이만 확인)."""
    from policy_control.cpu_plan import INT8_ENV
    from policy_control.joint_policy import JointPolicy
    c = _ap05_contract()
    monkeypatch.delenv(INT8_ENV, raising=False)
    off = JointPolicy(c, "cpu")
    assert off._int8_net is None
    ref = []
    off.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):                                  # 예전 forward 를 그대로 다시 쓴 값
        import torch
        t = torch.as_tensor(np.asarray(rng.normal(scale=0.5, size=off._dim), np.float32)).unsqueeze(0)
        with torch.no_grad():
            ref.append(off._policy.get_action(t)[0].numpy().astype(np.float64))
    assert np.array_equal(_roll(off), np.stack(ref))
    monkeypatch.setenv(INT8_ENV, "1")
    on = JointPolicy(c, "cpu")
    assert on._int8_net is not None
    d = np.abs(_roll(on) - np.stack(ref)).max()
    assert 0.0 < d < 0.2
    assert JointPolicy(c, "cpu")._int8_net is not None and _roll(on).shape == (40, c.action_dim)


def test_policy_device_log_names_int8_only_on_cpu():
    from policy_control.cpu_plan import INT8_ENV, policy_device
    assert policy_device("cpu", {}) == ("cpu", "정책 장치: cpu (미션 device:=)")
    assert "int8" in policy_device("cpu", {INT8_ENV: "1"})[1]
    assert "int8" not in policy_device("cuda", {INT8_ENV: "1"})[1]

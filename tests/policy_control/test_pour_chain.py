"""End-to-end offline pour chain (measure -> obs -> policy -> decoder [-> fabric]) vs the golden trace.

Teacher-forced: the policy backend replays trace actions; measurement for step t is trace row t-1."""
import numpy as np
import pytest

from policy_control.pour_chain import PourChain, PourChainError, PourMeasure, RecordedPolicy
from policy_control.pour_obs import CupPose, segment_slices
from pour_trace_util import contract, make_fk, trace
from test_pour_golden import NOISE_BOUND

LAST = 898


def _measure(c, z, meta, row, e, fill=True):
    names = meta["joint_names"]
    return PourMeasure(
        joint_pos=dict(zip(names, z["joint_pos"][row, e])), joint_vel=dict(zip(names, z["joint_vel"][row, e])),
        cups={s.role: CupPose(z[f"{s.role}_cup_pos"][row, e], z[f"{s.role}_cup_quat"][row, e]) for s in c.sides},
        forces={s.role: (z[f"{s.role}_f_mid"][row, e], z[f"{s.role}_f_dist"][row, e]) for s in c.sides},
        fill_level=float(z["fill_level"][row, e]) if fill else None)


@pytest.fixture(scope="module")
def ctx():
    c = contract()
    z, meta = trace()
    return c, z, meta, {s.role: make_fk(s) for s in c.sides}


@pytest.mark.parametrize("env", [0, 3])
def test_chain_reproduces_obs_gate_and_targets(ctx, env):
    c, z, meta, fks = ctx
    chain = PourChain(c, RecordedPolicy(z["actions"][:, env]), fks)
    chain.reset()
    chain.step(_measure(c, z, meta, 0, env))                     # step 0: no pre-step row exists in the trace
    sl, worst = segment_slices(c), {}
    for t in range(1, LAST):
        out = chain.step(_measure(c, z, meta, t - 1, env))
        d = np.abs(out.obs - z["obs_next"][t - 1, env])
        for k, s_ in sl.items():
            worst[f"obs/{k}"] = max(worst.get(f"obs/{k}", 0.0), float(d[s_].max()))
        for s in c.sides:
            r = s.role
            for k, v, ref in (("gate", out.gates[r], z[f"{r}_close_gate"][t - 1, env]),
                              ("palm_target", out.targets[r].palm_target, z[f"{r}_palm_tgt"][t, env]),
                              ("hand_target", out.targets[r].hand_target, z[f"{r}_syn_target"][t, env])):
                worst[f"{r}/{k}"] = max(worst.get(f"{r}/{k}", 0.0), float(np.abs(np.asarray(v) - ref).max()))
    print("chain max abs err:", {k: f"{v:.2e}" for k, v in worst.items()})
    for k, v in worst.items():
        bound = NOISE_BOUND.get(k.split("/")[-1], 1e-5) if k.startswith("obs/") else \
            (2e-2 if k.endswith("gate") else 1e-5)
        assert v < bound, (k, v)
    assert chain.fabric is None and out.joint_targets is None


def test_chain_without_fill_level_uses_contract_default_or_raises(ctx):
    c, z, meta, fks = ctx
    chain = PourChain(c, RecordedPolicy(z["actions"][:, 0]), fks)
    chain.reset()
    if c.fill_level.default is None:
        with pytest.raises(Exception):
            chain.step(_measure(c, z, meta, 0, 0, fill=False))
    else:
        chain.step(_measure(c, z, meta, 0, 0, fill=False))


def test_chain_requires_reset_and_both_sides(ctx):
    c, z, meta, fks = ctx
    chain = PourChain(c, RecordedPolicy(z["actions"][:, 0]), fks)
    with pytest.raises(PourChainError):
        chain.step(_measure(c, z, meta, 0, 0))
    with pytest.raises(PourChainError):
        PourChain(c, RecordedPolicy(z["actions"][:, 0]), {"src": fks["src"]})


def test_chain_with_fake_fabric_passes_targets_through(ctx):
    c, z, meta, fks = ctx

    class _Fab:
        def __init__(self): self.calls = []
        def reset(self, home): self.home = home
        def step(self, palm, hand, hold=False):
            self.calls.append((palm, hand, hold))
            return {r: ("jt", r) for r in palm}

    fab = _Fab()
    chain = PourChain(c, RecordedPolicy(z["actions"][:, 0]), fks, fabric=fab)
    chain.reset(home={"src": np.zeros(27), "rcv": np.zeros(27)})
    out = chain.step(_measure(c, z, meta, 0, 0))
    assert out.joint_targets == {"src": ("jt", "src"), "rcv": ("jt", "rcv")}
    palm, hand, hold = fab.calls[0]
    assert np.allclose(palm["src"], out.targets["src"].palm_target) and hold == (not out.active)

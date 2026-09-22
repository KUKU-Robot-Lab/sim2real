"""pour_bimanual fabric step: contract fabric section, world box, fake-backend wiring, GPU parity."""
import dataclasses

import numpy as np
import pytest

from policy_control.fabric_core import FabricBackend, FabricError
from policy_control.pour_contract import PourContractError, from_dict, to_dict, validate
from policy_control.pour_fabric import PourFabricCore, PourFabricPair, pour_world_dict
from pour_trace_util import contract, trace


class _FakeFabric:
    def __init__(self, n):
        self.num_joints = n
        self.default_config = _Cfg()
        self.calls = []

    def set_features(self, pca, feat, orient, q, qd, ids, ind, damping):
        self.calls.append((feat.clone(), orient, q.clone()))


class _Cfg:
    def copy_(self, q):
        self.q = q.clone()


class _FakeIntegrator:
    def step(self, q, qd, qdd, dt):
        return q + 0.01, qd + 1.0, qdd


def _fake(n=27):
    return FabricBackend(fabric=_FakeFabric(n), integrator=_FakeIntegrator(),
                         object_ids=None, object_indicator=None, device="cpu")


def test_contract_carries_the_env_fabric_settings():
    f = contract().fabric
    assert (f.damping, f.max_objects, f.vel_ff_scale) == (10.0, 8, 1.0)
    assert (f.use_hand_repulsion, f.use_body_repulsion_pairs) == (False, True)
    assert (f.table_obstacle, f.table_margin_xy, f.table_thickness, f.table_z) == (True, 0.1, 0.05, 0.205)


def test_fabric_section_round_trips_and_is_validated():
    c = contract()
    assert from_dict(to_dict(c)) == c
    bad = dataclasses.replace(c, fabric=dataclasses.replace(c.fabric, table_thickness=0.0))
    with pytest.raises(PourContractError):
        validate(bad)


def test_world_is_the_union_of_both_palm_boxes():
    c = contract()
    w = pour_world_dict(c)["table"]
    lo = [min(s.box_lo[i] for s in c.sides) for i in range(2)]
    hi = [max(s.box_hi[i] for s in c.sides) for i in range(2)]
    sx, sy, th = (float(v) for v in w["scaling"].split())
    cx, cy, cz = (float(v) for v in w["transform"].split()[:3])
    assert sx == pytest.approx(hi[0] - lo[0] + 0.2) and sy == pytest.approx(hi[1] - lo[1] + 0.2)
    assert (cx, cy) == pytest.approx((0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])))
    assert th == 0.05 and cz == pytest.approx(0.205 - 0.025)


def test_world_is_none_when_the_run_has_no_table_obstacle():
    c = contract()
    assert pour_world_dict(dataclasses.replace(c, fabric=dataclasses.replace(c.fabric, table_obstacle=False))) is None


def test_core_syncs_hand_then_integrates_decimation_times():
    c = contract()
    s = c.side("src")
    home = np.linspace(0.0, 0.26, 27)
    core = PourFabricCore(c, "src", "cpu", home, backend=_fake())
    assert core.joint_names == tuple(s.fabric_joint_order)
    hand = np.arange(20, dtype=float) / 100.0          # contract hand_joints order
    out = core.step(np.arange(6.0), hand)
    fab = core.backend.fabric
    feat, orient, q_in = fab.calls[0]
    assert orient == "euler_zyx" and np.allclose(feat[0].numpy(), np.arange(6.0))
    want_hand = [hand[list(s.hand_joints).index(n)] for n in s.fabric_joint_order[7:]]
    assert np.allclose(q_in[0, 7:].numpy(), want_hand)
    assert out.substeps.shape == (c.fabric_decimation, 27)
    assert np.allclose(out.q_arm, home[:7] + 0.01 * c.fabric_decimation, atol=1e-6)
    assert np.allclose(out.qd_arm, c.fabric_decimation * c.fabric.vel_ff_scale)


def test_core_requires_a_hand_target_and_checks_sizes():
    c = contract()
    core = PourFabricCore(c, "rcv", "cpu", np.zeros(27), backend=_fake())
    with pytest.raises(FabricError):
        core.step(np.zeros(6), None)
    with pytest.raises(FabricError):
        core.step(np.zeros(5), np.zeros(20))
    with pytest.raises(FabricError):
        PourFabricCore(c, "src", "cpu", np.zeros(27), backend=_fake(26))


def test_hold_pins_state_and_zeroes_velocity():
    c = contract()
    home = np.full(27, 0.1)
    core = PourFabricCore(c, "src", "cpu", home, backend=_fake())
    out = core.step(np.zeros(6), np.full(20, 0.3), hold=True)
    assert np.allclose(out.q_arm, 0.1) and np.allclose(out.qd_arm, 0.0)
    assert np.allclose(core.q[7:], 0.3)               # pin point is taken after the hand sync (env order)


def test_pair_steps_both_roles():
    c = contract()
    pair = PourFabricPair(c, "cpu", {"src": np.zeros(27), "rcv": np.zeros(27)},
                          backends={"src": _fake(), "rcv": _fake()})
    out = pair.step({r: np.zeros(6) for r in c.roles}, {r: np.zeros(20) for r in c.roles})
    assert set(out) == {"src", "rcv"} and out["rcv"].q_full.shape == (27,)


# ------------------------------------------------------------------ GPU parity vs the sim trace
def _gpu_pair(c, home):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():       # fabrics_sim 을 import 하기 전에 판정한다 (warp 가 장치를 잡는다)
        pytest.skip("no CUDA device")
    pytest.importorskip("fabrics_sim")
    return PourFabricPair(c, "cuda:0", home)


def _first_active_row(hold):
    # the snapshot is taken after env.step, so its hold flag already belongs to the NEXT step:
    # the row after the last hold-flagged row was still pinned (its fabric_qd is exactly 0).
    return int(np.flatnonzero(hold[:200])[-1]) + 2


@pytest.mark.gpu
@pytest.mark.parametrize("env", [0, 1])
def test_fabric_one_step_matches_the_sim_trace(env):
    """Teacher-forced: state from trace row t-1, targets of row t -> fabric_q of row t."""
    c = contract()
    z, _ = trace()
    t1 = _first_active_row(z["hold"][:, env].astype(bool))
    assert np.all(z["src_fabric_qd"][t1 - 1, env] == 0.0)
    pair = _gpu_pair(c, {r: z[f"{r}_fabric_q"][t1 - 1, env] for r in c.roles})
    errs = []
    for t in range(t1, t1 + 60):
        for r in c.roles:
            pair.cores[r].set_state(z[f"{r}_fabric_q"][t - 1, env], z[f"{r}_fabric_qd"][t - 1, env])
        out = pair.step({r: z[f"{r}_palm_tgt"][t, env] for r in c.roles},
                        {r: z[f"{r}_syn_target"][t, env] for r in c.roles})
        errs.append(max(float(np.abs(out[r].q_full - z[f"{r}_fabric_q"][t, env]).max()) for r in c.roles))
    print(f"[pour fabric one-step parity] env{env} rows {t1}..{t1 + 59}: "
          f"max {max(errs):.3e} mean {np.mean(errs):.3e} rad")
    # fabrics_sim on cuda is NOT deterministic (measured 2026-09-17): one tick with identical (q, qd, targets)
    # repeated 200x gave 67..129 distinct outputs, spread up to 5.2e-4 rad. Over 30 repeats of these 60 ticks the
    # max ranged 5.7e-4..1.8e-3, and one full-suite run logged 2.823e-3, which made the original `max < 2e-3`
    # bound flaky. The mean is the tight check; the max only catches gross breakage.
    assert np.mean(errs) < 4e-4
    assert max(errs) < 5e-3

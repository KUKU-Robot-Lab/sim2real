"""Bimanual pour family vs the sim golden trace (i11 checkpoint). Row 898 is post-reset, so rows 0..897 are used.

Trace alignment (measured): row t = snapshot after env.step(actions[t]); obs_next[t] is the actor obs built
from that snapshot. Decoder gate/force inputs for step t are the measurements of row t-1 (state before the step).
"""
import numpy as np
import pytest

from policy_control.pour_decoder import PourDecoder, SideInputs
from policy_control.pour_obs import PourObsError, build_obs, clip_obs, resolve_fill_level, segment_slices
from pour_trace_util import contract, make_fk, side_state, trace

LAST = 898
# Actor obs carries gaussian training noise on q / qd / palm+tip positions (env.yaml obs_noise_qpos 0.002,
# obs_noise_qvel 0.05, obs_noise_body 0.005); the trace holds no clean copy of those, so they are bounded at
# 5 sigma (sqrt(2) sigma where two noisy points are subtracted). Cup noise/delay was 0 in this play session
# (ADR level 0), so every cup-derived segment is exact. Everything not listed here must match within 1e-5.
NOISE_BOUND = {"arm_q": 5 * 0.002, "hand_q": 5 * 0.002, "arm_qd": 5 * 0.05, "palm_pos": 5 * 0.005,
               "cup_rel_palm": 5 * 0.005, "tips_rel_cup": 5 * 0.005, "tips_rel_palm": 5 * 0.005 * 2 ** 0.5}


@pytest.fixture(scope="module")
def ctx():
    c = contract()
    z, meta = trace()
    return c, z, meta


def _inputs(c, z, t, e):
    p = max(t - 1, 0)
    return {s.role: SideInputs(float(z[f"{s.role}_close_gate"][p, e]), z[f"{s.role}_f_mid"][p, e],
                               z[f"{s.role}_f_dist"][p, e]) for s in c.sides}


def _run_decoder(c, z, e):
    dec, errs, prevs = PourDecoder(c), {}, []
    for t in range(LAST):
        out = dec.step(z["actions"][t, e], _inputs(c, z, t, e))
        prevs.append(dec.prev_action_obs(z["actions"][t, e]))
        for i, s in enumerate(c.sides):
            r = s.role
            for k, v, ref in (("palm_cmd", out[r].palm_cmd, z["palm_cmd"][t, e, i]),
                              ("palm_target", out[r].palm_target, z[f"{r}_palm_tgt"][t, e]),
                              ("syn_close", out[r].syn_close, z[f"{r}_syn_close"][t, e]),
                              ("hand_target", out[r].hand_target, z[f"{r}_syn_target"][t, e])):
                errs[f"{r}/{k}"] = max(errs.get(f"{r}/{k}", 0.0), float(np.abs(v - ref).max()))
    return errs, prevs


def test_decoder_reproduces_trace_targets_for_every_env(ctx):
    c, z, _ = ctx
    worst = {}
    for e in range(z["actions"].shape[1]):
        errs, _ = _run_decoder(c, z, e)
        worst = {k: max(worst.get(k, 0.0), v) for k, v in errs.items()}
    print("decoder max abs err:", {k: f"{v:.2e}" for k, v in worst.items()})
    assert max(worst.values()) < 1e-5, worst


def test_obs_deterministic_segments_match_trace(ctx):
    c, z, meta = ctx
    sl = segment_slices(c)
    fks = {s.role: make_fk(s) for s in c.sides}
    worst = {}
    for e in (0, 3):
        _, prevs = _run_decoder(c, z, e)
        for t in range(0, LAST, 7):
            st = {s.role: side_state(z, meta, s, fks[s.role], t, e, use_fk=False) for s in c.sides}
            obs = clip_obs(c, build_obs(c, st, float(z["fill_level"][t, e]), prevs[t]))
            d = np.abs(obs - z["obs_next"][t, e])
            for k, s_ in sl.items():
                worst[k] = max(worst.get(k, 0.0), float(d[s_].max()))
    print("obs max abs err:", {k: f"{v:.2e}" for k, v in worst.items()})
    for k, v in worst.items():
        assert v < NOISE_BOUND.get(k.split("/")[1], 1e-5), (k, v)


def test_fk_palm_matches_sim_body_pose(ctx):
    c, z, meta = ctx
    for s in c.sides:
        fk = make_fk(s)
        for t in (0, 200, 600, 897):
            st = side_state(z, meta, s, fk, t, 0, use_fk=True)
            assert np.abs(st.palm_pos - z[f"{s.role}_palm_pose"][t, 0, :3]).max() < 2e-3
            assert np.abs(st.tips - z[f"{s.role}_tips"][t, 0]).max() < 2e-3


def test_fill_level_is_never_silently_defaulted(ctx):
    c = ctx[0]
    if c.fill_level.default is None:
        with pytest.raises(PourObsError):
            resolve_fill_level(c, None)
    with pytest.raises(PourObsError):
        resolve_fill_level(c, 1.5)
    assert resolve_fill_level(c, 0.25) == 0.25


def test_decoder_rejects_bad_action(ctx):
    c = ctx[0]
    from policy_control.pour_decoder import PourDecodeError
    with pytest.raises(PourDecodeError):
        PourDecoder(c).step(np.zeros(5), {})
    with pytest.raises(PourDecodeError):
        PourDecoder(c).step(np.full(c.action_dim, np.nan), {})

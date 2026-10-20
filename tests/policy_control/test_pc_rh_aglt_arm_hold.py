"""★10.19 팔 들림 래치(hdgp arm_hold_rise_m) — 배포 디코더 RaDecoder 가 학습 규칙과 같은 팔 목표를 내는지.

hdgp 51094780 tasks/rh_aglt_l/arm_action.py update_arm_latch · step_arm_hold · rh_aglt_env.py :957-966 · :1355-1358.
aglt_l_neo03(10.19 왼손 실기 후보)가 이 규칙으로 학습했다 — 디코더가 증분만 쓰면 들어 올린 뒤 팔 거동이 sim 과 달라진다."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from policy_control import pour_fj_node as N
from policy_control import rh_aglt as A

FIX = Path(__file__).resolve().parent / "fixtures"
LEFT = FIX / "rh_aglt_left_i05_contract.json"
HDGP = Path.home() / "rl_ws/hdgp/source/openarm/openarm/agnostic/tasks/rh_aglt_l/arm_action.py"
HOLD = {"arm_hold_rise_m": 0.07, "arm_hold_release_m": 0.03, "arm_hold_release_steps": 30}


def _c(**kw):
    return replace(A.load_contract(LEFT), **{**HOLD, **kw})


def _hand_q(c):
    return np.asarray(c.side().hand_open, float)


def _run(dec, acts, zs, active_from=0):
    out = []
    for t, (a, z) in enumerate(zip(acts, zs)):
        tg = dec.step(a, active=t >= active_from, hand_q=_hand_q(dec.c), tactile_n=np.zeros(5), cup_z=z)
        out.append((tg["arm"][0].copy(), dec.arm_latched))
    return out


def _z_path(n_up=60, n_hold=40, n_down=50, n_up2=40, z0=0.08):
    """올림(0 → 0.12 m) · 유지 · 내림(0.01 m, 해제) · 다시 올림 — 래치 · 해제 · 재래치를 다 지난다."""
    up = np.linspace(0.0, 0.12, n_up)
    return z0 + np.concatenate([up, np.full(n_hold, 0.12), np.full(n_down, 0.01), np.linspace(0.01, 0.1, n_up2)])


def test_old_contracts_keep_the_plain_increment():
    c = A.load_contract(LEFT)
    assert c.arm_hold_rise_m == 0.0
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1, 1, (50, 13))
    a, b = A.RaDecoder(c), A.RaDecoder(c)
    ra = _run(a, acts, [None] * 50)                          # 래치 없는 계약은 cup_z 가 없어도 된다
    rb = _run(b, acts, list(0.08 + np.linspace(0, 0.3, 50)))  # 컵이 올라가도 증분 그대로
    assert all(np.allclose(x[0], y[0]) for x, y in zip(ra, rb)) and not any(x[1] for x in rb)
    assert a.arm_hold_view() == {}


def test_latch_step_matches_increment_and_then_does_not_accumulate():
    c = _c()
    dec, inc = A.RaDecoder(c), A.RaDecoder(A.load_contract(LEFT))
    a = np.zeros(13)
    a[:7] = 1.0                                              # 같은 행동을 계속 낸다
    zs = 0.08 + np.r_[np.zeros(5), np.full(200, 0.08)]       # 6 번째 스텝에 +0.08 m(> 0.07) 들림
    out = _run(dec, [a] * len(zs), zs)
    ref = _run(inc, [a] * len(zs), [None] * len(zs))
    assert not out[4][1] and out[5][1]
    assert np.allclose(out[5][0], ref[5][0])                 # 래치 스텝 = 증분과 같은 값(튀지 않음)
    q_latch = ref[4][0]
    want = np.clip(q_latch + c.k_arm * 1.0, c.side().arm_lo, c.side().arm_hi)
    assert np.allclose(out[-1][0], want, atol=1e-6)          # q_latch + k·a 로 수렴(누적 없음)
    assert not np.allclose(ref[-1][0], want, atol=1e-3)      # 증분이면 계속 간다


def test_z0_is_the_first_active_step_and_release_needs_30_low_steps():
    c = _c()
    dec = A.RaDecoder(c)
    a = np.zeros(13)
    zs = [0.50] * 10 + [0.08] + [0.16] * 5 + [0.10] * 29 + [0.10] + [0.16]
    # 대기 10 스텝(z 0.50 은 쓰지 않는다) → z0 = 0.08 · +0.08 래치 · +0.02(< 0.03) 30 스텝 연속 → 해제 · 다시 +0.08 → 새 래치
    out = _run(dec, [a] * len(zs), zs, active_from=10)
    lat = [x[1] for x in out]
    assert not any(lat[:11]) and all(lat[11:16])
    assert all(lat[16:45]) and not lat[45] and lat[46]
    assert dec.arm_hold_view() == {"arm_latched": True, "arm_rise": pytest.approx(0.08), "arm_z0": pytest.approx(0.08)}
    dec.reset()
    assert not dec.arm_latched and dec._z0 is None


def test_missing_or_nan_cup_z():
    c = _c()
    with pytest.raises(A.RhAgltError, match="cup_z"):
        A.RaDecoder(c).step(np.zeros(13), active=True, hand_q=_hand_q(c), tactile_n=np.zeros(5))
    dec = A.RaDecoder(c)
    out = _run(dec, [np.zeros(13)] * 4, [float("nan"), 0.08, float("nan"), 0.2])
    assert [x[1] for x in out] == [False, False, False, True]   # z0 는 첫 유한 값 · nan 은 판정에 쓰지 않는다


def test_contract_validation():
    with pytest.raises(A.RhAgltError):
        A.validate(_c(arm_hold_release_m=0.08))              # 해제 높이 ≥ 래치 높이
    with pytest.raises(A.RhAgltError):
        A.validate(_c(arm_hold_rise_m=0.0))                  # 해제만 있음
    with pytest.raises(A.RhAgltError):
        A.validate(_c(arm_hold_release_steps=0))
    A.validate(_c(arm_hold_release_m=0.0))                   # 풀지 않는 래치(hdgp release_m 0)


def test_chain_feeds_the_observed_cup_z():
    c = _c()

    class Zero:
        def forward(self, obs):
            return np.zeros(13)

    ch = N.AgltChain(c, Zero())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_pc_rh_aglt as T
    m = T._meas(c, cup=(0.35, 0.16, 0.08))
    ch.reset({"arm": m})
    for _ in range(c.hold_steps + 2):
        ch.step({"arm": m})
    assert ch.dec._z0 == pytest.approx(0.08) and not ch.dec.arm_latched
    up = T._meas(c, cup=(0.35, 0.16, 0.16))
    ch.step({"arm": up})
    assert ch.dec.arm_latched


def _load_hdgp():
    if not HDGP.is_file():
        pytest.skip(f"hdgp 없음: {HDGP}")
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("_hdgp_arm_action", HDGP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "update_arm_latch"):
        pytest.skip("hdgp arm_action 에 update_arm_latch 없음(10.18 전 트리)")
    return mod, torch


def test_parity_with_hdgp_arm_action():
    """학습 함수(torch, env 1 개)와 배포 디코더가 같은 팔 목표를 낸다 — 래치 · 해제 · 재래치 경로 전체."""
    H, torch = _load_hdgp()
    c = _c()
    s = c.side()
    lo, hi = torch.tensor([s.arm_lo], dtype=torch.float64), torch.tensor([s.arm_hi], dtype=torch.float64)
    rng = np.random.default_rng(3)
    zs = _z_path()
    acts = rng.uniform(-1.2, 1.2, (len(zs), 13))
    hold = 10
    dec = A.RaDecoder(c)
    q = torch.tensor([s.arm_home], dtype=torch.float64)
    latched, q_latch = torch.zeros(1, dtype=torch.bool), q.clone()
    below = torch.zeros(1, dtype=torch.long)
    z0, z0_ok = torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.bool)
    seen = set()
    for t, (a, z) in enumerate(zip(acts, np.r_[np.full(hold, 0.3), zs])):
        active = t >= hold
        # hdgp _get_observations: 정책이 처음 움직이는 스텝의 관측 컵 z 를 z0 로(관측은 그 스텝 pre_physics 가 쓴다)
        if active and not bool(z0_ok):
            z0, z0_ok = torch.tensor([z], dtype=torch.float64), torch.ones(1, dtype=torch.bool)
        at = torch.tensor([a[:7]], dtype=torch.float64)
        q_new = H.step_arm_increment(q, at, k=c.k_arm, alpha=c.arm_ema, q_lo=lo, q_hi=hi)
        latched, q_latch, below, _ = H.update_arm_latch(latched, q_latch, q, torch.tensor([z], dtype=torch.float64) - z0,
                                                        z0_ok, below, rise_m=0.07, release_m=0.03, release_steps=30)
        q_hold = H.step_arm_hold(q, at, q_latch, k=c.k_arm, alpha=c.arm_ema, q_lo=lo, q_hi=hi)
        q_new = torch.where(latched.unsqueeze(1), q_hold, q_new)
        q = q_new if active else torch.tensor([s.arm_home], dtype=torch.float64)
        got = dec.step(a, active=active, hand_q=_hand_q(c), tactile_n=np.zeros(5), cup_z=float(z))["arm"][0]
        assert dec.arm_latched == bool(latched[0]), t
        assert np.allclose(got, q[0].numpy(), atol=1e-9), t
        seen.add(bool(latched[0]))
    assert seen == {True, False}

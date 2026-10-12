"""10.12 손가락 막힘 신호(grasping 확정 정의) — 배포 numpy 판이 hdgp hand_block.HandBlock(torch, DR 끔)과 같은 값을 낸다."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from policy_control import hand_block as HB

HDGP_HB = Path.home() / "rl_ws/hdgp/source/openarm/openarm/agnostic/tasks/rh_aglt_r/hand_block.py"
#: RH56F1 JOINTS 순(thumb_1 회전, thumb_2 굽힘, index, middle, ring, pinky) — rh_aglt 계약 hand_joints 순서
Q_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Q_GRIP = np.array([1.2, 0.5, 1.4, 1.4, 1.4, 1.4])
NAMES = ["r_hj_thumb_1", "r_hj_thumb_2", "r_hj_index_1", "r_hj_middle_1", "r_hj_ring_1", "r_hj_pinky_1"]
NOM = {"g_rad": 0.0218, "eps_rad": 0.0109, "n_ticks": 3, "lag_ticks": 6,
       "axes": ["thumb_flex", "index", "middle", "ring", "pinky"]}


def _blk(**kw):
    return HB.HandBlockNp.from_contract(dict(NOM, **kw), NAMES, Q_OPEN, Q_GRIP)


def _scenario(stop_at: float = 0.30, steps: int = 60, rate: float = 0.025, lag: int = 5):
    """q* 가 스텝당 rate 로 닫히고, 실측은 lag 틱 뒤 따라가다 stop_at 에서 막힌다(엄지굽힘 · 검지 · 중지)."""
    qs, qm = [], []
    star = Q_OPEN.copy()
    hist = [star.copy()]
    for t in range(steps):
        star = np.minimum(star + rate * np.sign(Q_GRIP - Q_OPEN), Q_GRIP)
        hist.append(star.copy())
        q = hist[max(0, len(hist) - 1 - lag)].copy()
        q[[1, 2, 3]] = np.minimum(q[[1, 2, 3]], stop_at)
        qs.append(star.copy()), qm.append(q)
    return qs, qm


def test_the_contract_axes_map_to_the_rh56f1_joint_names():
    b = _blk()
    assert b.axes.tolist() == [False, True, True, True, True, True]      # 엄지 회전은 위치 제어 축
    assert b.thumb_flex == 1 and b.fingers == [2, 3, 4, 5]
    assert b.cell[1] == pytest.approx(1 / 527) and b.cell[2] == pytest.approx(1 / 550)
    with pytest.raises(HB.HandBlockError):
        _blk(axes=["thumb_flex", "wrist"])
    with pytest.raises(HB.HandBlockError):
        _blk(g_rad=-1.0)
    with pytest.raises(HB.HandBlockError):
        HB.HandBlockNp.from_contract({k: v for k, v in NOM.items() if k != "lag_ticks"}, NAMES, Q_OPEN, Q_GRIP)


def test_free_closing_with_command_lag_is_not_blocked_and_a_stopped_finger_latches():
    b = _blk()
    qs, qm = _scenario()
    prev = Q_OPEN.copy()
    first = None
    for t, (s, q) in enumerate(zip(qs, qm)):
        latch = b.step(prev, q)
        if t < 15:                                     # 자유 닫힘(지연 5 틱 < lag 6) — 막힘 아님
            assert not latch.any(), (t, latch)
        if first is None and b.grasped():
            first = t
        prev = s
    assert first is not None and b.latch[[1, 2, 3]].all() and not b.latch[[0, 4, 5]].any()


def test_opening_the_target_releases_the_latch():
    b = _blk()
    qs, qm = _scenario()
    prev = Q_OPEN.copy()
    for s, q in zip(qs, qm):
        b.step(prev, q)
        prev = s
    assert b.grasped()
    opened = prev - 0.05 * np.sign(Q_GRIP - Q_OPEN)
    b.step(opened, qm[-1])                            # 직전 q* → 지금 q* 가 여는 쪽
    b.step(opened - 0.05 * np.sign(Q_GRIP - Q_OPEN), qm[-1])
    assert not b.latch.any()


def test_reset_flushes_the_history():
    b = _blk()
    qs, qm = _scenario()
    prev = Q_OPEN.copy()
    for s, q in zip(qs, qm):
        b.step(prev, q)
        prev = s
    b.reset()
    assert not b.latch.any() and not b.grasped()
    assert not b.step(Q_OPEN, Q_OPEN).any()


def _torch_block():
    torch = pytest.importorskip("torch")
    if not HDGP_HB.is_file():
        pytest.skip("hdgp hand_block.py 없음")
    spec = importlib.util.spec_from_file_location("hdgp_hand_block", HDGP_HB)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod                    # dataclass 가 모듈을 찾는다
    spec.loader.exec_module(mod)
    axes = torch.tensor([False, True, True, True, True, True])
    p = mod.BlockParams(g_rad=(NOM["g_rad"],) * 2, eps_rad=(NOM["eps_rad"],) * 2, n_ticks=(NOM["n_ticks"],) * 2,
                        miss_p=(0.0, 0.0), cell_rad=tuple((1 / 527) if i == 1 else (1 / 550) for i in range(6)),
                        lag_ticks=NOM["lag_ticks"])
    return torch, mod.HandBlock(1, axes, 1, [2, 3, 4, 5], torch.tensor(Q_OPEN, dtype=torch.float32),
                                torch.tensor(Q_GRIP, dtype=torch.float32), p, "cpu")


@pytest.mark.parametrize("stop_at,lag", [(0.30, 5), (0.12, 2), (0.45, 6)])
def test_the_numpy_block_matches_the_sim_torch_block_tick_by_tick(stop_at, lag):
    torch, tb = _torch_block()
    b = _blk()
    qs, qm = _scenario(stop_at=stop_at, lag=lag)
    qs = qs + [qs[-1] - 0.04 * k * np.sign(Q_GRIP - Q_OPEN) for k in range(1, 6)]   # 끝에 여는 구간
    qm = qm + [qm[-1]] * 5
    prev = Q_OPEN.copy()
    for s, q in zip(qs, qm):
        want = tb.step(torch.tensor(prev, dtype=torch.float32).unsqueeze(0),
                       torch.tensor(q, dtype=torch.float32).unsqueeze(0))[0].numpy()
        got = b.step(prev, q)
        assert got.tolist() == want.tolist()
        assert b.grasped() == bool(tb.grasped()[0])
        prev = s


# ---------------------------------------------------------------- 계약 · 디코더 · 체인(rh_aglt)
from policy_control import rh_aglt as A  # noqa: E402

RUN = Path(__file__).resolve().parents[2] / "deploy/policies/rh56f1/aglt/right_rg3mft"
BLOCK = dict(NOM)


@pytest.fixture(scope="module")
def cb():
    c = A.load_contract(RUN / "rh_aglt_contract.json")
    return A.select_object(A.with_run(c, hand_block=BLOCK, grasp_rule="thumb_flex_and_any_finger", hand_lead_cap_rad=0.2,
                                      actor_tactile_obs=False, obs_dim=94), "source240_real")


def test_the_block_contract_validates_and_refuses_half_settings(cb):
    assert cb.hand_block == BLOCK and cb.grasp_rule
    with pytest.raises(A.RhAgltError, match="HAND_MAX_LEAD"):
        A.with_run(cb, hand_lead_cap_rad=0.3)
    with pytest.raises(A.RhAgltError, match="grasp_rule"):
        A.with_run(cb, grasp_rule="any")
    with pytest.raises(A.RhAgltError, match="grasp_rule"):
        A.with_run(cb, hand_block={}, hand_lead_cap_rad=0.0)
    old = A.load_contract(RUN / "rh_aglt_contract.json")
    assert old.hand_block == {} and A.RaDecoder(old).grasped() is None   # 이전 계약은 촉각 멈춤 그대로


def _closing_run(cb, stop_frac=0.4, steps=90):
    """정책이 계속 쥐라고 하고(행동 +1), 실측은 지연 뒤 따라가다 엄지굽힘 · 검지 · 중지가 범위의 stop_frac 에서 막힌다."""
    dec = A.RaDecoder(cb)
    s = cb.side()
    lo, hi = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
    i_flex, i_idx, i_mid = (s.hand_joints.index(n) for n in ("r_hj_thumb_2", "r_hj_index_1", "r_hj_middle_1"))
    a = np.concatenate([np.zeros(7), np.ones(6)])
    hist, frozen_at = [lo.copy()], None
    for t in range(steps):
        q = hist[max(0, len(hist) - 1 - 4)].copy()
        for i in (i_flex, i_idx, i_mid):
            stop = lo[i] + stop_frac * (hi[i] - lo[i])
            q[i] = min(q[i], stop) if hi[i] > lo[i] else max(q[i], stop)
        before = dec.state["arm"].hand_target.copy()
        dec.step(a, active=True, hand_q=q)
        after = dec.state["arm"].hand_target
        if frozen_at is None and dec.grasped():
            frozen_at = t
        hist.append(after.copy())
    return dec, frozen_at, (before, after), (i_flex, i_idx, i_mid)


def test_the_decoder_freezes_the_blocked_fingers_and_reports_the_grasp(cb):
    with pytest.raises(A.RhAgltError, match="hand_q"):
        A.RaDecoder(cb).step(np.zeros(13), active=True)
    dec, frozen_at, (before, after), axes = _closing_run(cb)
    assert frozen_at is not None and dec.grasped()
    assert np.allclose(after[list(axes)], before[list(axes)])           # 막힌 축은 닫는 쪽으로 더 안 간다
    dec.reset()
    assert dec.grasped() is False


def test_the_chain_uses_the_block_grasp_for_goal_success(cb, monkeypatch):
    from policy_control.pour_fj_node import AgltChain

    class P:
        def forward(self, obs):
            return np.concatenate([np.zeros(7), np.ones(6)])

    ch = AgltChain(cb, P())
    ch.object_name = "source240_real"
    s = cb.side()
    m = A.RaMeas(arm_q=np.asarray(s.arm_home), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, s.hand_open)),
                 palm_pos=np.array([0.2, -0.16, 0.3]), palm_R=np.eye(3), tips=np.tile([0.25, -0.16, 0.28], (5, 1)),
                 cup_pos=np.array([0.35, -0.16, 0.29]), cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.full(5, 9.0))
    ch.reset({"arm": m})
    seen = []
    monkeypatch.setattr(ch.goals, "step", lambda *a, is_grasped=None, **k: seen.append(is_grasped))
    for _ in range(cb.hold_steps + 3):
        ch.step({"arm": m})
    assert seen and all(g is False for g in seen)                       # 손끝 촉각 9 N 이어도 막힘이 아니면 쥔 게 아니다


def test_the_builder_reads_the_block_from_the_run_env():
    env = {"hand_block_enable": True, "hand_block_g_rad": [0.0145, 0.0273], "hand_block_eps_rad": [0.0073, 0.0182],
           "hand_block_n_ticks": [2, 4], "hand_block_lag_ticks": 6, "hand_firmware_stop_enable": False,
           "hand_lead_cap_rad": 0.2}
    block, rule, cap = A.hand_block_of(env)
    assert block == BLOCK and rule == "thumb_flex_and_any_finger" and cap == 0.2
    assert A.hand_block_of({}) == ({}, "", 0.0)                                   # 이전 런
    with pytest.raises(A.RhAgltError, match="firmware_stop"):
        A.hand_block_of(dict(env, hand_firmware_stop_enable=True))
    with pytest.raises(A.RhAgltError, match="명목값"):
        A.hand_block_of(dict(env, hand_block_g_rad=[0.0145, 0.018]))             # 배포 G 12 칸을 sim 이 학습하지 않았다

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


def _torch_block(g: float = NOM["g_rad"], **extra):
    """hdgp HandBlock(DR 끔). extra = 10.14 위치제어 키(g_axis_rad (H,) · release_on_open)."""
    torch = pytest.importorskip("torch")
    if not HDGP_HB.is_file():
        pytest.skip("hdgp hand_block.py 없음")
    spec = importlib.util.spec_from_file_location("hdgp_hand_block", HDGP_HB)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod                    # dataclass 가 모듈을 찾는다
    spec.loader.exec_module(mod)
    if extra and "g_axis_rad" not in mod.BlockParams.__dataclass_fields__:
        pytest.skip("hdgp hand_block.py 가 10.14 이전(g_axis_rad 없음)")
    axes = torch.tensor([False, True, True, True, True, True])
    p = mod.BlockParams(g_rad=(g,) * 2, eps_rad=(NOM["eps_rad"],) * 2, n_ticks=(NOM["n_ticks"],) * 2,
                        miss_p=(0.0, 0.0), cell_rad=tuple((1 / 527) if i == 1 else (1 / 550) for i in range(6)),
                        lag_ticks=NOM["lag_ticks"], **extra)
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

RUN = Path(__file__).resolve().parents[2] / "deploy/policies_retired/rh56f1/aglt/right_rg3wo"   # 10.18 rg3mft 폴더 삭제 — 같은 다물체 rg3 계열 배포 정책
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
    old = A.load_contract(Path(__file__).resolve().parent / "fixtures" / "rh_aglt_right_i03_contract.json")   # 09.30 i03 사본
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


def test_a_position_contact_judge_run_clamps_g_to_its_trained_range():
    """10.18 rg3wg: contact 판정 위치제어 런(G 7 칸 고정)은 pos 판정 런처럼 G 를 학습 범위 끝으로 붙인다. 어드민턴스면 그대로 거부."""
    env = {"hand_block_enable": True, "hand_block_g_rad": [0.0128, 0.0128], "hand_block_eps_rad": [0.0109, 0.0109],
           "hand_block_n_ticks": [3, 3], "hand_block_lag_ticks": 6, "hand_firmware_stop_enable": False,
           "hand_grasp_judge": "contact", "hand_adm_enable": False}
    block, _rule, _cap = A.hand_block_of(env)
    assert block["g_rad"] == 0.0128 and block["eps_rad"] == 0.0109 and block["n_ticks"] == 3
    with pytest.raises(A.RhAgltError, match="명목값"):
        A.hand_block_of(dict(env, hand_adm_enable=True))
    with pytest.raises(A.RhAgltError, match="명목값"):
        A.hand_block_of(dict(env, hand_block_eps_rad=[0.02, 0.03]))               # G 말고는 붙이지 않는다


# ---------------------------------------------------------------- 10.14 위치제어 블록(축별 G · 여는 쪽으로 안 풀림)
#: grasping aglt_r_rg3hp3 env(hand_block_pos): 엄지굽힘 15 칸 · 검지 · 중지 35 칸, 약지 · 새끼 = env 의 g_rad
GAX = {"thumb_flex": 0.0285, "index": 0.0636, "middle": 0.0636, "ring": 0.0, "pinky": 0.0}
GAX_H = (0.0, 0.0285, 0.0636, 0.0636, 0.0, 0.0)          # NAMES 순(thumb_1 은 대상 축 아님)
POS_ENV = {"hand_block_enable": True, "hand_block_g_rad": [0.0073, 0.0182], "hand_block_eps_rad": [0.0073, 0.0182],
           "hand_block_n_ticks": [2, 4], "hand_block_lag_ticks": 6, "hand_firmware_stop_enable": False,
           "hand_lead_cap_rad": 0.2, "hand_block_g_axis_rad": [0.0285, 0.0636, 0.0636, 0.0, 0.0],
           "hand_block_release_on_open": False, "hand_block_hold_lead_rad": [0.0, 0.0]}


@pytest.mark.parametrize("stop_at,lag", [(0.30, 5), (0.12, 2), (0.45, 6)])
@pytest.mark.parametrize("release_on_open", [False, True])
def test_the_numpy_block_matches_the_sim_with_axis_thresholds(stop_at, lag, release_on_open):
    torch, tb = _torch_block(g=0.0182, g_axis_rad=GAX_H, release_on_open=release_on_open)
    b = _blk(g_rad=0.0182, g_axis_rad=GAX, release_on_open=release_on_open)
    qs, qm = _scenario(stop_at=stop_at, lag=lag)
    qs = qs + [qs[-1] - 0.01 * k * np.sign(Q_GRIP - Q_OPEN) for k in range(1, 9)]   # 끝에 천천히 여는 구간
    qm = qm + [qm[-1]] * 8
    prev = Q_OPEN.copy()
    seen = np.zeros(6, bool)
    for s, q in zip(qs, qm):
        want = tb.step(torch.tensor(prev, dtype=torch.float32).unsqueeze(0),
                       torch.tensor(q, dtype=torch.float32).unsqueeze(0))[0].numpy()
        got = b.step(prev, q)
        assert got.tolist() == want.tolist()
        assert b.grasped() == bool(tb.grasped()[0])
        seen |= got
        prev = s
    assert seen[[1, 2, 3]].all()                          # 막힌 세 축이 실제로 걸렸다(비교가 빈 시험이 아니다)


def test_the_axis_threshold_latches_the_index_later_than_the_common_g():
    qs, qm = _scenario(stop_at=0.30, lag=0)
    first = {}
    for name, b in (("nominal", _blk(g_rad=0.0182)), ("axis", _blk(g_rad=0.0182, g_axis_rad=GAX))):
        prev = Q_OPEN.copy()
        for t, (s, q) in enumerate(zip(qs, qm)):
            if b.step(prev, q)[2] and name not in first:
                first[name] = t
            prev = s
    assert first["axis"] > first["nominal"]               # 검지 35 칸은 10 칸보다 늦게 — 실기에서 더 깊게 쥔다


def test_without_release_on_open_a_small_opening_keeps_the_latch():
    qs, qm = _scenario(stop_at=0.30, lag=0)
    sg = np.sign(Q_GRIP - Q_OPEN)
    tail = [qs[-1] - 0.002 * k * sg for k in range(1, 4)]  # 여는 쪽 1 칸씩 — 앞섬은 여전히 G/2 위
    for roo, kept in ((True, False), (False, True)):
        b = _blk(g_rad=0.0182, g_axis_rad=GAX, release_on_open=roo)
        prev = Q_OPEN.copy()
        for s, q in zip(qs + tail, qm + [qm[-1]] * 3):
            b.step(prev, q)
            prev = s
        assert bool(b.latch[2]) is kept


def test_the_block_contract_accepts_the_position_keys_and_refuses_bad_ones():
    assert _blk(g_axis_rad=GAX, release_on_open=False).g.tolist() == [0.0218, 0.0285, 0.0636, 0.0636, 0.0218, 0.0218]
    assert _blk().g.tolist() == [0.0218] * 6                                      # 10.12 계약 그대로
    with pytest.raises(HB.HandBlockError):
        _blk(g_axis_rad={"thumb_rot": 0.03})                                      # 모르는 축
    with pytest.raises(HB.HandBlockError):
        _blk(g_axis_rad={"index": -0.01})
    with pytest.raises(HB.HandBlockError):
        _blk(release_on_open="no")
    with pytest.raises(HB.HandBlockError):
        _blk(hold_lead_rad=0.01)                                                  # 모르는 키


def test_the_builder_reads_the_position_block_and_keeps_g_inside_the_trained_range():
    block, rule, cap = A.hand_block_of(POS_ENV)
    assert block == dict(BLOCK, g_rad=0.0182, g_axis_rad=GAX, release_on_open=False)   # 명목 12 칸 → 학습 끝 10 칸
    assert rule == "thumb_flex_and_any_finger" and cap == 0.2
    HB.HandBlockNp.from_contract(block, NAMES, Q_OPEN, Q_GRIP)                   # 계약으로 그대로 만들어진다
    with pytest.raises(A.RhAgltError, match="hold_lead"):
        A.hand_block_of(dict(POS_ENV, hand_block_hold_lead_rad=[0.0, 0.01]))      # 배포에 없는 유지 앞섬
    with pytest.raises(A.RhAgltError, match="g_axis"):
        A.hand_block_of(dict(POS_ENV, hand_block_g_axis_rad=[0.03, 0.06]))
    with pytest.raises(A.RhAgltError, match="명목값"):                           # 축별 G 없는 옛 런은 그대로 거부
        A.hand_block_of({k: v for k, v in POS_ENV.items() if k != "hand_block_g_axis_rad"})
    assert "release_on_open" not in A.hand_block_of(dict(POS_ENV, hand_block_release_on_open=True))[0]


# ---------------------------------------------------------------- 10.13 쥔 판정 유지(grasp_hold_rule)
HOLD_X = (0.0, 0.10, 0.10, 0.10, 0.10)


def _hold_scenario(jitter_every: int = 4, open_at: int = 70, steps: int = 110):
    """닫혀 막힌 뒤 q* 가 jitter_every 틱마다 여는 쪽으로 1 칸 흔들리고, open_at 부터 q* · 실측이 함께 연다."""
    qs, qm = _scenario(stop_at=0.30, steps=60, lag=3)
    sg = np.sign(Q_GRIP - Q_OPEN)
    star, q = qs[-1].copy(), qm[-1].copy()
    for t in range(60, steps):
        if t < open_at:
            s = star - (0.003 * sg if t % jitter_every == 0 else 0.0)
            qs.append(s), qm.append(q.copy())
        else:
            star = np.maximum(star - 0.05 * sg, Q_OPEN) if (sg > 0).all() else star
            q = np.where(sg * (q - star) > 0, star, q)              # 실측은 열린 목표를 바로 따라간다
            qs.append(star.copy()), qm.append(q.copy())
    return qs, qm


def _run_np(hold_x=HOLD_X, steps=5):
    b = _blk()
    h = HB.GraspHoldNp(b, hold_x, steps)
    raws, holds = [], []
    prev = Q_OPEN.copy()
    for s, q in zip(*_hold_scenario()):
        holds.append(h.step(b.grasped(), prev, q))
        b.step(prev, q)
        raws.append(b.grasped())
        prev = s
    return raws, holds


def test_the_hold_keeps_the_grasp_through_target_jitter_and_releases_when_the_fingers_open():
    raws, holds = _run_np()
    first = raws.index(True)
    held = holds[first + 1: 70]
    assert not all(raws[first:70])                       # 원 판정은 흔들림에 풀린다(고치려는 문제)
    assert all(held)                                     # 유지 판정은 그동안 계속 참
    assert holds[-1] is False                            # 손가락이 X 이상 열리면 풀린다


def test_the_hold_stays_when_the_fingers_open_less_than_x():
    _, holds = _run_np(hold_x=(0.0, 2.0, 2.0, 2.0, 2.0))
    assert holds[-1] is True


def test_the_hold_refuses_bad_settings_and_resets():
    b = _blk()
    for x, n in (((0.1,) * 4, 5), ((0.0, -0.1, 0.1, 0.1, 0.1), 5), (HOLD_X, 0)):
        with pytest.raises(HB.HandBlockError):
            HB.GraspHoldNp(b, x, n)
    h = HB.GraspHoldNp(b, HOLD_X, 5)
    h.hold = True
    h.reset()
    assert h.hold is False and h.cnt == 0


def test_the_numpy_hold_matches_the_sim_torch_grasp_hold_tick_by_tick():
    torch, tb = _torch_block()
    mod = sys_modules_hdgp()
    if not hasattr(mod, "GraspHold"):
        pytest.skip("hdgp GraspHold 없음")
    th = mod.GraspHold(tb, torch.tensor([0.0, *HOLD_X], dtype=torch.float32), 5)
    b = _blk()
    h = HB.GraspHoldNp(b, HOLD_X, 5)
    f = lambda v: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    qs, qm = _hold_scenario()
    prev, sim_hold = Q_OPEN.copy(), False
    for k, (s, q) in enumerate(zip(qs, qm)):
        got = h.step(b.grasped(), prev, q)               # 배포: 이번 틱 block.step 전
        if k:
            assert got == sim_hold
        b.step(prev, q)
        tb.step(f(prev), f(q))                           # sim tick k: pre_physics 막힘 → 물리 → 쥔 판정(q_{k+1}, q*_k)
        nxt = qm[k + 1] if k + 1 < len(qm) else q
        sim_hold = bool(th.step(tb.grasped(), f(s), f(nxt))[0])
        prev = s


def sys_modules_hdgp():
    import sys
    return sys.modules["hdgp_hand_block"]


def test_the_contract_and_decoder_use_the_hold_when_the_run_env_asks():
    env = {"grasp_hold_rule": "block_follow_open", "grasp_release_open_rad": list(HOLD_X), "grasp_release_steps": 5}
    assert A.grasp_hold_of({}, BLOCK) == {} and A.grasp_hold_of({"grasp_hold_rule": "none"}, BLOCK) == {}
    with pytest.raises(A.RhAgltError, match="hand_block_enable"):
        A.grasp_hold_of(env, {})
    c = A.load_contract(RUN / "rh_aglt_contract.json")
    hb = A.replace(c, hand_block=dict(BLOCK), grasp_rule=HB.GRASP_RULES[0], hand_lead_cap_rad=0.2,
                   **A.grasp_hold_of(env, BLOCK))
    A.validate(hb)
    dec = A.RaDecoder(hb)
    assert dec.hold is not None and dec.grasped() is False
    with pytest.raises(A.RhAgltError, match="grasp_hold_rule"):
        A.validate(A.replace(hb, grasp_hold_rule="magic"))
    with pytest.raises(A.RhAgltError):
        A.validate(A.replace(hb, grasp_release_open_rad=[0.1] * 4))

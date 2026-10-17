"""10.14 쥔 규칙 latched_axes_at_least — 배포 numpy 판이 hdgp hand_block_pos(HandBlockOpp · HandBlockPos)와 틱마다 같다.

opp(locking capg_j10, wt_capsplit) = 네 손가락 중 래치 k 개 이상(엄지굽힘 제외) AND 캡-손바닥 < hand_grasp_near_m.
pos(grasping, hdgp main) = 판정 축(hand_block_judge_axes) 중 래치 k 개 이상.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from policy_control import hand_block as HB
from policy_control import rh_aglt as A

TASK = "source/openarm/openarm/agnostic/tasks/rh_aglt_r"
SIM_OPP = Path.home() / "rl_ws/wt_capsplit" / TASK          # capg_j10 이 학습하는 코드
SIM_POS = Path.home() / "rl_ws/hdgp" / TASK
Q_OPEN = np.zeros(6)
Q_GRIP = np.array([1.2, 0.5, 1.4, 1.4, 1.4, 1.4])
NAMES = ["r_hj_thumb_1", "r_hj_thumb_2", "r_hj_index_1", "r_hj_middle_1", "r_hj_ring_1", "r_hj_pinky_1"]
NOM = {"g_rad": 0.0182, "eps_rad": 0.0109, "n_ticks": 3, "lag_ticks": 6,
       "axes": ["thumb_flex", "index", "middle", "ring", "pinky"]}
FOUR = ["index", "middle", "ring", "pinky"]
#: NAMES 색인: 1 엄지굽힘 · 2 검지 · 3 중지 · 4 약지 · 5 새끼
STOPS = {"index_only": {2: 0.30}, "thumb_only": {1: 0.20}, "thumb_index": {1: 0.20, 2: 0.35},
         "index_middle_late": {2: 0.30, 3: 0.55}, "none": {}}


def _sim_module(root: Path, tag: str):
    """hand_block_pos 는 `from .hand_block import` 를 쓴다 — 가짜 패키지로 isaaclab 없이 두 파일만 읽는다."""
    pytest.importorskip("torch")
    if not (root / "hand_block_pos.py").is_file():
        pytest.skip(f"{root} 에 hand_block_pos.py 없음")
    pkg = f"_sim_{tag}"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(root)]
        sys.modules[pkg] = m
    return importlib.import_module(f"{pkg}.hand_block_pos")


def _sim_block(mod, cls: str, **kw):
    import torch
    if not hasattr(mod, cls):
        pytest.skip(f"{cls} 없음")
    hb = sys.modules[mod.__name__.rsplit(".", 1)[0] + ".hand_block"]
    p = hb.BlockParams(g_rad=(NOM["g_rad"],) * 2, eps_rad=(NOM["eps_rad"],) * 2, n_ticks=(NOM["n_ticks"],) * 2,
                       miss_p=(0.0, 0.0), cell_rad=tuple((1 / 527) if i == 1 else (1 / 550) for i in range(6)),
                       lag_ticks=NOM["lag_ticks"])
    axes = torch.tensor([False, True, True, True, True, True])
    return torch, getattr(mod, cls)(1, axes, 1, [2, 3, 4, 5], torch.tensor(Q_OPEN, dtype=torch.float32),
                                    torch.tensor(Q_GRIP, dtype=torch.float32), p, "cpu", **kw)


def _np_block(judge_axes, k):
    return HB.HandBlockNp.from_contract(NOM, NAMES, Q_OPEN, Q_GRIP, HB.GRASP_RULES[1], judge_axes, k)


def _scenario(stops: dict, steps: int = 70, rate: float = 0.025, lag: int = 4):
    """q* 가 스텝당 rate 로 닫히고 실측은 lag 틱 뒤 따라간다. stops 의 축은 그 각도에서 막힌다. 끝 8 틱은 연다."""
    sg = np.sign(Q_GRIP - Q_OPEN)
    qs, qm, hist, star = [], [], [Q_OPEN.copy()], Q_OPEN.copy()
    for _ in range(steps):
        star = np.minimum(star + rate * sg, Q_GRIP)
        hist.append(star.copy())
        q = hist[max(0, len(hist) - 1 - lag)].copy()
        for i, v in stops.items():
            q[i] = min(q[i], v)
        qs.append(star.copy()), qm.append(q)
    qs += [qs[-1] - 0.03 * k * sg for k in range(1, 9)]
    qm += [qm[-1]] * 8
    return qs, qm


def _compare(torch, tb, b, stops):
    qs, qm = _scenario(stops)
    f = lambda v: torch.tensor(v, dtype=torch.float32).unsqueeze(0)  # noqa: E731
    prev, seen = Q_OPEN.copy(), []
    for s, q in zip(qs, qm):
        want = tb.step(f(prev), f(q))[0].numpy()
        got = b.step(prev, q)
        assert got.tolist() == want.tolist()
        assert b.grasped() == bool(tb.grasped()[0])
        seen.append(b.grasped())
        prev = s
    return seen


@pytest.mark.parametrize("name", list(STOPS))
def test_opp_matches_the_capg_j10_sim_tick_by_tick(name):
    mod = _sim_module(SIM_OPP, "opp")
    torch, tb = _sim_block(mod, "HandBlockOpp", min_axes=1)
    seen = _compare(torch, tb, _np_block(FOUR, 1), STOPS[name])
    finger_stops = any(i >= 2 for i in STOPS[name])
    assert any(seen) is finger_stops                       # 엄지굽힘만 막히면 쥔 것이 아니다(opp)


@pytest.mark.parametrize("name", list(STOPS))
@pytest.mark.parametrize("k", [1, 2])
def test_opp_with_two_fingers_matches_too(name, k):
    mod = _sim_module(SIM_OPP, "opp")
    torch, tb = _sim_block(mod, "HandBlockOpp", min_axes=k)
    _compare(torch, tb, _np_block(FOUR, k), STOPS[name])


@pytest.mark.parametrize("name", list(STOPS))
def test_pos_matches_the_hdgp_sim_tick_by_tick(name):
    mod = _sim_module(SIM_POS, "pos")
    import inspect
    kw = {"min_axes": 2}
    axes = ["thumb_flex", *FOUR]
    if "judge_axes" in inspect.signature(mod.HandBlockPos.__init__).parameters:
        kw["judge_axes"] = [1, 2, 3]                        # 엄지굽힘 · 검지 · 중지(hand_block_judge_axes)
        axes = ["thumb_flex", "index", "middle"]
    torch, tb = _sim_block(mod, "HandBlockPos", **kw)
    _compare(torch, tb, _np_block(axes, 2), STOPS[name])


def test_the_judge_refuses_half_settings():
    with pytest.raises(HB.HandBlockError, match="min_axes"):
        _np_block(FOUR, 0)
    with pytest.raises(HB.HandBlockError, match="min_axes"):
        _np_block(FOUR, 5)
    with pytest.raises(HB.HandBlockError, match="부분집합"):
        _np_block(["index", "index"], 1)
    with pytest.raises(HB.HandBlockError, match="부분집합"):
        _np_block(["wrist"], 1)
    with pytest.raises(HB.HandBlockError, match="받지 않는다"):
        HB.HandBlockNp.from_contract(NOM, NAMES, Q_OPEN, Q_GRIP, HB.GRASP_RULES[0], FOUR, 1)
    with pytest.raises(HB.HandBlockError, match="쥔 규칙"):
        HB.HandBlockNp.from_contract(NOM, NAMES, Q_OPEN, Q_GRIP, "magic")


# ---------------------------------------------------------------- 빌더(hdgp env.yaml → 계약 키)
OPP_ENV = {"hand_block_enable": True, "hand_block_g_rad": [0.0073, 0.0182], "hand_block_eps_rad": [0.0073, 0.0182],
           "hand_block_n_ticks": [2, 4], "hand_block_lag_ticks": 6, "hand_firmware_stop_enable": False,
           "hand_lead_cap_rad": 0.2, "hand_grasp_judge": "opp", "hand_block_min_axes": 1, "hand_grasp_near_m": 0.08}


def test_the_builder_reads_opp_and_keeps_g_inside_the_trained_range():
    block, rule, cap = A.hand_block_of(OPP_ENV)
    assert block["g_rad"] == 0.0182 and "g_axis_rad" not in block   # 명목 12 칸 → 학습 끝 10 칸(capg_j10 은 축별 G 없음)
    assert A.grasp_judge_of(OPP_ENV, block) == {"grasp_rule": "latched_axes_at_least", "grasp_judge_axes": FOUR,
                                                "grasp_min_axes": 1, "grasp_near_m": 0.08}
    old = {k: v for k, v in OPP_ENV.items() if k not in ("hand_grasp_judge", "hand_grasp_near_m")}
    with pytest.raises(A.RhAgltError, match="명목값"):
        A.hand_block_of(old)                                       # 옛 판정 런은 G 범위 밖이면 그대로 거부


def test_the_builder_reads_pos_axes_and_refuses_what_the_deploy_cannot_do():
    env = dict(OPP_ENV, hand_grasp_judge="pos", hand_block_min_axes=2, hand_grasp_near_m=0.0,
               hand_block_judge_axes=["thumb_2", "index", "middle"])
    block, _, _ = A.hand_block_of(env)
    assert A.grasp_judge_of(env, block)["grasp_judge_axes"] == ["thumb_flex", "index", "middle"]
    assert A.grasp_judge_of({k: v for k, v in env.items() if k != "hand_block_judge_axes"}, block)["grasp_judge_axes"] \
        == ["thumb_flex", *FOUR]
    with pytest.raises(A.RhAgltError, match="judge_axes"):
        A.grasp_judge_of(dict(env, hand_block_judge_axes=["thumb_1"]), block)
    with pytest.raises(A.RhAgltError, match="거리 조건"):
        A.grasp_judge_of(dict(env, hand_grasp_judge="block", hand_grasp_near_m=0.08), block)
    with pytest.raises(A.RhAgltError, match="hand_block_enable"):
        A.grasp_judge_of(env, {})
    assert A.grasp_judge_of(dict(env, hand_grasp_judge="contact"), block) == {}   # rg3hp3 계열 = 옛 규칙 그대로


# ---------------------------------------------------------------- 계약 · 디코더(거리 조건)
RUN = Path(__file__).resolve().parents[2] / "deploy/policies/rh56f1/aglt/right_rg3wo"   # 10.18 rg3mft 폴더 삭제 — 같은 다물체 rg3 계열 배포 정책


@pytest.fixture(scope="module")
def co():
    c = A.load_contract(RUN / "rh_aglt_contract.json")
    block, _, cap = A.hand_block_of(OPP_ENV)
    j = A.grasp_judge_of(OPP_ENV, block)
    return A.select_object(A.with_run(c, hand_block=block, hand_lead_cap_rad=cap, actor_tactile_obs=False,
                                      obs_dim=94, **j), "source240_real")


def _close(dec, c, obj_palm_m, steps=80):
    """행동 +1 로 닫고, 실측 검지만 범위 0.3 에서 막힌다(엄지굽힘 · 다른 손가락은 따라감)."""
    s = c.side()
    lo, hi = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
    i = s.hand_joints.index("r_hj_index_1")
    a = np.concatenate([np.zeros(7), np.ones(6)])
    hist, seen = [lo.copy()], []
    for _ in range(steps):
        q = hist[max(0, len(hist) - 5)].copy()
        q[i] = lo[i] + min((q[i] - lo[i]) / (hi[i] - lo[i]), 0.3) * (hi[i] - lo[i])
        dec.step(a, active=True, hand_q=q, obj_palm_m=obj_palm_m)
        hist.append(dec.state["arm"].hand_target.copy())
        seen.append(dec.grasped())
    return seen


def test_the_decoder_counts_one_blocked_finger_only_near_the_palm(co):
    assert co.grasp_rule == "latched_axes_at_least" and co.grasp_near_m == 0.08
    assert any(_close(A.RaDecoder(co), co, 0.05))                    # 검지 하나 막힘 + 캡 5 cm → 쥠
    assert not any(_close(A.RaDecoder(co), co, 0.12))                # 같은 손, 캡 12 cm → 쥔 것 아님(허공 · 상판 막힘)
    with pytest.raises(A.RhAgltError, match="obj_palm_m"):
        A.RaDecoder(co).step(np.zeros(13), active=True, hand_q=np.asarray(co.side().hand_open, float))
    assert not any(_close(A.RaDecoder(co), co, float("nan"), steps=40))   # 거리 결손 = 쥔 것으로 치지 않는다


def test_the_contract_refuses_half_judge_settings(co, tmp_path):
    with pytest.raises(A.RhAgltError, match="min_axes"):
        A.with_run(co, grasp_min_axes=0)
    with pytest.raises(A.RhAgltError, match="받지 않는다"):
        A.with_run(co, grasp_rule="thumb_flex_and_any_finger")
    with pytest.raises(A.RhAgltError, match="grasp_near_m"):
        A.with_run(co, grasp_near_m=-0.1)
    with pytest.raises(A.RhAgltError, match="hand_block 과 같이"):
        A.with_run(co, hand_block={}, hand_lead_cap_rad=0.0, grasp_rule="", grasp_judge_axes=[], grasp_min_axes=0)
    f = tmp_path / "c.json"
    f.write_text(co.to_json())
    back = A.load_contract(f)                                         # 계약 파일로 저장 · 읽기 뒤에도 규칙이 남는다
    assert (back.grasp_rule, back.grasp_judge_axes, back.grasp_min_axes, back.grasp_near_m) == \
        ("latched_axes_at_least", FOUR, 1, 0.08)

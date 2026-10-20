"""10.12 hdgp actor_tactile_obs(3a1f897f) — actor 에서 손끝 촉각 5 칸을 뺀 계약(91 + 물체 one-hot)을 배포가 학습과 같은 순서로 만든다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_control import rh_aglt as A

REPO = Path(__file__).resolve().parents[2]
#: 10.18 옛 정책 폴더(rg3mft 등) 삭제 — 배포 촉각 없는 다물체 계약(rg3wo, 91 + 3)을 기준으로 둔다
RUN = REPO / "deploy" / "policies_retired" / "rh56f1" / "aglt" / "right_rg3wo"   # ★10.20 시험 고정본
OLD = Path(__file__).resolve().parent / "fixtures" / "rh_aglt_right_i03_contract.json"   # actor_tactile_obs 필드 전 계약(09.30 i03)
NAMES = ["source240_real", "source200_real", "shaker_c_real"]


@pytest.fixture(scope="module")
def nt():
    return A.load_contract(RUN / "rh_aglt_contract.json")


@pytest.fixture(scope="module")
def c(nt):
    """촉각을 가진 다물체 계약(10.12 rg3mft 와 같은 모양, 96 + 3) — rg3wo 에서 촉각만 켜서 만든다."""
    return A.with_run(nt, actor_tactile_obs=True, obs_dim=99)


def _meas(c, tactile):
    s = c.side()
    return A.RaMeas(arm_q=np.asarray(s.arm_home), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, s.hand_open)),
                    palm_pos=np.array([0.2, -0.16, 0.3]), palm_R=np.eye(3), tips=np.tile([0.25, -0.16, 0.28], (5, 1)),
                    cup_pos=np.array([0.35, -0.16, 0.29]), cup_quat=np.array([1.0, 0, 0, 0]),
                    tactile_n=np.asarray(tactile, float))


def _span(layout, name):
    i = 0
    for n, w in layout:
        if n == name:
            return i, i + w
        i += w
    raise KeyError(name)


def test_old_contracts_keep_the_tactile_and_96_plus_k(c):
    old = A.load_contract(OLD)
    assert old.actor_tactile_obs is True and old.obs_dim == 96          # 필드가 없는 계약 = 촉각 있음
    assert c.actor_tactile_obs is True and c.obs_dim == 99
    assert [n for n, _ in A.actor_layout(c)].count("tactile") == 1
    assert A.ACTOR_BASE_DIMS == (96, 91)


def test_a_no_tactile_contract_is_91_plus_k_and_drops_only_the_tactile(c, nt):
    assert nt.actor_tactile_obs is False and A.obs_dim_of(nt) == 94
    assert A.with_run(c, actor_tactile_obs=False, obs_dim=94) == nt
    assert [n for n, _ in A.actor_layout(nt)] == [n for n, _ in A.actor_layout(c) if n != "tactile"]
    assert A.actor_layout(nt)[-1] == ("object_id", 3)                # hdgp layout: 촉각을 뺀 base 뒤에 one-hot
    with pytest.raises(A.RhAgltError, match="obs/action"):
        A.with_run(c, actor_tactile_obs=False, obs_dim=99)


@pytest.mark.parametrize("name", NAMES)
def test_the_no_tactile_obs_is_the_full_obs_without_the_tactile_slice(c, nt, name):
    cc = A.select_object(c, name)
    nt = A.select_object(nt, name)
    m = _meas(cc, [3.0, 2.0, 1.0, 0.5, 0.2])
    goal = A.first_goal(cc, m.cup_pos, m.cup_quat)
    full = A.build_obs(cc, m, A.RaDecoder(cc), goal, np.zeros(13))
    lo, hi = _span(A.actor_layout(cc), "tactile")
    got = A.build_obs(nt, m, A.RaDecoder(nt), goal, np.zeros(13))
    assert got.size == 94 and got == pytest.approx(np.concatenate([full[:lo], full[hi:]]))
    other = A.build_obs(nt, _meas(cc, [0.0] * 5), A.RaDecoder(nt), goal, np.zeros(13))
    assert other == pytest.approx(got)                               # 촉각이 actor 관측을 바꾸지 않는다


def test_a_no_tactile_run_is_recognised_and_its_flag_is_read():
    env = yaml.unsafe_load((RUN / "params" / "env.yaml").read_text())
    env = env.get("env", env)
    assert A.is_rh_aglt_run(env) and A._actor_tactile(env) is False and A._base_dim(env) == 91
    assert A._actor_tactile(dict(env, actor_tactile_obs=True)) is True
    nt = dict(env, actor_tactile_obs=False, observation_space=91)
    assert A.is_rh_aglt_run(nt) and A._actor_tactile(nt) is False and A._base_dim(nt) == 91
    assert A.is_rh_aglt_run(dict(nt, observation_space=94))         # 해석 뒤 값(+물체 one-hot)도
    assert A.is_rh_aglt_run(dict(nt, observation_space=96))         # 10.13 rg3s: hdgp 는 해석 전 cfg 기본값 96 을 덤프한다
    assert not A.is_rh_aglt_run(dict(nt, observation_space=97))     # 그 밖의 폭은 거부

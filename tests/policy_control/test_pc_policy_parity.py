"""policy_parity: bag obs → 배포 로더 재현 → 기록 행동 대조(합성 + 실제 bag 있으면)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "deploy" / "policy_control" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import policy_parity as PP  # noqa: E402
from policy_control import codec  # noqa: E402


class _Gru:
    """은닉 상태 하나를 이어 가는 가짜 정책 — reset 을 안 하거나 틱을 건너뛰면 행동이 달라진다."""

    def __init__(self):
        self.h = 0.0

    def reset(self):
        self.h = 0.0

    def forward(self, o):
        self.h = 0.5 * self.h + float(np.sum(o))
        return np.array([self.h, -self.h])


def _msgs(rows, seqs, label):
    return [(float(s), codec.encode_float_array(np.asarray(r, float), [label], [len(r)], int(s))) for r, s in zip(rows, seqs)]


def test_pair_by_seq_and_first_mismatch():
    O = np.arange(10, dtype=float).reshape(5, 2) / 10
    pol = _Gru()
    U = PP.replay(pol, O, 1.0)
    act = _msgs(U[[0, 1, 3, 4]], [1, 2, 4, 5], "action")    # seq 3 은 행동 없음(settling 같은 틱)
    seq, O2, U2 = PP.pair(_msgs(O, [1, 2, 3, 4, 5], "obs"), act)
    assert seq.tolist() == [1, 2, 4, 5] and np.allclose(O2, O[[0, 1, 3, 4]])
    good = PP.compare(U, U)
    assert good["max_abs"] == 0.0 and good["first_mismatch"] is None
    bad = U.copy()
    bad[3, 1] += 1e-2
    assert PP.compare(U, bad)["first_mismatch"] == 3
    pol.forward(O[0])                                          # 은닉 상태가 남아 있어도 replay 는 reset 부터
    assert np.array_equal(PP.replay(pol, O, 1.0), U)


EP = ROOT / "logs/runs/261116/episode/150115_cocktail_1c_n07b"


@pytest.mark.skipif(not EP.is_dir(), reason="실기 회차 261116 없음")
def test_real_bag_parity_cocktail_n07b():
    import run_bag as RB
    import sim_real_compare as C
    try:
        bag = RB.find_bag(EP, RB.default_roots(EP))
    except FileNotFoundError:
        pytest.skip("bag 없음")
    ep = C.episode(EP, [bag.parent])
    par = [n["parity"] for n in ep["nodes"] if n["stage"] in ("grasp", "place")]
    assert len(par) == 4
    for p in par:
        assert p["n"] > 0 and p["max_abs"] <= PP.TOL and p["first_mismatch"] is None

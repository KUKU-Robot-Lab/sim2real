"""grasp_s2r 을 상속한 다른 과제는 같은 family 로 판정된다 — 빌더는 차원으로 걸러 **이유를 말해야** 한다.

09.21: `grasp_fj_t2r`(obs 133 / act 26) 덤프가 grasp_s2r 로 판정돼 'no value for joint r_hj_thumb_1'
에서 우연히 멈췄다. 그 관절이 있는 자산이었다면 더 깊이 들어갔을 것이다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from policy_control import contract_build as B

ENV = Path("x/params/env.yaml")


def test_matching_dims_pass():
    B._refuse_other_interface("observation_space: 155\naction_space: 21\n", ENV, obs=155, act=21)


def test_other_dims_are_refused_with_both_numbers():
    with pytest.raises(SystemExit) as e:
        B._refuse_other_interface("observation_space: 133\naction_space: 26\n", ENV, obs=155, act=21)
    msg = str(e.value)
    assert "133" in msg and "26" in msg and "155" in msg and "21" in msg and "no contract builder" in msg


@pytest.mark.parametrize("text", ["", "action_space: 21\n", "observation_space: 155\n"])
def test_an_undeclared_dim_is_left_to_the_later_checks(text):
    B._refuse_other_interface(text, ENV, obs=155, act=21)       # 모르면 막지 않는다 — 뒤의 검증이 본다


def test_nested_keys_do_not_count():
    B._refuse_other_interface("cfg:\n  observation_space: 133\n  action_space: 26\n", ENV, obs=155, act=21)


def test_the_guard_runs_before_any_joint_lookup():
    src = Path(B.__file__).read_text()
    body = src[src.index("def _build_right"):]
    assert body.index("_refuse_other_interface(") < body.index("_home_values(")

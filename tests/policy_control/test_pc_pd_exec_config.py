"""실기 발행 사본 pd_dg5f_m_short_exec.yaml — 원본과 `execute` 한 줄만 다르다.

pd_node 의 발행 = launch execute:=true AND yaml execute. 원본이 false 라 launch 만으로는 켜지지 않았고(09.22 실기),
그래서 발행 전용 사본을 두었다. 사본이 원본과 어긋나기 시작하면(게인·리미터를 한쪽만 고친다) 무발행 점검이
발행 때의 설정을 대표하지 못한다 — 그것을 막는다.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "config"


def test_the_exec_copy_differs_from_the_real_config_only_in_execute():
    base = yaml.safe_load((CFG / "pd_dg5f_m_short.yaml").read_text(encoding="utf-8"))
    exe = yaml.safe_load((CFG / "pd_dg5f_m_short_exec.yaml").read_text(encoding="utf-8"))
    assert base["execute"] is False and exe["execute"] is True
    assert {k: v for k, v in base.items() if k != "execute"} == {k: v for k, v in exe.items() if k != "execute"}

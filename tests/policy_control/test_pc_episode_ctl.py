"""episode_ctl 의 일부 실행(--only · --hold-s) — 콘솔 미션의 engage → 제자리 → 홈 이 이 옵션에 기댄다.

09.22: 미션이 `--steps 0` 으로 불렀는데 episode_ctl 은 그 값을 거부한다 — goto_home 단계는 처음부터 돌 수 없었다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools" / "episode_ctl.py"
_spec = importlib.util.spec_from_file_location("episode_ctl", _PATH)
ec = importlib.util.module_from_spec(_spec)
sys.modules["episode_ctl"] = ec             # dataclass 가 제 모듈을 sys.modules 에서 찾는다
_spec.loader.exec_module(ec)


def test_only_keeps_the_declared_order_whatever_the_argument_order():
    assert [s.id for s in ec.selected(("pd_goto_home", "pd_engage"))] == ["pd_engage", "pd_goto_home"]
    assert "pd_hand_home" not in [s.id for s in ec.selected(())]          # --only 로만 부르는 단계
    assert [s.id for s in ec.selected(("pd_hand_home",))] == ["pd_hand_home"]
    with pytest.raises(KeyError):
        ec.selected(("nope",))


def test_only_asks_approval_just_for_the_real_stages_it_runs():
    assert ec.missing_approvals(frozenset(), ("pd_engage",)) == ["pd_engage"]
    assert ec.missing_approvals(frozenset({"pd_engage"}), ("pd_engage",)) == []
    assert ec.missing_approvals(frozenset(), ("ep_stop",)) == []           # 실기를 움직이지 않는 단계
    assert "ep_start" in ec.missing_approvals(frozenset({"pd_engage", "pd_goto_home"}))   # 전부 부르면 전부 필요


def test_hold_passes_only_while_pd_keeps_the_arm():
    assert ec.hold_ok({"phase": "TRACKING"}) and ec.hold_ok({"phase": "RAMPING"})
    assert not ec.hold_ok({"phase": "HOLD"}) and not ec.hold_ok({"phase": "IDLE"}) and not ec.hold_ok(None)


def test_cli_refuses_a_hold_without_engage_and_unknown_stages(capsys):
    assert ec.main(["--only", "pd_goto_home", "--hold-s", "5"]) == 2
    assert ec.main(["--only", "nope"]) == 2
    assert ec.main(["--only", "pd_engage", "--hold-s", "10"]) == 0       # 계획만 — 서비스를 부르지 않는다
    assert "DRY RUN" in capsys.readouterr().out
    assert ec.main(["--only", "pd_engage", "--execute"]) == 3            # 승인 없이는 실행하지 않는다

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


def test_a_lost_response_is_told_apart_from_a_refusal_and_idempotent_pd_calls_are_retried():
    """09.28 실기: goto_home 응답이 유실돼 45 s 뒤 실패 → pd 해제 → JTC 가 넘겨받는 사이 팔이 움직였다."""
    from episode_ctl import RETRY_ON_LOST, keep_engaged_after_lost, lost_response

    assert lost_response(["service /policy_control/pd_left/goto_home timeout"])
    assert not lost_response(["service /policy_control/pd_left/goto_home unavailable"])
    assert not lost_response(["phase IDLE is not RAMPING/TRACKING (engage first)"])
    assert {"pd_goto_home", "pd_hand_home"} <= RETRY_ON_LOST and "pd_engage" not in RETRY_ON_LOST
    assert keep_engaged_after_lost({"phase": "TRACKING", "ok": True})
    assert not keep_engaged_after_lost({"phase": "HOLD", "ok": False})
    assert not keep_engaged_after_lost(None)


def test_skip_engaged_only_skips_a_pd_that_already_holds_the_arm():
    """10.04 rehome: 정책이 멈춘 뒤 pd 는 그 자리를 붙들고 있다(TRACKING) — engage 는 IDLE 에서만 받으니 건너뛴다."""
    from types import SimpleNamespace as NS
    for phase, want in (("TRACKING", True), ("RAMPING", True), ("IDLE", False), ("HOLD", False)):
        assert ec.already_engaged(NS(pd_status={"phase": phase}, spin=lambda s: None), wait_s=0.0) is want
    assert ec.already_engaged(NS(pd_status=None, spin=lambda s: None), wait_s=0.05) is False   # status 를 못 받으면 engage 한다

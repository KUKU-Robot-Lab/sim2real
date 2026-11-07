"""★11.07 continue_on_fail — 놓기가 안전 코드 외 이유(시간 초과 포함)로 실패하면 멈추지 않고 다음 노드(home)로 넘어간다
(사용자 "어차피 place 이후엔 실패하든 성공하든 그냥 home 자세로 가는게 맞는데"). 손이 컵에 걸린 놓기는 기존대로 실패."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from policy_control import episode_spec as S
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_failure import F, NodeResult
from policy_control.episode_runner import FAILURE, SUCCESS, EpisodeManager

EPISODES = Path(__file__).resolve().parents[2] / "config" / "episodes"


def _ep(**over):
    return S.parse({**yaml.safe_load((EPISODES / "pick_place_right.yaml").read_text()), **over})


class _Ex(FakeExecutor):
    """res=(status, reason, signals) 를 주면 place_cup 정책 결과를 그것으로 바꾼다."""

    def __init__(self, ep, inject=None, res=None):
        super().__init__(ep, inject)
        self.res = res

    def run_policies(self, node, plans, world):
        out = super().run_policies(node, plans, world)
        if self.res is not None and node.id == "place_cup":
            out[0] = NodeResult(self.res[0], self.res[1], 1.0, out[0].role, dict(self.res[2]))
        return out


def _run(ep, ex):
    rows = []
    m = EpisodeManager(ep, ex, approve=lambda w, y: True, log=rows.append)
    return m, m.run(), rows


def test_place_timeout_continues_to_home():
    ep = _ep(continue_on_fail=["place"])
    ex = FakeExecutor(ep, inject={"place_cup": [F.POLICY_TIMEOUT] * 5})
    m, st, rows = _run(ep, ex)
    assert st == SUCCESS                                                      # 지금 흐름 그대로 — 남은 노드가 다 끝나면 SUCCESS
    assert ("trajectory", "go_home_end", "go_home") in ex.calls               # 기다리지 않고 home 으로
    assert m.world.hand("right") == "EMPTY"
    assert m.world.objects["CUP"]["at"] == "unknown"                          # 홀더로 단정하지 않는다
    assert "cup_placed" not in m.world.flags
    assert m.runs["place_cup"]["state"] == "failed" and m.runs["place_cup"]["code"] == F.POLICY_TIMEOUT.value
    assert m.place_failed_continued == ["place_cup"] and m.view()["place_failed_continued"] == ["place_cup"]
    events = [r["event"] for r in rows]
    assert "place_failed_continued" in events
    term = [r for r in rows if r["event"] == "terminal"][-1]
    assert term["place_failed_continued"] == ["place_cup"] and term["result"] == SUCCESS


def test_without_opt_in_place_timeout_still_fails():
    ep = _ep()
    assert ep.continue_on_fail == ()
    m, st, _ = _run(ep, FakeExecutor(ep, inject={"place_cup": [F.POLICY_TIMEOUT] * 5}))
    assert st == FAILURE and m.last["code"] == F.POLICY_TIMEOUT.value and m.place_failed_continued == []


def test_safety_codes_are_not_continued():
    ep = _ep(continue_on_fail=["place"])
    m, st, _ = _run(ep, FakeExecutor(ep, inject={"place_cup": [F.CONTROLLER_ERROR]}))
    assert st == FAILURE and m.place_failed_continued == []


def test_continue_kinds_are_checked():
    with pytest.raises(S.EpisodeSpecError, match="continue_on_fail"):
        _ep(continue_on_fail=["aglt"])
    assert S.load(EPISODES / "cocktail_1c_e4000.yaml").continue_on_fail == ("place",)


@pytest.mark.parametrize("sig", [{"cup_in_holder": False, "hand_empty": False},
                                 {"cup_in_holder": True, "hand_empty": True, "outcome": "withdraw_timeout"}])
def test_hand_still_on_cup_is_not_continued(sig):
    ep = _ep(continue_on_fail=["place"])
    ex = _Ex(ep, res=("completed", "", sig))
    m, st, _ = _run(ep, ex)
    assert st == FAILURE and m.place_failed_continued == []
    assert not any(c[:2] == ("trajectory", "go_home_end") for c in ex.calls)


def test_place_failed_with_empty_hand_is_continued():
    ep = _ep(continue_on_fail=["place"])
    m, st, _ = _run(ep, _Ex(ep, res=("completed", "", {"cup_in_holder": False, "hand_empty": True, "outcome": "clear"})))
    assert st == SUCCESS and m.place_failed_continued == ["place_cup"]
    assert m.runs["place_cup"]["code"] == F.PLACE_FAILED.value


@pytest.mark.parametrize("reason,code", [("holder pose missing", F.HOLDER_NOT_FOUND), ("cup pose missing", F.TARGET_NOT_FOUND),
                                         ("rad from start pose", F.POSE_MISMATCH), ("not grasped", F.GRASP_FAILED_RIGHT)])
def test_refused_before_the_policy_ran_is_not_continued(reason, code):
    ep = _ep(continue_on_fail=["place"])
    m, st, _ = _run(ep, _Ex(ep, res=("refused", reason, {})))
    assert st == FAILURE and m.last["code"] == code.value and m.place_failed_continued == []

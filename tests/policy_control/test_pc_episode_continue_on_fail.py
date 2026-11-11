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
                                 {"cup_in_holder": True, "hand_empty": True, "outcome": "withdraw_timeout"},
                                 # ★11.11 래치 전 손 폄 · 손 빔 모름 · 컵 홀더 밖 → 걸림(이어 가지 않는다)
                                 {"cup_in_holder": False, "hand_empty": None, "outcome": "opened_before_latch"}])
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


def test_episode_end_but_camera_says_outside_continues():
    """★11.10 학습 시간 끝(outcome episode_end)으로 멈췄는데 카메라가 홀더 밖으로 보면 PLACE_FAILED → home 으로 이어 간다."""
    ep = _ep(continue_on_fail=["place"])
    ex = _Ex(ep, res=("completed", "placed — 학습 시간 5.0 s 끝", {"cup_in_holder": False, "hand_empty": None, "outcome": "episode_end"}))
    m, st, _ = _run(ep, ex)
    assert st == SUCCESS and m.place_failed_continued == ["place_cup"]
    assert ("trajectory", "go_home_end", "go_home") in ex.calls


def test_episode_end_in_holder_is_success_without_continue():
    """★11.10 학습 시간 끝 · 손 빔 모름(None) · 컵 홀더 안 → 성공(실패 이어 가기 없음)."""
    ep = _ep(continue_on_fail=["place"])
    ex = _Ex(ep, res=("completed", "placed — 학습 시간 5.0 s 끝", {"cup_in_holder": True, "hand_empty": None, "outcome": "episode_end"}))
    m, st, _ = _run(ep, ex)
    assert st == SUCCESS and m.place_failed_continued == []


def test_place_timeout_reason_is_policy_timeout_and_opened_hand_continues():
    """★11.09 'place 시간 초과' 는 UNKNOWN 이 아니라 POLICY_TIMEOUT (261109 161346 은 UNKNOWN 으로 멈췄다)."""
    ep = _ep(continue_on_fail=["place"])
    ex = _Ex(ep, res=("aborted", "place 시간 초과 5.0 s(학습 episode_length_s) — 놓기 전 (래치 False · 0 틱)",
                      {"hand_empty": True}))
    m, st, _ = _run(ep, ex)
    assert st == SUCCESS and m.runs["place_cup"]["code"] == F.POLICY_TIMEOUT.value


def test_cocktail_place_blue_continue_reaches_grasp_pink():
    """칵테일: place_blue 가 실패로 넘어가도 home_r_after_blue → grasp_pink → … → done."""
    ep = S.load(EPISODES / "cocktail_1c_e4000.yaml")

    class _C(FakeExecutor):
        def run_policies(self, node, plans, world):
            out = super().run_policies(node, plans, world)
            if node.id == "place_blue":
                out[0] = NodeResult("completed", "placed — 학습 시간 5.0 s 끝", 1.0, out[0].role,
                                    {"cup_in_holder": False, "hand_empty": None, "outcome": "episode_end"})
            return out

    ex = _C(ep)
    m, st, _ = _run(ep, ex)
    ids = [c[1] for c in ex.calls]
    assert st == SUCCESS and m.place_failed_continued == ["place_blue"]
    for nid in ("home_r_after_blue", "grasp_pink", "tilt_pink", "place_pink", "place_shaker", "go_home_end"):
        assert nid in ids, (nid, ids)


def test_auto_node_runs_after_step_without_approval():
    """★11.09 사용자 — 구분 실행(step)에서 place_blue 다음 home_r_after_blue(auto) 는 승인 없이 바로 이어 간다."""
    ep = S.load(EPISODES / "cocktail_1c_e4000.yaml")
    asked = []
    m = EpisodeManager(ep, FakeExecutor(ep), approve=lambda w, y: asked.append(w) or True, log=lambda r: None)
    while m.node is not None and m.node.id != "place_blue":
        m.step()
    asked.clear()
    m.step()                                            # place_blue 한 번 승인 → home_r_after_blue 까지
    assert asked == ["place_blue"] and m.node.id == "grasp_pink", (asked, m.node.id)


def test_auto_node_failure_still_waits_for_recovery_approval():
    """auto 는 승인만 생략 — 실패 복구는 다음 명령(승인)을 기다린다."""
    ep = S.load(EPISODES / "cocktail_1c_e4000.yaml")
    ex = FakeExecutor(ep, inject={"home_r_after_blue": [F.CONTROLLER_ERROR] * 3})
    m = EpisodeManager(ep, ex, approve=lambda w, y: True, log=lambda r: None)
    while m.node is not None and m.node.id != "place_blue":
        m.step()
    m.step()
    assert m.node.id == "home_r_after_blue" and m.status != SUCCESS


def test_auto_must_be_bool():
    raw = yaml.safe_load((EPISODES / "cocktail_1c_e4000.yaml").read_text())
    raw["sequence"][0]["auto"] = "yes"
    with pytest.raises(S.EpisodeSpecError):
        S.parse(raw)


def test_opened_before_latch_in_holder_is_success():
    """★11.11 래치 전 손 폄 · 손 빔 모름 · 컵 홀더 안 → 지금처럼 성공."""
    ep = _ep(continue_on_fail=["place"])
    ex = _Ex(ep, res=("completed", "", {"cup_in_holder": True, "hand_empty": None, "outcome": "opened_before_latch"}))
    m, st, _ = _run(ep, ex)
    assert st == SUCCESS and m.place_failed_continued == []

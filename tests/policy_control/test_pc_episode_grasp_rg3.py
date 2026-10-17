"""rg3 파지 전용 에피소드(grasp_rg3_right · grasp_rg3_left) — 제자리에서 잡고 들어 첫 목표 성공에 끝낸다(10.13 사용자 방안 A).

기본 물체: 오른손 source240_real · 왼손 shaker_c_real. 실행기 파라미터로 바꾼다(override_grasp_object)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from policy_control import episode_spec as S
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_runner import SUCCESS, EpisodeManager

REPO = Path(__file__).resolve().parents[2]
EPISODES = REPO / "config" / "episodes"
POLICY = {"right": "rh56f1/aglt/right_rg3wo", "left": "rh56f1/aglt/left_rg3wo"}   # 10.13 rg3s(actor 94 · hand_block · grasp_hold)
DEFAULT_OBJECT = {"right": "source240_real", "left": "shaker_c_real"}
CUP_TOPIC = {"right": "/objects/source240b_blue/pose", "left": "/objects/shaker_c_orange/pose"}


def _ep(side: str):
    return S.load(EPISODES / f"grasp_rg3_{side}.yaml")


def _aglt_jobs(ep):
    return [j for n in ep.nodes for j in n.jobs]


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_episode_grasps_lifts_and_ends_without_placing(side):
    ep = _ep(side)
    assert [n.type for n in ep.nodes] == ["trajectory", "snapshot", "policy", "terminal"]
    assert ep.nodes[-1].result == "SUCCESS"
    assert {b.kind for b in ep.policies.values()} == {"aglt"}
    (b,) = ep.policies.values()
    assert b.policy == POLICY[side] and b.side == side
    assert ep.nodes[2].expect == {f"{side}_hand": "CUP"}


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_default_object_and_fpp_topic_follow_the_user_choice(side):
    ep = _ep(side)
    (job,) = _aglt_jobs(ep)
    assert job.policy_object == DEFAULT_OBJECT[side]
    assert ep.objects["CUP"]["topic"] == CUP_TOPIC[side]


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_lift_only_episode_sends_no_setting_so_the_goal_stays_the_sim_first_goal(side):
    """10.13 방안 A: SETTING 을 보내면 GoalBook 이 중간 목표를 쪼개 넣는다(정책이 본 적 없는 목표 바꿈) — 보내지 않는다."""
    ep = _ep(side)
    assert side not in ep.setting
    m = EpisodeManager(ep, FakeExecutor(ep), approve=lambda what, why: True, episode_id="t")
    (plan,) = m._plans(ep.nodes[2], m.world)
    assert plan.setting is None


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_policy_node_ends_at_the_first_goal_like_training(side):
    import yaml
    for kind in ("control", "fake"):
        m = yaml.safe_load((REPO / "config" / f"mission_rh56f1_{kind}.yaml").read_text(encoding="utf-8"))
        argv = next(c["argv"] for c in m["run"][f"episode_grasp_rg3_{side}"]
                    if any("rh_aglt_node.py" in a for a in c["argv"]))
        assert "goal_max:=1" in argv and "stop_on_target:=true" not in argv
        # ★10.13 lift_hold · ★10.14 오른손은 5 s(300 틱) — 정책이 sim 목표 성공(10 cm, 2 s 유지)까지 가게 늦춘다
        hold = {"right": 300, "left": 30}[side]
        for kv in ("freeze_rule:=lift_hold", "freeze_lift_m:=0.07", f"freeze_hold_steps:={hold}"):
            assert argv[argv.index(kv) - 1] == "-p"
        assert "max_episode_s:=15.0" in argv


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_fake_run_ends_in_success_with_the_cup_in_the_hand(side):
    ep = _ep(side)
    m = EpisodeManager(ep, FakeExecutor(ep), approve=lambda what, why: True, episode_id="t")
    m.run()
    assert m.status == SUCCESS and m.world.hand(side) == "CUP"


def test_override_replaces_the_object_without_touching_the_original():
    ep = _ep("right")
    new = S.override_grasp_object(ep, policy_object="source200_real", cup_object="source200_pink")
    assert _aglt_jobs(new)[0].policy_object == "source200_real"
    assert new.objects["CUP"]["topic"] == "/objects/source200_pink/pose"
    assert _aglt_jobs(ep)[0].policy_object == "source240_real"
    assert ep.objects["CUP"]["topic"] == CUP_TOPIC["right"]


def test_override_with_empty_values_keeps_the_defaults():
    ep = _ep("left")
    assert S.override_grasp_object(ep, policy_object="", cup_object="") == ep


def test_override_cup_object_alone_moves_only_the_fpp_topic():
    new = S.override_grasp_object(_ep("left"), policy_object="", cup_object="shaker_c_pink")
    assert _aglt_jobs(new)[0].policy_object == "shaker_c_real"
    assert new.objects["CUP"]["topic"] == "/objects/shaker_c_pink/pose"


def test_an_unknown_policy_object_is_refused_by_availability():
    from policy_control import policy_registry as R
    entries = {e.id: e for e in R.scan(REPO / "deploy" / "policies", deep=False)}
    if POLICY['right'] not in entries or not entries[POLICY['right']].ok:
        pytest.skip("rg3 가중치(nn/)가 이 호스트에 없다")
    bad = S.override_grasp_object(_ep("right"), policy_object="cyl60", cup_object="")
    assert "cyl60" in S.availability(bad, entries)["rh_aglt_r"]
    good = S.override_grasp_object(_ep("right"), policy_object="shaker_c_real", cup_object="")
    assert S.availability(good, entries)["rh_aglt_r"] == ""


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_grasp_episode_stage_does_not_wait_for_cup_holders(side):
    import yaml
    for kind in ("control", "fake"):
        m = yaml.safe_load((REPO / "config" / f"mission_rh56f1_{kind}.yaml").read_text(encoding="utf-8"))
        stages = {s["id"]: s for s in m["stages"]}
        assert "cup_holders" not in stages[f"episode_grasp_rg3_{side}"]["needs"]
        assert "cup_holders" in stages[f"episode_pick_place_{side}"]["needs"]


@pytest.mark.parametrize('side', ['right', 'left'])
def test_grasp_recovery_is_operator_only(side):
    ep = _ep(side)
    assert all(ep.max_retry(node) == 0 for node in ep.nodes)


def test_a_multi_object_policy_node_keeps_its_parameter_services():
    """실행기가 reset 전에 노드 파라미터 object 를 SetParameters 로 넣는다 — lean 노드(서비스 끔)에는 그 서비스가 없다(10.12 실기)."""
    from policy_control.lean_node import lean_node_kwargs
    assert lean_node_kwargs() == {"start_parameter_services": False}
    assert lean_node_kwargs(parameter_services=True) == {"start_parameter_services": True}

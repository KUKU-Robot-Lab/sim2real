"""rg3 파지 전용 에피소드(grasp_rg3_right · grasp_rg3_left) — 잡고 들어 SETTING 에 닿으면 팔을 멈추고 끝낸다(10.12 사용자).

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
POLICY = {"right": "rh56f1/aglt/right_rg3mft", "left": "rh56f1/aglt/left_rg3"}
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
    assert ep.nodes[2].expect == {"pose": "SETTING", f"{side}_hand": "CUP"}


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_default_object_and_fpp_topic_follow_the_user_choice(side):
    ep = _ep(side)
    (job,) = _aglt_jobs(ep)
    assert job.policy_object == DEFAULT_OBJECT[side]
    assert ep.objects["CUP"]["topic"] == CUP_TOPIC[side]


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_setting_goal_is_inside_the_contract_goal_box(side):
    ep = _ep(side)
    c = json.loads((REPO / "deploy/policies" / POLICY[side] / "rh_aglt_contract.json").read_text())
    x, y, z = ep.setting[side]
    lo, hi = c["goal_box_min"], c["goal_box_max"]
    assert lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and lo[2] <= z <= hi[2]


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
    if not all(entries[p].ok for p in POLICY.values()):
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


def test_a_multi_object_policy_node_keeps_its_parameter_services():
    """실행기가 reset 전에 노드 파라미터 object 를 SetParameters 로 넣는다 — lean 노드(서비스 끔)에는 그 서비스가 없다(10.12 실기)."""
    from policy_control.lean_node import lean_node_kwargs
    assert lean_node_kwargs() == {"start_parameter_services": False}
    assert lean_node_kwargs(parameter_services=True) == {"start_parameter_services": True}

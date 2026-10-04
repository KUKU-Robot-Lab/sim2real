"""에피소드 실행기 Step 1 · 2 — YAML 순서 · WorldState · 실패 판정 · 복구 · 구분/연속 실행 · 로그. 10.04 사용자(가이드, VLM 전까지)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from policy_control import episode_spec as S
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_failure import F, FailureDetector, NodeResult, RecoveryManager
from policy_control.episode_runner import FAILURE, READY, SUCCESS, EpisodeManager, jsonl_logger
from policy_control.episode_world import World

REPO = Path(__file__).resolve().parents[2]
EPISODES = REPO / "config" / "episodes"


def _ep(name="pick_place_right"):
    return S.load(EPISODES / f"{name}.yaml")


def _yes(log=None):
    def approve(what, why):
        if log is not None:
            log.append(what)
        return True
    return approve


# ---------------------------------------------------------------- 정의
@pytest.mark.parametrize("name", ["pick_place_right", "pick_place_left", "pick_place_both", "bead_mix_episode"])
def test_the_committed_episodes_load(name):
    ep = _ep(name)
    assert ep.nodes[-1].type == "terminal" and ep.nodes[0].type == "trajectory"
    assert any(n.type == "snapshot" for n in ep.nodes)


def test_the_guide_sequence_is_kept_in_the_bead_mix_episode():
    ep = _ep("bead_mix_episode")
    assert [n.id for n in ep.nodes] == [
        "go_home_start", "scene", "pick_red_blue", "pour_blue_to_red", "place_blue", "return_home_after_blue", "pick_green",
        "pour_green_to_red", "place_green", "lock_red", "shake_red", "place_red", "go_home_end", "episode_success"]
    assert {j.role for j in ep.nodes[2].jobs} == {"rh_aglt_l", "rh_aglt_r"}


def _raw(**over):
    raw = yaml.safe_load((EPISODES / "pick_place_right.yaml").read_text())
    raw.update(over)
    return raw


@pytest.mark.parametrize("bad, match", [
    ({"sequence": [{"id": "a", "type": "jump"}, {"id": "z", "type": "terminal"}]}, "type"),
    ({"sequence": [{"id": "a", "type": "policy", "name": "rh_nope"}, {"id": "z", "type": "terminal"}]}, "rh_nope"),
    ({"sequence": [{"id": "a", "type": "trajectory", "name": "go_home"}]}, "terminal"),
    ({"objects": {"CUP": {"topic": "/x"}, "CUP2": {"topic": "/x"}}}, "topic"),
    ({"sequence": [{"id": "p", "type": "policy", "name": "rh_aglt_r", "target_object": "CUP"},
                   {"id": "z", "type": "terminal"}]}, "snapshot"),
    ({"holder_poses": ""}, "holder_poses"),
])
def test_bad_episodes_are_refused_with_the_reason(bad, match):
    with pytest.raises(S.EpisodeSpecError, match=match):
        S.parse(_raw(**bad))


def test_two_policies_on_the_same_arm_cannot_run_in_parallel():
    raw = _raw(sequence=[{"id": "s", "type": "snapshot"},
                         {"id": "p", "type": "parallel_policy", "policies": [{"name": "rh_aglt_r", "target_object": "CUP"},
                                                                             {"name": "rh_place_r", "source_object": "CUP"}]},
                         {"id": "z", "type": "terminal"}])
    with pytest.raises(S.EpisodeSpecError, match="같은 팔"):
        S.parse(raw)


def test_availability_names_what_blocks_a_real_run():
    from policy_control import policy_registry as R
    entries = {e.id: e for e in R.scan(REPO / "deploy" / "policies", deep=False)}
    av = S.availability(_ep("bead_mix_episode"), entries)
    assert "없다" in av["rh_pour"] and "없다" in av["rh_lock"] and "없다" in av["rh_shake"]
    if all(e.ok for e in entries.values()):
        assert S.availability(_ep(), entries) == {"rh_aglt_r": "", "rh_place_r": ""}


# ---------------------------------------------------------------- 상태
def test_world_applies_expectations_without_changing_the_old_one():
    w0 = World()
    w1 = w0.apply({"pose": "SETTING", "right_hand": "CUP", "cup_placed": False, "x_done": True})
    assert w0.pose == "UNKNOWN" and w0.hand("right") == "EMPTY"
    assert w1.pose == "SETTING" and w1.hand("right") == "CUP" and w1.flags == frozenset({"x_done"})
    assert w1.locate("CUP", "right_hand").objects["CUP"] == {"at": "right_hand", "pos": None}


# ---------------------------------------------------------------- 정상 실행(Step 1)
def test_step_mode_asks_for_every_node_and_ends_in_success():
    ep, asked = _ep(), []
    ex = FakeExecutor(ep)
    m = EpisodeManager(ep, ex, approve=_yes(asked))
    states = []
    while m.status not in (SUCCESS, FAILURE):
        states.append(m.step())
    assert m.status == SUCCESS and asked == ["go_home_start", "scene", "pick_cup", "place_cup", "go_home_end"]
    assert m.world.pose == "HOME" and m.world.hand("right") == "EMPTY" and "cup_placed" in m.world.flags
    assert m.world.objects["CUP"]["at"] == "holder:CENTER_HOLDER"
    assert [c[0] for c in ex.calls] == ["trajectory", "snapshot", "policy", "policy", "trajectory"]
    assert "HOME_START" in m.checkpoints


def test_the_snapshot_pose_is_handed_to_the_grasp_policy():
    ep = _ep()
    seen = {}

    class Ex(FakeExecutor):
        def run_policies(self, node, plans, world):
            seen[node.id] = plans
            return super().run_policies(node, plans, world)

    m = EpisodeManager(ep, Ex(ep), approve=_yes())
    m.run()
    pick = seen["pick_cup"][0]
    assert pick.object_pose is not None and pick.setting == (0.25, -0.12, 0.41) and pick.policy == "right_rh_aglt_cyl60g"
    place = seen["place_cup"][0]
    assert place.holder_id == 1 and place.target_holder == "CENTER_HOLDER"


def test_auto_mode_asks_once_and_runs_the_parallel_grasp():
    ep, asked = _ep("pick_place_both"), []
    ex = FakeExecutor(ep)
    m = EpisodeManager(ep, ex, approve=_yes(asked))
    assert m.run() == SUCCESS and asked == ["episode:pick_place_both"]
    assert ("policy", "pick_both", ("rh_aglt_l", "rh_aglt_r")) in ex.calls
    assert m.world.objects["CUP_R"]["at"] == "holder:RIGHT_HOLDER" and m.world.objects["CUP_L"]["at"] == "holder:LEFT_HOLDER"


def test_the_whole_guide_episode_runs_in_the_dry_run():
    ep = _ep("bead_mix_episode")
    m = EpisodeManager(ep, FakeExecutor(ep), approve=_yes())
    assert m.run() == SUCCESS
    assert {"blue_poured", "green_poured", "red_locked", "shake_done", "red_placed"} <= m.world.flags
    assert m.world.hand("left") == "EMPTY" and m.world.pose == "HOME"


def test_a_declined_node_stops_without_running_it():
    ep = _ep()
    ex = FakeExecutor(ep)
    m = EpisodeManager(ep, ex, approve=lambda what, _why: what != "pick_cup")
    m.step()
    m.step()
    assert m.step() == FAILURE and not any(c[0] == "policy" for c in ex.calls) and ex.stopped


def test_the_log_has_the_guide_fields(tmp_path):
    ep, path = _ep(), tmp_path / "episode.jsonl"
    EpisodeManager(ep, FakeExecutor(ep), approve=_yes(), log=jsonl_logger(path)).run()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    need = {"timestamp", "episode_id", "state_id", "executor_type", "policy_name", "left_hand_object", "right_hand_object",
            "pose_state", "task_flags", "event"}
    assert all(need <= set(r) for r in rows)
    exits = [r for r in rows if r["event"] == "exit"]
    assert [r["state_id"] for r in exits] == ["go_home_start", "scene", "pick_cup", "place_cup", "go_home_end"]
    assert all({"start_time", "end_time", "duration", "next"} <= set(r) for r in exits)


# ---------------------------------------------------------------- 실패 · 복구(Step 2)
def test_a_failed_grasp_goes_back_home_rerecords_the_cup_and_retries():
    ep = _ep()
    ex = FakeExecutor(ep, inject={"pick_cup": [F.GRASP_FAILED_RIGHT]})
    m = EpisodeManager(ep, ex, approve=_yes())
    assert m.run() == SUCCESS and m.attempts == {"pick_cup": 1}
    kinds = [c[:2] for c in ex.calls]
    i = kinds.index(("policy", "pick_cup"))
    assert kinds[i + 1:i + 5] == [("prepare", "open_hand"), ("trajectory", "go_home_start"), ("snapshot", "pick_cup:resnap"),
                                  ("policy", "pick_cup")]


def test_retries_stop_at_the_cap_and_then_the_episode_stops_safely():
    ep = _ep()
    ex = FakeExecutor(ep, inject={"pick_cup": [F.GRASP_FAILED_RIGHT] * 3})
    m = EpisodeManager(ep, ex, approve=_yes())
    assert m.run() == FAILURE and m.attempts == {"pick_cup": 2} and ex.stopped
    assert not any(c[:2] == ("policy", "place_cup") for c in ex.calls)            # 실패 뒤 다음 정책은 돌지 않는다


@pytest.mark.parametrize("code", [F.CONTROLLER_ERROR, F.OBSERVATION_ERROR, F.OBJECT_DROPPED_RIGHT])
def test_safety_and_unrecoverable_failures_stop_at_once(code):
    ep = _ep()
    ex = FakeExecutor(ep, inject={"pick_cup": [code]})
    m = EpisodeManager(ep, ex, approve=_yes())
    assert m.run() == FAILURE and m.attempts == {} and m.last["code"] == code.value


def test_a_failed_place_stops_because_there_is_no_regrasp_skill():
    ep = _ep()
    m = EpisodeManager(ep, FakeExecutor(ep, inject={"place_cup": [F.PLACE_FAILED]}), approve=_yes())
    assert m.run() == FAILURE and m.last["code"] == "place_failed"


def test_a_missing_holder_is_retried_after_a_perception_refresh():
    ep = _ep()
    ex = FakeExecutor(ep, inject={"place_cup": [F.HOLDER_NOT_FOUND]})
    rc = EpisodeManager(ep, ex, approve=_yes(), episode_id="t").run()
    assert rc == FAILURE                                                         # rh_place_r 재시도 0 — 상한
    ep2 = S.parse({**yaml.safe_load((EPISODES / "pick_place_right.yaml").read_text()),
                   "failure_policy": {"default": 1}})
    ex2 = FakeExecutor(ep2, inject={"place_cup": [F.HOLDER_NOT_FOUND]})
    assert EpisodeManager(ep2, ex2, approve=_yes()).run() == SUCCESS
    assert ("prepare", "refresh_perception", ("rh_place_r",)) in ex2.calls


def test_no_rollback_while_holding_an_object():
    """홈 checkpoint 는 빈손 — 컵을 든 채(place 실패 · 자세 어긋남) 홈으로 돌아가지 않는다(계획기가 든 컵을 모른다)."""
    ep = S.parse({**yaml.safe_load((EPISODES / "pick_place_right.yaml").read_text()), "failure_policy": {"default": 2}})
    m = EpisodeManager(ep, FakeExecutor(ep), approve=_yes())
    for _ in range(3):                                                           # 집기까지 — 오른손 CUP
        m.step()
    assert m.world.hand("right") == "CUP"
    rec = RecoveryManager().decide(ep, ep.nodes[3], F.POSE_MISMATCH, 0, m.world, m.checkpoints)
    assert rec.action == "safe_stop" and "checkpoint" in rec.reason


def test_step_mode_asks_before_a_recovery_motion():
    ep, asked = _ep(), []
    ex = FakeExecutor(ep, inject={"pick_cup": [F.GRASP_FAILED_RIGHT]})
    m = EpisodeManager(ep, ex, approve=_yes(asked))
    for _ in range(3):
        m.step()
    assert m.status == READY and asked[-1].startswith("recover:pick_cup:rollback_retry")


def test_the_detector_reads_refusal_text_and_signals():
    ep, d = _ep(), FailureDetector()
    node = ep.nodes[2]

    def code(res):
        return d.evaluate(ep, node, [res])[0]

    assert code(NodeResult("refused", "arm 컵 자세가 없다(또는 0.5 s 넘게 끊겼다)", role="rh_aglt_r")) is F.TARGET_NOT_FOUND
    assert code(NodeResult("refused", "right arm is 0.400 rad from the training start pose (tol 0.15)", role="rh_aglt_r")) \
        is F.POSE_MISMATCH
    assert code(NodeResult("aborted", "4 ticks without a valid step: source 'arm' is stale", role="rh_aglt_r")) \
        is F.OBSERVATION_ERROR
    assert code(NodeResult("timeout", "episode time >= 15.0 s", role="rh_aglt_r")) is F.POLICY_TIMEOUT
    assert code(NodeResult("completed", "", role="rh_aglt_r", signals={"attached": True, "setting_err_m": 0.01})) is F.NONE
    place = ep.nodes[3]
    assert d.evaluate(ep, place, [NodeResult("refused", "reset: holder pose missing — …", role="rh_place_r")])[0] \
        is F.HOLDER_NOT_FOUND


def test_a_timeout_after_the_grasp_is_not_retried_from_home():
    ep = S.parse({**yaml.safe_load((EPISODES / "pick_place_right.yaml").read_text()), "failure_policy": {"default": 2}})
    m = EpisodeManager(ep, FakeExecutor(ep, inject={"place_cup": [F.POLICY_TIMEOUT]}), approve=_yes())
    assert m.run() == FAILURE and "timeout" in m.last["code"]

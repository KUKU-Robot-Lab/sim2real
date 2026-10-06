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
    assert pick.object_pose is not None and pick.setting == (0.25, -0.12, 0.41) and pick.policy == "rh56f1/aglt/right_cyl60g"
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


def test_a_node_without_approval_does_not_run_and_the_episode_waits():
    ep = _ep()
    ex = FakeExecutor(ep)
    ok = {"pick_cup": False}
    m = EpisodeManager(ep, ex, approve=lambda what, _why: ok.get(what, True))
    m.step()
    m.step()
    assert m.step() == READY and m.next_action() == "pick_cup" and not any(c[0] == "policy" for c in ex.calls)
    assert m.last["code"] == "operator_declined" and not ex.stopped
    ok["pick_cup"] = True
    assert m.step() == READY and m.next_action() == "place_cup"


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
    assert ("prepare", "refresh_holders", ("rh_place_r",)) in ex2.calls


def test_no_rollback_while_holding_an_object():
    """홈 checkpoint 는 빈손 — 컵을 든 채(place 실패 · 자세 어긋남) 홈으로 돌아가지 않는다(계획기가 든 컵을 모른다)."""
    ep = S.parse({**yaml.safe_load((EPISODES / "pick_place_right.yaml").read_text()), "failure_policy": {"default": 2}})
    m = EpisodeManager(ep, FakeExecutor(ep), approve=_yes())
    for _ in range(3):                                                           # 집기까지 — 오른손 CUP
        m.step()
    assert m.world.hand("right") == "CUP"
    rec = RecoveryManager().decide(ep, ep.nodes[3], F.POSE_MISMATCH, 0, m.world, m.checkpoints)
    assert rec.action == "safe_stop" and "checkpoint" in rec.reason


def test_step_mode_makes_the_recovery_its_own_approved_command():
    """구분 실행: 실패한 노드 뒤 다음 명령은 복구(되돌아가기) 하나 — 그다음 명령이 그 노드를 다시 돈다."""
    ep, asked = _ep(), []
    ex = FakeExecutor(ep, inject={"pick_cup": [F.GRASP_FAILED_RIGHT]})
    m = EpisodeManager(ep, ex, approve=_yes(asked))
    for _ in range(3):
        m.step()
    assert m.status == READY and m.next_action() == "recover:pick_cup:rollback_retry"
    assert m.view()["pending"]["action"] == "rollback_retry"
    m.step()
    assert asked[-1] == "recover:pick_cup:rollback_retry" and m.next_action() == "pick_cup"
    assert [c[:2] for c in ex.calls][-3:] == [("prepare", "open_hand"), ("trajectory", "go_home_start"),
                                              ("snapshot", "pick_cup:resnap")]
    while m.status not in (SUCCESS, FAILURE):
        m.step()
    assert m.status == SUCCESS


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


# ---------------------------------------------------------------- 명령 · 승인 파일 · ROS 순수부
def test_the_command_tool_only_sends_the_name_the_runner_will_ask_for():
    import importlib.util
    spec = importlib.util.spec_from_file_location("episode_cmd", REPO / "deploy/policy_control/tools/episode_cmd.py")
    cmd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmd)
    st = {"episode": "pick_place_right", "phase": "READY", "next_action": "pick_cup", "busy": False}
    assert cmd.refusal("next", "pick_cup", st) is None
    assert "≠" in cmd.refusal("next", "place_cup", st)
    assert cmd.refusal("run", "episode:pick_place_right", st) is None
    assert "busy" not in (cmd.refusal("run", "x", st) or "") and cmd.refusal("run", "x", st)
    assert "돌고" in cmd.refusal("next", "pick_cup", {**st, "busy": True})
    assert cmd.refusal("stop", None, {**st, "busy": True}) is None                 # 멈춤은 언제나
    assert "없다" in cmd.refusal("next", None, {**st, "next_action": None, "phase": "SUCCESS"})


def test_an_approval_line_is_used_once_and_only_after_the_runner_started(tmp_path):
    from policy_control.episode_runner_node import Approvals, write_approval
    path = tmp_path / "approvals.jsonl"
    write_approval("pick_cup", "old", path)                                        # 뜨기 전 승인은 안 쓴다
    import time as _t
    _t.sleep(0.01)
    a = Approvals(path, _t.time())
    assert not a.take("pick_cup")
    write_approval("pick_cup", "op", path)
    assert a.take("pick_cup") and not a.take("pick_cup")                           # 한 번만
    write_approval("episode:pick_place_right", "op", path)
    assert not a.take("pick_cup") and a.take("episode:pick_place_right")


def test_snapshot_needs_enough_still_frames_and_the_holder_check_uses_the_seat():
    from policy_control.episode_ros import SEAT_DZ_CYL60, cup_in_holder, load_holder_poses, snapshot_of
    q = (1.0, 0.0, 0.0, 0.0)
    still = [((0.25, -0.2, 0.29 + 0.001 * (k % 2)), q, 0.0) for k in range(8)]
    pos, quat = snapshot_of(still)
    assert pos == pytest.approx((0.25, -0.2, 0.2905), abs=1e-3) and quat == q
    assert snapshot_of(still[:3]) is None                                          # 프레임 모자람
    moving = [((0.25 + 0.01 * k, -0.2, 0.29), q, 0.0) for k in range(8)]
    assert snapshot_of(moving) is None                                             # 움직이는 중
    assert SEAT_DZ_CYL60 == pytest.approx(0.060)
    assert cup_in_holder((0.381, -0.003, 0.296), (0.38, -0.002, 0.235), SEAT_DZ_CYL60)
    assert not cup_in_holder((0.43, -0.003, 0.296), (0.38, -0.002, 0.235), SEAT_DZ_CYL60)   # 구멍 밖
    assert not cup_in_holder((0.38, -0.002, 0.36), (0.38, -0.002, 0.235), SEAT_DZ_CYL60)    # 링 위에 걸침
    assert load_holder_poses(Path("/nonexistent.yaml")) == {}


def test_holder_problems_names_missing_files_and_ids():
    ep = _ep()
    assert S.holder_problems(ep, {1: ((0.38, -0.002, 0.235), 0.0)}) == []
    assert "CENTER_HOLDER" in S.holder_problems(ep, {0: ((0.38, 0.15, 0.235), 0.0)})[0]
    assert "cup_holders" in S.holder_problems(ep, {})[0]                         # 파일이 없다 — cup_holders --write 를 돌릴 것


def test_the_plan_tool_refuses_a_real_run_without_the_holder_file(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("episode_run", REPO / "deploy/policy_control/tools/episode_run.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    raw = yaml.safe_load((EPISODES / "pick_place_right.yaml").read_text())
    raw["holder_poses"] = str(tmp_path / "none.yaml")
    path = tmp_path / "ep.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    assert tool.main(["--episode", str(path), "--plan", "--require-holders"]) == 1
    (tmp_path / "none.yaml").write_text("holders:\n  cup_holder_1: {marker_id: 1, position: [0.38, -0.002, 0.235], yaw_rad: 0.0}\n")
    rc = tool.main(["--episode", str(path), "--plan", "--require-holders"])
    from policy_control import policy_registry as R
    if all(e.ok for e in R.scan(REPO / "deploy" / "policies", deep=False)):
        assert rc == 0


# ---------------------------------------------------------------- 흐름 그림(상황판)
def test_the_flow_draws_the_grasp_retry_as_a_feedback_back_to_the_empty_handed_home():
    """10.04 사용자: 상황판이 연결창처럼 순서대로, 두 번 반복하는 부분은 피드백 연결처럼."""
    from policy_control.episode_flow import flow_of
    f = flow_of(_ep())
    assert [n["id"] for n in f["nodes"]] == ["go_home_start", "scene", "pick_cup", "place_cup", "go_home_end", "done"]
    back = [e for e in f["feedback"] if e["kind"] == "rollback"]
    assert back == [{"from": "pick_cup", "to": "go_home_start", "kind": "rollback", "max": 2, "label": back[0]["label"]}]
    assert not any(e["from"] == "place_cup" for e in f["feedback"])            # 놓기는 재시도 0 — 실패면 정지
    pick = next(n for n in f["nodes"] if n["id"] == "pick_cup")
    assert "rh56f1/aglt/right_cyl60g" in " ".join(pick["lines"]) and "SETTING" in pick["expect"]
    assert next(n for n in f["nodes"] if n["id"] == "go_home_start")["checkpoint"] == "HOME_START"


def test_no_feedback_back_home_while_a_hand_holds_a_cup():
    """bead_mix 의 두 번째 집기(왼손이 RED 를 든 채) — 빈손 홈 checkpoint 와 손이 달라 되돌아가기 화살표가 그 checkpoint 로만."""
    from policy_control.episode_flow import flow_of
    f = flow_of(_ep("bead_mix_episode"))
    back = {e["from"]: e["to"] for e in f["feedback"] if e["kind"] == "rollback"}
    assert back["pick_red_blue"] == "go_home_start"
    assert back["pick_green"] == "return_home_after_blue"                      # 왼손 RED · 오른손 빈손인 홈


def test_the_view_carries_the_flow_and_what_happened_at_each_node():
    ep = _ep()
    m = EpisodeManager(ep, FakeExecutor(ep, inject={"pick_cup": [F.GRASP_FAILED_RIGHT]}), approve=_yes())
    m.run()
    v = m.view()
    assert v["flow"]["nodes"][0]["id"] == "go_home_start"
    assert v["runs"]["pick_cup"]["n"] == 2 and v["runs"]["pick_cup"]["state"] == "done"
    assert v["runs"]["place_cup"]["state"] == "done" and v["runs"]["go_home_start"]["n"] == 2   # 되돌아가기도 센다

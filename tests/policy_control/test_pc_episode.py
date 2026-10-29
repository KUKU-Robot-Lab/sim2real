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
    if all(b.policy in entries and entries[b.policy].ok for b in _ep().policies.values()):
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
    # ★10.20 마지막 동작 노드의 [다음] 한 번으로 끝난다 — terminal 은 묻지 않고 바로 닫는다(남는 READY 없음)
    assert states[-1] == SUCCESS and len(states) == len(asked) and m.next_action() is None


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


@pytest.mark.parametrize("name, flags, pose", [
    ("cocktail_1", {"blue_poured", "blue_placed", "pink_poured", "pink_placed", "shaker_placed"}, "HOME"),
    ("cocktail_1b", {"blue_poured", "blue_placed", "pink_poured", "pink_placed", "shaker_placed"}, "HOME"),
    ("cocktail_2", {"blue_poured", "blue_placed", "pink_poured", "pink_placed", "shaker_capped", "shaker_placed"}, "SETTING"),
])
def test_the_cocktail_episodes_run_in_the_dry_run(name, flags, pose):
    """10.27 사용자 칵테일 순서 — 세 홀더를 다 쓰고(shaker → 홀더 0, 파랑 · 분홍 배정은 순서 A · B 가 다르다 — 아래),
    오른팔만 홈 동안 왼손은 shaker 를 든다."""
    ep = _ep(name)
    S.validate(ep)
    holders = {j.target_holder for n in ep.nodes for j in n.jobs if j.target_holder}
    assert {ep.holders[h] for h in holders} == {0, 1, 2}
    m = EpisodeManager(ep, FakeExecutor(ep), approve=_yes())
    assert m.run() == SUCCESS
    assert flags <= m.world.flags
    assert m.world.hand("right") == m.world.hand("left") == "EMPTY" and m.world.pose == pose


# ---------------------------------------------------------------- ★10.28 사람이 나중에 놓는 물체(칵테일 1 분홍)
def _pink_gate():
    """분홍 확인만 따로 허락하는 승인 — ok 를 바꿔 운영자 확인을 흉내 낸다(나머지는 허락)."""
    ok = {"confirm_pink_placed": False}
    return ok, (lambda what, _why: ok.get(what, True))


def test_cocktail_1_starts_without_pink_and_waits_for_the_operator_to_place_it():
    """10.28 사용자 "블루컵 놓기 이후에, 내가 핑크컵을 놓아주는 식" — 처음 snapshot 은 분홍을 찾지 않는다. 파랑을 홀더 1 에
    놓고 오른팔이 홈에 가면 확인에서 멈추고(왼손은 shaker), 확인하면 분홍만 다시 찍어 잡기 → 붓기 → 홀더 2 로 잇는다.
    홀더 배정은 순서 B 와 같다(10.28 사용자 "놓는순서 그러면 통일시켜주고")."""
    ep = _ep("cocktail_1")
    ok, approve = _pink_gate()
    ex = FakeExecutor(ep)
    m = EpisodeManager(ep, ex, approve=approve)
    assert m.world.objects["SRC_PINK"]["at"] == "away"
    assert m.run() == READY and m.waiting == "confirm_pink_placed"
    assert m.world.objects["SRC_BLUE"]["at"] == "holder:HOLDER_1" and m.world.objects["SRC_PINK"]["at"] == "away"
    assert m.world.hand("right") == "EMPTY" and m.world.hand("left") == "SHAKER"
    assert m.view()["confirm"]["node"] == "confirm_pink_placed"
    ok["confirm_pink_placed"] = True
    assert m.run() == SUCCESS
    assert [c[1] for c in ex.calls if c[0] == "snapshot"] == ["scene", "scene_pink"]
    assert {k: m.world.objects[k]["at"] for k in ("SRC_BLUE", "SRC_PINK", "SHAKER")} == {
        "SRC_BLUE": "holder:HOLDER_1", "SRC_PINK": "holder:HOLDER_2", "SHAKER": "holder:HOLDER_0"}
    back = [f for f in m.flow["feedback"] if f["from"] == "scene_pink"]
    assert [(f["to"], f["max"]) for f in back] == [("confirm_pink_placed", S.PLACED_SNAPSHOT_TRIES - 1)]


def test_a_failed_pink_rescan_goes_back_to_the_confirm_and_the_third_stops():
    """못 찾으면 로봇은 움직이지 않았다 — 그 확인으로 돌아가 운영자를 다시 기다린다(연속 실행도 멈춘다). 세 번째 실패는
    보통 복구(재시도 0) = 정지 — fake 자동 승인이 끝없이 돌지 않게."""
    ep = _ep("cocktail_1")
    ok, approve = _pink_gate()
    ex = FakeExecutor(ep, inject={"scene_pink": ["target_not_found"] * 3})
    m = EpisodeManager(ep, ex, approve=approve)
    m.run()
    for k in (1, 2):
        ok["confirm_pink_placed"] = True
        since = m.node_since
        assert m.run() == READY and m.waiting == "confirm_pink_placed" and m.node.id == "confirm_pink_placed"
        assert m.placed_tries == {"scene_pink": k} and m.last["code"] == "waiting_confirm"
        assert m.runs["confirm_pink_placed"]["state"] == "waiting" and m.node_since >= since   # 옛 승인은 다시 안 쓴다
        assert not any(c[0] == "policy" and c[1] == "grasp_pink" for c in ex.calls) and not ex.stopped
    assert m.run() == FAILURE and m.last["code"] == "target_not_found" and ex.stopped


def test_a_pink_outside_the_verified_box_is_refused_and_not_recorded():
    """region = sim e2e 오른손 잡기 검증 상자(x 0.20~0.30 · y −0.20~−0.10) — 밖이면 기록하지 않고 확인으로 돌아간다."""
    ep = _ep("cocktail_1")

    class Off(FakeExecutor):
        def snapshot(self, node, world):
            res, poses = super().snapshot(node, world)
            if node.id == "scene_pink":
                poses = {k: ((0.35, -0.15, 0.29), q) for k, (_p, q) in poses.items()}
            return res, poses

    ok, approve = _pink_gate()
    m = EpisodeManager(ep, Off(ep), approve=approve)
    m.run()
    ok["confirm_pink_placed"] = True
    assert m.run() == READY and m.node.id == "confirm_pink_placed"
    assert "cup outside region" in m.last["reason"] and "x 0.350" in m.last["reason"]
    assert m.world.objects["SRC_PINK"]["at"] == "away"


def _episode_raw(name):
    """커밋된 에피소드 파일 그대로(dict) — 위 _raw(**over) 는 손으로 만든 최소 에피소드다."""
    return yaml.safe_load((EPISODES / f"{name}.yaml").read_text(encoding="utf-8"))


def _cocktail_raw():
    return _episode_raw("cocktail_1")


def _without(raw, *ids):
    raw["sequence"] = [n for n in raw["sequence"] if n["id"] not in ids]
    return raw


@pytest.mark.parametrize("edit, match", [
    (lambda r: _without(r, "confirm_pink_placed", "scene_pink"), "at_start away"),     # 찍지 않은 분홍으로 손을 뻗는다
    (lambda r: _without(r, "confirm_pink_placed"), "바로 앞이 confirm"),                  # 사람 확인 없이 다시 찍기
    (lambda r: r["objects"]["SRC_PINK"].update(at_start="later") or r, "at_start"),
    (lambda r: r["objects"]["SRC_PINK"].update(region={"x": [0.3, 0.2]}) or r, "lo < hi"),
    (lambda r: r["objects"]["SRC_PINK"].update(region={"z": [0.2, 0.3]}) or r, "x · y"),
    (lambda r: r["sequence"][0].update(objects=["SRC_PINK"]) or r, "snapshot 에만"),
])
def test_bad_placed_later_setups_are_refused(edit, match):
    with pytest.raises(S.EpisodeSpecError, match=match):
        S.parse(edit(_cocktail_raw()))


def test_the_pink_rescan_asks_fpp_for_the_real_object_and_needs_it_found():
    """scene_pink = 미션 cups 단계와 같은 도구(fpp_rescan)로 그 물체만 다시 찍기 — 다 찾아야 통과(--require-all)."""
    from dataclasses import replace

    from policy_control.episode_ros import RESCAN_TOOL, rescan_argv
    ep = _ep("cocktail_1")
    argv = rescan_argv(ep, ["SRC_PINK"])
    assert argv[1] == str(RESCAN_TOOL) and RESCAN_TOOL.is_file()
    assert argv[2] == "source240p_pink" and "--require-all" in argv              # ★10.28 분홍만 든 FP++ 묶음 source240p
    odd = replace(ep, objects={**ep.objects, "SRC_PINK": {"topic": "/camera/pink"}})
    with pytest.raises(ValueError, match="실물 이름"):
        rescan_argv(odd, ["SRC_PINK"])


# ---------------------------------------------------------------- ★10.28 칵테일 1 순서 B(cocktail_1b)
def test_cocktail_1b_has_pink_on_the_table_from_the_start_and_places_blue_1_pink_2():
    """순서 B(10.27 사용자 정의 · 10.28 sim 통과 뒤 추가) — 분홍은 처음부터 탁자에 있어 처음 snapshot 이 함께 찍고(사람 확인 ·
    다시 찍기 없음), 파랑 → 홀더 1 · 분홍 → 홀더 2 · shaker → 홀더 0."""
    ep = _ep("cocktail_1b")
    ex = FakeExecutor(ep)
    m = EpisodeManager(ep, ex, approve=_yes())
    assert not any(n.type == "confirm" for n in ep.nodes) and "at_start" not in ep.objects["SRC_PINK"]
    assert m.run() == SUCCESS
    assert [c[1] for c in ex.calls if c[0] == "snapshot"] == ["scene"]
    assert {k: m.world.objects[k]["at"] for k in ("SRC_BLUE", "SRC_PINK", "SHAKER")} == {
        "SRC_BLUE": "holder:HOLDER_1", "SRC_PINK": "holder:HOLDER_2", "SHAKER": "holder:HOLDER_0"}


def test_cocktail_1b_differs_from_cocktail_1_only_where_order_b_says():
    """B 는 A 의 변형 — 정책 · 궤적 · 실패 정책 · 홀더 · 홀더 배정이 같고, 순서는 사람 확인 · 다시 찍기만 뺀다(홀더 배정은
    10.28 사용자 "놓는순서 그러면 통일시켜주고" — A v3 부터 파랑 → 1 · 분홍 → 2). 분홍은 at_start 가 없고(처음부터 탁자)
    region 만 다르다. A 를 고치면 B 도 같이 고치라는 알림."""
    a, b = _episode_raw("cocktail_1"), _episode_raw("cocktail_1b")
    for k in ("policies", "trajectories", "failure_policy", "holders", "holder_poses"):
        assert a[k] == b[k], k
    assert b["sequence"] == [n for n in a["sequence"] if n["id"] not in ("confirm_pink_placed", "scene_pink")]
    pa, pb = a["objects"].pop("SRC_PINK"), b["objects"].pop("SRC_PINK")
    assert a["objects"] == b["objects"] and pa["topic"] == pb["topic"]
    assert pa["at_start"] == "away" and "at_start" not in pb
    assert pb["region"] == {"x": [0.26, 0.30], "y": [-0.365, -0.340]}          # ★10.29 사용자 v1 상자 ±1 cm(인식 잡음 여유, v5)


@pytest.mark.parametrize("xy, ok", [
    ((0.28, -0.3525), True), ((0.27, -0.355), True), ((0.29, -0.350), True),   # ★10.29 구멍 옆 v1 상자 안
    ((0.26, -0.365), True), ((0.30, -0.340), True), ((0.28, -0.359), True),    # v5 ±1 cm 인식 여유(끝 포함)
    ((0.26, -0.3275), False), ((0.25, -0.3525), False),                        # 10.28 v2 둘레 영역 · x 밖(손목이 분홍을 침)
    ((0.31, -0.3525), False),                                                  # x 밖(잡는 동안 병이 홀더 2 에 닿음)
    ((0.28, -0.37), False), ((0.28, -0.335), False),                           # y 밖(−0.36 바깥은 10.23 불가 자세 규칙 밖)
])
def test_cocktail_1b_stops_at_the_first_snapshot_when_pink_is_off_its_spot(xy, ok):
    """분홍이 구멍 옆 v1 상자 ±1 cm(x 0.26~0.30 · y −0.365~−0.340, 사용자 10.29 v5) 밖이면 처음 snapshot 이 거부하고 정지한다 — 로봇은 홈에서
    아무것도 쥐지 않았다. 콘솔 경고 줄(last.reason)에 어디가 벗어났는지 보여 운영자가 다시 놓고 새로 시작한다."""
    ep = _ep("cocktail_1b")

    class At(FakeExecutor):
        def snapshot(self, node, world):
            res, poses = super().snapshot(node, world)
            if "SRC_PINK" in poses:
                poses["SRC_PINK"] = ((*xy, 0.29), poses["SRC_PINK"][1])
            return res, poses

    ex = At(ep)
    m = EpisodeManager(ep, ex, approve=_yes())
    if ok:
        assert m.run() == SUCCESS
        return
    assert m.run() == FAILURE and m.last["node"] == "scene" and m.last["code"] == "target_not_found" and ex.stopped
    assert "cup outside region" in m.last["reason"] and "SRC_PINK" in m.last["reason"]
    assert not any(c[0] == "policy" for c in ex.calls)
    assert m.world.hand("right") == m.world.hand("left") == "EMPTY"


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
    lift = "lift_hold: cup +0.111 m above reset (> 0.07) and grasped for 30 ticks (sim freeze_rule lift_hold)"
    # 10.13 실기 왼손: 들기 성공 규칙으로 끝났으면 SETTING 거리는 보지 않는다
    assert code(NodeResult("completed", lift, role="rh_aglt_r",
                           signals={"attached": True, "setting_err_m": 0.088, "live_off_m": 0.007})) is F.NONE
    assert code(NodeResult("completed", "", role="rh_aglt_r",
                           signals={"attached": True, "setting_err_m": 0.088})) is F.POSE_MISMATCH
    # 10.13 실기 오른손: FP++ 실측 컵이 붙은 컵에서 9 cm 떨어졌다 — 손에 없다
    assert code(NodeResult("completed", lift, role="rh_aglt_r",
                           signals={"attached": True, "setting_err_m": 0.04, "live_off_m": 0.091})) is F.GRASP_FAILED_RIGHT
    # 10.13 실기 오른손 223839: 첫 쥠이 미끄러져 감시가 뗐고(live_off 0.094 는 그 부착의 기록) 두 번째 쥠으로 FP++ 실측 컵을
    #   9.6 cm 들었다 — 붙지 않은 채 들기 성공이면 그 들림은 FP++ 가 직접 본 것이다. 지난 부착의 거리는 보지 않는다
    assert code(NodeResult("completed", lift, role="rh_aglt_r",
                           signals={"attached": False, "setting_err_m": 0.056, "live_off_m": 0.0939})) is F.NONE
    assert code(NodeResult("completed", "", role="rh_aglt_r",
                           signals={"attached": False, "setting_err_m": 0.01})) is F.GRASP_FAILED_RIGHT
    assert code(NodeResult("completed", lift, role="rh_aglt_r",
                           signals={"attached": False, "dropped": True})) is F.OBJECT_DROPPED_RIGHT
    # ★10.14 붙이기를 끈 위치제어 노드: attached = None(모름) — 안 붙은 것을 쥠 실패로 보지 않는다
    assert code(NodeResult("completed", "", role="rh_aglt_r",
                           signals={"attached": None, "setting_err_m": 0.01, "live_off_m": None})) is F.NONE
    assert code(NodeResult("completed", "", role="rh_aglt_r",
                           signals={"attached": None, "setting_err_m": 0.088})) is F.POSE_MISMATCH
    assert d.evaluate(ep, place, [NodeResult("refused", "reset: holder pose missing — …", role="rh_place_r")])[0] \
        is F.HOLDER_NOT_FOUND


def test_a_timeout_after_the_grasp_is_not_retried_from_home():
    ep = S.parse({**yaml.safe_load((EPISODES / "pick_place_right.yaml").read_text()), "failure_policy": {"default": 2}})
    m = EpisodeManager(ep, FakeExecutor(ep, inject={"place_cup": [F.POLICY_TIMEOUT]}), approve=_yes())
    assert m.run() == FAILURE and "timeout" in m.last["code"]


def test_grasp_timeout_never_opens_hand_rolls_back_or_retries():
    ep = S.parse(_raw(failure_policy={'default': 3}))
    ex = FakeExecutor(ep, inject={'pick_cup': [F.POLICY_TIMEOUT]})
    m = EpisodeManager(ep, ex, approve=_yes())
    assert m.run() == FAILURE
    assert not any(c[0] == 'prepare' for c in ex.calls)
    assert sum(c[:2] == ('trajectory', 'go_home_start') for c in ex.calls) == 1


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


def test_holder_numbers_come_from_the_name_not_the_marker(tmp_path):
    """★10.20 사용자 "순서를 다시 0,1,2로" — 번호는 이름에서 읽는다. 예: 왼쪽부터 마커 0·2·1 이면 CENTER_HOLDER 1 은 가운데 홀더(마커 2)."""
    from policy_control.episode_ros import load_holder_poses
    f = tmp_path / "holders.yaml"
    f.write_text("holders:\n"
                 "  cup_holder_0: {marker_id: 0, position: [0.38, 0.154, 0.235], yaw_rad: 0.0}\n"
                 "  cup_holder_1: {marker_id: 2, position: [0.38, -0.052, 0.235], yaw_rad: 0.0}\n"
                 "  cup_holder_2: {marker_id: 1, position: [0.38, -0.329, 0.235], yaw_rad: 0.0}\n")
    poses = load_holder_poses(f)
    assert {h: p[1] for h, (p, _) in poses.items()} == {0: 0.154, 1: -0.052, 2: -0.329}
    f.write_text("holders:\n  holder_left: {marker_id: 0, position: [0.38, 0.15, 0.235]}\n")
    with pytest.raises(ValueError, match="cup_holder_"):
        load_holder_poses(f)


def test_the_holder_config_numbers_holders_left_to_right():
    """설정의 홀더 이름은 cup_holder_0·1·2 순서이고 마커는 겹치지 않는다(마커 번호는 실물 따라 자유)."""
    raw = yaml.safe_load((REPO / "config/cup_holders.yaml").read_text())
    hs = raw["holders"]
    assert [h["name"] for h in hs] == [f"cup_holder_{i}" for i in range(len(hs))]
    assert len({h["marker_id"] for h in hs}) == len(hs)


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
    entries = {e.id: e for e in R.scan(REPO / "deploy" / "policies", deep=False)}
    if all(b.policy in entries and entries[b.policy].ok for b in S.load(path).policies.values()):
        assert rc == 0
    else:
        assert rc == 1  # Deleted policy bundles must remain unavailable.


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


def test_an_episode_object_can_follow_the_active_fpp_object_of_an_arm(tmp_path):
    """10.09 사용자: cyl60 · source200 · source240 을 바꿔 가며 쓴다 — 에피소드 물체가 config/fpp_active.yaml 의 팔별 물체를
    따른다(CUP: {active: right}). 활성을 바꾸면(fpp_object.py activate) 에피소드 파일은 그대로."""
    (tmp_path / "episodes").mkdir()
    (tmp_path / "fpp_active.yaml").write_text(yaml.safe_dump({"group": "g", "sides": {"right": "source200_pink",
                                                                                     "left": "source200_orange"}}))
    raw = _raw()
    raw["objects"] = {"CUP": {"active": "right"}}
    ep = S.parse(raw, path=str(tmp_path / "episodes" / "x.yaml"))
    assert ep.objects["CUP"]["topic"] == "/objects/source200_pink/pose"
    raw["objects"] = {"CUP": {"active": "middle"}}
    with pytest.raises(S.EpisodeSpecError, match="active"):
        S.parse(raw, path=str(tmp_path / "episodes" / "x.yaml"))


def test_the_shipped_pick_place_episodes_follow_the_active_objects():
    active = yaml.safe_load((REPO / "config" / "fpp_active.yaml").read_text())
    for name, side in (("pick_place_right", "right"), ("pick_place_left", "left")):
        ep = S.load(EPISODES / f"{name}.yaml")
        assert ep.objects["CUP"]["topic"] == f"/objects/{active['sides'][side]}/pose"


STALE_CUP = (False, ["reset: FP++ live 컵 자세가 0.5 s 안에 없다 ['arm'] — 컵이 보이는지 · live 추적(fpp 컨테이너)이 도는지 본다"])


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_a_reset_refused_for_a_stale_live_cup_is_retried_until_the_cup_comes_back():
    """10.14 실기 183230: FP++ live 추적이 5~6 s 마다 다시 등록하며 약 3.3 s 끊긴다 — 그 사이 reset 이 거부돼 에피소드가 안전 정지했다."""
    from policy_control.episode_ros import reset_until_cup_fresh
    answers = iter([STALE_CUP] * 3 + [(True, [])])
    clock, calls = _Clock(), []

    def call():
        calls.append(clock())
        return next(answers)

    ok, why = reset_until_cup_fresh(call, retry_s=6.0, sleep=clock.sleep, clock=clock)
    assert ok and why == [] and len(calls) == 4


def test_other_refusals_and_a_cup_that_never_comes_back_are_not_hidden():
    from policy_control.episode_ros import reset_until_cup_fresh
    other = (False, ["reset: 측정이 아직 없다 ['arm']"])
    calls = []
    ok, why = reset_until_cup_fresh(lambda: (calls.append(1), other)[1], retry_s=6.0,
                                    sleep=lambda s: None, clock=lambda: 0.0)
    assert (ok, why, len(calls)) == (False, other[1], 1)
    clock = _Clock()
    ok, why = reset_until_cup_fresh(lambda: STALE_CUP, retry_s=6.0, sleep=clock.sleep, clock=clock)
    assert not ok and why == STALE_CUP[1] and 6.0 <= clock() < 6.5


def test_the_stale_cup_retry_stops_when_the_runner_is_stopped():
    from policy_control.episode_ros import reset_until_cup_fresh
    calls = []
    ok, _ = reset_until_cup_fresh(lambda: (calls.append(1), STALE_CUP)[1], retry_s=6.0,
                                  sleep=lambda s: None, clock=lambda: 0.0, cancelled=lambda: True)
    assert not ok and len(calls) == 1


def test_the_retried_refusal_text_is_the_one_the_policy_node_sends():
    from policy_control.episode_ros import RESET_CUP_STALE_TEXT
    node = Path(__file__).resolve().parents[2] / "deploy/policy_control/policy_control/pour_fj_node.py"
    assert f"reset: {RESET_CUP_STALE_TEXT}" in node.read_text()

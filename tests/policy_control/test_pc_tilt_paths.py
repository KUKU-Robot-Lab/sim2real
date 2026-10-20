"""10.17 tilt 평가 경로 B · C 와 한 팔씩 이송(docs/PLAN_GRASP_TILT_EVAL_2026-10-17.md): 에피소드 형식(confirm · hand · carry
sides/cups), 실행기(확인 노드는 연속 실행에서도 멈춤), 손 쥐기 규칙(policy_control.hand_grasp), ROS 실행기의 거부 조건,
fake 손의 손 안 컵, 평가 기록(결정 1 · 4)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from policy_control import episode_spec as S
from policy_control import rh_aglt as A
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_failure import F
from policy_control.episode_runner import FAILURE, READY, SUCCESS, EpisodeManager
from policy_control.hand_grasp import GraspCfg, HandGrasp, HandGraspError

REPO = Path(__file__).resolve().parents[2]
EP = {k: REPO / "config" / "episodes" / f"tilt_base_{k}.yaml" for k in "abcd"}   # ★10.20 A 는 폐지(시험 고정본), D 가 대신
TOOLS = REPO / "deploy" / "policy_control" / "tools"
CONTRACT = REPO / "config" / "hand" / "rh56f1_right_hand.json"   # ★10.20 1-B 손 계약 고정본(옛 right_approach 계약 그대로)


def _tool(name: str, path: Path = TOOLS):
    spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 에피소드 형식
def test_three_paths_parse_and_share_the_carry_and_tilt():
    eps = {k: S.load(p) for k, p in EP.items()}
    for k, ep in eps.items():
        assert S.episode_sides(ep) == ["right", "left"]
        assert ep.trajectories["tilt"].sides == ("right", "left") and ep.trajectories["tilt"].cups
        assert (ep.trajectories["to_tilt_start_r"].sides, ep.trajectories["to_tilt_start_l"].sides) == (("right",), ("left",))
    assert not eps["b"].trajectories["to_tilt_start_r"].cups and eps["a"].trajectories["to_tilt_start_r"].cups
    # ★10.20 C = ap05 사전 파지 자리에서 멈춤(stop_ready) · D = 같은 자리에서 rcg03 이 넘겨받음(handoff)
    assert all(b.approach_grasp == {"stop_ready": 1} for b in eps["c"].policies.values())
    assert {b.handoff for b in eps["d"].policies.values()} == {"rh56f1/aglt/right_rcg03", "rh56f1/aglt/left_rcg03"}
    assert all(b.approach_grasp == {"handoff": 1} for b in eps["d"].policies.values())
    b = eps["b"]
    assert not b.policies and not any(n.type == "snapshot" for n in b.nodes)
    assert [n.id for n in b.nodes if n.type == "confirm"] == ["confirm_cup_right", "confirm_close_right", "confirm_cup_left",
                                                               "confirm_close_left"]
    # 확인 두 번(컵 대기 전 · 닫기 전) 뒤에 그 손 쥐기
    ids = [n.id for n in b.nodes]
    assert ids.index("confirm_close_right") + 1 == ids.index("grasp_right")
    assert ids.index("confirm_close_left") + 1 == ids.index("grasp_left")


def _b_raw():
    return yaml.safe_load(EP["b"].read_text(encoding="utf-8"))


@pytest.mark.parametrize("patch", [
    {"id": "x", "type": "confirm"},                                                        # text 없음
    {"id": "x", "type": "confirm", "text": "t", "expect": {"pose": "HOME"}},
    {"id": "x", "type": "hand", "action": "squeeze", "sides": ["right"]},
    {"id": "x", "type": "hand", "action": "open", "sides": []},
    {"id": "x", "type": "hand", "action": "grasp", "sides": ["right", "left"], "hand_contract": "a", "expect": {"right_hand": "SRC"}},
    {"id": "x", "type": "hand", "action": "grasp", "sides": ["right"], "expect": {"right_hand": "SRC"}},   # hand_contract 없음
    {"id": "x", "type": "hand", "action": "grasp", "sides": ["right"], "hand_contract": "a"},              # 쥘 물체 없음
    {"id": "x", "type": "hand", "action": "grasp", "sides": ["right"], "hand_contract": "a", "expect": {"right_hand": "SRC"},
     "cfg": {"close_rad_s": 9.0}},                                                         # 안전 범위 밖
    {"id": "x", "type": "hand", "action": "grasp", "sides": ["right"], "hand_contract": "a", "expect": {"right_hand": "SRC"},
     "cfg": {"lift_m": 0.1}},                                                              # 모르는 키
    {"id": "x", "type": "hand", "action": "open", "sides": ["right"], "expect": {"right_hand": "SRC"}},
])
def test_bad_confirm_and_hand_nodes_are_refused(patch):
    d = _b_raw()
    d["sequence"].insert(1, patch)
    with pytest.raises(S.EpisodeSpecError):
        S.parse(d, path=str(EP["b"]))


@pytest.mark.parametrize("traj", [{"kind": "carry", "sides": ["left", "right"], "asset": "tilt/base_arc6"},
                                  {"kind": "carry", "sides": ["both"], "asset": "tilt/base_arc6"},
                                  {"kind": "rehome", "sides": ["right"], "cups": False}])
def test_bad_carry_sides_and_cups_on_rehome_are_refused(traj):
    d = _b_raw()
    d["trajectories"]["to_tilt_start_r"] = traj
    with pytest.raises(S.EpisodeSpecError):
        S.parse(d, path=str(EP["b"]))


# ---------------------------------------------------------------- 실행기(모의)
def test_path_b_continuous_run_stops_at_every_confirm_until_that_node_is_approved():
    ep = S.load(EP["b"])
    ok: set = set()
    m = EpisodeManager(ep, FakeExecutor(ep), approve=lambda what, why: not what.startswith("confirm_") or what in ok,
                       episode_id="t")
    m.run()
    assert m.status == READY and m.waiting == "confirm_cup_right" and m.node.id == "confirm_cup_right"
    assert m.view()["confirm"] == {"node": "confirm_cup_right", "text": ep.nodes[m.index].text}
    assert m.world.pose == "SETTING" and m.world.hand("right") == "EMPTY"     # 빈손으로 시작 자세까지 왔다
    m.run()                                                                   # 승인 없이 다시 눌러도 그 자리
    assert m.node.id == "confirm_cup_right"
    for cid in ("confirm_cup_right", "confirm_close_right"):
        ok.add(cid)
        m.run()
    assert m.waiting == "confirm_cup_left" and m.world.hand("right") == "SRC" and m.world.objects["SRC"]["at"] == "right_hand"
    ok |= {"confirm_cup_left", "confirm_close_left"}
    m.run()
    assert m.status == SUCCESS and (m.world.hand("right"), m.world.hand("left")) == ("SRC", "RCV")
    calls = [c for c in m.executor.calls if c[0] in ("hand", "trajectory")]
    assert calls == [("trajectory", "go_home_start", "go_home"), ("trajectory", "to_tilt_start_r", "to_tilt_start_r"),
                     ("trajectory", "to_tilt_start_l", "to_tilt_start_l"), ("hand", "open_hands", "open", ("right", "left")),
                     ("hand", "grasp_right", "grasp", ("right",)), ("hand", "grasp_left", "grasp", ("left",)),
                     ("trajectory", "tilt", "tilt")]


def test_path_b_step_mode_asks_the_confirm_by_its_own_name():
    ep = S.load(EP["b"])
    asked = []
    m = EpisodeManager(ep, FakeExecutor(ep), approve=lambda what, why: asked.append((what, why)) or True, episode_id="t")
    while m.status not in (SUCCESS, FAILURE):
        m.step()
    names = [w for w, _ in asked]
    assert names.count("confirm_cup_right") == 1 and "episode:tilt_base_b" not in names
    assert dict(asked)["confirm_close_left"].startswith("확인: ")


def test_a_failed_hand_grasp_stops_without_recovery_and_names_the_hand():
    ep = S.load(EP["b"])
    ex = FakeExecutor(ep, {"grasp_left": ["grasp_failed_left"]})
    m = EpisodeManager(ep, ex, approve=lambda what, why: True, episode_id="t")
    m.run()
    assert m.status == FAILURE and m.last["code"] == F.GRASP_FAILED_LEFT.value and ex.stopped
    assert ("trajectory", "tilt", "tilt") not in ex.calls


def test_path_c_fake_run_carries_one_arm_at_a_time_after_both_grasps():
    ep = S.load(EP["c"])
    ex = FakeExecutor(ep)
    m = EpisodeManager(ep, ex, approve=lambda what, why: True, episode_id="t")
    m.run()
    assert m.status == SUCCESS and (m.world.hand("right"), m.world.hand("left")) == ("SRC", "RCV")
    seq = [c[1] for c in ex.calls if c[0] in ("policy", "trajectory")]
    assert seq == ["go_home_start", "grasp_right", "grasp_left", "to_tilt_start_r", "to_tilt_start_l", "tilt"]


# ---------------------------------------------------------------- ROS 실행기(ROS 없이)
def _ros(tmp_path, holds: dict, rcs: list[int], open_err: float | None = 0.01):
    from policy_control.episode_ros import RosExecutor
    calls = []

    def run(argv):
        calls.append(argv)
        return rcs[len(calls) - 1], "tail not grasped"
    me = SimpleNamespace(_wait_pd_tracking=lambda side: True, _pd_view=lambda side: {"pd_phase": "TRACKING", "pd_reasons": []},
                         run_dir=tmp_path, _run=run, cancel=threading.Event(), _hand_open_err=lambda side: open_err)
    world = SimpleNamespace(hand=lambda side: holds[side])
    return me, world, calls, RosExecutor


def test_a_one_arm_carry_plans_and_replays_only_that_arm(tmp_path):
    me, world, calls, R = _ros(tmp_path, {"right": "SRC", "left": "EMPTY"}, [0, 0])
    traj = S.load(EP["a"]).trajectories["to_tilt_start_r"]
    r = R._run_planned(me, SimpleNamespace(id="to_tilt_start_r"), traj, world)
    assert r.status == "completed"
    plan, rp = calls
    assert plan[plan.index("--sides") + 1] == "right"
    assert rp[rp.index("--joints") + 1] == ",".join(f"r_aj_{i}" for i in range(1, 8))


def test_an_empty_hand_carry_refuses_a_held_cup_and_a_cup_carry_refuses_an_empty_hand(tmp_path):
    me, world, calls, R = _ros(tmp_path, {"right": "SRC", "left": "EMPTY"}, [])
    b, a = S.load(EP["b"]).trajectories, S.load(EP["a"]).trajectories
    assert R._run_planned(me, SimpleNamespace(id="x"), b["to_tilt_start_r"], world).status == "refused"
    assert R._run_planned(me, SimpleNamespace(id="x"), a["to_tilt_start_l"], world).status == "refused"
    assert calls == []


def test_hand_open_refuses_a_held_cup_and_grasp_maps_the_tool_exit_codes(tmp_path):
    me, world, calls, R = _ros(tmp_path, {"right": "SRC", "left": "EMPTY"}, [2, 0])
    ep = S.load(EP["b"])
    node = {n.id: n for n in ep.nodes}
    assert R.run_hand(me, node["open_hands"], world).status == "refused" and calls == []
    r = R.run_hand(me, node["grasp_left"], world)                    # rc 2 = 못 쥠
    assert r.status == "aborted" and "not grasped" in r.reason
    argv = calls[0]
    assert Path(argv[1]).name == "hand_grasp.py" and argv[argv.index("--side") + 1] == "left" and "--execute" in argv
    assert argv[argv.index("--contract") + 1].endswith("config/hand/rh56f1_left_hand.json")   # ★10.20 1-B 손 계약 고정본
    r = R.run_hand(me, node["grasp_left"], world)
    assert r.status == "completed"


# ---------------------------------------------------------------- 손 쥐기 규칙
def _sim(hg: HandGrasp, stop: dict | None, ticks: int, drop_at: int | None = None, v: float = 1.5) -> list[str]:
    """손 모형: 축마다 v rad/s 로 목표를 따라가고, 컵이 있으면 stop 에서 더 닫히지 못한다."""
    names = hg.names
    q = np.array(hg.q_open, float)
    hg.start(q)
    phases = []
    for k in range(ticks):
        q_cmd = hg.step(q)
        nxt = q + np.clip(q_cmd - q, -v / hg.hz, v / hg.hz)
        if stop and (drop_at is None or k < drop_at):
            for i, n in enumerate(names):
                s = stop.get(n.split("_hj_")[1])
                if s is not None and nxt[i] > s:
                    nxt[i] = max(q[i], s) if q[i] <= s else q[i]
        q = nxt
        phases.append(hg.phase)
        if hg.finished:
            break
    return phases


STOP = {"thumb_2": 0.16, "index_1": 0.70, "middle_1": 0.70, "ring_1": 0.60, "pinky_1": 0.60}


def _hg(**over) -> HandGrasp:
    c = A.load_contract(CONTRACT)
    return HandGrasp(c, GraspCfg(**over), c.policy_hz)


def test_a_cup_in_the_hand_latches_then_holds_three_seconds():
    hg = _hg()
    ph = _sim(hg, STOP, 1200)
    assert ph[-1] == "done" and "hold" in ph and hg.latched_fingers() >= 2
    i_hold = ph.index("hold")
    assert len(ph) - i_hold == pytest.approx(3.0 * hg.hz, abs=2)
    th = hg.names.index("r_hj_thumb_1")
    assert hg.q_cmd[th] == pytest.approx(hg.q_open[th])                     # 엄지 회전은 그대로


def test_an_empty_hand_fails_after_close_max_without_latching():
    hg = _hg()
    ph = _sim(hg, None, 1200)
    assert ph[-1] == "failed" and "not grasped" in hg.why and "hold" not in ph
    assert len(ph) == pytest.approx(4.0 * hg.hz, abs=2)


def test_a_cup_pulled_out_during_the_hold_fails():
    hg = _hg()
    ph = _sim(hg, STOP, 1200, drop_at=170)
    assert "hold" in ph and ph[-1] == "failed" and "유지 중 풀림" in hg.why


def test_grasp_cfg_ranges_and_keys():
    with pytest.raises(HandGraspError):
        GraspCfg(judge_ticks=0)
    with pytest.raises(HandGraspError):
        GraspCfg.from_mapping({"near_m": 0.1})
    assert GraspCfg.from_mapping({"judge_min_axes": 3}).judge_min_axes == 3


def test_the_hand_grasp_tool_validates_without_ros():
    t = _tool("hand_grasp")
    assert t.main(["--side", "right", "--contract", str(CONTRACT)]) == 0
    assert t.main(["--side", "left", "--contract", str(CONTRACT)]) == 1        # 오른손 계약으로 왼손
    assert t.main(["--side", "right", "--contract", str(CONTRACT), "--cfg", '{"close_rad_s": 5}']) == 1


# ---------------------------------------------------------------- fake 손 · 평가 기록
def test_the_fake_hand_cup_stops_closing_but_never_opening():
    fh = _tool("fake_rh56f1_hand", REPO / "scripts" / "fakes")
    open_reg, stop = np.array([1740.0, 1350.0]), np.array([1410.0, 1266.0])
    on = np.array([True, False])
    cur = np.array([1420.0, 1300.0])
    out = fh.cup_clamp(np.array([1400.0, 1250.0]), cur, stop, open_reg, on)
    assert out.tolist() == [1410.0, 1250.0]                                   # 막는 축만 컵 면에서
    assert fh.cup_clamp(np.array([1500.0, 1300.0]), cur, stop, open_reg, on).tolist() == [1500.0, 1300.0]   # 여는 쪽


def test_eval_record_judges_with_the_d4_rule_and_numbers_the_trials(tmp_path):
    ev = _tool("eval_record")
    assert ev.judge_tilt(10, 10, 0, 0)["success"] and not ev.judge_tilt(10, 9, 0, 1)["success"]
    with pytest.raises(ev.EvalError):
        ev.judge_tilt(10, 8, 0, 1)                                            # 합이 안 맞으면 다시 센다
    assert not ev.judge_grasp("ok", 2.0)["success"] and ev.judge_grasp("ok", 3.0)["success"]
    assert ev.main(["--root", str(tmp_path), "tilt", "--path", "A", "--in-rcv", "10", "--in-src", "0", "--spilled", "0"]) == 0
    assert ev.main(["--root", str(tmp_path), "tilt", "--path", "A", "--in-rcv", "7", "--in-src", "1", "--spilled", "2"]) == 0
    assert ev.main(["--root", str(tmp_path), "tilt", "--path", "A", "--in-rcv", "7", "--in-src", "1", "--spilled", "1"]) == 2
    s = ev.summary(tmp_path)
    assert s["tilt"]["A"] == {"n": 2, "success": 1, "fill_mean": 0.85, "spilled_total": 2}
    rows = [json.loads(x) for x in next(tmp_path.glob("*/tilt.jsonl")).read_text().splitlines()]
    assert [r["trial"] for r in rows] == [1, 2]


def test_d4_success_numbers_match_the_training_run_when_it_is_here():
    cands = [Path(p).expanduser() for p in ("~/rl_ws/our_source/policy/pour_res_s2h4_ep5000/params/env.yaml",
                                             "~/rl_ws/s2r_2nd/replay_chain/pour/ref/d4/params/env.yaml")]   # arm4090 · 로컬 사본
    env = next((p for p in cands if p.is_file()), None)
    if env is None:
        pytest.skip("D4 params 가 이 호스트에 없다")
    ev = _tool("eval_record")
    keys = {}
    for line in env.read_text().splitlines():       # env.yaml 은 python 태그가 섞여 safe_load 가 못 읽는다 — 최상위 키 줄만
        k, sep, v = line.partition(": ")
        if sep and k in ev.D4_SUCCESS:
            keys[k] = yaml.safe_load(v)
    assert keys == ev.D4_SUCCESS


def test_the_start_check_names_a_missing_or_wrong_hand_contract(tmp_path):
    ep = S.load(EP["b"])
    assert S.hand_problems(ep, REPO / "deploy" / "policies") == {"grasp_right": "", "grasp_left": ""}
    from dataclasses import replace
    swapped = replace(ep, nodes=tuple(replace(n, hand_contract="config/hand/rh56f1_left_hand.json") if n.id == "grasp_right" else n
                                      for n in ep.nodes))
    assert "right 손 계약이 아니다" in S.hand_problems(swapped, REPO / "deploy" / "policies")["grasp_right"]
    assert "가 없다" in S.hand_problems(ep, tmp_path)["grasp_left"]


# ---------------------------------------------------------------- 리뷰 반영(10.17)
def test_hand_open_waits_for_the_measured_hand_to_open(tmp_path, monkeypatch):
    from policy_control import episode_ros as ER
    monkeypatch.setattr(ER, "HAND_OPEN_WAIT_S", 0.3)
    node = {n.id: n for n in S.load(EP["b"]).nodes}["open_hands"]
    me, world, calls, R = _ros(tmp_path, {"right": "EMPTY", "left": "EMPTY"}, [0, 0])
    assert R.run_hand(me, node, world).status == "completed" and len(calls) == 2
    for err in (0.5, None):                                          # 덜 펴짐 · 손 실측 없음
        me, world, calls, R = _ros(tmp_path, {"right": "EMPTY", "left": "EMPTY"}, [0, 0], open_err=err)
        r = R.run_hand(me, node, world)
        assert r.status == "error" and "안 펴졌다" in r.reason


def test_a_hand_grasp_needs_two_confirms_right_before_and_no_retry():
    d = _b_raw()
    d["sequence"] = [n for n in d["sequence"] if n["id"] != "confirm_close_left"]
    with pytest.raises(S.EpisodeSpecError, match="confirm 두 개"):
        S.parse(d, path=str(EP["b"]))
    d = _b_raw()
    d["failure_policy"] = {"default": 1}
    with pytest.raises(S.EpisodeSpecError, match="재시도 상한 0"):
        S.parse(d, path=str(EP["b"]))


def test_the_hand_tool_holds_the_arm_at_the_pd_setpoint_not_the_measurement():
    t = _tool("hand_grasp")
    meas = np.zeros(7)
    q, why = t.arm_hold_target({"arms": {"right": {"q_set": [0.01] * 7}}}, "right", meas)
    assert why == "" and np.allclose(q, 0.01)
    assert t.arm_hold_target({"arms": {"right": {}}}, "right", meas)[0] is None                     # 옛 pd
    assert t.arm_hold_target({"arms": {"right": {"q_set": [0.3] * 7}}}, "right", meas)[0] is None   # 못 따르는 중


def test_a_confirm_takes_only_an_approval_written_after_it_was_reached(tmp_path):
    import time as _t
    sys.path.insert(0, str(REPO / "deploy" / "policy_control" / "policy_control"))
    from policy_control.episode_runner_node import Approvals, write_approval
    path = tmp_path / "approvals.jsonl"
    ap = Approvals(path, _t.time() - 10)
    write_approval("confirm_cup_right", "op", path)          # 노드에 오기 전에 남은 승인
    _t.sleep(0.02)
    reached = _t.time()
    assert not ap.take("confirm_cup_right", since=reached)
    write_approval("confirm_cup_right", "op", path)
    assert ap.take("confirm_cup_right", since=reached) and not ap.take("confirm_cup_right", since=reached)
    ep = S.load(EP["b"])
    m = EpisodeManager(ep, FakeExecutor(ep), approve=lambda what, why: not what.startswith("confirm_"), episode_id="t")
    t0 = m.node_since
    m.run()
    assert m.node.id == "confirm_cup_right" and m.node_since >= t0


def test_a_hand_that_answers_late_is_not_taken_for_a_cup():
    """10.18 fake: 명령 뒤 0.6 s 동안 손이 안 움직였다 — 그 멈춤을 막힘으로 래치하면 안 된다(움직인 뒤의 멈춤만 막힘)."""
    hg = _hg()
    names, q = hg.names, np.array(hg.q_open, float)
    hg.start(q)
    ph, dead = [], 36                                      # 0.6 s 응답 지연
    for k in range(1200):
        q_cmd = hg.step(q)
        if k >= dead:
            nxt = q + np.clip(q_cmd - q, -1.5 / hg.hz, 1.5 / hg.hz)
            for i, n in enumerate(names):
                s = STOP.get(n.split("_hj_")[1])
                if s is not None and nxt[i] > s:
                    nxt[i] = max(q[i], s) if q[i] <= s else q[i]
            q = nxt
        ph.append(hg.phase)
        if hg.finished:
            break
    assert "hold" not in ph[:dead] and ph[-1] == "done"


def test_a_hand_that_never_moves_fails_and_says_so():
    hg = _hg()
    q = np.array(hg.q_open, float)
    hg.start(q)
    for _ in range(1200):
        hg.step(q)
        if hg.finished:
            break
    assert hg.phase == "failed" and "하나도 움직이지 않았다" in hg.why

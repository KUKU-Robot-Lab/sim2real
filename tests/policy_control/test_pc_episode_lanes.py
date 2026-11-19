"""★11.08 lanes — 두 팔이 각자 자기 순서(잡기 → tilt 시작 자세)로 가고, 둘 다 끝나면 다음 노드
(사용자 "왼손, 오른손 정책들은 기본이 독립으로 운영되는게 맞음"). 한 줄이 실패하면 다른 줄도 멈추고 복구 없이 정지."""
from __future__ import annotations

import copy
import dataclasses
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from policy_control import episode_spec as S
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_failure import F, NodeResult
from policy_control.episode_flow import flow_of
from policy_control.episode_runner import FAILURE, STOPPED, SUCCESS, EpisodeManager

EPISODES = Path(__file__).resolve().parents[2] / "config" / "episodes"
RAW = yaml.safe_load((EPISODES / "cocktail_e2e.yaml").read_text())
LANES_AT = next(i for i, n in enumerate(RAW["sequence"]) if n.get("type") == "lanes")


def _raw_with(edit):
    raw = copy.deepcopy(RAW)
    edit(raw["sequence"][LANES_AT])
    return raw


def _unsynced():
    """★11.18 sync 를 뺀 e4000 — 묶지 않은 줄 궤적(traj_lock 으로 한 팔씩)의 옛 동작을 본다."""
    def strip(node):
        for ln in node["lanes"]:
            for it in ln["nodes"]:
                it.pop("sync", None)
    return S.parse(_raw_with(strip))


def _lane(raw_node, side):
    return next(ln for ln in raw_node["lanes"] if ln["side"] == side)["nodes"]


def _run(ep, ex, **kw):
    rows = []
    m = EpisodeManager(ep, ex, approve=lambda w, y: True, log=rows.append, **kw)
    t0 = time.monotonic()
    return m, m.run(), rows, time.monotonic() - t0


def _ev(rows, event, sid):
    return next(i for i, r in enumerate(rows) if r["event"] == event and r["state_id"] == sid)


def test_cocktail_e4000_loads_as_one_lanes_node_with_the_guard_kept():
    ep = S.load(EPISODES / "cocktail_e2e.yaml")
    ids = [n.id for n in ep.nodes]
    i = ids.index("grasp_to_tilt_start")
    assert ids[i:i + 6] == ["grasp_to_tilt_start", "grasp_blue", "to_tilt_start_r1", "grasp_shaker", "to_tilt_start_l",
                            "tilt_blue"]
    lanes = ep.nodes[i]
    assert lanes.type == "lanes" and dict(lanes.expect) == {"pose": "SETTING"} and not lanes.jobs
    items = {n.id: n for n in ep.nodes if n.group == lanes.id}
    assert {k: n.lane for k, n in items.items()} == {"grasp_blue": "right", "to_tilt_start_r1": "right",
                                                       "grasp_shaker": "left", "to_tilt_start_l": "left"}
    assert items["grasp_blue"].jobs[0].lift_guard == ("SRC_PINK", 40.0, 0.08)          # 들기 가드 G40H8 그대로
    assert dict(items["grasp_shaker"].expect) == {"left_hand": "SHAKER"}
    assert not items["to_tilt_start_l"].expect                                        # 자세는 두 줄이 끝난 뒤(lanes expect)
    assert S.episode_sides(ep) == ["right", "left"]


def test_the_first_lane_to_finish_does_not_wait_for_the_other():
    ep = _unsynced()
    ex = FakeExecutor(ep, delay={"grasp_blue": 0.05, "to_tilt_start_r1": 0.05, "grasp_shaker": 0.6, "to_tilt_start_l": 0.05})
    m, status, rows, _ = _run(ep, ex)
    assert status == SUCCESS
    # 오른손 이송은 왼손 잡기가 끝나기 전에 시작하고 끝난다
    assert _ev(rows, "exit", "grasp_blue") < _ev(rows, "enter", "to_tilt_start_r1") < _ev(rows, "exit", "to_tilt_start_r1") \
        < _ev(rows, "exit", "grasp_shaker") < _ev(rows, "enter", "to_tilt_start_l")
    # 다음 노드(tilt_blue)는 두 줄이 다 끝난 뒤
    assert _ev(rows, "exit", "to_tilt_start_l") < _ev(rows, "exit", "grasp_to_tilt_start") < _ev(rows, "enter", "tilt_blue")
    assert {r["lane"] for r in rows if r["state_id"] == "grasp_shaker"} == {"left"}
    assert [c[1] for c in ex.calls if c[0] == "trajectory"].count("to_tilt_start_r1") == 1


def test_both_lanes_update_the_world_and_the_lanes_expect_comes_last():
    def add_flags(node):
        _lane(node, "right")[1]["expect"] = {"r_carried": True}
        _lane(node, "left")[1]["expect"] = {"l_carried": True}
    ep = S.parse(_raw_with(add_flags))
    ex = FakeExecutor(ep, delay={"to_tilt_start_r1": 0.02, "to_tilt_start_l": 0.02})
    m = EpisodeManager(ep, ex, approve=lambda w, y: True)
    m.index = ep.node_index("grasp_to_tilt_start")
    m.world = m.world.locate("SRC_BLUE", "table", (0.3, -0.2, 0.3)).locate("SHAKER", "table", (0.3, 0.2, 0.3))
    m.step()
    w = m.world
    assert (w.pose, w.hand("right"), w.hand("left")) == ("SETTING", "SRC_BLUE", "SHAKER")
    assert {"r_carried", "l_carried"} <= set(w.flags)
    assert (w.objects["SRC_BLUE"]["at"], w.objects["SHAKER"]["at"]) == ("right_hand", "left_hand")
    assert m.node.id == "tilt_blue"                                              # 줄 항목을 건너 다음 순서 노드로
    assert {k: (v["state"], v.get("lane")) for k, v in m.runs.items() if v.get("lane")} == {
        "grasp_blue": ("done", "right"), "to_tilt_start_r1": ("done", "right"),
        "grasp_shaker": ("done", "left"), "to_tilt_start_l": ("done", "left")}


@pytest.mark.parametrize("failing, other, code", [("grasp_shaker", "to_tilt_start_r1", F.GRASP_FAILED_LEFT),
                                                  ("grasp_blue", "grasp_shaker", F.CONTROLLER_ERROR)])
def test_a_failed_lane_stops_the_other_lane_at_once_without_recovery(failing, other, code):
    raw = copy.deepcopy(RAW)
    raw["failure_policy"] = {"default": 0}
    ep = S.parse(raw)
    ex = FakeExecutor(ep, inject={failing: [code.value]}, delay={failing: 0.1, other: 3.0})
    m, status, rows, dt = _run(ep, ex)
    assert status == FAILURE and dt < 2.0                                         # 다른 줄의 3 s 를 기다리지 않았다
    assert m.last["code"] == code.value and failing in m.last["reason"]
    assert rows[_ev(rows, "stopped", other)]["lane"] in ("right", "left")
    assert ex.stopped and failing in ex.stopped[0]                                # 실패 그 자리에서 safe_stop
    assert not m.attempts and not any(c[1] == "tilt_blue" for c in ex.calls)     # 재시도 · 다음 노드 없음
    assert m.runs[failing]["state"] == "failed" and m.runs[other]["state"] == "stopped"


def test_lanes_node_never_retries_or_continues_even_if_the_policy_allows_it():
    ep = dataclasses.replace(S.load(EPISODES / "cocktail_e2e.yaml"), failure_policy={"default": 3})
    ex = FakeExecutor(ep, inject={"grasp_shaker": [F.TARGET_NOT_FOUND.value]})     # 보통이면 RETRY 인 실패
    m, status, _, _ = _run(ep, ex)
    assert status == FAILURE and not m.attempts and m.pending is None


def test_operator_stop_during_lanes_stops_both_lanes():
    ep = S.load(EPISODES / "cocktail_e2e.yaml")
    ex = FakeExecutor(ep, delay={"grasp_blue": 3.0, "grasp_shaker": 3.0})
    rows = []
    m = EpisodeManager(ep, ex, approve=lambda w, y: True, log=rows.append)
    th = threading.Thread(target=m.run)
    th.start()
    time.sleep(0.3)
    m.stop("operator")
    th.join(timeout=2.0)
    assert not th.is_alive() and m.status == STOPPED
    assert {r["state_id"] for r in rows if r["event"] == "stopped"} >= {"grasp_blue", "grasp_shaker"}


def _reject(edit, match):
    with pytest.raises(S.EpisodeSpecError, match=match):
        S.parse(_raw_with(edit))


def test_spec_rejects_two_lanes_on_one_arm():
    def same(node):
        node["lanes"][1]["side"] = "right"
    _reject(same, "right 줄 하나")


def test_spec_rejects_an_item_using_the_other_arm():
    def swap(node):
        _lane(node, "right")[0], _lane(node, "left")[0] = _lane(node, "left")[0], _lane(node, "right")[0]
    _reject(swap, "줄 항목은 그 줄 팔 하나만")


def test_spec_rejects_a_two_arm_trajectory_in_a_lane():
    def tilt(node):
        _lane(node, "right")[1] = {"id": "tilt_in_lane", "type": "trajectory", "name": "tilt"}
    _reject(tilt, "줄 항목은 그 줄 팔 하나만")


@pytest.mark.parametrize("item, match", [
    ({"id": "c", "type": "confirm", "text": "x"}, "policy"),
    ({"id": "c", "type": "trajectory", "name": "to_tilt_start_r", "expect": {"pose": "SETTING"}}, "자세는 lanes"),
    ({"id": "c", "type": "trajectory", "name": "to_tilt_start_r", "expect": {"left_hand": "SHAKER"}}, "그 팔 손 칸"),
    ({"id": "c", "type": "trajectory", "name": "to_tilt_start_r", "checkpoint": "X"}, "checkpoint"),
])
def test_spec_rejects_bad_lane_items(item, match):
    def put(node):
        _lane(node, "right")[1] = item
    _reject(put, match)


def test_flow_shows_each_lane_and_no_feedback_for_lane_items():
    ep = S.load(EPISODES / "cocktail_e2e.yaml")
    f = flow_of(ep)
    box = next(n for n in f["nodes"] if n["id"] == "grasp_to_tilt_start")
    assert box["lines"][:2] == ["오른손 줄: grasp_blue → to_tilt_start_r1", "왼손 줄: grasp_shaker → to_tilt_start_l"]
    assert "오른손 줄" in next(n for n in f["nodes"] if n["id"] == "grasp_blue")["title"]
    assert not [fb for fb in f["feedback"] if fb["from"] in ("grasp_blue", "grasp_shaker")]


def test_ros_safe_stop_kills_every_running_tool_not_only_the_last(tmp_path):
    """두 줄이 replay_to_pd 를 동시에 띄운다 — safe_stop 이 하나만 끊으면 다른 팔이 계속 움직인다."""
    from policy_control import episode_ros as R
    me = _ros_me(tmp_path)
    out = {}
    ths = [threading.Thread(target=lambda k=k: out.__setitem__(k, R.RosExecutor._run(me, ["sleep", "30"]))) for k in range(2)]
    for th in ths:
        th.start()
    t0 = time.monotonic()
    while len(me._procs) < 2 and time.monotonic() - t0 < 5:
        time.sleep(0.02)
    R.RosExecutor.safe_stop(me, "test")
    for th in ths:
        th.join(timeout=5)
    assert all(rc != 0 for rc, _ in out.values()) and len(out) == 2 and not me._procs


def _ros_me(tmp_path):
    from policy_control import episode_ros as R
    me = SimpleNamespace(run_dir=tmp_path, _lock=threading.Lock(), _procs=set(), cancel=threading.Event(), sides=(),
                         _tool_seq=0)
    me._log_line = lambda text: R.RosExecutor._log_line(me, text)
    return me


# ---- ★11.08 리뷰 반영 ------------------------------------------------------------------------------------------

class _TrackingFake(FakeExecutor):
    """궤적이 동시에 몇 개 도는지 잰다."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.now, self.peak, self.mx = 0, 0, threading.Lock()

    def run_trajectory(self, node, traj, world):
        with self.mx:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            return super().run_trajectory(node, traj, world)
        finally:
            with self.mx:
                self.now -= 1


def test_lane_trajectories_run_one_arm_at_a_time_and_the_wait_is_logged():
    ep = _unsynced()
    ex = _TrackingFake(ep, delay={"to_tilt_start_r1": 0.4, "to_tilt_start_l": 0.4})
    m, status, rows, _ = _run(ep, ex)
    assert status == SUCCESS and ex.peak == 1
    waits = {r["state_id"]: r["traj_wait_s"] for r in rows if r["event"] == "plan"}
    assert set(waits) == {"to_tilt_start_r1", "to_tilt_start_l"} and max(waits.values()) >= 0.3


def test_planning_while_the_other_arm_runs_a_policy_is_noted():
    ep = _unsynced()
    ex = FakeExecutor(ep, delay={"grasp_blue": 0.1, "grasp_shaker": 0.6})
    m, status, rows, _ = _run(ep, ex)
    assert status == SUCCESS
    plan = next(r for r in rows if r["event"] == "plan" and r["state_id"] == "to_tilt_start_r1")
    assert "다른 팔 정책 진행 중(left:grasp_shaker)" in plan["other_arm_policy"]
    ex_row = next(r for r in rows if r["event"] == "exit" and r["state_id"] == "to_tilt_start_r1")
    assert "다른 팔 정책 진행 중" in ex_row["termination_reason"]
    assert "다른 팔 정책 진행 중" in m.runs["to_tilt_start_r1"]["note"]
    assert not any(r.get("other_arm_policy") for r in rows if r["event"] == "plan" and r["state_id"] == "to_tilt_start_l")


def test_the_approval_text_warns_and_names_arms_and_guard():
    ep = S.load(EPISODES / "cocktail_e2e.yaml")
    asked = []
    m = EpisodeManager(ep, FakeExecutor(ep), approve=lambda w, y: asked.append(y) or False)
    m.run()
    assert "두 팔 동시 진행: trajectory 는 한 팔씩, 정책 중인 팔과는 충돌 미검사" in asked[0]
    d = m._describe(next(n for n in ep.nodes if n.type == "lanes"))
    assert "right 팔: grasp_blue(" in d and "가드 SRC_PINK(40 mm · 0.08 m)" in d and "left 팔: grasp_shaker(" in d


class _SlowStopFake(FakeExecutor):
    """실기처럼 cancel 을 먼저 세우고 노드 stop 서비스에 시간이 걸린다."""

    def safe_stop(self, reason):
        self.cancel.set()
        time.sleep(0.4)
        self.stopped.append(reason)


def test_operator_stop_with_a_slow_safe_stop_is_stopped_not_failure():
    ep = S.load(EPISODES / "cocktail_e2e.yaml")
    ex = _SlowStopFake(ep, delay={"grasp_blue": 3.0, "grasp_shaker": 3.0})
    rows = []
    m = EpisodeManager(ep, ex, approve=lambda w, y: True, log=rows.append)
    th = threading.Thread(target=m.run)
    th.start()
    time.sleep(0.3)
    assert m.stop("operator stop") == STOPPED
    th.join(timeout=5)
    assert m.status == STOPPED and len(ex.stopped) == 1           # 줄이 safe_stop 을 다시 부르지 않았다
    assert not any(r["event"] == "failure" for r in rows)
    assert {m.runs[k]["state"] for k in ("grasp_blue", "grasp_shaker")} == {"stopped"}


def test_a_safety_code_from_the_other_lane_outranks_the_first_failure():
    ep = S.load(EPISODES / "cocktail_e2e.yaml")

    class Both(FakeExecutor):                                    # 왼손이 먼저 못 쥐고, 오른손은 가드 판정 실패(정지와 무관)
        def run_policies(self, node, plans, world):
            if node.id == "grasp_shaker":
                return [NodeResult("refused", "not grasped", 0.0, plans[0].role)]
            if node.id == "grasp_blue":
                time.sleep(0.2)
                return [NodeResult("aborted", "lift_guard hold failed: 상승 0 cm", 0.0, plans[0].role)]
            return super().run_policies(node, plans, world)

    m, status, rows, _ = _run(ep, Both(ep))
    assert status == FAILURE
    assert m.last["code"] == F.LIFT_GUARD_FAILED.value
    assert "grasp_blue(right) lift_guard_failed" in m.last["reason"] and "grasp_shaker(left) grasp_failed_left" in m.last["reason"]


def test_a_lane_cut_by_the_stop_is_stopped_even_if_its_result_looks_like_a_failure():
    ep = S.load(EPISODES / "cocktail_e2e.yaml")

    class Cut(FakeExecutor):
        def run_policies(self, node, plans, world):
            if node.id == "grasp_shaker":
                return [NodeResult("refused", "not grasped", 0.0, plans[0].role)]
            if node.id == "grasp_blue":
                self.cancel.wait(3.0)
                return [NodeResult("aborted", "episode runner stop — 가드 고정 판정 중 정지", 0.0, plans[0].role,
                                   {"guard_hold_cancelled": True, "guard_hold_ok": False})]
            return super().run_policies(node, plans, world)

    ex = Cut(ep)
    m, status, rows, _ = _run(ep, ex)
    assert m.last["code"] == F.GRASP_FAILED_LEFT.value and m.runs["grasp_blue"]["state"] == "stopped"
    assert sum("다른 줄도 멈춘다" in r for r in ex.stopped) == 1


def test_lanes_node_expect_with_a_hand_slot_is_rejected():
    raw = copy.deepcopy(RAW)
    raw["sequence"][LANES_AT]["expect"] = {"pose": "SETTING", "right_hand": "SRC_BLUE"}
    with pytest.raises(S.EpisodeSpecError, match="lanes 노드 expect 는 자세 · 플래그만"):
        S.parse(raw)


def test_ros_run_writes_each_tool_to_its_own_log_and_reads_its_tail(tmp_path):
    from policy_control import episode_ros as R
    me = _ros_me(tmp_path)
    out = {}
    ths = [threading.Thread(target=lambda k=k: out.__setitem__(k, R.RosExecutor._run(
        me, ["sh", "-c", f"for i in 1 2 3; do echo {k}-$i; sleep 0.05; done"]))) for k in ("a", "b")]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=5)
    assert out == {"a": (0, "a-3"), "b": (0, "b-3")}
    assert len(list((tmp_path / "tools").glob("*.log"))) == 2 and "→ tools/" in (tmp_path / "tools.log").read_text()


def test_ros_run_times_out_and_does_not_launch_after_stop(tmp_path, monkeypatch):
    from policy_control import episode_ros as R
    from policy_control.episode_failure import _from_reason
    me = _ros_me(tmp_path)
    monkeypatch.setitem(R.TOOL_TIMEOUT_S, "-c", 0.3)
    t0 = time.monotonic()
    rc, tail = R.RosExecutor._run(me, ["sh", "-c", "sleep 30"])
    assert rc == 124 and tail.startswith("tool timeout") and time.monotonic() - t0 < 5 and not me._procs
    assert _from_reason(f"plan_carry.py rc {rc}: {tail}", None) is F.CONTROLLER_ERROR
    me.cancel.set()
    rc, tail = R.RosExecutor._run(me, ["sh", "-c", "touch launched"])
    assert rc != 0 and tail.startswith("episode runner stop") and not (tmp_path / "launched").exists()


def test_replay_limit_scales_with_npz_length_and_rate(tmp_path):
    """★11.08 재검토 — 재생 상한은 궤적 길이 / rate-scale + 여유, 고정 600 s 아래로는 내려가지 않는다."""
    import numpy as np
    from policy_control import episode_ros as er

    npz = tmp_path / "carry_cmd.npz"
    np.savez(npz, fabric_q=np.zeros((3000, 7)), meta_step_dt=np.array(0.02))   # 60 s 궤적
    argv = ["py", "replay_to_pd.py", "--npz", str(npz), "--rate-scale", "0.1"]
    assert er._replay_limit(argv) == 60.0 / 0.1 + er.REPLAY_TIMEOUT_MARGIN_S
    short = ["py", "replay_to_pd.py", "--npz", str(npz), "--rate-scale", "1.0"]
    assert er._replay_limit(short) == er.TOOL_TIMEOUT_S["replay_to_pd.py"]
    missing = ["py", "replay_to_pd.py", "--npz", str(tmp_path / "none.npz")]
    assert er._replay_limit(missing) == er.TOOL_TIMEOUT_S["replay_to_pd.py"]


# ---- ★11.18 sync — 두 팔이 멈춘 뒤 함께 계획, 한 시각표로 재생(사용자 "계획은 함께, 재생은 동시") ------------------------

GRASP = {"right": "grasp_blue", "left": "grasp_shaker"}
CARRY = {"right": "to_tilt_start_r1", "left": "to_tilt_start_l"}
JOINT = "+".join(CARRY.values())


def _sync_raw(edit=lambda node: None):
    """sync 시험용 raw — 운영 yaml 이 sync 를 켜 두었든 꺼 두었든 두 carry 항목에 sync: to_tilt 를 직접 넣는다."""
    def on(node):
        for side in CARRY:
            _lane(node, side)[1]["sync"] = "to_tilt"
        edit(node)
    return _raw_with(on)


def _synced():
    return S.parse(_sync_raw())


class _TrajFake(FakeExecutor):
    """result = 양팔 carry 가 돌려줄 결과(예외면 던진다). by = 궤적을 부른 스레드(늦게 온 줄이 양팔 carry 를 실행한다)."""

    def __init__(self, *a, result=None, **kw):
        super().__init__(*a, **kw)
        self.trajs, self.by, self.result = [], {}, result

    def run_trajectory(self, node, traj, world):
        self.trajs.append((node.id, traj.sides, traj.rate_scale, world.hand("right"), world.hand("left")))
        self.by[node.id] = threading.current_thread().name
        if isinstance(self.result, Exception) and node.id == JOINT:
            raise self.result
        if self.result is not None and node.id == JOINT:
            return self.result
        return dataclasses.replace(super().run_trajectory(node, traj, world),
                                   signals={"arms_clear": {"ok": True, "min_arm_point_m": 0.21, "window": {"ok": True}}})


def _carries(ex):
    """줄 carry 로 돈 궤적 — 양팔(JOINT)이든 팔마다든(뒤 순서의 to_tilt_start_r2 는 뺀다)."""
    return [t for t in ex.trajs if t[0] in (JOINT, *CARRY.values())]


@pytest.mark.parametrize("late", ["left", "right"])            # 늦게 온 줄이 양팔 carry 를 실행한다 — 어느 팔이 늦어도 같다
def test_synced_carries_wait_for_both_grasps_then_run_once_for_both_arms(late):
    early = "left" if late == "right" else "right"
    ep = _synced()
    ex = _TrajFake(ep, delay={GRASP[early]: 0.05, GRASP[late]: 0.5})
    m, status, rows, _ = _run(ep, ex)
    assert status == SUCCESS
    # 양팔 carry 한 번 — 두 팔 다, 두 손 다 컵을 쥔 World, yaml 배속 그대로
    rate = ep.trajectories["to_tilt_start_r"].rate_scale
    assert _carries(ex) == [(JOINT, ("right", "left"), rate, "SRC_BLUE", "SHAKER")]
    assert ex.by[JOINT] == f"lane_{late}"
    plans = {r["state_id"]: r for r in rows if r["event"] == "plan"}
    assert set(plans) == set(CARRY.values())
    assert _ev(rows, "exit", GRASP[late]) < min(_ev(rows, "plan", k) for k in plans)      # 늦은 잡기가 끝난 뒤에 계획
    assert plans[CARRY[early]]["traj_wait_s"] >= 0.3 and plans[CARRY[late]]["traj_wait_s"] < 0.3
    # 먼저 온 줄만 기다렸다 — 그 까닭이 상황판 note 와 jsonl 한 줄(sync_wait)에 남는다
    assert [(r["state_id"], r["lane"]) for r in rows if r["event"] == "sync_wait"] == [(CARRY[early], early)]
    assert m.runs[CARRY[early]]["note"].startswith("sync 대기") and GRASP[late] in m.runs[CARRY[early]]["note"]
    assert "note" not in m.runs[CARRY[late]]
    for k in plans:
        ex_row = rows[_ev(rows, "exit", k)]
        assert ex_row["sync"] == "to_tilt" and ex_row["joint_plan"] == JOINT and ex_row["arms_clear"]["min_arm_point_m"] == 0.21
        assert m.runs[k]["state"] == "done"


def _both_failed_once(m, ex, rows, text):
    """두 carry 항목이 같은 사유로 failed · 줄이 부른 safe_stop 은 한 번(그 뒤 노드 _fail 이 한 번 더) · 다음 노드는 안 돈다."""
    assert m.status == FAILURE and text in m.last["reason"]
    assert [m.runs[k]["state"] for k in CARRY.values()] == ["failed", "failed"]
    assert all(text in rows[_ev(rows, "failure", k)]["termination_reason"] for k in CARRY.values())
    assert ["다른 줄도 멈춘다" in r for r in ex.stopped] == [True, False]
    assert sum(text in r["reason"] for r in m.last["results"]) == 1     # 두 줄이 같이 받은 결과 — 노드 결과에는 한 번만
    assert not any(c[1] == "tilt_blue" for c in ex.calls)


@pytest.mark.parametrize("result, text", [
    (NodeResult("refused", "plan_carry.py rc 2: arms_clear window 0.08 m"), "plan_carry.py rc 2"),   # 계획 거부 — 아무 팔도 안 움직였다
    (NodeResult("error", "replay_to_pd.py rc 1: start gap"), "replay_to_pd.py rc 1"),                # 양팔 재생 실패(rc ≠ 0)
    (RuntimeError("실행기 예외"), "RuntimeError: 실행기 예외"),                                         # 실행기가 예외를 냈다
])
def test_a_joint_carry_that_fails_fails_both_items_and_stops_the_lanes_once(result, text):
    ep = _synced()
    ex = _TrajFake(ep, result=result)
    m, _, rows, _ = _run(ep, ex)
    assert [t[0] for t in _carries(ex)] == [JOINT]              # 양팔 carry 한 번 — 팔마다 다시 돌리지 않는다
    _both_failed_once(m, ex, rows, text)


def test_an_error_before_the_joint_carry_runs_stays_in_both_failure_rows():
    """함께 계획 앞부분(plan 기록 · 궤적 조회) 예외 — 그 문구가 실행하던 줄 항목의 failure 행에도 남고 running 으로 남지 않는다."""
    ep = _synced()
    ex, rows = _TrajFake(ep), []

    def log(row):
        if row["event"] == "plan":
            raise OSError("jsonl 못 씀")
        rows.append(row)
    m = EpisodeManager(ep, ex, approve=lambda w, y: True, log=log)
    m.run()
    assert not _carries(ex)                                     # 계획 전 — 아무 팔도 안 움직였다
    _both_failed_once(m, ex, rows, "OSError: jsonl 못 씀")


def _stop_after(ep, ex, wait_s):
    rows = []
    m = EpisodeManager(ep, ex, approve=lambda w, y: True, log=rows.append)
    th = threading.Thread(target=m.run)
    th.start()
    time.sleep(wait_s)
    m.stop("operator stop")
    th.join(timeout=5)
    assert not th.is_alive() and m.status == STOPPED and len(ex.stopped) == 1    # 줄이 safe_stop 을 다시 부르지 않았다
    assert not any(r["event"] == "failure" for r in rows)
    return m, rows


def test_operator_stop_while_one_arm_waits_at_the_sync_stops_without_planning():
    ep = _synced()
    ex = _TrajFake(ep, delay={"grasp_blue": 0.05, "grasp_shaker": 3.0})
    m, rows = _stop_after(ep, ex, 0.3)
    assert not _carries(ex) and not any(r["event"] == "plan" for r in rows)
    assert {k: m.runs[k].get("state") for k in ("grasp_blue", "to_tilt_start_r1", "grasp_shaker", "to_tilt_start_l")} == {
        "grasp_blue": "done", "to_tilt_start_r1": "stopped", "grasp_shaker": "stopped", "to_tilt_start_l": None}
    assert m.runs["to_tilt_start_r1"]["note"].startswith("sync 대기")


def test_operator_stop_during_the_joint_replay_stops_both_items():
    ep = _synced()
    ex = _TrajFake(ep, delay={JOINT: 3.0})
    m, rows = _stop_after(ep, ex, 0.3)
    assert [t[0] for t in _carries(ex)] == [JOINT]
    assert [m.runs[k]["state"] for k in CARRY.values()] == ["stopped", "stopped"]
    assert (m.world.hand("right"), m.world.hand("left")) == ("SRC_BLUE", "SHAKER")   # 쥔 컵은 정지에도 World 에 그대로


def test_a_failed_grasp_stops_the_arm_waiting_at_the_sync_without_planning():
    ep = _synced()
    ex = _TrajFake(ep, inject={"grasp_shaker": [F.GRASP_FAILED_LEFT.value]}, delay={"grasp_shaker": 0.3})
    m, status, rows, dt = _run(ep, ex)
    assert status == FAILURE and dt < 2.0
    assert not _carries(ex)
    assert m.runs["to_tilt_start_r1"]["state"] == "stopped"


def test_unsynced_lane_trajectories_still_run_one_at_a_time_per_arm():
    ep = _unsynced()
    ex = _TrajFake(ep)
    m, status, _, _ = _run(ep, ex)
    assert status == SUCCESS
    assert sorted(t[:2] for t in ex.trajs if t[0] in ("to_tilt_start_r1", "to_tilt_start_l")) == [("to_tilt_start_l", ("left",)), ("to_tilt_start_r1", ("right",))]


def test_the_approval_text_names_the_synced_pair():
    ep = _synced()
    m = EpisodeManager(ep, FakeExecutor(ep), approve=lambda w, y: False)
    assert f"sync to_tilt: {JOINT} 은 두 팔이 멈춘 뒤 함께 계획" in m._describe(ep.nodes[ep.node_index("grasp_to_tilt_start")])


@pytest.mark.parametrize("edit, match", [
    (lambda n: _lane(n, "left")[1].pop("sync"), "right 줄 · left 줄 항목 하나씩"),
    (lambda n: _lane(n, "left")[0].update(sync="to_tilt"), "carry 궤적 항목만"),
    (lambda n: _lane(n, "left")[1].update(sync="other"), "right 줄 · left 줄 항목 하나씩"),
])
def test_spec_rejects_bad_sync(edit, match):
    with pytest.raises(S.EpisodeSpecError, match=match):
        S.parse(_sync_raw(edit))


@pytest.mark.parametrize("bad", [False, True, 0, "", " ", None])
def test_spec_rejects_a_sync_that_is_not_a_non_empty_string(bad):
    """두 줄 다 sync: false · 0 · 빈 값이면 조용히 sync 없는 직렬 경로로 빠졌다(true 는 "True" 라는 키가 됐다)."""
    def both(node):
        for side in CARRY:
            _lane(node, side)[1]["sync"] = bad
    with pytest.raises(S.EpisodeSpecError, match="sync 는 비어 있지 않은 문자열"):
        S.parse(_raw_with(both))


def test_spec_rejects_sync_outside_lanes_and_mismatched_rate():
    raw = copy.deepcopy(RAW)
    raw["sequence"][-2]["sync"] = "x"
    with pytest.raises(S.EpisodeSpecError, match="sync 는 lanes 줄의 carry 궤적 항목에만"):
        S.parse(raw)
    raw = _sync_raw()
    raw["trajectories"]["to_tilt_start_l"]["rate_scale"] = 1.0
    with pytest.raises(S.EpisodeSpecError, match="asset · rate_scale · cups 가 같아야"):
        S.parse(raw)


def _ros_carry(tmp_path, rcs, left="SHAKER", check='{"arms_clear": {"ok": false, "min_arm_point_m": 0.08}}', warn=lambda msg: None):
    from policy_control.episode_ros import RosExecutor
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[1].endswith("plan_carry.py"):
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "check.json").write_text(check)
        return rcs[len(calls) - 1], "tail"
    me = SimpleNamespace(_tin_rec=None, ep=SimpleNamespace(tin_fallback=""), _wait_pd_tracking=lambda side: True,
                         _wait_arm_still=lambda sides: {"arm_still": True}, _pd_view=lambda side: {}, run_dir=tmp_path,
                         _run=run, cancel=threading.Event(), _log_line=warn,
                         node=SimpleNamespace(get_logger=lambda: SimpleNamespace(warning=warn)))
    base = S.load(EPISODES / "cocktail_e2e.yaml").trajectories["to_tilt_start_r"]
    traj = dataclasses.replace(base, sides=("right", "left"))
    world = SimpleNamespace(hand=lambda side: {"right": "SRC_BLUE", "left": left}[side])
    return (lambda: RosExecutor._run_planned(me, SimpleNamespace(id=JOINT), traj, world)), calls


def test_ros_joint_carry_plans_both_arms_and_a_refused_plan_replays_nothing(tmp_path):
    go, calls = _ros_carry(tmp_path, [2])
    r = go()
    assert r.status == "refused" and len(calls) == 1 and r.signals["arms_clear"]["min_arm_point_m"] == 0.08
    plan = calls[0]
    assert plan[plan.index("--sides") + 1] == "right,left" and "--empty" not in plan


def test_ros_joint_carry_replays_fourteen_joints_in_one_process(tmp_path):
    go, calls = _ros_carry(tmp_path, [0, 0])
    assert go().status == "completed" and len(calls) == 2
    rp = calls[1]
    assert rp[rp.index("--joints") + 1] == ",".join([*(f"r_aj_{i}" for i in range(1, 8)), *(f"l_aj_{i}" for i in range(1, 8))])
    assert rp[rp.index("--rate-scale") + 1] == "0.5" and "--execute" in rp


def test_ros_carry_refused_before_planning_does_not_log_an_earlier_attempts_check(tmp_path):
    (tmp_path / f"carry_{JOINT}").mkdir()
    (tmp_path / f"carry_{JOINT}" / "check.json").write_text('{"arms_clear": {"ok": true}}')   # 같은 런의 앞 시도가 남긴 파일
    go, calls = _ros_carry(tmp_path, [0, 0], left="EMPTY")
    r = go()
    assert r.status == "refused" and not calls and "arms_clear" not in r.signals


@pytest.mark.parametrize("check", ["", "[1, 2]", '{"ok": true}'])         # 쓰다 만 파일 · 최상위 list · arms_clear 없음
def test_ros_carry_with_an_unreadable_check_stays_completed_and_warns(tmp_path, check):
    warned = []
    go, calls = _ros_carry(tmp_path, [0, 0], check=check, warn=warned.append)
    r = go()
    assert r.status == "completed" and len(calls) == 2 and "arms_clear" not in r.signals   # 이미 재생한 carry 는 그대로
    assert len(warned) == 2 and all("arms_clear 를 못 읽었다" in w for w in warned)        # ROS logger · tools.log 둘 다


def test_ros_carry_stays_completed_when_the_warning_itself_cannot_be_written(tmp_path):
    def boom(msg):
        raise OSError("tools.log 못 씀")
    go, calls = _ros_carry(tmp_path, [0, 0], check="", warn=boom)
    r = go()
    assert r.status == "completed" and len(calls) == 2 and "arms_clear" not in r.signals   # 경고 기록이 예외를 내도 그대로


def test_plan_carry_window_catches_a_pass_that_only_clears_at_the_same_time(monkeypatch):
    """★11.18 함께 계획 ±창 — 같은 시각에는 멀어도, 한 팔이 조금 늦게 출발하면 겹치는 쌍을 잡는다."""
    import importlib.util
    tools = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("plan_carry", tools / "plan_carry.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    import torch
    t = torch.linspace(0.0, 1.0, 61)
    right = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], 1)[:, None, :]          # x 0 → 1 m
    left = torch.stack([torch.full_like(t, 0.5), 0.5 - 0.5 * t, torch.zeros_like(t)], 1)[:, None, :]   # (0.5, 0.5 → 0)
    same, win = pc._pair_min(right, left, 0), pc._pair_min(right, left, 30)
    assert same > 0.2 and win < 0.01                       # 같은 시각 최소 0.22 m · 0.5 s 어긋나면 (0.5, 0) 에서 만난다
    assert pc._pair_min(right, left, 30) <= pc._pair_min(right, left, 10) <= same
    # lag 0 지름길(프레임별 cdist) = 마스크 방식(전 프레임 쌍 행렬의 대각) 값 그대로 — 프레임당 점 2 · 3 개
    x, y = torch.cat([right, right + 0.3], 1), torch.cat([left, left - 0.2, left + 0.1], 1)
    d = torch.cdist(x.reshape(-1, 3), y.reshape(-1, 3), compute_mode="donot_use_mm_for_euclid_dist").view(61, 2, 61, 3)
    assert pc._pair_min(x, y, 0) == float(d.amin(dim=(1, 3)).diagonal().min())

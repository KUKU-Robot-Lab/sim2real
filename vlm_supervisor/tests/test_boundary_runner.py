"""경계 러너 — 진짜 EpisodeManager + FakeExecutor + 실기 yaml(cocktail_1c_e4000), 가짜 시계 · 판정기 · 사진."""
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from policy_control import episode_spec as S
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_failure import F
from policy_control.episode_runner import FAILURE, READY, STOPPED, SUCCESS, EpisodeManager
from vlm_supervisor import runner as R
from vlm_supervisor.runner import HANDED, BoundaryRunner, Question

EP = S.load(Path(__file__).resolve().parents[2] / "config" / "episodes" / "cocktail_1c_e4000.yaml")
ORDER = [n.id for n in EP.nodes if not n.lane and n.type != "terminal"]     # 경계 = 순서 노드(lanes 는 한 덩어리)
EID = "ep_test"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class Frames:
    def __init__(self, clock):
        self.clock, self.calls = clock, []

    def frame(self, after):
        self.calls.append(after)
        return b"jpg", self.clock()


class Arm:
    def __init__(self, still=True):
        self.still = still

    def arm_still(self):
        return self.still


class FakeJudge:
    """ask 호출 순서대로 plan 을 하나씩 쓴다(없으면 사진 '예' · advance). 항목: pick · action · delay(가짜 시계) · block · pre."""

    def __init__(self, clock, plan=()):
        self.clock, self.plan, self.asks, self.decides = clock, list(plan), [], 0
        self._tl = threading.local()

    def ask(self, images, questions):
        self.asks.append((list(images), list(questions)))
        e = self._tl.e = self.plan.pop(0) if self.plan else {}
        if "pre" in e:
            e["pre"]()
        self.clock.t += e.get("delay", 0.0)
        if "block" in e:
            e["block"].wait(5)
        pick = e.get("pick", "yes")
        if pick == "raise":
            raise RuntimeError("vlm down")
        if pick == "low":
            return [{"scores": {}, "low_mass": True} for _ in questions]
        if pick == "bad":
            return ["yes"]
        if "scores" in e:
            return [{"scores": e["scores"]} for _ in questions]
        return [{"scores": {"yes": 0.9, "no": 0.1} if pick == "yes" else {"yes": 0.2, "no": 0.8}} for _ in questions]

    def decide(self, answers, signal, allowed):
        self.decides += 1
        if self._tl.e.get("action") == "raise":
            raise RuntimeError("decide down")
        return self._tl.e.get("action", "advance"), "fake"


def grant(path, **kw):
    line = {"operator": "yoon", "episode_id": EID, "scope": "episode", "node": "", "action": "", "since": 100.0, **kw}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def make(tmp_path, mode="auto", plan=(), inject=None, arm=None, ep=EP, frames=Frames, **kw):
    clock, recs = Clock(), []
    ex, judge = FakeExecutor(ep, inject=inject), FakeJudge(clock, plan)
    path = tmp_path / "vlm_grants.jsonl"
    args = dict(mode=mode, judge=judge, frames=frames(clock), arm=arm or Arm(), sink=recs.append, poll_s=0.01,
                questions=lambda n: [Question(f"{n.id} 됐나?")], clock=clock, sleep=clock.sleep, wall=clock, episode_id=EID)
    args.update({"approve": lambda w, y: True} if mode == "record" else {"grants_path": path})
    args.update(kw)
    return BoundaryRunner(ep, ex, **args), ex, judge, recs, path, clock


def ran(ex):
    return [c[1] for c in ex.calls]


def bounds(recs):
    return [r for r in recs if "event" not in r]


def test_auto_advances_through_every_boundary_and_lanes_is_one(tmp_path):
    r, ex, judge, recs, path, _ = make(tmp_path)
    grant(path)
    assert r.run() == SUCCESS
    plain = FakeExecutor(EP)
    assert EpisodeManager(EP, plain, approve=lambda w, y: True).run() == SUCCESS
    assert ex.calls == plain.calls                                          # 순서 · 실행은 기존 러너와 같다
    b = bounds(recs)
    assert [x["node"] for x in b] == ORDER and [x["seq"] for x in b] == list(range(1, len(ORDER) + 1))
    assert all(x["decision"] == "advance" and x["stamp"]["after"] >= x["stamp"]["before"] for x in b)
    assert len(judge.asks) == len(ORDER) and judge.asks[0][0] == [b"jpg", b"jpg"]
    lanes = next(x for x in b if x["node"] == "grasp_to_tilt_start")        # 줄 항목 넷이 한 경계의 신호에 들어 있다
    assert len(lanes["signal"]["results"]) == 4 and "grasp_blue" not in [x["node"] for x in b]
    assert recs[0]["event"] == "start" and recs[0]["next"] == ORDER[0]


@pytest.mark.parametrize("inject, pick", [(None, "no"), ({"place_blue": [F.PLACE_FAILED.value]}, "yes")])
def test_runner_and_photo_disagree_hands_off(tmp_path, inject, pick):
    at = ORDER.index("place_blue")
    r, ex, judge, recs, path, _ = make(tmp_path, plan=[{}] * at + [{"pick": pick}], inject=inject)
    grant(path)
    assert r.run() == HANDED and r.em.status == READY
    last = bounds(recs)[-1]
    assert last["node"] == "place_blue" and last["decision"] == "handoff" and "어긋남" in last["reason"]
    assert last["signal"]["ok"] is (inject is None) and judge.decides == at     # 어긋나면 2단을 부르지 않는다
    assert "home_r_after_blue" not in ran(ex)


@pytest.mark.parametrize("entry", [{"delay": 25.0}, {"delay": 25.0, "block": threading.Event()}, {"pick": "raise"},
                                   {"pick": "low"}, {"pick": "bad"}, {"action": "retry"}, {"action": "raise"},
                                   {"scores": {"yes": 0.5, "no": 0.5}}, {"scores": {"yes": 0.55, "no": 0.45}},
                                   {"scores": {"yes": float("nan"), "no": 0.1}}])
def test_no_late_or_malformed_answer_hands_off(tmp_path, entry):
    r, ex, _, recs, path, _ = make(tmp_path, plan=[entry])
    grant(path)
    assert r.run() == HANDED
    assert ran(ex) == [ORDER[0]] and bounds(recs)[-1]["decision"] == "handoff"
    if "delay" in entry:
        assert "마감" in bounds(recs)[-1]["reason"]
    entry.get("block", threading.Event()).set()


def test_stale_answer_from_earlier_seq_is_dropped(tmp_path):
    gate = threading.Event()
    r, ex, _, recs, path, clock = make(tmp_path, plan=[{"delay": 25.0, "block": gate, "action": "advance"},
                                                       {"action": "stop"}])
    grant(path)
    assert r.run() == HANDED
    gate.set()
    r._judging.join(5)                                      # 1번 경계의 늦은 advance 가 큐에 먼저 들어가 있다
    assert r._q.qsize() == 1
    clock.t += 10
    grant(path, since=clock())                              # 사람이 새로 승인
    assert r.run() == STOPPED                               # 늦은 advance 가 아니라 2번 경계의 stop 을 따른다
    assert ran(ex) == ORDER[:2] and len(ex.stopped) == 1
    last = bounds(recs)[-1]
    assert (last["seq"], last["decision"], last["discarded"]) == (2, "stop", 1)


def test_previous_judge_still_alive_hands_off(tmp_path):
    gate = threading.Event()
    r, ex, judge, recs, path, clock = make(tmp_path, plan=[{"delay": 25.0, "block": gate}])
    grant(path)
    assert r.run() == HANDED
    clock.t += 10
    grant(path, since=clock())
    assert r.run() == HANDED and "이전 판정" in bounds(recs)[-1]["reason"]
    assert ran(ex) == ORDER[:2] and len(judge.asks) == 1    # 새 판정을 시작하지 않았다
    gate.set()


def test_look_again_once_then_rejudge_twice_hands_off(tmp_path):
    r, ex, judge, recs, path, _ = make(tmp_path, plan=[{"action": "look_again"}, {}, {"action": "look_again"},
                                                       {"action": "look_again"}])
    grant(path)
    assert r.run() == HANDED
    b = bounds(recs)
    assert (b[0]["decision"], b[0]["looks"]) == ("advance", 1) and len(r.frames.calls) == 2 + 1 + 2 + 1
    assert b[1]["decision"] == "handoff" and "look_again" in b[1]["reason"] and len(judge.asks) == 4
    assert ran(ex) == ORDER[:2]


def test_failure_skips_judge_and_ends_grant(tmp_path):
    at = ORDER.index("grasp_pink")
    r, ex, judge, recs, path, _ = make(tmp_path, inject={"grasp_pink": [F.GRASP_FAILED_RIGHT.value]})
    grant(path)
    assert r.run() == FAILURE and len(ex.stopped) == 1
    assert len(judge.asks) == at and bounds(recs)[-1]["decision"] == "stop" and not bounds(recs)[-1]["answers"]
    assert r.grants.episode() is None


def test_record_mode_does_not_change_flow_even_if_judge_raises(tmp_path):
    r, ex, judge, recs, _, _ = make(tmp_path, mode="record", plan=[{"pick": "raise"}] * 20)
    assert r.em.ep is EP                                    # auto 를 끄지 않는다
    plain = FakeExecutor(EP)
    m = EpisodeManager(EP, plain, approve=lambda w, y: True)
    while m.step() == READY:
        pass
    assert r.run() == SUCCESS == m.status and ex.calls == plain.calls
    for t in r.workers:
        t.join(5)
    assert len(recs) == len(judge.asks) > 0 and all(x["decision"] == "handoff" for x in recs)   # 기록만, 진행은 그대로
    r2, _, _, recs2, _, _ = make(tmp_path, mode="record", questions=lambda n: [])
    assert r2.run() == SUCCESS and recs2 and all(x["decision"] == "unjudged" for x in recs2)
    with pytest.raises(ValueError):
        make(tmp_path, mode="record", grants_path=tmp_path / "g", approve=None)


def test_record_mode_keeps_outer_approval(tmp_path):
    r, ex, judge, _, _, _ = make(tmp_path, mode="record", approve=lambda w, y: False)
    assert r.run() == READY and not ex.calls and not judge.asks


def test_propose_needs_human_line_for_that_node_and_action(tmp_path):
    r, ex, _, recs, path, clock = make(tmp_path, mode="propose")
    grant(path)                                                             # 에피소드 승인은 propose 에서 안 통한다
    grant(path, scope="node", node=ORDER[1], action="advance")              # 다른 노드
    grant(path, scope="node", node=ORDER[0], action="stop")                 # 다른 결정
    assert r.run() == HANDED and not ex.calls and recs[-1]["event"] == "no_grant"
    assert clock() >= 130.0                                                 # hold_limit_s 만큼 기다렸다
    clock.t += 10
    grant(path, scope="node", node=ORDER[0], action="advance", since=clock())
    grant(path, scope="node", node=ORDER[1], action="advance", since=clock() - 0.5)   # 제안보다 먼저 쓴 줄 — 무시
    assert r.run() == HANDED and ran(ex) == [ORDER[0]]
    assert bounds(recs)[-1]["decision"] == "advance"                        # 제안은 났고 사람 승인이 없었다


@pytest.mark.parametrize("line", [{"operator": "vlm_supervisor"}, {"operator": " VLM"}, {"operator": ""},
                                  {"episode_id": "other"}, {"since": 99.0}, {"since": True}, {"since": float("nan")},
                                  {"since": float("inf")}, {"since": 1e9}, {"since": "100.0"}])
def test_auto_rejects_forged_foreign_or_old_grant(tmp_path, line):
    r, ex, _, _, path, _ = make(tmp_path)
    path.write_text("not json\n[1]\n")
    grant(path, **line)
    assert r.run() == HANDED and not ex.calls


def test_handoff_ends_grant_until_a_new_line(tmp_path):
    r, ex, _, _, path, clock = make(tmp_path, plan=[{}, {"action": "handoff"}])
    grant(path)
    assert r.run() == HANDED and ran(ex) == ORDER[:2]
    assert r.run() == HANDED and ran(ex) == ORDER[:2]       # 같은 줄로는 다시 못 간다
    grant(path, since=clock() + 3.0)                        # 종료 시각 + 시계 오차 허용(5 s) 안의 줄도 죽은 줄이다
    assert r.run() == HANDED and ran(ex) == ORDER[:2]
    clock.t += 10
    grant(path, since=clock())
    assert r.run() == SUCCESS


def test_budget_must_be_below_hold_limit(tmp_path):
    for budget, hold in ((30.0, 30.0), (31.0, 30.0), (0.0, 30.0)):
        with pytest.raises(ValueError):
            make(tmp_path, boundary_budget_s=budget, hold_limit_s=hold)


def test_arm_not_still_hands_off_without_asking(tmp_path):
    r, ex, judge, recs, path, clock = make(tmp_path, arm=Arm(still=False))
    grant(path)
    assert r.run() == HANDED and not judge.asks and ran(ex) == [ORDER[0]]
    assert "멈추지" in bounds(recs)[-1]["reason"] and bounds(recs)[-1]["elapsed_s"] >= 20.0


def test_stop_while_judging(tmp_path):
    box = {}
    r, ex, _, recs, path, _ = make(tmp_path, plan=[{"pre": lambda: box["r"].stop("operator")}])
    box["r"] = r
    grant(path)
    assert r.run() == STOPPED and ran(ex) == [ORDER[0]] and r.grants.episode() is None


@pytest.mark.parametrize("junk", [b'\xff\xfe{"operator": "yoon"}\n', b"[" * 100000 + b"\n",
                                  b'{"operator": "yoon", "episode_id": "ep_test", "scope": "episode", "since": 100'])
def test_broken_grant_file_is_no_grant_and_stop_still_reaches_robot(tmp_path, junk):
    r, ex, _, _, path, _ = make(tmp_path)
    path.write_bytes(junk)
    assert r.grants.episode() is None and r.run() == HANDED and not ex.calls
    assert r.stop("op") == STOPPED and ex.stopped == ["op"]


def boom(*a, **k):
    raise ZeroDivisionError("boom")


def test_safe_stop_goes_first_even_if_ending_the_grant_fails(tmp_path):
    r, ex, _, _, path, _ = make(tmp_path, plan=[{"action": "stop"}])
    grant(path)
    r.grants.end = boom
    with pytest.raises(ZeroDivisionError):
        r.run()                                             # 판정기의 stop — em.stop 뒤에 grant 종료가 터진다
    assert len(ex.stopped) == 1 and r.em.status == STOPPED
    r2, ex2, _, _, _, _ = make(tmp_path)
    r2.grants.end = boom
    with pytest.raises(ZeroDivisionError):
        r2.stop("op")
    assert ex2.stopped == ["op"]


def test_grant_unreadable_at_end_does_not_revive(tmp_path):
    box = {}
    r, ex, _, _, path, _ = make(tmp_path, plan=[{"pre": lambda: setattr(box["r"].grants, "path", tmp_path)}, ],
                                questions=lambda n: [Question("됐나?", ok="no")])     # 어긋남 → handoff
    box["r"] = r
    grant(path)
    assert r.run() == HANDED and ran(ex) == [ORDER[0]]      # 종료 순간 경로가 디렉터리 — 읽기 실패
    r.grants.path = path                                    # 회복
    assert r.grants.episode() is None and r.run() == HANDED and ran(ex) == [ORDER[0]]


def test_em_step_error_while_running_stops_and_ends_grant(tmp_path):
    r, ex, _, recs, path, _ = make(tmp_path)
    grant(path)
    ex.joints = boom                                        # 첫 노드 checkpoint — em.step() 이 RUNNING 중에 터진다
    with pytest.raises(ZeroDivisionError):
        r.run()
    assert ex.stopped == ["runner error"] and r.em.status == STOPPED and r.grants.episode() is None
    assert recs[-1]["event"] == "error" and "ZeroDivisionError" in recs[-1]["reason"]
    assert r.run() == STOPPED and ran(ex) == [ORDER[0]]


def test_runner_error_before_node_ends_grant_and_nothing_runs_after(tmp_path):
    r, ex, _, recs, path, _ = make(tmp_path)
    grant(path)
    step, r.em.step = r.em.step, boom
    with pytest.raises(ZeroDivisionError):
        r.run()
    r.em.step = step
    assert not ex.stopped and recs[-1]["event"] == "error"  # 돌던 것이 없어 정지는 안 부른다
    assert r.run() == HANDED and not ex.calls               # 옛 승인으로는 못 간다


@pytest.mark.parametrize("questions, why", [(boom, "질문 오류"), (lambda n: [], "질문 없음")])
def test_questions_error_or_empty_hands_off_and_ends_grant(tmp_path, questions, why):
    r, ex, judge, recs, path, _ = make(tmp_path, questions=questions)
    grant(path)
    assert r.run() == HANDED and why in bounds(recs)[-1]["reason"] and not judge.asks
    assert bool(bounds(recs)[-1]["errors"]) == (questions is boom)
    assert r.grants.episode() is None and r.run() == HANDED and ran(ex) == [ORDER[0]]


@pytest.mark.parametrize("mode", ["auto", "propose"])
def test_no_advance_without_a_record(tmp_path, mode):
    def sink(row):
        if "event" not in row:
            raise OSError("disk full")                      # 경계 기록만 실패

    r, ex, _, _, path, clock = make(tmp_path, mode=mode, sink=sink)
    grant(path)
    grant(path, scope="node", node=ORDER[0], action="advance")
    assert r.run() == HANDED and ran(ex) == [ORDER[0]] and r._next is None
    assert r.grants.episode() is None
    r2, ex2, _, _, path2, _ = make(tmp_path, mode=mode, sink=boom)      # 시작 기록부터 실패 — 아무것도 안 돈다
    assert r2.run() == HANDED and not ex2.calls


def test_record_mode_flow_survives_sink_error(tmp_path):
    r, ex, _, _, _, _ = make(tmp_path, mode="record", sink=boom)
    plain = FakeExecutor(EP)
    assert EpisodeManager(EP, plain, approve=lambda w, y: True).run() == SUCCESS
    assert r.run() == SUCCESS and ex.calls == plain.calls


@pytest.mark.parametrize("action", ["advance", "look_again"])
def test_failed_runner_signal_never_advances(tmp_path, action):
    """실제 EpisodeManager 의 놓기 실패 뒤 이어 가기(_continue_after_fail) — 사진도 "실패" 라 어긋남은 없다."""
    at = ORDER.index("place_blue")
    r, ex, judge, recs, path, _ = make(tmp_path, plan=[{}] * at + [{"pick": "no", "action": action}],
                                       inject={"place_blue": [F.PLACE_FAILED.value]})
    grant(path)
    assert r.run() == HANDED and r.em.place_failed_continued == ["place_blue"]
    last = bounds(recs)[-1]
    assert last["signal"]["ok"] is False and last["decision"] == "handoff" and "허용 밖" in last["reason"]
    assert judge.decides == at + 1 and "home_r_after_blue" not in ran(ex) and r.grants.episode() is None


class BadFrames(Frames):
    bad = ""

    def frame(self, after):
        jpg, t = super().frame(after)
        return {"empty": (b"", t), "old": (jpg, after - 1.0), "nan": (jpg, float("nan")), "none": None}[self.bad]


@pytest.mark.parametrize("bad", ["empty", "old", "nan", "none"])
def test_bad_photo_hands_off_without_asking(tmp_path, bad):
    r, ex, judge, recs, path, _ = make(tmp_path, frames=type("B", (BadFrames,), {"bad": bad}))
    grant(path)
    assert r.run() == HANDED and not judge.asks and ran(ex) == [ORDER[0]]
    assert bounds(recs)[-1]["reason"] == "사진 없음" and bounds(recs)[-1]["errors"]


def test_confirm_node_needs_its_own_human_line_in_auto(tmp_path):
    ep = replace(EP, nodes=(S.Node(id="check", type="confirm", text="컵을 놓았나"),) + EP.nodes)
    r, ex, _, recs, path, clock = make(tmp_path, ep=ep)
    grant(path)
    assert r.run() == HANDED and not ex.calls and recs[-1]["event"] == "no_grant"   # 에피소드 승인으로는 못 지난다
    clock.t += 10
    grant(path, since=clock())
    grant(path, scope="node", node="check", action="advance", since=clock())
    assert r.run() == SUCCESS and bounds(recs)[0]["node"] == "check" and ORDER[0] in ran(ex)


def test_poll_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, poll_s=0.0)


def test_runner_package_cannot_touch_the_grant_file():
    src = "".join(p.read_text() for p in Path(R.__file__).parent.glob("*.py"))
    assert not any(w in src for w in ("open(", "write", '"a"', '"w"', "interface", "approval"))


def test_sensor_fail_lowers_runner_ok_and_hands_off(tmp_path):
    """★11.17 센서 확실한 실패 = 러너 신호 실패 — 사진이 '됐다' 해도 어긋남으로 handoff. 애매 · 성공은 그대로."""
    from vlm_supervisor.signals import FAIL, UNSURE, Verdict
    seen = []

    def sensors(node, t0, t1):
        seen.append((node.id, t1 >= t0))
        return [Verdict(FAIL if node.id == "place_blue" else UNSURE, ("센서 테스트",), {"side": "right"})]
    at = ORDER.index("place_blue")
    r, ex, judge, recs, path, _ = make(tmp_path, plan=[{}] * at + [{"pick": "yes"}], sensors=sensors)
    grant(path)
    assert r.run() == HANDED
    last = bounds(recs)[-1]
    assert last["node"] == "place_blue" and last["decision"] == "handoff" and "어긋남" in last["reason"]
    assert last["signal"]["ok"] is False and last["signal"]["reason"].startswith("센서:")
    assert all(x["signal"]["ok"] for x in bounds(recs)[:-1]) and all(ok for _, ok in seen)


def test_sensor_error_is_logged_not_fatal(tmp_path):
    def sensors(node, t0, t1):
        raise RuntimeError("buffer down")
    r, ex, judge, recs, path, _ = make(tmp_path, sensors=sensors)
    grant(path)
    assert r.run() == SUCCESS
    assert any("센서 판정 오류" in e for x in bounds(recs) for e in x["errors"])

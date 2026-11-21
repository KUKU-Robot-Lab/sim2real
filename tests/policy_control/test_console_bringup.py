"""tilt_comp_ab --bringup(console_bringup) — 단계 순서 · 건너뛰기 · 승인 · 수동 확인 · 실패에서 멈춤 · payload 복원.
콘솔은 HTTP 대역(FakeConsole)이다 — 실제 콘솔 · ROS 를 띄우지 않는다."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools"
sys.path.insert(0, str(TOOLS))


def _load(name):
    s = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


CB = _load("console_bringup")
ab = _load("tilt_comp_ab")
ORDER = ["preflight", "drivers", "hand_right", "hand_check_right", "pd_load_right", "pd_arm_right",
         "hand_left", "hand_check_left", "pd_load_left", "pd_arm_left"]
NOTHING = {"topics": set(), "pd": {"left": None, "right": None}}


class FakeConsole:
    """콘솔 API 대역 — 단계 실행은 곧바로 끝나고(수동 단계는 ack 를 기다린다), fail 에 든 단계는 FAILED."""

    def __init__(self, profile="rh56f1_real_cam5090", domain=50, done=(), fail=(), session=True, units=(), refuse=()):
        self.stages = {s["id"]: s for s in CB.plan(CB.profile_mission(profile))}
        self.profile = {"id": profile, "domain": domain, "mission": str(CB.profile_mission(profile))}
        self.done, self.fail, self.calls, self.runners = set(done), set(fail), [], {}
        self.approved, self.open = set(), session
        self.units, self.refuse = {k: {"alive": True} for k in units}, set(refuse)   # refuse = 끄기를 거부할 유닛 키

    def __call__(self, method, path, body, headers):
        self.calls.append((method, path, dict(body or {})))
        if method == "GET" and path == "/api/state":
            return 200, {"session": None if not self.open else {
                "profile": self.profile, "run_id": "r1", "units": self.units,
                "mission": {"rows": [{"id": k, "done": k in self.done, "approved": k in self.approved} for k in self.stages]},
                "runners": self.runners}}
        if method == "GET":
            return 200, {"text": "log tail"}
        stage = (body or {}).get("stage")
        if path == "/api/lease":
            return 200, {"token": "t"}
        if path == "/api/run/open":
            self.open = True
        elif path == "/api/approve":
            self.approved.add(stage)
        elif path == "/api/stage/skip":
            self.done.add(stage)
        elif path == "/api/stage/rewind":
            self.done.discard(stage)
        elif path == "/api/unit":
            if body["key"] in self.refuse:
                return 409, {"error": f"{body['key']} 를 지금 끌 수 없다", "reasons": ["pd 가 HOLD 다 — 먼저 PD 해제를 할 것"]}
            self.units[body["key"]]["alive"] = False
        elif path == "/api/stage/run":
            if self.stages[stage]["touches_real"] and stage not in self.approved:
                return 400, {"error": "승인 없음"}
            steps = [{"index": i, "kind": c["kind"], "note": c["note"], "key": f"{stage}#{i}",
                      "status": "waiting" if c["kind"] == "manual" else "done"} for i, c in enumerate(self.stages[stage]["commands"])]
            failed = stage in self.fail
            if failed:
                steps[-1]["status"] = "failed"
            waiting = any(s["status"] == "waiting" for s in steps)
            self.runners[stage] = {"stage": stage, "steps": steps, "active": waiting and not failed,
                                   "outcome": None if waiting and not failed else ("FAILED" if failed else "DONE"), "note": ""}
            if not waiting and not failed:
                self.done.add(stage)
        elif path == "/api/stage/ack":
            r = self.runners[stage]
            r["steps"][body["index"]]["status"] = "done" if body["ok"] else "failed"
            if not any(s["status"] == "waiting" for s in r["steps"]):
                ok = all(s["status"] == "done" for s in r["steps"])
                r.update(active=False, outcome="DONE" if ok else "ABORTED")
                if ok:
                    self.done.add(stage)
        return 200, {"ok": True}

    def posted(self, path):
        return [b.get("stage") for m, p, b in self.calls if p == path and m == "POST"]


def _bringup(fc, graph=NOTHING, answers=(), profile="rh56f1_real_cam5090", domain=50, tmp=Path("/nonexistent")):
    asked = list(answers)
    api = CB.Api(http=fc)
    return CB.bringup(api, profile, lambda: graph, domain, log=tmp / "c.log", ask=lambda q: asked.pop(0), out=lambda *a: None), asked


def test_plan_follows_needs_in_order_and_real_has_manual_power_check():
    assert [s["id"] for s in CB.plan(CB.profile_mission("rh56f1_fake"))] == ORDER
    real = {s["id"]: s for s in CB.plan(CB.profile_mission("rh56f1_real_cam5090"))}
    assert list(real) == ORDER
    assert real["drivers"]["commands"][0]["kind"] == "manual" and "모터 전원" in real["drivers"]["commands"][0]["note"]
    assert any(c["kind"] == "manual" for c in real["hand_left"]["commands"])
    assert "probe_left" not in real                                  # needs 가 아니면 넣지 않는다


def test_decide_and_already_up():
    g = {"topics": {"/joint_states", "/hand_left/joint_states"},
         "pd": {"left": {"execute": True}, "right": {"execute": False}}}
    assert CB.already_up("drivers", g) and CB.already_up("hand_left", g) and not CB.already_up("hand_right", g)
    assert CB.already_up("pd_arm_left", g) and CB.already_up("pd_load_right", g) and not CB.already_up("pd_arm_right", g)
    assert CB.already_up("hand_check_left", g) is None and CB.already_up("preflight", g) is None
    skippable, fixed = {"skippable": True}, {"skippable": False}
    assert CB.decide(skippable, True, True)[0] == "skip"
    assert CB.decide(skippable, True, False)[0] == "fail"                # 콘솔은 끝났다는데 없다
    assert CB.decide(skippable, False, True)[0] == "console_skip"
    assert CB.decide(fixed, False, True)[0] == "fail"                   # 두 번 띄우지 않는다
    assert CB.decide(fixed, False, None)[0] == "run"


def test_bringup_runs_all_in_order_with_one_approval_and_manual_acks():
    fc = FakeConsole()
    rec, left = _bringup(fc, answers=["bringup", "", "", ""])        # 승인 1 · 모터 전원 · 손 2
    assert rec["ok"], rec.get("error")
    assert left == []
    assert fc.posted("/api/stage/run") == ORDER
    real = [s for s in ORDER if fc.stages[s]["touches_real"]]
    assert fc.posted("/api/approve") == real
    acks = [(b["stage"], b["ok"]) for m, p, b in fc.calls if p == "/api/stage/ack"]
    assert acks == [("drivers", True), ("hand_right", True), ("hand_left", True)]
    assert fc.calls[-1][:2] == ("DELETE", "/api/lease")             # lease 를 놓고 끝난다


def test_bringup_skips_what_is_already_up():
    fc = FakeConsole(done={"preflight", "drivers", "hand_right", "hand_check_right", "pd_load_right", "pd_arm_right"})
    g = {"topics": {"/joint_states", "/hand_right/joint_states", "/hand_left/joint_states"},
         "pd": {"right": {"execute": True}, "left": None}}
    rec, _ = _bringup(fc, graph=g, answers=["bringup"])
    assert rec["ok"], rec.get("error")
    assert fc.posted("/api/stage/skip") == ["hand_left"]                # 콘솔 밖에서 떠 있는 손
    assert fc.posted("/api/stage/run") == ["hand_check_left", "pd_load_left", "pd_arm_left"]
    assert {s["id"]: s["action"] for s in rec["stages"]}["drivers"] == "skip"


def test_bringup_stops_at_first_failure_and_on_refusals():
    fc = FakeConsole(fail={"drivers"})
    rec, _ = _bringup(fc, answers=["bringup", ""])
    assert not rec["ok"] and rec["error"].startswith("drivers ") and "실패 단계: ★모터 응답 확인" in rec["error"]
    assert fc.posted("/api/stage/run") == ["preflight", "drivers"]

    fc = FakeConsole()
    rec, _ = _bringup(fc, answers=["no"])
    assert not rec["ok"] and "승인 안 함" in rec["error"] and fc.posted("/api/stage/run") == []

    fc = FakeConsole()                                               # 모터 전원이 정상이 아니라고 답함
    rec, _ = _bringup(fc, answers=["bringup", "n"])
    assert not rec["ok"] and fc.posted("/api/stage/run") == ["preflight", "drivers"]

    rec, _ = _bringup(FakeConsole(profile="rh56f1_fake", domain=97), domain=97)
    assert not rec["ok"] and "다른 프로파일" in rec["error"]
    rec, _ = _bringup(FakeConsole(), domain=88)
    assert not rec["ok"] and "도메인" in rec["error"]


def test_bringup_rewinds_stage_done_in_console_but_missing_in_ros():
    fc = FakeConsole(done={"preflight", "drivers"})                  # 실험 뒤 drivers 를 내림 — /joint_states 없음
    rec, _ = _bringup(fc, answers=["bringup", "", "", ""])
    assert rec["ok"], rec.get("error")
    assert fc.posted("/api/stage/rewind") == ["drivers"] and rec["rewound"] == ["drivers"]
    assert fc.posted("/api/stage/run") == [s for s in ORDER if s != "preflight"]   # preflight 는 판단 거리 없음 → 그대로
    assert "drivers" in fc.posted("/api/approve")                     # 다시 띄우는 실기 단계는 승인을 다시 받는다
    fc = FakeConsole(done={"preflight", "drivers"})
    rec, _ = _bringup(fc, answers=["no"])
    assert not rec["ok"] and "승인 안 함" in rec["error"] and fc.posted("/api/stage/run") == []


def test_bringup_opens_run_when_console_has_none():
    fc = FakeConsole(profile="rh56f1_fake", domain=97, session=False)
    rec, _ = _bringup(fc, profile="rh56f1_fake", domain=97)
    assert rec["ok"], rec.get("error")
    assert ("POST", "/api/run/open", {"profile": "rh56f1_fake"}) in fc.calls
    assert fc.posted("/api/approve") == []                           # fake 미션은 실기 단계가 없다


def test_dry_run_prints_bringup_and_payload_plan_and_moves_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    monkeypatch.setattr(CB.Api, "up", lambda self: False)
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *a, **k: pytest.fail("계획만 모드에서 명령을 실행했다"))
    assert ab.main(["--cycles", "2", "--out", str(tmp_path / "run")]) == 0
    text = capsys.readouterr().out
    assert "사이클 2 (A B)" in text and "--tilt-comp" in text and "payload → 0 kg" in text
    assert "rh56f1_real_cam5090" in text and "★손으로 확인" in text and "모터 전원" in text
    assert all(f" {s} " in text for s in ORDER)
    assert "check_can_rx.py" in text                                 # 수동 확인 뒤 자동 확인
    assert not (tmp_path / "run").exists()
    ab.main(["--cycles", "2", "--no-bringup", "--out", str(tmp_path / "run")])
    assert "[bringup]" not in capsys.readouterr().out


class FakeWatch:
    def __init__(self, refuse=()):
        self.p, self.sent, self.refuse = [0.117, 0.0174, 0.0167, 0.1945], [], set(refuse)
        self.status = {"left": {"phase": "IDLE"}}

    def payload(self, side="left"):
        return list(self.p)

    def set_payload(self, kg, side="left", timeout=0):
        self.sent.append(kg)
        if kg in self.refuse:
            return "pd: TRACKING"
        self.p[0] = kg
        return None


@pytest.mark.parametrize("boom", [RuntimeError("예외"), KeyboardInterrupt()])
def test_payload_applied_then_restored_even_on_exception(tmp_path, monkeypatch, boom):
    pd_yaml = tmp_path / "pd.yaml"
    pd_yaml.write_text("tilt_comp: {}\n")
    monkeypatch.setattr(ab, "running_pd_config", lambda side="left": pd_yaml)
    monkeypatch.setattr(ab, "md5", lambda p: "x")
    seen = {}

    def cycles(a, conds, traj, out, w, base):
        seen["during"] = w.payload()[0]
        raise boom
    monkeypatch.setattr(ab, "_cycles", cycles)
    w = FakeWatch()
    a = SimpleNamespace(payload_kg=0.0, rate_scale=1.0, cycles=2, keep_bringup=True)
    with pytest.raises(type(boom)):
        ab._experiment(a, ["A", "B"], pd_yaml, tmp_path, w, None)
    assert seen["during"] == 0.0 and w.sent == [0.0, 0.117] and w.p[0] == 0.117
    import json
    pay = json.loads((tmp_path / "meta.json").read_text())["payload"]
    assert pay["before"][0] == 0.117 and pay["applied"][0] == 0.0 and pay["restored"] and pay["after"][0] == 0.117


def test_payload_refused_moves_nothing_and_failed_restore_is_reported(tmp_path, monkeypatch):
    pd_yaml = tmp_path / "pd.yaml"
    pd_yaml.write_text("tilt_comp: {}\n")
    monkeypatch.setattr(ab, "running_pd_config", lambda side="left": pd_yaml)
    monkeypatch.setattr(ab, "md5", lambda p: "x")
    monkeypatch.setattr(ab, "_cycles", lambda *a: pytest.fail("거부됐는데 사이클을 돌렸다"))
    a = SimpleNamespace(payload_kg=0.0, rate_scale=1.0, cycles=2, keep_bringup=True)
    assert ab._experiment(a, ["A"], pd_yaml, tmp_path, FakeWatch(refuse={0.0}), None) == 3

    monkeypatch.setattr(ab, "_cycles", lambda *a: None)
    monkeypatch.setattr(ab, "write_meta", lambda out, **kw: seen.update(kw))
    seen: dict = {}
    assert ab._experiment(a, ["A"], pd_yaml, tmp_path, FakeWatch(refuse={0.117}), None) == 2   # 복원 실패 = 실패 종료
    assert seen["payload"]["restored"] is False and "TRACKING" in seen["payload"]["restore_error"]

    held = FakeWatch()                                   # 실패 중단 = 붙든 채(TRACKING) — 기다리지 않고 이유를 남긴다
    held.status["left"]["phase"] = "TRACKING"
    assert ab._experiment(a, ["A"], pd_yaml, tmp_path, held, None) == 2
    assert held.sent == [0.0] and "pd release 뒤" in seen["payload"]["restore_error"]


# ---------------------------------------------------------------- teardown(11.17 사용자: 실험이 끝나면 브링업까지 내린다)
UNITS = ("drivers#0", "hand_right#0", "pd_arm_left#1", "hand_left#0", "pd_load_right#0", "probe_x#0")
TD_STAGES = ["hand_left", "hand_right", "pd_arm_left", "pd_load_left", "pd_arm_right", "pd_load_right"]


def _teardown(fc, keep=None, stages=TD_STAGES):
    lines = []
    rec = CB.teardown(CB.Api(http=fc), stages, out=lines.append, keep_drivers=keep, poll=0)
    return rec, [b["key"] for m, p, b in fc.calls if p == "/api/unit"], lines


def test_teardown_finished_turns_off_hands_pd_then_drivers_and_rewinds_all():
    fc = FakeConsole(done=ORDER, units=UNITS)
    rec, off, _ = _teardown(fc)
    assert off == ["hand_left#0", "hand_right#0", "pd_arm_left#1", "pd_load_right#0", "drivers#0"]   # 손 → pd → drivers
    assert fc.units["probe_x#0"]["alive"] and not rec["failed"] and rec["drivers_kept"] is None
    assert rec["rewound"] == [s for s in CB.REWIND_ORDER if s in ORDER] == fc.posted("/api/stage/rewind")
    assert fc.done == {"preflight", "hand_check_right", "hand_check_left"}   # 실제 콘솔은 의존 단계도 같이 되감는다


def test_teardown_keeps_drivers_when_told_or_when_a_unit_refuses():
    fc = FakeConsole(done=ORDER, units=UNITS)
    rec, off, lines = _teardown(fc, keep="중단")
    assert "drivers#0" not in off and fc.units["drivers#0"]["alive"] and "drivers" not in fc.posted("/api/stage/rewind")
    assert rec["drivers_kept"] == "중단" and any("drivers)은 남김 — 중단" in x for x in lines)
    fc = FakeConsole(done=ORDER, units=UNITS, refuse={"hand_left#0"})
    rec, off, _ = _teardown(fc)
    assert "drivers#0" not in off and "hand_left" in rec["drivers_kept"]               # 손 · pd 를 다 못 내리면 drivers 남김
    assert "hand_left" not in rec["rewound"] and "drivers" not in rec["rewound"] and "pd_arm_left" in rec["rewound"]
    assert any(f.startswith("hand_left#0:") for f in rec["failed"])


def test_teardown_console_down_is_recorded_not_raised():
    def down(*a):
        raise OSError("연결 거부")
    down.calls = []
    rec, _, _ = _teardown(down)
    assert rec["off"] == [] and "연결 거부" in rec["drivers_kept"] and rec["failed"]

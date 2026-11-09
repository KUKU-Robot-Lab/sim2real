"""stationary_batch — 한 프로세스 배치의 오프라인 부분(정지 판정 · 감시 기준선 · 프레임 검사 · rebase · capture).

로봇 · ROS 를 띄우지 않는다. 11.09 F1 실험 폴더가 있는 호스트(5090)에서는 실제 기록으로 rebase · capture 를 대조한다.
"""
import collections
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/policy_control/tools"))
import stationary_batch as B  # noqa: E402

F1 = ROOT / "analysis/friction/261109_F1off_right"
REAL = pytest.mark.skipif(not (F1 / "setup/plan/plan.json").is_file(), reason="11.09 F1 실기 기록 없음")


def _js(t_ns, q, side="right"):
    return {"topic": "/joint_states", "recv_mono_ns": int(t_ns),
            "message": {"name": [f"openarm_{side}_joint{i}" for i in range(1, 8)], "position": list(q)}}


def test_settled_needs_a_full_quiet_window_after_reset():
    now = 10e9
    assert B.SETTLE_WINDOW_S == 2.0                                   # 원래 capture --seconds 2
    quiet = [_js(now - 2.2e9 + k * 1e7, [0.1] * 7) for k in range(221)]
    assert B.settled(quiet, "right", now)
    assert not B.settled(quiet[-150:], "right", now)                 # 버퍼가 2 s 를 못 덮는다(막 끝난 trial)
    sag = [_js(r["recv_mono_ns"], [0.1 + 0.002 * k / 220] + [0.1] * 6) for k, r in enumerate(quiet)]
    assert not B.settled(sag, "right", now)                           # 2 s 에 1.8 mrad 처짐
    one_tick = [_js(r["recv_mono_ns"], [0.1 + 0.000381 * (k % 2)] + [0.1] * 6) for k, r in enumerate(quiet)]
    assert B.settled(one_tick, "right", now)                          # 엔코더 한 칸 떨림은 정지
    assert not B.settled(quiet, "left", now)                          # 다른 팔 이름이면 판정 없음


def test_guard_baseline_follows_contact_guard_rules():
    rng = np.random.default_rng(0)
    ok = (100 + rng.normal(0, 1, (600, 6))).tolist()
    base, why = B.guard_baseline(ok)
    assert why is None and np.allclose(base, np.mean(ok[-500:], axis=0))
    assert B.guard_baseline(ok[:50]) == (None, "invalid_baseline")
    assert B.guard_baseline((100 + rng.normal(0, 9, (600, 6))).tolist()) == (None, "unstable_or_loaded_baseline")
    assert B.guard_baseline([[500.0] * 6] * 200) == (None, "unstable_or_loaded_baseline")
    assert B.guard_baseline([[1.0] * 5] * 200) == (None, "invalid_baseline")


def test_trial_frames_keeps_timing_and_limits():
    ramp = np.linspace(0, 0.0005, 101)[:, None] * np.ones(7)
    trial = {"approach_frames": ramp.tolist(), "frames": ramp[-1:].repeat(3, 0).tolist(),
             "return_frames": ramp[::-1].tolist(), "hold_s": 15.0}
    frames, end = B.trial_frames(trial, 0.01)
    assert end == 102 and len(frames) == 103 + 1500 + 100
    with pytest.raises(ValueError, match="dimensions/timing"):
        B.trial_frames({**trial, "hold_s": 10.0}, 0.01)
    fast = {**trial, "frames": (ramp[-1:] + 0.01).tolist()}
    with pytest.raises(ValueError, match="velocity"):
        B.trial_frames(fast, 0.01)


def test_dry_run_sends_nothing_and_refuses_existing_outputs(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    args = ["--side", "right", "--plan", str(plan), "--trial", "t1", str(tmp_path / "rb/t1_a01"), str(tmp_path / "runs/t1")]
    assert B.main(args) == 0 and "0 commands" in capsys.readouterr().out
    (tmp_path / "runs/t1").mkdir(parents=True)
    with pytest.raises(SystemExit):
        B.main(args)
    with pytest.raises(SystemExit):
        B.main(args[:4] + ["--trial", "a", "x", "y", "--trial", "a", "z", "w"])


@REAL
def test_rebase_in_process_is_byte_identical_to_the_script(tmp_path):
    """world 를 한 번 만들고 되돌려 써도 trial 마다 rebase_trial.py 를 새로 띄운 것과 plan.json 이 같다."""
    plan = F1 / "setup/plan/plan.json"
    rb = B.Rebaser()
    for i, t in enumerate(["F1_right_fit_01_j1_+1_r2", "F1_right_fit_01_j2_-1_r1"]):
        snap = F1 / f"setup/rebase/{t}_a01/capture/snapshot.json"
        mine = rb(plan, snap, t, tmp_path / f"in{i}")
        subprocess.run([sys.executable, str(B.REBASE), "--plan", str(plan), "--snapshot", str(snap),
                        "--trial", t, "--out", str(tmp_path / f"sub{i}")], check=True, capture_output=True)
        assert mine.read_text() == (tmp_path / f"sub{i}/plan.json").read_text()
        B.trial_frames(json.loads(mine.read_text())["trials"][0], 0.01)   # 실제 계획은 프레임 검사를 지난다


@REAL
def test_capture_from_rolling_buffer_matches_the_recorder_capture(tmp_path, monkeypatch):
    """같은 기록을 굴리는 버퍼로 넣으면 snapshot 이 capture 명령의 것과 같고 raw/ 형식도 같다."""
    src = F1 / "setup/rebase/F1_right_fit_01_j1_+1_r2_a01/capture"
    rows = [json.loads(line) for line in (src / "raw/events.jsonl").read_text().splitlines()]
    shift = time.monotonic_ns() - max(r["recv_mono_ns"] for r in rows)
    for r in rows:
        r["recv_mono_ns"] += shift
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    monkeypatch.setattr(B, "SETTLE_WINDOW_S", 1.0)                    # 이 옛 capture 기록은 1.02 s 다
    b = B.Batch.__new__(B.Batch)
    b.side, b.buffer, b.force = "right", collections.deque(rows), collections.deque()
    b.latched = {r["topic"]: r for r in rows if B.recorder.is_latched(r["topic"])}
    snap = b.capture(tmp_path / "capture")
    got, want = json.loads(snap.read_text()), json.loads((src / "snapshot.json").read_text())
    assert got["q"] == want["q"] and got["hand"] == want["hand"] and got["scene"] == want["scene"]
    for name in ("manifest.json", "summary.json", "report.md", "tracking.png", "events.jsonl",
                 "raw/events.jsonl", "raw/events.npz", "raw/metadata.json"):
        assert (tmp_path / "capture" / name).exists(), name
    meta = json.loads((tmp_path / "capture/raw/metadata.json").read_text())
    assert meta["commands_sent"] == 0 and meta["n_events"] > 0
    shutil.rmtree(tmp_path / "capture")


def _guard_batch(force_age_s, still=True):
    """버퍼 = 지금까지 3.5 s 정지(또는 움직임), 힘 = force_age_s 초 전에 끝난 3.5 s 치."""
    now = time.monotonic_ns()
    b = B.Batch.__new__(B.Batch)
    b.side, b.guard = "right", None
    b.buffer = collections.deque(_js(now - 3.5e9 + k * 1e7, [0.1 + (0 if still else 0.003 * k / 350)] + [0.1] * 6)
                                 for k in range(351))
    end = now / 1e9 - force_age_s
    b.force = collections.deque((end - 3.5 + k * .01, [100.0] * 6) for k in range(351))
    b.spin = lambda seconds: time.sleep(seconds)              # 새 메시지 없음
    b.node = type("N", (), {"get_logger": lambda self: type("L", (), {"info": lambda self, m: None})()})()
    return b


def test_guard_baseline_uses_only_recent_force_while_still(monkeypatch):
    monkeypatch.setattr(B, "GUARD_TIMEOUT_S", .3)
    b = _guard_batch(0.0)
    b._guard_on()
    assert b.guard["latched"] is None and b.guard["baseline"].tolist() == [100.0] * 6
    stale = _guard_batch(10.0)                                    # 힘이 10 s 전에 끊겼다 — 오래된 힘으로 잡지 않는다
    with pytest.raises(ValueError, match="guard baseline"):
        stale._guard_on()
    assert stale.guard is None
    moving = _guard_batch(0.0, still=False)                       # 3 s 동안 팔이 움직였다
    with pytest.raises(ValueError, match="guard baseline"):
        moving._guard_on()


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


def test_second_sigterm_still_publishes_abort_and_writes_manifest(tmp_path, monkeypatch):
    """도는 trial 에 SIGTERM 두 번 — 첫 번은 abort(=pd hold), 두 번째는 무시되고 abort · manifest 가 끝난다."""
    import os
    import signal
    import types

    side, diag, status = "right", "/policy_control/pd_right/diagnostics", "/policy_control/status/pd_right"
    d = {"contact": False, "stale": False, "protected": False, "temperature": [30.0] * 7,
         "torque_feedback": [0.0] * 7, "measured": [0.0] * 7, "velocity": [0.0] * 7,
         "gains": {"runtime": {"verified": True}, "real_kp": [1.0] * 7, "real_kd": [1.0] * 7},
         "compensation_config": {"c": 1}, "hand_names": [], "measured_hand": []}
    js = {"name": [f"openarm_{s}_joint{i}" for s in ("right", "left") for i in range(1, 8)], "position": [0.0] * 14}
    calls = {"n": 0, "signals": 0}

    def spin_once(node, timeout_sec=0.0):
        calls["n"] += 1
        if b.ep_pub.sent and calls["signals"] < 2 and calls["n"] % 20 == 0:
            calls["signals"] += 1                                   # 동작 중 한 번, abort 뒤 spin 중 한 번
            os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(min(timeout_sec, .001))

    b = B.Batch.__new__(B.Batch)
    b.side, b.other, b.diag_topic, b.status_topic, b.hand = side, "left", diag, status, "/hand_right"
    b.rclpy = types.SimpleNamespace(spin_once=spin_once)
    b.node = types.SimpleNamespace(
        get_name=lambda: "n", get_publishers_info_by_topic=lambda t: [types.SimpleNamespace(node_name="n")],
        get_logger=lambda: types.SimpleNamespace(info=lambda m: None, warning=lambda m: None))
    b.String = b.Bool = lambda data: types.SimpleNamespace(data=data)
    b.ep_pub, b.target_pub, b.contact_pub, b.guard_pub = _Pub(), _Pub(), _Pub(), _Pub()
    b.latest = {diag: {"data": json.dumps(d)}, status: {"data": json.dumps(
        {"execute": True, "ok": True, "phase": "TRACKING", "estop": False})}, "/joint_states": js}
    b.received = type("Fresh", (dict,), {"get": lambda self, k, default=0: time.monotonic() if k in b.latest else default})()  # 스트림은 계속 온다
    b.latched, b.sink, b.guard = {}, None, None
    b.buffer, b.force = collections.deque(), collections.deque()
    b._guard_on = lambda: setattr(b, "guard", {"baseline": None, "latched": None})
    monkeypatch.setattr(sys.modules["policy_control.sources"] if "policy_control.sources" in sys.modules
                        else __import__("policy_control.sources", fromlist=["x"]), "load_profile", lambda p: {})
    import policy_control.fk_numpy as fk
    from policy_control import codec
    monkeypatch.setattr(codec, "encode_joint_target", lambda *a, **k: a)      # ROS 메시지 첫 import 가 20 ms 를 넘긴다
    monkeypatch.setattr(fk, "UrdfTree", lambda p: types.SimpleNamespace(expand_mimic=lambda h: h))
    z = [[0.0] * 7] * 2
    plan = {"dt": 0.01, "kp": [1.0] * 7, "kd": [1.0] * 7, "compensation": {"c": 1}, "fixed_hand": {},
            "scene_joints": [], "scene": {f"l_aj_{i}": 0.0 for i in range(1, 8)},
            "inputs": {"profile": {"path": "x"}, "urdf": {"path": "x"}}}
    trial = {"approach_frames": z, "frames": z, "return_frames": z, "hold_s": 15.0, "approach_start_q": [0.0] * 7}
    manifest = {"metadata": {}}
    monkeypatch.setitem(B.SIG, "count", 0)
    old = signal.signal(signal.SIGTERM, B._interrupt)
    try:
        rows, path_end, failure, count = b._execute(plan, trial, manifest, tmp_path)
    finally:
        signal.signal(signal.SIGTERM, old)
    assert calls["signals"] == 2 and B.SIG["count"] == 2, (failure, b.ep_pub.sent)
    assert failure == "operator_interrupt" and 0 < count < 1500
    assert [json.loads(m.data)["event"] for m in b.ep_pub.sent] == ["reset", "abort"]
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["metadata"]["failure"] == "operator_interrupt" and saved["commands_sent"] == count
    assert b.guard is None and b.sink is None


def test_analysis_finish_after_direct_run_does_not_join_unstarted_thread(monkeypatch):
    """11.09 F4 r2: 실패 경로에서 run() 직접 호출 뒤 finish() 가 RuntimeError 로 batch 를 죽였다."""
    import stationary_batch as B
    monkeypatch.setattr(B, "analyze_run", lambda *a: None)
    job = B.Analysis("run_dir")
    job.run()
    job.finish()

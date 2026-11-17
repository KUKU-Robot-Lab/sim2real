"""place_fixed: pd 손 명령이 없으면 계획 전에 rc 4 · 있으면 그 값이 --hand-grip 으로, descend 는 --until · 손 출발은 applied."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"))
import place_fixed as PFW  # noqa: E402
import plan_tilt  # noqa: E402

Q = {"right": [0.0] * 7, "left": [0.0] * 7}
GRIP = [1.2, 0.24, 1.08, 1.08, 0.85, 0.85]


def _setup(monkeypatch, tmp_path, hand, plan_rc=0):
    calls = []

    def fake_run(argv):
        calls.append(argv)
        if argv[1].endswith("plan_place_fixed.py"):
            if plan_rc:
                return plan_rc
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "meta.json").write_text(json.dumps({"replay_max_rate_scale_reduced": 0.345}))
        return 0

    monkeypatch.setattr(PFW.run_paths, "day_dir", lambda kind: tmp_path / kind)
    monkeypatch.setattr(PFW, "write_meta", lambda out, **kw: None)
    monkeypatch.setattr(PFW, "run", fake_run)
    monkeypatch.setattr(PFW, "pd_hand", lambda side, names: hand)
    monkeypatch.setattr(plan_tilt, "current_q", lambda q_json: Q)
    return calls


def test_no_pd_hand_command_refuses_before_planning(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, None)
    assert PFW.main(["--cup", "blue", "--until", "descend"]) == 4 and calls == []


def test_pd_hand_command_drives_plan_and_replay(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, GRIP)
    assert PFW.main(["--cup", "shaker", "--until", "descend"]) == 0
    plan, replay = calls
    assert [float(v) for v in plan[plan.index("--hand-grip") + 1].split(",")] == GRIP
    out = next((tmp_path / "place_fixed").iterdir())
    assert json.loads((out / "hand.json").read_text())["side"] == "left"
    assert replay[replay.index("--until") + 1] == "descend" and replay[replay.index("--rate-scale") + 1] == "0.34"
    assert replay[replay.index("--hand-start") + 1] == "applied" and "--execute" not in replay


def test_pick_hand_exact_six_or_none():
    from policy_control import place_teach as PT
    names = PT.hand_names("right")
    msg_n = ["openarm_right_joint1", "l_hj_1", *reversed(names)]
    msg_p = [9.0, 8.0, *reversed(GRIP)]
    assert PFW.pick_hand(msg_n, msg_p, names) == GRIP           # 팔 · 다른 손 이름은 빼고 순서대로
    assert PFW.pick_hand(msg_n[:-1], msg_p[:-1], names) is None   # 손 6 중 하나라도 없으면 None


def test_plan_fail_rc_passthrough_no_replay(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, GRIP, plan_rc=3)
    assert PFW.main(["--cup", "blue", "--until", "all", "--execute"]) == 3
    assert len(calls) == 1 and calls[0][1].endswith("plan_place_fixed.py")


def test_execute_flag_only_with_execute(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, GRIP)
    assert PFW.main(["--cup", "blue", "--until", "all", "--execute"]) == 0
    replay = calls[1]
    assert "--execute" in replay and "--until" not in replay

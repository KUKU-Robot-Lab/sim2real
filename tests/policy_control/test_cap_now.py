"""11.18 cap_now — cap 학생 시작 자세로(tilt_now 를 자산 config/cap/cap_start · 오른손 빈손으로). 실기 · ROS 없음(run 을 가짜로)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "deploy" / "policy_control" / "tools"
CONFIG = REPO / "deploy" / "policy_control" / "config"
RCV_L = [0.5277, -0.2809, -0.0247, 1.5046, -0.1040, -0.2581, -0.7205]    # rcv_bank_rcg04_base_ik.npz l_aj_1..7(모든 행 같음)


def test_start_pose_is_the_student_start_and_the_receiver_pose():
    from policy_control import rh_cap_student_node as N

    pose = yaml.safe_load((CONFIG / "homes" / "rh56f1_cap_start.yaml").read_text(encoding="utf-8"))
    assert pose["right"] == list(N.LIVE_ARM_START_Q)
    assert pose["left"] == RCV_L


def test_asset_differs_from_base_arc6_only_in_start_pose():
    cap = yaml.safe_load((CONFIG / "cap" / "cap_start" / "policy.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((CONFIG / "tilt" / "base_arc6" / "policy.yaml").read_text(encoding="utf-8"))
    assert cap["hdgp"] == base["hdgp"] and cap["ref"] == base["ref"] and cap["table"] == base["table"]
    assert (REPO / cap["start_pose"]).resolve() == (CONFIG / "homes" / "rh56f1_cap_start.yaml").resolve()
    assert (CONFIG / "cap" / "cap_start" / "ref" / "single734_t120.npz").is_file()


@pytest.fixture()
def cap_now(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "rclpy", None)           # 패치를 빼먹은 시험이 실 ROS 로 가지 않게 — import rclpy 가 ImportError
    import cap_now as K
    import tilt_now as T

    monkeypatch.setattr(T, "current_q", lambda: {"right": [0.0] * 7, "left": [0.0] * 7})
    calls, plan_rc = [], {}

    def run(argv):
        calls.append(argv)
        if argv[1].endswith("plan_carry.py"):
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            rc = plan_rc.get(argv[argv.index("--sides") + 1], 0)
            (out / "check.json").write_text(json.dumps({"ok": rc == 0, "checks": {}, "arms_clear": {"ok": True}}))
            return rc
        return 0

    monkeypatch.setattr(T, "run", run)
    monkeypatch.setattr(K, "hand_q", lambda names: [1.57, 0.0, 0.0, 0.0, 0.0, 0.0])          # 손 구독 · 발행은 늘 가짜
    monkeypatch.setattr(K, "applied_hand", lambda names: [1.57, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(K.H, "run", lambda side, hg, c: (0, {"phase": "done"}))
    return K, calls, plan_rc, tmp_path / "cap"


def test_plan_only_plans_both_arms_with_right_empty_and_never_replays(cap_now):
    K, calls, _, root = cap_now
    assert K.main(["--out-root", str(root)]) == 0
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py", "plan_carry.py"]
    for c in calls:
        assert Path(c[c.index("--asset") + 1]) == K.ASSET
        assert c[c.index("--empty") + 1] == "right"


def test_execute_replays_each_arm_after_its_check(cap_now):
    K, calls, _, root = cap_now
    assert K.main(["--out-root", str(root), "--execute"]) == 0
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py", "replay_to_pd.py", "plan_carry.py", "replay_to_pd.py"]
    assert all("--execute" in c for c in calls if c[1].endswith("replay_to_pd.py"))


def test_plan_failure_rc_is_returned_and_nothing_moves(cap_now):
    K, calls, plan_rc, root = cap_now
    plan_rc["right"] = 2
    assert K.main(["--out-root", str(root), "--execute"]) == 2
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py"]


# ---- 11.18 마지막 단계: 오른손 → rh_cap.CAPTOP_OPEN(hand_grasp.run 은 가짜)
@pytest.fixture()
def hand(cap_now, monkeypatch):
    K, calls, plan_rc, root = cap_now
    pub = []                                                  # hand_grasp.run 이 발행했을 목표(이름, 값)

    def fake_run(side, hg, c):
        s = c.side()
        names = list(s.arm_joints) + list(s.hand_joints)       # hand_grasp.run send() 와 같은 이름
        hg.start([9.0] * 6)                                   # 실측은 무시하고 applied 에서 시작해야 한다
        meas = monkeypatch_meas[0]
        while not hg.finished:
            q = hg.step(hg.q_cmd if meas is None else meas)
            pub.append((names, q.copy()))
        return (0 if hg.phase == "done" else 2), {**hg.as_dict()}

    monkeypatch_meas = [None]
    monkeypatch.setattr(K.H, "run", fake_run)
    return K, calls, plan_rc, root, pub, monkeypatch_meas


def test_hand_tol_is_inside_student_node_start_tol(monkeypatch):
    from policy_control import rh_cap_student_node as N
    monkeypatch.syspath_prepend(str(TOOLS))
    import cap_now as K

    assert K.HAND_TOL_RAD < N.HAND_OPEN_TOL_RAD


def test_contract_is_right_side_and_target_inside_limits():
    c = json.loads((REPO / "config" / "hand" / "rh56f1_right_hand.json").read_text())
    s = c["sides"]["arm"]
    assert all(j.startswith("r_") for j in s["arm_joints"] + s["hand_joints"])
    from policy_control.rh_cap import CAPTOP_OPEN
    t = np.asarray(CAPTOP_OPEN, float)
    assert np.all(t >= np.asarray(s["hand_lim_lo"])) and np.all(t <= np.asarray(s["hand_lim_hi"]))


def test_plan_only_never_publishes_hand(hand):
    K, calls, _, root, pub, _ = hand
    assert K.main(["--out-root", str(root)]) == 0
    assert pub == []


def test_execute_ramps_right_hand_to_captop_open_only(hand):
    from policy_control.rh_cap import CAPTOP_OPEN
    K, calls, _, root, pub, _ = hand
    assert K.main(["--out-root", str(root), "--execute"]) == 0
    assert pub and all(not n.startswith("l_") for names, _ in pub for n in names)
    assert all(n.startswith("r_") for n in pub[0][0])
    q = np.array([v for _, v in pub])
    assert np.allclose(q[-1], CAPTOP_OPEN)
    steps = np.abs(np.diff(np.vstack([[1.57, 0, 0, 0, 0, 0], q]), axis=0)).max()
    assert steps <= K.HAND_RAD_S / 60.0 + 1e-9                 # 계약 policy_hz 60 · applied 에서 시작(첫 틱도)


def test_arm_replay_failure_skips_hand(hand):
    K, calls, plan_rc, root, pub, _ = hand
    plan_rc["left"] = 2
    assert K.main(["--out-root", str(root), "--execute"]) == 2
    assert pub == []


@pytest.mark.parametrize("extra", [["--no-hand"], ["--sides", "left"]])
def test_no_hand_or_left_only_skips_hand(hand, extra):
    K, calls, _, root, pub, _ = hand
    assert K.main(["--out-root", str(root), "--execute", *extra]) == 0
    assert pub == []
    assert all("--no-hand" not in c for c in calls)


def test_target_outside_hand_limits_is_refused(hand, monkeypatch):
    K, calls, _, root, pub, _ = hand
    monkeypatch.setattr(K, "TARGET", (0.60, 0.60, 0.0, 0.0, 0.0, 0.0))   # thumb_2 한계 0.475
    assert K.main(["--out-root", str(root), "--execute"]) == 2
    assert pub == [] and calls == []                          # 팔도 움직이기 전에(T.main 이 계획도 안 돌림)


def test_contract_load_failure_is_refused_before_arms(hand, monkeypatch, tmp_path, capsys):
    K, calls, _, root, pub, _ = hand
    monkeypatch.setattr(K, "CONTRACT", tmp_path / "missing.json")
    monkeypatch.setattr(K.tilt_now, "main", lambda argv: pytest.fail("T.main 이 불렸다"))
    assert K.main(["--out-root", str(root), "--execute"]) == 2
    assert pub == [] and calls == []
    assert "손 계약" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [["--asset", "x"], ["--empty", "left"], ["--asset=x"], ["--emp", "left"]])
def test_fixed_args_from_user_are_input_error(cap_now, extra):
    K, calls, _, root = cap_now
    assert K.main(["--out-root", str(root), *extra]) == 3
    assert calls == []


def test_missing_applied_hand_is_refused(hand, monkeypatch):
    K, calls, _, root, pub, _ = hand
    monkeypatch.setattr(K, "applied_hand", lambda names: None)
    assert K.main(["--out-root", str(root), "--execute"]) == 2
    assert pub == []


def test_stuck_thumb_fails_with_per_joint_deviation(hand, capsys):
    K, calls, _, root, pub, meas = hand
    meas[0] = np.array([1.57, 0.0, 0.0, 0.0, 0.0, 0.0])            # thumb_1 이 1.57 에서 안 내려온다
    assert K.main(["--out-root", str(root), "--execute"]) == 2
    out = capsys.readouterr().out
    assert "r_hj_thumb_1 +0.970" in out

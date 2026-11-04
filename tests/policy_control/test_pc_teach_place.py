"""place 직접 교시(11.04) — pd 를 붙든 채 손으로 끌어 홀더까지: 따라가기 · 속도 · 한계 · 손 키 · 녹화가 재생 도구에 그대로 들어가는지.

모의 팔 = pd 위치 루프를 줄인 것: 손이 없으면 실측 = 목표 − 처짐, 손이 있으면 손이 정한 자리. 시간은 틱 수로 굴린다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from policy_control import place_teach as P
from policy_control import rh_replay as R

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "deploy" / "policy_control" / "tools"
DT = 1.0 / P.POLICY_HZ
LO, HI = np.full(7, -1.5), np.full(7, 1.5)
Q0 = np.array([-0.5814, 0.2908, -0.0098, 1.0, 0.1172, 0.1432, 0.6816])
DROOP = np.array([0.0, 0.004, 0.0, 0.003, 0.0, 0.0, 0.0])
HAND0 = np.array([0.8, 0.3, 1.45, 1.45, 1.45, 1.45])


def _tool(name):
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _teacher(**kw):
    return P.Teacher("right", LO, HI, np.array(P.GRIP_RAD), np.array(P.OPEN_RAD), **kw)


def _run(teacher, ticks, hand_path=None, keys=None):
    """hand_path(k, cmd) → 손이 정한 실측(또는 None). keys = {틱: 키}."""
    teacher.start(Q0, Q0 - DROOP, HAND0, 0.0)
    cmd = teacher.command
    for k in range(ticks):
        t = k * DT
        if keys and k in keys:
            teacher.key(keys[k], t)
        imposed = hand_path(k, cmd) if hand_path else None
        meas = (cmd - DROOP) if imposed is None else imposed
        cmd, _, _ = teacher.tick(meas, t)
    return teacher


def _push_j4(start_k=90, ticks=150, total=0.3):
    """틱 start_k 부터 ticks 동안 관절 4 를 total 만큼 손으로 옮긴 뒤 놓는다(그 뒤 손 없음)."""
    def path(k, cmd):
        if start_k <= k < start_k + ticks:
            q = Q0 - DROOP
            q = q.copy()
            q[3] += total * (k - start_k + 1) / ticks
            return q
        return None
    return path


def test_without_a_push_the_target_stays_at_the_pd_setpoint():
    t = _run(_teacher(), 200)
    assert np.allclose(t.command, Q0)
    assert np.allclose(np.vstack([r[2] for r in t.rows]), Q0)


def test_a_push_is_followed_within_the_speed_limit_and_kept_after_letting_go():
    t = _run(_teacher(), 600, _push_j4())
    cmds = np.vstack([r[2] for r in t.rows])
    assert np.abs(np.diff(cmds, axis=0)).max() <= P.MAX_VEL_RAD_S * DT + 1e-12
    assert t.command[3] > Q0[3] + 0.25          # 0.3 rad 옮긴 자리 근처에 남는다
    assert np.allclose(np.delete(t.command, 3), np.delete(Q0, 3), atol=0.02)


def test_the_target_never_leaves_the_joint_limits():
    def shove(k, cmd):
        q = Q0 - DROOP
        q = q.copy()
        q[3] = 1.6 if k > 90 else q[3]
        return q if k > 90 else None
    t = _run(_teacher(), 900, shove)
    assert np.vstack([r[2] for r in t.rows]).max() <= HI[3] - P.LIMIT_MARGIN_RAD + 1e-12


def test_keys_set_the_hand_and_keep_the_events():
    t = _run(_teacher(), 120, keys={10: "g", 100: "o"})
    hands = np.vstack([r[3] for r in t.rows])
    assert np.allclose(hands[5], HAND0) and np.allclose(hands[50], P.GRIP_RAD) and np.allclose(hands[-1], P.OPEN_RAD)
    assert [e.name for e in t.events] == ["grip", "open"]


def test_space_locks_the_arm_against_a_push():
    t = _run(_teacher(), 600, _push_j4(), keys={5: " "})
    assert np.allclose(t.command, Q0, atol=1e-9)


#: 쥔 컵 무게로 더 처지는 몫(손목 게인이 낮아 j5 · j7 이 push 0.02 rad 를 넘는다) — 11.04 실기: g 뒤 j7 이 −0.73 rad 까지 흘렀다.
PAYLOAD = np.array([0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.05])


def _loaded(grip_k, then=None):
    """grip_k 틱부터 컵 무게만큼 더 처지는 팔. then(k, cmd) 는 그 위에 손이 정한 실측(또는 None)."""
    def path(k, cmd):
        load = PAYLOAD if k >= grip_k else 0.0
        imposed = then(k, cmd) if then else None
        if imposed is not None:
            return imposed - load                              # 밀어도 손목은 컵 무게만큼 처진 채
        return cmd - DROOP - load
    return path


def test_the_weight_of_a_gripped_cup_is_not_followed_as_a_push():
    t = _run(_teacher(), 900, _loaded(60), keys={60: "g"})
    assert np.allclose(t.command, Q0, atol=1e-9)          # 처짐을 따라 내려가지 않는다
    assert not t.state.locked                               # 기준선을 다시 잡은 뒤 잠금이 풀렸다


def test_after_the_grip_rebaseline_a_push_is_still_followed():
    t = _run(_teacher(), 900, _loaded(60, _push_j4(start_k=400)), keys={60: "g"})
    assert t.command[3] > Q0[3] + 0.25
    assert np.allclose(np.delete(t.command, 3), np.delete(Q0, 3), atol=0.02)


def test_a_grip_keeps_a_lock_the_user_set():
    t = _run(_teacher(), 900, _loaded(60, _push_j4(start_k=400)), keys={5: " ", 60: "g"})
    assert t.state.locked
    assert np.allclose(t.command, Q0, atol=1e-9)


def _taught(tmp_path):
    # 쥔 뒤 팔이 잠긴 동안(손 이동 + settle_sec ≈ 1.4 s)은 기다렸다가 민다 — 틱 200 부터. 끝 틱 900 = 15 s(녹화 끝보다 뒤)
    t = _run(_teacher(), 900, _push_j4(start_k=200), keys={30: "g", 400: "o"})
    path = P.save_recording(tmp_path / "holder1.npz", P.build_recording(t))
    return t, path


def test_the_recording_is_trimmed_and_replays_through_rh_replay_at_full_rate(tmp_path):
    t, path = _taught(tmp_path)
    traj = R.load_replay(path)
    assert traj.side == "right" and traj.names == R.expected_names("right")
    # 앞 정지(쥐기 전 기다림)는 잘려 밀기 시작 0.5 s 전부터, 끝은 폄(틱 400) 뒤 손이 다 펴지고(1.08 rad / 1.0 rad/s) 1 s 더
    # 그 뒤 들어온 길을 거꾸로 되짚는 후진이 붙는다(같은 길이 − 겹친 1 프레임).
    first_s, last_s = 199 * DT - P.PRE_ROLL_S, 400 * DT + 1.08 / P.HAND_VEL_RAD_S + P.POST_ROLL_S
    fwd = traj.segments[1]["end"] + 1
    assert fwd == pytest.approx((last_s - first_s) / DT, abs=3)
    assert traj.n_frames == 2 * fwd - 1
    assert np.allclose(traj.q[0, 7:], P.GRIP_RAD) and np.allclose(traj.q[-1, 7:], P.OPEN_RAD)
    assert [s["label"] for s in traj.segments] == ["carry", "release", "retreat"]
    # 그대로(rate_scale 1.0) 재생 가능 — replay_traj_rh 의 단계 한계(reduced 0.25 · 손 2.1 의 0.8)
    assert R.max_rate_scale(traj, arm_vmax=0.8 * 0.25, hand_vmax=0.8 * 2.1) == pytest.approx(1.0)
    plan = R.build_plan(traj, traj.q[0], rate_scale=1.0)
    assert R.speed_violations(plan, traj.names, arm_vmax=0.8 * 0.25, hand_vmax=0.8 * 2.1) == []


def test_the_recording_is_also_an_arm_only_replay_to_pd_file(tmp_path):
    _, path = _taught(tmp_path)
    rp = _tool("replay_to_pd")
    args = type("A", (), {"npz": path, "npz_key": None, "joints": list(P.arm_names("right")), "env": 0, "dt": 0.02})()
    q, dt = rp.load_frames(args)
    assert q.shape[1] == 7 and dt == pytest.approx(DT)
    with np.load(path) as z:
        assert list(z["hand_event_names"]) == ["grip", "open"] and z["hand_event_t"][0] < 0 < z["hand_event_t"][1]


def test_nothing_to_save_without_motion():
    t = _run(_teacher(), 100, keys={10: "g"})
    with pytest.raises(P.TeachError, match="움직이지 않았다"):
        P.build_recording(t)


def test_bad_poses_and_files_are_refused(tmp_path):
    with pytest.raises(P.TeachError):
        P.parse_pose("1,2", P.GRIP_RAD, "--grip")
    with pytest.raises(P.TeachError):
        P.save_recording(tmp_path / "x.npy", {})


def test_tool_refuses_a_non_npz_or_existing_output_before_ros(tmp_path, capsys):
    tool = _tool("teach_place")
    assert tool.main(["--side", "right", "--out", str(tmp_path / "a.txt")]) == 1
    (tmp_path / "b.npz").write_bytes(b"")
    assert tool.main(["--side", "right", "--out", str(tmp_path / "b.npz")]) == 1
    assert "이미 있다" in capsys.readouterr().err


def test_tool_pd_gate():
    tool = _tool("teach_place")
    assert tool.pd_fault({}, 10.0)
    assert tool.pd_fault({"_t": 9.9, "phase": "IDLE"}, 10.0)
    assert tool.pd_fault({"_t": 9.9, "phase": "TRACKING", "estop": True}, 10.0)
    assert tool.pd_fault({"_t": 9.9, "phase": "TRACKING"}, 10.0) is None
    assert tool.pd_fault({"_t": 9.9, "phase": "RAMPING"}, 10.0, tool.MOVING) is None
    lo, hi = tool.arm_limits("left")
    assert lo.shape == (7,) and np.all(lo < hi)


def test_the_recording_starts_with_the_hand_already_gripped(tmp_path):
    """쥘 때 컵 무게로 처지는 것이 움직임으로 잡혀도, 첫 프레임 손은 쥔 자리 — 재생이 시작하며 손을 펴지 않는다."""
    t = _run(_teacher(), 900, _loaded(30, _push_j4(start_k=200)), keys={30: "g", 400: "o"})
    rec = P.build_recording(t)
    assert np.allclose(rec["hand_q_cmd"][0], P.GRIP_RAD)
    carry = json.loads(str(rec["meta_segments"]))[0]
    assert np.allclose(rec["hand_q_cmd"][carry["start"]:carry["end"] + 1], P.GRIP_RAD)
    assert np.all(rec["hand_event_t"][rec["hand_event_names"] == "grip"] < 0)   # 쥠은 녹화 앞


def test_a_recording_without_a_grip_is_refused():
    t = _run(_teacher(), 600, _push_j4())
    with pytest.raises(P.TeachError, match="쥠"):
        P.build_recording(t)


def _push_j4_wrist_flops(start_k=90, ticks=150, total=0.3, flop=0.06):
    """관절 4 를 끄는 동안 손목이 손 무게로 flop 만큼 돌아간다(손목은 아무도 잡지 않았다)."""
    def path(k, cmd):
        if start_k <= k < start_k + ticks:
            q = (Q0 - DROOP).copy()
            frac = (k - start_k + 1) / ticks
            q[3] += total * frac
            q[P.WRIST] = cmd[P.WRIST] - DROOP[P.WRIST] - flop * min(1.0, 4 * frac)
            return q
        return None
    return path


def test_dragging_the_arm_does_not_carry_the_wrist_along():
    t = _run(_teacher(), 600, _push_j4_wrist_flops())
    cmds = np.vstack([r[2] for r in t.rows])
    assert t.command[3] > Q0[3] + 0.25                            # 팔은 따라왔다
    assert np.allclose(cmds[:, P.WRIST], Q0[P.WRIST])             # 손목 명령은 그 자리
    assert "follow" in {r[4] for r in t.rows}


def test_twisting_the_wrist_itself_is_followed():
    def twist(k, cmd):
        if 90 <= k < 240:
            q = (Q0 - DROOP).copy()
            q[P.WRIST] += 0.4 * (k - 89) / 150
            return q
        return None
    t = _run(_teacher(), 600, twist)
    assert t.command[P.WRIST] > Q0[P.WRIST] + 0.3


def test_raw_rows_keep_every_tick_with_its_mode():
    t = _run(_teacher(), 300, _push_j4(), keys={20: "g"})
    raw = P.raw_rows(t)
    assert raw["q_cmd"].shape == (300, 7) and raw["mode"].shape == (300,)
    assert set(raw["mode"]) <= {"hold", "follow"} and list(raw["event_names"]) == ["grip"]




def test_the_retreat_retraces_the_carry_path_with_the_hand_open(tmp_path):
    _, path = _taught(tmp_path)
    traj = R.load_replay(path)
    rel, ret = traj.segments[1], traj.segments[2]
    fwd = traj.q[: rel["end"] + 1, :7]
    back = traj.q[ret["start"]:, :7]
    assert np.allclose(back, fwd[-2::-1])                       # 같은 길을 거꾸로
    assert np.allclose(traj.q[-1, :7], traj.q[0, :7])           # 시작 자세로 돌아온다
    assert np.allclose(traj.q[ret["start"]:, 7:], P.OPEN_RAD)   # 내내 편 손


def test_arm_motion_after_the_open_is_dropped():
    # 폄(틱 400) 뒤 손이 다 펴지고 1 s 지나(틱 ≈ 525) 사람이 팔을 다시 끌어도(틱 700~) 녹화에는 들어가지 않는다
    def path(k, cmd):
        a = _push_j4(start_k=200)(k, cmd)
        return a if a is not None else _push_j4(start_k=700, ticks=100, total=0.1)(k, cmd)
    t = _run(_teacher(), 1000, path, keys={30: "g", 400: "o"})
    rec = P.build_recording(t)
    segs = json.loads(str(rec["meta_segments"]))
    fwd = rec["arm_q"][: segs[1]["end"] + 1]
    assert np.abs(fwd[-60:, 3] - fwd[-1, 3]).max() < 0.005      # 끝 1 s 는 놓은 자리에 그대로
    assert fwd[-1, 3] > Q0[3] + 0.25                             # 두 번째 끌기는 안 들어갔다
    assert str(rec["meta_retreat"]) == "reverse"


def test_without_an_open_there_is_no_retreat():
    t = _run(_teacher(), 600, _push_j4(start_k=200), keys={30: "g"})
    rec = P.build_recording(t)
    assert [s["label"] for s in json.loads(str(rec["meta_segments"]))] == ["place"]
    assert str(rec["meta_retreat"]) == "none"


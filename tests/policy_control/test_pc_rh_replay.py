"""rh_replay — 방안 2 재생 계획(sim 성공 궤적 → pd 입력) 계약 테스트 (ROS 없음).

지키는 것:
  ① 이름·측·순서가 canonical 과 같아야 읽는다(손 열이 섞이면 엄지 지령이 새끼로 간다).
  ② 손 기본 열은 지령 q*(쥐는 힘), 팔 기본 열은 측정 q.
  ③ 시간만 늘리고 경로는 그대로 — 끝 프레임은 정확히, 발행 간격은 워치독보다 짧게.
  ④ 손 입력 모드 사건은 구간 경계에서, 바뀔 때만.
  ⑤ 한계·속도 위반을 미리 잡는다.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from policy_control import rh_replay as R

pytestmark = pytest.mark.unit

T = 31


def write(tmp_path, *, side="left", hand_names=None, segs=None, arm=None, hand_cmd=None, **extra):
    p = side[0]
    arm = np.linspace(0, 0.3, T)[:, None].repeat(7, 1) if arm is None else arm
    hand_cmd = np.full((T, 6), 0.8) if hand_cmd is None else hand_cmd
    segs = segs or [{"label": "grasp", "start": 0, "end": 19, "hand_command": "admittance"},
                    {"label": "place", "start": 20, "end": T - 1, "hand_command": "position"}]
    d = {
        "meta_schema": np.array(R.SCHEMA), "meta_side": np.array(side),
        "meta_arm_names": np.array([f"{p}_aj_{i}" for i in range(1, 8)]),
        "meta_hand_names": np.array(hand_names or [f"{p}_hj_{a}" for a in R.HAND_AXES]),
        "meta_step_dt": np.array(1 / 60), "meta_segments": np.array(json.dumps(segs)),
        "arm_q": arm, "arm_q_cmd": arm + 0.01, "arm_q_pd": arm + 0.02,
        "hand_q": np.full((T, 6), 0.5), "hand_q_cmd": hand_cmd, "hand_q_pd": np.full((T, 6), 0.7),
        "cup": np.zeros((T, 13)),
    }
    d.update(extra)
    f = tmp_path / "traj.npz"
    np.savez(f, **d)
    return f


def test_default_columns(tmp_path):
    tr = R.load_replay(write(tmp_path))
    assert tr.names == R.expected_names("left")
    assert tr.q[0, 7] == pytest.approx(0.8)          # 손 = 지령
    assert tr.q[-1, 0] == pytest.approx(0.3)         # 팔 = 측정
    tr2 = R.load_replay(write(tmp_path), arm_src="arm_q_cmd", hand_src="hand_q")
    assert tr2.q[0, 7] == pytest.approx(0.5) and tr2.q[0, 0] == pytest.approx(0.01)


def test_hand_order_must_match(tmp_path):
    names = [f"l_hj_{a}" for a in reversed(R.HAND_AXES)]
    with pytest.raises(R.ReplayError, match="canonical"):
        R.load_replay(write(tmp_path, hand_names=names))


def test_side_prefix_must_match(tmp_path):
    with pytest.raises(R.ReplayError):
        R.load_replay(write(tmp_path, side="right", hand_names=[f"l_hj_{a}" for a in R.HAND_AXES]))


def test_nonfinite_rejected(tmp_path):
    arm = np.zeros((T, 7))
    arm[5, 3] = np.nan
    with pytest.raises(R.ReplayError, match="프레임 5"):
        R.load_replay(write(tmp_path, arm=arm))


def test_segments_must_cover(tmp_path):
    bad = [{"label": "a", "start": 0, "end": 10, "hand_command": "admittance"},
           {"label": "b", "start": 12, "end": T - 1, "hand_command": "admittance"}]
    with pytest.raises(R.ReplayError, match="비거나"):
        R.load_replay(write(tmp_path, segs=bad))
    with pytest.raises(R.ReplayError, match="hand_command"):
        R.load_replay(write(tmp_path, segs=[{"label": "a", "start": 0, "end": T - 1, "hand_command": "force"}]))


def test_resample_stretches_time_keeps_path():
    q = np.linspace(0, 1, 11)[:, None]
    out, pos = R.resample(q, 0.1, 0.5, 0.05)
    assert out[0, 0] == 0.0 and out[-1, 0] == pytest.approx(1.0)
    assert pos[-1] == 10.0
    assert len(out) == 41                                 # 1 s → 2 s, 0.05 s 격자
    assert np.all(np.diff(out[:, 0]) >= -1e-12)
    with pytest.raises(R.ReplayError):
        R.resample(q, 0.1, 1.5, 0.05)


def test_plan_ramp_modes(tmp_path):
    tr = R.load_replay(write(tmp_path))
    start = tr.q[0].copy()
    start[0] -= 0.05                                      # 실측이 첫 프레임에서 0.05 rad
    plan = R.build_plan(tr, start, rate_scale=0.25, pub_hz=60)
    assert plan.n_ramp == 31                              # 0.05 / (0.1 rad/s / 60) = 30 칸 + 시작
    assert np.allclose(plan.q[0], start) and np.allclose(plan.q[plan.n_ramp], tr.q[0])
    assert np.all(plan.qd[: plan.n_ramp] == 0.0)
    assert plan.mode_at[0] == (0, "admittance")
    k, mode = plan.mode_at[1]
    assert mode == "position" and plan.frame_of[k] >= 20 > plan.frame_of[k - 1]
    assert plan.pub_dt < 0.25                             # 워치독 안


def test_hand_ramp_faster_than_arm(tmp_path):
    tr = R.load_replay(write(tmp_path))
    start = tr.q[0].copy()
    start[7:] += 1.0                                      # 차렷 주먹 → 편 손: 1 rad
    slow = R.build_plan(tr, start, rate_scale=1.0)
    fast = R.build_plan(tr, start, rate_scale=1.0, hand_ramp_speed=0.5)
    assert slow.n_ramp == 601 and fast.n_ramp == 121      # 1 rad ÷ 0.1 · 0.5 rad/s × 60 Hz (+ 시작)


def test_no_duplicate_mode_event(tmp_path):
    segs = [{"label": "a", "start": 0, "end": 9, "hand_command": "admittance"},
            {"label": "b", "start": 10, "end": T - 1, "hand_command": "admittance"}]
    plan = R.build_plan(R.load_replay(write(tmp_path, segs=segs)), np.zeros(13) + 0.0, rate_scale=1.0)
    assert len(plan.mode_at) == 1


def test_speed_and_max_rate(tmp_path):
    tr = R.load_replay(write(tmp_path))                   # 팔 0.3 rad / 0.5 s = 0.6 rad/s
    assert R.max_rate_scale(tr, arm_vmax=0.25, hand_vmax=2.0) == pytest.approx(0.25 / 0.6, rel=1e-6)
    fast = R.build_plan(tr, tr.q[0], rate_scale=1.0)
    assert R.speed_violations(fast, tr.names, arm_vmax=0.25, hand_vmax=2.0)
    slow = R.build_plan(tr, tr.q[0], rate_scale=0.4)
    assert not R.speed_violations(slow, tr.names, arm_vmax=0.25, hand_vmax=2.0)


def test_limit_violations():
    q = np.array([[0.0, 0.5], [0.2, 0.9]])
    v = R.limit_violations(q, np.array([0.1, 0.0]), np.array([1.0, 0.8]), ("a", "b"))
    assert len(v) == 2 and v[0].startswith("a: 프레임 0") and v[1].startswith("b: 프레임 1")

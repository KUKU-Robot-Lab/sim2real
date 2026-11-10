"""★11.10 qd_target_diff — joint_target 차분으로 q̇* 를 만들어 vel_ff 경로가 실제 모터 qd 명령에 닿게 한다.

잠그는 것:
  · 기본(키 없음)은 꺼짐: 배포 pd yaml 이 그대로 로드되고, take_target 은 메시지 velocity 를 비트 단위로 그대로 쓴다
  · 켜면 q̇* = Δq/Δt(같은 에피소드 · 간격 ≤ max_dt), 첫 목표 · 에피소드 바뀜 · 긴 간격 · hold_sec 지난 목표는 0
  · 켜면 pd_law.step 의 qd 명령 = clip(vel_ff_scale·q̇*, ±vel_ff_cap), q 명령은 q̇* 와 무관(이중 보상 회귀는 test_pc_pd_law)
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control import pd_arm as A
from policy_control import pd_law as L
from policy_control.codec import JointSample

CONFIG = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "config"
ARM = [f"r_aj_{i}" for i in range(1, 8)]
N = 7

pytestmark = pytest.mark.unit


def _sample(q, qd):
    return JointSample(names=tuple(ARM), position=np.asarray(q, float), velocity=np.asarray(qd, float),
                       effort=None, stamp=0.0)


def _make(diff: L.QdTargetDiff | None):
    """take_target · _select_target 가 보는 것만 가진 팔(test_pc_node_pd._recovery_unit 과 같은 방식)."""
    u = SimpleNamespace(side="right", arm_joints=ARM, hand_joints=[], hold=None, target=None, hand_target=None,
                        stopped_episode=None, cfg=SimpleNamespace(watchdog_sec=0.25, qd_target_diff=diff))
    u._diff_qd = lambda *a: A.ArmUnit._diff_qd(u, *a)
    return u


def _law_cfg(cap=0.5, scale=1.0):
    return L.PdLawCfg(max_vel=100.0, lead_sec=0.1, lead_vel=100.0, vel_ff_scale=scale, vel_ff_cap=cap,
                      effort_cap=20.0, lower=np.full(N, -3.0), upper=np.full(N, 3.0), gravity_mode="off")


def _law_cmd(target, cfg):
    inp = L.PdInputs(q_target=target.q, qd_target=target.qd, tau_request=np.zeros(N), q_meas=np.zeros(N),
                     qd_meas=np.zeros(N), target_fresh=True, new_policy_step=True, advance=True, tracking=True,
                     dt=1 / 120)
    return L.step(L.initial_state(np.zeros(N)), inp, cfg)[1]


# ---------------------------------------------------------------- 기본 꺼짐
def test_deploy_pd_yamls_leave_the_diff_off():
    for name in ("pd_left.yaml", "pd_right.yaml"):
        assert L.load_pd_config(CONFIG / name).qd_target_diff is None


def test_rh56f1_mission_pd_turns_on_stage_1_and_t2_adds_friction():
    """★11.10 사용자 "속도 상한은 풀어야": 미션 PD(pd_rh56f1·_exec)는 1단계(q̇* 차분 + vel_ff_cap 0.5),
    tracking_1110/*_T2 는 거기에 fric_ff 만 더한다. 두 미션 yaml 은 execute 한 줄만 달라야 한다."""
    for name in ("pd_rh56f1.yaml", "pd_rh56f1_exec.yaml"):
        c = L.load_pd_config(CONFIG / name)
        assert c.qd_target_diff == L.QdTargetDiff(max_dt=0.1, hold_sec=0.07) and c.vel_ff_cap == 0.5
        assert c.fric_ff is None
        t2 = L.load_pd_config(CONFIG / "tracking_1110" / name.replace(".yaml", "_T2.yaml"))
        assert t2.fric_ff is not None and t2.qd_target_diff == c.qd_target_diff and t2.vel_ff_cap == 0.5
        assert t2.execute == c.execute


@pytest.mark.parametrize("msg_qd", [np.zeros(N), np.linspace(-0.3, 0.4, N)])
def test_off_keeps_the_message_velocity_bit_for_bit(msg_qd):
    u = _make(None)
    for k, q in enumerate((np.zeros(N), np.full(N, 0.01), np.full(N, 0.03))):
        assert A.ArmUnit.take_target(u, _sample(q, msg_qd), k + 1, 0.033 * k, episode="1")
        assert u.target.qd.tobytes() == np.asarray(msg_qd, float).tobytes()
        assert A.ArmUnit._select_target(u, 10.0, np.zeros(N)) is u.target    # 오래된 목표도 손대지 않는다
    assert not hasattr(u, "_qd_prev")


# ---------------------------------------------------------------- 차분
def test_difference_velocity_rules():
    q1, q0 = np.full(N, 0.02), np.zeros(N)
    assert np.allclose(L.target_difference_velocity(q1, 1.04, (q0, 1.0), 0.1), 0.5)
    assert not L.target_difference_velocity(q1, 1.0, None, 0.1).any()          # 첫 목표
    assert not L.target_difference_velocity(q1, 1.2, (q0, 1.0), 0.1).any()     # 간격 > max_dt
    assert not L.target_difference_velocity(q1, 1.0, (q0, 1.0), 0.1).any()     # 간격 0
    assert not L.target_difference_velocity(q1, 0.9, (q0, 1.0), 0.1).any()     # 시각 역행


def test_on_differentiates_within_an_episode_and_resets_across():
    u = _make(L.QdTargetDiff(max_dt=0.1, hold_sec=0.07))
    zero = np.zeros(N)
    A.ArmUnit.take_target(u, _sample(zero, zero), 1, 1.00, episode="1")
    assert not u.target.qd.any()                                               # 첫 목표 = 0
    A.ArmUnit.take_target(u, _sample(np.full(N, 0.01), zero), 2, 1.04, episode="1")
    assert np.allclose(u.target.qd, 0.25)
    A.ArmUnit.take_target(u, _sample(np.full(N, 0.02), zero), 1, 1.07, episode="2")
    assert not u.target.qd.any()                                               # 새 에피소드 = 0
    A.ArmUnit.take_target(u, _sample(np.full(N, 0.05), zero), 2, 1.30, episode="2")
    assert not u.target.qd.any()                                               # 정책 멈춤(간격 > max_dt) = 0


def test_on_zeroes_the_feed_forward_once_the_policy_goes_quiet():
    u = _make(L.QdTargetDiff(max_dt=0.1, hold_sec=0.07))
    A.ArmUnit.take_target(u, _sample(np.zeros(N), np.zeros(N)), 1, 1.00, episode="1")
    A.ArmUnit.take_target(u, _sample(np.full(N, 0.01), np.zeros(N)), 2, 1.04, episode="1")
    fresh = A.ArmUnit._select_target(u, 1.05, np.zeros(N))
    stale = A.ArmUnit._select_target(u, 1.04 + 0.08, np.zeros(N))
    assert np.allclose(fresh.qd, 0.25)
    assert not stale.qd.any() and np.array_equal(stale.q, fresh.q) and stale.seq == fresh.seq


# ---------------------------------------------------------------- pd_law 까지
def test_on_reaches_the_motor_qd_command_through_vel_ff_and_off_does_not():
    zero = np.zeros(N)
    on, off = _make(L.QdTargetDiff()), None
    A.ArmUnit.take_target(on, _sample(zero, zero), 1, 1.00, episode="1")
    A.ArmUnit.take_target(on, _sample(np.full(N, 0.03), zero), 2, 1.033, episode="1")
    cmd = _law_cmd(on.target, _law_cfg(cap=0.5))
    assert np.allclose(cmd.qd, 0.5)                                            # 0.909 rad/s → cap 0.5
    assert np.allclose(_law_cmd(on.target, _law_cfg(cap=2.0, scale=0.5)).qd, 0.5 * 0.03 / 0.033)
    assert not _law_cmd(on.target, _law_cfg(cap=0.0)).qd.any()                 # cap 0 = 배포 기본 → 여전히 0
    off = _make(None)
    A.ArmUnit.take_target(off, _sample(zero, zero), 1, 1.00, episode="1")
    A.ArmUnit.take_target(off, _sample(np.full(N, 0.03), zero), 2, 1.033, episode="1")
    assert not _law_cmd(off.target, _law_cfg(cap=0.5)).qd.any()               # 꺼짐 = 메시지 0 → qd 명령 0
    assert np.array_equal(cmd.q, _law_cmd(off.target, _law_cfg(cap=0.5)).q)   # q 명령은 q̇* 와 무관


# ---------------------------------------------------------------- 로더
def test_loader_parses_the_block_and_rejects_bad_values(tmp_path):
    src = (CONFIG / "pd_rh56f1.yaml").read_text()
    p = tmp_path / "pd.yaml"
    p.write_text(src + "\nqd_target_diff: {max_dt: 0.08, hold_sec: 0.05}\n")
    assert L.load_pd_config(p).qd_target_diff == L.QdTargetDiff(max_dt=0.08, hold_sec=0.05)
    p.write_text(src + "\nqd_target_diff: {}\n")
    assert L.load_pd_config(p).qd_target_diff == L.QdTargetDiff()
    for bad in ("{hold_sec: 0}", "{max_dt: -1}", "{alpha: 0.5}", "{max_dt: true}"):
        p.write_text(src + f"\nqd_target_diff: {bad}\n")
        with pytest.raises(L.PdConfigError):
            L.load_pd_config(p)


# ---------------------------------------------------------------- ★11.10 마찰 전향 fric_ff
FRIC = L.FricFf(bv=(0.0, 2.6, 3.5, 2.8, 1.1, 0.9, 1.7), fc=(0.3, 0.7, 0.55, 1.28, 0.06, 0.09, 0.0),
                cap=(1.0,) * N, v_eps=0.02)


def _fric_law_cmd(qd, fric, tracking=True, effort_cap=20.0, tau_req=None):
    cfg = L.PdLawCfg(max_vel=100.0, lead_sec=0.1, lead_vel=100.0, vel_ff_scale=1.0, vel_ff_cap=0.5,
                     effort_cap=effort_cap, lower=np.full(N, -3.0), upper=np.full(N, 3.0), gravity_mode="off",
                     fric_ff=fric)
    inp = L.PdInputs(q_target=np.zeros(N), qd_target=np.asarray(qd, float),
                     tau_request=np.zeros(N) if tau_req is None else tau_req, q_meas=np.zeros(N),
                     qd_meas=np.full(N, -5.0), target_fresh=True, new_policy_step=True, advance=True,
                     tracking=tracking, dt=1 / 120)
    return L.step(L.initial_state(np.zeros(N)), inp, cfg)[1]


def test_deploy_pd_yamls_leave_the_friction_off():
    for name in ("pd_rh56f1.yaml", "pd_rh56f1_exec.yaml", "pd_left.yaml", "pd_right.yaml"):
        assert L.load_pd_config(CONFIG / name).fric_ff is None


def test_friction_off_and_zero_target_velocity_keep_tau_bit_for_bit():
    qd = np.linspace(-0.3, 0.4, N)
    off = _fric_law_cmd(qd, None)
    assert np.array_equal(off.tau, np.zeros(N))
    assert np.array_equal(_fric_law_cmd(np.zeros(N), FRIC).tau, off.tau)      # q̇* = 0 → τ_fric = 0


def test_friction_follows_the_target_velocity_not_the_measured_one():
    qd = np.array([0.0, 0.1, -0.1, 0.2, 0.05, -0.05, 0.3])
    cmd = _fric_law_cmd(qd, FRIC)                     # qd_meas = −5 rad/s 는 결과에 들어가지 않아야 한다
    want = np.clip(np.array(FRIC.bv) * qd + np.array(FRIC.fc) * np.tanh(qd / 0.02), -1.0, 1.0)
    assert np.allclose(cmd.tau, want) and not cmd.effort_fault
    assert np.all(np.abs(cmd.tau) <= 1.0 + 1e-12)       # 관절별 cap


def test_friction_is_off_outside_tracking_and_respects_the_effort_cap():
    qd = np.full(N, 0.2)
    assert np.array_equal(_fric_law_cmd(qd, FRIC, tracking=False).tau, np.zeros(N))
    big = _fric_law_cmd(qd, FRIC, effort_cap=0.5, tau_req=np.full(N, 0.4))   # 합 τ 를 effort_cap 으로 자른다
    assert np.all(np.abs(big.tau) <= 0.5 + 1e-12) and not big.effort_fault


def test_friction_loader_parses_and_rejects(tmp_path):
    src = (CONFIG / "pd_rh56f1.yaml").read_text()
    p = tmp_path / "pd.yaml"
    p.write_text(src + "\nfric_ff: {bv: [0,2.6,3.5,2.8,1.1,0.9,1.7], fc: [0.3,0.7,0.55,1.28,0.06,0.09,0],"
                       " cap: [1,1,1,1,1,1,1]}\n")
    got = L.load_pd_config(p).fric_ff
    assert got.bv[2] == 3.5 and got.v_eps == 0.02
    for bad in ("{bv: [1], fc: [1, 2], cap: [1]}", "{bv: [-1], fc: [0], cap: [1]}", "{bv: [1], fc: [0]}",
                "{bv: [1], fc: [0], cap: [1], v_eps: 0}", "{bv: [1], fc: [0], cap: [1], gain: 2}"):
        p.write_text(src + f"\nfric_ff: {bad}\n")
        with pytest.raises(L.PdConfigError):
            L.load_pd_config(p)

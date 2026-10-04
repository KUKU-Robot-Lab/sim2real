"""pour_fj(RH56F1 양팔 붓기) 계약 · 디코더 · 관측 — hdgp pour_fabric_mimic 의 행동 · 관측 규칙을 그대로 옮겼는가. 09.29.

기준: hdgp tasks/pour_fabric_mimic/side_rig.py step_joint_arm · direct_hand_targets · joint_err,
pour_fabric_env.py _pre_physics_step · _side_obs · _get_observations.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from policy_control import _paths
from policy_control import pour_fj as F
from policy_control.pour_profiles import load_mimic_pair

F01 = Path("/media/user/DATA/kuku_ai/rl/rl_runs_20260926/pour_bi_rh/t2r_rh4_f01")
F01_CKPT = F01 / "last_open-rh_b_pour_fj-lstm_ep_2300_rew_1107.8425.pth"
URDF = _paths.RL_WS / "hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
HOMES = _paths.SIM2REAL / "deploy/policy_control/config/homes/rh56f1_pour_fj.yaml"
# pour_bi_rh 세션 결정론 재생(git 밖): f01 ep800 = arm4090 ~/logs/t2r/det_rh/ · f02 ep3000 = server ~/logs/t2r/det_rh/
RH5_RUNS = [_paths.SIM2REAL / f"logs/policy/{n}" for n in ("t2r_rh5_f01_ep800", "t2r_rh5_f02_ep3000")]


def _side(role: str) -> F.FjSide:
    p = "r" if role == "src" else "l"
    hand = [f"{p}_hj_{n}" for n in ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")]
    return F.FjSide(role=role, side=F.SIDE_OF[role], arm_joints=[f"{p}_aj_{i}" for i in range(1, 8)],
                    arm_home=[0.0] * 7, arm_lo=[-1.0] * 7, arm_hi=[1.0] * 7, hand_joints=hand,
                    hand_obs_order=[f"{p}_hj_{n}" for n in F.ASSUMED_OBS_ORDER],
                    hand_open=[1.57, 0, 0, 0, 0, 0], hand_grip=[1.2, 0.24, 1.08, 1.08, 0.85, 0.85],
                    hand_lim_lo=[0.0] * 6, hand_lim_hi=[2.0944, 0.47456, 1.52856, 1.52856, 1.52856, 1.52856],
                    hand_finger=[0, 0, 1, 2, 3, 4], hand_freeze=[True] * 6,
                    palm_body=f"{p}_hl_palm_sensor", tip_bodies=[f"{p}_hl_{f}_tip" for f in F.FINGERS])


def _contract(**over) -> F.FjContract:
    c = F.FjContract(schema=F.SCHEMA, task=F.TASK_PREFIX, run_dir="", checkpoint="", checkpoint_md5="",
                     env_yaml_sha1="", agent_yaml_sha1="", asset="openarm_rh56f1_bi_rl", policy_hz=60.0, episode_s=15.0,
                     obs_dim=165, action_dim=26, obs_clip=5.0, action_clip=1.0, recurrent=True, hold_steps=30,
                     arm_mode="absolute", k_arm=0.05, arm_ema=0.1, arm_abs_vmax=0.3, hand_ema=0.1, hand_full_range_s=1.0,
                     hand_range="grip", hand_freeze=True, freeze_threshold_n=1.0, joint_pos_err_max=1.2,
                     tactile_clip_n=10.0, tactile_tanh_n=3.0, obs_tips_rel_palm=False, obs_mouth_diff=False,
                     cup_mouth_z=0.053885, hand_obs_order_source="assumed", sides={r: _side(r) for r in F.ROLES})
    return replace(c, **over)


# ---------------------------------------------------------------- 팔
def test_hold_keeps_the_arm_at_home_but_the_hand_follows_the_action():
    d = F.FjDecoder(_contract())
    out = d.step(np.ones(26), active=False)
    assert np.allclose(out["src"][0], 0.0)                                       # 팔 = home
    assert not np.allclose(out["src"][1], [1.57, 0, 0, 0, 0, 0])                 # 손은 움직였다


def test_absolute_arm_is_piecewise_linear_around_home_with_a_speed_cap():
    c = _contract(arm_mode="absolute")
    d = F.FjDecoder(c)
    a = np.zeros(26)
    a[0] = 1.0                                                                    # src j1 → hi
    q = d.step(a, active=True)["src"][0]
    assert q[0] == pytest.approx(0.3 / 60.0)                                      # α·(1−0)=0.1 > cap 0.005
    for _ in range(2000):
        q = d.step(a, active=True)["src"][0]
    assert q[0] == pytest.approx(1.0, abs=1e-3) and np.allclose(q[1:], 0.0)
    a[0] = -0.5
    for _ in range(2000):
        q = d.step(a, active=True)["src"][0]
    assert q[0] == pytest.approx(-0.5, abs=1e-3)                                  # home + a·(home − lo)


def test_increment_arm_is_the_grasp_fj_law():
    d = F.FjDecoder(_contract(arm_mode="increment"))
    a = np.zeros(26)
    a[13] = 1.0                                                                   # rcv j1
    q = d.step(a, active=True)["rcv"][0]
    assert q[0] == pytest.approx(0.1 * 0.05)                                      # α · k


# ---------------------------------------------------------------- 손
def test_hand_grip_range_ema_and_per_step_cap():
    c = _contract()
    d = F.FjDecoder(c)
    lo, hi = d.hand_range(c.sides["src"])
    assert np.allclose(lo, [1.2, 0, 0, 0, 0, 0]) and np.allclose(hi, [1.57, 0.24, 1.08, 1.08, 0.85, 0.85])
    a = np.zeros(26)
    a[7:13] = 1.0                                                                 # 전부 hi
    h = d.step(a, active=True)["src"][1]
    cap = np.array([2.0944, 0.47456, 1.52856, 1.52856, 1.52856, 1.52856]) / 60.0
    assert np.allclose(h[2:], np.minimum(0.1 * hi[2:], cap[2:]))
    for _ in range(400):
        h = d.step(a, active=True)["src"][1]
    assert np.allclose(h, hi, atol=1e-3)


def test_limits_range_is_the_whole_joint_before_the_grip_feature():
    c = _contract(hand_range="limits")
    lo, hi = F.FjDecoder(c).hand_range(c.sides["rcv"])
    assert np.allclose(lo, 0.0) and hi[0] == pytest.approx(2.0944)


def test_freeze_stops_only_closing_of_a_touching_finger():
    c = _contract()
    d = F.FjDecoder(c)
    a = np.zeros(26)
    a[7:13] = 1.0                                                                 # 닫는 쪽(4지 grip > open)
    touch = {"src": [False, True, False, False, False], "rcv": [False] * 5}       # 검지만 닿음
    h = d.step(a, active=True, touch=touch)["src"][1]
    assert h[2] == 0.0 and h[3] > 0.0                                             # 검지 멈춤 · 중지 닫힘
    a[7:13] = -1.0                                                                # 여는 쪽은 막지 않는다
    d.state["src"].hand_target[:] = [1.4, 0.1, 0.5, 0.5, 0.5, 0.5]
    h2 = d.step(a, active=True, touch={"src": [True] * 5, "rcv": [False] * 5})["src"][1]
    assert h2[2] < 0.5


# ---------------------------------------------------------------- 관측
def _meas(role, k=0.0):
    s = _side(role)
    return F.FjSideMeas(arm_q=np.arange(7) + k, arm_qd=np.zeros(7),
                        hand_q={n: 0.01 * i + k for i, n in enumerate(s.hand_joints)},
                        palm_pos=np.array([0.3, 0.0, 0.4]), palm_R=np.eye(3),
                        tips=np.tile([0.35, 0.0, 0.3], (5, 1)), cup_pos=np.array([0.38, -0.16, 0.26]),
                        cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.array([0, 3.0, 0, 0, 20.0]))


def test_obs_layout_matches_the_training_actor():
    c = _contract()
    d = F.FjDecoder(c)
    prev = np.linspace(-1, 1, 26)
    o = F.build_obs(c, {"src": _meas("src"), "rcv": _meas("rcv", 1.0)}, d, prev)
    assert o.size == 165 == F.obs_dim_of(c)
    s = c.sides["src"]
    hand_obs = o[14:20]
    want = [dict(zip(s.hand_joints, range(6)))[n] * 0.01 for n in s.hand_obs_order]
    assert np.allclose(hand_obs, want)                                            # ★PhysX 순
    assert np.allclose(o[20:23], [0.3, 0.0, 0.4]) and np.allclose(o[23:29], [1, 0, 0, 0, 1, 0])   # palm · R 열0+열1
    assert np.allclose(o[29:32], [0.08, -0.16, -0.14])                            # cup − palm
    err = o[47:53]
    assert np.allclose(err, np.clip((np.array(s.hand_open) - 0.01 * np.arange(6)) / 1.2, -1, 1))  # 프로필 순
    assert np.allclose(o[53:56], [0, 0, 1]) and np.allclose(o[56:63], 0.0)        # cup_up · arm q*(home)
    assert np.allclose(o[126:129], 0.0)                                           # rcv 컵 − src 컵
    assert np.allclose(o[129:134], np.tanh(np.clip([0, 3.0, 0, 0, 20.0], 0, 10) / 3.0))
    assert np.allclose(o[139:165], np.clip(prev, -5, 5))


def test_optional_segments_change_the_dimension():
    c = _contract(obs_tips_rel_palm=True, obs_mouth_diff=True, obs_dim=165 + 30 + 3)
    F.validate(c)
    o = F.build_obs(c, {"src": _meas("src"), "rcv": _meas("rcv")}, F.FjDecoder(c), np.zeros(26))
    assert o.size == 198


def test_contract_round_trip_and_measured_obs_order(tmp_path):
    c = _contract()
    p = tmp_path / "pour_fj_contract.json"
    p.write_text(c.to_json())
    back = F.load_contract(p)
    assert back.sides["rcv"].hand_obs_order == c.sides["rcv"].hand_obs_order
    order = {r: list(c.sides[r].hand_joints) for r in F.ROLES}
    m = F.with_obs_order(back, order, "measured:trace_meta.json")
    assert m.hand_obs_order_source.startswith("measured") and m.sides["src"].hand_obs_order == order["src"]
    with pytest.raises(F.PourFjError):
        F.with_obs_order(back, {"src": order["src"][:5], "rcv": order["rcv"]}, "x")


# ---------------------------------------------------------------- 실제 런(f01 아카이브)
@pytest.mark.skipif(not F01_CKPT.exists(), reason="f01 아카이브 없음")
def test_building_from_the_f01_run_uses_its_era_behaviour_and_the_hdgp_homes():
    import yaml
    pair = load_mimic_pair(_paths.RL_WS / "hdgp")
    c = F.build(F01, F01_CKPT, pair, URDF, asset="openarm_rh56f1_bi_rl")
    assert (c.obs_dim, c.action_dim, c.hold_steps, round(c.policy_hz)) == (165, 26, 30, 60)
    assert c.arm_mode == "increment" and c.hand_range == "limits" and not c.hand_freeze   # f01 = 기능 이전 런
    assert c.notes and "arm_joint_mode" in c.notes[0]
    homes = yaml.safe_load(HOMES.read_text())
    assert c.sides["src"].arm_home == pytest.approx(homes["right"]) and c.sides["rcv"].arm_home == pytest.approx(homes["left"])
    assert c.sides["src"].palm_body == "r_hl_palm_sensor" and c.sides["rcv"].side == "left"
    assert c.hand_obs_order_source.startswith("assumed")
    json.loads(c.to_json())


def test_hand_speed_cap_is_common_with_thumb_flex_apart():
    """hdgp side_rig init_real_control: thumb_2 만 hand_thumb_flex_vel_cap_rad_s(0 이면 공통 값)."""
    assert F.hand_vel_cap(_contract()) == ()
    assert F.hand_vel_cap(_contract(hand_vel_cap_rad_s=2.1)) == (2.1,) * 6
    c = _contract(hand_vel_cap_rad_s=2.1, hand_thumb_flex_vel_cap_rad_s=0.56, hand_finger_open_floor_rad=0.065)
    assert F.hand_vel_cap(c) == (2.1, 0.56, 2.1, 2.1, 2.1, 2.1)
    law = F.FjDecoder(c).law(c.sides["src"])
    assert law.finger_open_floor == 0.065 and law.vel_cap_rad_s == F.hand_vel_cap(c)


@pytest.mark.parametrize("RH5", RH5_RUNS, ids=lambda p: p.name)
def test_rh5_trace_one_step_decoder_replay_is_exact(RH5):
    """10.01 pour_bi_rh 세션 결정론 재생(64 env · 900 스텝, f01 · f02 같은 행동 법칙): 관측 안의 q*(팔 마지막 7 · 손 joint_err 복원)를
    직전 q* + 이번 행동으로 한 스텝씩 다시 만든다. 동결은 직전 스텝 손가락 첫마디 OR 손끝 컵 접촉력(link_mid · link_tip) —
    hdgp side_rig `(mid > thr) | (dist > thr)`. f02 왼손은 첫마디 접촉으로 동결되는 스텝이 많다(손끝만 쓰면 1만 스텝 넘게 어긋남)."""
    if not (RH5 / "trace.npz").is_file():
        pytest.skip(f"{RH5.name} trace 없음(git 밖)")
    pair = load_mimic_pair(_paths.RL_WS / "hdgp")
    meta = json.loads((RH5 / "trace_meta.json").read_text())
    order = {r: meta[f"{r}_hand_obs_joint_names"] for r in F.ROLES}
    c = F.build(RH5, next((RH5 / "nn").glob("*.pth")), pair, URDF, asset="openarm_rh56f1_bi_rl",
                hand_obs_order=order, obs_order_source="measured:trace_meta.json")
    assert [n.split("_hj_")[1] for n in order["src"]] == list(F.ASSUMED_OBS_ORDER)      # 가정했던 순서가 맞았다
    assert (c.arm_mode, c.hand_range, c.hand_freeze) == ("absolute", "grip", True)
    assert (c.hand_finger_open_floor_rad, c.hand_vel_cap_rad_s, c.hand_thumb_flex_vel_cap_rad_s) == (0.065, 2.1, 0.56)
    d = np.load(RH5 / "trace.npz")
    O, A = d["obs_next"].astype(float), d["actions"].astype(float)
    assert np.abs(O[:, :, 139:165] - np.clip(A, -1, 1)).max() == 0.0                    # prev_action
    dec = F.FjDecoder(c)
    arm_err, hand_err = 0.0, 0.0
    for ri, r in enumerate(F.ROLES):
        s, off = c.sides[r], 0 if r == "src" else 63
        law, idx = dec.law(s), [list(s.hand_obs_order).index(j) for j in s.hand_joints]
        touch = np.maximum(d[f"{r}_link_mid"], d[f"{r}_link_tip"])            # npz 키 접근은 매번 압축을 푼다 — 한 번만
        for n in (0, 21, 42, 63):
            home = np.array(s.arm_home)
            back = [t for t in range(c.hold_steps + 1, O.shape[0]) if np.abs(O[t, n, off + 56:off + 63] - home).max() < 1e-6]
            for t in range(c.hold_steps + 1, back[0] if back else O.shape[0]):     # 다음 에피소드 리셋 전까지
                q = O[t - 1, n, off + 56:off + 63]
                st = F._SideState(q, np.zeros(6))                                 # f01 · f02 는 가속 한계 없음(prev 무관)
                arm_err = max(arm_err, np.abs(dec._arm(s, st, A[t, n, ri * 13:ri * 13 + 7]) - O[t, n, off + 56:off + 63]).max())
                e0, e1 = O[t - 1, n, off + 47:off + 53], O[t, n, off + 47:off + 53]
                if max(np.abs(e0).max(), np.abs(e1).max()) > 0.999:                   # joint_err 가 잘린 칸은 복원 불가
                    continue
                q0 = O[t - 1, n, off + 14:off + 20][idx] + e0 * c.joint_pos_err_max
                q1 = O[t, n, off + 14:off + 20][idx] + e1 * c.joint_pos_err_max
                fz = law.touch_to_freeze(touch[t - 1, n])
                hand_err = max(hand_err, np.abs(law.step(q0, A[t, n, ri * 13 + 7:(ri + 1) * 13], active=True, freeze=fz) - q1).max())
    assert arm_err < 1e-6 and hand_err < 1e-5, (arm_err, hand_err)


# ---------------------------------------------------------------- 노드의 ROS 없는 절반 · FK
def test_fk_resolves_two_level_thumb_mimic_and_puts_the_palm_where_hdgp_says():
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK
    s = _side("src")
    fk = UrdfChainFK(ASSETS["openarm_rh56f1_bi_rl"].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
    home = [-0.2069, 0.4203, 0.3880, 1.5941, 0.4569, 0.3426, -0.4174]
    p0 = fk.palm_pose(home, [1.57, 0, 0, 0, 0, 0])
    assert np.allclose(p0.palm_pos, [0.310, -0.300, 0.419], atol=2e-3)          # hdgp RH56F1 프로필 주석의 실측
    p1 = fk.palm_pose(home, [1.57, 0.47, 0, 0, 0, 0])
    assert np.linalg.norm(p1.tips[0] - p0.tips[0]) > 0.05                         # thumb_2 → thumb_3 → thumb_4 가 끝을 옮긴다


class _Policy:
    def __init__(self):
        self.calls, self.resets = [], 0

    def forward(self, obs):
        self.calls.append(np.asarray(obs).copy())
        return np.full(26, 0.5)

    def reset(self):
        self.resets += 1


def test_chain_holds_the_arm_for_hold_steps_and_feeds_back_the_clamped_action():
    from policy_control.pour_fj_node import PourFjChain, start_refusals, target_arrays
    c = _contract(hold_steps=2)
    pol = _Policy()
    ch = PourFjChain(c, pol)
    ch.reset()
    meas = {"src": _meas("src"), "rcv": _meas("rcv")}
    for _ in range(2):
        _, _, t = ch.step(meas)
        assert np.allclose(t["src"][0], c.sides["src"].arm_home)                  # 대기
    _, _, t = ch.step(meas)
    assert not np.allclose(t["src"][0], c.sides["src"].arm_home)
    assert np.allclose(pol.calls[1][139:165], 0.5)                                # 직전 행동이 관측 끝 26 칸
    names, q, qd = target_arrays(c, t)
    assert names[:7] == tuple(c.sides["src"].arm_joints) and names[13:20] == tuple(c.sides["rcv"].arm_joints)
    assert q.size == 26 and not qd.any() and pol.resets == 1
    far = dict(meas, rcv=_meas("rcv", 1.0))
    assert any("left arm" in r for r in start_refusals(c, far, 0.15))



def test_hand_obs_order_is_read_from_both_trace_meta_formats():
    """hdgp 6f2ced31(09.30 pour_bi_rh 세션): trace_meta 에 {src,rcv}_hand_obs_joint_names · _hand_action_slot 이 생겼다."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "deploy/policy_control/tools/build_pour_fj_contract.py"
    spec = importlib.util.spec_from_file_location("_build_pour_fj", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = [f"r_hj_{j}" for j in ("index_1", "middle_1", "pinky_1", "ring_1", "thumb_1", "thumb_2")]
    l = [n.replace("r_", "l_", 1) for n in r]
    assert mod.hand_obs_order_from_meta({"hand_joint_names": {"src": r, "rcv": l}}) == {"src": r, "rcv": l}
    new = {"src_hand_obs_joint_names": r, "rcv_hand_obs_joint_names": l,
           "src_hand_action_slot": [0, 1, 2, 3, 4, 5], "rcv_hand_action_slot": [0, 1, 2, 3, 4, 5]}
    assert mod.hand_obs_order_from_meta(new) == {"src": r, "rcv": l}
    assert mod.hand_obs_order_from_meta({}) is None
    with pytest.raises(SystemExit, match="항등"):
        mod.hand_obs_order_from_meta({**new, "src_hand_action_slot": [1, 0, 2, 3, 4, 5]})


# ---------------------------------------------------------------- 10.04 b16 손 닫힘 상한
Q0 = [1.40, 0.10, 0.43, 0.43, 0.40, 0.40]          # 인계 쥔 목표(엄지 회전은 닫을수록 작아진다: grip 1.2 < open 1.57)


def test_b16_close_margin_caps_closing_at_the_handoff_target_plus_margin():
    """hdgp t2r_rh5_b16: 정책이 손을 +1 로 포화해도 닫는 쪽은 q*_0 + 0.02 까지(엄지 회전은 반대 방향), 펴는 쪽은 자유."""
    c = _contract(hand_close_margin_rad=0.02, hand_freeze=False)
    d = F.FjDecoder(c)
    d.reset(hand_start={"src": Q0, "rcv": Q0})
    close = np.ones(26)
    close[7] = close[20] = -1.0                                    # 엄지 회전은 −1 이 닫는 쪽(각도 감소, grip 1.2 < open 1.57)
    for _ in range(300):
        out = d.step(close, active=True)
    h = out["src"][1]
    assert h[0] == pytest.approx(1.38, abs=1e-6)                   # 엄지 회전: 1.40 − 0.02 에서 멈춘다
    assert h[1:] == pytest.approx(np.array(Q0[1:]) + 0.02, abs=1e-6)
    for _ in range(300):
        out = d.step(-close, active=True)
    h = out["src"][1]
    assert h[2] < 0.43 - 0.2 and h[0] > 1.40                      # 펴는 쪽은 q*_0 를 지나 연다


def test_b16_contract_starts_from_the_measured_hand_angles_and_refuses_without_them():
    """sim 뱅크(--targets_from_state)처럼 q*_0 = 인계 순간 손 실측 관절각 — 파지 정책의 마지막 목표가 아니다(T2R Pouring 10.04)."""
    from policy_control import pour_fj_node as N
    c = _contract(hand_close_margin_rad=0.02)
    d = F.FjDecoder(c)
    with pytest.raises(F.PourFjError, match="인계"):
        d.step(np.zeros(26), active=True)
    ch = N.PourFjChain(c, policy=None)
    with pytest.raises(N.PourFjNodeError, match="실측"):
        ch.reset(None)
    meas = {r: _meas(r) for r in F.ROLES}
    measured = {r: [meas[r].hand_q[j] for j in c.sides[r].hand_joints] for r in F.ROLES}
    ch.reset(meas)
    assert ch.dec.state["src"].hand_target == pytest.approx(measured["src"])   # q* 도 실측에서 시작
    assert ch.dec.hand_q0["rcv"] == pytest.approx(measured["rcv"])


def test_old_contracts_still_start_from_the_open_hand_and_are_not_capped():
    d = F.FjDecoder(_contract(hand_freeze=False))
    assert d.state["src"].hand_target == pytest.approx([1.57, 0, 0, 0, 0, 0])
    for _ in range(300):
        out = d.step(np.ones(26), active=True)
    assert out["src"][1][2] == pytest.approx(1.08, abs=1e-3)      # grip 끝까지


def test_close_margin_round_trips_through_the_contract_json(tmp_path):
    c = _contract(hand_close_margin_rad=0.02)
    p = tmp_path / "c.json"
    p.write_text(c.to_json())
    assert F.load_contract(p).hand_close_margin_rad == 0.02


# ---------------------------------------------------------------- 10.04 b16 a=0 자세 · b17 가속 한계
B16_HOME = (-0.0729, 0.2479, -0.0574, 0.8405, 0.2238, 0.1198, 0.7866)       # t2r_rh5_b16 env arm_abs_home_rad


def test_arm_abs_home_is_mirrored_for_the_left_arm_like_hdgp():
    assert F.arm_abs_home_for("right", B16_HOME) == pytest.approx(list(B16_HOME))
    assert F.arm_abs_home_for("left", B16_HOME) == pytest.approx([0.0729, -0.2479, 0.0574, 0.8405, -0.2238, -0.1198, -0.7866])
    assert F.arm_abs_home_for("right", None) == [] and F.arm_abs_home_for("left", ()) == []
    with pytest.raises(F.PourFjError):
        F.arm_abs_home_for("right", (0.1, 0.2))


def test_absolute_action_zero_drives_to_arm_abs_home_not_the_start_pose():
    sides = {r: replace(_side(r), arm_abs_home=[0.3] * 7) for r in F.ROLES}
    d = F.FjDecoder(_contract(arm_mode="absolute", sides=sides))
    for _ in range(600):
        out = d.step(np.zeros(26), active=True)
    assert out["src"][0] == pytest.approx([0.3] * 7, abs=1e-3)         # arm_home 0 이 아니라 a=0 자세 0.3


def _sim_step(q, prev_q, a, home, lo, hi, alpha, vmax, amax, dt):
    """hdgp 92237130 side_rig.step_joint_arm(absolute) + _accel_limit 를 그대로 옮긴 기준."""
    q_raw = np.where(a >= 0, home + a * (hi - home), home + a * (home - lo))
    err = q_raw - q
    step = np.clip(alpha * err, -vmax * dt, vmax * dt)
    if amax > 0:
        prev_step = np.zeros_like(step) if prev_q is None else q - prev_q
        v_stop = np.sqrt(2 * amax * np.abs(err)) * dt
        step = np.maximum(np.minimum(step, v_stop), -v_stop)
        dv = amax * dt * dt
        step = np.maximum(np.minimum(step, prev_step + dv), prev_step - dv)
    return np.clip(q + step, lo, hi)


def test_accel_limit_matches_the_sim_law_step_by_step_and_has_no_velocity_flips():
    c = _contract(arm_mode="absolute", arm_abs_amax=2.0)
    d = F.FjDecoder(c)
    rng = np.random.default_rng(0)
    s = c.sides["src"]
    q, prev = np.array(s.arm_home, float), np.array(s.arm_home, float)
    steps = []
    for t in range(400):
        a = np.clip(rng.normal(0, 1, 26), -1, 1) * (1 if (t // 20) % 2 else -1)   # 20 스텝마다 뒤집는 거친 행동
        ref = _sim_step(q, prev, a[:7], np.array(s.arm_home), np.array(s.arm_lo), np.array(s.arm_hi), 0.1, 0.3, 2.0, 1 / 60)
        out = d.step(a, active=True)
        assert out["src"][0] == pytest.approx(ref, abs=1e-12)
        prev, q = q, ref
        steps.append(q - prev)
    dsteps = np.diff(np.array(steps), axis=0)
    assert np.abs(dsteps).max() <= 2.0 / 3600 + 1e-12                   # |Δstep| ≤ amax·dt²


def test_reset_starts_at_zero_target_velocity_and_handoff_arm_starts_measured():
    from policy_control import pour_fj_node as N
    c = _contract(arm_mode="absolute", arm_abs_amax=2.0, hand_close_margin_rad=0.02)
    ch = N.PourFjChain(c, policy=None)
    meas = {r: _meas(r) for r in F.ROLES}
    meas = {r: replace(m, arm_q=np.full(7, 0.2)) for r, m in meas.items()}
    ch.reset(meas)
    st = ch.dec.state["src"]
    assert st.arm_target == pytest.approx([0.2] * 7) and st.arm_prev_q == pytest.approx([0.2] * 7)   # 실측 · 속도 0
    out = ch.dec.step(np.ones(26), active=True)
    assert np.abs(out["src"][0] - 0.2).max() <= 2.0 / 3600 + 1e-12     # 첫 스텝은 amax·dt² 까지만


def test_bank_start_contract_skips_the_hold_and_starts_from_the_measured_arm():
    """sim start_bank.restore 가 episode_length_buf = hold_steps — 인계 시작은 첫 스텝부터 정책, 팔 q* 는 실측에서(T2R Pouring 10.04)."""
    from policy_control import pour_fj_node as N

    class _P:
        def forward(self, obs):
            return np.ones(26)

        def reset(self):
            pass

    c = _contract(arm_mode="absolute", bank_start=True, hold_steps=30)
    ch = N.PourFjChain(c, policy=_P())
    with pytest.raises(N.PourFjNodeError, match="실측"):
        ch.reset(None)
    meas = {r: replace(_meas(r), arm_q=np.full(7, 0.2), cup_pos=np.array([0.38, -0.16, 0.26])) for r in F.ROLES}
    ch.reset(meas)
    assert ch.step_i == c.hold_steps
    _, _, t = ch.step(meas)
    assert not np.allclose(t["src"][0], c.sides["src"].arm_home)        # hold 였다면 arm_home(0)
    assert np.abs(t["src"][0] - 0.2).max() < 0.05                      # 실측 0.2 에서 정책으로 한 스텝


def test_old_contracts_keep_the_hold_from_home():
    from policy_control import pour_fj_node as N

    class _P:
        def forward(self, obs):
            return np.ones(26)

    c = _contract(arm_mode="absolute", hold_steps=30)
    ch = N.PourFjChain(c, policy=_P())
    ch.reset(None)
    assert ch.step_i == 0


def test_cup_mouth_follows_the_cup_object_not_the_stale_dump():
    """10.04: b16~b18 은 cup_object=cyl60_box32w25 인데 env.yaml 덤프의 cup_mouth_z 는 기본 shaker × 0.65(0.0539) —
    train.py 가 hydra 오버라이드 뒤 · resolve_cfg 재호출 전에 덤프한다. hdgp pour_fabric_mimic CUP_OBJECTS 대로 다시 푼다."""
    dump = {"cup_scale": 0.65, "cup_mouth_z": 0.053885}
    assert F.cup_mouth_z_of(dump) == pytest.approx(0.053885)                              # 키 없는 옛 런 = shaker = 덤프
    assert F.cup_mouth_z_of({**dump, "cup_object": "shaker"}) == pytest.approx(0.053885)
    for name in ("cyl60", "cyl60_box32", "cyl60_box32w25", "cyl60_sdf256"):
        assert F.cup_mouth_z_of({**dump, "cup_object": name}) == pytest.approx(0.085)
    with pytest.raises(F.PourFjError, match="cup_object"):
        F.cup_mouth_z_of({**dump, "cup_object": "mug"})


def test_runs_with_action_law_features_the_decoder_lacks_are_refused(tmp_path):
    """10.04 pour_bi_rh b23(hdgp 733b3365): 팔 행동 저역 필터 · 증분 목표 상자 · 두 손 고정 · 리시버 팔 고정은 배포 디코더에
    아직 없다 — 켜진 런은 계약을 만들지 않는다(조용히 무시하면 실기 목표가 학습과 어긋난다). 끔 값과 키 없음(옛 런)은 통과."""
    base = "pair_name: rh\nhand_control: direct\naction_space: 26\n"
    off = (base + "arm_action_lpf: 1.0\narm_abs_range_rad: !!python/tuple []\nhand_hold: false\nrcv_arm_hold: false\n"
           "hand_force_stop_nm: 0.0\noppose_grip_delta_rad: 0.0\n")
    on = base + "arm_action_lpf: 0.3\narm_abs_range_rad: !!python/tuple\n- 0.6\n- 0.6\nhand_hold: true\nrcv_arm_hold: true\n"
    run = tmp_path / "run" / "params"
    run.mkdir(parents=True)
    for text, want in ((base, []), (off, []),
                       (on, ["arm_abs_range_rad", "arm_action_lpf", "hand_hold", "rcv_arm_hold"])):
        (run / "env.yaml").write_text(text)
        assert [k.split("=")[0] for k in F.unported_keys(F.read_env(run / "env.yaml"))] == want
    with pytest.raises(F.PourFjError, match=r"arm_action_lpf=0\.3"):
        F.build(run.parent, tmp_path / "x.pth", None, tmp_path / "x.urdf", asset="a")

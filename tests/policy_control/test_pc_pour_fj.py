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
                arm_err = max(arm_err, np.abs(dec._arm(s, q, A[t, n, ri * 13:ri * 13 + 7]) - O[t, n, off + 56:off + 63]).max())
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

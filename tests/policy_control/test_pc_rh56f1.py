"""Inspire RH56F1 손 — 변환표 · pd 백엔드 · 상태 노드 · fake 손 · 제어 전용 계약. 09.29 사용자: rh56f1 제어 연결.

여기서 잠그는 것:
  ① 네 손가락은 레지스터가 클수록 펴진다(매뉴얼) — sim 0 rad(편 손) → 1740. 옛 isaacsim 보정(0 rad → 900)의 반대
  ② 슬롯 순서는 드라이버(새끼부터), pd · 정책 순서는 자산(엄지부터) — 변환표 한 곳에서만 바꾼다
  ③ 방향 확인 안 된 축은 -1(움직이지 않음) — 엄지 둘은 arm4090 실측 전까지 pd 가 건드리지 않는다
  ④ 명령은 바뀐 것만 · 최대 command_max_hz — 벤더 노드가 쓰기마다 응답을 기다려 읽기 주기가 밀린다
  ⑤ 변환표 · robot_control 프로필 · 자산 manifest 의 관절 · 한계가 같다
"""
from __future__ import annotations

import math

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
sys.path.insert(0, str(SIM2REAL / "scripts" / "fakes"))

from policy_control import rh56f1_map as M  # noqa: E402
from policy_control.pd_backends import HandCmd, Rh56f1AngleBackend  # noqa: E402
from policy_control.rh56f1_state_node import HandStateCore  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HMAP = M.load()
PROFILE = RL_WS / "robot_control/src/robot_control/profiles/openarm_rh56f1.yaml"
MANIFEST = RL_WS / "hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl_manifest.yaml"
CONTRACT = SIM2REAL / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
OPEN = [1.57, 0.0, 0.0, 0.0, 0.0, 0.0]          # hdgp RH56F1 hand_open_pose (joint_order)


def test_fingers_open_at_the_high_register_like_the_manual():
    reg = HMAP.to_register([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    slots = {n: HMAP.axes[HMAP.joint_order.index(n)].slot for n in HMAP.joint_order}
    for f in ("index_1", "middle_1", "ring_1", "pinky_1"):
        assert reg[slots[f]] == 1740, f                                     # 편 손
    bent = HMAP.to_register([0.0, 0.0, 1.5285594, 1.5285594, 1.5285594, 1.5285594])
    assert all(bent[slots[f]] == 900 for f in ("index_1", "middle_1", "ring_1", "pinky_1"))


def test_the_driver_slot_order_is_pinky_first():
    assert [a.name for a in sorted(HMAP.axes, key=lambda a: a.slot)] == \
        ["pinky_1", "ring_1", "middle_1", "index_1", "thumb_2", "thumb_1"]
    reg = HMAP.to_register([0.0, 0.0, 0.0, 0.5, 1.0, 1.5])                  # 검지 0 · 중지 0.5 · 약지 1.0 · 새끼 1.5
    assert reg[:4] == [HMAP.axes[5].to_reg(1.5), HMAP.axes[4].to_reg(1.0), HMAP.axes[3].to_reg(0.5), 1740]


def test_all_axes_are_verified_after_the_0930_probes_but_an_unverified_axis_is_left_alone():
    """09.30 양손 probe 뒤 전부 확인됨. 확인 안 된 축은 -1(LEAVE) — 변환표를 고쳐 흉내 낸다."""
    assert HMAP.unverified() == [] and all(r != M.LEAVE for r in HMAP.to_register(OPEN))
    raw = yaml.safe_load((REPO / "deploy/policy_control/config/rh56f1_hand_map.yaml").read_text())
    raw["joints"]["thumb_1"]["verified"] = False
    raw["joints"]["thumb_2"]["verified"] = ["right"]
    m = M.parse(raw)
    assert m.to_register(OPEN)[5] == M.LEAVE and m.to_register(OPEN)[4] == M.LEAVE
    assert m.to_register(OPEN, side="right")[4] != M.LEAVE and m.to_register(OPEN, side="left")[4] == M.LEAVE
    assert all(r != M.LEAVE for r in m.to_register(OPEN, allow_unverified=True))


def test_rad_and_register_round_trip_and_clamp():
    q = np.array([1.0, 0.2, 0.3, 0.7, 1.1, 1.4])
    back = HMAP.to_rad(HMAP.to_register(q, allow_unverified=True))
    assert np.allclose(back, q, atol=2e-3)                                  # 0.1° 양자화
    assert HMAP.to_register([9, 9, 9, 9, 9, 9], allow_unverified=True)[3] == 900     # 한계 밖은 끝점으로
    with pytest.raises(M.HandMapError):
        HMAP.to_register([0.0] * 5)
    with pytest.raises(M.HandMapError):
        HMAP.to_register([np.nan] * 6)


def test_mimic_joints_follow_the_asset_urdf():
    full = HMAP.with_mimic([0.0, 0.4, 1.0, 1.0, 1.0, 1.0])
    assert full["thumb_3"] == pytest.approx(0.4 * 1.1425) and full["thumb_4"] == pytest.approx(0.4 * 1.1425 * 0.7508)
    assert full["index_2"] == pytest.approx(1.1169)


def test_touch_is_reordered_thumb_first():
    assert np.allclose(HMAP.touch_sim_order([10, 20, 30, 40, 50]), [0.5, 0.4, 0.3, 0.2, 0.1])  # 벤더 새끼 → 엄지


def test_map_matches_the_robot_control_profile_and_the_asset():
    prof = {j["canonical"]: j for j in yaml.safe_load(PROFILE.read_text())["joints"]}
    order = [j for j in yaml.safe_load(MANIFEST.read_text())["control_joint_order"] if j.startswith("r_hj_")]
    assert order == HMAP.names("right")                                     # pd · 정책의 손 순서 = 자산 순서
    for a in HMAP.axes:
        j = prof[f"r_hj_{a.name}"]
        assert a.rad == pytest.approx((j["lower"], j["upper"]), abs=1e-4), a.name


def test_the_old_isaacsim_calibration_direction_is_not_what_we_use():
    """옛 robot/isaacsim_bridge 보정은 0 rad → reg 900(굽힘) — 그대로 쓰면 편 손 명령이 주먹이 된다."""
    old = yaml.safe_load((SIM2REAL / "robot/isaacsim_bridge/config/rh56f1_hand_calibration.yaml").read_text())
    idx = next(j for j in old["right"]["joints"] if j["suffix"] == "index_1")
    assert idx["reg_lo"] == 900 and HMAP.axes[HMAP.joint_order.index("index_1")].reg[0] == 1740


# ---------------------------------------------------------------- pd 백엔드
class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class _Node:
    def __init__(self):
        self.pubs = {}

    def create_publisher(self, _t, topic, _q):
        return self.pubs.setdefault(topic, _Pub())


def _backend(execute=False, clock=None):
    names = HMAP.names("right")
    lo = [0.0] * 6
    hi = [2.0943951, 0.474555, 1.5285594, 1.5285594, 1.5285594, 1.5285594]
    return Rh56f1AngleBackend(_Node(), "/hand_right/angle_set", names, lo, hi, HMAP, 1.0, execute=execute, clock=clock)


def test_backend_ramps_in_rad_and_turns_the_result_into_registers():
    b = _backend()
    w = b.write(HandCmd(q_star=np.array([1.57, 0, 1.0, 1.0, 1.0, 1.0]), qd_star=None, dt=0.01,
                        q_meas=np.array(OPEN)))
    assert w.limited and np.allclose(w.q_cmd[2:], 0.01)                      # 1 rad/s × 10 ms
    assert b.last_register[3] == HMAP.axes_of("right")[2].to_reg(0.01) and b.last_register[4] != M.LEAVE   # 오른손 보정 · 엄지 확인됨
    assert b.publish_count == 0                                               # execute=False → 발행 없음


def test_backend_refuses_a_hand_order_that_is_not_the_map_order():
    with pytest.raises(ValueError, match="순서"):
        Rh56f1AngleBackend(_Node(), "/t", list(reversed(HMAP.names("right"))), [0] * 6, [1] * 6, HMAP, 1.0,
                           execute=False)


def test_backend_sends_only_changes_and_at_most_command_max_hz():
    pytest.importorskip("rh56f1_interfaces.msg", reason="robot_control 설치 공간(rh56f1_interfaces)이 source 되지 않았다")
    t = [0.0]
    b = _backend(execute=True, clock=lambda: t[0])
    pub = b._pub._pub
    cmd = lambda q: HandCmd(q_star=np.array(q), qd_star=None, dt=0.01, q_meas=np.array(OPEN))  # noqa: E731
    b.write(cmd(OPEN))
    assert len(pub.msgs) == 1 and pub.msgs[0].hand_id == 0 and -1 not in list(pub.msgs[0].joint_values)   # 오른손 엄지 확인됨
    b.write(cmd(OPEN))
    assert len(pub.msgs) == 1                                                # 같은 레지스터 — 안 보낸다
    b.write(cmd([1.57, 0, 1, 1, 1, 1]))
    assert len(pub.msgs) == 1                                                # 바뀌었지만 1/30 s 전 — 안 보낸다
    t[0] = 0.05
    b.write(cmd([1.57, 0, 1, 1, 1, 1]))
    assert len(pub.msgs) == 2
    last = list(pub.msgs[-1].joint_values)
    t[0] = 0.5
    b.write(cmd([1.57, 0, 1, 1, 1, 1]))
    if list(b.last_register) == last:
        assert len(pub.msgs) == 2                                            # 같은 값 · 1 s 안 — 안 보낸다
    t[0] = 1.2
    b.write(cmd([1.57, 0, 1, 1, 1, 1]))
    assert len(pub.msgs) >= 3                                                # 1 s 마다 다시(연결 직후 유실 대비)


def test_backend_sets_the_hand_speed_and_force_before_the_first_angle_and_then_every_few_seconds():
    """09.30 사용자: 손 액션 구조 · 튜닝 — 손 자체 속도(speed_set) · 힘 멈춤(force_set)을 pd yaml 에서 준다. 0 은 보내지 않는다."""
    pytest.importorskip("rh56f1_interfaces.msg", reason="robot_control 설치 공간(rh56f1_interfaces)이 source 되지 않았다")
    from policy_control.pd_backends import RH56F1_HW_RESEND_S
    t = [0.0]
    node = _Node()
    names = HMAP.names("right")
    b = Rh56f1AngleBackend(node, "/hand_right/angle_set", names, [0.0] * 6, [2.1, 0.48, 1.53, 1.53, 1.53, 1.53], HMAP, 1.0,
                           execute=True, clock=lambda: t[0], hw_speed=2000)
    assert "/hand_right/speed_set" in node.pubs and "/hand_right/force_set" not in node.pubs       # force 0(인자 기본) = 안 보낸다
    cmd = HandCmd(q_star=np.array(OPEN), qd_star=None, dt=0.01, q_meas=np.array(OPEN))
    b.write(cmd)
    speed = node.pubs["/hand_right/speed_set"].msgs
    assert len(speed) == 1 and list(speed[0].joint_values) == [2000] * 6
    t[0] = RH56F1_HW_RESEND_S + 0.1
    b.write(cmd)
    assert len(speed) == 2
    with pytest.raises(ValueError, match="speed"):
        Rh56f1AngleBackend(_Node(), "/t", names, [0] * 6, [1] * 6, HMAP, 1.0, execute=False, hw_speed=5000)


def test_pd_yaml_hand_settings_are_loaded():
    from policy_control.pd_law import load_pd_config
    for name in ("pd_rh56f1.yaml", "pd_rh56f1_exec.yaml"):
        h = load_pd_config(REPO / "deploy/policy_control/config" / name).hand
        assert h.hw_speed == 2000 and h.hw_force == 600


# ---------------------------------------------------------------- 상태 노드 · fake 손
def test_state_core_publishes_rad_with_mimic_and_velocity():
    core = HandStateCore(HMAP, "left")
    q0 = [1.2, 0.2, 0.5, 0.5, 0.5, 0.5]                                      # 보정 범위 안(편 손 0 은 명령 끝점 1740 에서 잘린다)
    reg0 = HMAP.to_register(q0, side="left")
    names, pos, vel = core.on_angle(reg0, 0.0)
    assert names[:6] == HMAP.names("left") and "l_hj_thumb_3" in names and len(names) == 12
    assert np.allclose(pos[:6], q0, atol=3e-3) and not vel.any()
    reg1 = HMAP.to_register([1.2, 0.2, 0.7, 0.5, 0.5, 0.5], side="left")
    _, pos1, vel1 = core.on_angle(reg1, 0.1)
    assert vel1[2] == pytest.approx((pos1[2] - pos[2]) / 0.1)                # 첫 차분은 그대로
    _, pos2, vel2 = core.on_angle(reg1, 0.2)                                  # 멈춤 → 시정수 EMA 로 줄어든다
    a = 1.0 - math.exp(-0.1 / core.tau_s)
    assert vel2[2] == pytest.approx((1.0 - a) * vel1[2])


def test_state_velocity_filter_does_not_depend_on_the_publish_rate():
    """10.03: 상태 발행이 50 → 200 Hz 로 바뀌어도 같은 램프에서 같은 속도 추정(시정수 필터)."""
    def run(hz):
        core = HandStateCore(HMAP, "left")
        v = None
        for k in range(int(0.5 * hz) + 1):
            t = k / hz
            q = [1.2, 0.2, 0.5 + 0.4 * t, 0.5, 0.5, 0.5]                     # 0.4 rad/s 램프
            _, _, v = core.on_angle(HMAP.to_register(q, side="left"), t)
        return v[2]
    assert run(50) == pytest.approx(run(200), rel=0.1) and run(200) == pytest.approx(0.4, rel=0.15)


def test_fake_hand_moves_a_full_stroke_in_a_second_and_keeps_minus_one_axes():
    import fake_rh56f1_hand as F
    cur = np.array([1740.0] * 4 + [1350.0, 1750.0])
    span = np.array([840.0] * 4 + [250.0, 1150.0])
    target = F.apply_command(cur.copy(), [900, -1, -1, -1, -1, -1])
    assert target[0] == 900 and target[1] == 1740
    for _ in range(50):
        cur = F.step(cur, target, span, 0.02)
    assert cur[0] == pytest.approx(900.0)


# ---------------------------------------------------------------- 제어 전용 계약
def test_control_contract_has_no_fabric_and_opens_the_hand():
    if not CONTRACT.exists():
        pytest.skip("계약 없음 — tools/build_deploy_contract.py --asset openarm_rh56f1_bi_rl --home arms:config/homes/rh56f1_pour_fj.yaml")
    from policy_control import contract as C
    c = C.load_contract(CONTRACT)
    assert c.fabric is None and c.control_only and c.asset.ee_kind == "rh56f1"
    for side in ("right", "left"):
        s = c.side(side)
        assert s.palm_body == f"{side[0]}_hl_palm_sensor" and s.fabric is None
        assert list(s.hand_joints) == HMAP.names(side)
        assert s.home_hand[f"{side[0]}_hj_thumb_1"] == 1.57 and s.pd_groups == [f"{side}_arm", f"{side}_hand"]
    raw = json.loads(CONTRACT.read_text())
    homes = yaml.safe_load((SIM2REAL / "deploy/policy_control/config/homes/rh56f1_aglt.yaml").read_text())
    assert raw["sides"]["right"]["home_arm"] == pytest.approx(homes["right"])        # 홈 = rh_aglt 시작 자세(09.29)
    assert raw["sides"]["left"]["home_arm"] == pytest.approx(homes["left"])


def test_a_fabric_less_contract_is_only_valid_when_control_only():
    if not CONTRACT.exists():
        pytest.skip("계약 없음")
    from policy_control import contract as C
    raw = json.loads(CONTRACT.read_text())
    raw["control_only"] = False
    with pytest.raises(C.ContractError, match="control"):
        C.validate(C.from_dict(raw))


def test_robot_yamls_use_the_rh56f1_backend_and_state_topics():
    from policy_control import sources as S
    for side in ("right", "left"):
        for kind in ("real", "fake"):
            cfg = S.load_robot_cfg(SIM2REAL / f"deploy/policy_control/config/robots/rh56f1_{side}_{kind}.yaml")
            hand = cfg.groups[f"{side}_hand"]
            assert hand["backend"] == "rh56f1_angle" and hand["topic"] == f"/hand_{side}/angle_set"
            assert cfg.sources["ee"].topic == f"/hand_{side}/joint_states"
            assert list(cfg.sources["ee"].joints) == HMAP.names(side)


def test_verified_list_names_only_hands():
    with pytest.raises(M.HandMapError, match="right"):
        raw = yaml.safe_load((REPO / "deploy/policy_control/config/rh56f1_hand_map.yaml").read_text())
        raw["joints"]["thumb_1"]["verified"] = ["up"]
        M.parse(raw)


@pytest.mark.parametrize("side", ["right", "left"])
def test_calibrated_map_reproduces_the_0930_sweep_within_a_degree(side):
    """09.30 레지스터 스윕(posAct → 벤더 표 → 관절각)을 손 · 축별 보정이 1° 안으로 재현한다. 기본 끝점 변환은 최대 6.5° 틀렸다."""
    import csv
    path = REPO / "logs/rh56f1_probe_0930/sweep_joint_rad.npz"
    if not path.is_file():
        pytest.skip("스윕 원자료 없음(git 밖) — logs/rh56f1_probe_0930")
    d = np.load(path)
    names = {"index": "index_1", "middle": "middle_1", "ring": "ring_1", "pinky": "pinky_1",
             "thumb_bend": "thumb_2", "thumb_rot": "thumb_1"}
    worst_cal, worst_default = 0.0, 0.0
    for key, joint in names.items():
        reg, q = d[f"{side}_{key}_reg"], d[f"{side}_{key}_rad"]
        cal = HMAP.axes_of(side)[HMAP.joint_order.index(joint)]
        base = HMAP.axes[HMAP.joint_order.index(joint)]
        worst_cal = max(worst_cal, max(abs(cal.to_rad(r) - v) for r, v in zip(reg, q)))
        worst_default = max(worst_default, max(abs(base.to_rad(r) - v) for r, v in zip(reg, q)))
    assert np.degrees(worst_cal) < 1.0 < np.degrees(worst_default)


def test_calibrated_commands_stay_in_the_vendor_register_range():
    for side in ("right", "left"):
        reg = HMAP.to_register([0.0] * 6, side=side)                        # 관절 0 = 보정 영점(1760~1818) — 명령은 끝점까지
        lim = {a.slot: (min(a.reg), max(a.reg)) for a in HMAP.axes}
        assert all(lim[i][0] <= r <= lim[i][1] for i, r in enumerate(reg))
        back = HMAP.to_rad(HMAP.to_register([0.8, 0.2, 0.7, 0.7, 0.7, 0.7], side=side), side)
        assert back == pytest.approx([0.8, 0.2, 0.7, 0.7, 0.7, 0.7], abs=0.003)


def test_tactile_is_the_contact_force_magnitude_like_the_training_sensor():
    """09.30 누름: 접선이 법선의 0.4~1.0배 — 학습 촉각(손끝 링크 합력 크기)과 같게 √(법선² + 접선²). 3000 포화."""
    n = [300, 0, 0, 400, 3000]                                          # 벤더 순 새끼 → 엄지
    t = [400, 0, 0, 300, 9999]
    got = HMAP.touch_sim_order(n, t)                                    # sim 순 엄지 → 새끼
    assert got == pytest.approx([np.hypot(3000, 3000) * 0.01, 5.0, 0.0, 0.0, 5.0])
    assert HMAP.touch_sim_order(n) == pytest.approx([30.0, 4.0, 0.0, 0.0, 3.0])     # 접선이 없으면 법선만
    core = HandStateCore(HMAP, "right")
    assert core.tip_forces(n, t)[1] == pytest.approx(5.0)


def test_engage_seed_just_outside_a_joint_limit_is_clipped_in_and_far_outside_is_refused():
    """10.03 실기: 왼팔 차렷 j4 −0.8° 가 하한 0 밖 → engage 시작점이 '한계 밖 목표'로 즉시 HOLD 였다."""
    from policy_control.pd_arm import PdArmError, SEED_CLIP_TOL_RAD, clip_seed
    lo, hi = np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.4, 1.0])
    names = ["l_aj_3", "l_aj_4", "l_aj_5"]
    seed, notes = clip_seed(np.array([0.2, -0.014, 0.0]), lo, hi, names)
    assert seed[1] == 0.0 and seed[0] == 0.2 and len(notes) == 1 and "l_aj_4" in notes[0]
    same, none = clip_seed(np.array([0.2, 0.3, 0.0]), lo, hi, names)
    assert np.allclose(same, [0.2, 0.3, 0.0]) and none == []
    with pytest.raises(PdArmError, match="l_aj_4"):
        clip_seed(np.array([0.2, -(SEED_CLIP_TOL_RAD + 0.01), 0.0]), lo, hi, names)


def test_joint_forces_are_republished_thumb_first():
    """10.04 force_actual(슬롯 새끼 · 약지 · 중지 · 검지 · 엄지 굽힘 · 엄지 회전) → joint_forces(엄지 굽힘 · 엄지 회전 · 검지 · 중지 · 약지 · 새끼)."""
    from policy_control.rh56f1_state_node import joint_forces_out
    assert joint_forces_out([10, 20, 30, 40, 50, 60]) == [50.0, 60.0, 40.0, 30.0, 20.0, 10.0]
    with pytest.raises(ValueError):
        joint_forces_out([1, 2, 3])


# ---------------------------------------------------------------- ★10.08 손 명령 입력(위치 · 어드민턴스)
def test_backend_sends_admittance_on_angle_target_and_switches_input_at_once():
    """정책 계약 hand_command admittance → /hand_<s>/angle_target(마스터의 손가락별 어드민턴스). 입력을 바꾸면 같은 값 ·
    1/30 s 안이라도 곧바로 보낸다 — 마스터는 축마다 마지막으로 받은 입력을 따른다."""
    pytest.importorskip("rh56f1_interfaces.msg", reason="robot_control 설치 공간(rh56f1_interfaces)이 source 되지 않았다")
    t = [0.0]
    node = _Node()
    names = HMAP.names("right")
    b = Rh56f1AngleBackend(node, "/hand_right/angle_set", names, [0.0] * 6, [2.1, 0.48, 1.53, 1.53, 1.53, 1.53], HMAP, 1.0,
                           execute=True, clock=lambda: t[0])
    pos, adm = node.pubs["/hand_right/angle_set"].msgs, node.pubs["/hand_right/angle_target"].msgs
    cmd = lambda mode: HandCmd(q_star=np.array(OPEN), qd_star=None, dt=0.01, q_meas=np.array(OPEN), mode=mode)  # noqa: E731
    b.write(cmd("admittance"))
    assert (len(pos), len(adm)) == (0, 1) and b.mode == "admittance" and b.admittance_count == 1
    b.write(cmd("admittance"))
    assert len(adm) == 1                                                      # 같은 값 — 안 보낸다
    b.write(cmd("position"))
    assert (len(pos), len(adm)) == (1, 1) and b.mode == "position"                # 입력이 바뀌면 곧바로
    assert b.publish_count == 2
    with pytest.raises(ValueError, match="손 명령 입력"):
        b.write(cmd("force"))


def test_backend_without_execute_publishes_on_neither_input():
    b = _backend()
    b.write(HandCmd(q_star=np.array(OPEN), qd_star=None, dt=0.01, q_meas=np.array(OPEN), mode="admittance"))
    assert b.publish_count == 0 and b.admittance_count == 0 and b.mode == "admittance"


def test_fake_hand_takes_both_command_inputs():
    import fake_rh56f1_hand as F
    assert F.COMMAND_TOPICS == ("angle_set", "angle_target")


def test_leaving_admittance_restarts_the_position_command_from_the_measured_hand():
    """10.08 리뷰: 어드민턴스로 쥔 손가락은 목표보다 앞(접촉)에 막혀 있다 — 위치로 바꾸는 순간 마지막 목표를 그대로 보내면 힘 제한 없이
    짓누른다. 위치로 바뀌면 실측 각에서 속도 상한으로 다시 출발한다."""
    pytest.importorskip("rh56f1_interfaces.msg", reason="robot_control 설치 공간(rh56f1_interfaces)이 source 되지 않았다")
    node = _Node()
    names = HMAP.names("right")
    b = Rh56f1AngleBackend(node, "/hand_right/angle_set", names, [0.0] * 6, [2.1, 0.48, 1.53, 1.53, 1.53, 1.53], HMAP, 1.0,
                           execute=True, clock=lambda: 0.0)
    closed = np.array(OPEN) + np.array([0, 0, 1.2, 1.2, 1.2, 1.2])
    blocked = np.array(OPEN) + np.array([0, 0, 0.6, 0.6, 0.6, 0.6])         # 컵에 막힌 실측
    for _ in range(400):                                                    # 어드민턴스로 쥔 목표까지 다 갔다
        b.write(HandCmd(q_star=closed, qd_star=None, dt=0.01, q_meas=blocked, mode="admittance"))
    w = b.write(HandCmd(q_star=closed, qd_star=None, dt=0.01, q_meas=blocked, mode="position"))
    assert np.max(np.abs(w.q_cmd - blocked)) <= 1.0 * 0.01 + 1e-9           # 실측 + 한 틱(max_vel 1 rad/s · 10 ms)


# ---------------------------------------------------------------- ★11.04 손끝 소프트웨어 영점
def _fill(z, t_end, normal, tang=(0,) * 5, q=(1.57, 0, 0, 0, 0, 0), hz=200, span=0.6):
    for k in range(int(span * hz)):
        t = t_end - span + k / hz
        z.add_touch(t, normal, tang)
        z.add_angle(t, q)


def test_tip_zero_takes_the_window_mean_and_subtracts_it_from_raw_without_touching_tip_forces():
    """11.04 사용자: home · rehome 에서 bias 를 잡아 영점. tip_forces 원값(정책 입력)은 그대로, zeroed 는 법선 · 접선에서 각각 뺀다."""
    from policy_control.rh56f1_state_node import TipZero
    z = TipZero(HMAP)
    _fill(z, 10.0, [10, 0, 0, 40, 30], [0, 0, 0, 30, 0])
    ok, why, info = z.capture(10.0)
    assert ok, why
    assert info["normal_bias_raw"] == pytest.approx([10, 0, 0, 40, 30]) and info["tangential_bias_raw"] == pytest.approx([0, 0, 0, 30, 0])
    assert z.bias_n() == pytest.approx([0.3, 0.5, 0.0, 0.0, 0.1])                       # sim 순 엄지 → 새끼, N
    n, t = [110, 0, 0, 40, 330], [0, 0, 0, 430, 0]
    assert z.zeroed(n, t) == pytest.approx([3.0, 4.0, 0.0, 0.0, 1.0])
    assert z.zeroed([0] * 5, [0] * 5) == pytest.approx([0.0] * 5)                       # 음수로 내려가지 않는다
    assert HandStateCore(HMAP, "right").tip_forces(n, t)[0] == pytest.approx(3.3)        # 원값은 그대로


@pytest.mark.parametrize("case, kw, needle", [
    ("bent", dict(q=(1.57, 0, 0.6, 0, 0, 0)), "펴지지 않았다"),
    ("touching", dict(normal=[0, 0, 0, 0, 900]), "원시값이 크다"),
])
def test_tip_zero_refuses_a_closed_hand_or_a_touching_tip_and_keeps_the_old_bias(case, kw, needle):
    from policy_control.rh56f1_state_node import TipZero
    z = TipZero(HMAP)
    _fill(z, 1.0, [5, 5, 5, 5, 5])
    assert z.capture(1.0)[0]
    args = dict(normal=[5] * 5) | kw
    _fill(z, 2.0, **args)
    ok, why, info = z.capture(2.0)
    assert not ok and any(needle in w for w in why), why
    assert info["normal_bias_raw"] == pytest.approx([5] * 5)


def test_tip_zero_refuses_a_moving_hand_noisy_values_or_no_data():
    from policy_control.rh56f1_state_node import ZERO_MOVING, TipZero
    z = TipZero(HMAP)
    assert not z.capture(0.0)[0]                                                         # 아무것도 안 왔다
    for k in range(120):                                                                 # 손가락이 펴지는 중
        z.add_touch(k / 200, [0] * 5, [0] * 5)
        z.add_angle(k / 200, (1.57, 0, 0.2 - k / 1000, 0, 0, 0))
    ok, why, _ = z.capture(119 / 200)
    assert not ok and any(ZERO_MOVING in w for w in why)
    z = TipZero(HMAP)
    for k in range(120):
        z.add_touch(k / 200, [0, 0, 0, 0, 100 * (k % 2)], [0] * 5)
        z.add_angle(k / 200, (1.57, 0, 0, 0, 0, 0))
    ok, why, _ = z.capture(119 / 200)
    assert not ok and any("튄다" in w for w in why)
    _fill(z, 5.0, [0] * 5)
    assert not z.capture(5.5)[0]                                                         # 0.5 s 넘게 끊겼다


# ---------------------------------------------------------------- ★11.04 관절 힘 소프트웨어 영점
def _fill_jf(z, t_end, jf, q=(1.57, 0, 0, 0, 0, 0), hz=200, span=0.6):
    _fill(z, t_end, [0] * 5, q=q, hz=hz, span=span)
    for k in range(int(span * hz)):
        z.add_joint_force(t_end - span + k / hz, jf)


def test_joint_force_bias_is_not_published_before_the_first_capture_and_a_jump_is_taken_with_a_warning():
    """11.04 실기: 오른손 엄지 굽힘이 편 손에서 +207 g 로 떠 엄지 받침 B(80 g)가 늘 참. 영점 전에는 zeroed 를 내지 않고(None),
    직전 bias 에서 30 g 넘게 옮겨 가도 받되 joint_jump_g 에 남긴다. 손끝 영점과 따로 받아들인다."""
    from policy_control.rh56f1_state_node import TipZero
    z = TipZero(HMAP)
    assert z.joint_zeroed([207, 0, 45, 0, 0, 0]) is None                                         # 영점 전 = 내지 않는다
    _fill_jf(z, 1.0, [207, 0, 45, 0, 0, 0])
    ok, _, info = z.capture(1.0)
    assert ok and info["joint_accepted"] and info["joint_jump_g"] == {} and info["joint_bias_g"] == pytest.approx([207, 0, 45, 0, 0, 0])
    assert z.joint_zeroed([290, 5, 45, 0, 0, 0]) == pytest.approx([83, 5, 0, 0, 0, 0])
    assert z.joint_zeroed([150, 0, 0, 0, 0, 0])[0] == pytest.approx(-57)                         # 부호 그대로(자르지 않는다)
    _fill_jf(z, 2.0, [237, 0, 75.5, 0, 0, 0])                                                     # 엄지 +30(경계, 경고 아님) · 검지 +30.5
    _, _, info = z.capture(2.0)
    assert info["joint_accepted"] and info["joint_jump_g"] == {"index": 30.5}
    assert info["joint_bias_g"] == pytest.approx([237, 0, 75.5, 0, 0, 0])
    z2 = TipZero(HMAP)
    _fill(z2, 1.0, [0] * 5)                                                                       # 관절 힘이 안 왔다 — 손끝만 받는다
    ok, _, info = z2.capture(1.0)
    assert ok and not info["joint_accepted"] and not z2.joint_bias_set and z2.joint_zeroed([0] * 6) is None


def test_joint_force_bias_refuses_noise_and_a_large_mean_and_force_only_skips_the_mean_limit():
    from policy_control.rh56f1_state_node import JOINT_ZERO_MAX_ABS_G, JOINT_ZERO_MAX_STD_G, TipZero
    z = TipZero(HMAP)
    _fill_jf(z, 1.0, [JOINT_ZERO_MAX_ABS_G, -JOINT_ZERO_MAX_ABS_G, 0, 0, 0, 0])                    # 경계 = 받는다
    assert z.capture(1.0)[2]["joint_accepted"]
    _fill_jf(z, 2.0, [0, 0, 0, 0, 0, -JOINT_ZERO_MAX_ABS_G - 1])
    _, _, info = z.capture(2.0)
    assert not info["joint_accepted"] and any("pinky" in w and "tip_zero_force" in w for w in info["joint_reasons"])
    assert info["joint_bias_g"][0] == pytest.approx(JOINT_ZERO_MAX_ABS_G)                         # 이전 bias 유지
    _, _, info = z.capture(2.0, force_joint=True)                                                  # 강제 = 절대 상한만 넘는다
    assert info["joint_accepted"] and info["joint_forced"] and info["joint_bias_g"][5] == pytest.approx(-JOINT_ZERO_MAX_ABS_G - 1)
    _fill_jf(z, 3.0, [0] * 6, q=(1.57, 0, 0.6, 0, 0, 0))                                          # 강제여도 편 손 조건은 그대로
    _, _, info = z.capture(3.0, force_joint=True)
    assert not info["joint_accepted"] and any("펴지지 않았다" in w for w in info["joint_reasons"])

    def alt(sd):                                                                                   # 검지가 ±sd 로 번갈아 = 표준편차 sd
        z = TipZero(HMAP)
        _fill(z, 1.0, [0] * 5)
        for k in range(120):
            z.add_joint_force(1.0 - 0.6 + k / 200, [0, 0, 100 + sd * (1 if k % 2 else -1), 0, 0, 0])
        return z.capture(1.0, force_joint=True)[2]
    assert alt(JOINT_ZERO_MAX_STD_G)["joint_accepted"]                                            # 경계 = 받는다
    info = alt(JOINT_ZERO_MAX_STD_G + 0.1)
    assert not info["joint_accepted"] and any("튄다" in w and "index" in w for w in info["joint_reasons"])  # 강제여도 sd 는 본다


def test_trigger_warns_when_the_joint_zero_was_refused():
    sys.path.insert(0, str(REPO / "deploy/policy_control/tools"))
    import trigger as T
    assert T.joint_zero_warning(json.dumps({"ok": True, "reasons": [], "joint_ok": False, "joint_reasons": ["관절 힘이 튄다"]}))
    assert T.joint_zero_warning(json.dumps({"ok": True, "reasons": [], "joint_ok": True})) is None
    assert T.joint_zero_warning(json.dumps({"ok": True, "reasons": []})) is None and T.joint_zero_warning("x") is None


@pytest.mark.parametrize("name", ["rh56f1_right_real", "rh56f1_left_real", "rh56f1_right_fake", "rh56f1_left_fake",
                                  "rh56f1_bi_real", "rh56f1_bi_fake"])
def test_grasp_reads_the_zeroed_joint_forces_and_a_missing_topic_is_none_not_the_raw(name):
    """11.04: 정책 노드의 joint_force(쥠 판정 · 엄지 받침 B · 놓기 is_free)는 영점 뺀 토픽. 안 오면 원값으로 돌아가지 않고 None
    (B = 받침 아님) — 선택 소스라 reset 을 막지도 않는다."""
    from policy_control import sources
    cfg = sources.load_robot_cfg(REPO / f"deploy/policy_control/config/robots/{name}.yaml")
    jf = {s.side or name.split("_")[1]: s for s in cfg.sources.values() if s.role == "joint_force"}
    assert jf and all(s.topic == f"/hand_{side}/joint_forces_zeroed" and not s.required for side, s in jf.items())
    for side in jf:
        st = sources.SourceSet(sources.select_side(cfg, side) if "_bi_" in name else cfg).snapshot(0.0)
        assert st.joint_force is None and not any("joint_force" in m for m in st.missing)


def test_a_stale_joint_force_becomes_none_but_other_optional_sources_keep_the_last_value():
    """11.04: stale_to_none 은 joint_force 만 켰다 — stale_sec(0.5 s) 넘게 끊기면 None, 경계 안이면 값. tip_force 는 예전대로 마지막 값."""
    from policy_control import sources
    cfg = sources.load_robot_cfg(REPO / "deploy/policy_control/config/robots/rh56f1_right_real.yaml")
    assert cfg.sources["joint_force"].stale_to_none and not cfg.sources["tip_force"].stale_to_none
    ss = sources.SourceSet(cfg)
    ss._latest["joint_force"] = sources._Reading(value=np.zeros((6, 1)), t_recv=10.0, stamp=10.0)
    ss._latest["tip_force"] = sources._Reading(value=np.zeros((5, 1)), t_recv=10.0, stamp=10.0)
    assert ss.snapshot(10.5).joint_force is not None
    st = ss.snapshot(10.51)
    assert st.joint_force is None and st.tip_force is not None

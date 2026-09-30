"""Inspire RH56F1 손 — 변환표 · pd 백엔드 · 상태 노드 · fake 손 · 제어 전용 계약. 09.29 사용자: rh56f1 제어 연결.

여기서 잠그는 것:
  ① 네 손가락은 레지스터가 클수록 펴진다(매뉴얼) — sim 0 rad(편 손) → 1740. 옛 isaacsim 보정(0 rad → 900)의 반대
  ② 슬롯 순서는 드라이버(새끼부터), pd · 정책 순서는 자산(엄지부터) — 변환표 한 곳에서만 바꾼다
  ③ 방향 확인 안 된 축은 -1(움직이지 않음) — 엄지 둘은 arm4090 실측 전까지 pd 가 건드리지 않는다
  ④ 명령은 바뀐 것만 · 최대 command_max_hz — 벤더 노드가 쓰기마다 응답을 기다려 읽기 주기가 밀린다
  ⑤ 변환표 · robot_control 프로필 · 자산 manifest 의 관절 · 한계가 같다
"""
from __future__ import annotations

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


def test_unverified_thumb_axes_are_left_alone():
    assert set(HMAP.unverified()) == {"thumb_1", "thumb_2"}
    reg = HMAP.to_register(OPEN)
    assert reg[4] == M.LEAVE and reg[5] == M.LEAVE
    assert all(r != M.LEAVE for r in HMAP.to_register(OPEN, allow_unverified=True))


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
    assert b.last_register[3] == HMAP.axes[2].to_reg(0.01) and b.last_register[4] != M.LEAVE   # 오른손 엄지 확인됨(09.30)
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
    assert "/hand_right/speed_set" in node.pubs and "/hand_right/force_set" not in node.pubs       # force 0 = 안 보낸다
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
        assert h.hw_speed == 2000 and h.hw_force == 0


# ---------------------------------------------------------------- 상태 노드 · fake 손
def test_state_core_publishes_rad_with_mimic_and_velocity():
    core = HandStateCore(HMAP, "left")
    reg0 = HMAP.to_register(OPEN, allow_unverified=True)
    names, pos, vel = core.on_angle(reg0, 0.0)
    assert names[:6] == HMAP.names("left") and "l_hj_thumb_3" in names and len(names) == 12
    assert np.allclose(pos[:6], OPEN, atol=2e-3) and not vel.any()
    reg1 = HMAP.to_register([1.57, 0, 0.2, 0, 0, 0], allow_unverified=True)
    _, pos1, vel1 = core.on_angle(reg1, 0.1)
    assert vel1[2] == pytest.approx((pos1[2] - pos[2]) / 0.1)                # 첫 차분은 그대로
    _, pos2, vel2 = core.on_angle(reg1, 0.2)                                  # 멈춤 → EMA α 0.3 로 줄어든다
    assert vel2[2] == pytest.approx(0.7 * vel1[2])


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


def test_thumbs_are_verified_on_the_right_hand_only_until_the_left_is_checked():
    """09.30 사용자: 오른손 엄지 두 축 방향이 맞다 — 변환표 verified: [right]. 왼손 엄지는 아직 -1."""
    q = [1.2, 0.2, 0.5, 0.5, 0.5, 0.5]
    right, left = HMAP.to_register(q, side="right"), HMAP.to_register(q, side="left")
    assert right[4] != M.LEAVE and right[5] != M.LEAVE and left[4] == M.LEAVE and left[5] == M.LEAVE
    assert HMAP.unverified("right") == [] and HMAP.unverified("left") == ["thumb_1", "thumb_2"]
    b = _backend()
    b.write(HandCmd(q_star=np.array(OPEN), qd_star=None, dt=0.01, q_meas=np.array(OPEN)))
    assert M.LEAVE not in b.last_register                                  # 오른손 백엔드는 엄지도 보낸다
    with pytest.raises(M.HandMapError, match="right"):
        raw = yaml.safe_load((REPO / "deploy/policy_control/config/rh56f1_hand_map.yaml").read_text())
        raw["joints"]["thumb_1"]["verified"] = ["up"]
        M.parse(raw)

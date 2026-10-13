"""10.13 place IK 이송(carry) — hdgp hdgp_place_r1 c3009166 rh_place_env._carry_actions 와 같은 식(배포 노이즈 0)."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control import place_carry as PC
from policy_control import rh_aglt as A
from policy_control import rh_place as P

from test_pc_rh_place_rg3 import HOLDER, _contract, _meas  # noqa: E402

CFG = {"dz": 0.085, "tol": 0.01, "min_steps": 20, "max_steps": 240, "speed": 0.0015, "gain": 0.1,
       "rot_gain": 0.1, "rot_max": 0.01, "damp": 0.01}
ENV = {"carry_enable": True, **{v: CFG[k] for k, v in PC.ENV_KEYS.items()}}


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _toy_pose(q):
    """관절 0~2 = 손바닥 xyz(직선), 3 = z 회전, 4 = x 회전, 5·6 = 영향 없음."""
    q = np.asarray(q, float)
    return q[:3].copy(), _rz(q[3]) @ _rx(q[4])


# ---------------------------------------------------------------- 계약 키
def test_env_keys_must_all_be_dumped_and_disabled_means_no_carry():
    assert PC.CarryCfg.from_env({"carry_enable": False}) is None
    assert PC.CarryCfg.from_env(ENV).as_dict() == CFG
    bad = dict(ENV)
    bad.pop("carry_dls_damp")
    with pytest.raises(PC.CarryError, match="carry_dls_damp"):
        PC.CarryCfg.from_env(bad)
    with pytest.raises(PC.CarryError):
        PC.CarryCfg.from_dict({**CFG, "tol": 0.0})


def test_target_is_holder_xy_at_table_plus_origin_plus_dz():
    t = PC.carry_target([0.3, -0.13, 0.10], origin_z=0.085, dz=0.085)
    assert t == pytest.approx([0.3, -0.13, 0.10 - 0.030 + 0.085 + 0.085])     # 홀더 원점 − 0.030 = 상판


# ---------------------------------------------------------------- 수치
def test_numeric_jacobian_matches_the_toy_arm():
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    J = PC.palm_jacobian(_toy_pose, q)
    want = np.zeros((6, 7))
    want[:3, :3] = np.eye(3)
    want[5, 3] = 1.0                                                  # z 회전
    want[3:, 4] = _rz(0.4) @ np.array([1.0, 0.0, 0.0])                # x 회전 축(세계 틀)
    assert J == pytest.approx(want, abs=1e-6)


def test_dls_step_matches_the_sim_formula():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    J, e = rng.normal(size=(6, 7)), rng.normal(size=6) * 0.01
    JJt = torch.tensor(J) @ torch.tensor(J).T                          # hdgp invert_probe.dls_step 그대로(env 1 개)
    x = torch.linalg.solve(JJt + (0.01 ** 2) * torch.eye(6, dtype=JJt.dtype), torch.tensor(e).unsqueeze(-1))
    want = (torch.tensor(J).T @ x).squeeze(-1).clamp(-1.0, 1.0).numpy()
    assert PC.dls_step(J, e, 0.01, 1.0) == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_tilt_twist_rights_the_cup_and_leaves_yaw_free():
    up_x = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])            # x 로 0.2 rad 기운 컵
    w = PC.tilt_twist(up_x, A.quat_apply, 0.1, 0.01)
    assert w[0] == pytest.approx(-0.01) and abs(w[1]) < 1e-12 and w[2] == 0.0   # 상한 0.01, 세우는 쪽
    yawed = np.array([np.cos(0.5), 0.0, 0.0, np.sin(0.5)])
    assert PC.tilt_twist(yawed, A.quat_apply, 0.1, 0.01) == pytest.approx(np.zeros(3))


# ---------------------------------------------------------------- 상태 기계
def _run_toy(cfg: PC.CarryCfg, cup_off=np.array([0.0, 0.0, -0.05]), target=np.array([0.30, -0.10, 0.25]), steps=400):
    q = np.array([0.30, -0.25, 0.20, 0.0, 0.0, 0.0, 0.0])
    k_arm, ema = 0.05, 0.5
    c = PC.Carry(cfg, target, np.full(6, 0.3), k_arm, ema)
    log = []
    while c.on and len(log) < steps:
        p, R = _toy_pose(q)
        cup = p + cup_off
        a = c.step(p, cup, np.array([1.0, 0, 0, 0]), PC.palm_jacobian(_toy_pose, q), A.quat_apply)
        log.append(a)
        q = q + a[:7] * k_arm * ema                                    # 팔 디코더 한 틱(EMA 정상상태)
    return c, log, q


def test_carry_moves_the_perceived_cup_onto_the_target_then_hands_over():
    cfg = PC.CarryCfg.from_dict(CFG)
    c, log, q = _run_toy(cfg)
    assert not c.on and c.arrived and cfg.min_steps <= c.t < cfg.max_steps
    assert np.linalg.norm(_toy_pose(q)[0] + [0.0, 0.0, -0.05] - c.target) < 0.0125
    assert all(np.all(a[7:] == 0.3) for a in log)                      # 손 = 리셋 손 목표 그대로
    assert max(np.abs(a[:3]).max() for a in log) * 0.05 * 0.5 <= cfg.speed + 1e-12   # 축당 속도 상한


def test_carry_stops_at_max_steps_even_if_far():
    c, log, _ = _run_toy(PC.CarryCfg.from_dict({**CFG, "max_steps": 30}), target=np.array([0.5, 0.3, 0.5]))
    assert not c.on and not c.arrived and c.t == 30 and len(log) == 30


def test_min_steps_hold_the_carry_even_when_already_on_target():
    cfg = PC.CarryCfg.from_dict(CFG)
    c, log, _ = _run_toy(cfg, target=np.array([0.30, -0.25, 0.15]))   # 시작부터 목표
    assert c.arrived and c.t == cfg.min_steps


def test_dp_is_frozen_at_the_first_tick_in_the_base_frame():
    c = PC.Carry(PC.CarryCfg.from_dict(CFG), np.zeros(3), np.zeros(6), 0.05, 0.5)
    J = np.hstack([np.vstack([np.eye(3), np.zeros((3, 3))]), np.zeros((6, 4))])
    c.step(np.array([0.1, 0.0, 0.0]), np.zeros(3), np.array([1.0, 0, 0, 0]), J, A.quat_apply)
    c.step(np.array([0.1, 0.0, 0.0]), np.array([0.05, 0.0, 0.0]), np.array([1.0, 0, 0, 0]), J, A.quat_apply)
    assert c.dp == pytest.approx([0.1, 0.0, 0.0])


# ---------------------------------------------------------------- PlaceChain
class _CountPol:
    def __init__(self, a):
        self.a, self.n = np.asarray(a, float), 0

    def forward(self, obs):
        self.n += 1
        return self.a


class _FK:
    """palm = 팔 관절 0~2, R = I(손바닥이 컵 위 2 cm) — 장난감 팔."""
    def palm_pose(self, q, hand_q):
        p, R = _toy_pose(q)
        return SimpleNamespace(palm_pos=p, extra={"palm_rot": R})


def _chain(**kw):
    c = _contract(carry=dict(CFG), **kw)
    pol = _CountPol(np.full(13, 0.7))
    ch = P.PlaceChain(c, pol)
    ch.object_name, ch.fk = "source240_real", _FK()
    return c, ch, pol


def _toy_meas(c, q, hand):
    m = _meas(c, hand)
    p, R = _toy_pose(q)
    return replace(m, arm_q=np.asarray(q, float), palm_pos=p, palm_R=R)


def test_contract_refuses_bad_carry_and_needs_fk():
    with pytest.raises(P.RhPlaceError):
        _contract(carry={**CFG, "speed": -1.0})
    c = _contract(carry=dict(CFG))
    ch = P.PlaceChain(c, _CountPol(np.zeros(13)))
    ch.object_name = "source240_real"
    hand = np.asarray(c.side().hand_grip, float) * 0.5
    with pytest.raises(P.RhPlaceError, match="FK"):
        ch.reset({"arm": _meas(c, hand)}, holder=HOLDER, held=(np.asarray(c.side().arm_home), hand))


def test_chain_runs_the_policy_every_tick_but_executes_the_carry_first():
    c, ch, pol = _chain()
    hand = np.asarray(c.side().hand_grip, float) * 0.5
    q = np.asarray(c.side().arm_home, float).copy()
    q[:3] = [0.30, -0.25, 0.40]
    ch.reset({"arm": _toy_meas(c, q, hand)}, holder=HOLDER, held=(q.copy(), hand))
    assert ch.carry is not None and ch.as_dict()["phase"] == "carry"
    want_t = PC.carry_target(HOLDER, 0.085, 0.085)
    assert ch.carry.target == pytest.approx(want_t)
    executed = []
    for _ in range(ch.c.carry["max_steps"] + 5):
        _, a, targets = ch.step({"arm": _toy_meas(c, q, hand)})
        executed.append(a)
        assert ch.free_streak == 0 or not ch.carry.on
        assert ch.prev == pytest.approx(np.clip(a, -1, 1))             # prev_action = 실행한 행동
        q = np.asarray(targets["arm"][0], float)                       # 장난감 팔은 목표를 그대로 따른다
        if not ch.carry.on:
            break
    n = ch.carry.t
    assert pol.n == n                                                  # 이송 틱에도 forward
    assert all(not np.allclose(a, 0.7) for a in executed[:n])          # 실행은 이송 행동
    assert executed[0][7:] == pytest.approx(P.hand_action_of(c, hand))  # 손 = 인계 손 목표
    _, a, _ = ch.step({"arm": _toy_meas(c, q, hand)})
    assert a == pytest.approx(np.full(13, 0.7)) and pol.n == n + 1     # 다음 틱부터 정책


def test_seat_goal_uses_the_selected_objects_bottom():
    c = _contract()
    a = A.select_object(c, "source240_real")
    b = A.select_object(c, "shaker_c_real")
    assert P.seat_goal(a, HOLDER).pos[2] == pytest.approx(HOLDER[2] + P.HOLDER_FLOOR_Z + 0.085)
    assert P.seat_goal(b, HOLDER).pos[2] == pytest.approx(HOLDER[2] + P.HOLDER_FLOOR_Z + 0.065)   # sim seat_pos(bottom −0.065)
    assert P.origin_z_of(b) == pytest.approx(0.065)


# ---------------------------------------------------------------- 빌더
def _build_with(tmp_path, extra: dict):
    import shutil
    from test_pc_rh_place import RUNS, URDF, _need
    run = RUNS["right"]
    _need(run)
    pth = next((run / "nn").glob("*.pth"), None)
    if pth is None:
        pytest.skip("right_i09 체크포인트 없음")
    dst = tmp_path / run.name
    shutil.copytree(run / "params", dst / "params")
    env_p = dst / "params" / "env.yaml"
    env_p.write_text(env_p.read_text() + "".join(f"{k}: {v}\n" for k, v in extra.items()))
    from policy_control.pour_profiles import load_profile
    from policy_control import _paths
    return P.build(dst, pth, load_profile(_paths.RL_WS / "hdgp", "rh56f1_right"), URDF, asset="openarm_rh56f1_bi_rl")


def test_the_builder_copies_every_carry_key_from_the_dump(tmp_path):
    c = _build_with(tmp_path, {k: str(v).lower() for k, v in ENV.items()})
    assert c.carry == CFG and any("IK 이송" in n for n in c.notes)


def test_the_builder_refuses_a_carry_run_with_a_missing_key(tmp_path):
    with pytest.raises(P.RhPlaceError, match="carry_dls_damp"):
        _build_with(tmp_path, {k: str(v).lower() for k, v in ENV.items() if k != "carry_dls_damp"})

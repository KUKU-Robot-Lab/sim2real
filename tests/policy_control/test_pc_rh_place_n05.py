"""★10.22 newton_05 place 계약(latch_s2r · 래치 GRU · 시간 기준 놓음 · 정책 빼기 · 넘김 묶음) — hdgp rh_place_r 16f25c219 와 같은 식."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from policy_control import place_handover as PH
from policy_control import place_latch_est as LE
from policy_control import rh_aglt as A
from policy_control import rh_place as P

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy/policies/rh56f1/place"
HDGP = REPO.parent / "hdgp"
CUP_EST = "source/openarm/openarm/agnostic/tasks/rh_place_r/cup_est.py"


def _contract(side="right") -> P.PlaceContract:
    return P.load_contract(POL / f"{side}_n05/rh_place_contract.json")


def _est_file(tmp_path: Path, d_in: int, seed: int = 0) -> Path:
    import torch
    torch.manual_seed(seed)
    h = 16
    sd = {"inp.0.weight": torch.randn(h, d_in) * 0.2, "inp.0.bias": torch.randn(h) * 0.1,
          "rnn.weight_ih_l0": torch.randn(3 * h, h) * 0.2, "rnn.weight_hh_l0": torch.randn(3 * h, h) * 0.2,
          "rnn.bias_ih_l0": torch.randn(3 * h) * 0.1, "rnn.bias_hh_l0": torch.randn(3 * h) * 0.1,
          "out.weight": torch.randn(6, h) * 0.5, "out.bias": torch.randn(6) * 0.5}
    p = tmp_path / f"est_{seed}.pt"
    torch.save({"sd": sd, "mu": torch.randn(d_in) * 0.1, "std": torch.rand(d_in) + 0.5}, p)
    return p


def test_the_built_contracts_are_101_with_latch_and_six_zero_neighbour_cells():
    for side, y in (("right", [-0.223, -0.037]), ("left", [0.037, 0.223])):
        c = _contract(side)
        assert c.obs_dim == 101 and A.obs_dim_of(c) == 101
        assert list(c.extra_actor_obs) == ["latch", *P.NB_OBS] and c.neighbor_obs == "zero"
        assert c.latch_s2r and c.latch_debounce == 3 and c.latch_dz_objs == {"shaker_c_real": 0.07}
        assert c.release_timed_steps == 15 and c.latch_arm_free and c.withdraw == {"clear_m": 0.05, "max_steps": 120, "hand_open": True}
        assert c.attach_source == "bundle" and c.holder_range == {"x": [0.375, 0.38], "y": y}
        assert c.episode_s == 5.0 and c.policy_hz == 60.0


def test_s2r_contracts_refuse_missing_pieces():
    c = _contract()
    with pytest.raises(P.RhPlaceError, match="latch_s2r"):
        P.validate(replace(c, release_timed_steps=0))
    with pytest.raises(P.RhPlaceError, match="extra_actor_obs"):
        P.validate(replace(c, neighbor_obs=""))                     # 이웃 칸이 있는 계약에서 neighbor_obs 를 빼면 칸이 안 맞는다
    with pytest.raises(P.RhPlaceError, match="latch_dz_objs"):
        P.validate(replace(c, latch_dz_objs={"cyl60": 0.07}))
    with pytest.raises(P.RhPlaceError, match="attach_source"):
        P.validate(replace(c, attach_source="magic"))


def test_the_latch_height_is_per_object():
    c = _contract()
    assert P.latch_dz_of(A.select_object(c, "shaker_c_real")) == 0.07
    assert P.latch_dz_of(A.select_object(c, "source240_real")) == 0.04


def test_holder_range_check_reports_the_excess_and_refuses_beyond_tolerance():
    c = _contract()
    bad, ex = P.holder_refusals(c, [0.3911, -0.0609, 0.235], 0.02)         # 10.20 스캔 홀더 1 — x 1.1 cm 밖
    assert not bad and ex["x"] == pytest.approx(0.0111, abs=1e-4) and ex["y"] == 0.0
    bad, _ = P.holder_refusals(c, [0.3911, -0.0609, 0.235], 0.005)
    assert bad and "x" in bad[0]
    bad, _ = P.holder_refusals(c, [0.378, -0.327, 0.235], 0.02)          # 홀더 2 — 오른손 범위 밖
    assert bad and "y" in bad[0]


def test_latch_est_matches_the_hdgp_cup_est(tmp_path):
    """같은 ckpt · 같은 관측 열 → hdgp CupEst(origin/main 사본)와 같은 보정 · 자세."""
    import torch
    try:
        src = subprocess.run(["git", "-C", str(HDGP), "show", f"origin/main:{CUP_EST}"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("hdgp 저장소 없음")

    def qmul(a, b):
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)

    def qapply(q, v):
        xyz = q[..., 1:]
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)

    ns: dict = {"quat_apply": qapply, "quat_mul": qmul, "__name__": "cup_est_ref"}
    src = src.replace("from openarm.agnostic.modules.simcompat.mathx import quat_apply, quat_mul", "")
    exec(compile(src, CUP_EST, "exec"), ns)
    d_in = 101
    f = _est_file(tmp_path, d_in)
    ref = ns["CupEst"](str(f), 1, "cpu")
    mine = LE.LatchEst(f, LE.md5(f), d_in)
    rng = np.random.default_rng(1)
    att_p, att_q = np.array([0.01, -0.03, 0.05]), np.array([0.9, 0.1, -0.3, 0.2])
    att_q = att_q / np.linalg.norm(att_q)
    for k in range(6):
        o = rng.normal(size=d_in) * 3.0
        y = mine.step(o)
        ref.step(torch.tensor(o[None], dtype=torch.float32), torch.tensor([k > 0]))
        assert np.allclose(y, ref.y[0].numpy(), atol=1e-5)
        p, q = mine.rel_pose(att_p, att_q)
        rp, rq = ref.rel_pose(torch.tensor(att_p[None], dtype=torch.float32), torch.tensor(att_q[None], dtype=torch.float32))
        assert np.allclose(p, rp[0].numpy(), atol=1e-6) and np.allclose(q, rq[0].numpy(), atol=1e-6)
    assert k > 0 and not np.allclose(mine.y, 0.0)                     # 첫 틱 뒤에는 보정이 생긴다
    with pytest.raises(LE.LatchEstError, match="md5"):
        LE.LatchEst(f, "0" * 32, d_in)
    with pytest.raises(LE.LatchEstError, match="입력"):
        LE.LatchEst(f, LE.md5(f), 97)


class _Pol:
    def __init__(self, a):
        self.a = np.asarray(a, float)
        self.n = 0

    def forward(self, obs):
        self.n += 1
        return self.a.copy()


class _Est:
    """보정 0 추정기 — 체인 순서 시험용(입력 폭 · 래치 칸만 본다)."""

    def __init__(self):
        self.seen = []
        self.y = np.zeros(6)

    def reset(self):
        self.seen = []

    def step(self, obs):
        self.seen.append(np.asarray(obs, float).copy())
        return self.y

    def rel_pose(self, p, q):
        return LE.rel_pose(p, q, self.y)

    def as_dict(self):
        return {}


HOLDER = np.array([0.378, -0.10, 0.235])


def _meas(c, hand, cup_z, cup_xy=(0.378, -0.10), palm_dz=0.02):
    s = c.side()
    palm = np.array([cup_xy[0], cup_xy[1], cup_z + palm_dz])
    return A.RaMeas(arm_q=np.asarray(s.arm_home), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, hand)),
                    palm_pos=palm, palm_R=np.eye(3), tips=np.tile(palm, (5, 1)),
                    cup_pos=np.zeros(3), cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5), joint_force=None)


def _chain(a=None):
    c = _contract()
    ch = P.PlaceChain(c, _Pol(np.full(13, 0.3) if a is None else a))
    ch.object_name = "source240_real"
    ch.est = _Est()
    return c, ch


def test_reset_needs_the_bundle_and_puts_the_cup_on_the_palm():
    c, ch = _chain()
    hand = np.asarray(c.side().hand_grip, float) * 0.5
    held = (np.asarray(c.side().arm_home), hand)
    m = _meas(c, hand, 0.40)
    with pytest.raises(P.RhPlaceError, match="묶음"):
        ch.reset({"arm": m}, holder=HOLDER, held=held)
    ch.reset({"arm": m}, holder=HOLDER, held=held, attach=(np.array([0, 0, -0.02]), np.array([1.0, 0, 0, 0])))
    pos, _ = ch.cup_from_palm(m.palm_pos, m.palm_R)
    assert np.allclose(pos, m.palm_pos + [0, 0, -0.02])
    c2, ch2 = _chain()
    ch2.est = None
    with pytest.raises(P.RhPlaceError, match="추정기"):
        ch2.reset({"arm": m}, holder=HOLDER, held=held, attach=(np.zeros(3), np.array([1.0, 0, 0, 0])))


def test_s2r_latch_release_withdraw_follow_the_sim_step_order():
    c, ch = _chain()
    s = c.side()
    hand = np.asarray(s.hand_grip, float) * 0.5
    held = (np.asarray(s.arm_home), hand)
    seat_z = HOLDER[2] + P.seat_dz_of(A.select_object(c, "source240_real"))
    att = (np.array([0.0, 0.0, -0.02]), np.array([1.0, 0, 0, 0]))
    ch.reset({"arm": _meas(c, hand, seat_z + 0.10)}, holder=HOLDER, held=held, attach=att)
    a_open = P.open_hand_action(ch.c)
    n_arm = len(s.arm_joints)
    obs, a, _ = ch.step({"arm": _meas(c, hand, seat_z + 0.10)})
    assert obs.size == 101 and obs[94] == 0.0 and np.all(obs[95:] == 0.0) and not ch.latch
    assert np.allclose(a[n_arm:], 0.3)                               # 래치 전 = 정책 그대로
    for k in range(3):                                               # 자리 위 3 cm(dz 0.04 안) — 3 틱 연속이어야 켜진다
        obs, a, _ = ch.step({"arm": _meas(c, hand, seat_z + 0.03)})
        assert ch.latch == (k == 2) and obs[94] == (1.0 if k == 2 else 0.0)
    assert np.allclose(a[n_arm:], a_open) and np.allclose(a[:n_arm], 0.3)   # 래치 틱부터 손 = 편 손, 팔 = 정책
    assert ch.est.seen[-1][94] == 0.0                               # 추정기 입력의 래치 칸 = 갱신 전 값
    for k in range(14):                                              # 래치 뒤 15 틱째에 놓음
        ch.step({"arm": _meas(c, hand, seat_z + 0.20)})
        assert ch.latch and not ch.settling and ch.latch_t == k + 1
    ch.step({"arm": _meas(c, hand, seat_z + 0.20)})
    assert ch.settling and ch.settle_t == 0 and ch.latch_t == 15 and ch.released_cup is not None
    _, a, _ = ch.step({"arm": _meas(c, hand, seat_z + 0.20, palm_dz=-0.02)})   # 손이 컵 안쪽 — 아직 붙어 있다
    assert np.allclose(a[n_arm:], a_open) and not ch.done and ch.clearance is not None and ch.clearance < 0.05
    rel = np.asarray(ch.released_cup[0])
    ch.step({"arm": _meas(c, hand, seat_z + 0.20, cup_xy=(rel[0] + 0.20, rel[1]))})   # 컵 축에서 20 cm — 빠졌다
    assert ch.done and ch.outcome == "clear"


def test_withdraw_ends_on_its_step_limit():
    c, ch = _chain()
    s = c.side()
    hand = np.asarray(s.hand_grip, float) * 0.5
    seat_z = HOLDER[2] + P.seat_dz_of(A.select_object(c, "source240_real"))
    ch.reset({"arm": _meas(c, hand, seat_z + 0.03)}, holder=HOLDER, held=(np.asarray(s.arm_home), hand),
             attach=(np.array([0.0, 0.0, -0.02]), np.array([1.0, 0, 0, 0])))
    k = 0
    while not ch.done and k < 400:
        ch.step({"arm": _meas(c, hand, seat_z + 0.03, palm_dz=-0.02)})   # 손이 컵에서 떨어지지 않는다
        k += 1
    assert ch.outcome == "withdraw_timeout" and ch.settle_t == 120
    assert k == 3 + 15 + 120 + 1 - 1                                 # 래치 3 틱(3 번째 틱에 켜짐) · 놓음 15 · 빼기 120


def test_the_latch_never_turns_on_once_released_and_stays_on():
    c, ch = _chain()
    s = c.side()
    hand = np.asarray(s.hand_grip, float) * 0.5
    seat_z = HOLDER[2] + P.seat_dz_of(A.select_object(c, "source240_real"))
    ch.reset({"arm": _meas(c, hand, seat_z + 0.03)}, holder=HOLDER, held=(np.asarray(s.arm_home), hand),
             attach=(np.array([0.0, 0.0, -0.02]), np.array([1.0, 0, 0, 0])))
    for _ in range(3):
        ch.step({"arm": _meas(c, hand, seat_z + 0.03)})
    assert ch.latch
    obs, _, _ = ch.step({"arm": _meas(c, hand, seat_z + 0.30, cup_xy=(0.0, 0.0))})
    assert ch.latch and obs[94] == 1.0                               # 리셋까지 유지


# ---------------------------------------------------------------- 넘김 묶음

def _bundle(path: Path, *, side="r", obj="source240_real", valid=True, hand=(1.3, 0.2, 1.0, 1.0, 0.5, 0.5), created=None):
    meta = {"format": PH.FORMAT, "side": side, "quat_order": "wxyz", "frame": "palm",
            "created": (created or __import__("datetime").datetime.now().astimezone()).isoformat(timespec="seconds"),
            "object_contract": obj, "node": f"rh_aglt_node_{'right' if side == 'r' else 'left'}", "episode": 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, objects=np.asarray(["source240_cvx"]), meta_json=np.asarray(json.dumps(meta)),
                        rel_p=np.array([[-0.04, 0.015, 0.018]]), rel_q=np.array([[0.67, 0.48, 0.38, 0.42]]) / 1.0001,
                        obj_axis_palm=np.zeros((1, 3)), grasp_height=np.array([0.04]), hand_q=np.array([hand]),
                        n_steps=np.array([30]), valid=np.array([valid]))
    return path


def test_bundle_pick_takes_the_newest_fresh_matching_one(tmp_path):
    import datetime as dt
    day = tmp_path / dt.date.today().strftime("%Y%m%d")
    hand = np.array([1.3, 0.2, 1.0, 1.0, 0.5, 0.5])
    old = _bundle(day / "100000_rh_aglt_node_right_1.npz")
    os.utime(old, (time.time() - 50, time.time() - 50))
    new = _bundle(day / "100100_rh_aglt_node_right_2.npz")
    b = PH.pick("right", object_name="source240_real", hand_q=hand, hand_tol_rad=0.05, max_age_s=900, root=tmp_path)
    assert b.path == new and b.valid and abs(np.linalg.norm(b.rel_q) - 1.0) < 1e-9
    with pytest.raises(PH.HandoverError, match="이미 썼다"):          # 쓴 묶음 · 그보다 옛 묶음으로 내려가지 않는다
        PH.pick("right", object_name="source240_real", hand_q=hand, hand_tol_rad=0.05, max_age_s=900, used=[str(new)], root=tmp_path)
    with pytest.raises(PH.HandoverError, match="손 실측"):
        PH.pick("right", object_name="source240_real", hand_q=hand + 0.2, hand_tol_rad=0.05, max_age_s=900, root=tmp_path)
    with pytest.raises(PH.HandoverError, match="물체"):
        PH.pick("right", object_name="shaker_c_real", hand_q=hand, hand_tol_rad=0.05, max_age_s=900, root=tmp_path)
    with pytest.raises(PH.HandoverError, match="없다"):
        PH.pick("left", object_name="source240_real", hand_q=hand, hand_tol_rad=0.05, max_age_s=900, root=tmp_path)


def test_bundle_check_refuses_stale_invalid_and_other_side(tmp_path):
    import datetime as dt
    hand = np.array([1.3, 0.2, 1.0, 1.0, 0.5, 0.5])
    b = PH.load(_bundle(tmp_path / "a.npz", valid=False, created=dt.datetime.now().astimezone() - dt.timedelta(hours=1)))
    bad = PH.check(b, side="left", object_name="source240_real", hand_q=hand, hand_tol_rad=0.05, max_age_s=900)
    assert any("valid" in x for x in bad) and any("전 것" in x for x in bad) and any("팔" in x for x in bad)


def test_real_fake_bundle_reads_if_present():
    files = PH.candidates("right")
    if not files:
        pytest.skip("logs/inhand 묶음 없음")
    b = PH.load(files[0])
    assert b.side == "r" and b.rel_p.shape == (3,) and b.hand_q.shape == (6,)

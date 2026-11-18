"""11.18 rh_cap 깊이 학생 depth_h01 정책 코어(cap_student) — 관측 칸 · 머리 각 · 끼우기 손 잔차 · 메타 검사 · step 상태 · 묶음 검사 · 실제 ckpt.

step 시험은 가짜 mod · model 로 돈다(ckpt 불필요). 실제 ckpt(git 밖) · hdgp 저장소 · torchvision 이 없으면 적재 시험만 건너뛴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control import cap_student as CS
from policy_control import hdgp_pin
from policy_control import rh56f1_hand as RH
from policy_control.rh_cap import CAPTOP_GRIP, CAPTOP_OPEN

REPO = Path(__file__).resolve().parents[2]
H01 = REPO / "deploy" / "policies" / "rh56f1" / "cap" / "right_student_h01"
H01_MD5 = "60e3ebd572060f5cf64d6ad9dd01514a"
META = {"kind": "depth", "task": "rh_cap", "n_obs": 62, "n_act": 15, "head": True, "img_hw": [90, 160], "hid": 512, "n_aux": 6,
        "stages": ["grasp", "carry", "seat", "approach", "search"], "d_min": 0.2, "d_max": 1.5,
        "head_cfg": {"pan_lim": (-45.0, 30.0), "tilt_lim": (-2.0, 82.0)}}


def _obs(**over):
    kw = dict(arm_q=np.arange(7) * 0.1, arm_qd=np.zeros(7), arm_q_target=np.arange(7) * 0.1, hand_q=np.zeros(6),
              hand_target=np.array([0.6, 0.0, 0.0, 0.0, 3.0, -3.0]), palm_pos=[0.3, -0.2, 0.4], palm_R=np.eye(3),
              prev_action=np.full(13, 0.5), stage=4, head_enc_deg=[-45.0, 82.0])
    return CS.build_obs(**{**kw, **over})


def test_obs_layout_stage_and_head_columns():
    o = _obs()
    assert o.shape == (62,) and o.dtype == np.float32 and CS.N_OBS == 62
    np.testing.assert_allclose(o[27:33], [0.5, 0, 0, 0, 1, -1])            # 손 오차 = (목표 − 실측)/1.2, ±1 자름
    np.testing.assert_allclose(o[33:36], [0.3, -0.2, 0.4])
    np.testing.assert_allclose(o[42:55], 0.5)
    np.testing.assert_allclose(o[55:60], [0, 0, 0, 0, 1])                  # search = 끝 칸
    np.testing.assert_allclose(o[60:62], [-1.0, 1.0])                      # pan −45 → −1 · tilt 82 → +1
    np.testing.assert_allclose(_obs(stage=0, head_enc_deg=[-7.5, 40.0])[55:], [1, 0, 0, 0, 0, 0, 0])
    # 회전행렬 열 0 · 열 1(행이 아니다)
    np.testing.assert_allclose(_obs(palm_R=np.arange(9.0).reshape(3, 3))[36:42], [0, 3, 6, 1, 4, 7])


@pytest.mark.parametrize("bad", [
    {"stage": 5}, {"stage": -1}, {"stage": True}, {"stage": 2.0},
    {"arm_qd": np.full(7, np.nan)}, {"arm_q": 0.0}, {"arm_q_target": np.zeros(6)}, {"hand_q": np.zeros(7)},
    {"hand_target": 0.5}, {"hand_target": np.zeros((1, 6))}, {"palm_pos": np.zeros(4)},
    {"palm_R": np.arange(9.0)}, {"palm_R": np.eye(4)}, {"prev_action": np.zeros(15)},
    {"head_enc_deg": [0.0]}, {"head_enc_deg": [0.0, np.nan]},
])
def test_obs_rejects_wrong_shape_type_or_nan(bad):
    with pytest.raises(ValueError):
        _obs(**bad)


def test_head_action_to_deg_bounds():
    np.testing.assert_allclose(CS.head_deg([-1.0, 1.0]), [-45.0, 82.0])
    np.testing.assert_allclose(CS.head_deg([1.0, -1.0]), [30.0, -2.0])
    np.testing.assert_allclose(CS.head_deg([0.0, 0.0]), [-7.5, 40.0])
    np.testing.assert_allclose(CS.head_deg([-3.0, 3.0]), [-45.0, 82.0])    # 밖은 자름
    np.testing.assert_allclose(CS.head_obs(CS.head_deg([0.3, -0.6])), [0.3, -0.6])


def test_head_enc_slightly_over_clips_far_over_rejected():
    np.testing.assert_allclose(CS.head_obs([-49.9, 86.9]), [-1.0, 1.0])    # 한계 밖 5° 이내는 자른다
    np.testing.assert_allclose(CS.head_obs([34.9, -6.9]), [1.0, -1.0])
    for far in ([-50.1, 40.0], [35.1, 40.0], [0.0, 87.1], [0.0, -7.1], [0.0, -250.0]):      # −250 = 11.02 틸트 wrap
        with pytest.raises(ValueError):
            CS.head_obs(far)


def test_seat_hand_residual_matches_hand_law():
    law = RH.HandLaw(q_open=CAPTOP_OPEN, q_grip=CAPTOP_GRIP, lim_lo=(0.0,) * 6, lim_hi=(1.7,) * 6, range_mode="grip", ema=0.1,
                     full_range_s=1.0, policy_hz=60.0, freeze=False, freeze_joints=(True,) * 6, freeze_threshold_n=1.0, hold="open")
    lo, hi = law.bounds()
    np.testing.assert_allclose(law.raw(-np.ones(6)), CAPTOP_OPEN)          # 규약: −1 → 편 자세 · +1 → 쥔 자세
    np.testing.assert_allclose(law.raw(np.ones(6)), CAPTOP_GRIP)
    base = np.array([1.2, 0.1, 0.9, 0.9, 0.9, 0.9])

    def target(a):
        return law.raw(CS.seat_hand_action(a, base, lo, hi))

    np.testing.assert_allclose(target(np.zeros(6)), base)                  # 잔차 0 = 넘긴 손 목표 그대로
    np.testing.assert_allclose(target(np.full(6, 0.1)), base + 0.1 * (hi - lo))
    np.testing.assert_allclose(target(np.full(6, -0.1)), base - 0.1 * (hi - lo))
    np.testing.assert_allclose(target(np.ones(6)), CAPTOP_GRIP)            # 범위 밖은 자름
    np.testing.assert_allclose(target(-np.ones(6)), CAPTOP_OPEN)
    for bad in ((np.zeros(5), base, lo, hi), (np.full(6, np.nan), base, lo, hi), (np.zeros(6), base[:5], lo, hi),
                (np.zeros(6), base, lo, lo), (np.zeros(6), base, hi, lo)):  # 폭 0 · 뒤집힌 범위
        with pytest.raises(ValueError):
            CS.seat_hand_action(*bad)


def test_meta_h01_accepted():
    CS.check_meta(META)
    CS.check_meta({**META, "delta": False})


@pytest.mark.parametrize("bad", [
    {"n_obs": 61}, {"n_act": 13}, {"stages": ["grasp", "carry", "seat"]}, {"head": False}, {"img_hw": [96, 160]},
    {"head_cfg": {"pan_lim": (-45.0, 30.0), "tilt_lim": (-2.0, 86.0)}}, {"head_cfg": {"pan_lim": (-40.0, 30.0), "tilt_lim": (-2.0, 82.0)}},
    {"head_cfg": None}, {"arch": "v2"}, {"kind": "mask"}, {"task": "rh_aglt"}, {"d_min": 0.3}, {"d_max": 1.3}, {"delta": True},
])
def test_meta_field_mismatch_rejected(bad):
    with pytest.raises(ValueError):
        CS.check_meta({**META, **bad})


@pytest.mark.parametrize("key", ["kind", "task", "n_obs", "n_act", "head", "img_hw", "stages", "d_min", "d_max", "head_cfg"])
def test_meta_missing_key_rejected(key):
    with pytest.raises(ValueError):
        CS.check_meta({k: v for k, v in META.items() if k != key})


class _FakeMod:
    @staticmethod
    def depth_input(d, d_min, d_max):
        return d


class _FakeModel:
    """모델에 들어간 깊이를 적고 상태를 1 씩 올린다. nan=True 면 NaN 을 낸다."""

    def __init__(self):
        self.seen, self.nan = [], False

    def zero_state(self, n, device):
        return 0

    def __call__(self, img, obs, hc, train_cnn):
        import torch

        self.seen.append(img[0].numpy().copy())
        return torch.full((1, CS.N_ACT), float("nan") if self.nan else 0.25), None, hc + 1


def test_step_blanks_first_depth_and_keeps_state_on_any_failure():
    pytest.importorskip("torch")
    pol, model = CS.CapStudent.__new__(CS.CapStudent), _FakeModel()
    pol._attach(_FakeMod, model, 0.2, 1.5)
    depth, obs = np.full(CS.IMG_HW, 0.8, np.float32), np.zeros(CS.N_OBS, np.float32)

    def fails(exc, state):
        mm = depth * 1000.0                                                # mm 단위 오입력
        neg, nan, obs_nan = depth.copy(), depth.copy(), obs.copy()
        neg[0, 0], nan[3, 5], obs_nan[7] = -0.1, np.nan, np.nan
        for err, args in ((CS.StaleDepth, (depth, obs, 1.0)), (CS.StaleDepth, (depth, obs, -0.01)),
                          (CS.StaleDepth, (depth, obs, float("nan"))), (ValueError, (depth[:, :-1], obs, 0.0)),
                          (ValueError, (nan, obs, 0.0)), (ValueError, (neg, obs, 0.0)), (ValueError, (mm, obs, 0.0)),
                          (ValueError, (depth, obs[:-1], 0.0)), (ValueError, (depth, obs_nan, 0.0))):
            n = len(model.seen)
            with pytest.raises(err):
                pol.step(*args)
            assert (pol.hc, pol.first) == state and len(model.seen) == n, f"{err.__name__} 뒤 상태가 바뀌었거나 모델을 불렀다"
        model.nan = True
        with pytest.raises(exc):
            pol.step(depth, obs, 0.0)
        model.nan = False
        assert (pol.hc, pol.first) == state                                # NaN 출력은 상태를 잇지 않는다

    fails(FloatingPointError, (0, True))                                   # 첫 틱 전 — 실패 뒤에도 첫 틱 표시가 남는다
    seen0 = len(model.seen)
    a, head = pol.step(depth, obs, 0.0)
    assert not model.seen[seen0].any() and (pol.hc, pol.first) == (1, False)       # 첫 틱: 모델에 들어간 깊이 0
    np.testing.assert_allclose(a, np.full(13, 0.25))
    np.testing.assert_allclose(head, CS.head_deg([0.25, 0.25]))
    pol.step(depth, obs, 0.2)                                              # 나이 = 상한은 통과
    np.testing.assert_array_equal(model.seen[-1], depth)                   # 둘째 틱: 원래 깊이
    assert pol.hc == 2
    fails(FloatingPointError, (2, False))
    pol.reset()
    assert (pol.hc, pol.first) == (0, True)


def _bundle(tmp_path, **over):
    """tmp 묶음 폴더(student.yaml + 가짜 ckpt). over 의 None 값 = 그 키를 뺀다."""
    import yaml

    d = tmp_path / "b"
    (d / "student").mkdir(parents=True)
    (d / "student" / "x.pt").write_bytes(b"not a ckpt")
    (tmp_path / "out.pt").write_bytes(b"not a ckpt")
    cfg = {"task": "rh_cap", "ckpt": {"path": "student/x.pt", "md5": "0" * 32},
           "hdgp": {"commit": hdgp_pin.load_snapshot_cfg(CS.SNAPSHOT_CFG)["commit"]}, **over}
    (d / CS.STUDENT_CARD).write_text(yaml.safe_dump({k: v for k, v in cfg.items() if v is not None}), encoding="utf-8")
    return d


@pytest.mark.parametrize("over, match", [
    ({}, "md5"),                                                           # 다른 파일의 md5
    ({"ckpt": {"path": "../out.pt", "md5": "0" * 32}}, "폴더 밖"),
    ({"ckpt": {"path": "student/none.pt", "md5": "0" * 32}}, "없거나"),
    ({"ckpt": {"path": "student/x.pt"}}, "필요하다"),
    ({"ckpt": None}, "필요하다"), ({"hdgp": None}, "필요하다"), ({"task": None}, "필요하다"),
    ({"task": "rh_aglt"}, "기대"), ({"hdgp": {"commit": "0" * 40}}, "기대"),
])
def test_from_dir_bundle_errors_are_pin_errors(tmp_path, monkeypatch, over, match):
    monkeypatch.delenv("HDGP_STUDENT_SNAPSHOT_COMMIT", raising=False)
    with pytest.raises(hdgp_pin.PinError, match=match):
        CS.CapStudent.from_dir(_bundle(tmp_path, **over))


def test_from_dir_without_card_is_pin_error(tmp_path):
    with pytest.raises(hdgp_pin.PinError):
        CS.CapStudent.from_dir(tmp_path)


def test_real_ckpt_forward_and_reset():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if not (H01 / "student" / "student_it06000.pt").is_file() or not (hdgp_pin.hdgp_tree() / ".git").exists():
        pytest.skip("h01 ckpt(git 밖) 또는 hdgp 저장소 없음")
    pol = CS.CapStudent.from_dir(H01)
    assert pol.md5 == H01_MD5 and pol.meta["it"] == 6000 and (pol.d_min, pol.d_max) == (0.2, 1.5)
    assert pol.first and all(float(h.abs().sum()) == 0.0 for h in pol.hc)  # 워밍업이 상태를 남기지 않는다
    obs = pol.model.mu.numpy().copy()             # 정규화 뒤 0 이 되는 관측
    depth = np.zeros(CS.IMG_HW, np.float32)
    outs = []
    for _ in range(2):
        pol.reset()
        a, head = pol.step(depth, obs, 0.0)
        a2, _ = pol.step(np.full(CS.IMG_HW, 0.8, np.float32), obs, 0.0)
        assert a.shape == (13,) and head.shape == (2,) and np.isfinite(np.r_[a, head, a2]).all()
        assert np.all(np.abs(a) <= 1.0) and np.all((head >= CS.HEAD_LO) & (head <= CS.HEAD_HI))
        outs.append(np.r_[a, head, a2])
    np.testing.assert_array_equal(outs[0], outs[1])                        # reset 뒤 같은 입력 → 같은 출력
    assert not np.array_equal(outs[0][:13], outs[0][15:])                  # LSTM 상태 · 깊이가 이어진다
    with pytest.raises(CS.StaleDepth):
        pol.step(depth, obs, 1.0)
    with pytest.raises(hdgp_pin.PinError):        # 다른 파일의 md5
        CS.CapStudent(H01 / "student" / "student_it06000.pt", "0" * 32)

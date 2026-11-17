"""plan_place_fixed: 손바닥 목표 = 컵 평행 이동 + 컵 축 둘레 yaw, 설정 오프셋 상한."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"))
import plan_place_fixed as PF  # noqa: E402

CFG = Path(__file__).resolve().parents[2] / "deploy/policy_control/config/place_fixed"


def test_palm_pose_translate_and_yaw_about_cup_axis():
    p0, cup0, cup = np.array([0.22, -0.16, 0.42]), np.array([0.25, -0.12, 0.41]), np.array([0.38, -0.06, 0.30])
    p, R = PF.palm_pose(p0, np.eye(3), cup0, cup, 0.0)
    assert np.allclose(p - cup, p0 - cup0) and np.allclose(R, np.eye(3))
    p, R = PF.palm_pose(p0, np.eye(3), cup0, cup, 90.0)
    assert np.isclose(np.linalg.norm((p - cup)[:2]), np.linalg.norm((p0 - cup0)[:2]))   # 컵 축에서 거리 유지
    assert np.isclose((p - cup)[2], (p0 - cup0)[2]) and np.allclose(R[:, 2], [0, 0, 1])
    assert np.allclose(p - cup, [0.04, -0.03, 0.01])                                     # (−0.03, −0.04) → +90°


@pytest.mark.parametrize("name", ["blue_h1", "pink_h2", "shaker_h0"])
def test_cfg_loads_and_offset_guard(name, tmp_path):
    PF.load_cfg(CFG / f"{name}.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text(re.sub(r"yaw_deg: -?[\d.]+", "yaw_deg: 75.0", (CFG / f"{name}.yaml").read_text()))
    with pytest.raises(ValueError):
        PF.load_cfg(bad)


def test_hand_schedule_grip_then_open_before_lift():
    grip, open_ = (1.2, 0.24, 1.08, 1.08, 0.85, 0.85), (1.57, 0.0, 0.0, 0.0, 0.0, 0.0)
    hand, at = PF.hand_schedule(grip, open_, n=200, r0=50, step=1.0 / 60)
    assert np.allclose(hand[:50], grip) and np.allclose(hand[at:], open_)
    assert at == 50 + 64                                    # 1.08 rad / (1/60 rad 틱) = 64.8 → 65 틱, r0 칸부터 움직임
    _, at = PF.hand_schedule(grip, open_, n=80, r0=50, step=1.0 / 60)
    assert at == 80                                         # 끝까지 못 펴면 n(= 개방 여유 음수 → 실패)


def test_all_ok_any_failure_blocks_write():
    checks = {"a": {"value": 1.0, "limit": 0.5}, "b": {"value": 0.1, "limit": [0.0, 0.2], "ok": True}}
    assert PF.all_ok(checks) and checks["a"]["ok"] is True
    checks["c"] = {"value": 0.004, "limit": 0.01}
    assert not PF.all_ok(checks) and checks["c"]["ok"] is False


def test_cfg_missing_keys_and_short_release(tmp_path):
    src = (CFG / "blue_h1.yaml").read_text()
    for bad_text in (src.replace("yaw_deg: -10.0", ""), src.replace(", hold: 1.0", ""), src.replace("release: 1.5", "release: 1.2")):
        bad = tmp_path / "bad.yaml"
        bad.write_text(bad_text)
        with pytest.raises(ValueError):
            PF.load_cfg(bad)


def test_check_grip_range():
    assert PF.check_grip((1.2, 0.24, 1.08, 1.08, 0.85, 0.85))
    for g in ((1.2, 0.24, 1.08, 1.08, 0.85), (1.2, -0.1, 1.08, 1.08, 0.85, 0.85), (1.2, 0.24, 1.6, 1.08, 0.85, 0.85)):
        with pytest.raises(ValueError):
            PF.check_grip(g)


def test_fit_radius_prefers_cfg_over_contract_radius():
    g = {"radius": 0.0354}
    assert PF.fit_radius({}, g) == {"fit_radius": 0.0354, "fit_radius_source": "contract.radius"}
    assert PF.fit_radius({"fit_radius_m": 0.031}, g) == {"fit_radius": 0.031, "fit_radius_source": "cfg.fit_radius_m"}
    with pytest.raises(ValueError):
        PF.fit_radius({"fit_radius_m": 0.04}, g)


@pytest.mark.parametrize("edit", [
    lambda s: s.replace("place_dz: 0.005", ""),
    lambda s: s.replace("cup_start: [0.25, -0.12, 0.41]", "cup_start: [0.25, -0.12]"),
    lambda s: s.replace("cup_start: [0.25, -0.12, 0.41]", "cup_start: [0.25, .nan, 0.41]"),
    lambda s: s.replace("holder: 1 ", "holder: -1 "),
    lambda s: s.replace("holder: 1 ", "holder: 1.5 "),
    lambda s: s.replace("contract: deploy", "contrac: deploy"),
    lambda s: s.replace("object: source240_blue", ""),
    lambda s: s.replace("cup: blue", ""),
    lambda s: "- 1\n- 2\n",
])
def test_cfg_required_keys(edit, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(edit((CFG / "blue_h1.yaml").read_text()))
    with pytest.raises(ValueError):
        PF.load_cfg(bad)


def test_release_checks_bounds():
    ok = PF.release_checks(0.03, [-0.005, 0.03], 0.002, 0.002, 0.3, 0.5, 3.0)
    assert all(c["ok"] for c in ok.values())
    bad = PF.release_checks(0.0301, [-0.005, 0.03], 0.0021, 0.002, 0.299, 0.5, 3.01)   # 정지는 ms 반올림 뒤 비교(원래 동작)
    assert not any(c["ok"] for c in bad.values())
    assert not PF.release_checks(-0.0051, [-0.005, 0.03], 0.0, 0.002, 1.0, 0.5, 0.0)["release_drop_m"]["ok"]


def test_q_start_dev_place_arm_only():
    nom = {"right": [0.0] * 7, "left": [0.0] * 7}
    # 분홍 놓기 뒤 오른팔은 lift 자세(크게 벗어남), shaker 는 왼팔이 놓는다
    q = {"right": [np.radians(40.0)] * 7, "left": [0.0] * 6 + [np.radians(2.9)]}
    dev = PF.q_start_dev_deg(q, nom, "left")
    assert np.isclose(dev, 2.9)
    assert PF.release_checks(0.0, [-0.005, 0.03], 0.0, 0.002, 1.0, 0.5, dev)["q_start_dev_deg"]["ok"]
    dev_r = PF.q_start_dev_deg(q, nom, "right")                        # 놓는 팔이 벗어나면 실패
    assert not PF.release_checks(0.0, [-0.005, 0.03], 0.0, 0.002, 1.0, 0.5, dev_r)["q_start_dev_deg"]["ok"]


def test_main_failed_plan_exit2_and_no_npz(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "place_cmd.npz").write_bytes(b"old")
    meta = {"checks": {"hand_open_settle_s": {"value": 0.1, "ok": False}}, "place_xyz": [0, 0, 0], "T_end_s": 1.0,
            "release_drop_m": 0.0, "release_xy_m": 0.0, "release_xy_tol_m": 0.0}
    monkeypatch.setattr(PF, "plan", lambda *a, **k: (False, meta))
    assert PF.main(["--cup", "blue", "--out", str(out)]) == 2
    assert not (out / "place_cmd.npz").exists()

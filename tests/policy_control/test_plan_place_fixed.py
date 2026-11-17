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


# ★11.18 콘솔 고정 놓기 튜닝(jog · 오프셋 저장 · 마저 놓기)의 순수 계산
def test_jog_limits_step():
    assert all(v["ok"] for v in PF.jog_limits([0.005, 0.0, 0.0], 1.0).values())
    assert not PF.jog_limits([0.011, 0.0, 0.0], 0.0)["jog_step_mm"]["ok"]
    assert not PF.jog_limits([0.0, 0.0, 0.0], 3.5)["jog_step_deg"]["ok"]


def test_jog_anchor_absolute_from_descend_end_plan():
    plan, R_plan = np.array([0.38, -0.06, 0.30]), PF.plan_end_R(2.0, np.eye(3))
    c = PF.jog_anchor(plan + [0.0, 0.039, 0.0], plan, PF.rz(16.0), R_plan, np.radians(90.0))
    assert c["jog_anchor_mm"]["ok"] and c["jog_anchor_deg"]["ok"]             # 14° · 39 mm
    assert np.allclose(c["jog_anchor_mm"]["holder_mm"], [39.0, 0.0, 0.0], atol=1e-6)   # 홀더 yaw 90° → base +y = 홀더 +x
    assert np.isclose(c["jog_anchor_deg"]["yaw_deg"], 14.0)
    c = PF.jog_anchor(plan + [0.0, 0.041, 0.0], plan, PF.rz(17.5), R_plan, 0.0)
    assert not c["jog_anchor_mm"]["ok"] and not c["jog_anchor_deg"]["ok"]   # 여러 번 나눠도 절대 기준으로 거부


def test_bore_clear_only_below_rim():
    hxy, top, r = np.array([0.38, -0.06, 0.2]), 0.23, 0.030               # 여유 HOLDER_BORE_R − r = 5 mm
    assert PF.bore_clear_m([[0.39, -0.06]], [0.24], top, hxy, r) is None   # 링 위면 판정 안 함
    assert np.isclose(PF.bore_clear_m([[0.384, -0.06], [0.50, 0.0]], [0.22, 0.30], top, hxy, r), 0.001)
    assert PF.bore_clear_m([[0.386, -0.06]], [0.22], top, hxy, r) < 0                         # 링 안에서 벽에 닿는다


def test_save_then_resume_start_check_passes():
    """오프셋 저장(offset_update) 직후 같은 자세로 마저 놓기 시작 검사(resume_checks · plan_end_R)가 통과한다."""
    Rn, hpos = PF.rz(-30.0) @ np.diag([1.0, -1.0, -1.0]), np.array([0.38, -0.06, 0.20])
    off = {"x": 0.002, "y": -0.001, "z": 0.0, "yaw_deg": 3.0}
    cup_plan = PF.cup_place(hpos, 0.05, 0.005, off)
    cup_now, R_now = cup_plan + [0.012, -0.020, 0.004], PF.rz(7.0) @ PF.plan_end_R(off["yaw_deg"], Rn)
    assert not all(c["ok"] for c in PF.resume_checks(cup_now, cup_plan, R_now, PF.plan_end_R(off["yaw_deg"], Rn)).values())
    new, _ = PF.offset_update(off, cup_now, cup_plan, R_now, PF.plan_end_R(off["yaw_deg"], Rn), 0.3)
    c = PF.resume_checks(cup_now, PF.cup_place(hpos, 0.05, 0.005, new), R_now, PF.plan_end_R(new["yaw_deg"], Rn))
    assert all(v["ok"] for v in c.values()), c


def test_write_offset_backs_up_with_ms_and_keeps_original_on_bad_yaml(tmp_path):
    src = (CFG / "blue_h1.yaml").read_text(encoding="utf-8")
    path = tmp_path / "blue_h1.yaml"
    path.write_text(src, encoding="utf-8")
    off = PF.load_cfg(path)["offset"]
    new = {**off, "x": round(off["x"] + 0.001, 4)}
    bak = PF.write_offset(path, new, off, now=1763430000.123)
    assert re.fullmatch(r"blue_h1_\d{6}_\d{6}_123\.yaml", bak.name) and bak.read_text(encoding="utf-8") == src
    assert PF.load_cfg(path)["offset"] == new
    with pytest.raises(FileExistsError):                                    # 같은 ms 백업은 덮어쓰지 않는다
        PF.write_offset(path, off, new, now=1763430000.123)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):                                         # 상한(10 cm) 초과 — 원본 그대로
        PF.write_offset(path, {**new, "x": 0.5}, new, now=1763430001.0)
    path.write_text(before.replace("cup:", "cup: [", 1), encoding="utf-8")  # 깨진 yaml
    broken = path.read_text(encoding="utf-8")
    with pytest.raises(PF.yaml.YAMLError):
        PF.write_offset(path, new, new, now=1763430002.0)
    assert path.read_text(encoding="utf-8") == broken and not path.with_suffix(".tmp.yaml").exists()


def test_offset_update_adds_base_delta_and_reports_holder_frame():
    off = {"x": 0.001, "y": 0.0, "z": -0.002, "yaw_deg": 1.0}
    R_plan = np.eye(3)
    R_now = PF.rz(2.0)
    new, d = PF.offset_update(off, [0.31, 0.10, 0.20], [0.30, 0.10, 0.20], R_now, R_plan, np.radians(90.0))
    assert new == {"x": 0.011, "y": 0.0, "z": -0.002, "yaw_deg": 3.0}
    assert np.allclose(d["base_mm"], [10.0, 0.0, 0.0]) and np.isclose(d["yaw_deg"], 2.0)
    assert np.allclose(d["holder_mm"], [0.0, -10.0, 0.0], atol=1e-6)   # 홀더 yaw 90° → base +x = 홀더 −y


def test_resume_checks_tolerance():
    assert all(v["ok"] for v in PF.resume_checks([0, 0, 0.004], [0, 0, 0], PF.rz(4.0), np.eye(3)).values())
    c = PF.resume_checks([0, 0, 0.006], [0, 0, 0], PF.rz(6.0), np.eye(3))
    assert not c["resume_pos_mm"]["ok"] and not c["resume_ang_deg"]["ok"]


def test_offset_line_replaces_only_offset_line():
    text = "cup: blue\n# 주석\noffset: {x: 0.0, y: 0.0, z: 0.0, yaw_deg: 0.0}   # 옛\nplace_dz: 0.005\n"
    out = PF.offset_line(text, {"x": 0.01, "y": -0.002, "z": 0.0, "yaw_deg": 1.5}, "콘솔 저장")
    assert out.splitlines()[0:2] == text.splitlines()[0:2] and out.splitlines()[3] == "place_dz: 0.005"
    assert out.splitlines()[2].startswith("offset: {x: 0.01, y: -0.002, z: 0.0, yaw_deg: 1.5}")
    with pytest.raises(ValueError):
        PF.offset_line("a: 1\n", {"x": 0, "y": 0, "z": 0, "yaw_deg": 0}, "x")

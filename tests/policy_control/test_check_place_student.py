"""scripts/ops/check_place_student.py — 가짜 놓기 묶음(test_pc_place_student._meta)으로 불일치 → 종료 코드 1."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
EXTR = REPO / "config/global_camera_extrinsics_arm4090_r20u5.yaml"
YOLO = REPO / "logs/policy/student_mask_A/yolo/seg_s640_A/best.pt"
#: 실제 hdgp 7ffc4f8f · 잡기 ckpt 에 있고 _meta 에 없는 키(가짜 묶음을 실 ckpt 모양에 가깝게)
EXTRA = {"one_obj": "source240_cvx", "head": {"rate_dps": [45.0, 75.0], "accel_dps2": [200.0, 400.0]},
         "depth_pre": {"out_wh": [160, 90], "render_wh": [1280, 720]}}


def load_check():
    spec = importlib.util.spec_from_file_location("check_place_student", REPO / "scripts/ops/check_place_student.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_bundle(d: Path, c, extra: dict = EXTRA) -> Path:
    """가짜 놓기 학생 묶음: student.yaml · ckpt(= _meta + extra, 가중치 없음) · r20u5 사본 · seg_s640_A 링크."""
    import torch
    from test_pc_place_student import _meta

    from policy_control import hdgp_pin
    from policy_control.student_policy import SNAPSHOT_CFG

    (d / "student").mkdir(parents=True)
    (d / "camera").mkdir()
    shutil.copy(EXTR, d / "camera/extr.yaml")
    (d / "student/yolo.pt").symlink_to(YOLO)
    ex_md5 = hdgp_pin.md5(EXTR)
    meta = _meta(c, ex_md5, 0.185)
    for k, v in extra.items():
        meta[k] = {**meta[k], **v} if isinstance(v, dict) else v
    torch.save(meta, d / "student/ckpt.pt")
    card = {"kind": "student_place", "side": "right", "task": "rh_place", "rgb": False,
            "mask": {"mode": "P", "target_class": "blue",
                     "yolo": {"path": "student/yolo.pt", "md5": hdgp_pin.md5(YOLO), "conf": 0.25, "imgsz": 640,
                              "retina_masks": True, "top1_per_class": True, "cross_class_iou": 0.5}},
            "ckpt": {"path": "student/ckpt.pt", "md5": hdgp_pin.md5(d / "student/ckpt.pt")},
            "extrinsics": {"path": "camera/extr.yaml", "md5": ex_md5},
            "hdgp": {"commit": hdgp_pin.load_snapshot_cfg(SNAPSHOT_CFG)["commit"]}}
    (d / "student.yaml").write_text(yaml.safe_dump(card, allow_unicode=True), encoding="utf-8")
    return d


def test_ckpt_md5_mismatch_exits_nonzero(tmp_path):
    if not YOLO.is_file():
        pytest.skip(f"YOLO 없음 {YOLO}")
    m = load_check()
    from policy_control import rh_place as P

    c = P.load_contract(m.CONTRACTS["right"])
    d = make_bundle(tmp_path / "b", c)
    mission = tmp_path / "mission.yaml"     # --max-tilt 82 를 준 목 드라이버 하나(지금 미션의 불일치와 떼어 본다)
    mission.write_text(yaml.safe_dump({"steps": [{"argv": ["python3", "head_joint_publisher.py", "--max-tilt", "82"]}]}))
    args = ["--student-dir", str(d), "--mission", str(mission)]
    rows = {r[1]: r[4] for r in m.run(args) if r[0] == "right"}
    assert set(rows.values()) <= {m.OK, m.NA}, rows
    card = yaml.safe_load((d / "student.yaml").read_text())
    card["ckpt"]["md5"] = "0" * 32
    (d / "student.yaml").write_text(yaml.safe_dump(card, allow_unicode=True))
    assert {r[1]: r[4] for r in m.run(args)}["ckpt md5"] == m.BAD
    assert m.main(args) == 1


def test_episode_scope_reads_only_the_place_episode_driver_and_a_missing_bundle_blocks(tmp_path):
    """★11.08 --max-tilt 82 는 놓기 에피소드 드라이버에만 있다 — --episode 면 그 단계만 보고 OK, 미션 전체면 head_home 이 불일치.
    묶음이 아직 없으면 student.yaml 불일치로 종료 코드 1."""
    m = load_check()
    args = ["--student-dir", str(tmp_path / "none"), "--side", "right"]
    rows = m.run([*args, "--episode", "place_m_right"])
    tilt = {r[1]: r[4] for r in rows if r[1].startswith("미션 --max-tilt")}
    assert tilt and set(tilt.values()) == {m.OK} and all("episode_place_m_right" in k for k in tilt), tilt
    assert {r[1]: r[4] for r in rows}["놓기 학생 묶음 student.yaml"] == m.BAD
    assert m.main([*args, "--episode", "place_m_right"]) == 1
    whole = {r[1]: r[4] for r in m.run(args) if r[1].startswith("미션 --max-tilt")}
    assert m.BAD in whole.values() and m.OK in whole.values(), whole      # head_home 드라이버는 기본 −10 그대로
    with pytest.raises(SystemExit):
        m.run([*args, "--episode", "no_such_episode"])


def test_episode_scope_skips_the_tail_driver_and_blocks_when_no_driver_is_found(tmp_path):
    """★11.08 리뷰: 에피소드 끝 꼬리 드라이버(기본 −10)는 학생 뒤라 보지 않는다. 에피소드에 드라이버가 없으면 NA 가 아니라 BAD."""
    m = load_check()
    args = ["--student-dir", str(tmp_path / "none"), "--side", "right"]
    tilt = [r for r in m.run([*args, "--episode", "place_m_right"]) if r[1].startswith("미션 --max-tilt")]
    assert len(tilt) == 1, tilt
    mission = tmp_path / "mission.yaml"
    mission.write_text(yaml.safe_dump({"run": {"episode_x": [{"argv": ["bash", "-lc", "true"]}]}}))
    rows = {r[1]: r[4] for r in m.run([*args, "--mission", str(mission), "--episode", "x"])}
    assert rows["미션 목 드라이버"] == m.BAD
    assert {r[1]: r[4] for r in m.run([*args, "--mission", str(mission)])}["미션 목 드라이버"] == m.NA

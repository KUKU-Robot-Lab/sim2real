"""11.18 rh_cap 깊이 학생 묶음(right_student_h01)의 계약 — 빌더 학생 모드(--student-dir)와 노드 계약 대조(contract_mismatch).
ROS · GPU 없음. 학생 ckpt(git 밖) · hdgp 캡 json 이 없는 호스트에서는 빌더 시험을 건너뛴다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from dataclasses import replace

from policy_control import rh_cap as C
from policy_control import rh_cap_node as RN
from policy_control import rh_cap_student_node as N

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "deploy/policies/rh56f1/cap/right_student_h01"
CONTRACT = BUNDLE / "rh_cap_contract.json"
CARD = yaml.safe_load((BUNDLE / "student.yaml").read_text(encoding="utf-8"))
HDGP = Path.home() / "rl_ws/hdgp"


def _builder():
    p = REPO / "deploy/policy_control/tools/build_rh_cap_contract.py"
    spec = importlib.util.spec_from_file_location("build_rh_cap_contract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bundle_contract_loads_and_matches_student_card():
    c = C.load_contract(CONTRACT)
    assert N.contract_mismatch(c, CARD) == []
    assert c.cap_phase == "chain" and c.action_dim == 13 and c.checkpoint_md5 == CARD["ckpt"]["md5"]


@pytest.mark.skipif(not (BUNDLE / CARD["ckpt"]["path"]).is_file()
                    or not (HDGP / "assets/cyl_cap/shaker_pair_donut12/cap_c6.json").is_file(),
                    reason="학생 ckpt(git 밖) 또는 hdgp 캡 기하 없음")
def test_student_mode_build_matches_bundle_params(tmp_path):
    out = tmp_path / "c.json"
    assert _builder().main(["--student-dir", str(BUNDLE), "--hdgp", str(HDGP), "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    env = yaml.unsafe_load((BUNDLE / "params/env.yaml").read_text())
    for k in ("cap_phase", "phase_arm_box_rad", "grasp_ik_lift_m", "seat_tol", "phase_success_steps", "k_arm", "arm_ema",
              "hand_ema", "hand_full_range_s", "joint_err_norm"):
        assert got[k] == env[k], k
    assert got["profile"] == env["profile_name"] == CARD["train"]["profile_name"]
    assert got["policy_hz"] == pytest.approx(CARD["train"]["policy_hz"])
    assert got["hand_command"] == "position" and got["hand_block"]
    assert got["checkpoint_md5"] == CARD["ckpt"]["md5"]
    want = json.loads(CONTRACT.read_text())                  # 묶음에 둔 계약 = 지금 빌더 결과(호스트마다 다른 절대 경로 3개 빼고)
    for k in ("run_dir", "checkpoint", "cap_json"):
        assert Path(got.pop(k)).name == Path(want.pop(k)).name, k
    assert got == want


def test_student_mode_still_refuses_teacher_phase_without_flag():
    env = {"cap_phase": "chain", "profile_name": "rh56f1_v2_right_captop"}
    assert C.is_rh_cap_run(env, student=True) and not C.is_rh_cap_run(env)
    assert not C.is_rh_cap_run({"cap_phase": "grasp", "profile_name": "rh56f1_v2_right_captop"}, student=True)


def test_teacher_node_refuses_chain_and_student_node_refuses_teacher_phases():
    chain = C.load_contract(CONTRACT)
    seat = C.load_contract(Path(__file__).resolve().parent / "fixtures" / "rh_cap_seat_contract.json")
    assert RN.CapNode.phase_refusal(seat) is None and "chain" in RN.CapNode.phase_refusal(chain)
    assert N.CapStudentNode.phase_refusal(chain) is None
    for ph in C.PHASES:
        assert N.CapStudentNode.phase_refusal(replace(seat, cap_phase=ph)) is not None


def test_contract_mismatch_catches_md5_and_phase():
    c = C.load_contract(CONTRACT)
    assert [b.split(":")[0] for b in N.contract_mismatch(replace(c, checkpoint_md5="0" * 32), CARD)] == ["checkpoint_md5"]
    assert [b.split(":")[0] for b in N.contract_mismatch(replace(c, cap_phase="seat"), CARD)] == ["cap_phase"]


def test_builder_student_refusals_write_nothing(tmp_path):
    d, out = tmp_path / "b", tmp_path / "c.json"
    d.mkdir()
    run = lambda *a: _builder().main(["--student-dir", str(d), "--out", str(out), *a])  # noqa: E731
    with pytest.raises(SystemExit, match="student.yaml"):
        run()
    (d / "student.yaml").write_text(yaml.safe_dump({**CARD, "ckpt": {"path": "x.pt", "md5": "0" * 32}}))
    with pytest.raises(SystemExit, match="ckpt"):
        run()
    (d / "x.pt").write_bytes(b"not the student")
    with pytest.raises(SystemExit, match="md5"):
        run()
    with pytest.raises(SystemExit, match="--checkpoint"):
        run("--checkpoint", "x.pt")
    assert not out.exists()

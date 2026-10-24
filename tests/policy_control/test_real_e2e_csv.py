"""★10.25 실기 e2e CSV 변환기 — 계약 35 열 순서 · 단계 판정 · 신호 없는 옛 기록."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy/policy_control/tools"))
import real_e2e_csv as E  # noqa: E402

CONTRACT_COLS = ("side, seed, cond, latch, layout, env, cup, x, y, ap_ok, ap_steps, grasp_ok, grasp_end, t_handoff_s, t_grasp_s, "
                 "ho_phi_deg, ho_dphi_deg, ho_cup_move_mm, est_err_mm, est_err_deg, carry_ok, carry_steps, carry_err_mm, "
                 "carry_tilt_deg, carry_palm_cup_mm, seam_drift_mm, seam_rot_deg, place_ok, grade, perfect, place_reasons, "
                 "phys_fault, t_place_s, e2e_ok, fail_stage")


def _run(tmp: Path, name: str, rows: list[dict], cup=(0.31, -0.07)) -> Path:
    d = tmp / f"20261026_1200{len(list(tmp.iterdir())):02d}_pick_place_f_right"
    d.mkdir()
    base = {"episode_id": d.name[:15], "episode": "pick_place_f_right"}
    (d / f"episode_{d.name[:15]}.jsonl").write_text("".join(json.dumps(base | r, ensure_ascii=False) + "\n" for r in rows))
    (d / "scene.json").write_text(json.dumps({"objects": {"CUP": [[cup[0], cup[1], 0.29], [1, 0, 0, 0]]}}))
    return d


G_OK = {"event": "exit", "policy_name": "rh_aglt_r", "duration": "12.5", "termination_reason": "lift_hold: cup +0.1 m"}
RC = {"on": False, "t": 214, "arrived": True, "carry_ok": True, "err_mm": 0.4, "plan": {"anchor_err_mm": 3.2}}


def _place(event="exit", hold=True, empty=True, rc=RC, reason="placed"):
    return {"event": event, "policy_name": "rh_place_r", "duration": "9.1", "termination_reason": reason,
            "signals": {"rh_place_r": {"hand_empty": empty, "cup_in_holder": hold, "ready_carry": rc}}}


def test_columns_follow_the_contract():
    assert E.COLS == tuple(c.strip() for c in CONTRACT_COLS.split(","))


@pytest.mark.parametrize("rows, stage", [
    ([G_OK, _place()], "none"),
    ([G_OK, _place("failure", hold=False)], "place"),
    ([G_OK, _place(rc=RC | {"carry_ok": False, "t": 600})], "carry"),
    ([{"event": "failure", "policy_name": "rh_aglt_r", "duration": "15", "termination_reason": "approach_grasp: 접근 시간 초과"}], "approach"),
    ([{"event": "failure", "policy_name": "rh_aglt_r", "duration": "33", "termination_reason": "경로 D 시간 초과 33.0 s"}], "grasp"),
    ([G_OK, {"event": "failure", "policy_name": "rh_place_r", "duration": "5.3", "termination_reason": "place 시간 초과"}], "place"),
    ([G_OK, {"event": "failure", "policy_name": "rh_place_r", "termination_reason": "set_parameters: 서비스가 없다"}], ""),
    ([G_OK, _place() | {"signals": {"rh_place_r": {"hand_empty": True, "cup_in_holder": True, "pd_fault": "estop"}}}], "phys"),
])
def test_fail_stage(tmp_path, rows, stage):
    r = E.row_of(_run(tmp_path, "x", rows), 1, "gru", "est", "real")
    assert r["fail_stage"] == stage
    assert r["e2e_ok"] == ("" if stage == "" else int(stage == "none"))


def test_success_row_fills_carry_and_place(tmp_path):
    r = E.row_of(_run(tmp_path, "x", [G_OK, _place()]), 3, "gru", "est", "real")
    assert (r["side"], r["seed"], r["cup"], r["x"], r["y"]) == ("r", 3, "source240_real", 0.31, -0.07)
    assert (r["ap_ok"], r["grasp_ok"], r["grasp_end"], r["t_grasp_s"]) == (1, 1, "lift_hold", 12.5)
    assert (r["carry_ok"], r["carry_steps"], r["carry_err_mm"]) == (1, 214, 3.2)
    assert (r["place_ok"], r["t_place_s"], r["grade"], r["perfect"]) == (1, 9.1, "", "")


def test_no_carry_stage_leaves_carry_blank(tmp_path):
    r = E.row_of(_run(tmp_path, "x", [G_OK, _place(rc=None)]), 1, "gru", "est", "real")
    assert (r["carry_ok"], r["carry_steps"], r["fail_stage"]) == ("", "", "none")


def test_main_writes_csv(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    a = _run(runs, "a", [G_OK, _place()])
    b = _run(runs, "b", [G_OK, _place("failure", hold=False)])
    out = tmp_path / "e.csv"
    assert E.main([str(a), str(b), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert [r["seed"] for r in rows] == ["1", "2"] and [r["fail_stage"] for r in rows] == ["none", "place"]
    assert list(rows[0]) == list(E.COLS)

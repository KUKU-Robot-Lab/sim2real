"""sim_real_compare: sim 분포 대비 틱 비교 · 토글 라벨 · 요약 묶음(합성 입력)."""
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import sim_real_compare as C  # noqa: E402


def test_tick_compare_flags_shifted_cell_and_const():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(5000, 3))
    S[:, 2] = 1.0                                  # sim 상수 칸
    R = rng.normal(size=(50, 3))
    R[:, 1] += 10.0                                # 칸 1만 크게 벗어남
    R[:, 2] = 1.5
    r = C.tick_compare(S, R, ["a", "b", "c"])
    assert r["n_cell_z3"] == 1 and r["top"][0][0] == "b"
    assert r["sim_const_absdiff"] == [("c", 0.5)]
    assert all(n != "c" for n, *_ in r["top"])


def test_toggles_label_and_summary_groups():
    assert C.toggles_label({}) == C.NO_TOGGLES
    t = dict(C.format_toggles.__globals__["TOGGLES"])
    lab = C.toggles_label({"toggles": t})
    assert "hand_unfreeze_on_grasp=1" in lab
    g = {"stage": "grasp", "node": "g", "policy": "p", "obs": {"n_tick": 3, "maha_med": 1.0, "n_cell_z3": 0, "top": []},
         "action": {}, "outcome": "COMPLETED", "lift_dz": 0.1, "notes": []}
    eps = [{"name": "e1", "toggles": lab, "nodes": [g], "notes": []},
           {"name": "e2", "toggles": C.NO_TOGGLES, "nodes": [], "notes": ["bag 없음"]}]
    md = C.summary(eps)
    assert md.count("## 토글") == 2 and "| e1 | g | grasp |" in md and "bag 없음" in md


def test_tick_compare_ks_ranks_shifted_cell_and_warns_small_ref():
    rng = np.random.default_rng(1)
    S, R = rng.normal(size=(300, 3)), rng.normal(size=(80, 3))
    R[:, 2] += 4.0
    r = C.tick_compare(S, R, ["a", "b", "c"])
    assert r["ks_top"][0][0] == "c" and r["ks_top"][0][1] > 0.8 and r["ks_top"][0][2] > 3.0
    assert r["small_ref"] is True


def test_sim_folder_table_unknown_policy_is_none():
    assert C.sim_folder("rh56f1/aglt/no_such") is None
    assert C.sim_folder("rh56f1/aglt/right_rcg04").name == "ready_r"

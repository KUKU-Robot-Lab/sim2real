import json
from pathlib import Path

import yaml

from vlm_supervisor.bench.calibrate import NODE_CHECKS, held_out_thr, load_groups, node_eval

CFG = Path(__file__).resolve().parents[1] / "config" / "judge_short.yaml"


def test_released_not_used_and_pink_judged_next_node():
    # 11.18 실기: cup_released_in_holder yes 재현 11/33 — 판정에서 뺐다. 안착은 cup_in_holder_b 가 본다
    assert not any(s.startswith("cup_released_in_holder") for c in NODE_CHECKS.values() for s, _ in c)
    assert NODE_CHECKS["place_blue"] == [("cup_in_holder_b:SRC_BLUE", "yes")]
    assert NODE_CHECKS["place_shaker"] == [("cup_in_holder_b:SHAKER", "yes")]
    assert NODE_CHECKS["place_pink"] == []
    assert ("cup_in_holder_b:SRC_PINK", "yes") in NODE_CHECKS["home_r_after_pink"]
    assert NODE_CHECKS["scene"] == [("holders_any_cup", "no")]


def _row(seed, node, slot_, truth, p):
    return {"id": f"{seed}/{node}/{slot_}", "seed": seed, "node": node, "question_id": slot_.split(":")[0],
            "truth": truth, "p_yes": p}


def test_loeo_threshold_leaves_own_seed_out():
    # 시드 c 의 yes 만 p 0.3 — c 를 뺀 문턱은 0.3 위라 c 의 yes 를 놓치고, 다른 시드 문턱은 c 를 넣어 0.3 이하로 내려간다
    rows = [_row(sd, "n", f"cup_lifted:{i}", t, p) for sd, yes_p in [("a", 0.9), ("b", 0.9), ("c", 0.3)]
            for i, (t, p) in enumerate([("yes", yes_p), ("no", 0.1)])]
    th = held_out_thr(rows, "loeo")
    assert set(th) == {r["id"] for r in rows}
    assert th["c/n/cup_lifted:0"] > 0.3 and th["a/n/cup_lifted:0"] <= 0.3
    assert th["a/n/cup_lifted:0"] == th["a/n/cup_lifted:1"]  # 질문 하나·시드 하나에 문턱 하나


def test_node_eval_merges_boundary_group(tmp_path):
    # 실기 경계 하나에 쉐이커·파랑 잡기와 기울기 시작 이동이 묶인다 — 쉐이커를 못 들었으면 그 경계는 실패다
    b, node = "s/02_to_tilt_start_r1", "to_tilt_start_r1"
    rows = [_row("s", "02_to_tilt_start_r1", sl, t, p) for sl, t, p in [
        ("cup_in_hand:right:SRC_BLUE", "yes", 0.9), ("cup_lifted:SRC_BLUE", "yes", 0.9),
        ("cup_in_hand:left:SHAKER", "yes", 0.9), ("cup_lifted:SHAKER", "no", 0.1)]]
    for r in rows:
        r["node"] = node
    thr = {r["id"]: 0.5 for r in rows}
    # 묶음 없음(sim): 항목 node 하나의 검사(오른손 파랑)만 봐서 성공
    assert node_eval(rows, thr)[node] == {"ok_pass": 1}
    nodes = ["grasp_shaker", "grasp_blue", "to_tilt_start_l", node]
    (tmp_path / "boundaries.jsonl").write_text(json.dumps({"id": b, "node": node, "nodes": nodes}) + "\n")
    g = load_groups(tmp_path / "items.jsonl")
    assert g == {b: nodes}
    assert node_eval(rows, thr, g)[node] == {"fail_flag": 1}  # 결과는 마지막 노드 이름으로 센다
    # 묶음 검사가 모두 비면(place_pink) 건너뛰고, 슬롯이 빠지면 skip 으로 센다
    assert not node_eval(rows, thr, {b: ["place_pink"]})
    assert node_eval(rows, thr, {b: ["grasp_pink"]})[node] == {"skip": 1}
    assert load_groups(tmp_path / "none" / "items.jsonl") == {}


def test_judge_short_yaml_is_short2_and_matches_node_checks():
    cfg = yaml.safe_load(CFG.read_text())
    assert cfg["source"]["variant"] == "short2"
    # 실기 사진 채점(11.18, 13 에피소드)으로 만든 판 — 출처가 남고 홈 절대 경로가 박히지 않는다
    src = cfg["source"]
    assert src["split"] == "loeo" and len(src["items_sha256"]) == 64
    assert "vlm_plan_1118" in src["results"] and not src["results"].startswith("/")
    assert "two photographs" in cfg["prompt"] and "{question}" in cfg["prompt"]  # runner 가 [before, after] 를 보낸다
    want = {n: [{"slot": s, "ok": ok} for s, ok in c] for n, c in NODE_CHECKS.items()}
    assert cfg["node_checks"] == want
    used = {s.split(":")[0] for c in NODE_CHECKS.values() for s, _ in c}
    assert set(cfg["thresholds"]) == used
    assert {s for c in NODE_CHECKS.values() for s, _ in c} <= set(cfg["questions"])

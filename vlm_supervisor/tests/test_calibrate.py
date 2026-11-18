from pathlib import Path

import yaml

from vlm_supervisor.bench.calibrate import NODE_CHECKS

CFG = Path(__file__).resolve().parents[1] / "config" / "judge_short.yaml"


def test_released_only_on_place_nodes_and_pink_judged_next_node():
    # short2 는 released 를 "이번 동작에서 놓았나"로 읽는다 — 놓기 노드 밖에서 쓰면 안 된다
    for node, checks in NODE_CHECKS.items():
        if any(s.startswith("cup_released_in_holder") for s, _ in checks):
            assert node.startswith("place_"), node
    assert NODE_CHECKS["place_pink"] == []
    assert ("cup_in_holder_b:SRC_PINK", "yes") in NODE_CHECKS["home_r_after_pink"]
    assert NODE_CHECKS["scene"] == [("holders_any_cup", "no")]


def test_judge_short_yaml_is_short2_and_matches_node_checks():
    cfg = yaml.safe_load(CFG.read_text())
    assert cfg["source"]["variant"] == "short2"
    assert "two photographs" in cfg["prompt"] and "{question}" in cfg["prompt"]  # runner 가 [before, after] 를 보낸다
    want = {n: [{"slot": s, "ok": ok} for s, ok in c] for n, c in NODE_CHECKS.items()}
    assert cfg["node_checks"] == want
    used = {s.split(":")[0] for c in NODE_CHECKS.values() for s, _ in c}
    assert set(cfg["thresholds"]) == used
    assert {s for c in NODE_CHECKS.values() for s, _ in c} <= set(cfg["questions"])

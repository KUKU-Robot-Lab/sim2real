"""학습 가능 영역 지도 — 10.13 사용자: "Deploy 화면에 가능영역이 오버레이로 보였으면 좋겠어".

그날 실기 121432 는 컵을 오른팔 소환 박스 y 밖 9 cm(0.342, −0.394)에 놓았다. 화면이 그 자리를 '범위 밖'으로 말해야 한다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from s2r_console import spawn_map as SM

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
RIGHT = "deploy/policies/rh56f1/aglt/right_rg3s/rh_aglt_contract.json"
LEFT = "deploy/policies/rh56f1/aglt/left_rg3/rh_aglt_contract.json"
SLOTS = {"both": "pourfj_both", "right": "aglt_right", "left": "aglt_left"}
ARTS = {"aglt_right": RIGHT, "aglt_left": LEFT}
JS = (REPO / "deploy/s2r_console/s2r_console/web/console.js").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def isolated_contracts(tmp_path, monkeypatch):
    """Geometry tests do not require optional deployed policy bundles."""
    for side, relative in [('right', RIGHT), ('left', LEFT)]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        lo_y, hi_y = (-.3, -.1) if side == 'right' else (.1, .3)
        path.write_text(json.dumps({'goal_box_min': [.1, lo_y, .37], 'goal_box_max': [.4, hi_y, .51]}))
    monkeypatch.setitem(globals(), 'REPO', tmp_path)


def summ(x, y, z=0.29):
    return {"pos": [x, y, z], "age_s": 0.05}


def test_boxes_come_from_the_picked_contracts_of_each_arm():
    boxes = SM.boxes_from(ARTS, SLOTS, REPO)
    by = {b["side"]: b for b in boxes}
    assert set(by) == {"right", "left"}
    assert by["right"]["lo"] == pytest.approx([0.10, -0.30]) and by["right"]["hi"] == pytest.approx([0.40, -0.10])
    assert by["left"]["lo"] == pytest.approx([0.10, 0.10]) and by["left"]["hi"] == pytest.approx([0.40, 0.30])
    assert by["right"]["policy"] == "right_rg3s"


def test_a_contract_without_a_goal_box_or_a_missing_file_draws_nothing(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"schema": "pour_fj"}))
    assert SM.boxes_from({"aglt_right": "c.json", "aglt_left": "nope.json"}, SLOTS, tmp_path) == []
    assert SM.view([], {"/objects/a/pose": summ(0.3, -0.2)}, False) is None


def test_the_121432_cup_is_out_of_every_range_and_a_centred_cup_is_in():
    boxes = SM.boxes_from(ARTS, SLOTS, REPO)
    m = SM.view(boxes, {"/objects/source240b_blue/pose": summ(0.342, -0.394),
                        "/objects/shaker/pose": summ(0.25, -0.20),
                        "/objects/cup_l/pose": summ(0.25, 0.315)}, False)
    o = {x["name"]: x for x in m["objects"]}
    assert o["source240b_blue"]["tone"] == "bad" and "9.4 cm" in o["source240b_blue"]["say"]
    assert o["shaker"]["tone"] == "ok" and "오른팔" in o["shaker"]["say"]
    assert o["cup_l"]["tone"] == "warn" and "왼팔" in o["cup_l"]["say"]


def test_stale_or_missing_poses_are_not_drawn():
    boxes = SM.boxes_from(ARTS, SLOTS, REPO)
    assert SM.view(boxes, {"/objects/a/pose": summ(0.3, -0.2)}, True)["objects"] == []
    assert SM.view(boxes, {"/objects/a/pose": None}, False)["objects"] == []


def test_the_margin_is_the_one_the_deploy_start_refusal_uses():
    from policy_control import pour_fj_node as N
    assert SM.START_SPAWN_TOL_M == N.START_SPAWN_TOL_M


def test_the_fpp_panel_draws_the_server_map():
    body = re.search(r"function renderFpp[\s\S]*?\n}\n", JS).group(0)
    assert "spawnMap(f.map)" in body
    fn = re.search(r"function spawnMap[\s\S]*?\n}\n", JS).group(0)
    assert "o.tone" in fn and "m.boxes" in fn and "m.tol_m" in fn         # 판정은 서버가 — 화면은 그리기만

"""★11.02 eval 탭(실기 평가) — 평가 에피소드 목록(11.05 cocktail_1c_e4000 · grasp_s_right · grasp_s_left) · 회차 기록 CSV · /api/eval/rounds.

기록은 로봇에 아무것도 보내지 않는다. 서버가 round · time · policy id 를 채우고(policy id 는 에피소드 yaml 에서),
날짜마다 파일 하나(runs/s2r_console/eval/<YYYY-MM-DD>.csv)에 붙인다.
"""
from __future__ import annotations

import csv
import json
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from s2r_console import api
from s2r_console import eval_log as EV

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "deploy" / "s2r_console" / "s2r_console" / "web"

EPISODE = """
episode: {{name: {name}, version: 2}}
holders: {{HOLDER_0: 0, HOLDER_1: 1, HOLDER_2: 2}}
policies:
  rh_aglt_r: {{policy: rh56f1/aglt/right_approach_ap05, kind: aglt, side: right}}
  rh_place_r: {{policy: {place_r}, kind: place, side: right}}
  rh_place_l: {{policy: rh56f1/place/left_n06, kind: place, side: left}}
sequence:
  - {{id: go_home_start, type: trajectory, name: go_home}}
  - {{id: grasp_blue, type: policy, name: rh_aglt_r}}
  - {{id: place_blue, type: policy, name: rh_place_r, source_object: SRC_BLUE, target_holder: HOLDER_1}}
  - {{id: place_shaker, type: policy, name: rh_place_l, source_object: SHAKER, target_holder: HOLDER_0}}
  - {{id: done, type: terminal, result: SUCCESS}}
"""

GRASP = """
episode: {{name: grasp_s_{side}, version: 1}}
policies:
  rh_aglt_{s}: {{policy: rh56f1/aglt/{side}_approach_ap05, kind: aglt, side: {side}, student: rh56f1/aglt/{side}_student1104e}}
sequence:
  - {{id: go_home_start, type: trajectory, name: go_home}}
  - {{id: grasp_lift, type: policy, name: rh_aglt_{s}}}
  - {{id: done, type: terminal, result: SUCCESS}}
"""

DAY1 = datetime(2026, 11, 2, 13, 5, 0)
DAY2 = datetime(2026, 11, 3, 9, 0, 0)


def _write_episode(repo: Path, name: str, place_r: str) -> None:
    d = repo / "config" / "episodes"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(textwrap.dedent(EPISODE.format(name=name, place_r=place_r)), encoding="utf-8")


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    _write_episode(r, "cocktail_1c_e1000", "rh56f1/place/right_n07_e1000")   # 목록 밖 — 파일은 있어도 안 보여야
    _write_episode(r, "cocktail_1c_e4000", "rh56f1/place/right_n07_e4000")
    _write_episode(r, "grasp_b_right", "rh56f1/place/right_x")               # 목록 밖
    for side, s in (("right", "r"), ("left", "l")):
        (r / "config" / "episodes" / f"grasp_s_{side}.yaml").write_text(
            textwrap.dedent(GRASP.format(side=side, s=s)), encoding="utf-8")
    return r


def _csv(repo: Path, date: str) -> list[list[str]]:
    with (repo / "runs" / "s2r_console" / "eval" / f"{date}.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


# ── 에피소드 목록 ───────────────────────────────────────────────────────
def test_only_the_three_eval_episodes_are_listed_in_order(repo):
    assert [e["name"] for e in EV.episodes(repo)] == ["cocktail_1c_e4000", "grasp_s_right", "grasp_s_left"]


def test_cocktail_episode_keeps_place_policies_and_holders(repo):
    e = {e["name"]: e for e in EV.episodes(repo)}["cocktail_1c_e4000"]
    assert e["policy_r"] == "rh56f1/place/right_n07_e4000" and e["policy_l"] == "rh56f1/place/left_n06"
    assert e["holders"] == [{"node": "place_blue", "object": "SRC_BLUE", "holder": 1},
                            {"node": "place_shaker", "object": "SHAKER", "holder": 0}]
    assert e["stages"] == ["go_home_start", "grasp_blue", "place_blue", "place_shaker"]     # terminal 은 뺀다


def test_grasp_episode_reports_the_student_policy_of_its_hand(repo):
    eps = {e["name"]: e for e in EV.episodes(repo)}
    assert (eps["grasp_s_right"]["policy_r"], eps["grasp_s_right"]["policy_l"]) == ("rh56f1/aglt/right_student1104e", "")
    assert (eps["grasp_s_left"]["policy_r"], eps["grasp_s_left"]["policy_l"]) == ("", "rh56f1/aglt/left_student1104e")
    assert eps["grasp_s_right"]["stages"] == ["go_home_start", "grasp_lift"]


def test_view_marks_which_episodes_have_a_mission_stage(repo):
    v = EV.view(repo, ["episode_grasp_s_right", "head_home"])
    has = {e["name"]: e["has_stage"] for e in v["episodes"]}
    assert has == {"cocktail_1c_e4000": False, "grasp_s_right": True, "grasp_s_left": False}
    assert v["episodes"][1]["stage"] == "episode_grasp_s_right"


def test_no_episode_files_is_an_empty_list_not_an_error(tmp_path):
    v = EV.view(tmp_path / "nothing", [])
    assert v["episodes"] == [] and "error" not in v


def test_a_broken_episode_file_shows_its_error_and_keeps_the_list(repo):
    (repo / "config" / "episodes" / "grasp_s_left.yaml").write_text("episode: [", encoding="utf-8")
    eps = {e["name"]: e for e in EV.view(repo, [])["episodes"]}
    assert eps["grasp_s_left"]["error"] and not eps["cocktail_1c_e4000"]["error"]


def test_a_missing_listed_file_is_skipped_and_reappears_without_restart(repo):
    path = repo / "config" / "episodes" / "grasp_s_left.yaml"
    text = path.read_text(encoding="utf-8")
    path.unlink()
    assert [e["name"] for e in EV.episodes(repo)] == ["cocktail_1c_e4000", "grasp_s_right"]
    path.write_text(text, encoding="utf-8")
    assert "grasp_s_left" in {e["name"] for e in EV.episodes(repo)}


# ── 회차 기록 ───────────────────────────────────────────────────────────
def test_append_writes_header_once_and_reads_back(repo):
    EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": "성공", "note": "첫 회"}, now=DAY1)
    EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": "실패", "fail_stage": "place_blue",
                           "fail_type": "놓지 못함"}, now=DAY1)
    raw = _csv(repo, "2026-11-02")
    assert raw[0] == list(EV.COLUMNS) and sum(r == list(EV.COLUMNS) for r in raw) == 1 and len(raw) == 3
    rows = EV.read_rows(repo, "2026-11-02")
    assert [r["result"] for r in rows] == ["성공", "실패"]
    assert rows[1]["fail_stage"] == "place_blue" and rows[1]["fail_type"] == "놓지 못함"
    assert rows[0]["time"] == "13:05:00" and rows[0]["note"] == "첫 회"


def test_round_number_increases_per_day(repo):
    got = [EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": "중단"}, now=DAY1)["round"] for _ in range(3)]
    assert got == ["1", "2", "3"]


def test_each_date_has_its_own_file_and_its_own_round_count(repo):
    EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": "성공"}, now=DAY1)
    EV.append_round(repo, {"episode": "grasp_s_right", "result": "성공"}, now=DAY1)
    row = EV.append_round(repo, {"episode": "grasp_s_left", "result": "성공"}, now=DAY2)
    assert row["round"] == "1" and row["date"] == "2026-11-03"
    assert len(EV.read_rows(repo, "2026-11-02")) == 2 and len(EV.read_rows(repo, "2026-11-03")) == 1


def test_server_fills_policy_ids_from_the_episode_file_not_the_client(repo):
    row = EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": "성공",
                                 "policy_r": "가짜", "round": "99", "time": "00:00:00"}, now=DAY1)
    assert row["policy_r"] == "rh56f1/place/right_n07_e4000" and row["policy_l"] == "rh56f1/place/left_n06"
    assert row["round"] == "1" and row["time"] == "13:05:00"


@pytest.mark.parametrize("bad", ["", "ok", "SUCCESSFUL", None, 1])
def test_a_wrong_result_is_rejected_and_nothing_is_written(repo, bad):
    with pytest.raises(EV.EvalError):
        EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": bad}, now=DAY1)
    assert not (repo / "runs" / "s2r_console" / "eval" / "2026-11-02.csv").exists()


def test_runner_phases_are_accepted_as_results(repo):
    out = [EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": r}, now=DAY1)["result"]
           for r in ("SUCCESS", "FAILURE", "STOPPED")]
    assert out == ["성공", "실패", "중단"]


def test_fail_stage_must_belong_to_the_episode_and_is_dropped_on_success(repo):
    with pytest.raises(EV.EvalError):
        EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": "실패", "fail_stage": "tilt_pink"}, now=DAY1)
    row = EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": "성공", "fail_stage": "place_blue"}, now=DAY1)
    assert row["fail_stage"] == ""


@pytest.mark.parametrize("name", ["grasp_b_right", "cocktail_1c_e1000", "cocktail_1", "cocktail_none",
                                  "../cocktail_1c_e4000", "cocktail_1c_e4000/../x", 3])
def test_unknown_or_unsafe_episode_names_are_rejected(repo, name):
    with pytest.raises(EV.EvalError):
        EV.append_round(repo, {"episode": name, "result": "성공"}, now=DAY1)


def test_text_is_one_line_and_csv_quoting_survives_commas(repo):
    EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": "실패", "fail_type": "컵, 넘어짐",
                           "note": "줄\n바꿈 \"따옴표\""}, now=DAY1)
    (row,) = EV.read_rows(repo, "2026-11-02")
    assert row["fail_type"] == "컵, 넘어짐" and row["note"] == "줄 바꿈 \"따옴표\""


def test_bad_date_is_rejected(repo):
    for d in ("2026-11-2", "../x", "2026-11-02.csv"):
        with pytest.raises(EV.EvalError):
            EV.read_rows(repo, d)


def test_summary_counts_success_rate_and_failures_by_stage(repo):
    for result, stage in (("성공", ""), ("실패", "place_blue"), ("실패", "place_blue"), ("중단", "grasp_blue"), ("실패", "")):
        EV.append_round(repo, {"episode": "cocktail_1c_e4000", "result": result, "fail_stage": stage,
                               "fail_type": f"t_{stage}"}, now=DAY1)
    v = EV.rounds_view(repo, None, now=DAY1)
    s = v["summary"]
    assert (s["n"], s["success"], s["fail"], s["abort"]) == (5, 1, 3, 1) and s["rate"] == pytest.approx(0.2)
    assert s["by_stage"] == {"place_blue": 2, "grasp_blue": 1, "단계 없음": 1}
    assert v["date"] == v["today"] == "2026-11-02" and v["file"] == "runs/s2r_console/eval/2026-11-02.csv"
    assert v["fail_types"][0] == "t_" and set(v["fail_types"]) == {"t_", "t_grasp_blue", "t_place_blue"}


# ── API ─────────────────────────────────────────────────────────────────
class _Console:
    """api.handle 이 eval 경로에서 쓰는 면만 — lease 는 일부러 없다(기록은 lease 없이 된다)."""

    def __init__(self, repo: Path) -> None:
        self.repo, self.session = repo, None


def _call(c, method, path, body=None, query=None, headers=None):
    h = {"Host": "127.0.0.1:8091", "X-S2R-Console": "1", **(headers or {})}
    return api.handle(c, method, path, query=query or {}, body=json.dumps(body or {}).encode(), headers=h, client="t")


def test_api_post_without_lease_appends_and_get_reads_today(repo):
    c = _Console(repo)
    code, out = _call(c, "POST", "/api/eval/rounds", {"episode": "cocktail_1c_e4000", "result": "성공"})
    assert code == 200 and out["row"]["round"] == "1" and out["row"]["policy_r"] == "rh56f1/place/right_n07_e4000"
    code, out = _call(c, "GET", "/api/eval/rounds", query={"date": out["row"]["date"]})
    assert code == 200 and len(out["rows"]) == 1 and out["summary"]["success"] == 1


def test_api_rejects_a_wrong_result_with_400(repo):
    code, out = _call(_Console(repo), "POST", "/api/eval/rounds", {"episode": "cocktail_1c_e4000", "result": "maybe"})
    assert code == 400 and out["ok"] is False and "result" in out["error"]


def test_api_still_requires_the_console_header_for_writes(repo):
    code, _ = _call(_Console(repo), "POST", "/api/eval/rounds", {"episode": "cocktail_1c_e4000", "result": "성공"},
                    headers={"X-S2R-Console": ""})
    assert code == 403


def test_api_bad_date_query_is_400(repo):
    code, _ = _call(_Console(repo), "GET", "/api/eval/rounds", query={"date": "../../etc"})
    assert code == 400


# ── 화면 ────────────────────────────────────────────────────────────────
def test_the_eval_tab_sits_next_to_the_other_tabs():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="tab-button-eval"' in html and 'data-arg="eval"' in html and 'id="tab-eval"' in html
    for pid in ("ev-cards", "ev-run", "ev-form", "ev-table", "ev-summary", "ev-fail-types"):
        assert f'id="{pid}"' in html, pid
    for word in ("성공", "실패", "중단"):
        assert 'role="radio"' in html and word in html


_VM = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const els = new Map();
const el = id => { if (!els.has(id)) els.set(id, {id, innerHTML:'', value:'', hidden:true, dataset:{}, style:{}, options:[],
  classList:{toggle(){}}, addEventListener(){}, setAttribute(k, v){ this[k] = v; }, querySelectorAll(){ return []; },
  appendChild(){}, removeAttribute(){}, getAttribute(){ return ''; }}); return els.get(id); };
const calls = [];
const ctx = vm.createContext({console, Date, JSON, Set, Map, Number, String, Object, Array, Math, encodeURIComponent,
 document:{getElementById:el, addEventListener(){}, querySelectorAll(){return [];}, body:{classList:{toggle(){}}},
   createElement(){ return {className:'', innerHTML:'', remove(){}}; }},
 localStorage:{getItem(){return 'operator';}, setItem(){}}, sessionStorage:{getItem(){return 'token';}},
 window:{addEventListener(){}}, setInterval(){}, setTimeout(){},
 fetch:async(path, opts)=>{calls.push([path, opts && opts.body && JSON.parse(opts.body)]); return {ok:true, json:async()=>({ok:true, rows:[], summary:{n:0,success:0,by_stage:{}}, fail_types:[], date:'2026-11-02', today:'2026-11-02'})};},
 EventSource:function(){this.addEventListener=()=>{}}, requestAnimationFrame(){}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').replace('refresh().then(connect);',''),ctx);
"""


def test_finished_episode_prefills_result_and_fail_stage_once():
    script = _VM + r"""
const ev = {episodes:[{name:'cocktail_1c_e1000', stage:'episode_cocktail_1c_e1000', has_stage:true, policy_r:'r/e1000', policy_l:'l/n06',
  holders:[], stages:['go_home_start','grasp_blue','place_blue'], error:''}]};
const runner = {runner:'episode_runner', episode:'cocktail_1c_e1000', episode_id:'ep7', phase:'FAILURE', index:2, busy:false, stale:false,
  flow:{nodes:[{id:'go_home_start'},{id:'grasp_blue'},{id:'place_blue'}]}, last:{node:'place_blue', code:'policy_timeout'}};
const s = {profile:{domain_class:'fake'}, mission:{rows:[{id:'episode_cocktail_1c_e1000', can_run:true, reasons:[]}], lanes:[]},
  episode_runners:[runner], episode_runner:runner, eval:ev};
vm.runInContext(`S = {lease:{holder:'operator'}, session:${JSON.stringify(s)}}; renderEval(S.session);`, ctx);
assert.equal(vm.runInContext('evForm.result', ctx), '실패');
assert.equal(vm.runInContext('evForm.fail_stage', ctx), 'place_blue');
assert.equal(vm.runInContext('evForm.fail_type', ctx), 'policy_timeout');
assert.match(el('ev-cards').innerHTML, /cocktail_1c_e1000/);
assert.match(el('ev-cards').innerHTML, /r\/e1000/);
// 사람이 고친 값은 다음 그리기에서 덮어쓰지 않는다(같은 실행)
vm.runInContext(`evForm.result = '중단'; renderEval(S.session);`, ctx);
assert.equal(vm.runInContext('evForm.result', ctx), '중단');
(async () => {
  await vm.runInContext(`acts['ev-save']()`, ctx);
  const post = calls.find(c => c[0] === '/api/eval/rounds' && c[1]);
  assert.ok(post, 'POST 가 나갔다');
  assert.equal(post[1].episode, 'cocktail_1c_e1000');
  assert.equal(post[1].result, '중단');
  assert.equal(post[1].fail_stage, 'place_blue');
})().catch(e => { console.error(e); process.exitCode = 1; });
"""
    result = subprocess.run(["node", "-e", script, str(WEB / "console.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

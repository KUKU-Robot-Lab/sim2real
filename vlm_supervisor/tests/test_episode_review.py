"""episode_review — 가짜 VLM 서버로 score(형식 · 묶음 검사 합치기 · 판정불가 분기 · 재실행 안전 · bench 변환)와 combine 노드 역할 · page 종합 목록 · agree 를 본다."""
import base64
import json

import pytest
import yaml

from vlm_supervisor.bench import calibrate as C
from vlm_supervisor.bench import episode_review as R
from vlm_supervisor.bench import run_bench as B
from vlm_supervisor.tests.test_vlm_http import Server, ok

Q_HAND, Q_LIFT, Q_HOLD, Q_ALL = "Hand holds blue?", "Blue lifted?", "Any cup in holders?", "All cups in holders?"
HAND, LIFT = "cup_in_hand:right:SRC_BLUE", "cup_lifted:SRC_BLUE"
NODE_CHECKS = {
    "scene": [{"slot": "holders_any_cup", "ok": "no"}],                     # state 노드 검사: 묻지 않는다
    "grasp_blue": [{"slot": HAND, "ok": "yes"}, {"slot": LIFT, "ok": "yes"}],
    "to_tilt_start_r1": [{"slot": HAND, "ok": "yes"}],
    "go_home_end": [{"slot": "all_in_holders", "ok": "yes"}],
    "home": []}


@pytest.fixture
def srv():
    s = Server()
    yield s
    s.httpd.shutdown()
    s.httpd.server_close()


@pytest.fixture
def work(tmp_path):
    cfg = tmp_path / "judge.yaml"
    cfg.write_text(yaml.safe_dump({
        "prompt": B.SHORT2, "choices": B.CHOICES, "min_mass": 0.5,
        "thresholds": {"cup_in_hand": 0.9, "cup_lifted": 0.5, "holders_any_cup": 0.3, "all_in_holders": 0.5},
        "questions": {HAND: Q_HAND, LIFT: Q_LIFT, "holders_any_cup": Q_HOLD, "all_in_holders": Q_ALL},
        "node_checks": NODE_CHECKS}))
    for n in ("a0", "a1", "a2", "a3"):
        (tmp_path / f"{n}.jpg").write_bytes(b"\xff\xd8" + n.encode())

    def node(i, res="COMPLETED"):
        return {"id": i, "result": res, "reason": "r"}
    bounds = [
        {"id": "e/00_home", "seed": "e", "k": 0, "node": "home", "nodes": [node("home")], "jpg": "a0.jpg", "before": None},
        {"id": "e/01_scene", "seed": "e", "k": 1, "node": "scene", "nodes": [node("scene")], "jpg": "a1.jpg", "before": "a0.jpg"},
        {"id": "e/02_grasp", "seed": "e", "k": 2, "node": "grasp_blue",
         "nodes": [node("grasp_blue", "lift_guard_failed"), node("to_tilt_start_r1")], "jpg": "a2.jpg", "before": "a1.jpg"},
        {"id": "e/03_grasp_nobefore", "seed": "e", "k": 3, "node": "grasp_blue", "nodes": [node("grasp_blue")],
         "jpg": "a3.jpg", "before": None},
        {"id": "e/04_end", "seed": "e", "k": 4, "node": "go_home_end", "nodes": [node("go_home_end")],
         "jpg": "a3.jpg", "before": "a2.jpg", "end": True},
    ]
    (tmp_path / "bounds.jsonl").write_text("".join(json.dumps(b) + "\n" for b in bounds))
    return tmp_path, cfg


def reply_by_question(table):
    return lambda body: table[body["prompt"].rsplit("Question: ", 1)[1]]


def run(work, url, **kw):
    d, cfg = work
    return R.main(["score", str(d / "bounds.jsonl"), "--cfg", str(cfg), "--url", url, *sum(([f"--{k}", v] for k, v in kw.items()), [])])


def rows(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def comb(r):
    return {e["node"]: (e["final"], e["note"]) for e in r["combined"]}


def test_score_rows_merge_and_branches(srv, work):
    d, _ = work
    srv.reply = reply_by_question({Q_ALL: (200, ok(0.9)), Q_HAND: (200, ok(0.95)), Q_LIFT: (200, ok(0.8))})
    run(work, srv.url)
    out = rows(d / "review.jsonl")
    assert out["e/00_home"]["verdict"] == "undecided" and out["e/00_home"]["why"] == "검사 없음"
    assert out["e/01_scene"]["why"] == "검사 없음" and out["e/01_scene"]["final"] == "성공"   # state 검사는 묻지 않음
    nb = out["e/03_grasp_nobefore"]
    assert (nb["verdict"], nb["why"], len(nb["checks"])) == ("undecided", "before 없음", 2)
    assert nb["final"] == "성공"                              # 러너 성공 + VLM 못 물음 → 러너
    g = out["e/02_grasp"]
    # grasp_blue + to_tilt_start 묶음: to_tilt_start 는 state(★11.18) — 검사는 grasp_blue 것만, 결과는 러너
    assert [c["slot"] for c in g["checks"]] == [HAND, LIFT]
    assert g["checks"][0]["thr"] == 0.9 and g["checks"][0]["ans"] == "yes" and g["checks"][0]["pass"] is True
    assert g["verdict"] == "pass" and g["runner_ok"] is False
    assert comb(g) == {"grasp_blue": ("성공", "과정 경고: r"), "to_tilt_start_r1": ("성공", "")}
    assert g["final"] == "성공"
    end = out["e/04_end"]
    assert comb(end) == {"go_home_end": ("성공", ""), "all_in_holders": ("성공", "")}
    assert len(srv.bodies) == 3                             # grasp 묶음 2 + 끝 1, scene · before 없음 · 검사 없음은 묻지 않는다
    imgs = json.loads(srv.bodies[0][1])["images_b64"]
    assert [base64.b64decode(x) for x in imgs] == [b"\xff\xd8a1", b"\xff\xd8a2"]   # 전 · 후 jpg 바이트 그대로


def test_block_error_lowmass(srv, work):
    d, _ = work
    srv.reply = reply_by_question({Q_ALL: (200, ok(0.1)), Q_HAND: (500, "boom"), Q_LIFT: (200, ok(0.8))})
    run(work, srv.url)
    out = rows(d / "review.jsonl")
    assert out["e/04_end"]["verdict"] == "block" and out["e/04_end"]["final"] == "실패"
    g = out["e/02_grasp"]
    assert (g["verdict"], g["why"]) == ("undecided", "오류") and "error" in g["checks"][0]
    # 오류인데 러너 실패면 러너 실패 신호를 결과로 믿지 않고 사람 확인
    assert comb(g) == {"grasp_blue": ("사람 확인", "러너 실패 · VLM 판정 못 함"), "to_tilt_start_r1": ("성공", "")}
    assert g["final"] == "사람 확인"
    (d / "review.jsonl").unlink()
    srv.reply = reply_by_question({Q_ALL: (200, ok(0.1, mass=0.2)), Q_HAND: (200, ok(0.95)), Q_LIFT: (200, ok(0.2))})
    run(work, srv.url)
    out = rows(d / "review.jsonl")
    assert (out["e/04_end"]["verdict"], out["e/04_end"]["why"]) == ("undecided", "low_mass")
    assert out["e/04_end"]["final"] == "사람 확인"
    assert out["e/02_grasp"]["verdict"] == "block"           # 확정 답 하나가 기대와 다르면 막음
    assert comb(out["e/02_grasp"])["grasp_blue"] == ("실패", "")


def test_combine_roles_from_stored_checks():
    """옛 review 행(state 검사 포함)을 다시 계산: 공유 슬롯은 outcome 쪽, drop 은 떨어뜨림만, state 검사는 무시."""
    nc = {"place_blue": [{"slot": "cup_in_holder_b:SRC_BLUE", "ok": "yes"}],
          "home_r_after_blue": [{"slot": "cup_in_holder_b:SRC_BLUE", "ok": "yes"}],
          "tilt_blue": [{"slot": HAND, "ok": "yes"}],
          "scene": [{"slot": "holders_any_cup", "ok": "no"}]}

    def row(nodes, checks):
        return {"nodes": [{"id": i, "result": res, "reason": why} for i, res, why in nodes],
                "checks": [{"slot": s, "expect": "yes", "pass": p} for s, p in checks]}
    place = [("place_blue", "place_failed", "opened before latch " * 5), ("home_r_after_blue", "COMPLETED", "")]
    bc = R.bound_checks([{"id": i} for i, _, _ in place], nc)
    assert [c["slot"] for c in bc["place_blue"]] == ["cup_in_holder_b:SRC_BLUE"] and bc["home_r_after_blue"] == []
    r = R.combine(row(place, [("cup_in_holder_b:SRC_BLUE", True)]), nc)
    assert r["final"] == "성공" and r["combined"][0]["note"] == "과정 경고: " + ("opened before latch " * 5)[:60]
    assert r["combined"][1]["vlm"] == "비전 대상 아님"
    assert R.combine(row(place, [("cup_in_holder_b:SRC_BLUE", False)]), nc)["final"] == "실패"
    r = R.combine(row([("tilt_blue", "COMPLETED", "")], [(HAND, False)]), nc)
    assert comb(r) == {"tilt_blue": ("실패", "떨어뜨림 의심")} and r["final"] == "실패"
    r = R.combine(row([("tilt_blue", "tilt_failed", "x")], [(HAND, True)]), nc)
    assert comb(r) == {"tilt_blue": ("실패", "떨어뜨림 없음")}                       # drop 결과는 러너
    r = R.combine(row([("scene", "scene_failed", "")], [("holders_any_cup", True)]), nc)
    assert comb(r) == {"scene": ("실패", "")}                                        # state 는 러너만
    r = R.combine(row([("place_blue", "place_failed", "x")], []), nc)
    assert comb(r) == {"place_blue": ("사람 확인", "검사 누락: cup_in_holder_b:SRC_BLUE")} and r["final"] == "사람 확인"


def test_cfg_check_missing_from_stored_is_human_not_success():
    """설정엔 검사가 있는데 저장 checks 에 없으면(설정이 바뀐 뒤 다시 계산) 러너 성공이어도 사람 확인. 설정에 검사가 원래 없으면 러너대로."""
    nc = {"grasp_blue": [{"slot": HAND, "ok": "yes"}], "place_pink": []}
    r = R.combine({"nodes": [{"id": "grasp_blue", "result": "COMPLETED", "reason": ""}], "checks": []}, nc)
    assert comb(r) == {"grasp_blue": ("사람 확인", f"검사 누락: {HAND}")} and r["final"] == "사람 확인"
    r = R.combine({"nodes": [{"id": "place_pink", "result": "COMPLETED", "reason": ""}], "checks": []}, nc)
    assert comb(r) == {"place_pink": ("성공", "VLM 판정 못 함")} and r["combined"][0]["vlm"] == "검사 없음"


def test_episode_slot_missing_from_stored_is_human():
    """에피소드 노드에 all_in_holders 검사가 설정돼 있는데 저장 checks 에 없으면 항목이 사라져 성공이 되면 안 된다."""
    nc = {"go_home_end": [{"slot": R.EPISODE_SLOT, "ok": "yes"}]}
    r = R.combine({"nodes": [{"id": "go_home_end", "result": "COMPLETED", "reason": ""}], "checks": []}, nc)
    assert comb(r)[R.EPISODE_SLOT] == ("사람 확인", "VLM 검사 없음") and r["final"] == "사람 확인"


def test_place_slot_moves_from_state_node():
    """place 에 검사가 없으면 같은 묶음 state 노드의 cup_in_holder_b:<물체> 를 그 물체의 place 로 옮긴다."""
    nc = {"place_pink": [], "place_shaker": [],
          "home_r_after_pink": [{"slot": "cup_in_holder_b:SRC_PINK", "ok": "yes"}, {"slot": "cup_in_hand:left:SHAKER", "ok": "yes"}],
          "home_l_after_shaker": [{"slot": "cup_in_holder_b:SHAKER", "ok": "yes"}],
          "home_x": [{"slot": "cup_in_holder_b:OTHER", "ok": "yes"}]}
    ids = lambda *xs: [{"id": x} for x in xs]
    bc = R.bound_checks(ids("place_pink", "place_shaker", "home_r_after_pink", "home_l_after_shaker", "home_x"), nc)
    assert [c["slot"] for c in bc["place_pink"]] == ["cup_in_holder_b:SRC_PINK"]
    assert [c["slot"] for c in bc["place_shaker"]] == ["cup_in_holder_b:SHAKER"]
    assert bc["home_r_after_pink"] == [] and bc["home_x"] == []                    # 물체 대응 없고 place 가 둘 → 버림
    bc = R.bound_checks(ids("place_pink", "home_x"), nc)                          # 다른 물체 place 로 대신하지 않는다
    assert bc["place_pink"] == [] and [c["slot"] for c in bc[R.UNPLACED]] == ["cup_in_holder_b:OTHER"]
    assert R.bound_checks(ids("home_r_after_pink"), nc)["home_r_after_pink"] == []  # place 없으면 옮기지 않음
    cfg = {"questions": {"cup_in_holder_b:SRC_PINK": "Q"}, "node_checks": nc}
    b = {"id": "e/8", "seed": "e", "k": 8, "node": "home_r_after_pink", "jpg": "a.jpg", "before": "b.jpg",
         "nodes": [{"id": "place_pink", "result": "place_failed", "reason": "opened before latch"},
                   {"id": "home_r_after_pink", "result": "COMPLETED"}]}
    r = R.review_row(b, cfg, lambda b, qs: [{"scores": {"yes": 0.9, "no": 0.1}, "p_yes": 0.9, "thr": 0.5}] * len(qs))
    assert [c["slot"] for c in r["checks"]] == ["cup_in_holder_b:SRC_PINK"]     # 옮긴 슬롯은 state 검사여도 묻는다
    assert comb(r) == {"place_pink": ("성공", "과정 경고: opened before latch"), "home_r_after_pink": ("성공", "")}


def test_unmatched_place_slot_is_human_not_dropped():
    """설정 어디에도 그 물체의 place 노드가 없으면(이름 불일치) 검사를 버리지 않고 에피소드 수준 사람 확인으로 올린다.
    그 물체의 place 가 다른 묶음에 있으면(이미 놓음) 전처럼 state 검사라 뺀다."""
    nc = {"place_blue": [], "home_x": [{"slot": "cup_in_holder_b:OTHER", "ok": "yes"}],
          "home_r_after_blue": [{"slot": "cup_in_holder_b:SRC_BLUE", "ok": "yes"}]}
    nodes = [{"id": "place_blue", "result": "COMPLETED", "reason": ""}, {"id": "home_x", "result": "COMPLETED", "reason": ""}]
    r = R.combine({"nodes": nodes, "checks": []}, nc)
    assert comb(r)["cup_in_holder_b:OTHER"] == ("사람 확인", "놓기 노드 못 찾음") and r["final"] == "사람 확인"
    r = R.combine({"nodes": [{"id": "home_r_after_blue", "result": "COMPLETED", "reason": ""}], "checks": []}, nc)
    assert r["final"] == "성공"


def test_agree_truth_and_counts():
    nc = {"grasp_blue": [{"slot": HAND, "ok": "yes"}]}

    def row(i, res, p):
        return {"id": i, "runner_ok": res == "COMPLETED", "nodes": [{"id": "grasp_blue", "result": res, "reason": ""}],
                "checks": [{"slot": HAND, "expect": "yes", "pass": p}] if p is not None else []}
    rs = [row("a", "lift_guard_failed", True), row("b", "COMPLETED", False), row("c", "COMPLETED", None),
          {"id": "s", "runner_ok": True, "nodes": [{"id": "scene", "result": "COMPLETED", "reason": ""}], "checks": []}]
    labels = [{"id": "a", "mark": "vlm", "runner_ok": False, "vlm": "pass"},
              {"id": "b", "mark": "runner", "runner_ok": True, "vlm": "block"},
              {"id": "c", "mark": "vlm", "runner_ok": True, "vlm": "undecided"},     # 정답 모름
              {"id": "s", "mark": "both_wrong", "runner_ok": True, "vlm": "undecided"},
              {"id": "z", "mark": "runner", "runner_ok": True, "vlm": "pass"}]
    a = R.agree(rs, labels + [labels[1]], nc)                                  # 같은 id 다시 표시는 한 번만 센다
    assert a["table"]["outcome"] == {"n": 2, "runner_bad": 1, "runner_ok": 1, "comb_ok": 1, "comb_bad": 1}
    assert a["wrong"] == {"runner": ["a"], "combined": ["b"]}
    assert (a["both_wrong"], a["no_truth"], a["missing"]) == (["s"], 2, ["z"])
    assert "| outcome | 2 | 1 | 1 | 1 | 1 | 0 |" in R.agree_text(a)


def test_rerun_skips_existing(srv, work):
    d, _ = work
    p = d / "review.jsonl"
    p.write_text(json.dumps({"id": "e/02_grasp", "verdict": "pass", "keep": 1}) + "\n")
    run(work, srv.url)
    assert len(srv.bodies) == 1                              # grasp 는 이미 있어 끝 경계만 묻는다
    before = p.read_text()
    assert json.loads(before.splitlines()[0])["keep"] == 1
    run(work, srv.url)
    assert len(srv.bodies) == 1 and p.read_text() == before  # 두 번째는 아무것도 안 묻고 안 쓴다


def test_from_bench_matches_prompt(work, tmp_path):
    d, cfg = work
    res = tmp_path / "results.jsonl"
    lines = [{"meta": {}}] + [
        {"variant": "short2", "id": f"e/02_grasp/{s}", "prompt": B.SHORT2.format(question=q), "resp": ok(p)}
        for s, q, p in [(HAND, Q_HAND, 0.95), (LIFT, Q_LIFT, 0.3)]]
    lines.append({"variant": "short1", "id": "e/04_end/all_in_holders", "prompt": B.SHORT2.format(question=Q_ALL), "resp": ok(0.9)})
    res.write_text("".join(json.dumps(x) + "\n" for x in lines))
    R.main(["score", str(d / "bounds.jsonl"), "--cfg", str(cfg), "--from-bench", str(res), "--out", str(tmp_path / "rv.jsonl")])
    out = rows(tmp_path / "rv.jsonl")
    assert out["e/02_grasp"]["verdict"] == "block" and out["e/02_grasp"]["checks"][1]["p_yes"] == 0.3
    assert out["e/04_end"]["checks"][0]["error"] == "bench 결과에 없음"     # short1 행은 쓰지 않는다


def test_page_final_lists(work, tmp_path):
    d, _ = work

    def r(i, res, checks, nid="grasp_blue", seed="e"):
        return {"id": i, "seed": seed, "k": int(i[-1]), "node": nid, "runner_ok": res == "COMPLETED",
                "verdict": R.verdict(checks)[0] if checks else "undecided", "why": "",
                "nodes": [{"id": nid, "result": res, "reason": "<b>x</b>"}], "jpg": "a1.jpg", "before": "a0.jpg",
                "checks": [{**c, "q": "q", "expect": "yes"} for c in checks]}
    yes = {"slot": HAND, "pass": True, "ans": "yes"}
    data = [r("e/1", "lift_guard_failed", [yes, {**yes, "slot": LIFT}]), r("e/2", "COMPLETED", [{**yes, "pass": False, "ans": "no"}]),
            r("e/3", "COMPLETED", [{"slot": "holders_any_cup", "pass": False, "ans": "yes"}], "scene"),
            r("f/5", "lift_guard_failed", [{"slot": HAND, "pass": None, "error": "boom"}], seed="f")]
    page = tmp_path / "out" / "index.html"
    page.parent.mkdir()
    summ = R.page(data, d, page, NODE_CHECKS)
    assert (summ["final_fail"], summ["final_human"], summ["process_warn"]) == (["e/2"], ["f/5"], ["e/1"])
    assert summ["final"] == {"성공": 2, "실패": 1, "사람 확인": 1}
    assert (summ["episodes"], summ["pass"], summ["block"], summ["undecided"], summ["runner_fail"]) == (2, 1, 2, 1, 2)
    html = page.read_text()
    assert '<meta charset="utf-8">' in html and 'href="#e/1"' in html and "&lt;b&gt;x" in html
    assert "상태 노드 검사라 종합에서 뺐다" in html and "과정 경고: &lt;b&gt;x&lt;/b&gt;" in html
    assert 'src="../a1.jpg"' in html and "://" not in html   # 외부 CDN · 절대 URL 없음


def test_score_reasks_error_rows_and_page_uses_last(tmp_path):
    """서버 오류로 판정불가가 된 경계는 재실행 때 다시 묻고, 같은 id 는 마지막 줄을 쓴다."""
    from vlm_supervisor.bench.episode_review import latest, read_jsonl, score

    bounds = tmp_path / "bounds.jsonl"
    out = tmp_path / "review.jsonl"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("questions: {}\nnode_checks: {}\n", encoding="utf-8")
    bounds.write_text("\n".join(json.dumps({"id": i, "seed": "s", "k": k, "node": "n", "nodes": [], "jpg": "x.jpg"})
                                for k, i in enumerate(["s/a", "s/b"])) + "\n", encoding="utf-8")
    out.write_text(json.dumps({"id": "s/a", "why": "오류", "verdict": "undecided"}) + "\n"
                   + json.dumps({"id": "s/b", "why": "검사 없음", "verdict": "undecided"}) + "\n", encoding="utf-8")
    new, done = score(bounds, out, cfg, None)
    assert (new, done) == (1, 1)
    rows = latest(read_jsonl(out))
    assert [r["id"] for r in rows] == ["s/a", "s/b"]
    assert rows[0]["why"] == "검사 없음"


E2E = R.CFG.parents[2] / "config" / "episodes" / "cocktail_e2e.yaml"
# 노드 → 역할(★11.18 사용자: 이동 · 상태 노드는 사진으로 판정하지 않는다). 기본 에피소드에 노드가 생기면 여기와 calibrate.NODE_CHECKS 를 같이 고친다.
E2E_ROLES = {
    "go_home_start": "state", "scene": "state",
    "grasp_blue": "outcome", "to_tilt_start_r1": "state", "grasp_shaker": "outcome", "to_tilt_start_l": "state",
    "tilt_blue": "drop", "place_blue": "outcome", "home_r_after_blue": "state",
    "grasp_pink": "outcome", "to_tilt_start_r2": "state", "tilt_pink": "drop", "place_pink": "outcome",
    "home_r_after_pink": "state", "mix_shaker": "state", "place_shaker": "outcome", "home_l_after_shaker": "state",
    "go_home_end": "state"}


def test_judge_short_covers_every_cocktail_e2e_node():
    """yaml 노드 전체(lanes 는 줄 항목, terminal 은 경계가 아니라 뺀다)가 judge_short node_checks 와 역할 표에 있다."""
    seq = yaml.safe_load(E2E.read_text(encoding="utf-8"))["sequence"]
    nc = yaml.safe_load(R.CFG.read_text(encoding="utf-8"))["node_checks"]
    lane_of = {}
    for n in seq:
        if n["type"] == "lanes":
            lane_of.update({m["id"]: lane["side"] for lane in n["lanes"] for m in lane["nodes"]})
        elif n["type"] != "terminal":
            lane_of[n["id"]] = None
    assert set(lane_of) == set(E2E_ROLES), sorted(set(lane_of) ^ set(E2E_ROLES))
    assert not set(lane_of) - set(nc), f"node_checks 에 없는 노드: {sorted(set(lane_of) - set(nc))}"
    assert {n: R.role(n) for n in lane_of} == E2E_ROLES
    assert set(C.NODE_CHECKS) == set(lane_of) | {"final"}      # calibrate 가 만드는 키도 같은 노드 목록
    with pytest.raises(ValueError, match="역할을 모르는"):
        R.role("pour_extra")
    order = list(lane_of)
    for nid, side in lane_of.items():
        slots = [c["slot"] for c in nc[nid]]
        if side:                                   # lanes: 줄마다 그 팔의 손 검사만
            assert all(s.split(":")[1] == side for s in slots if s.startswith("cup_in_hand:")), (nid, slots)
        asked = R.bound_checks([{"id": nid}], nc)[nid]
        if E2E_ROLES[nid] == "state":              # state 는 묻지 않는다 — 에피소드 결과 슬롯만 예외
            assert [c["slot"] for c in asked] == ([R.EPISODE_SLOT] if nid in R.EPISODE_NODES else []), nid
        elif E2E_ROLES[nid] == "outcome" and not asked:   # 끝 사진이 가려진 놓기(place_pink)는 다음 노드 검사를 받는다
            nxt = order[order.index(nid) + 1]
            assert R.bound_checks([{"id": nid}, {"id": nxt}], nc)[nid], nid
        else:
            assert asked, nid


def test_merge_checks_view():
    nodes = [{"id": "grasp_blue"}]
    assert R.merge_checks(nodes, {"grasp_blue": [{"slot": HAND, "ok": "yes", "view": "any"}]}) == [{"slot": HAND, "expect": "yes"}]
    with pytest.raises(ValueError, match="view global"):
        R.merge_checks(nodes, {"grasp_blue": [{"slot": HAND, "ok": "yes", "view": "global"}]})

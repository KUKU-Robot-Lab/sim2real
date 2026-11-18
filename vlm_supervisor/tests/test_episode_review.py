"""episode_review — 가짜 VLM 서버로 score(형식 · 묶음 검사 합치기 · 판정불가 분기 · 재실행 안전 · bench 변환)와 page 불일치 목록을 본다."""
import base64
import json

import pytest
import yaml

from vlm_supervisor.bench import episode_review as R
from vlm_supervisor.bench import run_bench as B
from vlm_supervisor.tests.test_vlm_http import Server, ok

Q_HAND, Q_LIFT, Q_HOLD = "Hand holds blue?", "Blue lifted?", "Any cup in holders?"


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
        "thresholds": {"cup_in_hand": 0.9, "cup_lifted": 0.5, "holders_any_cup": 0.3},
        "questions": {"cup_in_hand:right:SRC_BLUE": Q_HAND, "cup_lifted:SRC_BLUE": Q_LIFT, "holders_any_cup": Q_HOLD},
        "node_checks": {
            "scene": [{"slot": "holders_any_cup", "ok": "no"}],
            "grasp_blue": [{"slot": "cup_in_hand:right:SRC_BLUE", "ok": "yes"}, {"slot": "cup_lifted:SRC_BLUE", "ok": "yes"}],
            "to_tilt": [{"slot": "cup_in_hand:right:SRC_BLUE", "ok": "yes"}],
            "home": []}}))
    for n in ("a0", "a1", "a2", "a3"):
        (tmp_path / f"{n}.jpg").write_bytes(b"\xff\xd8" + n.encode())

    def node(i, res="COMPLETED"):
        return {"id": i, "result": res, "reason": "r"}
    bounds = [
        {"id": "e/00_home", "seed": "e", "k": 0, "node": "home", "nodes": [node("home")], "jpg": "a0.jpg", "before": None},
        {"id": "e/01_scene", "seed": "e", "k": 1, "node": "scene", "nodes": [node("scene")], "jpg": "a1.jpg", "before": "a0.jpg"},
        {"id": "e/02_grasp", "seed": "e", "k": 2, "node": "grasp_blue",
         "nodes": [node("grasp_blue", "lift_guard_failed"), node("to_tilt")], "jpg": "a2.jpg", "before": "a1.jpg"},
        {"id": "e/03_grasp_nobefore", "seed": "e", "k": 3, "node": "grasp_blue", "nodes": [node("grasp_blue")],
         "jpg": "a3.jpg", "before": None},
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


def test_score_rows_merge_and_branches(srv, work):
    d, _ = work
    srv.reply = reply_by_question({Q_HOLD: (200, ok(0.1)), Q_HAND: (200, ok(0.95)), Q_LIFT: (200, ok(0.8))})
    run(work, srv.url)
    out = rows(d / "review.jsonl")
    assert out["e/00_home"]["verdict"] == "undecided" and out["e/00_home"]["why"] == "검사 없음"
    nb = out["e/03_grasp_nobefore"]
    assert (nb["verdict"], nb["why"], len(nb["checks"])) == ("undecided", "before 없음", 2)
    g = out["e/02_grasp"]
    # grasp_blue + to_tilt 묶음: cup_in_hand 는 한 번만 묻는다
    assert [c["slot"] for c in g["checks"]] == ["cup_in_hand:right:SRC_BLUE", "cup_lifted:SRC_BLUE"]
    assert g["checks"][0]["thr"] == 0.9 and g["checks"][0]["ans"] == "yes" and g["checks"][0]["pass"] is True
    assert g["verdict"] == "pass" and g["runner_ok"] is False
    assert [n["result"] for n in g["nodes"]] == ["lift_guard_failed", "COMPLETED"]
    assert out["e/01_scene"]["verdict"] == "pass" and out["e/01_scene"]["checks"][0]["ans"] == "no"
    assert len(srv.bodies) == 3                             # scene 1 + grasp 묶음 2, before 없음 · 검사 없음은 묻지 않는다
    imgs = json.loads(srv.bodies[1][1])["images_b64"]
    assert [base64.b64decode(x) for x in imgs] == [b"\xff\xd8a1", b"\xff\xd8a2"]   # 전 · 후 jpg 바이트 그대로
    assert R.mismatch(g) and not R.mismatch(out["e/01_scene"])


def test_block_error_lowmass(srv, work):
    d, _ = work
    srv.reply = reply_by_question({Q_HOLD: (200, ok(0.9)), Q_HAND: (500, "boom"), Q_LIFT: (200, ok(0.8))})
    run(work, srv.url)
    out = rows(d / "review.jsonl")
    assert out["e/01_scene"]["verdict"] == "block"
    g = out["e/02_grasp"]
    assert (g["verdict"], g["why"]) == ("undecided", "오류") and "error" in g["checks"][0]
    (d / "review.jsonl").unlink()
    srv.reply = reply_by_question({Q_HOLD: (200, ok(0.1, mass=0.2)), Q_HAND: (200, ok(0.95)), Q_LIFT: (200, ok(0.2))})
    run(work, srv.url)
    out = rows(d / "review.jsonl")
    assert (out["e/01_scene"]["verdict"], out["e/01_scene"]["why"]) == ("undecided", "low_mass")
    assert out["e/02_grasp"]["verdict"] == "block"           # 확정 답 하나가 기대와 다르면 막음


def test_rerun_skips_existing(srv, work):
    d, _ = work
    p = d / "review.jsonl"
    p.write_text(json.dumps({"id": "e/01_scene", "verdict": "pass", "keep": 1}) + "\n")
    run(work, srv.url)
    assert len(srv.bodies) == 2                              # scene 은 이미 있어 묻지 않는다
    before = p.read_text()
    assert json.loads(before.splitlines()[0])["keep"] == 1
    run(work, srv.url)
    assert len(srv.bodies) == 2 and p.read_text() == before  # 두 번째는 아무것도 안 묻고 안 쓴다


def test_from_bench_matches_prompt(work, tmp_path):
    d, cfg = work
    res = tmp_path / "results.jsonl"
    lines = [{"meta": {}}] + [
        {"variant": "short2", "id": f"e/02_grasp/{s}", "prompt": B.SHORT2.format(question=q), "resp": ok(p)}
        for s, q, p in [("cup_in_hand:right:SRC_BLUE", Q_HAND, 0.95), ("cup_lifted:SRC_BLUE", Q_LIFT, 0.3)]]
    lines.append({"variant": "short1", "id": "e/01_scene/holders_any_cup", "prompt": B.SHORT2.format(question=Q_HOLD), "resp": ok(0.1)})
    res.write_text("".join(json.dumps(x) + "\n" for x in lines))
    R.main(["score", str(d / "bounds.jsonl"), "--cfg", str(cfg), "--from-bench", str(res), "--out", str(tmp_path / "rv.jsonl")])
    out = rows(tmp_path / "rv.jsonl")
    assert out["e/02_grasp"]["verdict"] == "block" and out["e/02_grasp"]["checks"][1]["p_yes"] == 0.3
    assert out["e/01_scene"]["checks"][0]["error"] == "bench 결과에 없음"     # short1 행은 쓰지 않는다


def test_page_mismatch_list(work, tmp_path):
    d, _ = work

    def r(i, ok_, v, seed="e"):
        return {"id": i, "seed": seed, "k": int(i[-1]), "node": "n", "runner_ok": ok_, "verdict": v, "why": "",
                "nodes": [{"id": "n", "result": "COMPLETED" if ok_ else "place_failed", "reason": "<b>x</b>"}],
                "jpg": "a1.jpg", "before": "a0.jpg", "checks": []}
    data = [r("e/1", False, "pass"), r("e/2", True, "block"), r("e/3", True, "pass"), r("e/4", False, "block"),
            r("f/5", False, "undecided", "f")]
    page = tmp_path / "out" / "index.html"
    page.parent.mkdir()
    summ = R.page(data, d, page)
    assert summ["mismatch"] == ["e/1", "e/2"]
    assert (summ["episodes"], summ["pass"], summ["block"], summ["undecided"], summ["runner_fail"]) == (2, 2, 2, 1, 3)
    html = page.read_text()
    assert '<meta charset="utf-8">' in html and 'href="#e/1"' in html and "&lt;b&gt;x" in html
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

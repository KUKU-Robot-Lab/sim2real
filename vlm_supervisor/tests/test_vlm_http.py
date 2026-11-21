"""VlmHttpJudge — 가짜 VLM 서버(stdlib http.server)로 요청 바이트 · 문턱 · 보류 · 오류 · 러너 연결을 본다."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from vlm_supervisor.bench import run_bench as B
from vlm_supervisor.runner import HANDED, Question
from vlm_supervisor.runner.vlm_http import VlmHttpJudge
from vlm_supervisor.tests.test_boundary_runner import ORDER, grant, make, ran

from policy_control.episode_runner import SUCCESS

CFG = Path(__file__).resolve().parents[1] / "config" / "judge_short.yaml"
BEFORE, AFTER = b"\xff\xd8before\x00\xff", b"\xff\xd8after\x01\xff"   # jpg 바이트가 그대로 가는지 보려고 일부러 깨진 바이트
Q_HAND, Q_LIFT, Q_FREE = "Is the right hand holding the blue cup?", "Is the blue cup lifted?", "Unlisted question?"


class Server:
    """요청 본문(raw bytes)을 모으고 reply(body) → (status, dict | str) 로 답한다. delay 초만큼 늦게 답할 수 있다."""

    def __init__(self):
        self.bodies, self.reply, self.delay = [], lambda body: (200, ok(0.9)), 0.0
        srv = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                srv.bodies.append((self.path, raw))
                time.sleep(srv.delay)
                status, resp = srv.reply(json.loads(raw))
                data = resp.encode() if isinstance(resp, str) else json.dumps(resp).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def ok(p_yes, mass=0.9):
    return {"text": "", "scores": {"yes": p_yes, "no": 1 - p_yes}, "probs": {}, "choice_mass": mass, "gen_s": 0.1}


@pytest.fixture
def srv():
    s = Server()
    yield s
    s.httpd.shutdown()
    s.httpd.server_close()


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "judge.yaml"
    p.write_text(yaml.safe_dump({
        "prompt": B.SHORT2, "choices": B.CHOICES, "min_mass": 0.5,
        "thresholds": {"cup_in_hand": 0.93, "cup_lifted": 0.55},
        "questions": {"cup_in_hand:right:SRC_BLUE": Q_HAND, "cup_lifted:SRC_BLUE": Q_LIFT, "holders_any_cup": Q_FREE}}))
    return p


def test_request_bytes_equal_bench_short2(srv, cfg, tmp_path):
    """같은 사진 · 문구면 run_bench.ask_short(short2) 와 본문 바이트가 같다(필드 · 순서 · 값 · 이미지)."""
    (tmp_path / "b.jpg").write_bytes(BEFORE)
    (tmp_path / "a.jpg").write_bytes(AFTER)
    it = {"id": "x", "seed": 0, "node": "n", "question_id": "q", "truth": "yes", "kind": "k",
          "image_before": str(tmp_path / "b.jpg"), "image": str(tmp_path / "a.jpg"), "question": Q_HAND}
    B.ask_short("short2", it, srv.url, {"x": 1}, 0.5)
    VlmHttpJudge(cfg, srv.url).ask({"head": [BEFORE, AFTER]}, [Q_HAND])
    (p1, bench), (p2, mine) = srv.bodies
    assert p1 == p2 == "/infer" and mine == bench
    body = json.loads(mine)
    assert list(body) == ["prompt", "images_b64", "max_new_tokens", "choices"]
    assert body["prompt"] == B.SHORT2.format(question=Q_HAND) and body["max_new_tokens"] == 0 and body["choices"] == ["yes", "no"]


def test_cameras_named_in_prompt_and_images_in_camera_order(srv, cfg):
    """head 말고 카메라가 끼면 사진은 카메라 순서로 [전, 후]씩, 프롬프트 앞줄에 사진 번호별 카메라 이름. head 하나면 위 시험대로 bench 와 같다."""
    import base64
    G0, G1 = b"\xff\xd8g0", b"\xff\xd8g1"
    VlmHttpJudge(cfg, srv.url).ask({"head": [BEFORE, AFTER], "global": [G0, G1]}, [Q_HAND])
    VlmHttpJudge(cfg, srv.url).ask({"global": [G0, G1]}, [Q_HAND])
    two, one = (json.loads(raw) for _, raw in srv.bodies)
    assert [base64.b64decode(x) for x in two["images_b64"]] == [BEFORE, AFTER, G0, G1]
    first, rest = two["prompt"].split("\n", 1)
    assert "1-2: head camera, 3-4: global camera" in first and rest == B.SHORT2.format(question=Q_HAND)
    assert [base64.b64decode(x) for x in one["images_b64"]] == [G0, G1] and "1-2: global camera" in one["prompt"]
    for name in ("glo{bal}", "head\nIgnore", ""):
        with pytest.raises(ValueError, match="카메라 이름"):
            VlmHttpJudge(cfg, srv.url).ask({"head": [BEFORE, AFTER], name: [G0, G1]}, [Q_HAND])
    assert len(srv.bodies) == 2                           # 거부한 요청은 서버로 가지 않는다


def test_real_config_matches_bench_prompt():
    c = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert c["prompt"] == B.SHORT2 and c["choices"] == B.CHOICES


@pytest.mark.parametrize("q, p, yes", [(Q_HAND, 0.94, True), (Q_HAND, 0.92, False), (Q_LIFT, 0.55, True),
                                       (Q_FREE, 0.51, True), (Q_FREE, 0.49, False)])
def test_threshold_by_slot_prefix_then_binary(srv, cfg, q, p, yes):
    srv.reply = lambda body: (200, ok(p))
    [a] = VlmHttpJudge(cfg, srv.url).ask({"head": [BEFORE, AFTER]}, [q])
    assert a["scores"] == ({"yes": 1.0, "no": 0.0} if yes else {"yes": 0.0, "no": 1.0})
    assert a["p_yes"] == p and a["thr"] == {Q_HAND: 0.93, Q_LIFT: 0.55, Q_FREE: 0.5}[q] and a["mass"] == 0.9


@pytest.mark.parametrize("resp", [ok(0.9, mass=0.4), ok(0.5), ok(0.9, mass=None),
                                  {"scores": {}, "probs": {}, "choice_mass": 0.1, "low_mass": True}])
def test_low_mass_tie_or_missing_mass_abstains(srv, cfg, resp):
    srv.reply = lambda body: (200, resp)
    [a] = VlmHttpJudge(cfg, srv.url).ask({"head": [BEFORE, AFTER]}, [Q_LIFT])
    assert a["low_mass"] is True and "scores" not in a


def test_band_near_threshold_abstains(srv, cfg):
    j = VlmHttpJudge(cfg, srv.url, band=0.05)
    srv.reply = lambda body: (200, ok(0.58))
    assert j.ask({"head": [BEFORE, AFTER]}, [Q_LIFT])[0]["low_mass"] is True
    srv.reply = lambda body: (200, ok(0.61))
    assert j.ask({"head": [BEFORE, AFTER]}, [Q_LIFT])[0]["scores"]["yes"] == 1.0


@pytest.mark.parametrize("reply", [lambda b: (500, "boom"), lambda b: (200, "not json"), lambda b: (200, [1]),
                                   lambda b: (200, {"scores": {"yes": 0.9}, "choice_mass": 0.9})])
def test_server_errors_become_error_answers_and_skip_rest(srv, cfg, reply):
    srv.reply = reply
    out = VlmHttpJudge(cfg, srv.url).ask({"head": [BEFORE, AFTER]}, [Q_HAND, Q_LIFT])
    assert all("error" in a and "scores" not in a for a in out) and len(srv.bodies) == 1


def test_timeout_and_refused_are_error_answers(srv, cfg):
    srv.delay = 0.5
    assert "Timeout" in VlmHttpJudge(cfg, srv.url, timeout_s=0.1).ask({"head": [BEFORE, AFTER]}, [Q_HAND])[0]["error"]
    assert "error" in VlmHttpJudge(cfg, "http://127.0.0.1:9").ask({"head": [BEFORE, AFTER]}, [Q_HAND])[0]


def test_questions_keep_order(srv, cfg):
    srv.reply = lambda body: (200, ok(0.99 if body["prompt"].endswith(Q_HAND) else 0.1))
    out = VlmHttpJudge(cfg, srv.url).ask({"head": [BEFORE, AFTER]}, [Q_HAND, Q_LIFT, Q_FREE])
    assert [a["scores"]["yes"] for a in out] == [1.0, 0.0, 0.0]
    assert [json.loads(b)["prompt"].rsplit(": ", 1)[1] for _, b in srv.bodies] == [Q_HAND, Q_LIFT, Q_FREE]


def test_same_text_two_thresholds_is_rejected(tmp_path):
    p = tmp_path / "j.yaml"
    p.write_text(yaml.safe_dump({"prompt": "{question}", "choices": ["yes", "no"], "min_mass": 0.5,
                                 "thresholds": {"a": 0.3, "b": 0.7}, "questions": {"a:x": "Q?", "b:y": "Q?"}}))
    with pytest.raises(ValueError):
        VlmHttpJudge(p, "http://x")


@pytest.mark.parametrize("ok_, allowed, want", [(True, ("advance", "look_again", "handoff", "stop"), "advance"),
                                                (False, ("handoff", "stop"), "handoff")])
def test_decide(cfg, ok_, allowed, want):
    assert VlmHttpJudge(cfg, "http://x").decide([], {"ok": ok_}, allowed)[0] == want


# ---------------------------------------------------------------- 경계 러너와 연결
def run_with(tmp_path, srv, cfg, p_yes):
    srv.reply = lambda body: (200, ok(p_yes))
    r, ex, _, recs, path, _ = make(tmp_path, judge=VlmHttpJudge(cfg, srv.url), questions=lambda n: [Question(Q_LIFT)])
    grant(path)
    return r.run(), ex, [x for x in recs if "event" not in x]


def test_runner_advances_on_real_http_judge(tmp_path, srv, cfg):
    out, ex, b = run_with(tmp_path, srv, cfg, 0.9)
    assert out == SUCCESS and [x["node"] for x in b] == ORDER and all(x["decision"] == "advance" for x in b)
    assert b[0]["answers"][0]["p_yes"] == 0.9 and len(srv.bodies) == len(ORDER)
    assert json.loads(srv.bodies[0][1])["images_b64"] == ["anBn", "anBn"]     # 러너 가짜 사진 b"jpg" 그대로


def test_runner_hands_off_when_photo_says_no(tmp_path, srv, cfg):
    out, ex, b = run_with(tmp_path, srv, cfg, 0.3)
    assert out == HANDED and ran(ex) == [ORDER[0]] and b[0]["decision"] == "handoff" and "어긋남" in b[0]["reason"]

import base64
import importlib
import io
import math
import sys

import pytest
from PIL import Image

from vlm_supervisor.tools import vlm_server as vs

# server 의 Cosmos-Reason2-8B 토크나이저에서 실제로 읽은 첫 토큰 id(11.16)
VOCAB = {"yes": [9693], "Yes": [9454], " yes": [9834], " Yes": [7414],
         "no": [2152], "No": [2753], " no": [902], " No": [2308],
         "yes.": [9693, 13], "Yes.": [9454, 13], " yes.": [9834, 13], " Yes.": [7414, 13]}


def b64(size, fmt):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def test_import_needs_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # 이 상태에서 import torch 는 ImportError
    monkeypatch.setitem(sys.modules, "transformers", None)
    importlib.reload(vs)


def test_parse_request_keeps_legacy_default_and_skips_generation_for_choices_only():
    assert vs.parse_request({"prompt": "p"}) == ("p", None, None, 512, 0.0)
    assert vs.parse_request({"prompt": "p", "max_new_tokens": 64}) == ("p", None, None, 64, 0.0)
    assert vs.parse_request({"prompt": "p", "choices": ["yes", "no"], "system": "s"}) == ("p", "s", ["yes", "no"], 0, 0.0)
    assert vs.parse_request({"prompt": "p", "choices": ["yes", "no"], "max_new_tokens": 8, "min_mass": 0.5})[3:] == (8, 0.5)
    assert vs.parse_request({"prompt": "p", "max_new_tokens": vs.MAX_NEW_TOKENS})[3] == vs.MAX_NEW_TOKENS


@pytest.mark.parametrize("extra", [
    {"system": ["s"]}, {"choices": "yes"}, {"choices": ["yes", ""]}, {"choices": [1, 2]},
    {"choices": []}, {"choices": ""}, {"choices": 0}, {"choices": {}}, {"choices": None},
    {"max_new_tokens": 0}, {"max_new_tokens": -1}, {"max_new_tokens": vs.MAX_NEW_TOKENS + 1},
    {"max_new_tokens": "64"}, {"max_new_tokens": 1.5}, {"max_new_tokens": True}, {"max_new_tokens": None},
    {"min_mass": "0.5"}, {"min_mass": True}, {"min_mass": -0.1}, {"min_mass": 1.5}, {"min_mass": float("nan")}])
def test_parse_request_rejects_wrong_types_and_ranges(extra):
    with pytest.raises(ValueError):
        vs.parse_request({"prompt": "p", **extra})
    with pytest.raises(ValueError):
        vs.parse_request({"prompt": 1})


@pytest.mark.parametrize("ok,value", [(True, "0"), (True, str(vs.MAX_BODY_BYTES)), (False, "-1"),
                                      (False, str(vs.MAX_BODY_BYTES + 1)), (False, "abc"), (False, None)])
def test_content_length_bounds(ok, value):
    if ok:
        assert vs.content_length(value) == int(value)
    else:
        with pytest.raises(vs.BAD_REQUEST):
            vs.content_length(value)


def test_bad_json_is_a_bad_request():
    import json
    with pytest.raises(vs.BAD_REQUEST):
        json.loads(b"{not json")


def test_load_images_keeps_order_and_accepts_png_jpg_and_legacy_key():
    imgs = vs.load_images({"images_b64": [b64((4, 2), "PNG"), b64((2, 6), "JPEG")]})
    assert [(im.size, im.mode) for im in imgs] == [((4, 2), "RGB"), ((2, 6), "RGB")]
    assert [im.size for im in vs.load_images({"image_png_b64": b64((3, 5), "PNG")})] == [(3, 5)]


@pytest.mark.parametrize("bad", [{"images_b64": []}, {"images_b64": "abc"}, {"images_b64": [1]}, {"image_png_b64": None},
                                 {"images_b64": ["aaaa"], "image_png_b64": "aaaa"},
                                 {"images_b64": [b64((2, 2), "PNG")] * (vs.MAX_IMAGES + 1)},
                                 {"image_png_b64": "!!not base64!!"},  # 깨진 b64
                                 {"image_png_b64": base64.b64encode(b"not an image").decode()},  # 깨진 사진
                                 {"image_png_b64": b64((8, 8), "PNG")[:40]}])  # 잘린 PNG
def test_load_images_rejects_bad_lists(bad):
    with pytest.raises(ValueError):
        vs.load_images(bad)


def test_load_images_without_image_is_an_error():
    with pytest.raises(KeyError):
        vs.load_images({"prompt": "p"})


def test_messages_put_images_before_text_and_system_first():
    assert vs.messages(["a"], "q") == [  # system 이 없으면 원본 서버와 같은 메시지
        {"role": "user", "content": [{"type": "image", "image": "a"}, {"type": "text", "text": "q"}]}]
    sys_msg, user = vs.messages(["a", "b"], "q", "s")
    assert sys_msg == {"role": "system", "content": "s"}
    content = user["content"]
    assert isinstance(content, list)
    assert [c.get("image", c.get("text")) for c in content] == ["a", "b", "q"]


def test_choice_token_ids_merges_variants_and_rejects_overlap():
    enc = VOCAB.__getitem__
    assert vs.choice_token_ids(["yes", "no"], enc) == {"yes": [7414, 9454, 9693, 9834], "no": [902, 2152, 2308, 2753]}
    assert vs.choice_token_ids(["Yes"], enc) == {"Yes": [7414, 9454]}  # 대문자 변형이 같으면 중복 제거
    for overlapping in (["yes", "Yes"], ["yes", "yes"]):
        with pytest.raises(ValueError, match="share"):
            vs.choice_token_ids(overlapping, enc)


def test_choice_token_ids_counts_single_token_variants_only():
    with pytest.raises(ValueError, match="single-token"):  # "yes." 는 모든 변형이 두 토큰
        vs.choice_token_ids(["yes", "yes."], VOCAB.__getitem__)
    enc = {"ok": [1], "Ok": [2, 3], " ok": [4], " Ok": [5, 6]}.__getitem__  # 가짜 id: 대문자 변형만 두 토큰
    assert vs.choice_token_ids(["ok"], enc) == {"ok": [1, 4]}


def test_choice_scores_normalizes_within_choices_and_reports_mass():
    logits = [math.log(x) for x in (1, 2, 3, 4, 10)]  # softmax = x / 20
    ids = {"yes": [1, 3], "no": [0]}
    for shift in (0.0, 1000.0):  # 큰 logit 에서도 넘치지 않는다
        out = vs.choice_scores([x + shift for x in logits], ids)
        assert set(out) == {"scores", "probs", "choice_mass"}
        assert out["choice_mass"] == pytest.approx(0.35) == pytest.approx(sum(out["probs"].values()))
        assert out["probs"] == pytest.approx({"yes": 0.3, "no": 0.05})
        assert out["scores"] == pytest.approx({"yes": 6 / 7, "no": 1 / 7})
    assert vs.choice_scores(logits, ids, min_mass=0.35)["scores"]  # 경계값은 통과(미만일 때만 비운다)
    low = vs.choice_scores(logits, ids, min_mass=0.5)
    assert low["scores"] == {} and low["low_mass"] is True
    assert low["choice_mass"] == pytest.approx(0.35) and low["probs"] == pytest.approx({"yes": 0.3, "no": 0.05})
    with pytest.raises(ValueError):
        vs.choice_scores([0.0, float("nan"), 1.0], {"yes": [0], "no": [2]})

"""VLM HTTP server (arm4090). POST /infer (JSON) -> JSON.

요청 {"prompt", "image_png_b64" | "images_b64": [..], "max_new_tokens"?, "choices"?: ["yes", "no"], "min_mass"?, "system"?}
응답 {"text", "gen_s", "peak_gib"}, choices 가 있으면 "scores": {선택지: 선택지끼리 정규화한 p},
"probs": {선택지: 전체 어휘 softmax 의 p}, "choice_mass"(= probs 합)를 더한다. 선택지는 한 토큰 단어.
choice_mass < min_mass 이면 "scores" 는 {} 이고 "low_mass": true 다(200).
choices 만 있고 max_new_tokens 가 0 이거나 없으면 생성을 건너뛴다(text 는 "").
오류: 요청 형식 400, 추론 중 예외 500, CUDA OOM 503.

stdlib http.server + transformers: vLLM would pin its own torch and is a heavy install for one-image
requests; a single-threaded server also serializes GPU use for free.
Run: ~/venvs/vlm/bin/python vlm_server.py --model nvidia/Cosmos-Reason2-2B --host 192.168.0.40 --port 8765
"""
import argparse, base64, io, itertools, json, math, time, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

MAX_BODY_BYTES = 64 * 2**20
MAX_IMAGES = 8
MAX_NEW_TOKENS = 4096
SOCKET_TIMEOUT_S = 30
BAD_REQUEST = (ValueError, KeyError, TypeError)  # json·b64 오류도 ValueError 계열


def content_length(value):
    """Content-Length 헤더 값 → int. 없거나 범위를 벗어나면 ValueError·TypeError(→ 400, 본문을 읽지 않는다)."""
    n = int(value)
    if not 0 <= n <= MAX_BODY_BYTES:
        raise ValueError(f"Content-Length must be 0..{MAX_BODY_BYTES}")
    return n


def parse_request(req):
    """요청 dict → (prompt, system, choices, max_new_tokens, min_mass). 형식이 틀리면 KeyError·ValueError(→ 400)."""
    prompt, system, choices = req["prompt"], req.get("system"), req.get("choices")
    if not isinstance(prompt, str) or not isinstance(system, (str, type(None))):
        raise ValueError("prompt and system must be strings")
    if "choices" in req and not (isinstance(choices, list) and choices and all(isinstance(c, str) and c for c in choices)):
        raise ValueError("choices must be a non-empty list of non-empty strings")
    # choices 만 오면 생성을 건너뛴다(기본 0). choices 가 없으면 예전 기본 512.
    max_new_tokens = req.get("max_new_tokens", 0 if choices else 512)
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= MAX_NEW_TOKENS:  # type(): bool 거부
        raise ValueError(f"max_new_tokens must be an int in 0..{MAX_NEW_TOKENS}")
    if not choices and max_new_tokens == 0:
        raise ValueError("nothing to do: no choices and max_new_tokens is 0")
    min_mass = req.get("min_mass", 0.0)  # 기본 0.0: 8B 의 실제 choice_mass 를 아직 재지 않아 문턱을 정하지 못했다
    if type(min_mass) not in (int, float) or not 0 <= min_mass <= 1:  # NaN 도 여기서 걸린다
        raise ValueError("min_mass must be a number in 0..1")
    return prompt, system, choices, max_new_tokens, min_mass


def load_images(req):
    """images_b64(여러 장) 또는 기존 image_png_b64(한 장) → RGB PIL 목록. 요청에 적힌 순서 그대로."""
    if "images_b64" in req and "image_png_b64" in req:
        raise ValueError("give images_b64 or image_png_b64, not both")
    b64 = req["images_b64"] if "images_b64" in req else [req["image_png_b64"]]
    if not (isinstance(b64, list) and 0 < len(b64) <= MAX_IMAGES and all(isinstance(s, str) for s in b64)):
        raise ValueError(f"images_b64 must be a list of 1..{MAX_IMAGES} base64 strings")
    try:
        return [Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB") for s in b64]
    except (OSError, Image.DecompressionBombError) as e:  # PIL 은 깨진 사진에 OSError 계열을 낸다
        raise ValueError(f"cannot decode image: {e}") from e


def messages(images, prompt, system=None):
    """chat template 입력. 사진을 글보다 앞에 둔다. system 은 있을 때만 맨 앞에 넣는다."""
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]


def choice_token_ids(choices, encode):
    """선택지(한 토큰 단어) → 토큰 id 목록. 변형(그대로, 첫 글자 대문자, 각각 앞 공백) 중 정확히 한 토큰인 것만 센다.

    한 토큰 변형이 없는 선택지, 선택지끼리 겹치는 id 는 ValueError.
    """
    ids = {}
    for c in choices:
        cap = c[:1].upper() + c[1:]
        ids[c] = sorted({t[0] for t in map(encode, (c, cap, " " + c, " " + cap)) if len(t) == 1})
        if not ids[c]:
            raise ValueError(f"choice {c!r} has no single-token variant")
    for a, b in itertools.combinations(choices, 2):
        if shared := set(ids[a]) & set(ids[b]):
            raise ValueError(f"choices {a!r} and {b!r} share first token ids {sorted(shared)}")
    return ids


def choice_scores(logits, ids, min_mass=0.0):
    """다음 토큰 logit(전체 어휘) → {"scores": 선택지끼리 정규화, "probs": 전체 어휘 softmax, "choice_mass": probs 합}.

    choice_mass < min_mass 면 scores 는 {} 이고 "low_mass": True 를 싣는다(첫 토큰이 선택지가 아닌 답).
    """
    m = max(logits)
    z = sum(math.exp(x - m) for x in logits)
    p = {c: sum(math.exp(logits[i] - m) for i in t) / z for c, t in ids.items()}
    mass = sum(p.values())
    if not mass > 0:  # NaN·inf logit 이면 JSON 에 NaN 을 싣지 않고 실패시킨다
        raise ValueError(f"choice mass is {mass}")
    if mass < min_mass:
        return {"scores": {}, "probs": p, "choice_mass": mass, "low_mass": True}
    return {"scores": {c: v / mass for c, v in p.items()}, "probs": p, "choice_mass": mass}


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    ap.add_argument("--host", default="192.168.0.40")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print("loaded", args.model, f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    def infer(msgs, ids, max_new_tokens, min_mass):
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        out: dict[str, object] = {"text": ""}
        with torch.inference_mode():
            if ids:
                # shortcut: logit 은 모델 dtype(bf16) 격자 값이다, 더 고운 확률이 필요하면 lm_head 만 float32 로 계산
                logits = model(**inp, logits_to_keep=1).logits[0, -1].float().tolist()
                out.update(choice_scores(logits, ids, min_mass))
            if max_new_tokens > 0:
                gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
                out["text"] = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return out

    class H(BaseHTTPRequestHandler):
        timeout = SOCKET_TIMEOUT_S

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(200, {"model": args.model, "peak_gib": torch.cuda.max_memory_allocated() / 2**30})

        def do_POST(self):
            if self.path != "/infer":
                return self._send(404, {"error": "use /infer"})
            try:
                try:
                    req = json.loads(self.rfile.read(content_length(self.headers["Content-Length"])))
                    prompt, system, choices, max_new_tokens, min_mass = parse_request(req)
                    msgs = messages(load_images(req), prompt, system)
                    # 겹치는 선택지는 GPU 를 쓰기 전에 400 으로 거른다
                    ids = choices and choice_token_ids(choices, lambda s: proc.tokenizer.encode(s, add_special_tokens=False))
                except BAD_REQUEST as e:
                    return self._send(400, {"error": repr(e)})
                t = time.perf_counter()
                out = infer(msgs, ids, max_new_tokens, min_mass)
                self._send(200, {**out, "gen_s": time.perf_counter() - t,
                                 "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                self._send(503, {"error": repr(e)})
            except Exception as e:  # 추론 쪽 예외는 서버를 죽이지 않고 500, 원인은 stderr 에
                traceback.print_exc()
                self._send(500, {"error": repr(e)})

    HTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()

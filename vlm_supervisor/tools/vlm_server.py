"""VLM HTTP server (arm4090). POST /infer {"image_png_b64", "prompt", "max_new_tokens"} -> {"text", "gen_s"}.

stdlib http.server + transformers: vLLM would pin its own torch and is a heavy install for one-image
requests; a single-threaded server also serializes GPU use for free.
Run: ~/venvs/vlm/bin/python vlm_server.py --model nvidia/Cosmos-Reason2-2B --host 192.168.0.40 --port 8765
"""
import argparse, base64, io, json, time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
ap.add_argument("--host", default="192.168.0.40")
ap.add_argument("--port", type=int, default=8765)
args = ap.parse_args()

proc = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
print("loaded", args.model, f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)


def infer(img, prompt, max_new_tokens):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class H(BaseHTTPRequestHandler):
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
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            img = Image.open(io.BytesIO(base64.b64decode(req["image_png_b64"]))).convert("RGB")
            t = time.perf_counter()
            text = infer(img, req["prompt"], int(req.get("max_new_tokens", 512)))
            self._send(200, {"text": text, "gen_s": time.perf_counter() - t,
                             "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
        except Exception as e:  # bad request must not kill the server
            self._send(400, {"error": repr(e)})


HTTPServer((args.host, args.port), H).serve_forever()

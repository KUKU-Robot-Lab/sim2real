"""VLM 백엔드 묶음 — Gemini(일일 한도 · 토큰 장부) → 로컬 Cosmos-Reason2(5080) 순서로 자동 전환.

장부: logs/vlm_supervisor/state/usage.json (태평양 시간 날짜별 · 모델별 호출 수 · 토큰 · 막힘 시각)
      logs/vlm_supervisor/state/calls.jsonl (호출 한 건 한 줄)
환경변수:
  VLM_CHAIN      기본 "gemini-robotics-er-2-preview,local" (쉼표 순서대로 시도)
  VLM_CAP_<모델> 일일 상한 덮어쓰기 (예 VLM_CAP_gemini_robotics_er_2_preview=200, 유료 전환 뒤)
  VLM_LOCAL_URL  기본 http://192.168.0.60:8765 (5080 Cosmos-Reason2-2B)
  GEMINI_KEY_FILE 기본 ~/.config/gemini_key
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import re
import time
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
from PIL import Image

ROOT = os.path.expanduser(os.environ.get("VLM_SUP_HOME", "logs/vlm_supervisor"))
STATE = os.path.join(ROOT, "state")
PT = ZoneInfo("America/Los_Angeles")          # Gemini 일일 한도는 태평양 시간 자정에 풀린다
# 무료 등급 일일 요청 상한(11.10 429 응답 quotaValue 실측: ER-2 preview = 20). 모르는 모델은 상한 없음(429 로 배운다).
DEFAULT_CAPS = {"gemini-robotics-er-2-preview": 20}
CAP_MARGIN = int(os.environ.get("VLM_CAP_MARGIN", "0"))   # 상한 전에 남겨 둘 호출 수


def pt_day(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, PT).strftime("%F")


def next_pt_midnight() -> float:
    now = dt.datetime.now(PT)
    return (now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1, minutes=1)).timestamp()


def cap_of(model: str) -> int | None:
    env = os.environ.get("VLM_CAP_" + re.sub(r"[^A-Za-z0-9]", "_", model))
    if env:
        return int(env)
    return DEFAULT_CAPS.get(model)


class Ledger:
    def __init__(self, state_dir=None):
        self.state_dir = Path(state_dir or STATE)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.usage = self.state_dir / 'usage.json'
        try:
            self.d = json.loads(self.usage.read_text(encoding='utf-8'))
        except FileNotFoundError:
            self.d = {}

    def _m(self, model: str) -> dict:
        day = self.d.setdefault(pt_day(), {})
        return day.setdefault(model, {"n": 0, "in_tok": 0, "out_tok": 0, "think_tok": 0, "blocked_until": 0.0, "err": 0})

    def save(self):
        tmp = self.usage.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.d, indent=1), encoding='utf-8')
        os.replace(tmp, self.usage)

    def remaining(self, model: str) -> int | None:
        cap = cap_of(model)
        return None if cap is None else cap - CAP_MARGIN - self._m(model)["n"]

    def blocked(self, model: str) -> float:
        b = self._m(model)["blocked_until"]
        return b if b > time.time() else 0.0

    def usable(self, model: str) -> bool:
        r = self.remaining(model)
        return not self.blocked(model) and (r is None or r > 0)

    def count(self, model: str, meta: dict):
        m = self._m(model)
        m["n"] += 1
        m["in_tok"] += meta.get("in_tok", 0)
        m["out_tok"] += meta.get("out_tok", 0)
        m["think_tok"] += meta.get("think_tok", 0)
        self.save()

    def block(self, model: str, until: float, why: str):
        m = self._m(model)
        m["blocked_until"] = until
        m["err"] += 1
        m["why"] = why[:200]
        self.save()

    def table(self, chain: list[str]) -> list[dict]:
        out = []
        for model in chain:
            m = self._m(model)
            b = self.blocked(model)
            out.append({"model": model, "used_today": m["n"], "cap": cap_of(model), "remaining": self.remaining(model),
                        "blocked_until": time.strftime("%H:%M", time.localtime(b)) if b else None,
                        "tokens_in/out/think": f'{m["in_tok"]}/{m["out_tok"]}/{m["think_tok"]}'})
        return out


def to_jpeg_b64(img: Image.Image, max_side: int = 768, q: int = 85) -> str:
    """토큰 절약 — 긴 변 768 px · JPEG 85. 1280x720 원본 대비 이미지 토큰이 줄고 판단 품질은 오프라인 시험과 같은 크기."""
    im = img.convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.BILINEAR)
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    return base64.b64encode(b.getvalue()).decode()


class QuotaExhausted(RuntimeError):
    pass


class Gemini:
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"

    def __init__(self, model: str, ledger: Ledger):
        self.model, self.ledger = model, ledger
        key = os.environ.get('GEMINI_API_KEY')
        if not key:
            key = Path(os.environ.get('GEMINI_KEY_FILE', '~/.config/gemini_key')).expanduser().read_text().strip()
        if not key:
            raise ValueError('empty Gemini API key')
        self.key = key

    def ask(self, images: list[Image.Image], text: str, level: str = "low", timeout: float = 90) -> tuple[str, dict]:
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": to_jpeg_b64(im)}} for im in images] + [{"text": text}]
        body = {"contents": [{"parts": parts}], "generationConfig": {"thinkingConfig": {"thinkingLevel": level}}}
        last = ""
        for k in range(4):
            t0 = time.perf_counter()
            try:
                r = requests.post(self.URL.format(m=self.model), headers={"x-goog-api-key": self.key}, json=body, timeout=timeout)
            except requests.RequestException as e:            # 망 끊김 — 짧게 두 번만 더
                last = f"net {type(e).__name__}"
                time.sleep(2 * (k + 1))
                continue
            if r.status_code == 429:
                info = r.text
                per_day = "PerDay" in info
                m = re.search(r'"retryDelay":\s*"(\d+)s"', info)
                delay = float(m.group(1)) if m else 60.0
                if per_day:                                     # 일일 한도 — 오늘은 이 모델을 더 쓰지 않는다
                    self.ledger.block(self.model, time.time() + delay if m else next_pt_midnight(), "daily quota")
                    raise QuotaExhausted(f"{self.model}: 일일 한도 소진 · {delay/60:.0f}분 뒤 풀림")
                if delay > 20:                                  # 분당 한도인데 오래 기다려야 함 — 다음 백엔드로
                    self.ledger.block(self.model, time.time() + delay, "rate limit")
                    raise QuotaExhausted(f"{self.model}: 분당 한도 · {delay:.0f}s")
                time.sleep(delay + 1)
                continue
            if r.status_code >= 500:
                last = f"http {r.status_code}"
                time.sleep(3 * (k + 1))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"{self.model}: http {r.status_code} {r.text[:200]}")
            j = r.json()
            u = j.get("usageMetadata", {})
            meta = {"model": self.model, "sec": round(time.perf_counter() - t0, 2), "in_tok": u.get("promptTokenCount", 0),
                    "out_tok": u.get("candidatesTokenCount", 0), "think_tok": u.get("thoughtsTokenCount", 0)}
            self.ledger.count(self.model, meta)
            try:
                txt = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"] if not p.get("thought"))
            except (KeyError, IndexError):
                txt = ""
            return txt, meta
        raise RuntimeError(f"{self.model}: 재시도 실패 ({last})")


class Local:
    """5080 Cosmos-Reason2-2B HTTP 서버(vlm_server.py) — 이미지 1장만 받는다(마지막 = 지금 장면)."""

    #: 이름 → 기본 주소. local = 5080 2B, local8b = 5090 위 Cosmos-Reason2-8B(띄웠을 때만)
    URLS = {"local": "http://192.168.0.60:8765", "local8b": "http://127.0.0.1:8766"}

    def __init__(self, ledger: Ledger, name: str = "local"):
        env = "VLM_LOCAL_URL" if name == "local" else "VLM_" + name.upper() + "_URL"
        self.url = os.environ.get(env, self.URLS.get(name, "")).rstrip("/")
        self.ledger, self.model = ledger, name

    def ask(self, images, text, level="low", timeout: float = 60, max_new_tokens: int = 400):
        if not images:
            raise ValueError('local vlm_server requires an image')
        b = io.BytesIO()
        images[-1].convert("RGB").save(b, format="PNG")
        t0 = time.perf_counter()
        r = requests.post(self.url + "/infer", timeout=timeout,
                          json={"image_png_b64": base64.b64encode(b.getvalue()).decode(),
                                "prompt": "(Only the CURRENT image is attached.)\n" + text, "max_new_tokens": max_new_tokens})
        r.raise_for_status()
        meta = {"model": self.model, "sec": round(time.perf_counter() - t0, 2)}
        self.ledger.count(self.model, meta)
        return r.json().get("text", ""), meta

    def alive(self) -> bool:
        try:
            return requests.get(self.url + "/", timeout=3).status_code < 500
        except requests.RequestException:
            return False


class Router:
    def __init__(self, chain: list[str] | None = None, state_dir=None):
        self.chain = chain or [c.strip() for c in os.environ.get("VLM_CHAIN", "gemini-robotics-er-2-preview,local").split(",") if c.strip()]
        self.ledger = Ledger(state_dir)
        self.backends = {}  # Lazy: a missing Gemini key must not prevent local fallback.

    def ask(self, images, text, level="low", **kwargs) -> tuple[str, dict]:
        errs = []
        for c in self.chain:
            if not self.ledger.usable(c):
                errs.append(f"{c}: 건너뜀(한도/막힘)")
                continue
            try:
                if c not in self.backends:
                    self.backends[c] = Local(self.ledger, c) if c.startswith('local') else Gemini(c, self.ledger)
                txt, meta = self.backends[c].ask(images, text, level, **(kwargs if c.startswith("local") else {}))
                meta["fallbacks"] = errs
                self._log(meta, text, txt)
                return txt, meta
            except QuotaExhausted as e:
                errs.append(str(e))
            except Exception as e:                              # noqa: BLE001 — 다음 백엔드로
                errs.append(f"{c}: {type(e).__name__} {str(e)[:120]}")
        self._log({"model": None, "fallbacks": errs}, text, "")
        raise RuntimeError("모든 VLM 백엔드 실패: " + " | ".join(errs))

    def calls_left(self) -> str:
        parts = []
        for c in self.chain:
            r = self.ledger.remaining(c)
            parts.append(f"{c}={'∞' if r is None else r}{'(막힘)' if self.ledger.blocked(c) else ''}")
        return " ".join(parts)

    def _log(self, meta, prompt, txt):
        with (self.ledger.state_dir / 'calls.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({"t": time.strftime("%F %T"), **meta, "prompt_head": prompt[-160:], "answer": txt[:400]},
                                ensure_ascii=False) + "\n")


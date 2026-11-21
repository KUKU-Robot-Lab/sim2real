"""Judge 구현 — judge_short.yaml + VLM 서버(tools/vlm_server.py). 요청은 bench run_bench 의 short2(ask_short → infer)와 같다.

ask: 질문마다 POST {url}/infer {"prompt": prompt.format(question=문구), "images_b64": [전, 후], "max_new_tokens": 0, "choices"}.
사진은 {카메라: [전, 후]} 로 받는다. head 하나면 요청이 위와 같고(문턱을 맞춘 판), 다른 카메라가 끼면 카메라 순서대로 [전, 후]를
이어 보내고 프롬프트 앞에 사진별 카메라 이름 한 줄을 붙인다(shortcut: 여러 카메라 프롬프트 · 문턱은 head 판 그대로, 글로벌 사진
시험대 결과가 생기면 calibrate 로 다시 맞춘다).
p_yes(선택지끼리 정규화한 yes) 와 choice_mass 를 bench answer_of 규칙으로 읽고, 보류(점수 없음 · 동점 · NaN · mass < min_mass ·
|p_yes - 문턱| < band)는 {"low_mass": True}, 아니면 문턱으로 이진화해 {"scores": {"yes": 1|0, "no": 0|1}}. 오류는 {"error": 사유}.
boundary.picks 는 low_mass · scores 없는 답을 ValueError 로 받아 handoff 한다. p_yes · thr · mass 는 기록용 키다(picks 는 안 읽는다).
"""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Mapping, Sequence

import requests
import yaml

from .protocols import ADVANCE, HANDOFF, HEAD

DEFAULT_THR = 0.5


def cams_line(images: Mapping[str, Sequence[bytes]]) -> str:
    """사진 순서 → 카메라 이름 한 줄. 예: Photographs 1-2: head camera, 3-4: global camera (each pair: before, after)."""
    k, parts = 1, []
    for c in images:
        n = len(images[c])
        parts.append(f"{k}-{k + n - 1}: {c} camera")
        k += n
    return ("Photographs " + ", ".join(parts) + " (each pair: before, then after the last robot action). "
            "Answer about the after photographs.")


class VlmHttpJudge:
    def __init__(self, cfg_path, url: str, timeout_s: float = 15.0, band: float = 0.0):
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.prompt, self.choices, self.min_mass = cfg["prompt"], list(cfg["choices"]), float(cfg["min_mass"])
        thr = cfg.get("thresholds") or {}
        self.thr: dict[str, float] = {}             # 문구 → 문턱(슬롯 ':' 앞이 문턱 키)
        for slot, text in (cfg.get("questions") or {}).items():
            t = float(thr.get(slot.split(":")[0], DEFAULT_THR))
            if self.thr.setdefault(text, t) != t:
                raise ValueError(f"같은 문구에 문턱이 둘: {text!r}")
        self.url, self.timeout_s, self.band = url.rstrip("/"), timeout_s, band

    def ask(self, images: Mapping[str, Sequence[bytes]], questions: Sequence[str]) -> list[dict]:
        bad = [c for c in images if not (isinstance(c, str) and re.fullmatch(r"[A-Za-z0-9_]+", c))]
        if bad:     # 이름이 프롬프트에 들어가고 format 을 거친다 — 중괄호 · 줄바꿈 등이 끼지 않게
            raise ValueError(f"카메라 이름은 [A-Za-z0-9_]+ 만: {bad!r}")
        imgs = [base64.b64encode(bytes(b)).decode() for c in images for b in images[c]]   # 재인코딩 없이 jpg 바이트 그대로
        prompt = self.prompt if list(images) == [HEAD] else cams_line(images) + "\n" + self.prompt
        out: list[dict] = []
        for q in questions:
            if out and "error" in out[-1]:          # 서버가 죽었으면 남은 질문마다 timeout 을 또 기다리지 않는다
                out.append({"error": "앞 질문 오류로 묻지 않음"})
                continue
            out.append(self._one(imgs, q, prompt))
        return out

    def _one(self, imgs: list, q: str, prompt: str) -> dict:
        body = {"prompt": prompt.format(question=q), "images_b64": imgs, "max_new_tokens": 0, "choices": self.choices}
        try:
            r = requests.post(self.url + "/infer", json=body, timeout=self.timeout_s)
            if r.status_code != 200:
                raise ValueError(f"http {r.status_code} {r.text[:200]}")
            resp = r.json()
            if not isinstance(resp, dict):
                raise ValueError(f"응답이 dict 가 아니다: {str(resp)[:200]}")
            scores = resp.get("scores")
            if not resp.get("low_mass") and not (
                    isinstance(scores, dict) and all(isinstance(scores.get(c), (int, float)) for c in self.choices)):
                raise ValueError(f"응답에 선택지 점수가 없다: {str(resp)[:200]}")
        except (requests.RequestException, ValueError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return self.read_resp(resp, q)

    def read_resp(self, resp: Mapping, q: str) -> dict:
        """서버 응답 하나(/infer 의 dict) → 판정. bench results.jsonl 의 resp 를 다시 판정할 때도 쓴다(episode_review)."""
        scores = resp.get("scores") or {}
        yes, no, mass = scores.get("yes"), scores.get("no"), resp.get("choice_mass")
        thr = self.thr.get(q, DEFAULT_THR)
        rec = {"p_yes": yes, "thr": thr, "mass": mass}
        # bench answer_of 와 같은 보류: 점수 없음 · 동점 · NaN, 질량이 없거나 NaN 이거나 min_mass 미만
        if yes is None or no is None or not (yes > no or no > yes) or (
                self.min_mass > 0 and not (isinstance(mass, (int, float)) and mass >= self.min_mass)):
            return {**rec, "low_mass": True}
        if abs(yes - thr) < self.band:
            return {**rec, "low_mass": True, "band": True}
        ok = yes >= thr
        return {**rec, "scores": {"yes": 1.0 if ok else 0.0, "no": 0.0 if ok else 1.0}}

    def decide(self, answers: Sequence[Mapping], signal: Mapping, allowed: Sequence[str]) -> tuple[str, str]:
        """러너는 사진 답과 신호가 맞을 때만 부른다. 신호 성공이면 advance, 아니면 handoff(사람이 본다).
        shortcut: 규칙 2단(VLM 아님), PLAN 의 VLM 2단 프롬프트가 정해지면 교체한다."""
        if signal.get("ok") and ADVANCE in allowed:
            return ADVANCE, "사진 · 러너 모두 성공"
        return HANDOFF, "러너 신호 실패 — 사람이 본다"

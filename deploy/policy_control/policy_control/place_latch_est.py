"""★10.22 place 래치 입력 = 손 안 컵 자세 추정기(hdgp tasks/rh_place_r/cup_est.py CupEst, 10.21 1단계) — 배포 사본.

newton_05 는 이 GRU 를 붙인 채 학습했다(learn:place 10.21 "빼면 안 됩니다"). 정책 관측의 컵은 그대로 붙인 자세(손바닥 FK ⊗
넘김 묶음)이고, 이 추정기는 래치(놓을 자리 도착) 판정의 컵 자세만 보정한다(나르는 동안 손 안 미끄러짐).
  입력  정책 틱마다 actor 관측(±5 자름) — 관측 래치 칸은 갱신 전 값(sim _cup_est_step: 데이터에서도 래치 전까지 0)
  출력  6 = [손바닥 좌표 원점 차 mm 3, up 벡터 차 × 1000 × BOT 3]
  live  리셋 뒤 첫 틱만 False(sim episode_length_buf > 0) — 그 틱은 상태 0 · 보정 0
  자세  p = 붙인 원점 + 차 / 1000, 축 = 붙인 축을 추정 축으로 최소 회전(yaw 그대로)
ckpt = {"sd", "mu", "std"}(place_slip_est_fit.py). 파일 md5 는 계약 latch_est 에 핀으로 둔다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

#: hdgp cup_est.BOT · CLIP_OBS — place_slip_est_fit.BOT 와 같아야 한다(축 출력 단위), agent clip_observations
BOT = 0.085
CLIP_OBS = 5.0


class LatchEstError(ValueError):
    pass


def md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[0], q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rel_pose(att_p, att_q, y) -> tuple[np.ndarray, np.ndarray]:
    """hdgp CupEst.rel_pose — 손바닥 좌표 추정 컵 자세(붙인 자세 + 원점 보정, 축 최소 회전)."""
    att_p, att_q, y = (np.asarray(v, float).reshape(-1) for v in (att_p, att_q, y))
    p = att_p + y[:3] / 1000.0
    a = _quat_apply(att_q, np.array([0.0, 0.0, 1.0]))
    b = a + y[3:] / (1000.0 * BOT)
    b = b / max(float(np.linalg.norm(b)), 1e-9)
    q = np.concatenate([[1.0 + float(a @ b)], np.cross(a, b)])
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    return p, _quat_mul(q, att_q)


class LatchEst:
    """한 팔의 GRU 상태를 들고 정책 틱마다 한 번 갱신한다(sim CupEst 와 같은 식, env 1 개)."""

    def __init__(self, path: str | Path, want_md5: str, d_in: int) -> None:
        import torch
        import torch.nn as nn

        p = Path(path)
        if not p.is_file():
            raise LatchEstError(f"래치 추정기 {p} 가 없다 — 정책 폴더 latch/ 에 받는다(md5 {want_md5})")
        got = md5(p)
        if got != want_md5:
            raise LatchEstError(f"래치 추정기 {p}: md5 {got} ≠ 계약 {want_md5}")
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(ck, Mapping) or not {"sd", "mu", "std"} <= set(ck):
            raise LatchEstError(f"{p}: {{sd, mu, std}} ckpt 가 아니다")
        sd = ck["sd"]
        din, h = sd["inp.0.weight"].shape[1], sd["inp.0.weight"].shape[0]
        if int(din) != int(d_in):
            raise LatchEstError(f"{p}: 입력 {din} ≠ 정책 관측 {d_in}")

        class Est(nn.Module):                       # hdgp cup_est.Est 와 같은 층 이름(state_dict 를 그대로 싣는다)
            def __init__(s) -> None:
                super().__init__()
                s.inp = nn.Sequential(nn.Linear(din, h), nn.ELU())
                s.rnn = nn.GRU(h, h, batch_first=True)
                s.out = nn.Linear(h, 6)

        self.torch = torch
        self.m = Est()
        self.m.load_state_dict(sd)
        self.m.eval()
        self.mu = torch.as_tensor(ck["mu"], dtype=torch.float32).reshape(-1)
        self.sd = torch.as_tensor(ck["std"], dtype=torch.float32).reshape(-1)
        if self.mu.numel() != din or self.sd.numel() != din:
            raise LatchEstError(f"{p}: mu · std 폭 {self.mu.numel()} · {self.sd.numel()} ≠ {din}")
        self.d_in, self.hidden, self.path = int(din), int(h), p
        self.reset()

    @staticmethod
    def stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """(mu, std) — 계약 빌드가 학습 데이터에서 늘 0 이던 칸(이웃 홀더)을 확인할 때 쓴다."""
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return (np.asarray(torch.as_tensor(ck["mu"]).reshape(-1), float),
                np.asarray(torch.as_tensor(ck["std"]).reshape(-1), float))

    def reset(self) -> None:
        self.h = self.torch.zeros(1, 1, self.hidden)
        self.y = np.zeros(6)
        self.k = 0

    def step(self, obs) -> np.ndarray:
        """이번 틱 actor 관측(래치 칸 = 갱신 전) → 보정 6. 리셋 뒤 첫 틱은 넣지 않는다(상태 0 · 보정 0)."""
        torch = self.torch
        o = np.asarray(obs, float).reshape(-1)
        if o.size != self.d_in:
            raise LatchEstError(f"래치 추정기 관측 폭 {o.size} ≠ {self.d_in}")
        live = self.k > 0
        self.k += 1
        if not live:
            self.h = torch.zeros_like(self.h)
            self.y = np.zeros(6)
            return self.y
        x = (torch.as_tensor(np.clip(o, -CLIP_OBS, CLIP_OBS), dtype=torch.float32) - self.mu) / self.sd
        with torch.no_grad():
            z, h = self.m.rnn(self.m.inp(x[None]).unsqueeze(1), self.h)
            y = self.m.out(z[:, 0])
        self.h = h
        self.y = y[0].numpy().astype(float)
        return self.y

    def rel_pose(self, att_p, att_q) -> tuple[np.ndarray, np.ndarray]:
        return rel_pose(att_p, att_q, self.y)

    def as_dict(self) -> dict:
        return {"ticks": self.k, "d_mm": [round(float(v), 2) for v in self.y[:3]],
                "axis": [round(float(v), 2) for v in self.y[3:]]}

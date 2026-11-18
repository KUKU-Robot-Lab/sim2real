"""11.18 rh_cap 깊이 학생 depth_h01(캡 찾기 → 접근 → 잡기 → 이송 → 끼우기, 움직이는 머리) 정책 코어 — ROS · 단계 전환은 여기 없다.

ckpt = hdgp learner.save(db1160bc): arch 키 없음 = v1(ResNet18 + 고유수용 MLP → LSTM 512 → 행동 15 · aux 6). model.py 는 기존 학생과 같은
고정 사본(student_policy.load_model_module, config/hdgp_student_snapshot.yaml)의 arch "v1" 로 읽는다 — v1 경로는 db1160bc 판과 식이 같다.
고유수용 정규화(mu · var, ±5) · 깊이 0~1 · ImageNet 정규화는 모델 안에 있다. 여기서 따로 정규화하지 않는다.

관측 62 = hdgp RhCapEnv._student_proprio(actor 앞 7 조각 + 직전 실행 행동 13 + 단계 one-hot 5) + learner._head_in(머리 각 2).
  직전 행동 = env 가 실제로 실행한 13 칸(상자 · 손 잔차 → 절대 · 이송/접근 스크립트로 바뀐 뒤 값, hdgp prev_actions = self.actions).
행동 15 = 팔 7 · 손 6 · 머리 2(절대 목표 인코더 각). 팔 · 손 13 칸 → 관절 목표는 기존 법칙을 그대로 쓴다:
  rh_aglt.RaDecoder.step(팔 증분 + EMA · rh56f1_hand.HandLaw) · rh_cap.box_arm_action(끼우기 팔 상자). 없는 것만 여기 둔다:
  끼우기 손 잔차 → 절대 행동(seat_hand_action) · 머리 행동 ↔ 각(head_deg · head_obs).
LSTM 은 reset() 에서만 0 — 단계가 바뀌어도, 이송(출력을 버리는 구간)에도 매 틱 step 을 불러 상태를 잇는다(sim learner 와 같다).
reset · step 은 락이 없다 — 노드가 한 스레드(제어 타이머)에서만 부른다.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from . import hdgp_pin
from .student_policy import MAX_DEPTH_AGE_S, SNAPSHOT_CFG, STUDENT_CARD, StaleDepth, load_model_module

TASK = "rh_cap"
#: hdgp RhCapEnv.chain_stage_names(chain_capa · chain_search 켬) — one-hot 칸 순서
STAGES = ("grasp", "carry", "seat", "approach", "search")
PROP_COLS = (("arm_q", 7), ("arm_qd", 7), ("arm_q_target", 7), ("hand_q", 6), ("hand_err", 6), ("palm_pos", 3),
             ("palm_rot", 6), ("prev_action", 13), ("stage", len(STAGES)), ("head", 2))
N_OBS = sum(w for _, w in PROP_COLS)
N_ARM_HAND, N_ACT = 13, 15
IMG_HW = (90, 160)
D_MIN_M, D_MAX_M = 0.2, 1.5           # 학습 깊이 창(ckpt d_min · d_max) — 밖은 모델이 0 으로 만든다
DEPTH_MAX_M = 10.0                    # 입력 상한 = 학습 카메라 head_depth_clip 끝. 넘으면 단위 오류(mm)로 보고 거부한다
JOINT_ERR_NORM = 1.2                  # 학습 env.yaml joint_err_norm
SEAT_HAND_RESIDUAL_SCALE = 1.0        # 학습 env.yaml chain_seat_hand_residual_scale
#: hdgp HeadCfg pan_lim · tilt_lim — 실기 인코더 각 deg(pan, tilt). ckpt head_cfg 와 다르면 거부한다
HEAD_LO, HEAD_HI = np.array([-45.0, -2.0]), np.array([30.0, 82.0])
HEAD_OVER_DEG = 5.0                   # 실측 머리 각이 한계를 이만큼 넘으면 자르지 않고 거부(11.02 틸트 wrap 전례)

log = logging.getLogger(__name__)


def _arr(name: str, x: ArrayLike, shape: tuple) -> np.ndarray:
    """정확히 그 모양의 유한한 float 배열만(브로드캐스트 · 평탄 입력 거부)."""
    a = np.asarray(x, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"'{name}' 은 유한한 {shape}: 모양 {a.shape} 값 {a}")
    return a


def head_obs(enc_deg: ArrayLike) -> np.ndarray:
    """머리 인코더 각 deg (pan, tilt) → 관측 2 칸(hdgp HeadModel.to_action: 한계 → ±1). 한계 밖 HEAD_OVER_DEG 까지만 자른다."""
    q = _arr("head_enc_deg", enc_deg, (2,))
    if np.any(q < HEAD_LO - HEAD_OVER_DEG) or np.any(q > HEAD_HI + HEAD_OVER_DEG):
        raise ValueError(f"머리 인코더 각 {q} deg 가 한계 {HEAD_LO} ~ {HEAD_HI} 를 {HEAD_OVER_DEG}° 넘게 벗어났다 — 인코더 wrap · 영점을 본다")
    return np.clip(2.0 * (q - HEAD_LO) / (HEAD_HI - HEAD_LO) - 1.0, -1.0, 1.0)


def head_deg(a_head: ArrayLike) -> np.ndarray:
    """행동 끝 2 칸 → 머리 절대 목표 인코더 각 deg(hdgp HeadModel.from_action)."""
    return HEAD_LO + (np.clip(_arr("a_head", a_head, (2,)), -1.0, 1.0) + 1.0) * 0.5 * (HEAD_HI - HEAD_LO)


def seat_hand_action(a_hand: ArrayLike, base: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> np.ndarray:
    """끼우기 단계 손 잔차 → 절대 법칙 행동(hdgp hand_residual_to_abs, 척도 SEAT_HAND_RESIDUAL_SCALE).
    목표 = clip(base + 척도·a·(hi − lo)), base = 넘긴 순간 손 목표, lo · hi = HandLaw.bounds(). 반환값을 HandLaw 에 넣는다."""
    a, b, lo, hi = (_arr(n, x, (6,)) for n, x in (("a_hand", a_hand), ("base", base), ("lo", lo), ("hi", hi)))
    span = hi - lo
    if np.any(span <= 0.0):
        raise ValueError(f"손 범위 hi − lo {span} 에 0 이하가 있다")
    raw = np.clip(b + SEAT_HAND_RESIDUAL_SCALE * np.clip(a, -1.0, 1.0) * span, lo, hi)
    return np.clip(2.0 * (raw - lo) / span - 1.0, -1.0, 1.0)


def build_obs(arm_q: ArrayLike, arm_qd: ArrayLike, arm_q_target: ArrayLike, hand_q: ArrayLike, hand_target: ArrayLike,
              palm_pos: ArrayLike, palm_R: ArrayLike, prev_action: ArrayLike, stage: int, head_enc_deg: ArrayLike) -> np.ndarray:
    """(62,) float32. hand_q · hand_target = thumb_1, thumb_2, index_1, middle_1, ring_1, pinky_1(rad), palm = r_hl_palm_sensor 의
    base 기준 위치 · 회전행렬 (3, 3), prev_action = 직전 틱 실행 행동 13(에피소드 첫 틱 0), stage = STAGES 번호(정수), 머리 = 실측 인코더 deg.
    sim 은 nan_to_num 으로 덮지만 실기 입력은 모양 · 유한성이 틀리면 거부한다(student_obs 와 같다)."""
    if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)) or not 0 <= stage < len(STAGES):
        raise ValueError(f"단계 {stage!r} 는 정수 0~{len(STAGES) - 1} {STAGES}")
    hq, R = _arr("hand_q", hand_q, (6,)), _arr("palm_R", palm_R, (3, 3))
    parts = [_arr("arm_q", arm_q, (7,)), _arr("arm_qd", arm_qd, (7,)), _arr("arm_q_target", arm_q_target, (7,)), hq,
             np.clip((_arr("hand_target", hand_target, (6,)) - hq) / JOINT_ERR_NORM, -1.0, 1.0), _arr("palm_pos", palm_pos, (3,)),
             np.concatenate([R[:, 0], R[:, 1]]), _arr("prev_action", prev_action, (N_ARM_HAND,)), np.eye(len(STAGES))[stage],
             head_obs(head_enc_deg)]
    assert [p.shape for p in parts] == [(w,) for _, w in PROP_COLS]
    return np.concatenate(parts).astype(np.float32)


def check_meta(d: dict) -> None:
    """ckpt 머리글이 h01 형식(v1 · 관측 62 · 행동 15 · 단계 5 · 머리 켬 · 깊이 창 · 절대 행동)인지 — 하나라도 다르면 ValueError."""
    want = {"kind": "depth", "task": TASK, "arch": None, "n_obs": N_OBS, "n_act": N_ACT, "head": True, "stages": list(STAGES),
            "img_hw": list(IMG_HW), "d_min": D_MIN_M, "d_max": D_MAX_M}
    bad = {k: d.get(k) for k, v in want.items() if d.get(k) != v}
    if d.get("delta", False):                       # 증분 행동 학생은 다른 법칙
        bad["delta"] = d["delta"]
    hc = d.get("head_cfg") or {}
    try:
        lim_ok = np.array_equal(np.asarray([hc["pan_lim"], hc["tilt_lim"]], float).T, [HEAD_LO, HEAD_HI])
    except (KeyError, TypeError, ValueError):
        lim_ok = False
    if not lim_ok:
        bad["head_cfg.pan_lim · tilt_lim"] = hc
    if bad:
        raise ValueError(f"rh_cap 학생 ckpt 형식 어긋남 {bad} — 기대 {want} · delta 없음 · 머리 한계 {HEAD_LO} ~ {HEAD_HI}")


class CapStudent:
    """depth_h01 추론. reset() → 매 틱 step(깊이, 관측, 깊이 나이) → (팔 · 손 행동 13, 머리 목표 deg 2). 한 스레드에서만 부른다."""

    @classmethod
    def from_dir(cls, policy_dir: str | Path, **kw) -> "CapStudent":
        """학생 묶음 폴더(student.yaml: ckpt.path · ckpt.md5 · hdgp.commit = model.py 를 읽는 스냅숏 커밋 · task) — 어긋나면 PinError.
        student_policy.load_student_dir 와 같은 검사다(h01 묶음에는 extrinsics · rgb 가 없어 그 함수를 그대로 못 쓴다)."""
        import yaml

        d = Path(policy_dir).expanduser().resolve()
        p = d / STUDENT_CARD
        if not p.is_file():
            raise hdgp_pin.PinError(f"{p} 가 없다")
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        try:
            ckpt, md5 = d / cfg["ckpt"]["path"], str(cfg["ckpt"]["md5"])
            commit, task = str(cfg["hdgp"]["commit"]), str(cfg["task"])
        except (KeyError, TypeError) as e:
            raise hdgp_pin.PinError(f"{p}: ckpt.path · ckpt.md5 · hdgp.commit · task 가 필요하다 ({e!r})")
        if not ckpt.is_file() or not ckpt.resolve().is_relative_to(d):
            raise hdgp_pin.PinError(f"{p}: ckpt {ckpt} 가 없거나 폴더 밖이다")
        snap = os.environ.get("HDGP_STUDENT_SNAPSHOT_COMMIT") or hdgp_pin.load_snapshot_cfg(SNAPSHOT_CFG)["commit"]
        if task != TASK or commit != snap:
            raise hdgp_pin.PinError(f"{p}: task {task!r} · hdgp.commit {commit} — 기대 {TASK} · 스냅숏 {snap}({SNAPSHOT_CFG.name})")
        return cls(ckpt, md5, **kw)

    def __init__(self, ckpt: str | Path, md5: str, device: str = "cpu", num_threads: int = 4,
                 max_depth_age_s: float = MAX_DEPTH_AGE_S) -> None:
        got = hdgp_pin.md5(Path(ckpt))
        if got != md5:
            raise hdgp_pin.PinError(f"학생 ckpt md5 {got} ≠ 기대 {md5}: {ckpt}")
        import torch

        torch.set_num_threads(int(num_threads))
        d = torch.load(ckpt, map_location=device, weights_only=True)
        check_meta(d)
        mod = load_model_module()
        model = mod.make_depth_student(N_OBS, N_ACT, hid=int(d["hid"]), n_aux=int(d["n_aux"]), pretrained=False, arch="v1").to(device)
        model.load_state_dict(d["model"])                 # strict: 빠지거나 남는 가중치는 오류
        model.eval()
        self.md5 = got
        self.meta = {k: v for k, v in d.items() if k != "model"}
        log.info("rh_cap 학생 ckpt %s md5 %s it %s · model.py %s", ckpt, got, d.get("it"), mod.pinned_src)
        self._attach(mod, model, d["d_min"], d["d_max"], device, max_depth_age_s)
        self._forward(np.zeros(IMG_HW, np.float32), np.zeros(N_OBS, np.float32))     # 워밍업(첫 틱 지연 제거) — 상태는 안 바뀐다

    def _attach(self, mod, model, d_min: float, d_max: float, device: str = "cpu",
                max_depth_age_s: float = MAX_DEPTH_AGE_S) -> None:
        """모델을 붙이고 reset — 시험은 ckpt 없이 가짜 mod · model 을 여기로 넣는다."""
        import torch

        self.torch, self.mod, self.model, self.device = torch, mod, model, device
        self.d_min, self.d_max, self.max_depth_age_s = float(d_min), float(d_max), float(max_depth_age_s)
        self.reset()

    def reset(self) -> None:
        """에피소드 시작에만 — LSTM 상태 0, 다음 틱 깊이 0(sim learner blank_new)."""
        self.hc = self.model.zero_state(1, self.device)
        self.first = True

    def _forward(self, depth_m: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, object]:
        """(행동 15 ±1, 새 LSTM 상태). self.hc 는 쓰지 않는다 — 출력이 유한하지 않으면 FloatingPointError."""
        t = self.torch
        with t.inference_mode():
            img = self.mod.depth_input(t.from_numpy(depth_m).to(self.device)[None], self.d_min, self.d_max)
            a, _, hc = self.model(img, t.from_numpy(obs).to(self.device)[None], self.hc, train_cnn=False)
        a = a[0].float().cpu().numpy()
        if a.shape != (N_ACT,) or not np.isfinite(a).all():
            raise FloatingPointError(f"학생 출력이 유한한 ({N_ACT},) 가 아니다: {a}")
        return np.clip(a, -1.0, 1.0), hc

    def step(self, depth_m: ArrayLike, obs: ArrayLike, depth_age_s: float) -> tuple[np.ndarray, np.ndarray]:
        """(90, 160) 깊이 [m](구멍 = 0, 유한, 0 ~ DEPTH_MAX_M) · build_obs (62,) · 깊이 나이 [s] → (팔 7 + 손 6 행동 ±1, 머리 목표 인코더 deg).
        나이가 0 ~ 상한 밖(NaN 포함)이면 StaleDepth, 입력 모양 · 값이 틀리면 ValueError, 출력이 유한하지 않으면 FloatingPointError.
        어느 것이든 LSTM 상태 · 첫 틱 표시는 그대로다."""
        if not 0.0 <= depth_age_s <= self.max_depth_age_s:
            raise StaleDepth(f"깊이 나이 {depth_age_s} s ∉ 0 ~ {self.max_depth_age_s} s")
        depth, o = np.asarray(depth_m, np.float32), np.asarray(obs, np.float32)
        if depth.shape != IMG_HW or not np.isfinite(depth).all() or depth.min() < 0.0 or depth.max() > DEPTH_MAX_M:
            raise ValueError(f"깊이는 유한한 {IMG_HW} [m] 0 ~ {DEPTH_MAX_M}: 모양 {depth.shape}"
                             + (f" 범위 {np.nanmin(depth)} ~ {np.nanmax(depth)}" if depth.size else ""))
        if o.shape != (N_OBS,) or not np.isfinite(o).all():
            raise ValueError(f"관측은 유한한 ({N_OBS},): 모양 {o.shape}")
        a, hc = self._forward(np.zeros_like(depth) if self.first else depth, o)
        self.hc, self.first = hc, False
        return a[:N_ARM_HAND], head_deg(a[N_ARM_HAND:])

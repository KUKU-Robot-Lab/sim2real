"""11.04 FP++ 없는 깊이 학생(rh_aglt 잡기) 추론 — hdgp distillation/student/model.py 를 파일로 불러 ckpt 를 적재한다.

model.py 는 torch · torchvision 만 쓴다. openarm 패키지 import 는 isaaclab 을 끌어오므로 파일 하나만 불러온다.
hdgp 코드는 공유 작업 트리가 아니라 고정 커밋 사본에서 읽는다(load_pinned: config/hdgp_student_snapshot.yaml 의 커밋 · md5 핀,
hdgp_pin.snapshot). scripts/nodes/head_depth_node.py 의 real_depth.py 도 같은 함수로 읽는다.
ckpt 키는 hdgp learner.save(n_obs · n_act · hid · img_hw · d_min · d_max · kind · arch · task · stages · n_aux · head · delta ·
depth_pre · prop_cols · handover · n_vis). 받는 형식: kind depth · arch v2(깊이) 또는 v3(깊이 + RGB) · 증분 없음 · 머리 칸 없음 ·
prop_cols = PROP_COLS · depth_pre 있음(real_pipe.meta). depth_pre 와 깊이 노드 설정(/head_depth/config)의 대조는 check_depth_config.
깊이: (90, 160) float32 m(0 = 구멍) → model.depth_input(d, d_min, d_max). v3 는 rgb (90, 160, 3) float32 0~1(head_depth_node
/head_depth/rgb, 깊이와 같은 stamp)을 (1, 3, h, w) 로 뒤에 붙여 6 채널. reset() 뒤 첫 틱은 깊이 · rgb 모두 0(sim learner
blank_new 와 같다: 새 에피소드 첫 틱의 렌더는 리셋 전 장면이라 0 을 넣는다).
학생 묶음 폴더(student.yaml: ckpt · 얼린 외부 파라미터 · hdgp 커밋 + md5)는 load_student_dir · StudentPolicy.from_dir 로 읽는다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType

import numpy as np

from . import hdgp_pin
from .rh_aglt import RhAgltError
from .student_mask import CLASSES, MaskPost, check_mask_meta, check_target, class_index, decode
from .student_obs import PROP_COLS, STAGES

N_ACT = 13
ARCHS = ("v2", "v3")      # v3 = 깊이 + RGB 줄기
STUDENT_CARD = "student.yaml"  # policy.yaml 이 아니다: policy_registry 가 policy.yaml 폴더를 정책 카드로 읽는다
SNAPSHOT_CFG = Path(__file__).resolve().parents[1] / "config" / "hdgp_student_snapshot.yaml"
REAL_DEPTH = "source/openarm/openarm/distillation/real_depth.py"
MODEL = "source/openarm/openarm/distillation/student/model.py"
MAX_DEPTH_AGE_S = 0.2
HANDOVER_ARGS = ("tmin_s", "still_s", "still_mps")
EMPTY_WARN_TICKS = 30

log = logging.getLogger(__name__)


class StaleDepth(RuntimeError):
    """깊이 프레임이 상한보다 오래됐다 — 호출자가 잡아 정지로 보낸다."""


def load_pinned(rel: str, name: str, cfg_path: Path = SNAPSHOT_CFG) -> tuple[ModuleType, dict]:
    """고정 커밋 사본의 hdgp 파일 하나 → (모듈, {path, md5, commit}). md5 핀과 다르면 PinError(커밋을 바꿔도 핀 검사는 그대로)."""
    cfg = hdgp_pin.load_snapshot_cfg(cfg_path)
    pins = cfg.get("md5")
    if rel not in cfg["paths"] or not isinstance(pins, dict) or not pins.get(rel):
        raise hdgp_pin.PinError(f"{cfg_path}: {rel} 가 paths · md5 핀에 없다")
    commit = os.environ.get("HDGP_STUDENT_SNAPSHOT_COMMIT") or cfg["commit"]
    snap = hdgp_pin.snapshot(hdgp_pin.hdgp_tree(), commit, cfg["paths"])
    path = snap / rel
    got = hdgp_pin.md5(path)
    if got != pins[rel]:
        raise hdgp_pin.PinError(f"hdgp {commit[:9]} {rel} md5 {got} ≠ 핀 {pins[rel]} — 커밋을 바꿨으면 {cfg_path.name} md5 도 같이 고친다")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sha = (snap / ".complete").read_text().split()[0]
    return mod, {"path": str(path), "md5": got, "commit": sha}


def load_model_module() -> ModuleType:
    """hdgp model.py(고정 사본)만 파일로 불러온다."""
    mod, src = load_pinned(MODEL, "hdgp_student_model")
    log.info("학생 model.py %s md5 %s 커밋 %s", src["path"], src["md5"], src["commit"])
    mod.pinned_src = src
    return mod


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def check_meta(d: dict, task: str) -> None:
    """ckpt 머리글이 이 배포 형식인지 — 하나라도 다르면 거부."""
    want = {"kind": "depth", "task": task, "n_obs": sum(w for _, w in PROP_COLS), "n_act": N_ACT,
            "delta": False, "head": False, "stages": list(STAGES)}
    bad = {k: d.get(k) for k, v in want.items() if d.get(k) != v}
    if d.get("arch") not in ARCHS:
        bad["arch"] = d.get("arch")
        want["arch"] = ARCHS
    if bad:
        raise ValueError(f"학생 ckpt 형식 어긋남 {bad} — 기대 { {k: want[k] for k in bad} }")
    cols = [tuple(x) for x in d.get("prop_cols") or ()]
    if cols != list(PROP_COLS):
        raise ValueError(f"학생 ckpt prop_cols {cols} ≠ {list(PROP_COLS)}")
    ho = d.get("handover") or {}
    if ho.get("rule") != "DeployHandover" or ho.get("deploy_uses_rr") or any(k not in ho for k in HANDOVER_ARGS):
        raise ValueError(f"학생 ckpt 넘김 규칙 {ho} — DeployHandover({' · '.join(HANDOVER_ARGS)}, deploy_uses_rr 없음)가 아니다")
    if not isinstance(d.get("depth_pre"), dict) or not d["depth_pre"]:
        raise ValueError("학생 ckpt depth_pre 없음 — 깊이 전처리를 노드 설정과 대조할 수 없다")
    if (d["arch"] == "v3") != ("rgb" in d["depth_pre"]):
        raise ValueError(f"학생 ckpt arch {d['arch']} 와 depth_pre rgb {'rgb' in d['depth_pre']} 가 맞지 않는다(v3 = rgb 학습)")
    if d.get("mask"):                  # 11.07 마스크 학생: 둘째 줄기 = 마스크 3 채널(rgb 칸 자리, depth_pre rgb 문구는 그대로 남아 있다)
        if d["arch"] != "v3":
            raise ValueError(f"마스크 학생 ckpt arch {d['arch']} — v3(둘째 줄기)만")
        check_mask_meta(d["mask"])


def check_depth_config(depth_pre: dict, node_cfg: dict, mask: bool = False) -> None:
    """ckpt depth_pre(hdgp real_pipe.meta) ↔ head_depth_node 설정(/head_depth/config JSON). 다르면 RhAgltError.
    mask(마스크 학생)면 depth_pre 에 rgb 문구가 있어도 노드 rgb 는 꺼져 있어야 한다(둘째 줄기는 /head_mask)."""
    def norm(x):
        return list(x) if isinstance(x, (list, tuple)) else x

    bad = []
    for k in ("block", "out_wh", "render_wh", "extrinsics_md5"):
        a, b = depth_pre.get(k), node_cfg.get(k)
        if a is None or b is None or norm(a) != norm(b):
            bad.append(f"{k} ckpt {a!r} ≠ 노드 {b!r}")
    for k, atol in (("min_valid", 1e-6), ("floor_cut_z_m", 1e-6), ("K", 1e-3)):
        a, b = depth_pre.get(k), node_cfg.get(k)
        if a is None or b is None or np.shape(a) != np.shape(b) or not np.allclose(a, b, rtol=0, atol=atol):
            bad.append(f"{k} ckpt {a!r} ≠ 노드 {b!r}")
    want_rgb = "rgb" in depth_pre and not mask
    if want_rgb != bool(node_cfg.get("rgb")):
        bad.append(f"rgb ckpt {want_rgb} ≠ 노드 {bool(node_cfg.get('rgb'))}(head_depth_node --rgb)")
    if bad:
        raise RhAgltError("깊이 전처리 불일치 — " + " · ".join(bad))


def load_student_dir(policy_dir: str | Path, snapshot_cfg: Path = SNAPSHOT_CFG) -> dict:
    """학생 묶음 폴더의 student.yaml → 검사한 설정(ckpt · extrinsics 는 절대 경로로). 하나라도 어긋나면 PinError.

    검사: 얼린 외부 파라미터 파일 md5 = student.yaml(ckpt 는 StudentPolicy 가 md5 · depth_pre 로 다시 본다),
    student.yaml hdgp.commit = hdgp 스냅숏 설정 커밋(HDGP_STUDENT_SNAPSHOT_COMMIT 로 바꿔 놓았어도 거부)."""
    import yaml

    d = Path(policy_dir).expanduser().resolve()
    p = d / STUDENT_CARD
    if not p.is_file():
        raise hdgp_pin.PinError(f"{p} 가 없다")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        ck, ex, commit = cfg["ckpt"], cfg["extrinsics"], str(cfg["hdgp"]["commit"])
        ck = {**ck, "path": d / ck["path"], "md5": str(ck["md5"])}
        ex = {**ex, "path": d / ex["path"], "md5": str(ex["md5"])}
        task, rgb = str(cfg["task"]), bool(cfg["rgb"])
    except (KeyError, TypeError) as e:
        raise hdgp_pin.PinError(f"{p}: ckpt.path · ckpt.md5 · extrinsics.path · extrinsics.md5 · hdgp.commit · task · rgb 가 필요하다 ({e!r})")
    for name, f in (("ckpt", ck), ("extrinsics", ex)):
        if not f["path"].is_file() or not f["path"].resolve().is_relative_to(d):
            raise hdgp_pin.PinError(f"{p}: {name} {f['path']} 가 없거나 폴더 밖이다")
    got = hdgp_pin.md5(ex["path"])
    if got != ex["md5"]:
        raise hdgp_pin.PinError(f"외부 파라미터 {ex['path']} md5 {got} ≠ student.yaml {ex['md5']}")
    snap = os.environ.get("HDGP_STUDENT_SNAPSHOT_COMMIT") or hdgp_pin.load_snapshot_cfg(snapshot_cfg)["commit"]
    if snap != commit:
        raise hdgp_pin.PinError(f"hdgp 스냅숏 커밋 {snap} ≠ student.yaml hdgp.commit {commit} — {snapshot_cfg.name} 와 묶음을 맞춘다")
    return {**cfg, "dir": d, "ckpt": ck, "extrinsics": ex, "task": task, "rgb": rgb}


class StudentPolicy:
    @classmethod
    def from_dir(cls, policy_dir: str | Path, target_class: str = "", **kw) -> "StudentPolicy":
        """학생 묶음 폴더(load_student_dir) → StudentPolicy. ckpt depth_pre 외부 파라미터 md5 · rgb · mask 가 student.yaml 과 다르면 거부.
        마스크 학생은 target_class(빈 값 = student.yaml mask.target_class)로 목표 클래스를 정한다."""
        cfg = load_student_dir(policy_dir)
        pol = cls(cfg["ckpt"]["path"], cfg["ckpt"]["md5"], cfg["task"], **kw)
        if pol.depth_pre.get("extrinsics_md5") != cfg["extrinsics"]["md5"]:
            raise RhAgltError(f"ckpt depth_pre extrinsics_md5 {pol.depth_pre.get('extrinsics_md5')} ≠ student.yaml {cfg['extrinsics']['md5']}")
        if pol.rgb_on != cfg["rgb"]:
            raise RhAgltError(f"ckpt arch {pol.meta['arch']}(rgb {pol.rgb_on}) ≠ student.yaml rgb {cfg['rgb']}")
        card = cfg.get("mask") or {}
        if pol.mask_on != bool(card):
            raise RhAgltError(f"ckpt mask {pol.meta.get('mask')} ↔ student.yaml mask {card or '없음'} — 마스크 학생과 묶음 종류가 다르다")
        if pol.mask_on:
            try:
                y = {"path": cfg["dir"] / card["yolo"]["path"], "md5": str(card["yolo"]["md5"])}
                pol.set_target(target_class or str(card["target_class"]))
            except (KeyError, TypeError, ValueError) as e:
                raise hdgp_pin.PinError(f"student.yaml mask: target_class · yolo.path · yolo.md5 가 필요하다 ({e!r})")
            if not y["path"].is_file() or hdgp_pin.md5(y["path"]) != y["md5"]:
                raise hdgp_pin.PinError(f"YOLO 가중치 {y['path']} 가 없거나 md5 ≠ student.yaml {y['md5']}")
            try:
                pol.target_warn = check_target(pol.meta.get("one_obj"), CLASSES[pol.target])
            except ValueError as exc:
                raise RhAgltError(str(exc)) from exc
            pol.yolo = {**card["yolo"], **y}          # conf · imgsz · 인스턴스 규칙은 /head_mask/config 대조용
        pol.policy_cfg = cfg
        return pol

    def __init__(self, ckpt: str | Path, md5: str, task: str, device: str = "cpu", num_threads: int = 4,
                 max_depth_age_s: float = MAX_DEPTH_AGE_S, empty_warn_ticks: int = EMPTY_WARN_TICKS) -> None:
        import torch

        ckpt = Path(ckpt)
        got = _md5(ckpt)
        if got != md5:
            raise ValueError(f"학생 ckpt md5 {got} ≠ 기대 {md5}: {ckpt}")
        torch.set_num_threads(int(num_threads))
        d = torch.load(ckpt, map_location=device, weights_only=True)
        check_meta(d, task)
        self.mod = load_model_module()
        self.model = self.mod.make_depth_student(int(d["n_obs"]), N_ACT, hid=int(d["hid"]), n_aux=int(d["n_aux"]),
                                                 pretrained=False, arch=d["arch"], n_vis=int(d.get("n_vis") or 0)).to(device)
        self.model.load_state_dict(d["model"])            # strict: 빠지거나 남는 가중치는 오류
        self.model.eval()
        self.torch, self.device = torch, device
        self.img_hw = tuple(int(x) for x in d["img_hw"])
        self.mask_on = bool(d.get("mask"))
        self.rgb_on = d["arch"] == "v3" and not self.mask_on     # 마스크 학생은 둘째 줄기에 rgb 대신 마스크 3 채널
        self.target = None                                       # 마스크 학생 목표 클래스 번호(set_target)
        self.target_warn = None                                  # 목표가 학습 목표 색과 다르면 문구(from_dir)
        self.d_min, self.d_max = float(d["d_min"]), float(d["d_max"])
        self.handover = {k: float(d["handover"][k]) for k in HANDOVER_ARGS}   # StudentHandover 인자(train_near_rr 는 학습 전용)
        self.depth_pre = d["depth_pre"]
        self.meta = {k: v for k, v in d.items() if k != "model"}
        self.md5 = got
        self.max_depth_age_s, self.empty_warn_ticks = float(max_depth_age_s), int(empty_warn_ticks)
        log.info("학생 ckpt %s md5 %s · model.py %s", ckpt, got, self.mod.pinned_src)
        self.reset()
        self.mpost = MaskPost(self.d_min, self.d_max) if self.mask_on else None
        rgb0 = np.zeros((*self.img_hw, 3), np.float32) if d["arch"] == "v3" else None
        self._forward(np.zeros(self.img_hw, np.float32), np.zeros(self.model.n_prop, np.float32), rgb0)   # 워밍업(첫 틱 지연 제거)
        self.reset()

    def set_target(self, name: str) -> None:
        """마스크 학생 목표 클래스(blue · pink · orange). 나머지 클래스는 이웃."""
        self.target = class_index(name)

    def reset(self) -> None:
        """에피소드 시작에만 — LSTM 상태 0, 다음 틱 깊이 0."""
        self.hc = self.model.zero_state(1, self.device)
        self.first = True
        self.n_empty = 0
        self.depth_empty = False      # 유효 화소 0 인 깊이가 empty_warn_ticks 틱 연속이면 True(호출자가 경고)

    def _forward(self, depth_m: np.ndarray, prop: np.ndarray, rgb: np.ndarray | None = None) -> np.ndarray:
        t = self.torch
        with t.inference_mode():
            img = self.mod.depth_input(t.from_numpy(depth_m).to(self.device)[None], self.d_min, self.d_max)
            if rgb is not None:                                   # (h, w, 3) → (1, 3, h, w), 깊이 3 채널 뒤(hdgp learner 와 같은 순서)
                img = t.cat([img, t.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None]], 1)
            a, _, self.hc = self.model(img, t.from_numpy(prop).to(self.device)[None], self.hc, train_cnn=False)
        return np.clip(a[0].float().cpu().numpy(), -1.0, 1.0)

    def check_inputs(self, depth_m: np.ndarray, depth_age_s: float, rgb: np.ndarray | None = None) -> None:
        """깊이 나이 · 깊이 모양 · rgb 형식 검사. StudentChain 이 넘김 · 디코더 상태를 바꾸기 전에 부른다(실행 안 된 틱을 세지 않게).
        나이 > 상한이면 StaleDepth, 모양 · 형식이 틀리면 ValueError. 마스크 학생은 rgb 자리에 /head_mask 비트 (h, w) uint8."""
        if not depth_age_s <= self.max_depth_age_s:          # NaN 도 거부
            raise StaleDepth(f"깊이 나이 {depth_age_s} s > 상한 {self.max_depth_age_s} s")
        if np.shape(depth_m) != self.img_hw:
            raise ValueError(f"깊이 모양 {np.shape(depth_m)} ≠ {self.img_hw}")
        if self.mask_on:
            if self.target is None:
                raise ValueError("마스크 학생 목표 클래스가 없다(set_target)")
            if rgb is None or np.shape(rgb) != self.img_hw or np.asarray(rgb).dtype != np.uint8:
                raise ValueError(f"마스크 비트는 {self.img_hw} uint8 이어야 한다: {None if rgb is None else np.shape(rgb)}")
            return
        if (rgb is not None) != self.rgb_on:
            raise ValueError(f"rgb 입력 {rgb is not None} ≠ ckpt arch {self.meta['arch']}(v3 만 rgb)")
        if rgb is not None:
            rgb = np.asarray(rgb, np.float32)
            if rgb.shape != (*self.img_hw, 3) or not (np.isfinite(rgb).all() and rgb.min() >= 0.0 and rgb.max() <= 1.0):
                raise ValueError(f"rgb 는 {(*self.img_hw, 3)} float 0~1 이어야 한다: {rgb.shape}")

    def step(self, depth_m: np.ndarray, prop: np.ndarray, depth_age_s: float, rgb: np.ndarray | None = None) -> np.ndarray:
        """(90, 160) 깊이 [m] · (60,) 관측 · 깊이 나이 [s] (· v3: 같은 프레임 rgb (90, 160, 3) 0~1) → (13,) 행동(±1 클립, env 와 같게).
        나이 > 상한이면 StaleDepth."""
        self.check_inputs(depth_m, depth_age_s, rgb)
        depth_m, prop = np.asarray(depth_m, np.float32), np.asarray(prop, np.float32)
        if prop.shape != (self.model.n_prop,):
            raise ValueError(f"관측 모양 {prop.shape} ≠ ({self.model.n_prop},)")
        if self.mask_on:            # hdgp post 는 렌더 깊이 경로 출력(리셋 첫 틱 포함)으로 돈다 — 첫 틱 0 으로 바꾸기 전에
            t = self.torch
            m = t.from_numpy(decode(rgb, self.target))[None]
            img2 = self.mpost.post(m, t.from_numpy(depth_m)[None], t.tensor([self.first]))
            rgb = img2[0].permute(1, 2, 0).contiguous().numpy()
        elif rgb is not None:
            rgb = np.asarray(rgb, np.float32)
        self.n_empty = 0 if (depth_m > 0).any() else self.n_empty + 1
        self.depth_empty = self.n_empty >= self.empty_warn_ticks
        if self.first:
            depth_m, self.first = np.zeros_like(depth_m), False
            rgb = None if rgb is None else np.zeros_like(rgb)
        return self._forward(depth_m, prop, rgb)

"""11.07 목표 마스크 학생(DISTILL_MASK=A) 실기 입력 — hdgp 78e75059 scripts/tools/student_adapters/mask_scene.py 를 옮긴다.

검출 노드(scripts/nodes/head_mask_node.py, .venv-yolo)가 YOLO11s-seg 클래스 마스크(1280x720)를 hdgp TargetMask.full 과 같은
  블록 규칙(8x8 에서 절반 이상)으로 160x90 비트 영상 하나로 줄여 /head_mask/image(mono8)로 낸다.
  비트 c(0~2) = 클래스 c, 비트 3+c = "c 가 아닌 클래스 & ~c"(hdgp full: 원해상도에서 o &= ~t 한 뒤 줄인다 — 목표를 모르는
  검출 노드가 목표마다 미리 줄여 둔다). 목표 클래스는 학생 노드가 고른다(decode).
학생 노드는 깊이 프레임과 stamp 로 짝지은 비트 영상을 MaskPost(= TargetMask.post 모드 A, 증강 없음)에 넣어
  [목표 마스크 안 깊이, 이웃 마스크 안 깊이, 보임 면] (h, w, 3) 0~1 을 만든다 → v3 둘째 줄기(rgb 자리, learner 와 같은 순서).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

CLASSES = ("blue", "pink", "orange")      # 1105_det data.yaml names {0: blue(source240_blue), 1: pink(source240_pink), 2: orange(shaker_c)}
MASK_TOPIC, MASK_CONFIG_TOPIC = "/head_mask/image", "/head_mask/config"
VIS_MIN_PX = 3                            # hdgp mask_scene.VIS_MIN_PX(ckpt mask.vis_min_px 와 대조)
IMG2 = ["tgt_mask_depth", "nbr_mask_depth", "visible_plane"]
MASK_SLOP_S = 0.5 / 30.0                  # 깊이 · 마스크(color) stamp 짝 허용 = head_depth_node.RGB_SLOP_S
MAX_MASK_AGE_S = 0.2                      # 쓰는 마스크의 나이(깊이 stamp 기준) 상한 = MAX_DEPTH_AGE_S. 넘으면 틱 오류(단절 · 노드 죽음)
BAD_FRAMES_MAX = 5                        # 불량 /head_mask/image 가 이만큼 연속이면 틱 오류
PAIR_WINDOW = 30                          # 최근 깊이 프레임 수 — 짝 없는 비율을 이 창에서 센다
UNPAIRED_WARN = 0.3                       # 창 안 짝 없는 비율이 이 이상이면 경고(YOLO 15 Hz 반반도 잡힌다)
#: ckpt one_obj → 허용 목표 클래스(첫 값 = 학습 목표 색). source240 파랑 · 분홍은 형상이 같다
OBJ_TARGETS = {"source240_cvx": ("blue", "pink"), "shaker_c_cvx": ("orange",)}
#: 검출 노드 인스턴스 고르기 기본(11.07 검증: 탁자 밖 오검출이 보임 판정을 뚫는다) — student.yaml mask.yolo 와 대조
DET_RULES = {"top1_per_class": True, "cross_class_iou": 0.5}


class MaskStale(RuntimeError):
    """마스크 스트림 단절 · 불량 연속 — 틱 오류(max_gap_ticks → abort). 검출이 빈 메시지는 여기 해당하지 않는다."""


def class_index(name: str) -> int:
    if name not in CLASSES:
        raise ValueError(f"target_class {name!r} 는 {CLASSES} 중 하나")
    return CLASSES.index(name)


def check_target(one_obj: str | None, name: str) -> str | None:
    """목표 클래스 ↔ ckpt one_obj. 대응이 없으면 ValueError, 학습 목표 색과 다르면 경고 문구(None = 그대로)."""
    class_index(name)
    ok = OBJ_TARGETS.get(str(one_obj))
    if ok is None:
        raise ValueError(f"ckpt one_obj {one_obj!r} 의 목표 클래스 대응이 없다({sorted(OBJ_TARGETS)})")
    if name not in ok:
        raise ValueError(f"target_class {name} 는 ckpt one_obj {one_obj} 와 맞지 않는다(허용 {ok})")
    return None if name == ok[0] else f"목표 {name} 는 학습 목표 색 {ok[0]} 과 다르다(같은 형상 {one_obj})"


def block_half(m: torch.Tensor, b: int) -> torch.Tensor:
    """(n, H, W) bool → (n, H/b, W/b) bool. hdgp TargetMask.full 의 줄을 그대로 쓴다."""
    n, H, W = m.shape
    return (m.float().reshape(n, H // b, b, W // b, b).mean((2, 4)) >= 0.5)


def encode_bits(cls_mask: torch.Tensor, b: int) -> torch.Tensor:
    """(3, H, W) bool 원해상도 클래스 마스크 → (H/b, W/b) uint8 비트(위 docstring). 장치는 입력과 같다."""
    if cls_mask.shape[0] != len(CLASSES) or cls_mask.shape[1] % b or cls_mask.shape[2] % b:
        raise ValueError(f"클래스 마스크 {tuple(cls_mask.shape)} 는 ({len(CLASSES)}, H, W), H · W 는 블록 {b} 의 배수")
    k = len(CLASSES)
    oth = torch.stack([cls_mask[[j for j in range(k) if j != c]].any(0) & ~cls_mask[c] for c in range(k)])
    t, o = block_half(cls_mask, b), block_half(oth, b)
    bits = torch.zeros(t.shape[1:], dtype=torch.uint8, device=cls_mask.device)
    for c in range(k):
        bits |= t[c].to(torch.uint8) << c
        bits |= o[c].to(torch.uint8) << (k + c)
    return bits


def decode(bits: np.ndarray, target: int) -> np.ndarray:
    """(h, w) uint8 비트 → (2, h, w) float32 [목표, 이웃] 0/1 (hdgp tm.raw 의 두 채널)."""
    b = np.asarray(bits, np.uint8)
    return np.stack([(b >> target) & 1, (b >> (len(CLASSES) + target)) & 1]).astype(np.float32)


def mask_array(msg, shape: tuple) -> np.ndarray:
    """sensor_msgs/Image mono8 (h, w) → uint8 비트. 모양 · 형식 · 비트 범위가 다르면 ValueError."""
    h, w = shape
    if msg.encoding != "mono8" or (msg.height, msg.width) != (h, w) or msg.step != w:
        raise ValueError(f"마스크 {msg.encoding} {msg.height}x{msg.width} step {msg.step} ≠ mono8 {h}x{w}")
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(h, w).copy()
    if int(a.max(initial=0)) >= 1 << (2 * len(CLASSES)):
        raise ValueError(f"마스크 비트 {int(a.max())} 범위 밖(6 비트)")
    return a


class MaskPost:
    """hdgp TargetMask.post 모드 A 를 줄 단위로 옮긴 것(N env, 증강 · 통계 · 덤프 없음 = aug False 경로).
    목표 화소 < VIS_MIN_PX 면 마지막으로 보인 마스크 유지 + 보임 0. new_ep(에피소드 첫 틱)에서 유지 마스크를 비운다."""

    def __init__(self, d_min: float, d_max: float) -> None:
        self.d_min, self.d_max = float(d_min), float(d_max)
        self.last = None
        self.vis = None

    def post(self, m: torch.Tensor, d: torch.Tensor, new_ep: torch.Tensor) -> torch.Tensor:
        """m (N, ≥2, h, w) (0 = 목표, 1 = 이웃), d (N, h, w) 학생 깊이 [m], new_ep (N,) bool → (N, 3, h, w)."""
        m = (m[:, :2] > 0.5).float()
        n, _, h, w = m.shape
        if self.last is None:
            self.last = torch.zeros_like(m)
        if bool(new_ep.any()):
            self.last[new_ep] = 0.0
        drop = torch.zeros(n, dtype=torch.bool, device=m.device)
        px = m[:, 0].sum((1, 2))
        seen = (px >= VIS_MIN_PX) & ~drop
        self.last = torch.where(seen[:, None, None, None], m, self.last)
        use = torch.where(drop[:, None, None, None], torch.zeros_like(m), self.last)    # 누락 = 빈 마스크, 가림 = 마지막 마스크
        self.vis = seen.float()
        dn = torch.where((d >= self.d_min) & (d <= self.d_max), d / self.d_max, torch.zeros_like(d))
        return torch.stack([dn * use[:, 0], dn * use[:, 1], self.vis[:, None, None].expand(n, h, w)], 1)


def check_mask_meta(mask: dict) -> None:
    """ckpt mask 머리글이 이 배포가 옮긴 규칙(모드 A · 3 채널 순서 · VIS_MIN_PX)인지."""
    bad = {k: mask.get(k) for k, v in (("mode", "A"), ("img2", IMG2), ("vis_min_px", VIS_MIN_PX)) if mask.get(k) != v}
    if bad:
        raise ValueError(f"학생 ckpt mask {bad} — 배포는 mode A · img2 {IMG2} · vis_min_px {VIS_MIN_PX} 만 옮겼다")


def check_mask_config(node_cfg: dict, yolo: dict, depth_pre: dict) -> None:
    """/head_mask/config(검출 노드) ↔ 묶음 student.yaml mask.yolo(md5 · conf · imgsz · retina_masks · 인스턴스 규칙)
    · ckpt depth_pre(block · out_wh · render_wh). 다르거나 student.yaml 에 값이 없으면 ValueError."""
    keys = ("conf", "imgsz", "retina_masks", *DET_RULES)
    miss = [k for k in ("md5", *keys) if k not in yolo]
    if miss:
        raise ValueError(f"student.yaml mask.yolo 에 {miss} 가 없다")
    want = {"weights_md5": str(yolo["md5"]), **{k: yolo[k] for k in keys}, "classes": list(CLASSES),
            "block": depth_pre.get("block"), "out_wh": list(depth_pre.get("out_wh") or ()),
            "render_wh": list(depth_pre.get("render_wh") or ())}
    got = {k: (list(node_cfg.get(k)) if isinstance(node_cfg.get(k), (list, tuple)) else node_cfg.get(k)) for k in want}
    bad = [f"{k} 노드 {got[k]!r} ≠ 기대 {want[k]!r}" for k in want if got[k] != want[k]]
    if bad:
        raise ValueError("마스크 노드 설정 불일치 — " + " · ".join(bad))


def pick_mask(buf, depth_stamp_ns: int, slop_s: float = MASK_SLOP_S):
    """[(stamp_ns, bits)] → (bits | None, 출처 'paired' | 'latest' | 'none', 나이 s | None).
    깊이 stamp 에 가장 가까운 것(slop 안) = paired, 없으면 가장 최근 것 = latest. 나이 = 깊이 stamp − 마스크 stamp(앞선 마스크는 0)."""
    if not buf:
        return None, "none", None
    t, m = min(buf, key=lambda x: abs(x[0] - depth_stamp_ns))
    if abs(t - depth_stamp_ns) <= slop_s * 1e9:
        return m, "paired", abs(t - depth_stamp_ns) * 1e-9
    t, m = max(buf, key=lambda x: x[0])
    return m, "latest", max(0, depth_stamp_ns - t) * 1e-9


class MaskStream:
    """/head_mask/image 수신 · 짝 · 생존(학생 노드 공용, rclpy 없이 시험한다).

    틱 오류(MaskStale)는 두 경우뿐이다: 쓸 마스크가 max_age_s 보다 낡음(한 번도 없음 포함 — 토픽 단절 · 노드 죽음),
    불량 프레임 bad_max 연속. 검출이 빈 메시지(목표 비트 < VIS_MIN_PX)는 정상이다 — MaskPost 가 마지막 마스크 유지
    + 보임 0 으로 처리한다(잡기 중 손 가림으로 19~25 프레임 놓침이 실측, sim_e2e 11.07). 연속 놓침은 status 에만 싣는다."""

    def __init__(self, hw: tuple[int, int], max_age_s: float = MAX_MASK_AGE_S, bad_max: int = BAD_FRAMES_MAX,
                 keep: int = 8, slop_s: float = MASK_SLOP_S, window: int = PAIR_WINDOW) -> None:
        if not (max_age_s > 0 and bad_max >= 1):
            raise ValueError(f"max_mask_age_s > 0 · mask_bad_frames_max ≥ 1 이어야 한다: {max_age_s} · {bad_max}")
        self.hw, self.max_age_s, self.bad_max, self.keep = tuple(hw), float(max_age_s), int(bad_max), int(keep)
        self.slop_ns, self.window = slop_s * 1e9, int(window)
        self.buf: deque = deque(maxlen=self.keep)
        self.last_st = None
        self.reset()

    def reset(self) -> None:
        """에피소드 시작 — 불량 연속 · 짝 · 놓침 통계만 비운다. 받은 마스크는 남긴다: 나이 상한(max_age_s)이 신선도를 보장하고,
        비우면 첫 틱들이 YOLO 다음 프레임까지 MaskStale 로 gap 예산을 써서 시작부터 abort 할 수 있다(11.07 재리뷰)."""
        self.bad, self.bad_why = 0, ""
        self.depth_win: deque = deque(maxlen=self.window)
        self.paired: set[int] = set()
        self.src, self.age, self.miss, self._miss_st = "none", None, 0, None

    def on_bad(self, why: str) -> None:
        self.bad += 1
        self.bad_why = why

    def on_msg(self, msg) -> None:
        """mono8 비트 메시지 하나. 형식 · 크기 · 비트가 틀리면(어떤 예외든) 불량으로 센다."""
        try:
            bits = mask_array(msg, self.hw)
            st = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        except Exception as exc:   # noqa: BLE001 — 깨진 메시지가 무엇을 던지든 불량 연속으로 센다
            self.on_bad(f"{type(exc).__name__}: {exc}")
            return
        self.bad = 0
        self.last_st = st if self.last_st is None else max(self.last_st, st)
        self.buf.append((st, bits))
        self.paired.update(d for d in self.depth_win if abs(st - d) <= self.slop_ns)

    def alive(self, now_ns: int) -> str | None:
        """reset 검사 — 가장 최근 정상 마스크의 stamp 가 지금(벽시계, 깊이 나이와 같은 기준)부터 max_age_s 안인가.
        수신 처리 시각이 아니라 stamp 로 본다(멈춘 노드의 큐에 남은 옛 메시지를 살아 있음으로 세지 않게). 목표 검출 여부는 안 본다."""
        if self.bad >= self.bad_max:
            return f"{MASK_TOPIC} 불량 {self.bad} 연속({self.bad_why})"
        if self.last_st is None:
            return f"{MASK_TOPIC} 를 아직 못 받았다 — head_mask_node 가 도는가"
        age = (now_ns - self.last_st) * 1e-9
        if age > self.max_age_s:
            return f"{MASK_TOPIC} 가장 최근 마스크가 {age:.2f} s 전 stamp(> {self.max_age_s} s) — 끊겼나"
        return None

    def bits_for(self, depth_st: int, target: int) -> np.ndarray:
        """깊이 stamp 에 쓸 비트(짝 → 상한 안 가장 최근). 단절 · 불량 연속이면 MaskStale."""
        if depth_st not in self.depth_win:
            self.depth_win.append(depth_st)
            self.paired &= set(self.depth_win)
        if self.bad >= self.bad_max:
            raise MaskStale(f"{MASK_TOPIC} 불량 {self.bad} 연속({self.bad_why})")
        bits, self.src, self.age = pick_mask(self.buf, depth_st, self.slop_ns * 1e-9)
        if self.src == "paired":
            self.paired.add(depth_st)
        if bits is None:
            raise MaskStale(f"{MASK_TOPIC} 를 아직 못 받았다 — head_mask_node 가 도는가")
        if self.age > self.max_age_s:
            raise MaskStale(f"{MASK_TOPIC} 마스크 나이 {self.age * 1e3:.0f} ms > {self.max_age_s * 1e3:.0f} ms(단절?)")
        if depth_st != self._miss_st:              # 목표 놓침은 깊이 프레임마다 한 번 센다(경고 · status 만)
            self._miss_st = depth_st
            hit = int(((bits >> target) & 1).sum()) >= VIS_MIN_PX
            self.miss = 0 if hit else self.miss + 1
        return bits

    def unpaired_ratio(self) -> float | None:
        """최근 깊이 프레임 창에서 짝 마스크가 끝내 안 온 비율(창이 덜 찼으면 None). 늦게 온 짝도 짝으로 센다."""
        n = len(self.depth_win)
        return None if n < self.window else 1.0 - len(self.paired) / n

    def status(self) -> dict:
        r = self.unpaired_ratio()
        return {"src": self.src, "age_ms": None if self.age is None else round(self.age * 1e3, 1),
                "unpaired_ratio": None if r is None else round(r, 3), "bad_streak": self.bad, "target_miss": self.miss}

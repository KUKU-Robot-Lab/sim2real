#!/usr/bin/env python3
"""11.07 머리 color(1280x720) → YOLO11s-seg 클래스 마스크 → 160x90 비트 영상(목표 마스크 학생 A 입력, 검출 쪽).

    source /opt/ros/humble/setup.bash && source config/dds/lan_profile.sh     # 64 MB SHM 프로파일(없으면 시작 거부)
    nice -n 5 .venv-yolo/bin/python scripts/nodes/head_mask_node.py --policy-dir deploy/policies/rh56f1/aglt/right_student1105mA

  in   /camera/camera/color/image_raw (rgb8 · bgr8, 1280x720 — 정렬 깊이와 같은 화소 격자)
  out  /head_mask/image (mono8, 160x90, header = color header 그대로) · /head_mask/config (latched JSON)

- 비트 규칙 · 블록 축소는 policy_control.student_mask.encode_bits(hdgp 78e75059 TargetMask.full 의 8x8 절반 이상 줄).
  YOLO 마스크는 retina_masks(원해상도 1280x720)로 받아 클래스별로 합친다. 학습 이미지는 jpg 를 cv2 로 읽은 BGR 이라 BGR 로 넣는다.
- 인스턴스 고르기(11.07 검증, 기본 켬, student.yaml mask.yolo 와 대조): 클래스마다 conf 최고 하나(top1_per_class),
  다른 클래스 인스턴스와 마스크 IoU > cross_class_iou 면 conf 낮은 쪽을 버린다(파랑 · 분홍은 형상이 같다).
- 검출이 0 개여도 color 프레임마다 0 비트를 낸다(학생은 메시지가 끊기면 abort, 빈 검출은 마지막 마스크 유지 + 보임 0).
- 수신 콜백은 최신 프레임만 받아 두고 추론 스레드가 가져간다 — 느린 추론이 카메라 RELIABLE writer 를 붙잡지 않게.
  추론보다 빨리 온 프레임은 버리고(dropped) 5 s 로그에 센다.
- 목표 클래스는 고르지 않는다(학생 노드 target_class). 깊이 게이팅 · 가림 유지는 학생 쪽 MaskPost.
- 5 s 마다 p50/p90 로그: 처리(color msg → 비트) · 발행 · stamp → 발행 끝(카메라 지연 포함) · 버린 프레임 · CUDA 메모리.
- ultralytics 는 .venv-yolo 에만 있다(scripts/setup/setup_venv_yolo.sh). .venv-student 에 넣지 않는다.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")     # torch import 전에 — 코어는 정책 · 제어 노드 몫(head_depth_node 와 같은 취지)

import argparse  # noqa: E402
import array  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "policy_control"))
from policy_control import student_mask as SM  # noqa: E402
from policy_control.hdgp_pin import md5 as file_md5  # noqa: E402

SRC_COLOR = "/camera/camera/color/image_raw"
WH, BLOCK = (1280, 720), 8                       # ckpt depth_pre render_wh · block(학생 노드가 /head_mask/config 로 대조)
DEFAULT_WEIGHTS = ROOT / "logs/policy/student_mask_A/yolo/seg_s640_A/best.pt"
LOG_S = 5.0
WORKER_FAIL_MAX = 30                             # 연속으로 못 낸 프레임(30 Hz 약 1 s) — 넘으면 노드를 끝낸다(11.07 재리뷰)
STOP_JOIN_S = 10.0                               # stop() 이 추론 한 번(수십 ms)을 기다리는 상한 — CUDA 가 멈췄을 때만 닿는다
SHM_MIN = 64 << 20                               # 11.05 FP++ 끊김: 기본 SHM 512 KiB 면 1280x720 이 UDP 조각으로 간다(1d4cff3)


def color_to_bgr(height: int, width: int, encoding: str, step: int, data) -> np.ndarray:
    """color Image 필드 → (720, 1280, 3) uint8 BGR. 형식 · 크기 · 바이트 수가 다르면 ValueError."""
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"color encoding {encoding} 미지원(rgb8 · bgr8)")
    if (int(width), int(height)) != WH or int(step) != int(width) * 3 or len(data) != int(step) * int(height):
        raise ValueError(f"color {width}x{height} step {step} 바이트 {len(data)} ≠ {WH[0]}x{WH[1]}x3")
    x = np.frombuffer(data, np.uint8).reshape(int(height), int(width), 3)
    return np.ascontiguousarray(x[..., ::-1]) if encoding == "rgb8" else x


def keep_instances(cls: torch.Tensor, conf: torch.Tensor, masks: torch.Tensor, top1: bool, cross_iou: float) -> list[int]:
    """conf 내림차순으로 보며 (a) top1: 클래스마다 하나 (b) cross_iou > 0: 이미 고른 다른 클래스와 마스크 IoU > cross_iou 면 버린다."""
    kept: list[int] = []
    for i in conf.argsort(descending=True).tolist():
        c = int(cls[i])
        if top1 and any(int(cls[j]) == c for j in kept):
            continue
        if cross_iou > 0 and any(int(cls[j]) != c and _iou(masks[i], masks[j]) > cross_iou for j in kept):
            continue
        kept.append(i)
    return kept


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


class MaskDetector:
    """YOLO11s-seg → 비트 영상. rclpy 없이 쓴다(오프라인 시험 · 노드 공용)."""

    def __init__(self, weights: str | Path, imgsz: int = 640, conf: float = 0.25, device: str = "cuda:0",
                 top1: bool = SM.DET_RULES["top1_per_class"], cross_iou: float = SM.DET_RULES["cross_class_iou"],
                 half: bool = False) -> None:
        from ultralytics import YOLO

        self.weights, self.imgsz, self.conf, self.device = Path(weights), int(imgsz), float(conf), device
        self.top1, self.cross_iou, self.half = bool(top1), float(cross_iou), bool(half) and device != "cpu"
        self.model = YOLO(str(self.weights))
        names = [self.model.names[i] for i in sorted(self.model.names)]
        if names != list(SM.CLASSES):
            raise ValueError(f"YOLO 클래스 {names} ≠ {list(SM.CLASSES)}")
        self.md5 = file_md5(self.weights)
        for _ in range(3):                                              # 워밍업(첫 프레임 지연 · cudnn 고르기)
            self.classes_full(np.zeros((WH[1], WH[0], 3), np.uint8))

    def classes_full(self, bgr: np.ndarray) -> torch.Tensor:
        """(720, 1280, 3) BGR → (3, 720, 1280) bool 클래스 마스크(고른 인스턴스를 클래스별로 합친다)."""
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, retina_masks=True, verbose=False,
                               device=self.device, **({"half": True} if self.half else {}))[0]  # ultralytics 8.4 는 half 인자만 넘겨도 프레임마다 경고(30 s 546줄)
        dev = torch.device(self.device)
        out = torch.zeros(len(SM.CLASSES), bgr.shape[0], bgr.shape[1], dtype=torch.bool, device=dev)
        if r.masks is None or not len(r.boxes):
            return out
        m, cls, cf = r.masks.data.to(dev) > 0.5, r.boxes.cls.to(dev).long(), r.boxes.conf.to(dev)
        if tuple(m.shape[1:]) != tuple(bgr.shape[:2]):
            raise ValueError(f"YOLO 마스크 {tuple(m.shape)} ≠ 원해상도 {bgr.shape[:2]}(retina_masks)")
        for i in keep_instances(cls, cf, m, self.top1, self.cross_iou):
            out[int(cls[i])] |= m[i]
        return out

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """(720, 1280, 3) BGR → (90, 160) uint8 비트."""
        return SM.encode_bits(self.classes_full(bgr), BLOCK).cpu().numpy()

    def config(self) -> dict:
        return {"weights": str(self.weights), "weights_md5": self.md5, "classes": list(SM.CLASSES), "imgsz": self.imgsz,
                "conf": self.conf, "retina_masks": True, "top1_per_class": self.top1, "cross_class_iou": self.cross_iou,
                "half": self.half, "block": BLOCK, "render_wh": list(WH), "out_wh": [WH[0] // BLOCK, WH[1] // BLOCK],
                "bits": "c = 클래스 c, 3+c = (다른 클래스 & ~c), 8x8 절반 이상", "src": SRC_COLOR}

    def cuda_mem_mb(self) -> str:
        if not self.device.startswith("cuda"):
            return "-"
        return (f"할당 {torch.cuda.memory_allocated(self.device) / 2**20:.0f} · "
                f"예약 최대 {torch.cuda.max_memory_reserved(self.device) / 2**20:.0f} MiB")


def to_msg(Image, header, bits: np.ndarray):
    out = Image()
    out.header = header
    out.height, out.width = bits.shape
    out.encoding = "mono8"
    out.step = out.width
    out.data = array.array("B", bits.tobytes())        # bytes 대입은 rclpy 가 원소별 검사해 느리다(head_depth_node 와 같은 방식)
    return out


def resolve_yolo(policy_dir: str | None, weights: str | None) -> tuple[Path, dict]:
    """--policy-dir(student.yaml mask.yolo: 경로 · md5 · conf · imgsz · 인스턴스 규칙) | --weights | 기본 seg_s640_A
    → (경로, 묶음 값 dict — --weights · 기본이면 빈 dict)."""
    if policy_dir:
        import yaml

        d = Path(policy_dir).expanduser().resolve()
        y = (yaml.safe_load((d / "student.yaml").read_text(encoding="utf-8")) or {}).get("mask", {}).get("yolo")
        if not y:
            raise SystemExit(f"[head_mask] {d}/student.yaml 에 mask.yolo 가 없다(마스크 학생 묶음이 아니다)")
        return d / y["path"], dict(y)
    return Path(weights) if weights else DEFAULT_WEIGHTS, {}


def dds_profile_problem(env=os.environ) -> tuple[str | None, str | None]:
    """→ (시작 거부 사유 | None, 경고 | None). 실기는 config/dds/lan_profile.sh 의 64 MB SHM 프로파일이어야 한다.
    ROS_LOCALHOST_ONLY=1 시험 경로는 lan_profile.sh 규칙대로 프로파일이 없어야 하고(있으면 격리가 깨진다) 기본 SHM 이라 경고만."""
    rmw = env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    if rmw != "rmw_fastrtps_cpp":
        return None, f"RMW_IMPLEMENTATION={rmw} — Fast-DDS 프로파일 검사를 건너뛴다"
    prof = env.get("FASTRTPS_DEFAULT_PROFILES_FILE")
    if env.get("ROS_LOCALHOST_ONLY") == "1":
        if prof:
            return "ROS_LOCALHOST_ONLY=1 인데 FASTRTPS_DEFAULT_PROFILES_FILE 이 있다 — localhost 격리가 깨진다(lan_profile.sh 규칙)", None
        return None, "★ROS_LOCALHOST_ONLY=1 — 기본 SHM 512 KiB(시험 전용). 실기 카메라 1280x720 에는 쓰지 않는다"
    if not prof:
        return "FASTRTPS_DEFAULT_PROFILES_FILE 이 없다 — config/dds/lan_profile.sh 를 source 한다(기본 SHM 512 KiB 면 1280x720 이 조각난다, 11.05)", None
    try:
        root = ET.parse(prof).getroot()
    except (OSError, ET.ParseError) as exc:
        return f"DDS 프로파일 {prof} 를 못 읽는다: {exc}", None
    sizes = [int(t.findtext("{*}segment_size") or 0) for t in root.findall(".//{*}transport_descriptor")
             if (t.findtext("{*}type") or "").strip() == "SHM"]
    if not sizes or max(sizes) < SHM_MIN:
        return f"DDS 프로파일 {prof} SHM segment_size {sizes or '없음'} < {SHM_MIN} — config/dds/fastdds_lan.xml.in(1d4cff3)", None
    return None, None


class HeadMask:
    """rclpy 노드 하나를 감싼다(rclpy import 는 생성 때). 수신 콜백 = 최신 프레임 보관, 추론 = 별도 스레드."""

    def __init__(self, node, det: MaskDetector) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self.node, self.det, self.log, self.Image = node, det, node.get_logger(), Image
        cfg = det.config()
        self.log.info(f"설정 {json.dumps(cfg, ensure_ascii=False)} · CUDA {det.cuda_mem_mb()}")
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_cfg = node.create_publisher(String, SM.MASK_CONFIG_TOPIC, latched)
        self.pub_cfg.publish(String(data=json.dumps(cfg, ensure_ascii=False)))
        self.pub = node.create_publisher(Image, SM.MASK_TOPIC, qos_profile_sensor_data)
        self._lock, self._ready, self._stop = threading.Lock(), threading.Event(), threading.Event()
        self._latest = None
        self.proc_ms: list[float] = []
        self.pub_ms: list[float] = []
        self.lat_ms: list[float] = []
        self.dropped = 0
        self.fails, self.dead = 0, ""             # 워커 연속 실패 · 죽은 이유(main 이 종료 코드로 낸다)
        # head_depth_node 와 같이 RELIABLE · KEEP_LAST 1(BEST_EFFORT 1280x720 은 30 Hz 중 약 10 Hz 만 왔다, 11.05)
        newest = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(Image, SRC_COLOR, self._on_color, newest)
        node.create_timer(LOG_S, self._log)
        self._worker = threading.Thread(target=self._run, name="head_mask_infer", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        """워커가 추론 · 발행을 끝내고 나간 뒤 돌아온다 — 그 뒤에 destroy_node 해야 죽은 publisher 에 쓰지 않는다."""
        self._stop.set()
        self._ready.set()
        self._worker.join(timeout=STOP_JOIN_S)
        if self._worker.is_alive():
            self.log.error(f"추론 스레드가 {STOP_JOIN_S:g} s 안에 끝나지 않았다(CUDA 멈춤?)")

    def _on_color(self, msg) -> None:
        with self._lock:
            if self._latest is not None:
                self.dropped += 1
            self._latest = msg
        self._ready.set()

    def _run(self) -> None:
        """워커가 어떤 이유로든 나가거나 WORKER_FAIL_MAX 연속 실패면 rclpy 를 내려 노드를 끝낸다(조용히 죽지 않게).
        학생 노드는 마스크 나이 0.2 s 로 이미 멈추지만, 운영자는 이 노드 종료 코드 · 로그로 원인을 본다."""
        try:
            while not self._stop.is_set():
                self._ready.wait(0.5)
                self._ready.clear()
                with self._lock:
                    msg, self._latest = self._latest, None
                if msg is None or self._stop.is_set():
                    continue
                try:
                    ok = self._process(msg)
                except Exception as exc:   # noqa: BLE001 — 추론 · 변환 · 발행 어디서 나든 센다
                    self.log.error(f"프레임 처리 실패: {type(exc).__name__}: {exc}", throttle_duration_sec=LOG_S)
                    ok = False
                self.fails = 0 if ok else self.fails + 1
                if self.fails >= WORKER_FAIL_MAX:
                    self.dead = f"{SM.MASK_TOPIC} 를 {self.fails} 프레임 연속 못 냈다"
                    return
        except BaseException as exc:   # noqa: BLE001 — 루프 자체가 깨져도 이유를 남긴다
            self.dead = f"추론 스레드 오류 {type(exc).__name__}: {exc}"
        finally:
            if not self._stop.is_set():
                self.dead = self.dead or "추론 스레드가 나갔다"
                self.log.fatal(f"{self.dead} — 노드를 끝낸다")
                import rclpy
                rclpy.try_shutdown()

    def _process(self, msg) -> bool:
        """한 프레임 → 발행. 형식 불량(ValueError)은 False, 그 밖의 예외(CUDA · ultralytics · 발행)는 _run 이 센다."""
        t0 = time.perf_counter()
        try:
            bits = self.det(color_to_bgr(msg.height, msg.width, msg.encoding, msg.step, msg.data))
        except ValueError as exc:
            self.log.error(f"이 프레임을 버린다: {exc}", throttle_duration_sec=LOG_S)
            return False
        t1 = time.perf_counter()
        self.pub.publish(to_msg(self.Image, msg.header, bits))
        st = msg.header.stamp
        lat = (self.node.get_clock().now().nanoseconds - (st.sec * 10**9 + st.nanosec)) * 1e-6
        with self._lock:
            self.proc_ms.append((t1 - t0) * 1e3)
            self.pub_ms.append((time.perf_counter() - t1) * 1e3)
            self.lat_ms.append(lat)
        return True

    def _log(self) -> None:
        with self._lock:
            p, q, a, drop = np.array(self.proc_ms), np.array(self.pub_ms), np.array(self.lat_ms), self.dropped
            self.proc_ms, self.pub_ms, self.lat_ms, self.dropped = [], [], [], 0
        if not len(p):
            self.log.warning(f"{LOG_S:g} s 동안 {SRC_COLOR} 를 처리하지 못했다")
            return
        self.log.info(f"{len(p)} 프레임(버림 {drop}) 처리 p50 {np.median(p):.1f} · p90 {np.percentile(p, 90):.1f} ms | "
                      f"발행 p50 {np.median(q):.2f} · p90 {np.percentile(q, 90):.2f} ms | "
                      f"stamp → 발행 p50 {np.median(a):.1f} · p90 {np.percentile(a, 90):.1f} · max {a.max():.1f} ms | "
                      f"CUDA {self.det.cuda_mem_mb()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="머리 color → YOLO11s-seg → 160x90 클래스 비트 마스크")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--policy-dir", help="마스크 학생 묶음(student.yaml mask.yolo 경로 · md5 · conf · imgsz · 인스턴스 규칙)")
    src.add_argument("--weights", help=f"YOLO 가중치(기본 {DEFAULT_WEIGHTS.relative_to(ROOT)})")
    ap.add_argument("--conf", type=float, help="기본 = 묶음 값 · 없으면 0.25")
    ap.add_argument("--imgsz", type=int, help="기본 = 묶음 값 · 없으면 640")
    ap.add_argument("--top1", action=argparse.BooleanOptionalAction, default=None, help="클래스마다 conf 최고 하나(기본 켬)")
    ap.add_argument("--cross-iou", type=float, help="다른 클래스와 마스크 IoU 상한(0 = 끔, 기본 0.5)")
    ap.add_argument("--half", action="store_true", help="fp16 추론(fp32 대비 IoU 확인 전까지 기본 끔)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-default-dds", action="store_true", help="64 MB SHM 프로파일 없이 띄운다(부하 비교 시험 전용)")
    args = ap.parse_args()
    bad, warn = dds_profile_problem()
    if bad and not args.allow_default_dds:
        raise SystemExit(f"[head_mask] {bad}")
    weights, card = resolve_yolo(args.policy_dir, args.weights)

    def pick(arg, key, default):
        return arg if arg is not None else card.get(key, default)

    det = MaskDetector(weights, pick(args.imgsz, "imgsz", 640), pick(args.conf, "conf", 0.25), args.device,
                       pick(args.top1, "top1_per_class", SM.DET_RULES["top1_per_class"]),
                       pick(args.cross_iou, "cross_class_iou", SM.DET_RULES["cross_class_iou"]), args.half)
    if card and det.md5 != str(card["md5"]):
        raise SystemExit(f"[head_mask] {weights} md5 {det.md5} ≠ student.yaml {card['md5']}")

    import cv2
    import rclpy

    cv2.setNumThreads(2)                      # ultralytics letterbox · 마스크 크기 바꾸기 — 코어는 정책 · 제어 노드 몫
    torch.set_num_threads(2)
    rclpy.init()
    node = rclpy.create_node("head_mask")
    for m in (bad, warn):
        if m:
            node.get_logger().warning(f"DDS: {m}")
    try:
        from policy_control.cpu_plan import keep_off_rt
        node.get_logger().info(keep_off_rt())     # 실시간 코어(EtherCAT 마스터 자리)를 비켜 간다
    except Exception as exc:   # noqa: BLE001 — 고정 실패는 로그만(다른 노드와 같은 취지)
        node.get_logger().warning(f"CPU 고정 안 함: {exc}")
    hm = HeadMask(node, det)
    from rclpy.executors import ExternalShutdownException
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # 후자 = 워커가 rclpy 를 내렸다(hm.dead)
        pass
    finally:
        hm.stop()
        node.destroy_node()
        rclpy.try_shutdown()
    if hm.dead:
        raise SystemExit(f"[head_mask] {hm.dead}")


if __name__ == "__main__":
    main()

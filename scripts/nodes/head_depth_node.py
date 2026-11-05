#!/usr/bin/env python3
"""머리 D435i 정렬 깊이(1280x720, mm)를 rh_cap 깊이 학생 입력(160x90, m, float32)으로 줄여 내보낸다.

  .venv-student/bin/python scripts/nodes/head_depth_node.py --policy-dir <학생 폴더> [--block 8] [--min-valid 0.5] [--table-z 0.205] ...
  .venv-student/bin/python scripts/nodes/head_depth_node.py --extrinsics <yaml> [--rgb] ...

입력  /camera/camera/aligned_depth_to_color/image_raw (16UC1) · /camera/camera/color/camera_info
      (--rgb) /camera/camera/color/image_raw (rgb8 · bgr8, 1280x720 — 정렬 깊이와 같은 화소 격자)
출력  /head_depth/image (32FC1, m, 구멍은 0) · /head_depth/camera_info (K 를 블록 크기로 줄임)
      (--rgb) /head_depth/rgb (32FC3, 0~1, 160x90, 깊이와 같은 header)

- 외부 파라미터는 --policy-dir(student.yaml 의 얼린 사본 · md5) 또는 --extrinsics 로 꼭 준다. 기본값은 없다
  (config/global_camera_extrinsics_* 는 deploy 세션이 다시 재는 파일이라 학생이 따라가면 안 된다).
  md5 가 student.yaml(--policy-dir) 또는 학습 값 EXTRINSICS_MD5(--extrinsics)와 다르면 시작하지 않는다.
- --rgb(student.yaml rgb: true 면 자동): 깊이 stamp 와 가장 가까운 color 프레임(RGB_SLOP_S 안)을 /255 → 8x8 블록 평균으로
  줄여 깊이와 같은 header 로 낸다(hdgp real_pipe._rgb 와 같은 순서, 바닥 컷 없음). 짝 color 가 없으면 그 깊이도 내지 않는다
  (정책이 짝 안 맞는 깊이 · rgb 를 받지 않게).
- ★11.05 외부 파라미터는 목 한 자세(그 yaml head_pose, r20u5 = pan 1825 · tilt 2815 ± 4 틱)에서만 맞는 정적 스냅샷이고 학생은
  그 시야로 배웠다. /head/joint_states 가 그 자세 밖이거나 HEAD_MAX_AGE_S 넘게 낡았으면 깊이를 내지 않는다(정책 노드는
  StaleDepth → abort). object_pose_node 의 check_head_at 을 그대로 쓴다. 미션 head_home 단계가 그 자세로 맞춘다.

- 줄이기 전에 sim 학습(real_pipe)과 같은 바닥 컷을 한다: 고정 T_base_cam(r20u5 yaml camera 블록, 머리 FK 안 씀)과
  1280x720 K(real_depth.D435I_K_1103, 학습 기본과 같음)로 base z < table_z − margin 인 화소를 0 으로.
  광선 rz 는 시작 때 real_depth.ray_z 로 한 번 만들고 프레임마다 real_depth.floor_cut 만 부른다(real_pipe 와 같은 방식).
- 처음 받은 camera_info 를 D435I_K_1103 과 대조한다: 1280x720 이 아니면 깊이를 발행하지 않고(error),
  K 가 0.5 px 넘게 다르면 warning 한 번(컷은 학습과 같은 상수 K 그대로). camera_info 를 받기 전에는 발행하지 않는다.
- 구멍 채우기 · 시간 필터를 하지 않는다. realsense 후처리 필터도 camera_up.sh 기본값(꺼짐) 그대로 둔다.
- 줄이기는 블록 중앙값이다. 8x8 블록에서 0 을 뺀 값의 중앙값을 쓰고,
  유효 픽셀 비율이 --min-valid 보다 적으면 0 으로 둔다.
- d/1.5 정규화 · 0.2~1.5 m 자르기 · 3 채널 복제는 정책 쪽 전처리라 여기서 하지 않는다.
- 헤더 stamp 는 원본 깊이 stamp 를 그대로 둔다. 지연은 /head_depth 수신 시각과 stamp 차이로 잰다.
- 줄이기 함수(블록 중앙값 · K 축소)는 sim 학습과 같은 hdgp real_depth.py 를 고정 커밋 사본에서 읽는다
  (policy_control.student_policy.load_pinned, config/hdgp_student_snapshot.yaml · HDGP_DIR 기본 ~/rl_ws/hdgp).
- 시작 때 설정(블록 · 바닥 컷 · K · 외부 파라미터 md5 · real_depth 출처)을 JSON 으로 /head_depth/config 에 한 번 낸다
  (transient_local). 정책은 ckpt depth_pre 와 student_policy.check_depth_config 로 대조한다.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

SRC_DEPTH = "/camera/camera/aligned_depth_to_color/image_raw"
SRC_INFO = "/camera/camera/color/camera_info"
SRC_COLOR = "/camera/camera/color/image_raw"
OUT_DEPTH = "/head_depth/image"
OUT_RGB = "/head_depth/rgb"
OUT_INFO = "/head_depth/camera_info"
OUT_CONFIG = "/head_depth/config"
EXTRINSICS_MD5 = "624677fee7e8ff6502f33a3ed8a96a1f"   # 40e6496(11.04 다시 잼) — --extrinsics 로 줄 때 학습 값과 같아야 한다
TABLE_Z = 0.205                                      # 탁자 상판 base z[m](CAD, rh_aglt env_cfg table_surface_z)
K_TOL_PX = 0.5
LOG_S = 5.0          # 오류 · 경고 throttle, 지연 통계 주기 [s]
INFO_WAIT_S = 3.0    # 시작 뒤 이 시간 안에 camera_info 가 없으면 warn
RGB_SLOP_S = 0.5 / 30.0   # 깊이 · color stamp 짝 허용(30 Hz 반 프레임)
COLOR_BUF = 4
PEND_MAX_S = max(RGB_SLOP_S, 1 / 30.0)   # 짝 color 를 기다리는 최대 시간(한 프레임)
HEAD_TOPIC = "/head/joint_states"
HEAD_MAX_AGE_S = 1.0      # object_pose_node --head-max-age 기본과 같다


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "policy_control"))
from policy_control.hdgp_pin import PinError  # noqa: E402
from policy_control.student_policy import REAL_DEPTH, load_pinned, load_student_dir  # noqa: E402

# hdgp real_depth.py 고정 커밋 사본(config/hdgp_student_snapshot.yaml 커밋 · md5 핀). 공유 작업 트리는 읽지 않는다
real_depth, REAL_DEPTH_SRC = load_pinned(REAL_DEPTH, "hdgp_real_depth")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_pose_check import load_head_pose  # noqa: E402
from object_pose_node import HeadOffCalibPose, check_head_at  # noqa: E402


def head_block(head_pose, head_deg: tuple[float, float] | None, head_t: float | None, now: float,
               max_age: float = HEAD_MAX_AGE_S) -> str | None:
    """깊이를 내면 안 되는 이유(목 각이 없음 · 낡음 · 캘리브 자세 밖) 또는 None. 순수."""
    if head_deg is None or head_t is None:
        return f"{HEAD_TOPIC} 를 아직 못 받았다"
    if now - head_t > max_age:
        return f"{HEAD_TOPIC} 가 {now - head_t:.1f} s 낡았다(허용 {max_age:g} s)"
    try:
        check_head_at(head_pose, head_deg)
    except HeadOffCalibPose as e:
        return str(e)
    return None


def load_extrinsics(path) -> tuple[tuple, str]:
    """r20u5 yaml → ((position, orientation_wxyz) = T_base_cam, 파일 md5)."""
    import yaml

    raw = Path(path).read_bytes()
    cam = yaml.safe_load(raw)["camera"]
    return ([float(v) for v in cam["position"]], [float(v) for v in cam["orientation_wxyz"]]), hashlib.md5(raw).hexdigest()


def make_cut(t_base_cam, table_z: float = TABLE_Z, margin: float = real_depth.FLOOR_MARGIN_M) -> tuple:
    """(pos, quat_wxyz) → floor_cut 인자 (cam_z, rz (720, 1280) float32, z_min). K 는 D435I_K_1103(1280x720)."""
    pos, quat = t_base_cam
    w, h = real_depth.D435I_WH
    return float(pos[2]), real_depth.ray_z(real_depth.D435I_K_1103, w, h, quat), float(table_z) - float(margin)


def check_info(width: int, height: int, k) -> tuple[bool, str]:
    """camera_info 대조 → (깊이를 발행해도 되나, 남길 말. 빈 문자열 = D435I_K_1103 과 같음)."""
    if (int(width), int(height)) != real_depth.D435I_WH:
        return False, f"camera_info {width}x{height} ≠ 1280x720 — 바닥 컷 K 와 맞지 않아 깊이를 발행하지 않는다"
    dk = float(np.abs(np.asarray(k, dtype=np.float64).reshape(3, 3) - np.array(real_depth.D435I_K_1103)).max())
    if dk > K_TOL_PX:
        return True, f"camera_info K 가 D435I_K_1103 과 최대 {dk:.3f} px 다르다 — 바닥 컷은 학습과 같은 상수 K 를 그대로 쓴다"
    return True, ""


def to_small(d_mm: np.ndarray, block: int, min_valid: float, cut: tuple | None = None) -> np.ndarray:
    """(H, W) uint16 mm 깊이 → (바닥 컷, cut = make_cut(...) 이 있을 때) → (H/block, W/block) float32 m 블록 중앙값(0 = 구멍)."""
    m = d_mm.astype(np.float32) / np.float32(1000.0)   # real_depth.mm_quantize 와 같은 나눗셈(* 1e-3 은 1 ulp 다르다)
    if cut is not None:
        m = real_depth.floor_cut(m, *cut)
    # 같은 hdgp block_median 의 torch 경로(값 동일, 시험이 대조). numpy 경로는 1280x720 에 16 ms 라 30 Hz 를 못 맞춘다(11.05)
    return real_depth.block_median(torch.from_numpy(m), block, min_valid).numpy()


class FrameError(ValueError):
    pass


def frame_to_small(height: int, width: int, encoding: str, step: int, data, block: int, min_valid: float,
                   cut: tuple | None = None) -> np.ndarray:
    """Image 필드 → 검사(1280x720 · 16UC1 · 바이트 수) → to_small. 어긋나면 FrameError."""
    if encoding != "16UC1":
        raise FrameError(f"depth encoding {encoding} 미지원(16UC1 만)")
    if (int(width), int(height)) != real_depth.D435I_WH:
        raise FrameError(f"depth {width}x{height} ≠ {real_depth.D435I_WH[0]}x{real_depth.D435I_WH[1]}")
    if int(step) != int(width) * 2 or len(data) != int(step) * int(height):
        raise FrameError(f"depth 바이트 {len(data)} · step {step} ≠ {width}x{height} uint16")
    d = np.frombuffer(data, np.uint16).reshape(int(height), int(width))     # rclpy uint8[] = array.array(버퍼)
    return to_small(d, block, min_valid, cut)


def rgb_to_small(height: int, width: int, encoding: str, step: int, data, block: int) -> np.ndarray:
    """color Image 필드 → 검사(1280x720 · rgb8|bgr8 · 바이트 수) → (H/block, W/block, 3) float32 0~1 블록 평균(rgb 순서)."""
    if encoding not in ("rgb8", "bgr8"):
        raise FrameError(f"color encoding {encoding} 미지원(rgb8 · bgr8)")
    if (int(width), int(height)) != real_depth.D435I_WH:
        raise FrameError(f"color {width}x{height} ≠ {real_depth.D435I_WH[0]}x{real_depth.D435I_WH[1]}")
    if int(step) != int(width) * 3 or len(data) != int(step) * int(height):
        raise FrameError(f"color 바이트 {len(data)} · step {step} ≠ {width}x{height}x3")
    h, w, b = int(height), int(width), int(block)
    if h % b or w % b:
        raise FrameError(f"color {w}x{h} 가 블록 {b} 의 정수배가 아니다(INTER_AREA ≠ 블록 평균)")
    x = np.frombuffer(data, np.uint8).reshape(h, w, 3).astype(np.float32)
    # INTER_AREA 는 정수배 축소에서 블록 평균과 같다(float 반올림 차 < 1e-6). numpy reshape 평균은 12 ms(11.05)
    x = cv2.resize(x, (w // b, h // b), interpolation=cv2.INTER_AREA) / np.float32(255.0)
    if encoding == "bgr8":
        x = x[..., ::-1]
    return np.ascontiguousarray(x)


def _ns(msg) -> int:
    st = msg.header.stamp
    return st.sec * 10**9 + st.nanosec


def nearest_color(buf, stamp_ns: int, slop_s: float = RGB_SLOP_S):
    """[(stamp_ns, msg)] 에서 stamp 가 가장 가까운 msg(차이 > slop 이면 None)."""
    if not buf:
        return None
    t, m = min(buf, key=lambda x: abs(x[0] - stamp_ns))
    return m if abs(t - stamp_ns) <= slop_s * 1e9 else None


def resolve_extrinsics(policy_dir: str | None, extrinsics: str | None, rgb: bool) -> tuple[str, str, bool]:
    """--policy-dir | --extrinsics → (yaml 경로, 기대 md5, rgb). 파일 md5 ≠ 기대면 PinError(시작 거부)."""
    if policy_dir:
        cfg = load_student_dir(policy_dir)          # 얼린 사본 md5 = student.yaml 을 여기서 검사한다
        if rgb and not cfg["rgb"]:
            raise PinError(f"--rgb 인데 {policy_dir} student.yaml rgb: false")
        return str(cfg["extrinsics"]["path"]), cfg["extrinsics"]["md5"], cfg["rgb"]
    got = hashlib.md5(Path(extrinsics).read_bytes()).hexdigest()
    if got != EXTRINSICS_MD5:
        raise PinError(f"외부 파라미터 {extrinsics} md5 {got} ≠ 학습 {EXTRINSICS_MD5}(r20u5 40e6496) — 시작하지 않는다")
    return str(extrinsics), EXTRINSICS_MD5, bool(rgb)


def node_config(block: int, min_valid: float, table_z: float, floor_margin: float, extrinsics: str, extr_md5: str,
                rgb: bool = False) -> dict:
    """/head_depth/config 내용 — 키 이름은 hdgp real_pipe.meta(ckpt depth_pre)와 같게(student_policy.check_depth_config)."""
    w, h = real_depth.D435I_WH
    k = real_depth.scale_k(np.array(real_depth.D435I_K_1103, dtype=np.float64), block)
    return {"block": int(block), "min_valid": float(min_valid), "floor_cut_z_m": float(table_z) - float(floor_margin),
            "render_wh": [w, h], "out_wh": [w // block, h // block], "K": k.tolist(),
            "extrinsics": str(extrinsics), "extrinsics_md5": extr_md5, "rgb": bool(rgb),
            "real_depth": REAL_DEPTH_SRC["path"], "real_depth_md5": REAL_DEPTH_SRC["md5"], "hdgp_commit": REAL_DEPTH_SRC["commit"]}


class HeadDepth:
    """rclpy 노드 하나를 감싼다(rclpy import 는 생성 때 — 시험은 rclpy 없이 이 모듈을 import 한다)."""

    def __init__(self, node, args, cut: tuple, cfg: dict, head_pose) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import String

        self.node, self.args, self.cut, self.log = node, args, cut, node.get_logger()
        self.CameraInfo, self.Image = CameraInfo, Image
        self.log.info(f"설정 {json.dumps(cfg, ensure_ascii=False)}")
        self.info = None
        self.info_ok: bool | None = None       # None = camera_info 아직 못 받음(발행 안 함)
        self.lat_ms: list[float] = []
        self.pending = None                    # 짝 color 를 기다리는 깊이 msg
        self.pending_t = 0.0                   # 보류 시작(monotonic)
        self.last_ns: int | None = None        # 마지막으로 낸 깊이 stamp
        self.head_pose, self.head_deg, self.head_t = head_pose, None, None
        node.create_subscription(JointState, HEAD_TOPIC, self._head, 10)
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_cfg = node.create_publisher(String, OUT_CONFIG, latched)
        self.pub_cfg.publish(String(data=json.dumps(cfg, ensure_ascii=False)))
        self.pub = node.create_publisher(Image, OUT_DEPTH, qos_profile_sensor_data)
        self.pub_info = node.create_publisher(CameraInfo, OUT_INFO, qos_profile_sensor_data)
        self.rgb = bool(cfg["rgb"])
        self.color: deque = deque(maxlen=COLOR_BUF)
        color_qos = QoSProfile(depth=COLOR_BUF, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE)
        if self.rgb:
            self.pub_rgb = node.create_publisher(Image, OUT_RGB, qos_profile_sensor_data)
            node.create_subscription(Image, SRC_COLOR, self._color, color_qos)
        # 11.05 실측: 1280x720 원본을 BEST_EFFORT 로 받으면 30 Hz 중 약 10 Hz 만 오고 color·depth stamp 짝이 53 % 였다.
        # RELIABLE 은 26~28 Hz, 짝 95 %(logs/analysis/camera_stream_probe_1105_123002). KEEP_LAST 1 이라 최신 프레임만 남는다.
        newest = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(CameraInfo, SRC_INFO, self._info, qos_profile_sensor_data)
        node.create_subscription(Image, SRC_DEPTH, self._depth, newest)
        self.wait = node.create_timer(INFO_WAIT_S, self._info_wait)
        node.create_timer(LOG_S, self._log_latency)

    def _info_wait(self) -> None:
        self.wait.cancel()
        if self.info is None:
            self.log.warning(f"{INFO_WAIT_S:g} s 동안 {SRC_INFO} 없음 — 깊이를 발행하지 않는다")

    def _log_latency(self) -> None:
        if self.lat_ms:
            a = np.array(self.lat_ms)
            self.log.info(f"stamp → 발행 {len(a)} 프레임 p50 {np.median(a):.1f} ms · max {a.max():.1f} ms")
            self.lat_ms = []

    def _info(self, msg) -> None:
        self.info = msg
        if self.info_ok is None:
            self.info_ok, note = check_info(msg.width, msg.height, msg.k)
            if not self.info_ok:
                self.log.error(note)
            elif note:
                self.log.warning(note)

    def _head(self, msg) -> None:
        names = list(msg.name)
        try:
            pan, tilt = (float(msg.position[names.index(n)]) for n in ("head_j_pan", "head_j_tilt"))
        except (ValueError, IndexError):
            return
        self.head_deg = (float(np.degrees(pan)), float(np.degrees(tilt)))
        self.head_t = time.monotonic()

    def _color(self, msg) -> None:
        st = msg.header.stamp
        self.color.append((st.sec * 10**9 + st.nanosec, msg))
        if self.pending is not None:              # 짝 color 보다 먼저 온 깊이를 지금 처리한다
            d, t0, self.pending = self.pending, self.pending_t, None
            self._depth(d)
            if self.pending is d:                 # 아직 짝이 없다
                self.pending_t = t0
                if time.monotonic() - t0 > PEND_MAX_S:
                    self._drop_pending(f"{PEND_MAX_S * 1e3:.0f} ms 넘게 보류")

    def _drop_pending(self, why: str) -> None:
        self.pending = None
        self.log.warning(f"{SRC_COLOR} 짝 없이 깊이를 버린다: {why}", throttle_duration_sec=LOG_S)

    def _depth(self, msg) -> None:
        if self.pending is not None:
            if _ns(msg) > _ns(self.pending):
                self._drop_pending("다음 깊이가 왔다")
            else:
                self.log.warning("깊이 stamp 가 보류 중인 깊이보다 이르거나 같아 버린다", throttle_duration_sec=LOG_S)
                return
        try:
            self._publish(msg)
        except FrameError as e:
            self.log.error(str(e), throttle_duration_sec=LOG_S)
        except Exception as e:   # noqa: BLE001 — 콜백 예외로 spin 이 죽지 않게
            self.log.error(f"깊이 처리 실패 {type(e).__name__}: {e}", throttle_duration_sec=LOG_S)

    def _publish(self, msg) -> None:
        if self.info_ok is None:
            self.log.warning(f"{SRC_INFO} 를 아직 못 받아 깊이를 버린다", throttle_duration_sec=LOG_S)
            return
        if not self.info_ok:
            return
        why = head_block(self.head_pose, self.head_deg, self.head_t, time.monotonic())
        if why:
            self.log.warning(f"깊이 발행 보류: {why}", throttle_duration_sec=LOG_S)
            return
        t = _ns(msg)
        if self.last_ns is not None and t <= self.last_ns:
            self.log.warning("깊이 stamp 가 앞서 낸 것보다 이르거나 같아 버린다", throttle_duration_sec=LOG_S)
            return
        rgb = None
        if self.rgb:
            c = nearest_color(self.color, t)
            if c is None and (not self.color or self.color[-1][0] < t + RGB_SLOP_S * 1e9):
                self.pending, self.pending_t = msg, time.monotonic()   # 짝 color 가 아직 안 왔다(11.05 실측 깊이 절반이 color 보다 먼저 옴)
                return
            if c is None:
                self.log.warning(f"깊이 stamp 와 {RGB_SLOP_S * 1e3:.0f} ms 안의 {SRC_COLOR} 가 없어 이 프레임을 버린다",
                                 throttle_duration_sec=LOG_S)
                return
            rgb = rgb_to_small(c.height, c.width, c.encoding, c.step, c.data, self.args.block)
        small = frame_to_small(msg.height, msg.width, msg.encoding, msg.step, msg.data, self.args.block, self.args.min_valid, self.cut)
        if not (small > 0).any():
            self.log.warning("축소 깊이에 유효 화소가 없다", throttle_duration_sec=LOG_S)
        out = self.Image()
        out.header = msg.header
        out.height, out.width = small.shape
        out.encoding = "32FC1"
        out.step = out.width * 4
        out.data = array.array("B", small.tobytes())   # bytes 대입은 rclpy 가 원소별 검사해 rgb 6.7 ms 든다
        if rgb is not None:                       # rgb 를 먼저 낸다 — 정책은 깊이를 받는 순간 같은 stamp rgb 를 찾는다
            ro = self.Image()
            ro.header = msg.header
            ro.height, ro.width = rgb.shape[:2]
            ro.encoding = "32FC3"
            ro.step = ro.width * 12
            ro.data = array.array("B", rgb.tobytes())
            self.pub_rgb.publish(ro)
        self.pub.publish(out)
        self.last_ns = t
        info = self.CameraInfo()
        info.header = msg.header
        info.height, info.width = small.shape
        info.distortion_model = self.info.distortion_model
        info.d = list(self.info.d)
        info.k = real_depth.scale_k(np.asarray(self.info.k).reshape(3, 3), self.args.block).ravel().tolist()
        self.pub_info.publish(info)
        st = msg.header.stamp
        self.lat_ms.append((self.node.get_clock().now().nanoseconds - (st.sec * 10**9 + st.nanosec)) * 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description="D435i 정렬 깊이를 160x90 m float32 로 줄인다")
    ap.add_argument("--block", type=int, default=real_depth.BLOCK)
    ap.add_argument("--min-valid", type=float, default=real_depth.MIN_VALID)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy-dir", help="학생 묶음 폴더(student.yaml: 얼린 외부 파라미터 · md5 · rgb)")
    src.add_argument("--extrinsics", help=f"T_base_cam yaml(camera.position · orientation_wxyz), md5 = {EXTRINSICS_MD5} 만")
    ap.add_argument("--rgb", action="store_true", help="/head_depth/rgb 도 낸다(--policy-dir 면 student.yaml rgb 를 따른다)")
    ap.add_argument("--table-z", type=float, default=TABLE_Z)
    ap.add_argument("--floor-margin", type=float, default=real_depth.FLOOR_MARGIN_M)
    args = ap.parse_args()
    try:
        extr, want_md5, rgb = resolve_extrinsics(args.policy_dir, args.extrinsics, args.rgb)
    except PinError as e:
        raise SystemExit(f"[head_depth] {e}") from None
    t_base_cam, extr_md5 = load_extrinsics(extr)
    if extr_md5 != want_md5:                      # 검사 뒤 파일이 바뀐 경우
        raise SystemExit(f"[head_depth] 외부 파라미터 {extr} md5 {extr_md5} ≠ {want_md5}")
    head_pose = load_head_pose(Path(extr))
    if head_pose is None:
        raise SystemExit(f"[head_depth] {extr} 에 head_pose 가 없다 — 목 자세를 확인할 수 없어 시작하지 않는다")
    cut = make_cut(t_base_cam, args.table_z, args.floor_margin)
    cfg = node_config(args.block, args.min_valid, args.table_z, args.floor_margin, extr, extr_md5, rgb)

    import rclpy

    cv2.setNumThreads(2)                      # INTER_AREA 축소 1~2 ms 면 충분, 코어는 정책 · 제어 노드 몫
    torch.set_num_threads(4)                  # block_median p50 5.5 ms. 8 은 p50 같고 꼬리만 길다(11.05). 정책 · 제어 노드 몫을 남긴다
    rclpy.init()
    node = rclpy.create_node("head_depth")
    HeadDepth(node, args, cut, cfg, head_pose)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:                 # SIGINT 종료는 정상 경로 — traceback 을 남기지 않는다
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()                  # SIGINT 때 rclpy 가 이미 shutdown 했으면 그냥 지나간다


if __name__ == "__main__":
    main()

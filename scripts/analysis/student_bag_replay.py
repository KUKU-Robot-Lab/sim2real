#!/usr/bin/env python3
"""11.05 1104e 잡기 학생(깊이 + RGB v3)을 실기 bag 으로 오프라인 재생 — 입력 경로 · 수치 건전성만 본다.

학생 행동은 로봇에 돌려 넣지 않는다(open-loop). ROS 노드를 띄우지 않고 bag 만 읽는다.
  깊이 · RGB: compressedDepth(12 바이트 헤더 + PNG 16UC1) · JPEG 를 풀어 head_depth_node 의 frame_to_small · rgb_to_small
              (같은 바닥 컷 · 묶음 얼린 외부 파라미터)로 160x90 을 만든다. color 짝은 head_depth_node.nearest_color.
  prop: bag 관절 → sources.SourceSet(rh56f1_bi_real.yaml 이름 · 부호) → pour_fj_node._side_raw(FK) → build_student_obs.
        컵 칸은 학생이 버리므로 고정값. 디코더 목표는 매 틱 측정 관절로 reset(목표 = 측정, 손 오차 0) — 아래 shortcut.
  틱: 계약 policy_hz 로 bag 시간을 걷는다. 깊이는 그 틱까지 받은 최신 프레임, 나이 = 틱 − header stamp.

  .venv-student/bin/python scripts/analysis/student_bag_replay.py \
      --bag logs/bags/20261104_125804_episode_cocktail_1c_e4000 --out logs/analysis/student_bag_replay_1105
"""
from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy" / "policy_control"))
from policy_control import codec  # noqa: E402
from policy_control import rh_aglt as A  # noqa: E402
from policy_control.contract_assets import ASSETS  # noqa: E402
from policy_control.fk_numpy import UrdfChainFK  # noqa: E402
from policy_control.pour_fj_node import _side_raw  # noqa: E402
from policy_control.sources import SourceSet, load_robot_cfg, select_side  # noqa: E402
from policy_control.student_handover import StudentHandover  # noqa: E402
from policy_control.student_obs import build_student_obs  # noqa: E402
from policy_control.student_policy import StaleDepth, StudentPolicy, check_depth_config  # noqa: E402

AGLT = REPO / "deploy" / "policies" / "rh56f1" / "aglt"
ROBOT = REPO / "deploy" / "policy_control" / "config" / "robots" / "rh56f1_bi_real.yaml"
T_DEPTH = "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth"
T_COLOR = "/camera/camera/color/image_raw/compressed"
T_INFO = "/camera/camera/color/camera_info"
T_ARM = "/joint_states"
T_HAND = {"right": "/hand_right/joint_states", "left": "/hand_left/joint_states"}
DEPTH_FMT = "16UC1; compressedDepth"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CD_HEADER = 12          # compressed_depth_image_transport ConfigHeader = int32 format + float depthParam[2]
APPROACH_MAX_S = 6.0    # student_handover: sim ap_max_steps 6 s(on_max handover)
N_PNG = 6


def load_head_depth_node():
    spec = importlib.util.spec_from_file_location("head_depth_node", REPO / "scripts" / "nodes" / "head_depth_node.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_compressed_depth(fmt: str, data: bytes) -> np.ndarray:
    """compressedDepth 메시지 → (H, W) uint16 mm. 16UC1 은 헤더 값을 쓰지 않는다(PNG 원값) — 형식 문자열 · PNG 시작만 검사."""
    import cv2

    if fmt.strip() != DEPTH_FMT:
        raise ValueError(f"compressedDepth format {fmt!r} ≠ {DEPTH_FMT!r}(PNG 16UC1 만)")
    if bytes(data[CD_HEADER:CD_HEADER + len(PNG_MAGIC)]) != PNG_MAGIC:
        raise ValueError(f"compressedDepth: {CD_HEADER} 바이트 헤더 뒤에 PNG 가 없다")
    d = cv2.imdecode(np.frombuffer(data, np.uint8)[CD_HEADER:], cv2.IMREAD_UNCHANGED)
    if d is None or d.dtype != np.uint16 or d.ndim != 2:
        raise ValueError(f"compressedDepth PNG 가 16 비트 1 채널이 아니다: {None if d is None else (d.dtype, d.shape)}")
    return d


def decode_jpeg_bgr(fmt: str, data: bytes) -> np.ndarray:
    """color compressed(JPEG) → (H, W, 3) uint8 BGR(cv2). image_transport 는 rgb8 원본도 bgr8 로 바꿔 압축한다(format 문자열)."""
    import cv2

    if "jpeg" not in fmt or "bgr8" not in fmt:
        raise ValueError(f"color format {fmt!r}: jpeg · bgr8 압축만")
    x = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if x is None:
        raise ValueError("JPEG 디코드 실패")
    return x


def read_bag(uri: Path, types: dict) -> list:
    """한 묶음 bag → [(recv_ns, topic, msg)] (topics = types 키만, 받은 순)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    r.set_filter(rosbag2_py.StorageFilter(topics=list(types)))
    out = []
    while r.has_next():
        t, d, ns = r.read_next()
        out.append((ns, t, deserialize_message(d, types[t])))
    return out


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)


def camera_frames(cam: list, hdn, cut, block: int, min_valid: float) -> tuple[list, dict]:
    """camera 묶음 → [(recv_s, stamp_s, depth (90,160) m, rgb (90,160,3) 0~1)] · 세기."""
    colors = sorted((stamp_ns(m), m) for _, t, m in cam if t == T_COLOR)
    keys = [c[0] for c in colors]
    w, h = hdn.real_depth.D435I_WH
    frames, n = [], {"depth_msgs": 0, "no_color_pair": 0, "decode_error": 0, "errors": []}
    for recv, t, m in cam:
        if t != T_DEPTH:
            continue
        n["depth_msgs"] += 1
        s = stamp_ns(m)
        i = bisect.bisect_left(keys, s)
        c = hdn.nearest_color(colors[max(0, i - 1):i + 1], s)
        if c is None:
            n["no_color_pair"] += 1
            continue
        try:
            d = decode_compressed_depth(m.format, bytes(m.data))
            bgr = decode_jpeg_bgr(c.format, bytes(c.data))
            small = hdn.frame_to_small(d.shape[0], d.shape[1], "16UC1", d.shape[1] * 2, d.tobytes(), block, min_valid, cut)
            rgb = hdn.rgb_to_small(bgr.shape[0], bgr.shape[1], "bgr8", bgr.shape[1] * 3, bgr.tobytes(), block)
        except (ValueError, hdn.FrameError) as e:
            n["decode_error"] += 1
            n["errors"] = (n["errors"] + [str(e)])[:5]
            continue
        frames.append((recv * 1e-9, s * 1e-9, small, rgb))
    n["frames"] = len(frames)
    n["wh"] = [w, h]
    return frames, n


class Side:
    """한 팔: 계약 · 소스 · FK · 학생 · 넘김 · 기록."""

    def __init__(self, side: str, cfg_bi, policy_dir: Path, contract: Path, num_threads: int):
        self.side = side
        self.pol = StudentPolicy.from_dir(policy_dir, num_threads=num_threads)
        obj = A.SIM_COLLIDER_ALIAS[self.pol.policy_cfg["one_obj"]]
        self.c = A.select_object(A.load_contract(contract), obj)
        cfg = select_side(cfg_bi, self.c.side().side)
        self.src = SourceSet(cfg)
        self.arm_names = tuple(cfg.sources["arm"].joints)
        s = self.c.side()
        self.fk = UrdfChainFK(ASSETS[self.c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
        self.dec = A.RaDecoder(self.c)
        self.ho = StudentHandover(hz=self.c.policy_hz, max_s=APPROACH_MAX_S, on_max="handover", **self.pol.handover)
        self.hold = A.ready_hold_action(self.c)
        self.prev = np.zeros(13)
        self.rows, self.lat, self.n = [], [], {"ticks": 0, "steps": 0, "stale_depth": 0, "meas_error": 0,
                                               "prop_nonfinite": 0, "act_nonfinite": 0, "errors": []}
        self.pol.reset()

    def tick(self, t: float, frame) -> None:
        self.n["ticks"] += 1
        try:
            m = A.RaMeas(**_side_raw(self.c.side(), self.src.snapshot(t), self.arm_names, self.fk,
                                      (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])), "arm"))
        except Exception as e:  # noqa: BLE001 — 결손 · stale 은 세고 다음 틱(실기 노드도 그 틱을 건너뛴다)
            self._err("meas_error", e)
            return
        hand_q = [m.hand_q[j] for j in self.c.side().hand_joints]
        # shortcut: 디코더 목표 = 측정(arm_q_target = arm_q, hand_err = 0) — open-loop 이라 학생 행동으로 굴리면 실측에서 벗어난다
        self.dec.reset(arm_start=m.arm_q, hand_start=hand_q)
        stage, stopping, _ = self.ho.step(m.palm_pos)
        prop = build_student_obs(self.c, m, self.dec, self.prev, stage)
        if not np.isfinite(prop).all():
            self.n["prop_nonfinite"] += 1
        _, stamp, depth, rgb = frame
        t0 = time.perf_counter()
        try:
            a = self.pol.step(depth, prop, t - stamp, rgb)
        except StaleDepth:
            self.n["stale_depth"] += 1
            return
        self.lat.append((time.perf_counter() - t0) * 1e3)
        self.n["steps"] += 1
        if not np.isfinite(a).all():
            self.n["act_nonfinite"] += 1
        self.prev = self.ho.action(stopping, a, self.hold)
        self.rows.append([round(t, 4), stage, int(stopping), round(t - stamp, 4), round(self.lat[-1], 3), *a.tolist()])

    def _err(self, key: str, e: Exception) -> None:
        self.n[key] += 1
        if len(self.n["errors"]) < 5:
            self.n["errors"].append(f"{type(e).__name__}: {e}")

    def summary(self) -> dict:
        a = np.array([r[5:] for r in self.rows], float).reshape(-1, 13)
        fin = a[np.isfinite(a).all(1)]
        lat = np.array(self.lat) if self.lat else np.array([np.nan])
        return {**self.n, "ckpt": str(self.pol.policy_cfg["ckpt"]["path"].relative_to(REPO)), "md5": self.pol.md5,
                "object": self.c.active_object, "policy_hz": self.c.policy_hz,
                "handover": self.ho.why, "grasp_ticks": sum(r[1] == "grasp" for r in self.rows),
                "nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum()),
                "act_min": fin.min(0).round(4).tolist() if len(fin) else [], "act_mean": fin.mean(0).round(4).tolist() if len(fin) else [],
                "act_max": fin.max(0).round(4).tolist() if len(fin) else [],
                "lat_ms_p50": float(np.percentile(lat, 50)), "lat_ms_p95": float(np.percentile(lat, 95))}


def image_stats(frames: list, d_min: float, d_max: float) -> dict:
    d = np.stack([f[2] for f in frames])
    rgb = np.stack([f[3] for f in frames])
    v = d > 0
    return {"depth_valid_frac": float(v.mean()), "depth_valid_frac_min_frame": float(v.mean((1, 2)).min()),
            "depth_in_ckpt_range_frac": float(((d >= d_min) & (d <= d_max)).mean()),
            "depth_valid_m_p5_p50_p95": np.percentile(d[v], [5, 50, 95]).round(3).tolist() if v.any() else [],
            "ckpt_d_min_max": [d_min, d_max],
            "rgb_mean": rgb.mean((0, 1, 2)).round(4).tolist(), "rgb_median": np.median(rgb.reshape(-1, 3), 0).round(4).tolist(),
            "depth_nonfinite": int((~np.isfinite(d)).sum()), "rgb_nonfinite": int((~np.isfinite(rgb)).sum())}


def save_pngs(frames: list, out: Path, d_max: float) -> list:
    import cv2

    names = []
    for k in np.linspace(0, len(frames) - 1, N_PNG).astype(int):
        _, s, d, rgb = frames[k]
        g = (np.clip(d / d_max, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(out / f"depth_{k:04d}.png"), g)
        cv2.imwrite(str(out / f"rgb_{k:04d}.png"), (rgb[..., ::-1] * 255).round().astype(np.uint8))
        names += [f"depth_{k:04d}.png", f"rgb_{k:04d}.png"]
    return names


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, r):
        self.lines.append(self.format(r))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag", default="logs/bags/20261104_125804_episode_cocktail_1c_e4000")
    ap.add_argument("--out", default="logs/analysis/student_bag_replay_1105")
    ap.add_argument("--sides", default="right,left")
    ap.add_argument("--max-s", type=float, default=0.0, help="앞에서 이만큼만(0 = 전부)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    from sensor_msgs.msg import CameraInfo, CompressedImage, JointState

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    cap = _Lines()
    logging.getLogger().addHandler(cap)
    bag, out = REPO / args.bag, REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    hdn = load_head_depth_node()
    cfg_bi = load_robot_cfg(ROBOT)
    sides = {s: Side(s, cfg_bi, AGLT / f"{s}_student1104e", AGLT / f"{s}_rcg04" / "rh_aglt_contract.json", args.threads)
             for s in args.sides.split(",")}

    pc = next(iter(sides.values())).pol.policy_cfg       # 외부 파라미터 · rgb 는 두 묶음이 같다(아래에서 확인)
    if len({s.pol.policy_cfg["extrinsics"]["md5"] for s in sides.values()}) != 1:
        raise SystemExit("두 묶음의 외부 파라미터 md5 가 다르다")
    t_base_cam, extr_md5 = hdn.load_extrinsics(pc["extrinsics"]["path"])
    block, min_valid = hdn.real_depth.BLOCK, hdn.real_depth.MIN_VALID
    cut = hdn.make_cut(t_base_cam)
    node_cfg = hdn.node_config(block, min_valid, hdn.TABLE_Z, hdn.real_depth.FLOOR_MARGIN_M, str(pc["extrinsics"]["path"]),
                               extr_md5, pc["rgb"])
    for s in sides.values():
        check_depth_config(s.pol.depth_pre, node_cfg)         # 전처리 = ckpt depth_pre(어긋나면 RhAgltError)

    cam = read_bag(bag / "camera", {T_DEPTH: CompressedImage, T_COLOR: CompressedImage, T_INFO: CameraInfo})
    info = next(m for _, t, m in cam if t == T_INFO)
    info_ok, info_msg = hdn.check_info(info.width, info.height, info.k)
    first_depth = next(m for _, t, m in cam if t == T_DEPTH)
    raw = bytes(first_depth.data)
    header = {"format": first_depth.format, "len": len(raw), "first16_hex": raw[:16].hex(),
              "png_magic_at": raw.find(PNG_MAGIC), "header_int32_float2": np.frombuffer(raw[:CD_HEADER], "<i4", 1).tolist()
              + np.frombuffer(raw[4:CD_HEADER], "<f4").tolist()}
    frames, cam_n = camera_frames(cam, hdn, cut, block, min_valid)
    del cam
    if not frames:
        raise SystemExit(f"깊이 프레임이 없다: {cam_n}")

    joints = read_bag(bag / "arm", {T_ARM: JointState}) + read_bag(bag / "hand", {T_HAND[s]: JointState for s in sides})
    joints.sort(key=lambda x: x[0])
    hz = next(iter(sides.values())).c.policy_hz
    t_end = frames[-1][0] if args.max_s <= 0 else min(frames[-1][0], frames[0][0] + args.max_s)
    ticks = np.arange(frames[0][0], t_end, 1.0 / hz)
    ji = fi = 0
    for t in ticks:
        while ji < len(joints) and joints[ji][0] * 1e-9 <= t:
            recv, topic, msg = joints[ji]
            js = codec.decode_joint_state(msg)
            for s, sd in sides.items():
                if topic == T_ARM:
                    sd.src.update_from_joint_state("arm", js, recv * 1e-9)
                elif topic == T_HAND[s]:
                    sd.src.update_from_joint_state("ee", js, recv * 1e-9)
            ji += 1
        while fi + 1 < len(frames) and frames[fi + 1][0] <= t:
            fi += 1
        for sd in sides.values():
            sd.tick(float(t), frames[fi])

    summ = {"bag": args.bag, "ticks_hz": hz, "duration_s": float(ticks[-1] - ticks[0]), "camera": cam_n,
            "camera_info": {"wh": [info.width, info.height], "ok": info_ok, "msg": info_msg},
            "compressed_depth_header": header, "depth_config": node_cfg,
            "images": image_stats(frames, *(next(iter(sides.values())).pol.d_min, next(iter(sides.values())).pol.d_max)),
            "sides": {}, "pngs": save_pngs(frames, out, next(iter(sides.values())).pol.d_max)}
    for s, sd in sides.items():
        summ["sides"][s] = sd.summary()
        with open(out / f"{s}_actions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "stage", "stopping", "depth_age_s", "lat_ms", *(f"a{i}" for i in range(13))])
            w.writerows(sd.rows)
    summ["log_md5_lines"] = [ln for ln in cap.lines if "md5" in ln]
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: summ[k] for k in ("camera", "images", "sides", "log_md5_lines")}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

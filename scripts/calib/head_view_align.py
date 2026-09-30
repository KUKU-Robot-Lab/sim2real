#!/usr/bin/env python3
"""머리 카메라 화면을 기준 로봇(5090 DG-5F)의 홈 화면과 맞춘다 — **읽기만**. 모터에 쓰지 않는다.

09.30 사용자: arm4090(RH56F1 로봇) 머리의 엔코더를 다시 맞춰 "5090 이랑 똑같이 화면 나오게". 머리 두 모터는
pan ID 1 · tilt ID 2 · 1M 로 이미 맞았지만 원점(Homing Offset)은 조립 상태를 따른다. 그래서
  1) 토크를 끈 머리를 손으로 움직이며 이 도구가 매초 내는 차이(숙임 · 옆 기울기 · 테이블 거리 · 좌우 밀림)를 0 으로 맞추고,
  2) 맞은 자세에서 읽은 엔코더로 **제안** Homing Offset(= 목표 틱 − 원래 틱)을 낸다. 쓰기는 사람이 승인한 뒤 따로 한다.

기준: config/head_view_ref_5090.npz — 5090 머리 홈(pan 2049 · tilt 1820 틱, homing offset 0)에서 찍은
  컬러 + 컬러 정렬 깊이[mm] + K. 목표 틱은 config/head_home.yaml 의 deg(pan 0 · tilt −20)를 틱으로 바꾼 값.

  python3 scripts/calib/head_view_align.py --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT763P8T-if00-port0
  → 한 줄씩: Δ숙임 · Δ옆기울기 · Δ거리 · Δ좌우(px) · pan/tilt 틱 · 제안 offset.  겹친 영상은 --overlay 파일(기본 /tmp/head_align.png)

숙임 · 옆 기울기 · 거리는 깊이로 맞춘 테이블 평면에서 나온다(scripts/calib/depth_table_plane.py 의 RANSAC).
좌우(pan)는 평면으로는 안 보인다 — 기준 영상과의 ORB 특징 대응에서 가로 이동의 중앙값을 낸다. 테이블 판이 달라 대응이
부족하면 '-' 로 두고 겹친 영상을 눈으로 본다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_table_plane import backproject, ransac_plane  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REF = REPO / "config" / "head_view_ref_5090.npz"
DEFAULT_HOME = REPO / "config" / "head_home.yaml"
TICKS_PER_REV = 4096
ADDR_HOMING_OFFSET = 20
ADDR_PRESENT_POSITION = 132
MIN_ORB_MATCHES = 15


def deg_to_tick(deg: float) -> int:
    """head_home.py 와 같은 규약: 0° = 2048 틱."""
    return int(round(2048 + float(deg) * TICKS_PER_REV / 360.0))


def proposed_homing_offset(present: int, current_offset: int, target: int) -> int:
    """지금 자세가 목표 자세일 때 써야 할 Homing Offset. present = raw + current_offset 이다."""
    return int(target - (present - current_offset))


def view_metrics(depth_m: np.ndarray, k: np.ndarray, seed: int = 0) -> dict:
    """테이블 평면(깊이에서 가장 큰 평면) 기준 카메라 자세. 법선은 카메라에서 멀어지는 쪽(+z)."""
    pts = backproject(depth_m, k)
    n, d, mask = ransac_plane(pts, np.random.default_rng(seed))
    return {"pitch_deg": float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))),
            "roll_deg": float(np.degrees(np.arctan2(-n[0], n[1]))),
            "dist_m": float(abs(d)), "inliers": float(mask.mean())}


def orb_shift_px(ref_rgb: np.ndarray, cur_rgb: np.ndarray) -> tuple[float | None, int]:
    """기준 → 현재 영상의 가로 이동 중앙값(px)과 대응 수. 대응이 모자라면 (None, n)."""
    import cv2

    orb = cv2.ORB_create(1500)
    g1, g2 = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (ref_rgb, cur_rgb))
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    good = [m for m in matches if m.distance < 40]
    if len(good) < MIN_ORB_MATCHES:
        return None, len(good)
    du = [k2[m.trainIdx].pt[0] - k1[m.queryIdx].pt[0] for m in good]
    return float(np.median(du)), len(good)


def load_ref(path: Path) -> dict:
    d = np.load(path)
    return {"rgb": d["rgb"], "depth": d["depth_mm"].astype(np.float32) * 1e-3, "K": d["K"],
            "pan_tick": int(d["pan_tick"]), "tilt_tick": int(d["tilt_tick"])}


def targets_from_home(path: Path) -> tuple[int, int]:
    raw = yaml.safe_load(path.read_text())
    return deg_to_tick(raw["motors"]["pan"]["deg"]), deg_to_tick(raw["motors"]["tilt"]["deg"])


def _camera():
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    for _ in range(30):
        pipe.wait_for_frames(5000)

    def grab():
        f = align.process(pipe.wait_for_frames(5000))
        c, dep = f.get_color_frame(), f.get_depth_frame()
        i = c.profile.as_video_stream_profile().get_intrinsics()
        k = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
        return np.asanyarray(c.get_data()).copy(), np.asanyarray(dep.get_data()).astype(np.float32) * scale, k

    return pipe, grab


def _reader(port: str, baud: int):
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

    ph, pk = PortHandler(port), PacketHandler(2.0)
    if not ph.openPort() or not ph.setBaudRate(baud):
        raise SystemExit(f"{port} @ {baud} 를 열지 못했다")

    def read(dxl_id: int, addr: int) -> int | None:
        v, rc, _ = pk.read4ByteTxRx(ph, dxl_id, addr)
        if rc != COMM_SUCCESS:
            return None
        return v - (1 << 32) if v > 0x7FFFFFFF else v

    return ph, read


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="U2D2 포트(/dev/serial/by-id/...)")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--home", type=Path, default=DEFAULT_HOME)
    ap.add_argument("--overlay", type=Path, default=Path("/tmp/head_align.png"))
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--count", type=int, default=0, help="0 = Ctrl-C 까지")
    args = ap.parse_args(argv)

    import cv2

    ref = load_ref(args.ref)
    ref_m = view_metrics(ref["depth"], ref["K"])
    pan_t, tilt_t = targets_from_home(args.home)
    print(f"기준 {args.ref.name}: 숙임 {ref_m['pitch_deg']:.1f}° · 옆기울기 {ref_m['roll_deg']:.1f}° · "
          f"거리 {ref_m['dist_m']:.3f} m · 목표 틱 pan {pan_t} · tilt {tilt_t}", flush=True)
    pipe, grab = _camera()
    ph, read = _reader(args.port, args.baud)
    n = 0
    try:
        while args.count == 0 or n < args.count:
            rgb, depth, k = grab()
            m = view_metrics(depth, k)
            du, nm = orb_shift_px(ref["rgb"], rgb)
            ticks = {}
            for name, dxl_id, target in (("pan", 1, pan_t), ("tilt", 2, tilt_t)):
                present, offset = read(dxl_id, ADDR_PRESENT_POSITION), read(dxl_id, ADDR_HOMING_OFFSET)
                ticks[name] = ("?" if present is None or offset is None else
                               f"{present}(offset {offset} → 제안 {proposed_homing_offset(present, offset, target)})")
            print(f"Δ숙임 {m['pitch_deg'] - ref_m['pitch_deg']:+6.1f}° · Δ옆 {m['roll_deg'] - ref_m['roll_deg']:+5.1f}° · "
                  f"Δ거리 {m['dist_m'] - ref_m['dist_m']:+.3f} m · Δ좌우 {'-' if du is None else f'{du:+.0f} px'} "
                  f"(대응 {nm}) · 평면 {m['inliers']:.2f} | pan {ticks['pan']} · tilt {ticks['tilt']}", flush=True)
            cv2.imwrite(str(args.overlay), cv2.addWeighted(ref["rgb"], 0.5, rgb, 0.5, 0)[:, :, ::-1])
            n += 1
            time.sleep(args.period)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        ph.closePort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

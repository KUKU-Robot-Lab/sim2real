#!/usr/bin/env python3
"""머리를 조금씩 움직여 카메라 화면을 5090 홈 화면(config/head_view_ref_5090.npz)과 맞춘다 — **--execute 일 때만 움직인다**.

09.30 사용자: "직접 비교 대조해서 조금씩 모터를 움직여서 똑같게". 화면 수치는 head_view_align.py 와 같다.
  tilt ← 테이블 평면의 부호 있는 숙임(pitch_signed) 차이      pan ← 기준 영상과의 ORB 가로 이동(px)
처음에 각 축을 +probe° 움직여 부호 · 이득을 잰 뒤, 한 번에 --max-step° 이하로 오차를 줄인다. 시작 자세에서 --window° 를
벗어나지 않고, 하드웨어 오류 · 55 °C · 테이블이 안 보임 · 수렴 실패면 그 자리에 멈춘다(토크는 켜 둔 채 버틴다).
끝나면 맞은 자세의 틱과 **제안** Homing Offset 을 낸다. offset 은 쓰지 않는다 — 따로 승인받아 쓴다.

쓰는 순서는 head_home.py 와 같다: 토크off → 모드(3) → I 게인 → 프로파일 → 토크on(목표 = 현재, 튀지 않음) → 목표.

  python3 scripts/calib/head_view_autoalign.py --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT763P8T-if00-port0            # 재기만
  python3 scripts/calib/head_view_autoalign.py --port … --execute                                                                          # 움직인다
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
from head_view_align import (DEFAULT_HOME, DEFAULT_REF, _camera, load_ref, orb_shift_px,  # noqa: E402
                             proposed_homing_offset, targets_from_home, view_metrics)

TICKS_PER_DEG = 4096 / 360.0
PAN, TILT = 1, 2
ADDR_OPERATING_MODE, ADDR_HOMING_OFFSET, ADDR_TORQUE_ENABLE = 11, 20, 64
ADDR_HW_ERROR, ADDR_POSITION_I_GAIN = 70, 82
ADDR_PROFILE_ACC, ADDR_PROFILE_VEL, ADDR_GOAL, ADDR_PRESENT, ADDR_TEMP = 108, 112, 116, 132, 146
MAX_TEMP_C = 55
MIN_INLIERS = 0.3            # 이보다 적으면 테이블을 못 본다 — 움직이지 않는다
MAX_START_PITCH_ERR = 45.0   # 시작 화면이 기준에서 이보다 멀면 사람이 먼저 대강 돌린다
PITCH_TOL_DEG, SHIFT_TOL_PX = 0.3, 3.0


def clamp_step(error: float, gain: float, max_step: float) -> float:
    """오차를 줄이는 한 걸음(°). gain = 측정값 변화 / 모터 각 변화."""
    if abs(gain) < 1e-6:
        return 0.0
    return float(np.clip(-error / gain, -max_step, max_step))


def within_window(target_tick: int, start_tick: int, window_deg: float) -> int:
    lim = int(round(window_deg * TICKS_PER_DEG))
    return int(np.clip(target_tick, start_tick - lim, start_tick + lim))


class Bus:
    def __init__(self, port: str, baud: int) -> None:
        from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
        self.ok, self.ph, self.pk = COMM_SUCCESS, PortHandler(port), PacketHandler(2.0)
        if not self.ph.openPort() or not self.ph.setBaudRate(baud):
            raise SystemExit(f"{port} @ {baud} 를 열지 못했다")

    def read(self, dxl_id: int, addr: int, n: int) -> int:
        fn = {1: self.pk.read1ByteTxRx, 2: self.pk.read2ByteTxRx, 4: self.pk.read4ByteTxRx}[n]
        v, rc, err = fn(self.ph, dxl_id, addr)
        if rc != self.ok:
            raise RuntimeError(f"id{dxl_id} addr {addr} 읽기 실패: {self.pk.getTxRxResult(rc)}")
        return v - (1 << 32) if n == 4 and v > 0x7FFFFFFF else v

    def write(self, dxl_id: int, addr: int, n: int, value: int) -> None:
        fn = {1: self.pk.write1ByteTxRx, 2: self.pk.write2ByteTxRx, 4: self.pk.write4ByteTxRx}[n]
        rc, err = fn(self.ph, dxl_id, addr, int(value) & 0xFFFFFFFF)
        if rc != self.ok or err:
            raise RuntimeError(f"id{dxl_id} addr {addr} ← {value} 쓰기 실패: {self.pk.getTxRxResult(rc)} err {err}")

    def health(self, dxl_id: int) -> None:
        hw, temp = self.read(dxl_id, ADDR_HW_ERROR, 1), self.read(dxl_id, ADDR_TEMP, 1)
        if hw or temp >= MAX_TEMP_C:
            raise RuntimeError(f"id{dxl_id} 하드웨어 오류 0x{hw:02x} · {temp} °C — 멈춘다")

    def close(self) -> None:
        self.ph.closePort()


def enable_hold(bus: Bus, dxl_id: int, vel: int, acc: int, i_gain: int) -> None:
    bus.write(dxl_id, ADDR_TORQUE_ENABLE, 1, 0)
    bus.write(dxl_id, ADDR_OPERATING_MODE, 1, 3)
    bus.write(dxl_id, ADDR_POSITION_I_GAIN, 2, i_gain)
    bus.write(dxl_id, ADDR_PROFILE_ACC, 4, acc)
    bus.write(dxl_id, ADDR_PROFILE_VEL, 4, vel)
    bus.write(dxl_id, ADDR_TORQUE_ENABLE, 1, 1)       # 목표가 현재로 덮인다 — 튀지 않는다


def move_to(bus: Bus, dxl_id: int, tick: int, timeout_s: float = 4.0) -> int:
    bus.write(dxl_id, ADDR_GOAL, 4, tick)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        now = bus.read(dxl_id, ADDR_PRESENT, 4)
        if abs(now - tick) <= 3:
            break
        time.sleep(0.05)
    time.sleep(0.4)
    bus.health(dxl_id)
    return bus.read(dxl_id, ADDR_PRESENT, 4)


def measure(grab, ref, frames: int = 3) -> dict:
    pitches, shifts, inl = [], [], []
    for _ in range(frames):
        rgb, depth, k = grab()
        m = view_metrics(depth, k)
        du, _ = orb_shift_px(ref["rgb"], rgb)
        pitches.append(m["pitch_signed_deg"])
        inl.append(m["inliers"])
        if du is not None:
            shifts.append(du)
    return {"pitch": float(np.median(pitches)), "inliers": float(np.median(inl)),
            "shift": float(np.median(shifts)) if len(shifts) >= 2 else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--home", type=Path, default=DEFAULT_HOME)
    ap.add_argument("--probe-deg", type=float, default=2.0)
    ap.add_argument("--max-step-deg", type=float, default=2.0)
    ap.add_argument("--window-deg", type=float, default=25.0, help="시작 자세에서 이만큼만 움직인다")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--vel", type=int, default=15, help="Profile Velocity(0.229 rpm 단위) — 15 ≈ 21°/s")
    ap.add_argument("--acc", type=int, default=10)
    ap.add_argument("--i-gain", type=int, default=400)
    ap.add_argument("--coarse-tilt-to", type=int, default=None,
                    help="먼저 tilt 를 이 틱 쪽으로 --coarse-step-deg 씩 옮기며 테이블이 잡히면 멈춘다(최대 --coarse-max-deg)")
    ap.add_argument("--coarse-step-deg", type=float, default=5.0)
    ap.add_argument("--coarse-max-deg", type=float, default=100.0)
    ap.add_argument("--execute", action="store_true", help="이게 있어야 모터에 쓴다")
    args = ap.parse_args(argv)

    ref = load_ref(args.ref)
    ref_pitch = view_metrics(ref["depth"], ref["K"])["pitch_signed_deg"]
    pan_t, tilt_t = targets_from_home(args.home)
    pipe, grab = _camera()
    bus = Bus(args.port, args.baud)
    try:
        start = {i: bus.read(i, ADDR_PRESENT, 4) for i in (PAN, TILT)}
        m = measure(grab, ref)
        print(f"시작: pan {start[PAN]} · tilt {start[TILT]} 틱 · 숙임 {m['pitch']:+.1f}°(기준 {ref_pitch:+.1f}) · "
              f"좌우 {m['shift']} px · 평면 {m['inliers']:.2f}", flush=True)
        coarse = args.coarse_tilt_to is not None
        if coarse:
            travel = (args.coarse_tilt_to - start[TILT]) / TICKS_PER_DEG
            if not 0 <= args.coarse_tilt_to <= 4095 or abs(travel) > args.coarse_max_deg:
                print(f"큰 이동 목표 {args.coarse_tilt_to} 가 한계(0~4095) 밖이거나 {travel:+.0f}° 로 너무 멀다 — 멈춘다")
                return 1
            print(f"큰 이동: tilt {start[TILT]} → {args.coarse_tilt_to} ({travel:+.0f}°) 를 {args.coarse_step_deg}° 씩")
        elif m["inliers"] < MIN_INLIERS or abs(m["pitch"] - ref_pitch) > MAX_START_PITCH_ERR:
            print("테이블이 안 보이거나 기준에서 너무 멀다 — 손으로 테이블 쪽으로 대강 돌린 뒤 다시(또는 --coarse-tilt-to).")
            return 1
        if not args.execute:
            print("--execute 없이 — 모터에 쓰지 않았다")
            return 0
        for i in (PAN, TILT):
            bus.health(i)
            enable_hold(bus, i, args.vel, args.acc, args.i_gain)
        cur = dict(start)
        if coarse:
            step = round(args.coarse_step_deg * TICKS_PER_DEG) * (1 if args.coarse_tilt_to > cur[TILT] else -1)
            while True:
                nxt = cur[TILT] + step
                if (step > 0 and nxt >= args.coarse_tilt_to) or (step < 0 and nxt <= args.coarse_tilt_to):
                    nxt = args.coarse_tilt_to
                cur[TILT] = move_to(bus, TILT, nxt)
                m = measure(grab, ref, frames=2)
                print(f"  tilt {cur[TILT]} · 숙임 {m['pitch']:+.1f}° · 평면 {m['inliers']:.2f}", flush=True)
                if m["inliers"] >= MIN_INLIERS and abs(m["pitch"] - ref_pitch) < 10.0:
                    print("  테이블이 기준 근처에 잡혔다 — 큰 이동 끝")
                    break
                if nxt == args.coarse_tilt_to:
                    break
            if m["inliers"] < MIN_INLIERS or abs(m["pitch"] - ref_pitch) > MAX_START_PITCH_ERR:
                print("큰 이동 뒤에도 테이블이 기준 근처에 없다 — 그 자리에서 멈춘다(토크 켜 둠)")
                return 1
            start = dict(cur)          # 미세 조정 창은 큰 이동이 끝난 자리 기준
        # 이득 재기: tilt → 숙임, pan → 좌우
        gains = {}
        for i, key in ((TILT, "pitch"), (PAN, "shift")):
            before = measure(grab, ref)[key]
            cur[i] = move_to(bus, i, within_window(cur[i] + round(args.probe_deg * TICKS_PER_DEG), start[i], args.window_deg))
            after = measure(grab, ref)[key]
            moved = (cur[i] - start[i]) / TICKS_PER_DEG
            gains[i] = None if before is None or after is None or abs(moved) < 0.5 else (after - before) / moved
            print(f"이득 id{i}: {moved:+.2f}° → {key} {before} → {after} · 이득 {gains[i]}", flush=True)
        if gains[TILT] is None or abs(gains[TILT]) < 0.3:
            print("tilt 를 움직여도 숙임이 안 바뀐다 — 멈춘다(토크 켜 둔 채)")
            return 1
        for it in range(args.iters):
            m = measure(grab, ref)
            e_pitch = m["pitch"] - ref_pitch
            e_shift = m["shift"]
            done_t = abs(e_pitch) < PITCH_TOL_DEG
            done_p = gains[PAN] is None or e_shift is None or abs(e_shift) < SHIFT_TOL_PX
            print(f"[{it}] pan {cur[PAN]} · tilt {cur[TILT]} · Δ숙임 {e_pitch:+.2f}° · Δ좌우 {e_shift} px · "
                  f"평면 {m['inliers']:.2f}", flush=True)
            if m["inliers"] < MIN_INLIERS:
                print("테이블을 놓쳤다 — 멈춘다")
                break
            if done_t and done_p:
                print("수렴")
                break
            if not done_t:
                step = clamp_step(e_pitch, gains[TILT], args.max_step_deg)
                cur[TILT] = move_to(bus, TILT, within_window(cur[TILT] + round(step * TICKS_PER_DEG), start[TILT], args.window_deg))
            if not done_p:
                step = clamp_step(e_shift, gains[PAN], args.max_step_deg)
                cur[PAN] = move_to(bus, PAN, within_window(cur[PAN] + round(step * TICKS_PER_DEG), start[PAN], args.window_deg))
        offs = {i: bus.read(i, ADDR_HOMING_OFFSET, 4) for i in (PAN, TILT)}
        print(f"끝: pan {cur[PAN]} · tilt {cur[TILT]} 틱(토크 켜 둠) · 제안 Homing Offset "
              f"pan {proposed_homing_offset(cur[PAN], offs[PAN], pan_t)} · tilt {proposed_homing_offset(cur[TILT], offs[TILT], tilt_t)}"
              + ("" if gains[PAN] is not None else " · ★pan 은 화면 대응이 없어 맞추지 못했다(겹친 영상으로 눈으로)"))
        return 0
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"멈춤: {exc} — 토크는 켜 둔 채 그 자리에서 버틴다")
        return 1
    finally:
        pipe.stop()
        bus.close()


if __name__ == "__main__":
    raise SystemExit(main())
